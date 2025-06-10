package com.example.artest

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.Image
import android.os.Environment
import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionStatus
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.ar.core.*
import io.github.sceneview.ar.ARScene
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

data class DetectedObject(
    val label: String,
    val confidence: Float,
    val boundingBox: RectF,
    val distance: Float
)

class TiltSensorManager(
    private val context: Context,
    private val onTiltChanged: (Float) -> Unit,
    private val onSensorAvailabilityChanged: (Boolean) -> Unit
) : SensorEventListener {
    
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    
    private val accelerometerReading = FloatArray(3)
    private val magnetometerReading = FloatArray(3)
    private val rotationMatrix = FloatArray(9)
    private val orientationAngles = FloatArray(3)
    
    val isSensorAvailable: Boolean
        get() = accelerometer != null && magnetometer != null
    
    fun startListening() {
        if (!isSensorAvailable) {
            Log.w("TiltSensor", "Accelerometer or magnetometer not available")
            onSensorAvailabilityChanged(false)
            return
        }
        
        val accelerometerRegistered = sensorManager.registerListener(
            this, accelerometer, SensorManager.SENSOR_DELAY_UI
        )
        val magnetometerRegistered = sensorManager.registerListener(
            this, magnetometer, SensorManager.SENSOR_DELAY_UI
        )
        
        val sensorsRegistered = accelerometerRegistered && magnetometerRegistered
        onSensorAvailabilityChanged(sensorsRegistered)
        
        if (!sensorsRegistered) {
            Log.w("TiltSensor", "Failed to register sensor listeners")
        }
    }
    
    fun stopListening() {
        sensorManager.unregisterListener(this)
    }
    
    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            when (it.sensor?.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    System.arraycopy(it.values, 0, accelerometerReading, 0, accelerometerReading.size)
                }
                Sensor.TYPE_MAGNETIC_FIELD -> {
                    System.arraycopy(it.values, 0, magnetometerReading, 0, magnetometerReading.size)
                }
            }
            
            // Calculate orientation
            SensorManager.getRotationMatrix(
                rotationMatrix, null,
                accelerometerReading, magnetometerReading
            )
            SensorManager.getOrientation(rotationMatrix, orientationAngles)
            
            // Convert pitch to degrees (orientationAngles[1] is pitch in radians)
            val pitchDegrees = Math.toDegrees(orientationAngles[1].toDouble()).toFloat()
            onTiltChanged(pitchDegrees)
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}

class TensorFlowObjectDetector(private val context: Context) {
    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    
    companion object {
        private const val MODEL_PATH = "1.tflite"
        private const val INPUT_SIZE = 320
        private const val DETECTION_THRESHOLD = 0.5f
    }
    
    init {
        initializeModel()
    }
    
    private fun initializeModel() {
        try {
            // Load the TFLite model
            val modelBuffer = loadModelFile()
            interpreter = Interpreter(modelBuffer)
            
            // Load labels (COCO dataset labels)
            labels = listOf(
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush"
            )
            
            Log.d("TFLite", "Model loaded successfully")
        } catch (e: Exception) {
            Log.e("TFLite", "Error loading model", e)
        }
    }
    
    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_PATH)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun detectObjects(bitmap: Bitmap): List<DetectedObject> {
        val detectedObjects = mutableListOf<DetectedObject>()
        
        try {
            // Resize bitmap to model input size
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            
            // Convert bitmap to ByteBuffer
            val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
            
            // For demonstration purposes, create mock detections
            // In a real implementation, you would run the actual model inference
            val mockDetections = generateMockDetections(bitmap)
            
            // Filter objects that are in the detection area (center of screen)
            val centerX = bitmap.width / 2f
            val centerY = bitmap.height / 2f
            val detectionWidth = bitmap.width * 0.4f
            val detectionHeight = bitmap.height * 0.3f
            
            val detectionRect = RectF(
                centerX - detectionWidth / 2,
                centerY - detectionHeight / 2,
                centerX + detectionWidth / 2,
                centerY + detectionHeight / 2
            )
            
            mockDetections.forEach { obj ->
                if (RectF.intersects(obj.boundingBox, detectionRect)) {
                    detectedObjects.add(obj)
                }
            }
            
        } catch (e: Exception) {
            Log.e("TFLite", "Error during detection", e)
        }
        
        return detectedObjects
    }
    
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val pixelValue = intValues[pixel++]
                byteBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
                byteBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
                byteBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
            }
        }
        
        return byteBuffer
    }
    
    private fun generateMockDetections(bitmap: Bitmap): List<DetectedObject> {
        // Generate some mock detections for demonstration
        // In a real app, this would come from actual model inference
        val random = Random()
        val detections = mutableListOf<DetectedObject>()
        
        // Add some random detections
        if (random.nextFloat() > 0.3f) {
            detections.add(
                DetectedObject(
                    label = labels[random.nextInt(min(10, labels.size))], // Use first 10 common objects
                    confidence = 0.7f + random.nextFloat() * 0.25f,
                    boundingBox = RectF(
                        random.nextFloat() * bitmap.width * 0.3f,
                        random.nextFloat() * bitmap.height * 0.3f,
                        random.nextFloat() * bitmap.width * 0.3f + bitmap.width * 0.4f,
                        random.nextFloat() * bitmap.height * 0.3f + bitmap.height * 0.4f
                    ),
                    distance = 1.0f + random.nextFloat() * 4.0f
                )
            )
        }
        
        return detections
    }
    
    fun close() {
        interpreter?.close()
    }
}

@Composable
fun TiltIndicator(
    currentAngle: Float,
    targetAngle: Float = 45f,
    isAvailable: Boolean = true,
    modifier: Modifier = Modifier
) {
    val isCorrectAngle = abs(currentAngle - targetAngle) < 5f // 5° tolerance
    val indicatorColor = if (!isAvailable) Color.Gray else if (isCorrectAngle) Color.Green else Color.Red
    
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Visual tilt indicator
        Box(
            modifier = Modifier
                .size(80.dp)
                .background(
                    Color.Black.copy(alpha = 0.7f),
                    CircleShape
                )
                .padding(8.dp),
            contentAlignment = Alignment.Center
        ) {
            if (!isAvailable) {
                // Show "X" when sensors not available
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val center = Offset(size.width / 2, size.height / 2)
                    val radius = size.minDimension / 2 - 8.dp.toPx()
                    
                    // Draw circle background
                    drawCircle(
                        color = Color.Gray.copy(alpha = 0.3f),
                        radius = radius,
                        center = center,
                        style = Stroke(width = 4.dp.toPx())
                    )
                    
                    // Draw X
                    val lineLength = radius * 0.6f
                    drawLine(
                        color = Color.Red,
                        start = Offset(center.x - lineLength, center.y - lineLength),
                        end = Offset(center.x + lineLength, center.y + lineLength),
                        strokeWidth = 4.dp.toPx()
                    )
                    drawLine(
                        color = Color.Red,
                        start = Offset(center.x + lineLength, center.y - lineLength),
                        end = Offset(center.x - lineLength, center.y + lineLength),
                        strokeWidth = 4.dp.toPx()
                    )
                }
            } else {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val center = Offset(size.width / 2, size.height / 2)
                    val radius = size.minDimension / 2 - 8.dp.toPx()
                    
                    // Draw circle background
                    drawCircle(
                        color = Color.Gray.copy(alpha = 0.3f),
                        radius = radius,
                        center = center,
                        style = Stroke(width = 4.dp.toPx())
                    )
                    
                    // Draw target angle indicator (45°)
                    val targetAngleRad = Math.toRadians((targetAngle - 90).toDouble())
                    val targetX = center.x + radius * cos(targetAngleRad).toFloat()
                    val targetY = center.y + radius * sin(targetAngleRad).toFloat()
                    
                    drawCircle(
                        color = Color.Green,
                        radius = 6.dp.toPx(),
                        center = Offset(targetX, targetY)
                    )
                    
                    // Draw current angle indicator
                    val currentAngleRad = Math.toRadians((currentAngle - 90).toDouble())
                    val currentX = center.x + radius * cos(currentAngleRad).toFloat()
                    val currentY = center.y + radius * sin(currentAngleRad).toFloat()
                    
                    drawCircle(
                        color = indicatorColor,
                        radius = 8.dp.toPx(),
                        center = Offset(currentX, currentY)
                    )
                    
                    // Draw center dot
                    drawCircle(
                        color = Color.White,
                        radius = 3.dp.toPx(),
                        center = center
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(8.dp))
        
        // Angle text or unavailable message
        Text(
            text = if (!isAvailable) "N/A" else "${currentAngle.roundToInt()}°",
            color = indicatorColor,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier
                .background(
                    Color.Black.copy(alpha = 0.7f),
                    RoundedCornerShape(8.dp)
                )
                .padding(horizontal = 12.dp, vertical = 4.dp)
        )
        
        // Status text
        Text(
            text = when {
                !isAvailable -> "⚠ NO SENSOR"
                isCorrectAngle -> "✓ CORRECT"
                else -> "↗ TILT TO 45°"
            },
            color = indicatorColor,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier
                .background(
                    Color.Black.copy(alpha = 0.7f),
                    RoundedCornerShape(4.dp)
                )
                .padding(horizontal = 8.dp, vertical = 2.dp)
        )
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun ARObjectDetectionScreen() {
    val context = LocalContext.current
    val cameraPermissionState = rememberPermissionState(android.Manifest.permission.CAMERA)
    val coroutineScope = rememberCoroutineScope()
    
    var detectedObjects by remember { mutableStateOf<List<DetectedObject>>(emptyList()) }
    var isDetecting by remember { mutableStateOf(false) }
    var objectDetector by remember { mutableStateOf<TensorFlowObjectDetector?>(null) }
    var arSession by remember { mutableStateOf<Session?>(null) }
    var currentFrame by remember { mutableStateOf<Frame?>(null) }
    var currentTiltAngle by remember { mutableStateOf(0f) }
    var tiltSensorManager by remember { mutableStateOf<TiltSensorManager?>(null) }
    var lastDetectionTime by remember { mutableStateOf(0L) }
    var isSensorAvailable by remember { mutableStateOf(true) }
    
    val targetAngle = 45f
    val isCorrectAngle = isSensorAvailable && abs(currentTiltAngle - targetAngle) < 5f
    val detectionZoneColor = if (!isSensorAvailable) Color.Gray else if (isCorrectAngle) Color.Green else Color.Red
    
    // Initialize object detector
    LaunchedEffect(Unit) {
        objectDetector = TensorFlowObjectDetector(context)
    }
    
    // Initialize tilt sensor
    LaunchedEffect(Unit) {
        tiltSensorManager = TiltSensorManager(
            context = context,
            onTiltChanged = { angle -> currentTiltAngle = angle },
            onSensorAvailabilityChanged = { available -> isSensorAvailable = available }
        )
        tiltSensorManager?.startListening()
    }
    
    // Clean up
    DisposableEffect(Unit) {
        onDispose {
            objectDetector?.close()
            tiltSensorManager?.stopListening()
        }
    }
    
    if (cameraPermissionState.status == PermissionStatus.Granted) {
        Box(modifier = Modifier.fillMaxSize()) {
            // AR Scene (Compose-native, no AndroidView needed)
            ARScene(
                modifier = Modifier.fillMaxSize(),
                onSessionCreated = { session ->
                    // Configure AR session
                    arSession = session
                    session.configure(session.config.apply {
                        updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                        focusMode = Config.FocusMode.AUTO
                    })
                    Log.d("ARSession", "AR Session configured")
                },
                onSessionUpdated = { arSceneView, arFrame ->
                    // Handle each AR frame
                    currentFrame = arFrame
                    
                    // Perform object detection only every 3 seconds and when angle is correct (or sensor unavailable)
                    val currentTime = System.currentTimeMillis()
                    if (!isDetecting && 
                        (isCorrectAngle || !isSensorAvailable) && 
                        (currentTime - lastDetectionTime) > 3000) {
                        
                        isDetecting = true
                        lastDetectionTime = currentTime
                        
                        coroutineScope.launch {
                            performObjectDetection(arFrame, objectDetector) { objects ->
                                detectedObjects = objects
                                isDetecting = false
                            }
                        }
                    }
                }
            )
            
            // Tilt indicator (top right)
            TiltIndicator(
                currentAngle = currentTiltAngle,
                targetAngle = targetAngle,
                isAvailable = isSensorAvailable,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
            )
            
            // Detection rectangle overlay with dynamic color
            Box(
                modifier = Modifier
                    .size(250.dp, 180.dp)
                    .align(Alignment.Center)
                    .border(
                        3.dp,
                        detectionZoneColor,
                        RoundedCornerShape(12.dp)
                    )
            ) {
                Text(
                    text = when {
                        !isSensorAvailable -> "⚠ SENSOR NOT AVAILABLE"
                        isCorrectAngle -> "✓ DETECTION ZONE"
                        else -> "↗ TILT TO 45°"
                    },
                    color = detectionZoneColor,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .background(
                            Color.Black.copy(alpha = 0.7f),
                            RoundedCornerShape(6.dp)
                        )
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
            
            // Detected objects info panel (only show for longer duration)
            if (detectedObjects.isNotEmpty()) {
                LazyColumn(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(16.dp)
                        .background(
                            Color.Black.copy(alpha = 0.8f),
                            RoundedCornerShape(12.dp)
                        )
                        .padding(12.dp)
                        .widthIn(max = 220.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    // Header with detection info
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = Color.Blue.copy(alpha = 0.9f)
                            ),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Column(
                                modifier = Modifier.padding(12.dp)
                            ) {
                                Text(
                                    text = "DETECTION RESULTS",
                                    color = Color.White,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "Next scan in 3s",
                                    color = Color.White.copy(alpha = 0.7f),
                                    fontSize = 10.sp
                                )
                            }
                        }
                    }
                    
                    items(detectedObjects.size) { index ->
                        val obj = detectedObjects[index]
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = Color.Green.copy(alpha = 0.9f)
                            ),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Column(
                                modifier = Modifier.padding(12.dp)
                            ) {
                                Text(
                                    text = obj.label.uppercase(),
                                    color = Color.White,
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = "Distance:",
                                        color = Color.White,
                                        fontSize = 11.sp
                                    )
                                    Text(
                                        text = "${String.format("%.1f", obj.distance)}m",
                                        color = Color.White,
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = "Confidence:",
                                        color = Color.White,
                                        fontSize = 11.sp
                                    )
                                    Text(
                                        text = "${(obj.confidence * 100).toInt()}%",
                                        color = Color.White,
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }
                        }
                    }
                }
            }
            
            // Camera capture button (only enabled when angle is correct or sensor unavailable)
            FloatingActionButton(
                onClick = {
                    if (isCorrectAngle || !isSensorAvailable) {
                        coroutineScope.launch {
                            currentFrame?.let { frame ->
                                capturePhoto(frame, context)
                            }
                        }
                    } else {
                        Toast.makeText(context, "Please tilt device to 45° first", Toast.LENGTH_SHORT).show()
                    }
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(32.dp),
                containerColor = if (isCorrectAngle || !isSensorAvailable) MaterialTheme.colorScheme.primary else Color.Gray
            ) {
                Icon(
                    imageVector = Icons.Default.Star,
                    contentDescription = "Take Photo",
                    tint = Color.White
                )
            }
            
            // Status text
            Text(
                text = when {
                    !isSensorAvailable -> "Device has no tilt sensor - detection always active"
                    !isCorrectAngle -> "Tilt device to 45° for detection"
                    detectedObjects.isNotEmpty() -> "${detectedObjects.size} objects detected"
                    isDetecting -> "Scanning..."
                    else -> "Hold steady - scanning every 3 seconds"
                },
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(16.dp)
                    .background(
                        Color.Black.copy(alpha = 0.8f),
                        RoundedCornerShape(20.dp)
                    )
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                color = Color.White,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
                textAlign = TextAlign.Center
            )
        }
    } else {
        // Permission request UI
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = Icons.Default.Star,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.height(24.dp))
            Text(
                text = "Camera Access Required",
                style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "This app needs camera permission to perform AR object detection and capture photos.",
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(32.dp))
            Button(
                onClick = { cameraPermissionState.launchPermissionRequest() },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Grant Camera Permission")
            }
        }
    }
}

private suspend fun performObjectDetection(
    frame: Frame,
    detector: TensorFlowObjectDetector?,
    onObjectsDetected: (List<DetectedObject>) -> Unit
) {
    withContext(Dispatchers.IO) {
        try {
            val image = frame.acquireCameraImage()
            image?.let { cameraImage ->
                val bitmap = imageProxyToBitmap(cameraImage)
                val detectedObjects = detector?.detectObjects(bitmap) ?: emptyList()
                
                withContext(Dispatchers.Main) {
                    onObjectsDetected(detectedObjects)
                }
                
                cameraImage.close()
            }
        } catch (e: Exception) {
            Log.e("AR Detection", "Error during object detection", e)
            withContext(Dispatchers.Main) {
                onObjectsDetected(emptyList())
            }
        }
    }
}

private fun imageProxyToBitmap(image: Image): Bitmap {
    val planes = image.planes
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer
    
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()
    
    val nv21 = ByteArray(ySize + uSize + vSize)
    
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)
    
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

private suspend fun capturePhoto(frame: Frame, context: Context) {
    withContext(Dispatchers.IO) {
        try {
            val image = frame.acquireCameraImage()
            image?.let { cameraImage ->
                val bitmap = imageProxyToBitmap(cameraImage)
                saveImageToGallery(bitmap, context)
                cameraImage.close()
                
                withContext(Dispatchers.Main) {
                    Toast.makeText(context, "Photo saved to gallery!", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Log.e("Photo Capture", "Error capturing photo", e)
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Failed to capture photo", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

private fun saveImageToGallery(bitmap: Bitmap, context: Context) {
    val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    val filename = "AR_Photo_$timestamp.jpg"
    
    val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
    val arDir = File(picturesDir, "AR_Test")
    if (!arDir.exists()) {
        arDir.mkdirs()
    }
    
    val file = File(arDir, filename)
    
    try {
        val outputStream = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
        outputStream.flush()
        outputStream.close()
        
        Log.d("Save Image", "Image saved to: ${file.absolutePath}")
        
    } catch (e: Exception) {
        Log.e("Save Image", "Error saving image", e)
    }
}