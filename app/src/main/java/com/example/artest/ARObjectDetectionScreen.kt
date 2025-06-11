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
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionStatus
import com.google.accompanist.permissions.rememberPermissionState
import com.google.ar.core.*
import io.github.sceneview.ar.ARScene
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

data class DetectedObject(
    val label: String,
    val confidence: Float,
    val boundingBox: RectF,
    val distance: Float
)

// Configuration constants
object ARDetectionConfig {

    // Tilt angle configuration
    const val TARGET_ANGLE = -45f           // Target tilt angle in degrees
    const val ANGLE_TOLERANCE = 20f         // Acceptable deviation from target angle

    // Detection configuration
    const val MIN_CONFIDENCE = 0.5f         // Minimum confidence threshold (60%)
    const val DETECTION_INTERVAL_MS = 3000L // Detection interval in milliseconds

    // Detection zone configuration
    const val DETECTION_ZONE_WIDTH_RATIO = 0.4f   // Detection zone width as ratio of screen width
    const val DETECTION_ZONE_HEIGHT_RATIO = 0.3f  // Detection zone height as ratio of screen height

    // Distance calculation
    const val CAMERA_FOCAL_LENGTH_PX = 800f // Approximate focal length in pixels
    const val DEFAULT_OBJECT_WIDTH_M = 0.3f // Default object width for unknown objects (30cm)

    // ARCore hit test configuration
    const val HIT_TEST_MAX_DISTANCE = 10f   // Maximum distance for hit test (10 meters)
    const val USE_ARCORE_DISTANCE = true    // Enable/disable ARCore distance measurement
}

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

/**
 * TensorFlow Lite Object Detector using the Task API
 * This implementation automatically extracts labels from the model metadata
 */
class TfLiteObjectDetector(
    private val context: Context,
    modelFileName: String = "2.tflite", // Change this to your model file name
    private val threshold: Float = 0.5f,
    private val maxResults: Int = 10,
    private val numThreads: Int = 4
) {

    private var objectDetector: org.tensorflow.lite.task.vision.detector.ObjectDetector? = null

    companion object {

        private const val TAG = "TfLiteObjectDetector"
    }

    init {
        setupDetector(modelFileName)
    }

    private fun setupDetector(modelFileName: String) {
        try {
            // Configure the object detector options
            val baseOptions = org.tensorflow.lite.task.core.BaseOptions.builder()
                .setNumThreads(numThreads)
                .build()

            val options = org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setMaxResults(maxResults)
                .setScoreThreshold(threshold)
                .build()

            // Create the object detector from model file
            objectDetector = org.tensorflow.lite.task.vision.detector.ObjectDetector.createFromFileAndOptions(
                context,
                modelFileName,
                options
            )

            Log.d(TAG, "Object detector initialized successfully with model: $modelFileName")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing object detector", e)
        }
    }

    /**
     * Detect objects in the given bitmap
     * Returns a list of DetectedObject with labels extracted from the model
     */
    fun detectObjects(bitmap: Bitmap): List<DetectedObject> {
        val detector = objectDetector ?: run {
            Log.e(TAG, "Object detector not initialized")
            return emptyList()
        }

        return try {
            // Create TensorImage from bitmap
            val tensorImage = org.tensorflow.lite.support.image.TensorImage.fromBitmap(bitmap)

            // Run object detection
            val results = detector.detect(tensorImage)

            // Convert Detection objects to our DetectedObject format
            results.map { detection ->
                val category = detection.categories.firstOrNull()
                DetectedObject(
                    label = category?.label ?: "Unknown",
                    confidence = category?.score ?: 0f,
                    boundingBox = detection.boundingBox,
                    distance = 0f // Will be calculated with ARCore
                )
            }.also { detectedObjects ->
                Log.d(TAG, "Detected ${detectedObjects.size} objects")
                detectedObjects.forEach { obj ->
                    Log.d(
                        TAG, "- ${obj.label}: ${(obj.confidence * 100).toInt()}% at " +
                            "[${obj.boundingBox.left}, ${obj.boundingBox.top}, " +
                            "${obj.boundingBox.right}, ${obj.boundingBox.bottom}]"
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during object detection", e)
            emptyList()
        }
    }

    /**
     * Update detected objects with ARCore distance measurements
     */
    fun updateObjectsWithARCoreDistances(
        frame: Frame,
        detectedObjects: List<DetectedObject>,
        imageWidth: Int,
        imageHeight: Int
    ): List<DetectedObject> {
        return detectedObjects.map { obj ->
            // Calculate center point of the bounding box
            val centerX = (obj.boundingBox.left + obj.boundingBox.right) / 2f
            val centerY = (obj.boundingBox.top + obj.boundingBox.bottom) / 2f

            // Calculate ARCore distance
            val arcoreDistance = calculateARCoreDistance(frame, centerX, centerY)

            // Use ARCore distance if available, otherwise use fallback estimation
            val finalDistance = if (arcoreDistance > 0) {
                arcoreDistance
            } else {
                // Fallback to size-based estimation
                estimateDistanceFromSize(obj.boundingBox, obj.label)
            }

            obj.copy(distance = finalDistance)
        }
    }

    /**
     * Calculate real distance using ARCore hit test
     */
    private fun calculateARCoreDistance(
        frame: Frame,
        screenX: Float,
        screenY: Float
    ): Float {
        return try {
            // Only use ARCore distance if enabled in config
            if (!ARDetectionConfig.USE_ARCORE_DISTANCE) {
                return -1f
            }

            // Perform hit test at the object's center point
            val hits = frame.hitTest(screenX, screenY)

            if (hits.isNotEmpty()) {
                val hit = hits[0]
//                val hitPose = hit.hitPose
//                val cameraPose = frame.camera.pose

                // Calculate distance between camera and hit point
//                val dx = hitPose.tx() - cameraPose.tx()
//                val dy = hitPose.ty() - cameraPose.ty()
//                val dz = hitPose.tz() - cameraPose.tz()

//                val distance = sqrt(dx * dx + dy * dy + dz * dz)

                // Validate distance
                if (hit.distance > 0 && hit.distance <= 10f) {
                    Log.d(TAG, "ARCore distance: ${String.format("%.2f", hit.distance)}m")
                    hit.distance
                } else {
                    -1f
                }
            } else {
                -1f
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating ARCore distance", e)
            -1f
        }
    }

    /**
     * Fallback distance estimation based on object size
     */
    private fun estimateDistanceFromSize(
        boundingBox: RectF,
        objectType: String
    ): Float {
        // Known object sizes in meters (width)
        val knownSizes = mapOf(
            "person" to 0.5f,
            "car" to 1.8f,
            "bicycle" to 0.6f,
            "motorcycle" to 0.8f,
            "bus" to 2.5f,
            "truck" to 2.4f,
            "bottle" to 0.07f,
            "cup" to 0.09f,
            "chair" to 0.5f,
            "laptop" to 0.35f,
            "cell phone" to 0.08f,
            "book" to 0.15f,
            "tv" to 1.0f,
            "monitor" to 0.5f,
            "keyboard" to 0.35f,
            "mouse" to 0.08f,
            "backpack" to 0.4f,
            "handbag" to 0.3f,
            "suitcase" to 0.6f,
            "dog" to 0.5f,
            "cat" to 0.3f,
            "bird" to 0.2f
        )

        val objectWidth = boundingBox.width()
        val assumedRealWidth = knownSizes[objectType.lowercase()] ?: 0.3f
        val focalLength = 800f // Approximate focal length in pixels

        return if (objectWidth > 0) {
            (assumedRealWidth * focalLength) / objectWidth
        } else {
            0f
        }
    }

    /**
     * Get available labels from the model
     * This is useful for debugging and understanding what the model can detect
     */
    fun getAvailableLabels(): List<String> {
        // The Task API automatically handles label extraction from model metadata
        // This method is for informational purposes
        return try {
            // Unfortunately, the Task API doesn't expose a direct way to get all labels
            // But they are automatically used when detection results are returned
            Log.d(TAG, "Labels are automatically extracted from model metadata")
            emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "Error getting labels", e)
            emptyList()
        }
    }

    /**
     * Clean up resources
     */
    fun close() {
        objectDetector?.close()
        objectDetector = null
        Log.d(TAG, "Object detector closed")
    }
}

@Composable
fun TiltIndicator(
    currentAngle: Float,
    targetAngle: Float = ARDetectionConfig.TARGET_ANGLE,
    isAvailable: Boolean = true,
    modifier: Modifier = Modifier
) {
    val isCorrectAngle = abs(currentAngle - targetAngle) < ARDetectionConfig.ANGLE_TOLERANCE
    val indicatorColor = if (!isAvailable) Color.Gray else if (isCorrectAngle) Color.Green else Color.Red

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
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
                else -> "↗ TILT TO ${ARDetectionConfig.TARGET_ANGLE.toInt()}°"
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
    var objectDetector by remember { mutableStateOf<TfLiteObjectDetector?>(null) }
    var arSession by remember { mutableStateOf<Session?>(null) }
    var currentFrame by remember { mutableStateOf<Frame?>(null) }
    var currentTiltAngle by remember { mutableStateOf(0f) }
    var tiltSensorManager by remember { mutableStateOf<TiltSensorManager?>(null) }
    var lastDetectionTime by remember { mutableStateOf(0L) }
    var isSensorAvailable by remember { mutableStateOf(true) }

    val targetAngle = ARDetectionConfig.TARGET_ANGLE
    val isCorrectAngle = isSensorAvailable && abs(currentTiltAngle - targetAngle) < ARDetectionConfig.ANGLE_TOLERANCE
    val detectionZoneColor = if (!isSensorAvailable) Color.Gray else if (isCorrectAngle) Color.Green else Color.Red

    // Initialize object detector with the EfficientDet model
    LaunchedEffect(Unit) {
        objectDetector = TfLiteObjectDetector(
            context = context,
            modelFileName = "2-metadata.tflite", // Make sure to place this file in assets folder
            threshold = ARDetectionConfig.MIN_CONFIDENCE,
            maxResults = 25,
            numThreads = 4
        )
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
                        (currentTime - lastDetectionTime) > ARDetectionConfig.DETECTION_INTERVAL_MS
                    ) {

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
                    .padding(top = 64.dp)
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
                        else -> "↗ TILT TO ${ARDetectionConfig.TARGET_ANGLE.toInt()}°"
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
                        .padding(top = 64.dp, start = 16.dp)
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
                        Toast.makeText(context, "Please tilt device to ${ARDetectionConfig.TARGET_ANGLE.toInt()}° first", Toast.LENGTH_SHORT).show()
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
                    !isCorrectAngle -> "Tilt device to ${ARDetectionConfig.TARGET_ANGLE.toInt()}° for detection"
                    detectedObjects.isNotEmpty() -> "${detectedObjects.size} objects detected"
                    isDetecting -> "Scanning..."
                    else -> "Hold steady - scanning every ${ARDetectionConfig.DETECTION_INTERVAL_MS / 1000} seconds"
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
    detector: TfLiteObjectDetector?,
    onObjectsDetected: (List<DetectedObject>) -> Unit
) {
    withContext(Dispatchers.IO) {
        try {
            val image = frame.acquireCameraImage()
            image?.let { cameraImage ->
                val bitmap = imageProxyToBitmap(cameraImage)

                // Get initial detections (without distance)
                val initialDetections = detector?.detectObjects(bitmap) ?: emptyList()

                // Update detections with ARCore distance measurements
                val detectionsWithDistance = detector?.updateObjectsWithARCoreDistances(
                    frame, initialDetections, bitmap.width, bitmap.height
                ) ?: emptyList()

                // Filter objects that are in the detection area (center of screen)
                val centerX = bitmap.width / 2f
                val centerY = bitmap.height / 2f
                val detectionWidth = bitmap.width * ARDetectionConfig.DETECTION_ZONE_WIDTH_RATIO
                val detectionHeight = bitmap.height * ARDetectionConfig.DETECTION_ZONE_HEIGHT_RATIO

                val detectionRect = RectF(
                    centerX - detectionWidth / 2,
                    centerY - detectionHeight / 2,
                    centerX + detectionWidth / 2,
                    centerY + detectionHeight / 2
                )

                // Only include objects that intersect with the detection zone
                val filteredObjects = detectionsWithDistance.filter { obj ->
                    RectF.intersects(obj.boundingBox, detectionRect)
                }

                Log.d("AR Detection", "Total detected: ${detectionsWithDistance.size}, In zone: ${filteredObjects.size}")

                // Log distance measurements for debugging
                filteredObjects.forEach { obj ->
                    Log.d("AR Distance", "${obj.label}: ${String.format("%.2f", obj.distance)}m")
                }

                withContext(Dispatchers.Main) {
                    onObjectsDetected(filteredObjects)
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