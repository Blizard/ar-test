# AR Object Detection App

This Android application demonstrates AR capabilities with TensorFlow Lite object detection using Jetpack Compose and Material Design 3.

## Features

1. **AR Scene Integration**: Uses ARSceneView for AR capabilities
2. **Object Detection**: TensorFlow Lite model (1.tflite) for real-time object recognition
3. **Detection Zone**: Red rectangle overlay indicating the detection area
4. **Real-time Results**: Shows detected objects with confidence scores and estimated distances
5. **Photo Capture**: Take photos using AR camera with `frame.acquireCameraImage()`
6. **Material Design 3**: Modern UI with Compose

## Technical Implementation

### Dependencies Added
- `io.github.sceneview:arsceneview:2.3.0` - AR Scene composable
- `org.tensorflow:tensorflow-lite:2.16.1` - TensorFlow Lite runtime
- `org.tensorflow:tensorflow-lite-support:2.16.1` - TensorFlow Lite support library
- `com.google.accompanist:accompanist-permissions:0.34.0` - Permission handling
- Camera2 libraries for image processing

### Key Components

#### ARObjectDetectionScreen
Main composable that:
- Handles camera permissions
- Uses native ARScene composable (no AndroidView wrapper needed)
- Overlays detection rectangle in center of screen  
- Shows detected objects with distance calculations
- Provides camera capture functionality

#### TensorFlowObjectDetector
Class responsible for:
- Loading the 1.tflite model from assets
- Processing camera frames for object detection
- Returning detected objects with bounding boxes and confidence scores
- Estimating distances using object size and camera focal length

#### Detection Features
- **Detection Zone**: Red rectangle overlay (250dp x 180dp) in screen center
- **Object Filtering**: Only shows objects detected within the red zone
- **Distance Calculation**: Uses object pixel size and assumed real-world dimensions
- **Real-time Processing**: Analyzes frames continuously without blocking UI

### File Structure
```
app/src/main/
├── assets/
│   └── 1.tflite                    # TensorFlow Lite model
├── java/com/example/artest/
│   ├── MainActivity.kt             # Main activity
│   ├── ARObjectDetectionScreen.kt  # AR composable with detection
│   └── ui/theme/                   # Material Design 3 theme
└── AndroidManifest.xml             # Permissions and AR metadata
```

### Permissions Required
- `CAMERA` - For AR camera access
- `WRITE_EXTERNAL_STORAGE` - For saving photos (API ≤ 28)
- `READ_EXTERNAL_STORAGE` - For reading saved photos (API ≤ 32)
- `READ_MEDIA_IMAGES` - For media access (API ≥ 33)

### AR Core Features
- `android.hardware.camera.ar` - AR capabilities
- `android.hardware.camera.autofocus` - Camera autofocus
- ARCore metadata with required flag

## Setup Instructions

1. **Prerequisites**:
   - Android device with ARCore support
   - API level 24+ (ARCore requirement)
   - Physical device (AR doesn't work in emulator)

2. **Build & Run**:
   ```bash
   ./gradlew assembleDebug
   ```

3. **Testing**:
   - Grant camera permission when prompted
   - Point camera at objects
   - Objects in red detection zone will be highlighted
   - Tap camera button to capture photos

## Model Integration

The app uses a TensorFlow Lite model (`1.tflite`) for object detection. The current implementation includes:
- Model loading from assets
- Image preprocessing (resize to 320x320)
- Post-processing of detection results
- COCO dataset labels (80 classes)

To replace with your own model:
1. Replace `1.tflite` in `app/src/main/assets/`
2. Update input size and labels in `TensorFlowObjectDetector`
3. Adjust output tensor parsing if needed

## Distance Calculation

Distance estimation uses the formula:
```
distance = (real_world_width × focal_length) / pixel_width
```

Where:
- `real_world_width`: Assumed physical size of detected object
- `focal_length`: Estimated camera focal length (800px)
- `pixel_width`: Object width in pixels

## UI Components

### Detection Overlay
- Red rounded rectangle marking detection zone
- Status text showing detection count
- Semi-transparent background for visibility

### Object Info Panel
- Cards showing detected objects
- Object name, confidence percentage, distance
- Green color coding for successful detections
- Scrollable list for multiple objects

### Camera Controls
- Floating action button for photo capture
- Material Design 3 styling
- Toast notifications for capture feedback

## Troubleshooting

1. **AR not working**: Ensure device supports ARCore
2. **No detections**: Check model file is in assets folder
3. **Camera permission denied**: Grant permission in app settings
4. **Photo save fails**: Check storage permissions

## Architecture

The app follows MVVM pattern with:
- **View**: Composable UI components
- **ViewModel**: State management (using Compose state)
- **Model**: TensorFlow Lite detector and AR session handling

Built with modern Android development practices:
- Jetpack Compose for UI
- Kotlin Coroutines for async operations
- Material Design 3 theming
- AndroidX Camera for image processing
