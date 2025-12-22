# Hailo YOLO8 Object Detection App

Real-time object detection application using Hailo SDK with YOLO8 model on Raspberry Pi.

## Features

- ✅ Real-time object detection using Hailo NPU
- ✅ YOLO8 model support (yolov8s)
- ✅ Raspberry Pi camera integration (picamera2)
- ✅ Live visualization with bounding boxes and labels
- ✅ FPS counter and detection statistics
- ✅ Save frames on demand

## Requirements

- Raspberry Pi 5 with Hailo NPU
- Raspberry Pi camera module
- Python packages:
  - `picamera2` - Camera interface
  - `hailo_platform` - Hailo SDK
  - `opencv-python` - Image processing
- `flask` - Web server for live streaming
  - `numpy` - Array operations

## Installation

```bash
# Install required packages
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv python3-numpy

# Install Hailo SDK (if not already installed)
# Follow Hailo SDK installation instructions for your system
```

## Usage

```bash
cd /home/raspberrypi-user/Desktop/Detection
python3 detection_app.py
```

The app will start a web server and display the detection feed in your browser.

### Display Options

#### GUI Window (if available)
- Opens an OpenCV window showing the detection feed
- **'q'** - Quit the application (or Ctrl+C in terminal)
- **'s'** - Save current frame with detections to a file

#### Web Interface

Once the app is running, access the live detection stream at:
- **http://[device-ip]:8080** (from other devices on the network)
- **http://localhost:8080** (from the Raspberry Pi itself)

The web interface shows:
- Live camera feed with YOLO8 detection bounding boxes
- Different colored boxes for different object classes
- Real-time FPS and detection count
- Detection statistics

### Controls

- **'q'** (in GUI window) or **Ctrl+C** - Quit the application
- **'s'** (in GUI window) - Save current frame with detections to a file

## Configuration

Edit these variables in `detection_app.py`:

- `CONFIDENCE_THRESHOLD` - Minimum confidence (0.0-1.0) to show detections (default: 0.5)
- `MODEL_INPUT_SIZE` - Model input size (default: 640)

## Model Files

The application automatically detects and uses:
- `/usr/share/hailo-models/yolov8s_h8l.hef` (for Hailo-8L)
- `/usr/share/hailo-models/yolov8s_h8.hef` (for Hailo-8)

## Detection Classes

The app detects 80 COCO classes including:
- person, bicycle, car, motorcycle, airplane, bus, train, truck
- bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- backpack, umbrella, handbag, suitcase, frisbee, skis, snowboard
- sports ball, kite, baseball bat, skateboard, surfboard, tennis racket
- bottle, wine glass, cup, fork, knife, spoon, bowl
- banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza
- chair, couch, potted plant, bed, dining table, toilet, tv, laptop
- mouse, remote, keyboard, cell phone, microwave, oven, toaster
- sink, refrigerator, book, clock, vase, scissors, teddy bear
- hair drier, toothbrush

## Performance

- Typical FPS: 20-30 FPS (depending on resolution and number of detections)
- Model: YOLOv8s (small variant for fast inference)
- Input resolution: 640x640 (letterboxed from camera resolution)

## Troubleshooting

### GUI not available
- If you see "OpenCV GUI not available", the GUI window won't open
- The web interface will still work - access it at http://localhost:8080
- To enable GUI, install: `sudo apt-get install libgtk2.0-dev pkg-config` and rebuild OpenCV

### Camera not found
- Ensure camera is properly connected
- Check camera permissions: `sudo usermod -a -G video $USER` (logout/login required)

### Hailo SDK not found
- Verify Hailo SDK installation
- Check that HEF files exist in `/usr/share/hailo-models/`

### Low FPS
- Reduce camera resolution in the code
- Increase confidence threshold to filter more detections

## License

This project is provided as-is for educational and development purposes.

