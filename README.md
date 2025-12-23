# Hailo YOLO Object Detection App

Real-time object detection application using Hailo SDK with YOLO models (YOLOv8/YOLOv11) on Raspberry Pi with Whisplay display support.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Models](#models)
- [Training & Model Conversion](#training--model-conversion)
- [Hailo Model Zoo](#hailo-model-zoo)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- ✅ **Real-time object detection** using Hailo NPU acceleration
- ✅ **YOLOv11 & YOLOv8 support** - Latest YOLO models with high accuracy
- ✅ **Raspberry Pi camera integration** - Native picamera2 support
- ✅ **Whisplay display support** - Live preview on Whisplay LCD (240x280)
- ✅ **Multiple display options** - GUI window, web interface, and Whisplay
- ✅ **Class filtering** - Filter detections by specific object classes
- ✅ **Live visualization** - Bounding boxes, labels, and confidence scores
- ✅ **FPS counter** - Real-time performance monitoring
- ✅ **Frame capture** - Save detection frames on demand

## Requirements

### Hardware
- Raspberry Pi 5 with Hailo NPU (Hailo-8L or Hailo-8)
- Raspberry Pi camera module
- Whisplay display (optional, for LCD preview)

### Software
- Raspberry Pi OS
- Python 3.10+
- Hailo SDK (`hailo_platform`)
- Python packages (see `requirements.txt`):
  - `picamera2` - Camera interface
  - `hailo_platform` - Hailo SDK
  - `opencv-python` - Image processing
  - `flask` - Web server
  - `numpy` - Array operations
  - `PIL` (Pillow) - Image processing for Whisplay

## Installation

### 1. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv python3-numpy python3-pil
```

### 2. Install Python Dependencies

```bash
cd /home/raspberrypi-user/Desktop/Detection
pip install -r requirements.txt --break-system-packages
```

### 3. Install Hailo SDK

Follow the official Hailo SDK installation instructions for Raspberry Pi 5.

### 4. Download Models (Optional)

If you need to download additional models, see [Downloading Models](#downloading-models) section below.

## Quick Start

```bash
cd /home/raspberrypi-user/Desktop/Detection
python3 detection_app.py
```

The app will:
1. Load the YOLO model (automatically detects Hailo-8L or Hailo-8)
2. Initialize the camera
3. Start detection loop
4. Display on available outputs (GUI, Web, Whisplay)

## Usage

### Display Options

#### 1. Whisplay Display (240x280 LCD)
- Automatically enabled if Whisplay hardware is detected
- Shows live detection preview with bounding boxes
- Optimized for small display size
- Configure in `config.json`: `"whisplay_preview": true`

#### 2. GUI Window (OpenCV)
- Opens if `DISPLAY` environment variable is set
- **Controls:**
  - **'q'** - Quit application
  - **'s'** - Save current frame with detections

#### 3. Web Interface
- Access at: `http://localhost:8080` or `http://[device-ip]:8080`
- Live streaming with MJPEG
- Works from any device on the network
- Shows FPS, detection count, and bounding boxes
- Configure in `config.json`: `"web_preview": true`

### Controls

- **'q'** (in GUI window) or **Ctrl+C** - Quit the application
- **'s'** (in GUI window) - Save current frame with detections to a file

## Configuration

Edit `config.json` to customize behavior:

```json
{
  "confidence_threshold": 0.5,
  "resolution": { "width": 2028, "height": 1520 },
  "web_preview": false,
  "gui_preview": true,
  "whisplay_preview": true,
  "detection_classes": "cat,person"
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `confidence_threshold` | float | 0.5 | Minimum confidence (0.0-1.0) to show detections |
| `resolution.width` | int | 2028 | Camera width |
| `resolution.height` | int | 1520 | Camera height |
| `web_preview` | bool | true | Enable web server at http://localhost:8080 |
| `gui_preview` | bool | true | Enable OpenCV GUI window |
| `whisplay_preview` | bool | true | Enable Whisplay LCD display |
| `detection_classes` | string | "all" | Comma-separated class filter (e.g., "cat,person") |

## Project Structure

```
Detection/
├── README.md                 # Complete documentation (this file)
├── detection_app.py          # Main application
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
│
└── models/                   # Downloaded model files
    ├── yolov11n_h8l.hef     # YOLOv11 Nano for Hailo-8L
    ├── yolov11n_h8.hef      # YOLOv11 Nano for Hailo-8
    ├── yolov11s_h8l.hef     # YOLOv11 Small for Hailo-8L
    ├── yolov11s_h8.hef      # YOLOv11 Small for Hailo-8
    └── downloaded_models/   # Backup storage
```

## Models

### Current Models

The application automatically detects and uses models from `/usr/share/hailo-models/`:

- **YOLOv11n** (Nano) - Fastest, smallest model
  - `yolov11n_h8l.hef` (10M) - For Hailo-8L
  - `yolov11n_h8.hef` (8.6M) - For Hailo-8
- **YOLOv11s** (Small) - Balanced speed/accuracy
  - `yolov11s_h8l.hef` (25M) - For Hailo-8L
  - `yolov11s_h8.hef` (19M) - For Hailo-8
- **YOLOv8s** (Legacy) - Also supported
  - `yolov8s_h8l.hef` (35M) - For Hailo-8L
  - `yolov8s_h8.hef` (10M) - For Hailo-8

**Note:** Downloaded model files are stored in `models/` directory, but the app reads from `/usr/share/hailo-models/` (system directory).

### Downloading Models

To download additional YOLO models from Hailo Model Zoo:

#### Step 1: Clone the Hailo Model Zoo Repository

```bash
cd ~/Desktop
git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
```

#### Step 2: Find Model Download URLs

The download URLs are documented in the repository's documentation files:

```bash
# For HAILO8L models
cat docs/public_models/HAILO8L/HAILO8L_object_detection.rst | grep -i "yolov11"

# For HAILO8 models
cat docs/public_models/HAILO8/HAILO8_object_detection.rst | grep -i "yolov11"
```

The documentation shows multiple links for each model. In the "Links" column, you'll see:
- **`S`** - Source: Link to the model's GitHub repository (for reference only)
- **`PT`** - Pretrained: Download the original PyTorch model file (.zip format) - not needed
- **`H`** - **HEF**: This is the pre-compiled HEF file ready to use. **This is what we need!**
- **`PR`** - Profiler Report: Performance report (HTML file) - optional

**For downloading models, you only need the `H` link** - that's the pre-compiled HEF file.

#### Step 3: Download Models

Create a directory and download the models:

```bash
cd ~/Desktop/Detection
mkdir -p models
cd models
```

Download YOLOv11 for HAILO8L:

```bash
# YOLOv11 Nano (smallest, fastest)
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov11n.hef \
  -O yolov11n_h8l.hef

# YOLOv11 Small (balanced)
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov11s.hef \
  -O yolov11s_h8l.hef
```

Download YOLOv11 for HAILO8:

```bash
# YOLOv11 Nano
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov11n.hef \
  -O yolov11n_h8.hef

# YOLOv11 Small
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov11s.hef \
  -O yolov11s_h8.hef
```

#### Step 4: Verify Model Architecture

Verify that each HEF file is compiled for the correct architecture:

```bash
# Check HAILO8L models
hailo parse-hef yolov11n_h8l.hef 2>/dev/null | grep -i "architecture"
# Should show: Architecture HEF was compiled for: HAILO8L

# Check HAILO8 models
hailo parse-hef yolov11n_h8.hef 2>/dev/null | grep -i "architecture"
# Should show: Architecture HEF was compiled for: HAILO8
```

#### Step 5: Install Models to System Directory

Copy the models to the system-wide Hailo models directory:

```bash
sudo cp *.hef /usr/share/hailo-models/
```

Verify installation:

```bash
ls -lh /usr/share/hailo-models/yolov11*.hef
```

### Changing Models

Edit `detection_app.py` lines 165-166:

```python
HEF_PATH_H8L = "/usr/share/hailo-models/yolov11n_h8l.hef"  # Change model here
HEF_PATH_H8 = "/usr/share/hailo-models/yolov11n_h8.hef"
```

### Available YOLOv11 Models

| Model | Size (HAILO8L) | Size (HAILO8) | Description |
|-------|----------------|---------------|-------------|
| **yolov11n** | 8.3M | 10M | Nano - Smallest, fastest inference |
| **yolov11s** | 25M | 25M | Small - Balanced speed/accuracy |
| **yolov11m** | ~50M | ~50M | Medium - Better accuracy |
| **yolov11l** | ~80M | ~80M | Large - High accuracy |
| **yolov11x** | ~120M | ~120M | X-Large - Highest accuracy |

## Detection Classes

The app detects **80 COCO classes** including:

**People & Animals:** person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Objects:** backpack, umbrella, handbag, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket

**Food:** banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture:** chair, couch, potted plant, bed, dining table, toilet

**Electronics:** tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster

**Other:** bottle, wine glass, cup, fork, knife, spoon, bowl, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### Filtering Classes

Set in `config.json`:
```json
{
  "detection_classes": "cat,person,dog"
}
```

Or set to `"all"` to detect all 80 classes.

## Training & Model Conversion

### Converting Custom Models

If you have a custom YOLO model (`.pt` file) and want to use it with Hailo NPU, you need to convert it to HEF format. The conversion process involves:

1. **Export PyTorch model to ONNX** (`.onnx`)
2. **Use Hailo Dataflow Compiler (DFC)** to convert ONNX to HEF

### Prerequisites

1. **Hailo Dataflow Compiler (DFC)**
   - Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/) (registration required)
   - The DFC is typically installed on a development machine (not necessarily the Raspberry Pi)
   - Available for Linux x86_64
   - Typically located at: `/opt/hailo/hailo_dataflow_compiler/` or similar

2. **Python environment** with:
   - PyTorch
   - Ultralytics YOLO (for YOLO8/YOLOv11 models)
   - ONNX

**Note:** Model compilation and training are typically done on a development machine (x86_64 with GPU), then the HEF files are transferred to Raspberry Pi for use.

### Step 1: Export PyTorch Model to ONNX

#### Method 1: Using Ultralytics CLI (Recommended)

The simplest way is to use the Ultralytics YOLO CLI directly:

```bash
yolo export model=/path/to/your_model.pt imgsz=640 format=onnx opset=11
```

**Command options:**
- `model` - Path to your trained YOLO model (.pt file)
- `imgsz` - Input image size (default: 640)
- `format` - Export format: `onnx`
- `opset` - ONNX opset version (11 or 12, recommended for Hailo)

**Examples:**
```bash
# Export YOLOv8s model
yolo export model=yolov8s.pt imgsz=640 format=onnx opset=11

# Export custom trained model
yolo export model=/path/to/trained/best.pt imgsz=640 format=onnx opset=11
```

**Note:** More export options are available. See [Ultralytics export documentation](https://docs.ultralytics.com/modes/export/).

#### Method 2: Using Ultralytics Python API

You can also export programmatically using Ultralytics:

```python
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('your_model.pt')  # or 'yolov8s.pt', 'yolov11n.pt', etc.

# Export to ONNX
# For Hailo, use opset_version=11 or 12
model.export(
    format='onnx',
    imgsz=640,  # Input size (must match your model)
    opset=11,   # ONNX opset version
    simplify=True,  # Simplify ONNX model
    dynamic=False,  # Static shapes (required for Hailo)
    half=False      # FP32 (Hailo will quantize)
)

# This will create 'your_model.onnx'
```

#### Method 3: Using Raw PyTorch

If using raw PyTorch (without Ultralytics):

```python
import torch
import torch.onnx

# Load your model
model = torch.load('your_model.pt')
model.eval()

# Create dummy input (batch_size=1, channels=3, height=640, width=640)
dummy_input = torch.randn(1, 3, 640, 640)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'your_model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None  # Static shapes for Hailo
)
```

### Step 2: Convert ONNX to HEF using Hailo DFC

The Hailo Dataflow Compiler (DFC) converts ONNX models to HEF format. This step requires the **Hailo Dataflow Compiler (DFC)** on a development machine (typically not on Raspberry Pi).

**Requirements:**
- Hailo Dataflow Compiler (DFC) - Available from Hailo (customer access required)
- Development machine with GPU (x86_64 Linux)
- CUDA support for compilation

#### Basic Conversion

```bash
# Basic conversion command
hailo compile your_model.onnx --output your_model.hef

# For Hailo-8L (13 TOPS)
hailo compile your_model.onnx --output your_model_h8l.hef --target hailo8l

# For Hailo-8 (26 TOPS)
hailo compile your_model.onnx --output your_model_h8.hef --target hailo8
```

#### Conversion with Calibration Dataset (Recommended)

For best accuracy, provide a calibration dataset:

```bash
hailo compile your_model.onnx \
    --output your_model.hef \
    --calibration-dataset path/to/calibration/images \
    --calibration-batch-size 32
```

**Calibration Dataset Requirements:**
- 100-1000 representative images from your use case
- Same resolution as model input (typically 640x640)
- Images should represent your typical inference scenarios

### Step 3: Install and Use the HEF File

Once you have the `.hef` file, transfer it to your Raspberry Pi and install:

```bash
# Copy to standard location
sudo cp your_model.hef /usr/share/hailo-models/

# Or use it directly in your code
# Update detection_app.py:
HEF_PATH_CUSTOM = "/path/to/your_model.hef"
```

### Important Notes

1. **Model Compatibility**: Not all YOLO operations are supported by Hailo. Some models may need modifications.
   - Use standard YOLO architecture (avoid custom layers)
   - Ensure all operations are in ONNX opset 11/12
   - Simplify the model if possible

2. **Quantization**: Hailo automatically quantizes models to INT8. Provide a calibration dataset for best accuracy.

3. **Input/Output Format**: 
   - Input: RGB, normalized to [0, 1] or [0, 255] (check your model)
   - Output: May need post-processing (NMS, coordinate transformation)

4. **Performance**: Hailo-8L vs Hailo-8 have different performance characteristics. Compile for your specific hardware.

### Troubleshooting Model Conversion

#### Model Not Supported

If conversion fails, the model may use unsupported operations. Check:
- Use standard YOLO architecture (avoid custom layers)
- Ensure all operations are in ONNX opset 11/12
- Simplify the model if possible

#### Accuracy Issues

- Provide a good calibration dataset
- Try different quantization strategies
- Check input preprocessing matches training

### Quick Reference Workflow

```bash
# Full workflow example
yolo export model=your_model.pt imgsz=640 format=onnx opset=11                   # Step 1: Create ONNX
hailo compile your_model.onnx --output your_model.hef --target hailo8l            # Step 2: Create HEF (on dev machine)
sudo cp your_model.hef /usr/share/hailo-models/                                    # Step 3: Install (on Raspberry Pi)
python3 detection_app.py                                                          # Step 4: Use it
```

### Additional Resources

- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) - Pre-compiled models
- [Hailo Developer Zone](https://hailo.ai/developer-zone/) - Documentation and tools
- [Hailo Community Forum](https://community.hailo.ai/) - Support and discussions

## Hailo Model Zoo

### What is Hailo Model Zoo?

The **Hailo Model Zoo** is a comprehensive repository that provides tools, models, and documentation for working with Hailo AI accelerators.

### Main Purpose

The Hailo Model Zoo serves as a **one-stop resource** for:
1. **Pre-trained models** - Ready-to-use AI models
2. **Model compilation tools** - Convert models to HEF format
3. **Model optimization** - Optimize models for Hailo hardware
4. **Training scripts** - Retrain models on custom datasets
5. **Documentation** - Complete guides and model information

### What You Can Do With It

#### 1. Download Pre-compiled Models (What We Did)
- Find download URLs for pre-compiled HEF files
- Browse available models for different Hailo devices
- Get model information (accuracy, FPS, size, etc.)

**Example:** We used it to find YOLOv11 download URLs

#### 2. Compile Your Own Models (Requires Hailo Dataflow Compiler)
If you have the Hailo Dataflow Compiler (DFC), you can:
- **Parse**: Convert ONNX/TensorFlow models to Hailo's internal format
- **Optimize**: Quantize models for better performance
- **Compile**: Generate HEF files from your own models
- **Profile**: Get performance reports before deployment

#### 3. Get Model Information
Query information about any model in the zoo:
```bash
hailomz info yolov11n
hailomz info mobilenet_v1
```

#### 4. Retrain Models on Custom Data

The repository includes training scripts and instructions for:
- YOLOv5, YOLOv8, YOLOv11
- Custom object detection
- Custom classification
- Fine-tuning for your specific use case

**How to Train Models Using Hailo Model Zoo:**

**Prerequisites:**
- Development machine with GPU (x86_64 Linux)
- Docker and nvidia-docker2 installed
- CUDA support

**Step 1: Clone and Navigate to Training Directory**

```bash
cd ~/Desktop/hailo_model_zoo/training/yolov8  # or yolov5, yolov11, etc.
```

**Step 2: Build Docker Image**

```bash
docker build --build-arg timezone=`cat /etc/timezone` -t yolov8:v0 .
```

**Step 3: Prepare Your Dataset**

- Create a `dataset.yaml` configuration file
- Prepare labels in YOLO format (one `.txt` file per image)
- Include number of classes in the YAML: `nc: 80`
- Follow [Ultralytics dataset format guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

**Step 4: Start Docker Container**

```bash
docker run --name "yolov8_training" -it --gpus all --ipc=host \
  -v /path/to/your/dataset:/workspace/dataset \
  yolov8:v0
```

**Step 5: Train the Model**

Inside the Docker container:

```bash
yolo detect train data=dataset.yaml model=yolov8s.pt name=my_custom_model epochs=100 batch=16
```

**Step 6: Export to ONNX**

After training completes:

```bash
yolo export model=/path/to/trained/best.pt imgsz=640 format=onnx opset=11
```

**Step 7: Compile to HEF**

Use Hailo Model Zoo to compile:

```bash
hailomz compile --ckpt best.onnx --calib-path /path/to/calibration/images \
  --yaml path/to/yolov8s.yaml
```

**Step 8: Transfer to Raspberry Pi**

```bash
# On development machine
scp best.hef user@raspberrypi:/tmp/

# On Raspberry Pi
sudo cp /tmp/best.hef /usr/share/hailo-models/
```

**Detailed Instructions:**
- YOLOv8: See `hailo_model_zoo/training/yolov8/README.rst`
- YOLOv5: See `hailo_model_zoo/training/yolov5/README.rst`
- YOLOv11: Similar process, check training directory for specific README

### For Raspberry Pi Users

**What you CAN do:**
- ✅ Browse documentation and find download URLs
- ✅ Download pre-compiled HEF files
- ✅ Use the models in your applications

**What you CANNOT do (on Raspberry Pi):**
- ❌ Compile new models (needs DFC + GPU)
- ❌ Optimize models (needs DFC)
- ❌ Run training (needs GPU)

**Note:** Model compilation and training are typically done on a development machine (x86_64 with GPU), then the HEF files are transferred to Raspberry Pi for use.

## Performance

- **FPS:** 20-30 FPS (depending on resolution and number of detections)
- **Model:** YOLOv11n (nano) - Fastest option
- **Input resolution:** 640x640 (letterboxed from camera resolution)
- **Latency:** ~30-50ms per frame (including inference + display)

### Performance Tips

- Use YOLOv11n for maximum speed
- Reduce camera resolution for higher FPS
- Increase confidence threshold to filter more detections
- Disable unused display outputs (web/GUI) if only using Whisplay

## Troubleshooting

### Model Not Found

**Error:** `HEF file not found`

**Solution:**
```bash
# Check if models exist
ls -lh /usr/share/hailo-models/yolov11*.hef

# If missing, download and install (see Downloading Models section)
```

### Architecture Mismatch

**Error:** `HEF format is not compatible with device. Device arch: HAILO8L, HEF arch: HAILO8`

**Solution:** Use the correct model for your device:
- For Hailo-8L: Use `*_h8l.hef` files
- For Hailo-8: Use `*_h8.hef` files

Verify architecture:
```bash
hailo parse-hef /usr/share/hailo-models/yolov11n_h8l.hef | grep -i "architecture"
```

### GUI Not Available

**Error:** `OpenCV GUI not available`

**Solution:**
- The web interface will still work - access at: `http://localhost:8080`
- To enable GUI: `sudo apt-get install libgtk2.0-dev pkg-config` and rebuild OpenCV

### Camera Not Found

**Error:** Camera initialization fails

**Solution:**
```bash
# Check camera connection
libcamera-hello --list-cameras

# Fix permissions
sudo usermod -a -G video $USER
# Logout and login again
```

### Whisplay Display Not Working

**Error:** Whisplay display not initializing

**Solution:**
- Check Whisplay hardware connection
- Verify driver is accessible: `ls -la ../Whisplay/Driver/WhisPlay.py`
- Check if display is enabled in `config.json`: `"whisplay_preview": true`

### Low FPS

**Solutions:**
- Use YOLOv11n (nano) instead of larger models
- Reduce camera resolution in `config.json`
- Increase confidence threshold to filter more detections
- Disable unused display outputs

### Hailo SDK Not Found

**Error:** `Hailo SDK not available`

**Solution:**
- Verify Hailo SDK installation: `python3 -c "import hailo_platform; print('OK')"`
- Check Hailo device: `hailortcli fw-control identify`
- Ensure Hailo NPU is properly connected

## License

This project is provided as-is for educational and development purposes.

## References

- [Hailo Model Zoo GitHub](https://github.com/hailo-ai/hailo_model_zoo)
- [Hailo Community Forum](https://community.hailo.ai/)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
