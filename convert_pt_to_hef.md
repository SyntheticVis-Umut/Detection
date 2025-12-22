# Converting PyTorch (.pt) YOLO8 Models to HEF Format

This guide explains how to convert your custom YOLO8 PyTorch models (`.pt` files) to HEF format for use with Hailo NPU.

## Overview

The conversion process involves:
1. **Export PyTorch model to ONNX** (`.onnx`)
2. **Use Hailo Dataflow Compiler (DFC)** to convert ONNX to HEF

## Prerequisites

1. **Hailo Dataflow Compiler (DFC)**
   - Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/) (registration required)
   - The DFC is typically installed on a development machine (not necessarily the Raspberry Pi)
   - Available for Linux x86_64

2. **Python environment** with:
   - PyTorch
   - Ultralytics YOLO (for YOLO8 models)
   - ONNX

## Step 1: Export PyTorch Model to ONNX

First, convert your `.pt` model to ONNX format:

```python
from ultralytics import YOLO
import torch

# Load your trained YOLO8 model
model = YOLO('your_model.pt')  # or 'yolov8s.pt', 'yolov8n.pt', etc.

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

**Alternative method (if using raw PyTorch):**

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

## Step 2: Convert ONNX to HEF using Hailo DFC

The Hailo Dataflow Compiler (DFC) converts ONNX models to HEF format.

### Install Hailo DFC

1. Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/)
2. Extract and follow installation instructions
3. Typically located at: `/opt/hailo/hailo_dataflow_compiler/` or similar

### Convert ONNX to HEF

```bash
# Basic conversion command
hailo compile your_model.onnx --output your_model.hef

# For Hailo-8L (13 TOPS)
hailo compile your_model.onnx --output your_model_h8l.hef --target hailo8l

# For Hailo-8 (26 TOPS)
hailo compile your_model.onnx --output your_model_h8.hef --target hailo8

# With calibration dataset for quantization (recommended)
hailo compile your_model.onnx \
    --output your_model.hef \
    --calibration-dataset path/to/calibration/images \
    --calibration-batch-size 32
```

### Calibration Dataset

For best accuracy, provide a calibration dataset:
- 100-1000 representative images from your use case
- Same resolution as model input (typically 640x640)
- Images should represent your typical inference scenarios

## Step 3: Use the HEF File

Once you have the `.hef` file, place it in your models directory:

```bash
# Copy to standard location
sudo cp your_model.hef /usr/share/hailo-models/

# Or use it directly in your code
# Update detection_app.py:
HEF_PATH_CUSTOM = "/path/to/your_model.hef"
```

## Important Notes

1. **Model Compatibility**: Not all YOLO8 operations are supported by Hailo. Some models may need modifications.

2. **Quantization**: Hailo automatically quantizes models to INT8. Provide a calibration dataset for best accuracy.

3. **Input/Output Format**: 
   - Input: RGB, normalized to [0, 1] or [0, 255] (check your model)
   - Output: May need post-processing (NMS, coordinate transformation)

4. **Performance**: Hailo-8L vs Hailo-8 have different performance characteristics. Compile for your specific hardware.

## Troubleshooting

### Model Not Supported
If conversion fails, the model may use unsupported operations. Check:
- Use standard YOLO8 architecture (avoid custom layers)
- Ensure all operations are in ONNX opset 11/12
- Simplify the model if possible

### Accuracy Issues
- Provide a good calibration dataset
- Try different quantization strategies
- Check input preprocessing matches training

### Resources
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) - Pre-compiled models
- [Hailo Developer Zone](https://hailo.ai/developer-zone/) - Documentation and tools
- [Hailo Community Forum](https://community.hailo.ai/) - Support and discussions

## Quick Reference

```bash
# Full workflow example
python3 export_to_onnx.py          # Step 1: Create ONNX
hailo compile model.onnx model.hef # Step 2: Create HEF
sudo cp model.hef /usr/share/hailo-models/
python3 detection_app.py           # Step 3: Use it
```


