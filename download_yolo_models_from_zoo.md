# Download YOLO Models from Hailo Model Zoo

This guide explains how to download pre-compiled YOLO model HEF files from the Hailo Model Zoo repository.

## Overview

The Hailo Model Zoo provides pre-compiled HEF (Hailo Executable Format) files for various YOLO models. These models are available for different Hailo hardware architectures:
- **HAILO8L** (13 TOPS) - For Raspberry Pi AI Kit with Hailo-8L
- **HAILO8** (26 TOPS) - For Hailo-8 devices

## Prerequisites

- Internet connection
- `wget` command-line tool (usually pre-installed on Linux)
- `sudo` access (to install models to system directory)

## Step-by-Step Instructions

### Step 1: Clone the Hailo Model Zoo Repository

Clone the repository to access the documentation with download URLs:

```bash
cd ~/Desktop
git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
```

### Step 2: Find Model Download URLs

The download URLs are documented in the repository's documentation files. Check the object detection documentation:

```bash
# For HAILO8L models
cat docs/public_models/HAILO8L/HAILO8L_object_detection.rst | grep -i "yolov11"

# For HAILO8 models
cat docs/public_models/HAILO8/HAILO8_object_detection.rst | grep -i "yolov11"
```

The documentation shows multiple links for each model. In the "Links" column, you'll see abbreviations like this:

**Example from the documentation:**
```
`S <github-link>` `PT <pretrained-link>` `H <hef-link>` `PR <profiler-link>`
```

**What each abbreviation means:**
- **`S`** - **Source**: Link to the model's GitHub repository (for reference only)
- **`PT`** - **Pretrained**: Download the original PyTorch model file (.zip format). This is the raw model before compilation - we don't need this.
- **`H`** - **HEF**: This is the pre-compiled HEF file ready to use on Hailo hardware. **This is what we need to download!**
- **`PR`** - **Profiler Report**: Performance report (HTML file) showing model stats - optional, not needed for running the model

**For downloading models, you only need the `H` link** - that's the pre-compiled HEF file that works directly with your Hailo device.

### Step 3: Create Download Directory

Create a directory to temporarily store the downloaded files:

```bash
cd ~/Desktop/Detection
mkdir -p yolo11_models
cd yolo11_models
```

### Step 4: Download YOLOv11 Models

Download the models directly from the S3 URLs. The URLs follow this pattern:

```
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/{architecture}/{model_name}.hef
```

#### Download YOLOv11 for HAILO8L:

```bash
# YOLOv11 Nano (smallest, fastest)
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov11n.hef \
  -O yolov11n_h8l.hef

# YOLOv11 Small (balanced)
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov11s.hef \
  -O yolov11s_h8l.hef

# YOLOv11 Medium (optional - larger file)
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov11m.hef \
  -O yolov11m_h8l.hef

# YOLOv11 Large (optional - even larger)
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov11l.hef \
  -O yolov11l_h8l.hef

# YOLOv11 X-Large (optional - largest)
wget -q --show-progress \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov11x.hef \
  -O yolov11x_h8l.hef
```

#### Download YOLOv11 for HAILO8:

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

### Step 5: Verify Downloaded Files

Check that the files were downloaded successfully:

```bash
ls -lh *.hef
```

You should see files like:
```
yolov11n_h8l.hef  8.3M
yolov11s_h8l.hef   25M
yolov11n_h8.hef    10M
yolov11s_h8.hef    25M
```

### Step 6: Verify Model Architecture

Verify that each HEF file is compiled for the correct architecture:

```bash
# Check HAILO8L models
hailo parse-hef yolov11n_h8l.hef 2>/dev/null | grep -i "architecture"
# Should show: Architecture HEF was compiled for: HAILO8L

hailo parse-hef yolov11s_h8l.hef 2>/dev/null | grep -i "architecture"
# Should show: Architecture HEF was compiled for: HAILO8L

# Check HAILO8 models
hailo parse-hef yolov11n_h8.hef 2>/dev/null | grep -i "architecture"
# Should show: Architecture HEF was compiled for: HAILO8
```

### Step 7: Install Models to System Directory

Copy the models to the system-wide Hailo models directory:

```bash
sudo cp *.hef /usr/share/hailo-models/
```

Verify installation:

```bash
ls -lh /usr/share/hailo-models/yolov11*.hef
```

### Step 8: Update Your Application

Update your `detection_app.py` to use the new models:

```python
# HEF model paths (try H8L first, fallback to H8)
HEF_PATH_H8L = "/usr/share/hailo-models/yolov11n_h8l.hef"  # or yolov11s_h8l.hef
HEF_PATH_H8 = "/usr/share/hailo-models/yolov11n_h8.hef"    # or yolov11s_h8.hef
```

## Available YOLOv11 Models

| Model | Size (HAILO8L) | Size (HAILO8) | Description |
|-------|----------------|---------------|-------------|
| **yolov11n** | 8.3M | 10M | Nano - Smallest, fastest inference |
| **yolov11s** | 25M | 25M | Small - Balanced speed/accuracy |
| **yolov11m** | ~50M | ~50M | Medium - Better accuracy |
| **yolov11l** | ~80M | ~80M | Large - High accuracy |
| **yolov11x** | ~120M | ~120M | X-Large - Highest accuracy |

## Quick Download Script

You can use this script to download all YOLOv11 models at once:

```bash
#!/bin/bash
# Download all YOLOv11 models for HAILO8L

BASE_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l"
MODELS=("yolov11n" "yolov11s" "yolov11m" "yolov11l" "yolov11x")

mkdir -p yolo11_models
cd yolo11_models

for model in "${MODELS[@]}"; do
    echo "Downloading ${model}..."
    wget -q --show-progress "${BASE_URL}/${model}.hef" -O "${model}_h8l.hef"
done

echo "✓ All models downloaded!"
ls -lh *.hef
```

## Troubleshooting

### Model Architecture Mismatch Error

If you see an error like:
```
HEF format is not compatible with device. Device arch: HAILO8L, HEF arch: HAILO8
```

**Solution:** Make sure you're using the correct model for your device:
- For HAILO8L devices: Use `*_h8l.hef` files
- For HAILO8 devices: Use `*_h8.hef` files

### Download Fails

If `wget` fails, try:
1. Check your internet connection
2. Verify the URL is correct (check the documentation)
3. Try downloading without `--show-progress` flag:
   ```bash
   wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov11n.hef
   ```

### Permission Denied

If you get permission errors when copying to `/usr/share/hailo-models/`:
- Make sure you're using `sudo`
- Check that the directory exists: `ls -ld /usr/share/hailo-models/`

## Alternative: Download Other YOLO Versions

The same process works for other YOLO versions. Check the documentation for:
- YOLOv8 models
- YOLOv10 models
- YOLOv5 models
- YOLOv6 models

Example for YOLOv8:
```bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/yolov8s.hef
```

## Notes

- The HEF files are pre-compiled and ready to use - no compilation needed
- File sizes are approximate and may vary slightly
- Always verify the architecture matches your device before using
- The `yolo11_models` folder can be kept as a backup or deleted after installation

## References

- [Hailo Model Zoo GitHub](https://github.com/hailo-ai/hailo_model_zoo)
- [Hailo Model Zoo Documentation](https://github.com/hailo-ai/hailo_model_zoo/tree/master/docs)
- [Hailo Community Forum](https://community.hailo.ai/)

