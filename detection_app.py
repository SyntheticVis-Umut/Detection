#!/usr/bin/env python3
"""
Hailo YOLO8 Object Detection App
Uses Hailo SDK with YOLO8 model for real-time object detection from Raspberry Pi camera.
Displays detections with bounding boxes in real-time.
"""

import cv2
import json
import numpy as np
import time
import threading
import socket
import os
from pathlib import Path
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string
import sys

# Access Whisplay driver
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Whisplay", "Driver"))
if DRIVER_DIR not in sys.path:
    sys.path.append(DRIVER_DIR)
try:
    from WhisPlay import WhisPlayBoard
except Exception:
    WhisPlayBoard = None

# Hailo SDK imports
try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
        InputVStreamParams, OutputVStreamParams, FormatType
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("ERROR: Hailo SDK not available. Install hailo_platform package.")
    exit(1)

# Configuration defaults (overridden by config.json if present)
MODEL_INPUT_SIZE = 640  # YOLO8 input size
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
WEB_PORT = 8080  # Web server port
ENABLE_GUI = True  # Enable OpenCV GUI window (if available)
ENABLE_WEB = True  # Enable web preview
ENABLE_WHISPLAY = True  # Enable Whisplay display
RESOLUTION_DEFAULT = (2028, 1520)  # Default camera resolution
DETECTION_CLASSES = "all"  # "all" or comma-separated list e.g. "cat,dog"
FILE_PATH = None  # None for camera, or path to video file
CONFIG_PATH = Path(__file__).parent / "config.json"

# Whisplay constants
WHISPLAY_WIDTH = 240
WHISPLAY_HEIGHT = 280

# GUI availability state
GUI_AVAILABLE = False
GUI_CHECK_DONE = False

# Global frame buffer for web streaming
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
allowed_classes_set = None  # set of lower-case class names when filtering


def load_config():
    """Load configuration from config.json if present."""
    global CONFIDENCE_THRESHOLD, ENABLE_GUI, ENABLE_WEB, ENABLE_WHISPLAY, RESOLUTION_DEFAULT, DETECTION_CLASSES, FILE_PATH, allowed_classes_set
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to read config.json: {e}")
            cfg = {}
    else:
        cfg = {}
    
    CONFIDENCE_THRESHOLD = float(cfg.get("confidence_threshold", CONFIDENCE_THRESHOLD))
    ENABLE_GUI = bool(cfg.get("gui_preview", ENABLE_GUI))
    ENABLE_WEB = bool(cfg.get("web_preview", ENABLE_WEB))
    # Support both 'display_preview' (new) and 'whisplay_preview' (legacy) for backward compatibility
    ENABLE_WHISPLAY = bool(cfg.get("display_preview", cfg.get("whisplay_preview", ENABLE_WHISPLAY)))
    DETECTION_CLASSES = cfg.get("detection_classes", DETECTION_CLASSES)
    
    # File path for video input (null/None means camera)
    raw_file_path = cfg.get("file_path", None)
    if raw_file_path is None or (isinstance(raw_file_path, str) and raw_file_path.strip().lower() == "null"):
        FILE_PATH = None
    else:
        FILE_PATH = str(raw_file_path).strip()
        # Resolve relative paths
        if FILE_PATH and not Path(FILE_PATH).is_absolute():
            FILE_PATH = str(Path(__file__).parent / FILE_PATH)
    
    # Resolution
    res_cfg = cfg.get("resolution", {})
    try:
        w = int(res_cfg.get("width", RESOLUTION_DEFAULT[0]))
        h = int(res_cfg.get("height", RESOLUTION_DEFAULT[1]))
        RESOLUTION_DEFAULT = (w, h)
    except Exception:
        RESOLUTION_DEFAULT = RESOLUTION_DEFAULT
    
    # Detection class filter
    if isinstance(DETECTION_CLASSES, str) and DETECTION_CLASSES.strip().lower() != "all":
        allowed_classes_set = {c.strip().lower() for c in DETECTION_CLASSES.split(",") if c.strip()}
        if not allowed_classes_set:
            allowed_classes_set = None
            DETECTION_CLASSES = "all"
    else:
        allowed_classes_set = None
        DETECTION_CLASSES = "all"


def check_gui_available():
    """Check if OpenCV GUI is available."""
    global GUI_AVAILABLE, GUI_CHECK_DONE
    if GUI_CHECK_DONE:
        return GUI_AVAILABLE
    
    if not ENABLE_GUI:
        GUI_CHECK_DONE = True
        return False
    
    display = os.environ.get('DISPLAY')
    if not display:
        print("⚠️  DISPLAY not set. GUI disabled. (Set DISPLAY, e.g., export DISPLAY=:0)")
        GUI_AVAILABLE = False
        GUI_CHECK_DONE = True
        return False
    
    try:
        test_img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.namedWindow("__test_gui__", cv2.WINDOW_NORMAL)
        cv2.imshow("__test_gui__", test_img)
        cv2.waitKey(1)
        cv2.destroyWindow("__test_gui__")
        GUI_AVAILABLE = True
        GUI_CHECK_DONE = True
        print(f"✓ GUI available (DISPLAY={display})")
        return True
    except (cv2.error, Exception) as e:
        GUI_AVAILABLE = False
        GUI_CHECK_DONE = True
        print(f"⚠️  OpenCV GUI not available: {e}")
        print("   GUI will be disabled. Web interface will still work.")
        return False


def image_to_rgb565(frame: np.ndarray) -> list:
    """Convert OpenCV BGR frame to RGB565 byte list for Whisplay display."""
    # Resize to fit Whisplay
    resized = cv2.resize(frame, (WHISPLAY_WIDTH, WHISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Efficient conversion to RGB565 using numpy
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    
    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    
    # Convert to big-endian bytes
    high_byte = (rgb565 >> 8).astype(np.uint8)
    low_byte = (rgb565 & 0xFF).astype(np.uint8)
    
    # Stack and flatten
    pixel_data = np.stack((high_byte, low_byte), axis=2).flatten().tolist()
    return pixel_data

# HEF model paths (try H8L first, fallback to H8)
HEF_PATH_H8L = "/usr/share/hailo-models/yolov11n_h8l.hef"
HEF_PATH_H8 = "/usr/share/hailo-models/yolov11n_h8.hef"

# COCO class names (YOLO8 uses COCO dataset)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class HailoYOLO8Detector:
    """Hailo detector using YOLO8 model."""
    
    def __init__(self, hef_path: str):
        """Initialize Hailo detector with HEF model."""
        if not Path(hef_path).exists():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")
        
        self.hef_path = hef_path
        self.vdevice = None
        self.network_group = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.input_shape = None
        self.output_shape = None
        
        # Performance optimization: Reuse inference pipeline
        self.infer_pipeline = None
        self.network_group_params = None
        
        # Letterbox tracking for coordinate transformation
        self.original_frame_size = None
        self.letterbox_offset = (0, 0)
        self.letterbox_scale = 1.0
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Hailo device and load model."""
        print(f"Loading HEF model: {self.hef_path}")
        
        # Create VDevice
        self.vdevice = VDevice()
        
        # Load HEF
        hef = HEF(self.hef_path)
        
        # Configure network group
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, 
            interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.vdevice.configure(hef, configure_params)
        self.network_group = self.network_groups[0]
        
        # Get input/output stream info
        input_vstream_infos = hef.get_input_vstream_infos()
        output_vstream_infos = hef.get_output_vstream_infos()
        
        # Create vstream params
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32
        )
        
        # Get input shape
        if input_vstream_infos:
            self.input_shape = input_vstream_infos[0].shape
            print(f"Model input shape: {self.input_shape}")
        
        # Get output shape
        if output_vstream_infos:
            self.output_shape = output_vstream_infos[0].shape
            print(f"Model output shape: {self.output_shape}")
        
        # Create reusable inference pipeline for performance
        print("⚡ Creating reusable inference pipeline (performance optimization)...")
        self.network_group_params = self.network_group.create_params()
        self.infer_pipeline = InferVStreams(
            self.network_group, 
            self.input_vstreams_params, 
            self.output_vstreams_params
        )
        self.infer_pipeline.__enter__()  # Initialize the pipeline
        print("✓ Inference pipeline ready (reusable for maximum performance)")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLO8 input using letterbox resizing.
        Maintains aspect ratio by adding black padding if necessary.
        """
        # Store original frame dimensions for coordinate scaling
        original_h, original_w = frame.shape[:2]
        self.original_frame_size = (original_w, original_h)
        
        # Calculate scaling factor to fit frame into MODEL_INPUT_SIZE while maintaining aspect ratio
        scale = min(MODEL_INPUT_SIZE / original_w, MODEL_INPUT_SIZE / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize maintaining aspect ratio (using INTER_AREA for better performance/quality balance)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create letterbox (add black padding to center the image)
        letterboxed = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, frame.shape[2]), dtype=frame.dtype)
        
        # Calculate padding offsets to center the image
        pad_x = (MODEL_INPUT_SIZE - new_w) // 2
        pad_y = (MODEL_INPUT_SIZE - new_h) // 2
        
        # Place resized image in center of letterbox
        letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Store letterbox offsets for coordinate transformation
        self.letterbox_offset = (pad_x, pad_y)
        self.letterbox_scale = scale
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        if len(letterboxed.shape) == 3:
            letterboxed = np.expand_dims(letterboxed, axis=0)
        
        return letterboxed
    
    def _add_detection(self, detections: list, class_id: int, ymin: float, xmin: float, 
                      ymax: float, xmax: float, confidence: float):
        """Helper method to add a detection to the list."""
        # Convert normalized coordinates [0, 1] to pixel coordinates in letterboxed space (640x640)
        x_min_letterbox = xmin * MODEL_INPUT_SIZE
        y_min_letterbox = ymin * MODEL_INPUT_SIZE
        x_max_letterbox = xmax * MODEL_INPUT_SIZE
        y_max_letterbox = ymax * MODEL_INPUT_SIZE
        
        # Remove letterbox padding offset
        pad_x, pad_y = self.letterbox_offset
        x_min_resized = x_min_letterbox - pad_x
        y_min_resized = y_min_letterbox - pad_y
        x_max_resized = x_max_letterbox - pad_x
        y_max_resized = y_max_letterbox - pad_y
        
        # Scale back to original image size
        scale = self.letterbox_scale
        if self.original_frame_size:
            orig_w, orig_h = self.original_frame_size
            x_min_px = x_min_resized / scale
            y_min_px = y_min_resized / scale
            x_max_px = x_max_resized / scale
            y_max_px = y_max_resized / scale
            
            # Clamp to image boundaries
            x_min_px = max(0, min(x_min_px, orig_w))
            y_min_px = max(0, min(y_min_px, orig_h))
            x_max_px = max(0, min(x_max_px, orig_w))
            y_max_px = max(0, min(y_max_px, orig_h))
        else:
            # Fallback if original size not set
            x_min_px = x_min_resized / scale
            y_min_px = y_min_resized / scale
            x_max_px = x_max_resized / scale
            y_max_px = y_max_resized / scale
        
        width_px = x_max_px - x_min_px
        height_px = y_max_px - y_min_px
        
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        
        detections.append({
            'class_id': int(class_id),
            'class_name': class_name,
            'confidence': float(confidence),
            'bbox': {
                'x_min': float(x_min_px),
                'y_min': float(y_min_px),
                'x_max': float(x_max_px),
                'y_max': float(y_max_px),
                'width': float(width_px),
                'height': float(height_px)
            }
        })
    
    def _parse_single_detection(self, det, class_id: int, detections: list):
        """Parse a single detection from list or numpy array format."""
        try:
            if isinstance(det, (list, np.ndarray)):
                if len(det) >= 5:
                    # Extract values: [ymin, xmin, ymax, xmax, confidence]
                    ymin, xmin, ymax, xmax, confidence = [float(v) for v in det[:5]]
                    # Only add if confidence is above threshold
                    if confidence >= CONFIDENCE_THRESHOLD:
                        self._add_detection(detections, class_id, ymin, xmin, ymax, xmax, confidence)
        except (ValueError, TypeError, IndexError):
            pass
    
    def parse_detections(self, output: dict) -> list:
        """Parse YOLO8 detection output from Hailo NPU."""
        detections = []
        
        if not output:
            return detections
        
        # Get the output tensor (assuming single output)
        output_key = list(output.keys())[0]
        output_data = output[output_key]
        
        # Handle nested list structure (common with Hailo NMS output)
        if isinstance(output_data, list):
            # Remove batch dimension if present (usually size 1)
            if len(output_data) > 0 and isinstance(output_data[0], (list, np.ndarray)):
                if len(output_data) == 1:
                    output_data = output_data[0]  # Remove batch dimension
            
            # Iterate over each class
            for class_id, class_detections in enumerate(output_data):
                if class_detections is None or (isinstance(class_detections, (list, np.ndarray)) and len(class_detections) == 0):
                    continue
                
                # Handle as list of detections
                if isinstance(class_detections, list):
                    for det in class_detections:
                        if det is not None:
                            self._parse_single_detection(det, class_id, detections)
                
                # Handle as numpy array
                elif isinstance(class_detections, np.ndarray):
                    if len(class_detections.shape) == 1:
                        # Single detection: [ymin, xmin, ymax, xmax, confidence]
                        if len(class_detections) >= 5:
                            self._parse_single_detection(class_detections, class_id, detections)
                    elif len(class_detections.shape) == 2:
                        # Multiple detections: (num_detections, 5)
                        if class_detections.shape[0] == 5 and class_detections.shape[1] > 5:
                            # Format: (5, num_detections) - transpose
                            class_detections = class_detections.T
                        # Now should be (num_detections, 5)
                        for det in class_detections:
                            self._parse_single_detection(det, class_id, detections)
        
        # Handle numpy array format directly
        elif isinstance(output_data, np.ndarray):
            if len(output_data.shape) == 3:
                # Format: (num_classes, dim1, dim2) - iterate over classes
                num_classes = output_data.shape[0]
                for class_id in range(num_classes):
                    class_detections = output_data[class_id]
                    if len(class_detections.shape) == 2:
                        for det in class_detections:
                            self._parse_single_detection(det, class_id, detections)
                    elif len(class_detections.shape) == 1:
                        self._parse_single_detection(class_detections, class_id, detections)
            elif len(output_data.shape) == 2:
                # Format: (num_detections, 6) with class_id
                if output_data.shape[1] >= 6:
                    for det in output_data:
                        if len(det) >= 6:
                            ymin, xmin, ymax, xmax, confidence, class_id = det[:6]
                            if confidence >= CONFIDENCE_THRESHOLD:
                                class_id = int(class_id)
                                self._add_detection(detections, class_id, ymin, xmin, ymax, xmax, confidence)
        
        return detections
    
    def infer(self, frame: np.ndarray) -> list:
        """Run inference on a frame and return detections."""
        # Preprocess frame
        preprocessed = self.preprocess(frame)
        
        # Prepare input dict
        input_vstream_info = self.network_group.get_input_vstream_infos()[0]
        input_name = input_vstream_info.name
        input_data = {input_name: preprocessed}
        
        # Run inference using reusable pipeline (much faster than creating new one each frame)
        with self.network_group.activate(self.network_group_params):
            output = self.infer_pipeline.infer(input_data)
        
        # Parse detections
        detections = self.parse_detections(output)
        
        return detections
    
    def cleanup(self):
        """Clean up resources."""
        # Clean up inference pipeline
        if self.infer_pipeline is not None:
            try:
                self.infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
            self.infer_pipeline = None
        
        if self.vdevice:
            pass  # VDevice cleanup is handled automatically


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    # Picamera2 with RGB888 already matches OpenCV expectations on Pi (BGR-like in practice).
    # To avoid color tint issues, skip color channel swaps and draw directly.
    # Use copy only if we need to preserve original (for now, copy for safety)
    frame_with_boxes = frame.copy()
    
    # Color palette for different classes
    colors = [
        (0, 255, 0),    # Green for person
        (255, 0, 0),    # Blue for bicycle
        (0, 0, 255),    # Red for car
        (255, 255, 0),  # Cyan for motorcycle
        (255, 0, 255),  # Magenta for airplane
        (0, 255, 255),  # Yellow for bus
    ]
    
    for det in detections:
        if isinstance(det, dict):
            bbox = det.get('bbox', {})
            class_name = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0.0)
            class_id = det.get('class_id', 0)
            
            # Get coordinates
            x_min = int(bbox.get('x_min', 0))
            y_min = int(bbox.get('y_min', 0))
            x_max = int(bbox.get('x_max', 0))
            y_max = int(bbox.get('y_max', 0))
            
            # Clamp to frame boundaries
            frame_h, frame_w = frame_with_boxes.shape[:2]
            x_min = max(0, min(x_min, frame_w))
            y_min = max(0, min(y_min, frame_h))
            x_max = max(0, min(x_max, frame_w))
            y_max = max(0, min(y_max, frame_h))
            
            # Select color based on class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y_min - 10, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(frame_with_boxes, 
                         (x_min, label_y - label_size[1] - 5),
                         (x_min + label_size[0], label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame_with_boxes, label,
                       (x_min, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame_with_boxes


# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hailo YOLO8 Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .video-container {
            text-align: center;
            background-color: #000;
            padding: 10px;
            border-radius: 10px;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 5px;
        }
        .info {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            margin: 5px;
            background-color: #4CAF50;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Hailo YOLO8 Object Detection</h1>
        <div class="info">
            <strong>Confidence Threshold:</strong> {{ confidence_threshold }} | 
            <strong>Status:</strong> <span class="status">LIVE</span>
        </div>
        <div class="video-container">
            <img src="/video_feed" alt="Live Detection Stream">
        </div>
        <div class="info">
            <p><strong>Instructions:</strong> This page shows live camera feed with YOLO8 detection bounding boxes.</p>
            <p>Different colored boxes indicate different object classes detected in real-time.</p>
        </div>
    </div>
</body>
</html>
"""


def get_local_ip():
    """Get the local IP address of the device."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def generate_frames():
    """Generate MJPEG frames for video streaming."""
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                # Create a placeholder frame if no frame available
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


def create_flask_app():
    """Create and configure Flask app."""
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Serve the main HTML page."""
        return render_template_string(HTML_TEMPLATE,
                                    confidence_threshold=CONFIDENCE_THRESHOLD)
    
    @app.route('/video_feed')
    def video_feed():
        """Video streaming route."""
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return app


def start_web_server():
    """Start Flask web server in a separate thread."""
    app = create_flask_app()
    local_ip = get_local_ip()
    print(f"\n🌐 Web server starting...")
    print(f"   Access at: http://{local_ip}:{WEB_PORT}")
    print(f"   Or locally: http://localhost:{WEB_PORT}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, debug=False)


def video_capture_loop(picam2, cap, detector: HailoYOLO8Detector, enable_gui: bool, board: WhisPlayBoard = None):
    """Continuous video capture loop for smooth web streaming and optional GUI/Whisplay.
    
    Args:
        picam2: Picamera2 instance (for camera input) or None
        cap: cv2.VideoCapture instance (for video file input) or None
        detector: HailoYOLO8Detector instance
        enable_gui: Whether to show GUI window
        board: WhisPlayBoard instance or None
    """
    global latest_frame, latest_detections
    
    print("\nStarting video capture loop...")
    print("Display outputs:")
    if enable_gui:
        print("  ✓ GUI window enabled")
    if board and ENABLE_WHISPLAY:
        print("  ✓ Whisplay display enabled")
    if ENABLE_WEB:
        print("  ✓ Web interface enabled")
    if not (enable_gui or (board and ENABLE_WHISPLAY) or ENABLE_WEB):
        print("  ⚠️  No display outputs enabled")
    print("Performance optimizations: Active")
    print("=" * 60)
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    resolution_logged = False
    using_video_file = cap is not None
    
    window_name = "Hailo YOLO Detection"
    if enable_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception as e:
            print(f"✗ Failed to create GUI window: {e}")
            enable_gui = False
    
    while True:
        try:
            # Capture frame from either camera or video file
            if using_video_file:
                ret, frame = cap.read()
                if not ret:
                    print("\n✓ End of video file reached. Looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                # OpenCV reads in BGR format, keep as-is
            else:
                frame = picam2.capture_array("main")
            
            # Run inference
            detections = detector.infer(frame)
            
            # Filter detections by allowed classes if configured
            if allowed_classes_set is not None:
                detections = [
                    det for det in detections
                    if isinstance(det, dict)
                    and det.get("class_name", "").lower() in allowed_classes_set
                ]
            
            # Draw detections
            frame_with_boxes = draw_detections(frame, detections)
            
            # Log resolutions once
            if not resolution_logged:
                cam_h, cam_w = frame.shape[:2]
                out_h, out_w = frame_with_boxes.shape[:2]
                match = (cam_h == out_h) and (cam_w == out_w)
                print(f"Camera input: {cam_w}x{cam_h}, output: {out_w}x{out_h} (match: {match})")
                resolution_logged = True
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30.0 / elapsed
                fps_start_time = time.time()
                print(f"FPS: {fps:.1f} | Detections: {len(detections)}")
            
            # Add FPS and detection count to frame
            info_text = f"FPS: {fps:.1f} | Detections: {len(detections)}"
            cv2.putText(frame_with_boxes, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update global frame buffer for web streaming
            with frame_lock:
                latest_frame = frame_with_boxes
                latest_detections = detections
            
            # Send to Whisplay display (only if enabled)
            if board is not None and ENABLE_WHISPLAY:
                try:
                    pixel_data = image_to_rgb565(frame_with_boxes)
                    board.draw_image(0, 0, WHISPLAY_WIDTH, WHISPLAY_HEIGHT, pixel_data)
                except Exception as e:
                    print(f"✗ Whisplay display error: {e}")
                    board = None  # Disable on error
            
            # GUI display and controls
            if enable_gui:
                try:
                    cv2.imshow(window_name, frame_with_boxes)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuit requested from GUI window.")
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"detection_{timestamp}.jpg"
                        cv2.imwrite(filename, frame_with_boxes)
                        print(f"Saved: {filename}")
                except (cv2.error, Exception) as e:
                    print(f"✗ GUI error: {e}. Disabling GUI.")
                    enable_gui = False
            
            # Remove artificial delay - let the system run at maximum speed
            # The inference and display operations naturally limit the frame rate
            
        except Exception as e:
            print(f"Error in video loop: {e}")
            time.sleep(1)
            # If GUI loop should stop on errors, break to allow cleanup
            if enable_gui:
                break
    
    # Cleanup GUI window
    if enable_gui:
        try:
            cv2.destroyAllWindows()
            print("✓ GUI window closed")
        except Exception as e:
            print(f"⚠️  Error closing GUI window: {e}")


def main():
    """Main application loop."""
    global RESOLUTION_DEFAULT
    load_config()
    
    print("=" * 60)
    print("Hailo YOLO Object Detection App (YOLOv8/YOLOv11)")
    print("=" * 60)
    print(f"Input source: {'Video file: ' + FILE_PATH if FILE_PATH else 'Camera'}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Resolution: {RESOLUTION_DEFAULT[0]}x{RESOLUTION_DEFAULT[1]}")
    print(f"Web preview: {'Enabled' if ENABLE_WEB else 'Disabled'}")
    print(f"GUI preview: {'Enabled' if ENABLE_GUI else 'Disabled'}")
    print(f"Display preview (Whisplay): {'Enabled' if ENABLE_WHISPLAY else 'Disabled'}")
    if allowed_classes_set is None:
        print("Detection classes: all")
    else:
        print(f"Detection classes filter: {sorted(allowed_classes_set)}")
    print("=" * 60)
    
    if not HAILO_AVAILABLE:
        print("\n✗ ERROR: Hailo SDK not available!")
        print("Please install hailo_platform package.")
        return
    
    # Find HEF file
    hef_path = None
    model_name = "Unknown"
    if Path(HEF_PATH_H8L).exists():
        hef_path = HEF_PATH_H8L
        model_name = Path(hef_path).stem
        print(f"\n📦 Model: {model_name} (Hailo-8L)")
        print(f"   Path: {hef_path}")
    elif Path(HEF_PATH_H8).exists():
        hef_path = HEF_PATH_H8
        model_name = Path(hef_path).stem
        print(f"\n📦 Model: {model_name} (Hailo-8)")
        print(f"   Path: {hef_path}")
    else:
        print(f"\n✗ ERROR: HEF file not found!")
        print(f"Expected at: {HEF_PATH_H8L} or {HEF_PATH_H8}")
        return
    
    # Initialize Hailo detector
    try:
        detector = HailoYOLO8Detector(hef_path)
    except Exception as e:
        print(f"\n✗ ERROR initializing Hailo detector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize input source (camera or video file)
    picam2 = None
    cap = None
    
    if FILE_PATH:
        # Video file input
        print(f"\n📹 Opening video file: {FILE_PATH}")
        try:
            if not Path(FILE_PATH).exists():
                print(f"\n✗ ERROR: Video file not found: {FILE_PATH}")
                detector.cleanup()
                return
            
            cap = cv2.VideoCapture(FILE_PATH)
            if not cap.isOpened():
                print(f"\n✗ ERROR: Could not open video file: {FILE_PATH}")
                detector.cleanup()
                return
            
            # Get video resolution and override config resolution
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            RESOLUTION_DEFAULT = (video_width, video_height)
            
            print(f"✓ Video file opened successfully")
            print(f"   Resolution: {video_width}x{video_height}")
            print(f"   FPS: {video_fps:.1f}")
            print(f"   Config resolution overridden to match video")
            
        except Exception as e:
            print(f"\n✗ ERROR opening video file: {e}")
            detector.cleanup()
            import traceback
            traceback.print_exc()
            return
    else:
        # Camera input
        print("\n📷 Initializing Raspberry Pi camera...")
        try:
            picam2 = Picamera2()
            
            # Configure camera for video capture
            config = picam2.create_video_configuration(
                main={"size": (RESOLUTION_DEFAULT[0], RESOLUTION_DEFAULT[1]), "format": "RGB888"},
                buffer_count=2
            )
            picam2.configure(config)
            picam2.start()
            
            # Allow camera to stabilize
            time.sleep(2)
            print("✓ Camera ready!")
            
        except Exception as e:
            print(f"\n✗ ERROR initializing camera: {e}")
            detector.cleanup()
            import traceback
            traceback.print_exc()
            return
    
    # Start web server in a separate thread (if enabled)
    if ENABLE_WEB:
        web_thread = threading.Thread(target=start_web_server, daemon=True)
        web_thread.start()
    
    # Decide GUI availability
    gui_available = check_gui_available() and ENABLE_GUI
    
    # Initialize Whisplay
    board = None
    if ENABLE_WHISPLAY and WhisPlayBoard is not None:
        try:
            print("\nInitializing Whisplay display...")
            board = WhisPlayBoard()
            board.set_backlight(80)
            print("✓ Whisplay display initialized!")
        except Exception as e:
            print(f"✗ Failed to initialize Whisplay: {e}")
            board = None

    if gui_available:
        print("\n✓ GUI available - starting detection loop with GUI window...")
        print("Press 'q' in GUI window or Ctrl+C to stop")
        print("Press 's' in GUI window to save current frame")
        print("=" * 60)
        try:
            video_capture_loop(picam2, cap, detector, enable_gui=True, board=board)
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
    else:
        # Start video capture loop in a separate thread (web/whisplay)
        video_thread = threading.Thread(target=video_capture_loop, args=(picam2, cap, detector, False, board), daemon=True)
        video_thread.start()
        
        time.sleep(2)  # Give threads time to start
        
        print("\nStarting detection loop (web only)...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
    
    # Cleanup
    print("\nCleaning up...")
    try:
        if picam2:
            picam2.stop()
        if cap:
            cap.release()
        detector.cleanup()
        if board:
            board.cleanup()
    finally:
        print("Done.")


if __name__ == "__main__":
    main()

