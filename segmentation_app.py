#!/usr/bin/env python3
"""
Hailo YOLO8 Instance Segmentation App
Uses Hailo SDK with YOLOv8 segmentation model for real-time instance segmentation.
Displays detections with bounding boxes and segmentation masks in real-time.
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

def vectorized_nms(preds, iou_thres):
    """
    Optimized NMS using vectorized NumPy operations.
    This provides high performance on Raspberry Pi without needing compiled C extensions.
    """
    if preds.shape[0] == 0:
        return np.array([], dtype=np.int32)
    
    boxes = preds[:, :4]
    scores = preds[:, 4]
    
    # Vectorized area calculation
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Vectorized IoU calculation for all remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        # Vectorized filtering
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)

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
FRAME_SKIP = 1  # Skip frames to reduce CPU load
DISPLAY_SCALE = 1.0  # Scale factor for display operations (0.5 = half size, reduces mask overlay work by 4x)
SHOW_BOXES_ONLY = False  # If True, show only cropped bounding boxes instead of full frame
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

# Predefined colors for mask overlay (BGR format)
MASK_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0)   # Olive
]


def load_config():
    """Load configuration from config.json if present."""
    global CONFIDENCE_THRESHOLD, ENABLE_GUI, ENABLE_WEB, ENABLE_WHISPLAY, RESOLUTION_DEFAULT, DETECTION_CLASSES, FILE_PATH, FRAME_SKIP, DISPLAY_SCALE, allowed_classes_set, SHOW_BOXES_ONLY
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
    FRAME_SKIP = int(cfg.get("frame_skip", FRAME_SKIP))
    DISPLAY_SCALE = float(cfg.get("display_scale", DISPLAY_SCALE))
    
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
    
    # Box-only preview mode
    SHOW_BOXES_ONLY = bool(cfg.get("show_boxes_only", SHOW_BOXES_ONLY))


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

# HEF model paths for YOLOv8n segmentation (try H8L first, fallback to H8)
HEF_PATH_H8L = "/usr/share/hailo-models/yolov8n_seg_h8l.hef"
HEF_PATH_H8 = "/usr/share/hailo-models/yolov8n_seg_h8.hef"

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


class HailoYOLO8Segmentation:
    """Hailo detector using YOLOv8 segmentation model."""
    
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

        # Debug flag for logging model outputs
        self.debug_logged = False
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Hailo device and load model."""
        print(f"Loading HEF model: {self.hef_path}")
        
        # Create VDevice
        try:
            self.vdevice = VDevice()
        except Exception as e:
            error_msg = str(e)
            if "HAILO_OUT_OF_PHYSICAL_DEVICES" in error_msg or "74" in error_msg:
                print("\n✗ ERROR: Hailo device is already in use!")
                print("   Another application (e.g., detection_app.py) may be using the Hailo NPU.")
                print("   Please close other Hailo applications and try again.")
                raise RuntimeError("Hailo device unavailable - already in use by another application") from e
            raise
        
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
            print(f"Number of output streams: {len(output_vstream_infos)}")
        
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
    
    def _transform_coordinates(self, x_min: float, y_min: float, x_max: float, y_max: float):
        """Transform coordinates from model output space to original frame space."""
        pad_x, pad_y = self.letterbox_offset
        scale = self.letterbox_scale
        orig_w, orig_h = self.original_frame_size
        
        # Remove letterbox padding and scale back
        x_min = (x_min - pad_x) / scale
        y_min = (y_min - pad_y) / scale
        x_max = (x_max - pad_x) / scale
        y_max = (y_max - pad_y) / scale
        
        # Clamp to original frame dimensions
        x_min = max(0, min(x_min, orig_w))
        y_min = max(0, min(y_min, orig_h))
        x_max = max(0, min(x_max, orig_w))
        y_max = max(0, min(y_max, orig_h))
        
        return int(x_min), int(y_min), int(x_max), int(y_max)
    
    def _process_mask(self, mask_coeffs: np.ndarray, mask_proto: np.ndarray, bbox: tuple, orig_shape: tuple):
        """
        Process segmentation mask from coefficients and prototype.
        
        Args:
            mask_coeffs: Mask coefficients (32 values)
            mask_proto: Mask prototype (160x160x32)
            bbox: Bounding box (x_min, y_min, x_max, y_max) in original coordinates
            orig_shape: Original frame shape (height, width)
        
        Returns:
            Resized mask array matching original frame dimensions
        """
        # Combine mask coefficients with prototype
        # mask_proto shape: (160, 160, 32)
        # mask_coeffs shape: (32,)
        mask = np.tensordot(mask_proto, mask_coeffs, axes=([2], [0]))  # Result: (160, 160)
        
        # Apply sigmoid activation
        mask = 1.0 / (1.0 + np.exp(-mask))
        
        # Crop mask to bounding box region
        x_min, y_min, x_max, y_max = bbox
        # Convert to 160x160 coordinates
        scale_x = 160.0 / MODEL_INPUT_SIZE
        scale_y = 160.0 / MODEL_INPUT_SIZE
        
        bbox_x_min = int(x_min * scale_x)
        bbox_y_min = int(y_min * scale_y)
        bbox_x_max = int(x_max * scale_x)
        bbox_y_max = int(y_max * scale_y)
        
        # Clamp to mask dimensions
        bbox_x_min = max(0, min(bbox_x_min, 160))
        bbox_y_min = max(0, min(bbox_y_min, 160))
        bbox_x_max = max(0, min(bbox_x_max, 160))
        bbox_y_max = max(0, min(bbox_y_max, 160))
        
        if bbox_x_max > bbox_x_min and bbox_y_max > bbox_y_min:
            cropped_mask = mask[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max]
        else:
            cropped_mask = mask
        
        # Resize mask to bounding box size in original frame
        orig_h, orig_w = orig_shape
        bbox_w = max(1, x_max - x_min)
        bbox_h = max(1, y_max - y_min)
        
        if cropped_mask.size > 0:
            resized_mask = cv2.resize(cropped_mask, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_mask = np.zeros((bbox_h, bbox_w), dtype=np.float32)
        
        return resized_mask
    
    def parse_detections(self, output: dict) -> list:
        """
        Parse YOLOv8 segmentation model outputs.
        YOLOv8 segmentation outputs typically include:
        - Detection boxes
        - Class scores
        - Mask coefficients
        - Mask prototypes
        """
        detections = []

        # Debug log the output keys and shapes once to help map tensors
        if not self.debug_logged:
            print("\n[DEBUG] Model outputs (keys and shapes):")
            for k, v in output.items():
                shape = None
                if hasattr(v, "shape"):
                    shape = v.shape
                elif isinstance(v, list):
                    shape = f"list(len={len(v)})"
                print(f"  - {k}: {shape}")
            self.debug_logged = True

        # Map tensors by name
        try:
            boxes_s8   = output.get("yolov8n_seg/conv44")  # (1, 80,80,64)
            scores_s8  = output.get("yolov8n_seg/conv45")  # (1, 80,80,80)
            masks_s8   = output.get("yolov8n_seg/conv46")  # (1, 80,80,32)

            boxes_s16  = output.get("yolov8n_seg/conv60")  # (1, 40,40,64)
            scores_s16 = output.get("yolov8n_seg/conv61")  # (1, 40,40,80)
            masks_s16  = output.get("yolov8n_seg/conv62")  # (1, 40,40,32)

            boxes_s32  = output.get("yolov8n_seg/conv73")  # (1, 20,20,64)
            scores_s32 = output.get("yolov8n_seg/conv74")  # (1, 20,20,80)
            masks_s32  = output.get("yolov8n_seg/conv75")  # (1, 20,20,32)

            mask_proto = output.get("yolov8n_seg/conv48")  # (1, 160,160,32)
        except Exception:
            return detections

        if mask_proto is None:
            return detections

        mask_proto = mask_proto[0]  # (160,160,32)
        reg_max = 16  # 64 channels / 4 directions

        def dfl_decode(dfl: np.ndarray):
            # dfl shape: (N,4,reg_max)
            # softmax implementation (numpy)
            e = np.exp(dfl - np.max(dfl, axis=2, keepdims=True))
            prob = e / np.sum(e, axis=2, keepdims=True)
            idx = np.arange(reg_max, dtype=np.float32)
            dist = np.sum(prob * idx, axis=2)
            return dist

        def decode_scale(boxes, scores, masks, stride):
            if boxes is None or scores is None or masks is None:
                return None
            b = boxes[0]   # (H,W,64)
            s = scores[0]  # (H,W,80)
            m = masks[0]   # (H,W,32)

            h, w = b.shape[:2]

            # reshape
            b = b.reshape(h, w, 4, reg_max).reshape(-1, 4, reg_max)
            s = s.reshape(-1, s.shape[2])
            m = m.reshape(-1, m.shape[2])

            # grid centers
            ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            xs = (xs.reshape(-1) + 0.5) * stride
            ys = (ys.reshape(-1) + 0.5) * stride

            # distances
            dist = dfl_decode(b) * stride  # (N,4)
            l, t, r, bttm = dist[:,0], dist[:,1], dist[:,2], dist[:,3]
            x_min = xs - l
            y_min = ys - t
            x_max = xs + r
            y_max = ys + bttm

            # scores
            cls_scores = 1 / (1 + np.exp(-s))  # sigmoid
            cls_conf = cls_scores.max(axis=1)
            cls_id = cls_scores.argmax(axis=1)

            # filter by threshold
            keep = cls_conf >= CONFIDENCE_THRESHOLD
            if not np.any(keep):
                return None

            boxes_out = np.stack([x_min, y_min, x_max, y_max], axis=1)[keep]
            scores_out = cls_conf[keep]
            cls_out = cls_id[keep]
            masks_out = m[keep]

            return boxes_out, scores_out, cls_out, masks_out

        decoded = []
        for tensors, stride in [
            ((boxes_s8, scores_s8, masks_s8), 8),
            ((boxes_s16, scores_s16, masks_s16), 16),
            ((boxes_s32, scores_s32, masks_s32), 32),
        ]:
            res = decode_scale(*tensors, stride)
            if res is not None:
                decoded.append(res)

        if not decoded:
            return detections

        boxes_all = np.concatenate([d[0] for d in decoded], axis=0)
        scores_all = np.concatenate([d[1] for d in decoded], axis=0)
        cls_all = np.concatenate([d[2] for d in decoded], axis=0)
        masks_all = np.concatenate([d[3] for d in decoded], axis=0)

        # Native NMS (Optimized Vectorized NumPy)
        # Format: [x1, y1, x2, y2, score] for vectorized_nms
        preds = np.hstack([boxes_all.astype(np.float32), scores_all[:, None].astype(np.float32)])
        keep_inds = vectorized_nms(preds, iou_thres=0.45)
        
        # Limit to top 100 detections
        if keep_inds.shape[0] > 100:
            keep_inds = keep_inds[:100]
        
        boxes_all = boxes_all[keep_inds]
        scores_all = scores_all[keep_inds]
        cls_all = cls_all[keep_inds]
        masks_all = masks_all[keep_inds]

        # Generate masks and transform boxes to original frame (vectorized where possible)
        orig_h, orig_w = self.original_frame_size[1], self.original_frame_size[0]
        scale_x = 160.0 / MODEL_INPUT_SIZE
        scale_y = 160.0 / MODEL_INPUT_SIZE
        
        # Vectorized coordinate transformation (batch process all boxes)
        boxes_transformed = []
        for box in boxes_all:
            x_min, y_min, x_max, y_max = box
            x_min_i, y_min_i, x_max_i, y_max_i = self._transform_coordinates(x_min, y_min, x_max, y_max)
            boxes_transformed.append((x_min_i, y_min_i, x_max_i, y_max_i))
        
        # Batch process masks (vectorized mask composition)
        # Vectorized mask composition using einsum (native compiled NumPy operation)
        if len(masks_all) > 0:
            # Batch compose all masks at once using einsum (compiled C operation)
            # mask_proto: (160, 160, 32), masks_all: (N, 32)
            # Result: (N, 160, 160)
            full_masks = np.einsum('ijk, nk -> nij', mask_proto, masks_all)
            full_masks = 1.0 / (1.0 + np.exp(-full_masks))  # sigmoid (vectorized)
            
            # Vectorized bbox calculations
            boxes_transformed_array = np.array(boxes_transformed)
            bbox_widths = boxes_transformed_array[:, 2] - boxes_transformed_array[:, 0]
            bbox_heights = boxes_transformed_array[:, 3] - boxes_transformed_array[:, 1]
            valid_mask = (bbox_widths > 0) & (bbox_heights > 0)
            
            # Process masks (vectorized where possible, minimal Python loops)
            for i, (box, score, cls_id, (x_min_i, y_min_i, x_max_i, y_max_i)) in enumerate(
                zip(boxes_all, scores_all, cls_all, boxes_transformed)
            ):
                if not valid_mask[i]:
                    mask_crop = None
                else:
                    bbox_w = int(bbox_widths[i])
                    bbox_h = int(bbox_heights[i])
                    
                    # Map bbox to proto coords
                    px_min = max(0, min(int(box[0] * scale_x), 160))
                    py_min = max(0, min(int(box[1] * scale_y), 160))
                    px_max = max(0, min(int(box[2] * scale_x), 160))
                    py_max = max(0, min(int(box[3] * scale_y), 160))
                    
                    if px_max > px_min and py_max > py_min:
                        mask_crop_proto = full_masks[i, py_min:py_max, px_min:px_max]
                    else:
                        mask_crop_proto = full_masks[i]
                    
                    # Resize mask to bbox size (OpenCV is compiled C)
                    mask_crop = cv2.resize(mask_crop_proto, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
                
                class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"

                detections.append({
                    'class_id': int(cls_id),
                    'class_name': class_name,
                    'confidence': float(score),
                    'bbox': {
                        'x_min': float(x_min_i),
                        'y_min': float(y_min_i),
                        'x_max': float(x_max_i),
                        'y_max': float(y_max_i),
                        'width': float(bbox_widths[i]),
                        'height': float(bbox_heights[i])
                    },
                    'mask': mask_crop
                })

        return detections
    
    def infer(self, frame: np.ndarray) -> list:
        """Run inference on a frame and return detections with masks."""
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


def draw_segmentation(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes, labels, and segmentation masks on frame.
    Based on hailo-rpi5-examples instance_segmentation.py mask rendering.
    """
    frame_with_overlay = frame.copy()
    alpha = 0.5  # Mask overlay transparency
    
    for idx, det in enumerate(detections):
        if isinstance(det, dict):
            bbox = det.get('bbox', {})
            class_name = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0.0)
            class_id = det.get('class_id', 0)
            mask = det.get('mask', None)
            
            # Get coordinates
            x_min = int(bbox.get('x_min', 0))
            y_min = int(bbox.get('y_min', 0))
            x_max = int(bbox.get('x_max', 0))
            y_max = int(bbox.get('y_max', 0))
            
            # Clamp to frame boundaries
            frame_h, frame_w = frame_with_overlay.shape[:2]
            x_min = max(0, min(x_min, frame_w))
            y_min = max(0, min(y_min, frame_h))
            x_max = max(0, min(x_max, frame_w))
            y_max = max(0, min(y_max, frame_h))
            
            # Select color based on class ID
            color = MASK_COLORS[class_id % len(MASK_COLORS)]
            
            # Draw bounding box
            cv2.rectangle(frame_with_overlay, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw mask if available
            if mask is not None and isinstance(mask, np.ndarray):
                # Ensure mask matches bbox dimensions
                mask_h, mask_w = mask.shape[:2]
                bbox_h = y_max - y_min
                bbox_w = x_max - x_min
                
                if mask_h != bbox_h or mask_w != bbox_w:
                    mask = cv2.resize(mask, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
                
                # Threshold mask
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # Create colored mask overlay
                mask_overlay = np.zeros((bbox_h, bbox_w, 3), dtype=np.uint8)
                mask_overlay[mask_binary > 0] = color
                
                # Apply mask overlay to frame
                roi = frame_with_overlay[y_min:y_max, x_min:x_max]
                if roi.shape[:2] == mask_overlay.shape[:2]:
                    frame_with_overlay[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                        roi, 1.0, mask_overlay, alpha, 0
                    )
            
            # Draw label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame_with_overlay,
                (x_min, y_min - text_height - baseline - 5),
                (x_min + text_width, y_min),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame_with_overlay,
                label,
                (x_min, y_min - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    return frame_with_overlay


def extract_boxes_grid(frame: np.ndarray, detections: list, max_boxes: int = 12, box_size: int = 320) -> np.ndarray:
    """
    Extract bounding boxes from frame and arrange them in a grid layout.
    Shows only the cropped content of each detection box (with masks if available).
    
    Args:
        frame: Original video frame
        detections: List of detection dictionaries with bbox and mask info
        max_boxes: Maximum number of boxes to display
        box_size: Size of each box in the grid (width and height)
    
    Returns:
        Grid image with cropped bounding boxes arranged in rows/columns
    """
    if not detections:
        # Return a placeholder if no detections
        placeholder = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No detections", (50, box_size // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder
    
    # Limit number of boxes
    detections = detections[:max_boxes]
    
    # Calculate grid dimensions
    num_boxes = len(detections)
    cols = int(np.ceil(np.sqrt(num_boxes)))
    rows = int(np.ceil(num_boxes / cols))
    
    # Create grid canvas
    grid_h = rows * box_size
    grid_w = cols * box_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    frame_h, frame_w = frame.shape[:2]
    alpha = 0.5  # Mask overlay transparency
    
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
            
        bbox = det.get('bbox', {})
        class_name = det.get('class_name', 'unknown')
        confidence = det.get('confidence', 0.0)
        class_id = det.get('class_id', 0)
        mask = det.get('mask', None)
        
        # Get coordinates
        x_min = int(bbox.get('x_min', 0))
        y_min = int(bbox.get('y_min', 0))
        x_max = int(bbox.get('x_max', 0))
        y_max = int(bbox.get('y_max', 0))
        
        # Clamp to frame boundaries
        x_min = max(0, min(x_min, frame_w))
        y_min = max(0, min(y_min, frame_h))
        x_max = max(0, min(x_max, frame_w))
        y_max = max(0, min(y_max, frame_h))
        
        # Ensure valid box dimensions
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Extract box region from frame
        box_region = frame[y_min:y_max, x_min:x_max].copy()
        box_h, box_w = box_region.shape[:2]
        
        if box_h == 0 or box_w == 0:
            continue
        
        # Apply mask if available
        if mask is not None and isinstance(mask, np.ndarray):
            mask_h, mask_w = mask.shape[:2]
            bbox_h = y_max - y_min
            bbox_w = x_max - x_min
            
            # Resize mask to match box region
            if mask_h != bbox_h or mask_w != bbox_w:
                mask_resized = cv2.resize(mask, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask
            
            # Ensure mask is binary (0 or 255)
            if mask_resized.dtype != np.uint8:
                mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Apply mask overlay
            color = MASK_COLORS[class_id % len(MASK_COLORS)]
            mask_colored = np.zeros_like(box_region)
            mask_colored[mask_resized > 0] = color
            
            # Blend mask with box region
            box_region = cv2.addWeighted(box_region, 1.0 - alpha, mask_colored, alpha, 0)
        
        # Resize box to fit grid cell (maintain aspect ratio)
        scale = min(box_size / box_w, box_size / box_h)
        new_w = int(box_w * scale)
        new_h = int(box_h * scale)
        
        if new_w > 0 and new_h > 0:
            resized_box = cv2.resize(box_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Calculate position in grid
            col = idx % cols
            row = idx // cols
            
            # Center the resized box in the grid cell
            y_offset = (box_size - new_h) // 2
            x_offset = (box_size - new_w) // 2
            
            grid_y = row * box_size + y_offset
            grid_x = col * box_size + x_offset
            
            # Place box in grid
            grid[grid_y:grid_y + new_h, grid_x:grid_x + new_w] = resized_box
            
            # Draw border and label
            color = MASK_COLORS[class_id % len(MASK_COLORS)]
            cv2.rectangle(grid, 
                         (col * box_size, row * box_size),
                         ((col + 1) * box_size - 1, (row + 1) * box_size - 1),
                         color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = row * box_size + 20
            
            # Label background
            cv2.rectangle(grid,
                         (col * box_size, label_y - label_size[1] - 5),
                         (col * box_size + label_size[0] + 5, label_y + 5),
                         color, -1)
            
            # Label text
            cv2.putText(grid, label,
                       (col * box_size + 2, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return grid


# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hailo YOLO8 Segmentation</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background: #1a1a1a; color: #fff; }
        h1 { color: #4CAF50; }
        img { max-width: 100%; height: auto; border: 2px solid #4CAF50; }
        .info { margin: 20px; padding: 10px; background: #2a2a2a; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Hailo YOLO8 Instance Segmentation</h1>
    <div class="info">
        <p>Real-time instance segmentation with mask overlay</p>
        <p>Confidence threshold: {{ confidence_threshold }}</p>
    </div>
    <img src="{{ url_for('video_feed') }}" alt="Video Feed">
</body>
</html>
"""


def get_local_ip():
    """Get local IP address for web interface."""
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


def video_capture_loop(picam2, cap, detector: HailoYOLO8Segmentation, enable_gui: bool, board: WhisPlayBoard = None):
    """Continuous video capture loop for segmentation with masks."""
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
    
    window_name = "Hailo YOLO8 Segmentation"
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

            frame_count += 1
            if FRAME_SKIP > 1 and (frame_count % FRAME_SKIP != 0):
                continue
            
            # Run inference
            detections = detector.infer(frame)
            
            # Filter detections by allowed classes if configured
            if allowed_classes_set is not None:
                detections = [
                    det for det in detections
                    if isinstance(det, dict)
                    and det.get("class_name", "").lower() in allowed_classes_set
                ]
            
            # Draw detections or show boxes only
            if SHOW_BOXES_ONLY:
                # Extract boxes from original frame (use full resolution for better quality)
                frame_with_overlay = extract_boxes_grid(frame, detections, max_boxes=12, box_size=320)
            else:
                # Early downscaling for display operations (reduces mask overlay work significantly)
                # Keep full resolution for inference, but work on smaller frame for display
                if DISPLAY_SCALE < 1.0:
                    orig_h, orig_w = frame.shape[:2]
                    display_w = int(orig_w * DISPLAY_SCALE)
                    display_h = int(orig_h * DISPLAY_SCALE)
                    frame_display = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
                    
                    # Scale detections to display dimensions (minimal Python work)
                    detections_scaled = []
                    for det in detections:
                        det_scaled = det.copy()
                        bbox = det_scaled['bbox']
                        det_scaled['bbox'] = {
                            'x_min': bbox['x_min'] * DISPLAY_SCALE,
                            'y_min': bbox['y_min'] * DISPLAY_SCALE,
                            'x_max': bbox['x_max'] * DISPLAY_SCALE,
                            'y_max': bbox['y_max'] * DISPLAY_SCALE,
                            'width': bbox['width'] * DISPLAY_SCALE,
                            'height': bbox['height'] * DISPLAY_SCALE
                        }
                        # Scale mask if present (only resize the bbox-sized mask, not full frame)
                        if det_scaled.get('mask') is not None:
                            mask = det_scaled['mask']
                            if mask is not None and mask.size > 0:
                                mask_h, mask_w = mask.shape[:2]
                                new_mask_w = int(mask_w * DISPLAY_SCALE)
                                new_mask_h = int(mask_h * DISPLAY_SCALE)
                                if new_mask_w > 0 and new_mask_h > 0:
                                    det_scaled['mask'] = cv2.resize(mask, (new_mask_w, new_mask_h), interpolation=cv2.INTER_LINEAR)
                        detections_scaled.append(det_scaled)
                    
                    # Draw on downscaled frame (much faster - 4x fewer pixels for mask overlay with 0.5 scale)
                    frame_with_overlay = draw_segmentation(frame_display, detections_scaled)
                else:
                    # No downscaling, use full resolution
                    frame_with_overlay = draw_segmentation(frame, detections)
            
            # Log resolutions once
            if not resolution_logged:
                cam_h, cam_w = frame.shape[:2]
                out_h, out_w = frame_with_overlay.shape[:2]
                if SHOW_BOXES_ONLY:
                    print(f"Camera input: {cam_w}x{cam_h}, output: {out_w}x{out_h} (box-only mode)")
                else:
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
            cv2.putText(frame_with_overlay, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update global frame buffer for web streaming
            with frame_lock:
                latest_frame = frame_with_overlay
                latest_detections = detections
            
            # Send to Whisplay display (only if enabled)
            if board is not None and ENABLE_WHISPLAY:
                try:
                    pixel_data = image_to_rgb565(frame_with_overlay)
                    board.draw_image(0, 0, WHISPLAY_WIDTH, WHISPLAY_HEIGHT, pixel_data)
                except Exception as e:
                    print(f"✗ Whisplay display error: {e}")
                    board = None  # Disable on error
            
            # GUI display and controls
            if enable_gui:
                try:
                    cv2.imshow(window_name, frame_with_overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuit requested from GUI window.")
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"segmentation_{timestamp}.jpg"
                        cv2.imwrite(filename, frame_with_overlay)
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
    print("Hailo YOLO8 Instance Segmentation App")
    print("=" * 60)
    print(f"Input source: {'Video file: ' + FILE_PATH if FILE_PATH else 'Camera'}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Resolution: {RESOLUTION_DEFAULT[0]}x{RESOLUTION_DEFAULT[1]}")
    print(f"Display scale: {DISPLAY_SCALE:.2f}x (early downscaling for performance)")
    print(f"Frame skip: {FRAME_SKIP} (process every {FRAME_SKIP} frame(s))")
    print(f"Web preview: {'Enabled' if ENABLE_WEB else 'Disabled'}")
    print(f"GUI preview: {'Enabled' if ENABLE_GUI else 'Disabled'}")
    print(f"Display preview (Whisplay): {'Enabled' if ENABLE_WHISPLAY else 'Disabled'}")
    print(f"Box-only preview: {'Enabled' if SHOW_BOXES_ONLY else 'Disabled'}")
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
        print("\nTo download YOLOv8n segmentation models:")
        print("  See README.md for instructions on downloading from Hailo Model Zoo")
        return
    
    # Initialize Hailo detector
    try:
        detector = HailoYOLO8Segmentation(hef_path)
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
        print("\n✓ GUI available - starting segmentation loop with GUI window...")
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
        
        print("\nStarting segmentation loop (web only)...")
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

