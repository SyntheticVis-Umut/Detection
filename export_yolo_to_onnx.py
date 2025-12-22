#!/usr/bin/env python3
"""
Export YOLO8 PyTorch model (.pt) to ONNX format for Hailo HEF conversion.

This script exports a YOLO8 model to ONNX format, which can then be converted
to HEF using the Hailo Dataflow Compiler (DFC) on a development machine.

Usage:
    python3 export_yolo_to_onnx.py --input model.pt --output model.onnx
    python3 export_yolo_to_onnx.py --input yolov8s.pt --output yolov8s.onnx --imgsz 640
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("ERROR: ultralytics package not found.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def export_yolo_to_onnx(
    input_path: str,
    output_path: str = None,
    imgsz: int = 640,
    opset: int = 11,
    simplify: bool = True,
    half: bool = False
):
    """
    Export YOLO8 model from PyTorch (.pt) to ONNX format.
    
    Args:
        input_path: Path to input .pt model file
        output_path: Path to output .onnx file (default: same name as input)
        imgsz: Input image size (default: 640)
        opset: ONNX opset version (default: 11, recommended for Hailo)
        simplify: Simplify ONNX model (default: True)
        half: Use FP16 (default: False, use FP32 for Hailo)
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    if output_path is None:
        output_path = input_file.with_suffix('.onnx')
    else:
        output_path = Path(output_path)
    
    print("=" * 60)
    print("YOLO8 to ONNX Export Tool")
    print("=" * 60)
    print(f"Input model:  {input_path}")
    print(f"Output ONNX: {output_path}")
    print(f"Image size:   {imgsz}x{imgsz}")
    print(f"ONNX opset:   {opset}")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {input_path}...")
    try:
        model = YOLO(str(input_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        sys.exit(1)
    
    # Export to ONNX
    print(f"\nExporting to ONNX format...")
    try:
        model.export(
            format='onnx',
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=False,  # Static shapes required for Hailo
            half=half,
            verbose=True
        )
        
        # Ultralytics exports to same directory with .onnx extension
        exported_path = input_file.with_suffix('.onnx')
        if exported_path.exists():
            if exported_path != output_path:
                # Move to desired output location
                exported_path.rename(output_path)
            print(f"✓ ONNX model exported successfully: {output_path}")
        else:
            print(f"✗ ERROR: ONNX file not found at expected location")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ ERROR during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify output
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Export complete!")
        print(f"  File: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"\nNext step: Convert ONNX to HEF using Hailo DFC:")
        print(f"  hailo compile {output_path} --output model.hef")
    else:
        print(f"\n✗ ERROR: Output file not found: {output_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO8 PyTorch model to ONNX format for Hailo conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export default YOLOv8s model
  python3 export_yolo_to_onnx.py --input yolov8s.pt

  # Export with custom image size
  python3 export_yolo_to_onnx.py --input model.pt --output model.onnx --imgsz 640

  # Export with specific ONNX opset
  python3 export_yolo_to_onnx.py --input model.pt --opset 12
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input PyTorch model file (.pt)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output ONNX file (.onnx). Default: same name as input with .onnx extension'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        choices=[10, 11, 12, 13],
        help='ONNX opset version (default: 11, recommended for Hailo)'
    )
    
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX model simplification'
    )
    
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use FP16 precision (default: FP32)'
    )
    
    args = parser.parse_args()
    
    export_yolo_to_onnx(
        input_path=args.input,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        half=args.half
    )


if __name__ == '__main__':
    main()


