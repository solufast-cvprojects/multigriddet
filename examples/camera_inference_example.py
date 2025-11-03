#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Camera Inference Example
Real-time object detection using camera feed with config file.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from multigriddet package
from multigriddet.config import ConfigLoader, build_model_for_inference
from multigriddet.utils.anchors import load_anchors, load_classes
from multigriddet.utils.tf_optimization import optimize_tf_gpu
from multigriddet.postprocess.multigrid_decode import MultiGridDecoder
from multigriddet.utils.preprocessing import preprocess_image
from multigriddet.utils.visualization import get_colors, draw_boxes


def main():
    """Run MultiGridDet camera inference example."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='MultiGridDet Camera Inference Example')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/infer_config.yaml',
        help='Path to inference config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Model weights path (overrides config)'
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=None,
        help='Camera device ID (overrides config)'
    )
    args = parser.parse_args()
    
    print("MultiGridDet Camera Inference Example")
    print("=" * 60)
    print(f"Config file: {args.config}")
    
    # Load configuration
    try:
        config = ConfigLoader.load_config(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] Config file not found: {args.config}")
        print(f"   {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get weights path
    weights_path = None
    if args.weights:
        weights_path = args.weights
        config['weights_path'] = weights_path
        print(f"   Weights: {weights_path} (from command line)")
    else:
        weights_path = config.get('weights_path')
        if weights_path:
            print(f"   Weights: {weights_path} (from config)")
        else:
            print("[ERROR] weights_path not specified in config")
            return False
    
    # Get camera ID
    camera_id = args.camera_id if args.camera_id is not None else config.get('camera', {}).get('device_id', 0)
    print(f"   Camera ID: {camera_id}")
    
    # Get configuration values
    model_config_path = config.get('model_config')
    if not model_config_path:
        print("[ERROR] model_config not specified in config")
        return False
    
    # Load model configuration
    model_config = ConfigLoader.load_config(model_config_path)
    
    # Merge configs
    full_config = ConfigLoader.merge_configs(model_config, config)
    
    # Get paths and parameters
    anchors_path = model_config['model']['preset']['anchors_path']
    classes_path = model_config['model']['preset'].get('classes_path')
    if not classes_path:
        classes_path = full_config.get('data', {}).get('classes_path')
    
    if not classes_path:
        print("[ERROR] classes_path not found in config")
        return False
    
    input_shape = tuple(model_config['model']['preset'].get('input_shape', [608, 608, 3]))
    detection_config = config.get('detection', {})
    confidence_threshold = detection_config.get('confidence_threshold', 0.5)
    nms_threshold = detection_config.get('nms_threshold', 0.45)
    nms_method = detection_config.get('nms_method', 'diou')
    use_wbf = detection_config.get('use_wbf', False)
    use_iol = detection_config.get('use_iol', True)
    max_boxes = detection_config.get('max_boxes', 100)
    
    camera_config = config.get('camera', {})
    resolution = camera_config.get('resolution', [640, 480])
    
    print()
    
    # Optimize TensorFlow
    optimize_tf_gpu()
    
    # Load classes and anchors
    print("Loading classes and anchors...")
    class_names = load_classes(classes_path)
    anchors = load_anchors(anchors_path)
    colors = get_colors(len(class_names))
    
    print(f"   Classes: {len(class_names)}")
    print(f"   Anchors: {len(anchors)} scales")
    print(f"   Input shape: {input_shape}")
    
    # Initialize decoder
    decoder = MultiGridDecoder(
        anchors=anchors,
        num_classes=len(class_names),
        input_shape=input_shape[:2],
        rescore_confidence=True
    )
    
    # Build model using config
    print("Building model...")
    model = build_model_for_inference(full_config, weights_path)
    
    print(f"   Model parameters: {model.count_params():,}")
    print()
    
    # Initialize camera
    print(f"Initializing camera (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera: {camera_id}")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"   Camera resolution: {width}x{height} @ {fps} FPS")
    print("Press 'q' to quit")
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Preprocess
            image_data = preprocess_image(image, input_shape[:2])
            image_shape = (height, width)
            
            # Run inference
            predictions = model.predict(image_data, verbose=0)
            
            # Post-process
            boxes, classes, scores = decoder.postprocess(
                predictions,
                image_shape,
                input_shape[:2],
                max_boxes=max_boxes,
                confidence=confidence_threshold,
                nms_threshold=nms_threshold,
                nms_method=nms_method,
                use_wbf=use_wbf,
                use_iol=use_iol,
                return_xyxy=True
            )
            
            # Draw boxes
            frame_rgb = np.array(image, dtype='uint8')
            annotated_frame = draw_boxes(
                frame_rgb,
                boxes,
                classes,
                scores,
                class_names,
                colors
            )
            
            # Convert back to BGR for OpenCV display
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Display
            cv2.imshow('MultiGridDet Camera Inference', annotated_frame_bgr)
            
            # Show detection info
            if len(boxes) > 0:
                info_text = f"Detections: {len(boxes)}"
                cv2.putText(annotated_frame_bgr, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Camera inference completed!")
    
    return True


if __name__ == '__main__':
    success = main()
    if success:
        print("\n[SUCCESS] MultiGridDet camera inference is working!")
    else:
        print("\n[ERROR] MultiGridDet camera inference example failed.")
