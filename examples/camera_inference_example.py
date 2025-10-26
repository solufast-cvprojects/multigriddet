#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Camera Inference Example
Real-time object detection using camera feed.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from multigriddet package
from multigriddet.models import build_multigriddet_darknet
from multigriddet.utils.anchors import load_anchors, load_classes
from multigriddet.utils.tf_optimization import optimize_tf_gpu
from multigriddet.postprocess.denseyolo_postprocess import denseyolo2_postprocess_np
from multigriddet.utils.preprocessing import preprocess_image
from multigriddet.utils.visualization import get_colors, draw_boxes


def main():
    """Run MultiGridDet camera inference example."""
    
    print("MultiGridDet Camera Inference Example")
    print("=" * 40)
    
    # Configuration
    weights_path = 'weights/model5.h5'
    anchors_path = 'configs/yolov3_coco_anchor.txt'
    classes_path = 'configs/coco_classes.txt'
    input_shape = (608, 608, 3)  # 608x608 for 80-class COCO
    confidence_threshold = 0.5
    nms_threshold = 0.4
    
    # Optimize TensorFlow
    optimize_tf_gpu()
    
    # Load classes and anchors
    print("Loading classes and anchors...")
    class_names = load_classes(classes_path)
    anchors = load_anchors(anchors_path)
    colors = get_colors(len(class_names))
    
    print(f"Loaded {len(class_names)} classes")
    print(f"Loaded {len(anchors)} anchor sets")
    print(f"Input shape: {input_shape}")
    
    # Create multigriddet_darknet model
    print("Creating multigriddet_darknet model...")
    num_anchors_per_head = [len(anchors[l]) for l in range(len(anchors))]
    num_classes = len(class_names)
    
    model, backbone_len = build_multigriddet_darknet(
        input_shape=input_shape,
        num_anchors_per_head=num_anchors_per_head,
        num_classes=num_classes,
        weights_path=None  # We'll load weights separately
    )
    
    print(f"Model created with {model.count_params()} parameters")
    print(f"Backbone length: {backbone_len}")
    
    # Load weights from model5.h5
    print(f"Loading weights from {weights_path}...")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("[OK] Weights loaded successfully!")
    else:
        print(f"[ERROR] Weights file not found: {weights_path}")
        print("Please download model5.h5 and place it in the weights/ directory")
        return False
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Use camera 0
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera initialized. Press 'q' to quit, 's' to save current frame")
    
    # Video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('examples/camera_inference_output.mp4', fourcc, 20.0, (640, 480))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            frame_count += 1
            
            # Skip frames for performance (process every 3rd frame)
            if frame_count % 3 != 0:
                out.write(frame)
                continue
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Preprocess image
            image_data = preprocess_image(pil_image, input_shape[:2])  # (608, 608)
            image_shape = tuple(reversed(pil_image.size))  # (width, height)
            
            # Run inference
            predictions = model.predict(image_data, verbose=0)
            
            # Post-process predictions
            boxes, classes, scores = denseyolo2_postprocess_np(
                predictions, 
                image_shape, 
                anchors, 
                num_classes, 
                input_shape[:2],  # (608, 608)
                max_boxes=100, 
                confidence=confidence_threshold, 
                rescore_confidence=True, 
                nms_threshold=nms_threshold, 
                use_iol=True
            )
            
            # Draw bounding boxes
            if len(boxes) > 0:
                annotated_frame = draw_boxes(
                    rgb_frame, 
                    boxes, 
                    classes, 
                    scores, 
                    class_names, 
                    colors
                )
                # Convert back to BGR for OpenCV
                frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Add FPS counter and model info
            fps_text = f"MultiGridDet - Frame: {frame_count}, Objects: {len(boxes)}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add model info
            model_text = f"Model: multigriddet_darknet (44.9M params)"
            cv2.putText(frame, model_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow('MultiGridDet Camera Inference', frame)
            out.write(frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f'examples/camera_frame_{frame_count}.jpg'
                cv2.imwrite(save_path, frame)
                print(f"Frame saved to: {save_path}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"MultiGridDet camera inference completed. Output saved to: examples/camera_inference_output.mp4")
        print(f"Total frames processed: {frame_count}")
        return True


if __name__ == '__main__':
    success = main()
    if success:
        print("\n[SUCCESS] MultiGridDet camera inference is working!")
    else:
        print("\n[ERROR] MultiGridDet camera inference example failed.")





