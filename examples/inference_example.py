#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Inference Example
Standalone inference script for MultiGridDet model.
"""

import os
import sys
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
    """Run MultiGridDet inference example."""
    
    print("MultiGridDet Inference Example")
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
    
    # Test with a sample image
    test_image_path = 'examples/images/dog.jpg'
    if os.path.exists(test_image_path):
        print(f"Testing with image: {test_image_path}")
        
        # Load and preprocess image
        image = Image.open(test_image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data = preprocess_image(image, input_shape[:2])  # (608, 608)
        image_shape = tuple(reversed(image.size))  # (width, height)
        
        # Run inference
        print("Running inference...")
        predictions = model.predict(image_data, verbose=0)
        
        print(f"Model outputs: {len(predictions)} prediction layers")
        for i, pred in enumerate(predictions):
            print(f"  Layer {i}: shape {pred.shape}")
        
        # Post-process predictions
        print("Post-processing predictions...")
        boxes, classes, scores = denseyolo2_postprocess_np(
            predictions, 
            image_shape, 
            anchors, 
            num_classes, 
            input_shape[:2],  # (608, 608)
            max_boxes=500, 
            confidence=confidence_threshold, 
            rescore_confidence=True, 
            nms_threshold=nms_threshold, 
            use_iol=True
        )
        
        print(f"Found {len(boxes)} objects:")
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            class_name = class_names[cls]
            x1, y1, x2, y2 = box
            print(f"  {i+1}. {class_name}: {score:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # Draw bounding boxes
        if len(boxes) > 0:
            image_array = np.array(image, dtype='uint8')
            annotated_image = draw_boxes(
                image_array, 
                boxes, 
                classes, 
                scores, 
                class_names, 
                colors
            )
            
            # Save result
            result_image = Image.fromarray(annotated_image)
            result_path = 'examples/inference_result.jpg'
            result_image.save(result_path)
            print(f"[OK] Result saved to: {result_path}")
        else:
            print("No objects detected")
            
    else:
        print(f"Test image not found: {test_image_path}")
        print("Available images in examples/images/:")
        for f in os.listdir('examples/images/'):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                print(f"  - {f}")
    
    print("[OK] MultiGridDet inference example completed!")
    return True


if __name__ == '__main__':
    success = main()
    if success:
        print("\n[SUCCESS] MultiGridDet inference is working!")
    else:
        print("\n[ERROR] MultiGridDet inference example failed.")





