#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Inference Example
Standalone inference script for MultiGridDet model using config file.
"""

import os
import sys
import argparse
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
    """Run MultiGridDet inference example."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='MultiGridDet Inference Example')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/infer_config.yaml',
        help='Path to inference config file'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input image path (overrides config)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Model weights path (overrides config)'
    )
    args = parser.parse_args()
    
    print("MultiGridDet Inference Example")
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
    
    if args.input:
        config['input']['source'] = args.input
        print(f"   Input: {args.input} (from command line)")
    else:
        input_source = config.get('input', {}).get('source')
        if input_source:
            print(f"   Input: {input_source} (from config)")
        else:
            print("[ERROR] input.source not specified in config")
            return False
    
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
    
    # Test with image from config
    test_image_path = config['input']['source']
    
    if not os.path.exists(test_image_path):
        print(f"[ERROR] Image not found: {test_image_path}")
        print("Please check the 'input.source' path in your config file")
        return False
    
    print(f"Processing image: {test_image_path}")
    
    # Load and preprocess image
    image = Image.open(test_image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_data = preprocess_image(image, input_shape[:2])
    image_shape = tuple(reversed(image.size))  # (height, width)
    
    # Run inference
    print("Running inference...")
    predictions = model.predict(image_data, verbose=0)
    
    print(f"   Model outputs: {len(predictions)} prediction layers")
    for i, pred in enumerate(predictions):
        print(f"      Layer {i}: shape {pred.shape}")
    
    # Post-process predictions
    print("Post-processing predictions...")
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
    
    print(f"\nDetected {len(boxes)} objects:")
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        class_name = class_names[cls]
        x1, y1, x2, y2 = box
        print(f"   {i+1}. {class_name}: {score:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
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
        output_config = config.get('output', {})
        if output_config.get('save_result', True):
            output_dir = output_config.get('output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            result_path = os.path.join(output_dir, f"result_{Path(test_image_path).name}")
            result_image = Image.fromarray(annotated_image)
            result_image.save(result_path)
            print(f"\n[OK] Result saved to: {result_path}")
        
        # Show result if configured
        if output_config.get('show_result', False):
            result_image = Image.fromarray(annotated_image)
            result_image.show()
    else:
        print("No objects detected")
    
    print("[OK] MultiGridDet inference example completed!")
    return True


if __name__ == '__main__':
    success = main()
    if success:
        print("\n[SUCCESS] MultiGridDet inference is working!")
    else:
        print("\n[ERROR] MultiGridDet inference example failed.")





