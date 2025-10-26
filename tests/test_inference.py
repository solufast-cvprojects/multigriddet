#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test inference functionality for MultiGridDet.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.models import build_multigriddet_darknet
from multigriddet.utils.anchors import load_anchors, load_classes
from multigriddet.utils.tf_optimization import optimize_tf_gpu


def test_model_creation():
    """Test that the model can be created successfully."""
    
    print("Testing model creation...")
    
    # Configuration
    input_shape = (608, 608, 3)
    num_classes = 80
    anchors_path = 'configs/yolov3_coco_anchor.txt'
    
    # Load anchors
    anchors = load_anchors(anchors_path)
    num_anchors_per_head = [len(anchors[l]) for l in range(len(anchors))]
    
    # Create model
    model, backbone_len = build_multigriddet_darknet(
        input_shape=input_shape,
        num_anchors_per_head=num_anchors_per_head,
        num_classes=num_classes,
        weights_path=None
    )
    
    # Verify model properties
    assert model is not None, "Model creation failed"
    assert backbone_len > 0, "Invalid backbone length"
    assert model.count_params() > 0, "Model has no parameters"
    
    print(f"[OK] Model created successfully with {model.count_params()} parameters")
    return True


def test_weight_loading():
    """Test that weights can be loaded successfully."""
    
    print("Testing weight loading...")
    
    # Configuration
    input_shape = (608, 608, 3)
    num_classes = 80
    anchors_path = 'configs/yolov3_coco_anchor.txt'
    weights_path = 'weights/model5.h5'
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"[WARNING] Weights file not found: {weights_path}")
        print("Skipping weight loading test")
        return True
    
    # Load anchors
    anchors = load_anchors(anchors_path)
    num_anchors_per_head = [len(anchors[l]) for l in range(len(anchors))]
    
    # Create model
    model, _ = build_multigriddet_darknet(
        input_shape=input_shape,
        num_anchors_per_head=num_anchors_per_head,
        num_classes=num_classes,
        weights_path=None
    )
    
    # Load weights
    model.load_weights(weights_path)
    
    print("[OK] Weights loaded successfully")
    return True


def test_inference():
    """Test that inference works with dummy data."""
    
    print("Testing inference...")
    
    # Configuration
    input_shape = (608, 608, 3)
    num_classes = 80
    anchors_path = 'configs/yolov3_coco_anchor.txt'
    weights_path = 'weights/model5.h5'
    
    # Load anchors
    anchors = load_anchors(anchors_path)
    num_anchors_per_head = [len(anchors[l]) for l in range(len(anchors))]
    
    # Create model
    model, _ = build_multigriddet_darknet(
        input_shape=input_shape,
        num_anchors_per_head=num_anchors_per_head,
        num_classes=num_classes,
        weights_path=None
    )
    
    # Load weights if available
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    
    # Create dummy input
    dummy_input = np.random.random((1, *input_shape)).astype(np.float32)
    
    # Run inference
    predictions = model.predict(dummy_input, verbose=0)
    
    # Verify output
    assert len(predictions) == 3, f"Expected 3 prediction layers, got {len(predictions)}"
    
    for i, pred in enumerate(predictions):
        expected_shape = (1, 608 // (2**(i+1)), 608 // (2**(i+1)), 5 + num_classes + num_anchors_per_head[i])
        assert pred.shape == expected_shape, f"Layer {i}: expected {expected_shape}, got {pred.shape}"
    
    print("[OK] Inference test passed")
    return True


def main():
    """Run all tests."""
    
    print("MultiGridDet Standalone Tests")
    print("=" * 40)
    
    # Optimize TensorFlow
    optimize_tf_gpu()
    
    tests = [
        test_model_creation,
        test_weight_loading,
        test_inference,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"[ERROR] {test.__name__} failed")
        except Exception as e:
            print(f"[ERROR] {test.__name__} failed with error: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! MultiGridDet is working correctly.")
        return True
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)





