#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity check to verify tf_preprocess_true_boxes and preprocess_true_boxes
produce identical tensors for the same input.

This test ensures compatibility between TensorFlow and NumPy data paths.
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multigriddet.data.generators import tf_preprocess_true_boxes, preprocess_true_boxes


def test_target_consistency():
    """Test that both functions produce identical targets."""
    print("=" * 60)
    print("Testing target consistency between tf and numpy implementations")
    print("=" * 60)
    
    # Create test data
    input_shape = (608, 608)
    num_classes = 1
    
    # Create a single box at a known location
    box_center_x, box_center_y = 304.0, 304.0  # Center of image
    box_w, box_h = 100.0, 80.0
    
    # Convert to (x1, y1, x2, y2, class) format
    x1 = box_center_x - box_w / 2.0
    y1 = box_center_y - box_h / 2.0
    x2 = box_center_x + box_w / 2.0
    y2 = box_center_y + box_h / 2.0
    
    true_boxes = np.array([[[x1, y1, x2, y2, 0]]], dtype=np.float32)  # (1, 1, 5)
    
    # Anchors and grid shapes
    anchors_np = [
        np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),
        np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),
        np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32),
    ]
    anchors_tf = [tf.constant(anchor, dtype=tf.float32) for anchor in anchors_np]
    grid_shapes = [(19, 19), (38, 38), (76, 76)]
    
    # Process with NumPy version
    print("\nProcessing with preprocess_true_boxes (NumPy)...")
    y_true_np = preprocess_true_boxes(
        true_boxes,
        input_shape,
        anchors_np,
        num_classes,
        multi_anchor_assign=False,
        grid_shapes=grid_shapes
    )
    
    # Process with TensorFlow version
    print("Processing with tf_preprocess_true_boxes (TensorFlow)...")
    true_boxes_tf = tf.constant(true_boxes, dtype=tf.float32)
    y_true_tf = tf_preprocess_true_boxes(
        true_boxes_tf,
        input_shape,
        anchors_tf,
        num_classes,
        multi_anchor_assign=False,
        grid_shapes=grid_shapes
    )
    y_true_tf_np = [y.numpy() for y in y_true_tf]
    
    # Compare results
    print("\nComparing results...")
    all_match = True
    
    for layer_idx in range(len(y_true_np)):
        y_np = y_true_np[layer_idx]
        y_tf = y_true_tf_np[layer_idx]
        
        print(f"\nLayer {layer_idx} (grid: {grid_shapes[layer_idx]}):")
        print(f"  NumPy shape: {y_np.shape}")
        print(f"  TF shape: {y_tf.shape}")
        
        # Check shapes match
        if y_np.shape != y_tf.shape:
            print(f"  ERROR: Shape mismatch!")
            all_match = False
            continue
        
        # Compare values
        # Extract objectness to find cells with objects
        objectness_np = y_np[0, ..., 4:5]
        objectness_tf = y_tf[0, ..., 4:5]
        
        # Find cells with objects
        mask_np = objectness_np[..., 0] > 0.5
        mask_tf = objectness_tf[..., 0] > 0.5
        
        # Check objectness masks match
        if not np.array_equal(mask_np, mask_tf):
            print(f"  ERROR: Objectness masks don't match!")
            print(f"    NumPy cells with objects: {np.sum(mask_np)}")
            print(f"    TF cells with objects: {np.sum(mask_tf)}")
            all_match = False
            continue
        
        # Compare values at object cells
        if np.any(mask_np):
            # Extract xy, wh, objectness, anchors, classes
            xy_np = y_np[0, ..., 0:2][mask_np]
            xy_tf = y_tf[0, ..., 0:2][mask_np]
            
            wh_np = y_np[0, ..., 2:4][mask_np]
            wh_tf = y_tf[0, ..., 2:4][mask_np]
            
            obj_np = y_np[0, ..., 4:5][mask_np]
            obj_tf = y_tf[0, ..., 4:5][mask_np]
            
            num_anchors = len(anchors_np[layer_idx])
            anchors_np_layer = y_np[0, ..., 5:5+num_anchors][mask_np]
            anchors_tf_layer = y_tf[0, ..., 5:5+num_anchors][mask_np]
            
            classes_np = y_np[0, ..., 5+num_anchors:][mask_np]
            classes_tf = y_tf[0, ..., 5+num_anchors:][mask_np]
            
            # Compare with tolerance
            tol = 1e-5
            xy_diff = np.abs(xy_np - xy_tf)
            wh_diff = np.abs(wh_np - wh_tf)
            obj_diff = np.abs(obj_np - obj_tf)
            anchors_diff = np.abs(anchors_np_layer - anchors_tf_layer)
            classes_diff = np.abs(classes_np - classes_tf)
            
            max_xy_diff = np.max(xy_diff)
            max_wh_diff = np.max(wh_diff)
            max_obj_diff = np.max(obj_diff)
            max_anchors_diff = np.max(anchors_diff)
            max_classes_diff = np.max(classes_diff)
            
            print(f"  Cells with objects: {np.sum(mask_np)}")
            print(f"  Max xy difference: {max_xy_diff:.6f} (tolerance: {tol})")
            print(f"  Max wh difference: {max_wh_diff:.6f} (tolerance: {tol})")
            print(f"  Max objectness difference: {max_obj_diff:.6f} (tolerance: {tol})")
            print(f"  Max anchors difference: {max_anchors_diff:.6f} (tolerance: {tol})")
            print(f"  Max classes difference: {max_classes_diff:.6f} (tolerance: {tol})")
            
            if (max_xy_diff > tol or max_wh_diff > tol or max_obj_diff > tol or 
                max_anchors_diff > tol or max_classes_diff > tol):
                print(f"  ERROR: Values don't match within tolerance!")
                all_match = False
            else:
                print(f"  PASS: All values match within tolerance")
        else:
            print(f"  No objects in this layer")
    
    if all_match:
        print("\n" + "=" * 60)
        print("PASS: All layers match!")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("FAIL: Some layers don't match!")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = test_target_consistency()
    sys.exit(0 if success else 1)

