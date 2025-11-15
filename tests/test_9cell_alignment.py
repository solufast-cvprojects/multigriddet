#!/usr/bin/env python3
"""
Test that all 9 cells in a 3x3 grid decode to the EXACT same box coordinates.
This is the core MultiGridDet innovation - multiple cells predict the same object.
"""

import numpy as np
import tensorflow as tf
from scipy.special import expit
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multigriddet.data.generators import tf_preprocess_true_boxes


def test_9cell_alignment():
    """Test that 9 cells decode to the same box center."""
    print("=" * 60)
    print("Testing 9-cell alignment (all should decode to same box)")
    print("=" * 60)
    
    # Create a single box at a known location
    input_shape = (608, 608)
    box_center_x, box_center_y = 311.999, 311.999  # Center of image in pixels
    box_w, box_h = 100.0, 80.0
    
    # Convert to (x1, y1, x2, y2, class) format
    x1 = box_center_x - box_w / 2.0
    y1 = box_center_y - box_h / 2.0
    x2 = box_center_x + box_w / 2.0
    y2 = box_center_y + box_h / 2.0
    
    true_boxes = np.array([[[x1, y1, x2, y2, 0]]], dtype=np.float32)  # (1, 1, 5)
    
    # Anchors and grid shapes
    anchors = [
        tf.constant([[10, 13], [16, 30], [33, 23]], dtype=tf.float32),
        tf.constant([[30, 61], [62, 45], [59, 119]], dtype=tf.float32),
        tf.constant([[116, 90], [156, 198], [373, 326]], dtype=tf.float32),
    ]
    num_classes = 1
    grid_shapes = [(19, 19), (38, 38), (76, 76)]
    
    # Encode
    true_boxes_tf = tf.constant(true_boxes, dtype=tf.float32)
    y_true_list = tf_preprocess_true_boxes(
        true_boxes_tf,
        input_shape,
        anchors,
        num_classes,
        multi_anchor_assign=False,
        grid_shapes=grid_shapes
    )
    
    # Find which layer got the box (should be layer 1 or 2 for this size)
    y_true_np = [y.numpy() for y in y_true_list]
    
    # Check each layer 
    for layer_idx, y_true in enumerate(y_true_np):
        grid_h, grid_w = grid_shapes[layer_idx]
        objectness = y_true[0, ..., 4:5]  # (grid_h, grid_w, 1)
        object_mask = objectness[..., 0] > 0.5  # (grid_h, grid_w)
        
        if not np.any(object_mask):
            continue
        
        print(f"\nLayer {layer_idx} (grid: {grid_h}x{grid_w}):")
        print(f"  Box center (pixels): ({box_center_x}, {box_center_y})")
        print(f"  Box center (normalized): ({box_center_x/input_shape[1]:.4f}, {box_center_y/input_shape[0]:.4f})")
        
        # Get all cells with objects
        cells_with_objects = np.where(object_mask)
        num_cells = len(cells_with_objects[0])
        print(f"  Number of cells with objects: {num_cells}")
        
        if num_cells == 0:
            continue
        
        # Decode each cell and check if they all produce the same box center
        decoded_centers = []
        raw_xy_values = []
        
        for cell_idx in range(num_cells):
            i, j = cells_with_objects[0][cell_idx], cells_with_objects[1][cell_idx]
            
            # Get stored_xy and raw_wh from y_true
            # NOTE: y_true now stores already-activated offsets [-kj + ty, -ki + tx]
            # (matching preprocess_true_boxes behavior)
            stored_xy = y_true[0, i, j, 0:2]  # (2,) - already-activated offset
            raw_wh = y_true[0, i, j, 2:4]  # (2,)
            anchor_one_hot = y_true[0, i, j, 5:5+len(anchors[layer_idx])]
            anchor_idx = np.argmax(anchor_one_hot)
            anchor = anchors[layer_idx][anchor_idx].numpy()
            
            raw_xy_values.append(stored_xy.copy())
            
            # Decode y_true targets (already-activated offsets, no need to apply activation)
            # Step 1: stored_xy is already activated (no activation needed)
            activated_xy = stored_xy  # Already activated: [-kj + ty, -ki + tx]
            
            # Step 2: Add cell_grid offset
            cell_grid = np.array([j, i])  # [x, y] = [j, i]
            box_xy_grid = activated_xy + cell_grid
            
            # Step 3: Normalize to [0, 1]
            box_xy_normalized = box_xy_grid / np.array([grid_w, grid_h])
            
            # Step 4: Convert to pixels
            decoded_cx = box_xy_normalized[0] * input_shape[1]
            decoded_cy = box_xy_normalized[1] * input_shape[0]
            
            decoded_centers.append([decoded_cx, decoded_cy])
            
            if cell_idx < 9:  # Print first 9
                print(f"    Cell ({i:2d}, {j:2d}): stored_xy=({stored_xy[0]:7.3f}, {stored_xy[1]:7.3f})",
                    f"box_xy_grid=({box_xy_grid[0]:7.3f}, {box_xy_grid[1]:7.3f})",
                    f"activated_xy=({activated_xy[0]:7.3f}, {activated_xy[1]:7.3f})",    
                    f"box_xy_normalized=({box_xy_normalized[0]:7.3f}, {box_xy_normalized[1]:7.3f})",
                    f"decoded=({decoded_cx:7.2f}, {decoded_cy:7.2f})")
        
        # Also decode width/height for each cell
        decoded_wh = []
        for cell_idx in range(num_cells):
            i, j = cells_with_objects[0][cell_idx], cells_with_objects[1][cell_idx]
            raw_wh = y_true[0, i, j, 2:4]
            anchor_one_hot = y_true[0, i, j, 5:5+len(anchors[layer_idx])]
            anchor_idx = np.argmax(anchor_one_hot)
            anchor = anchors[layer_idx][anchor_idx].numpy()
            
            # Decode width/height
            box_wh_pixels = anchor * np.exp(raw_wh)
            box_wh_normalized = box_wh_pixels / np.array([input_shape[1], input_shape[0]])
            decoded_w = box_wh_normalized[0] * input_shape[1]
            decoded_h = box_wh_normalized[1] * input_shape[0]
            decoded_wh.append([decoded_w, decoded_h])
        
        # Check alignment
        decoded_centers = np.array(decoded_centers)
        decoded_wh = np.array(decoded_wh)
        center_std_x = np.std(decoded_centers[:, 0])
        center_std_y = np.std(decoded_centers[:, 1])
        center_mean_x = np.mean(decoded_centers[:, 0])
        center_mean_y = np.mean(decoded_centers[:, 1])
        wh_std_w = np.std(decoded_wh[:, 0])
        wh_std_h = np.std(decoded_wh[:, 1])
        
        print(f"\n  Alignment check:")
        print(f"    Mean decoded center: ({center_mean_x:.2f}, {center_mean_y:.2f})")
        print(f"    Target center: ({box_center_x:.2f}, {box_center_y:.2f})")
        print(f"    Center std dev: ({center_std_x:.6f}, {center_std_y:.6f}) pixels")
        print(f"    Max center deviation: ({np.max(np.abs(decoded_centers[:, 0] - box_center_x)):.6f}, "
              f"{np.max(np.abs(decoded_centers[:, 1] - box_center_y)):.6f}) pixels")
        print(f"    Mean decoded wh: ({np.mean(decoded_wh[:, 0]):.2f}, {np.mean(decoded_wh[:, 1]):.2f})")
        print(f"    Target wh: ({box_w:.2f}, {box_h:.2f})")
        print(f"    WH std dev: ({wh_std_w:.6f}, {wh_std_h:.6f}) pixels")
        print(f"    Max WH deviation: ({np.max(np.abs(decoded_wh[:, 0] - box_w)):.6f}, "
              f"{np.max(np.abs(decoded_wh[:, 1] - box_h)):.6f}) pixels")
        
        # Check if all centers are within 1 pixel
        max_dev_x = np.max(np.abs(decoded_centers[:, 0] - box_center_x))
        max_dev_y = np.max(np.abs(decoded_centers[:, 1] - box_center_y))
        
        if max_dev_x < 1.0 and max_dev_y < 1.0:
            print(f"    ✓ PASS: All cells decode to within 1 pixel of target")
        else:
            print(f"    ✗ FAIL: Some cells are off by more than 1 pixel")
            print(f"    This suggests the numerical inversion may not be precise enough")
        
        # Also check stored_xy values - they should be different for each cell
        stored_xy_array = np.array(raw_xy_values)
        print(f"\n  Stored_xy values (already-activated offsets, should be different for each cell):")
        print(f"    Range x: [{np.min(stored_xy_array[:, 0]):.3f}, {np.max(stored_xy_array[:, 0]):.3f}]")
        print(f"    Range y: [{np.min(stored_xy_array[:, 1]):.3f}, {np.max(stored_xy_array[:, 1]):.3f}]")


if __name__ == '__main__':
    test_9cell_alignment()

