#!/usr/bin/env python3
"""
Test that all 9 cells in a 3x3 grid decode to the same box coordinates.

This verifies the core MultiGridDet mechanism where multiple grid cells
predict the same object, ensuring robust detection.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.data.generators import tf_preprocess_true_boxes


def test_9cell_alignment():
    """Verify that 9 cells decode to the same box center."""
    print("=" * 60)
    print("Testing 9-cell alignment")
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
            
            stored_xy = y_true[0, i, j, 0:2]
            raw_wh = y_true[0, i, j, 2:4]
            anchor_one_hot = y_true[0, i, j, 5:5+len(anchors[layer_idx])]
            anchor_idx = np.argmax(anchor_one_hot)
            anchor = anchors[layer_idx][anchor_idx].numpy()
            
            raw_xy_values.append(stored_xy.copy())
            
            activated_xy = stored_xy
            cell_grid = np.array([j, i])
            box_xy_grid = activated_xy + cell_grid
            box_xy_normalized = box_xy_grid / np.array([grid_w, grid_h])
            
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
        
        stored_xy_array = np.array(raw_xy_values)
        print(f"\n  Stored_xy values (should be different for each cell):")
        min_x, max_x = np.min(stored_xy_array[:, 0]), np.max(stored_xy_array[:, 0])
        min_y, max_y = np.min(stored_xy_array[:, 1]), np.max(stored_xy_array[:, 1])
        print(f"    Range x: [{min_x:.6f}, {max_x:.6f}] (expected: [-1, 2))")
        print(f"    Range y: [{min_y:.6f}, {max_y:.6f}] (expected: [-1, 2))")
        
        epsilon = 1e-6
        x_in_range = (min_x >= -1.0 - epsilon) and (max_x < 2.0 + epsilon)
        y_in_range = (min_y >= -1.0 - epsilon) and (max_y < 2.0 + epsilon)
        x_has_exactly_2 = np.any(np.abs(stored_xy_array[:, 0] - 2.0) < epsilon)
        y_has_exactly_2 = np.any(np.abs(stored_xy_array[:, 1] - 2.0) < epsilon)
        
        if x_in_range and y_in_range and not (x_has_exactly_2 or y_has_exactly_2):
            print(f"    ✓ PASS: stored_xy values are within expected range [-1, 2)")
        else:
            print(f"    ✗ FAIL: stored_xy values are outside expected range [-1, 2)")
            if not x_in_range:
                print(f"      X: min={min_x:.9f}, max={max_x:.9f}")
            if not y_in_range:
                print(f"      Y: min={min_y:.9f}, max={max_y:.9f}")
            if x_has_exactly_2 or y_has_exactly_2:
                print(f"      Values exactly 2.0 detected (should be < 2)")


if __name__ == '__main__':
    test_9cell_alignment()

