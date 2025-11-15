#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script to verify y_true assignments from tf_preprocess_true_boxes.

This script:
1. Loads a test annotation
2. Processes it through the data generator to get y_true tensors
3. Decodes y_true back to image coordinates
4. Visualizes original vs decoded boxes to verify correctness
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Tuple, Dict, Any
import yaml
from scipy.special import expit

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multigriddet.config import ConfigLoader
from multigriddet.data.generators import MultiGridDataGenerator, preprocess_true_boxes, tf_preprocess_true_boxes
from multigriddet.data.utils import load_annotation_lines
from multigriddet.utils.anchors import load_classes, load_anchors
import tensorflow as tf


def decode_y_true_to_boxes(y_true_list: List[np.ndarray], 
                          anchors: List[np.ndarray],
                          input_shape: Tuple[int, int],
                          grid_shapes: List[Tuple[int, int]],
                          num_classes: int,
                          deduplicate: bool = False) -> List[Dict[str, Any]]:
    """
    Decode y_true tensors back to bounding boxes in image coordinates.
    
    Args:
        y_true_list: List of y_true tensors for each layer [batch, grid_h, grid_w, features]
        anchors: List of anchor arrays for each layer
        input_shape: Input image shape (height, width)
        grid_shapes: Grid shapes for each layer [(grid_h, grid_w), ...]
        num_classes: Number of classes
        
    Returns:
        List of decoded boxes per layer, each containing:
        - boxes: (N, 4) array in format (x1, y1, x2, y2)
        - classes: (N,) array of class indices
        - anchors: (N,) array of anchor indices
        - grid_positions: (N, 2) array of (grid_i, grid_j) positions
    """
    input_h, input_w = input_shape
    decoded_boxes_per_layer = []
    
    for layer_idx, y_true in enumerate(y_true_list):
        # y_true shape: [batch, grid_h, grid_w, 5 + num_anchors + num_classes]
        batch_size = y_true.shape[0]
        # FIXED: Use actual grid dimensions from y_true.shape, not computed grid_shapes
        # This ensures we use the correct grid size that was actually used during encoding
        grid_h, grid_w = y_true.shape[1], y_true.shape[2]
        
        # Debug: Verify grid dimensions
        if layer_idx == 0 or layer_idx == 1:
            print(f"    DEBUG Layer {layer_idx}: y_true.shape={y_true.shape}, grid_h={grid_h}, grid_w={grid_w}, input_shape={input_shape}")
        num_anchors = len(anchors[layer_idx])
        
        # Extract components - y_true stores already-activated offsets and raw_wh
        # NOTE: y_true now stores already-activated offsets [-kj + ty, -ki + tx]
        # (matching preprocess_true_boxes behavior)
        stored_xy = y_true[..., 0:2]  # [batch, grid_h, grid_w, 2] - already-activated offset
        raw_wh = y_true[..., 2:4]  # [batch, grid_h, grid_w, 2] - log-space raw_wh
        objectness = y_true[..., 4:5]  # [batch, grid_h, grid_w, 1]
        anchor_one_hot = y_true[..., 5:5+num_anchors]  # [batch, grid_h, grid_w, num_anchors]
        class_one_hot = y_true[..., 5+num_anchors:]  # [batch, grid_h, grid_w, num_classes]
        
        # Find cells with objects (objectness > 0.5)
        object_mask = objectness[0, ...] > 0.5  # [grid_h, grid_w]
        
        # Create grid coordinates matching decoder logic
        # Decoder uses: grid_y = np.arange(grid_h), grid_x = np.arange(grid_w)
        # Then: x_offset, y_offset = np.meshgrid(grid_x, grid_y)
        # cell_grid[i, j] = [j, i] where i=row, j=col
        grid_y = np.arange(grid_h)
        grid_x = np.arange(grid_w)
        x_offset, y_offset = np.meshgrid(grid_x, grid_y)
        
        # Reshape to match y_true shape: (grid_h, grid_w, 2)
        cell_grid = np.stack([x_offset, y_offset], axis=-1)  # [grid_h, grid_w, 2] - [x, y] = [j, i]
        
        # Collect all valid boxes
        boxes = []
        classes = []
        anchor_indices = []
        grid_positions = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                if not object_mask[i, j]:
                    continue
                
                # Get values at this cell
                cell_stored_xy = stored_xy[0, i, j, :]  # [2] - already-activated offset
                cell_raw_wh = raw_wh[0, i, j, :]  # [2] - log-space raw_wh
                cell_anchor = anchor_one_hot[0, i, j, :]  # [num_anchors]
                cell_class = class_one_hot[0, i, j, :]  # [num_classes]
                
                # Get anchor index
                anchor_idx = np.argmax(cell_anchor)
                anchor = anchors[layer_idx][anchor_idx]
                
                # Get class index
                class_idx = np.argmax(cell_class)
                
                # Decode y_true targets (already-activated offsets, no need to apply activation)
                # Step 1: stored_xy is already activated (no activation needed)
                activated_xy = cell_stored_xy  # Already activated: [-kj + ty, -ki + tx]
                
                # Step 2: Add cell_grid offset
                cell_grid_val = cell_grid[i, j, :]  # [2] - [j, i] = [x, y]
                box_xy_grid = activated_xy + cell_grid_val  # [2] - in grid coordinates
                
                # Step 3: Normalize to [0, 1]
                # CRITICAL: grid_w and grid_h must match the actual grid dimensions used during encoding
                box_xy_normalized = box_xy_grid / np.array([grid_w, grid_h])  # [2] - normalized [x, y]
                
                # Step 4: Convert to pixel coordinates
                abs_x = box_xy_normalized[0] * input_w
                abs_y = box_xy_normalized[1] * input_h
                
                # Debug for first few cells
                if len(boxes) < 3 and i < 20 and j < 20:
                    print(f"      DEBUG cell ({i}, {j}): stored_xy=({cell_stored_xy[0]:.3f}, {cell_stored_xy[1]:.3f}), "
                          f"activated_xy=({activated_xy[0]:.3f}, {activated_xy[1]:.3f}), "
                          f"cell_grid=({cell_grid_val[0]}, {cell_grid_val[1]}), "
                          f"box_xy_grid=({box_xy_grid[0]:.3f}, {box_xy_grid[1]:.3f}), "
                          f"normalized=({box_xy_normalized[0]:.6f}, {box_xy_normalized[1]:.6f}), "
                          f"abs_xy=({abs_x:.2f}, {abs_y:.2f}), grid_w={grid_w}, grid_h={grid_h}")
                
                # Step 5: Decode width/height - matching decoder: box_wh = anchors * exp(raw_wh) / input_shape
                box_wh_pixels = anchor * np.exp(cell_raw_wh)  # [2] - in pixels
                box_wh_normalized = box_wh_pixels / np.array([input_w, input_h])  # [2] - normalized
                abs_w = box_wh_normalized[0] * input_w  # width in pixels
                abs_h = box_wh_normalized[1] * input_h  # height in pixels
                
                # Convert from center format to corner format
                x1 = abs_x - abs_w / 2.0
                y1 = abs_y - abs_h / 2.0
                x2 = abs_x + abs_w / 2.0
                y2 = abs_y + abs_h / 2.0
                
                # Clip to image bounds
                x1 = np.clip(x1, 0, input_w)
                y1 = np.clip(y1, 0, input_h)
                x2 = np.clip(x2, 0, input_w)
                y2 = np.clip(y2, 0, input_h)
                
                boxes.append([x1, y1, x2, y2])
                classes.append(class_idx)
                anchor_indices.append(anchor_idx)
                grid_positions.append([i, j])
        
        # Deduplicate boxes if requested (keep only one box per unique position)
        if deduplicate and len(boxes) > 0:
            boxes_array = np.array(boxes)
            # Round to nearest pixel for deduplication
            boxes_rounded = np.round(boxes_array).astype(int)
            # Find unique boxes (based on all 4 coordinates)
            _, unique_indices = np.unique(boxes_rounded, axis=0, return_index=True)
            boxes = [boxes[i] for i in unique_indices]
            classes = [classes[i] for i in unique_indices]
            anchor_indices = [anchor_indices[i] for i in unique_indices]
            grid_positions = [grid_positions[i] for i in unique_indices]
        
        decoded_boxes_per_layer.append({
            'boxes': np.array(boxes) if len(boxes) > 0 else np.zeros((0, 4)),
            'classes': np.array(classes) if len(classes) > 0 else np.zeros((0,), dtype=np.int32),
            'anchors': np.array(anchor_indices) if len(anchor_indices) > 0 else np.zeros((0,), dtype=np.int32),
            'grid_positions': np.array(grid_positions) if len(grid_positions) > 0 else np.zeros((0, 2), dtype=np.int32),
            'layer_idx': layer_idx,
            'grid_shape': (grid_h, grid_w)
        })
    
    return decoded_boxes_per_layer


def parse_annotation_line(annotation_line: str) -> Tuple[str, np.ndarray]:
    """
    Parse annotation line to extract image path and boxes.
    
    Args:
        annotation_line: Annotation line in format "image_path x1,y1,x2,y2,class ..."
        
    Returns:
        Tuple of (image_path, boxes) where boxes is (N, 5) array
    """
    parts = annotation_line.strip().split()
    image_path = parts[0]
    
    boxes = []
    for box_str in parts[1:]:
        coords = [float(x) for x in box_str.split(',')]
        if len(coords) >= 5:
            boxes.append(coords[:5])  # x1, y1, x2, y2, class
    
    boxes = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.zeros((0, 5))
    return image_path, boxes


def visualize_assignments(image: np.ndarray,
                          original_boxes: np.ndarray,
                          decoded_boxes_per_layer: List[Dict[str, Any]],
                          grid_shapes: List[Tuple[int, int]],
                          class_names: List[str],
                          input_shape: Tuple[int, int],
                          output_path: str):
    """
    Visualize original boxes vs decoded y_true boxes.
    
    Args:
        image: Original image array
        original_boxes: Original annotation boxes (N, 5) in format (x1, y1, x2, y2, class)
        decoded_boxes_per_layer: List of decoded boxes per layer
        grid_shapes: Grid shapes for each layer
        class_names: List of class names
        input_shape: Input image shape (height, width)
        output_path: Path to save visualization
    """
    input_h, input_w = input_shape
    # Create figure with subplots
    num_layers = len(decoded_boxes_per_layer)
    fig = plt.figure(figsize=(20, 5 * (num_layers + 1)))
    
    # Color scheme
    original_color = (0, 255, 0)  # Green for original
    layer_colors = [
        (255, 0, 0),    # Red for layer 0
        (0, 0, 255),    # Blue for layer 1
        (255, 165, 0),  # Orange for layer 2
        (255, 0, 255),  # Magenta for layer 3
        (0, 255, 255),  # Cyan for layer 4
    ]
    
    # Plot 1: Original image with original boxes
    ax1 = plt.subplot(num_layers + 1, 1, 1)
    ax1.imshow(image)
    ax1.set_title('Original Annotation Boxes', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    for box in original_boxes:
        x1, y1, x2, y2, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax1.add_patch(rect)
        
        if cls < len(class_names):
            label = class_names[cls]
        else:
            label = f'Class {cls}'
        
        ax1.text(x1, y1 - 5, label, fontsize=10, color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot each layer
    for layer_idx, decoded_data in enumerate(decoded_boxes_per_layer):
        ax = plt.subplot(num_layers + 1, 1, layer_idx + 2)
        ax.imshow(image)
        ax.set_title(f'Layer {layer_idx} Decoded Boxes (Grid: {decoded_data["grid_shape"][0]}x{decoded_data["grid_shape"][1]})', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        boxes = decoded_data['boxes']
        classes = decoded_data['classes']
        anchors = decoded_data['anchors']
        grid_positions = decoded_data['grid_positions']
        
        # Draw grid
        grid_h, grid_w = decoded_data['grid_shape']
        cell_h = input_h / grid_h
        cell_w = input_w / grid_w
        
        for i in range(grid_h + 1):
            y = i * cell_h
            ax.axhline(y=y, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(grid_w + 1):
            x = j * cell_w
            ax.axvline(x=x, color='gray', linewidth=0.5, alpha=0.3)
        
        # Draw decoded boxes
        color = layer_colors[layer_idx % len(layer_colors)]
        color_normalized = [c / 255.0 for c in color]
        
        # Debug: Group boxes by object (same class and similar coordinates) to check 9-cell alignment
        if len(boxes) > 0:
            print(f"  Debug - Layer {layer_idx} (grid: {grid_h}x{grid_w}):")
            print(f"    Total boxes: {len(boxes)}")
            
            # Group boxes by class to find objects with multiple cells
            boxes_by_class = {}
            for idx, (box, cls, anchor_idx, grid_pos) in enumerate(zip(boxes, classes, anchors, grid_positions)):
                cls_key = int(cls)
                if cls_key not in boxes_by_class:
                    boxes_by_class[cls_key] = []
                boxes_by_class[cls_key].append((idx, box, grid_pos))
            
            # Check each class for 9-cell assignments
            for cls_key, cls_boxes in list(boxes_by_class.items()):
                if len(cls_boxes) >= 9:
                    print(f"    Class {cls_key}: {len(cls_boxes)} boxes (checking first 9 for alignment):")
                    try:
                        # Get first 9 boxes
                        first_9 = cls_boxes[:9]
                        box_coords = [b[1] for b in first_9]  # Extract box coordinates
                        grid_poss = [b[2] for b in first_9]  # Extract grid positions
                        
                        # Check if they decode to the same coordinates
                        x1_values = [float(b[0]) for b in box_coords]
                        y1_values = [float(b[1]) for b in box_coords]
                        x2_values = [float(b[2]) for b in box_coords]
                        y2_values = [float(b[3]) for b in box_coords]
                        
                        x1_std = np.std(x1_values)
                        y1_std = np.std(y1_values)
                        x2_std = np.std(x2_values)
                        y2_std = np.std(y2_values)
                        
                        print(f"      Grid positions: {[tuple(gp) for gp in grid_poss]}")
                        print(f"      Box coordinate std dev: x1={x1_std:.6f}, y1={y1_std:.6f}, x2={x2_std:.6f}, y2={y2_std:.6f}")
                        if x1_std > 1.0 or y1_std > 1.0:
                            print(f"      ⚠️  WARNING: Boxes are NOT aligned! Max deviation: x={max(x1_values)-min(x1_values):.2f}, y={max(y1_values)-min(y1_values):.2f}")
                            print(f"      All boxes per class:")
                            for i in range(max(0, len(first_9))):
                                x1, y1, x2, y2 = box_coords[i]
                                print(f"        Cell {grid_poss[i]}: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
                        else:
                            print(f"      ✓ Boxes are aligned (std < 1 pixel)")
                    except Exception as e:
                        print(f"      Error checking alignment: {e}")
                    break  # Only check first class with 9+ boxes
        
        for box, cls, anchor_idx, grid_pos in zip(boxes, classes, anchors, grid_positions):
            x1, y1, x2, y2 = box
            # Store original float values for debugging
            x1_float, y1_float, x2_float, y2_float = x1, y1, x2, y2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            anchor_idx = int(anchor_idx)
            grid_i, grid_j = int(grid_pos[0]), int(grid_pos[1])
            
            # Draw box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color_normalized, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw label
            if cls < len(class_names):
                label = f'{class_names[cls]} (A{anchor_idx}, G{grid_i},{grid_j})'
            else:
                label = f'Class {cls} (A{anchor_idx}, G{grid_i},{grid_j})'
            
            ax.text(x1, y1 - 5, label, fontsize=8, color=color_normalized,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Draw original boxes for comparison (lighter)
        for box in original_boxes:
            x1, y1, x2, y2, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor='green', facecolor='none', linestyle='--', alpha=0.5
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize y_true assignments to verify correctness')
    parser.add_argument('--annotation', type=str, required=True,
                       help='Annotation line or path to annotation file (if file, uses first line)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML file')
    parser.add_argument('--output', type=str, default='y_true_visualization.png',
                       help='Output path for visualization image')
    parser.add_argument('--augment', action='store_true',
                       help='Enable augmentation (default: False, for testing)')
    parser.add_argument('--deduplicate', action='store_true',
                       help='Only show one box per object (deduplicate 9-cell assignments)')
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = ConfigLoader.load_config(args.config)
    model_config_path = config.get('model_config')
    if not model_config_path:
        raise ValueError("model_config not found in config")
    
    model_config = ConfigLoader.load_config(model_config_path)
    
    # Get paths and parameters
    data_config = config.get('data', {})
    classes_path = data_config.get('classes_path')
    if not classes_path:
        classes_path = model_config['model']['preset'].get('classes_path')
    
    anchors_path = model_config['model']['preset']['anchors_path']
    input_shape = tuple(model_config['model']['preset']['input_shape'][:2])
    
    # Load classes and anchors
    print("Loading classes and anchors...")
    class_names = load_classes(classes_path)
    anchors = load_anchors(anchors_path)
    num_classes = len(class_names)
    
    print(f"   Classes: {num_classes}")
    print(f"   Anchors: {len(anchors)} scales")
    print(f"   Input shape: {input_shape}")
    
    # Load annotation
    print("\nLoading annotation...")
    if os.path.isfile(args.annotation):
        # Load first line from file
        lines = load_annotation_lines(args.annotation, shuffle=False)
        if len(lines) == 0:
            raise ValueError(f"No annotations found in {args.annotation}")
        annotation_line = lines[0]
        print(f"   Using first annotation from file: {args.annotation}")
    else:
        annotation_line = args.annotation
    
    image_path, original_boxes = parse_annotation_line(annotation_line)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"   Image: {image_path}")
    print(f"   Original boxes: {len(original_boxes)}")
    
    # Load image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image)
    original_image_shape = image.size  # (width, height)
    print(f"   Original image shape: {original_image_shape}")
    
    # Resize image to input shape for processing
    image_resized = image.resize((input_shape[1], input_shape[0]))
    image_resized_array = np.array(image_resized)
    
    # Compute grid shapes
    num_layers = len(anchors)
    grid_shapes = [(input_shape[0] // {0: 32, 1: 16, 2: 8, 3: 4, 4: 2}[l],
                    input_shape[1] // {0: 32, 1: 16, 2: 8, 3: 4, 4: 2}[l])
                   for l in range(num_layers)]
    
    # FIXED: Use tf_preprocess_true_boxes directly instead of generator
    # The generator uses the old NumPy preprocess_true_boxes which has bugs
    print("\nProcessing boxes with tf_preprocess_true_boxes...")
    
    # Parse annotation to get boxes in (x1, y1, x2, y2, class) format
    parts = annotation_line.strip().split()
    boxes_list = []
    for box_str in parts[1:]:
        coords = [float(x) for x in box_str.split(',')]
        if len(coords) >= 5:
            boxes_list.append(coords[:5])  # x1, y1, x2, y2, class
    
    # Scale boxes to input_shape if image was resized
    if original_image_shape != (input_shape[1], input_shape[0]):
        scale_x = input_shape[1] / original_image_shape[0]
        scale_y = input_shape[0] / original_image_shape[1]
        for box in boxes_list:
            box[0] *= scale_x  # x1
            box[1] *= scale_y  # y1
            box[2] *= scale_x  # x2
            box[3] *= scale_y  # y2
    
    # Convert to numpy array and add batch dimension: (1, num_boxes, 5)
    if len(boxes_list) == 0:
        boxes_array = np.zeros((1, 1, 5), dtype=np.float32)
    else:
        boxes_array = np.array([boxes_list], dtype=np.float32)  # (1, num_boxes, 5)
    
    # Convert anchors to TensorFlow tensors
    anchors_tf = [tf.constant(anchor, dtype=tf.float32) for anchor in anchors]
    
    # Call tf_preprocess_true_boxes directly
    boxes_tf = tf.constant(boxes_array, dtype=tf.float32)
    y_true_list = tf_preprocess_true_boxes(
        boxes_tf,
        input_shape,
        anchors_tf,
        num_classes,
        multi_anchor_assign=False,
        grid_shapes=grid_shapes
    )
    
    # Convert to numpy for visualization
    y_true_list = [y.numpy() for y in y_true_list]
    
    # Create dummy image batch for visualization
    images_batch = np.expand_dims(image_resized_array, axis=0)  # (1, H, W, 3)
    
    print(f"   Image batch shape: {images_batch.shape}")
    print(f"   Number of y_true layers: {len(y_true_list)}")
    print(f"   grid_shapes: {grid_shapes}")
    print(f"   input_shape: {input_shape}")

    for i, y_true in enumerate(y_true_list):
        print(f"   Layer {i} y_true shape: {y_true.shape}")
    
    # Decode y_true to boxes
    print("\nDecoding y_true to boxes...")
    decoded_boxes_per_layer = decode_y_true_to_boxes(
        y_true_list, anchors, input_shape, grid_shapes, num_classes, deduplicate=args.deduplicate
    )
    
    # Print statistics
    print("\nDecoded boxes statistics:")
    for layer_idx, decoded_data in enumerate(decoded_boxes_per_layer):
        num_boxes = len(decoded_data['boxes'])
        print(f"   Layer {layer_idx}: {num_boxes} boxes")
        if num_boxes > 0:
            unique_classes = np.unique(decoded_data['classes'])
            unique_anchors = np.unique(decoded_data['anchors'])
            print(f"      Classes: {unique_classes}")
            print(f"      Anchors: {unique_anchors}")
    
    # Scale original boxes to input shape if needed
    if original_image_shape != (input_shape[1], input_shape[0]):
        scale_x = input_shape[1] / original_image_shape[0]
        scale_y = input_shape[0] / original_image_shape[1]
        original_boxes_scaled = original_boxes.copy()
        original_boxes_scaled[:, [0, 2]] *= scale_x
        original_boxes_scaled[:, [1, 3]] *= scale_y
    else:
        original_boxes_scaled = original_boxes
    
    # Visualize
    print("\nCreating visualization...")
    visualize_assignments(
        image_resized_array,
        original_boxes_scaled,
        decoded_boxes_per_layer,
        grid_shapes,
        class_names,
        input_shape,
        args.output
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

