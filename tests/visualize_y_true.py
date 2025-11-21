#!/usr/bin/env python3
"""
Visualize y_true assignments to verify encoding/decoding correctness.

Loads an annotation, processes it through the training pipeline to get y_true tensors,
decodes them back to image coordinates, and visualizes original vs decoded boxes.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
import tensorflow as tf
from PIL import Image

# Enable eager execution for .numpy() calls
tf.config.run_functions_eagerly(True)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.config import ConfigLoader
from multigriddet.data.generators import tf_preprocess_true_boxes
from multigriddet.data.utils import load_annotation_lines, get_classes, get_colors
from multigriddet.utils.anchors import load_anchors


def decode_y_true_to_boxes(y_true_list: List[np.ndarray], 
                          anchors: List[np.ndarray],
                          input_shape: Tuple[int, int],
                          grid_shapes: List[Tuple[int, int]],
                          num_classes: int,
                          deduplicate: bool = False) -> List[Dict[str, Any]]:
    """
    Decode y_true tensors back to bounding boxes in image coordinates.
    
    This mirrors the inverse of the encoding logic from tf_preprocess_true_boxes
    and the working reference implementation in tests/visualize_y_true_old.py.
    
    Args:
        y_true_list: List of y_true tensors for each layer [batch, grid_h, grid_w, features]
        anchors: List of anchor arrays for each layer
        input_shape: Input image shape (height, width)
        grid_shapes: Grid shapes for each layer [(grid_h, grid_w), ...]
        num_classes: Number of classes
        deduplicate: If True, keep only one box per unique position (rounded to pixel)
        
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
        # Use actual grid dimensions from y_true.shape to ensure alignment with encoding
        grid_h, grid_w = y_true.shape[1], y_true.shape[2]
        
        num_anchors = len(anchors[layer_idx])
        
        # Extract components - y_true stores already-activated offsets and raw_wh
        stored_xy = y_true[..., 0:2]        # [batch, grid_h, grid_w, 2] - already-activated offset
        raw_wh = y_true[..., 2:4]           # [batch, grid_h, grid_w, 2] - log-space raw_wh
        objectness = y_true[..., 4:5]       # [batch, grid_h, grid_w, 1]
        anchor_one_hot = y_true[..., 5:5+num_anchors]      # [batch, grid_h, grid_w, num_anchors]
        class_one_hot = y_true[..., 5+num_anchors:]        # [batch, grid_h, grid_w, num_classes]
        
        # Find cells with objects (objectness > 0.5)
        object_mask = objectness[0, ...] > 0.5  # [grid_h, grid_w]
        
        # Create grid coordinates matching encoder/decoder logic
        grid_y = np.arange(grid_h)
        grid_x = np.arange(grid_w)
        x_offset, y_offset = np.meshgrid(grid_x, grid_y)
        # [grid_h, grid_w, 2] where entry is [x, y] = [j, i]
        cell_grid = np.stack([x_offset, y_offset], axis=-1)
        
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
                cell_stored_xy = stored_xy[0, i, j, :]    # [2] - already-activated offset [-kj + ty, -ki + tx]
                cell_raw_wh = raw_wh[0, i, j, :]          # [2] - log-space raw_wh
                cell_anchor = anchor_one_hot[0, i, j, :]  # [num_anchors]
                cell_class = class_one_hot[0, i, j, :]    # [num_classes]
                
                # Get anchor index
                anchor_idx = int(np.argmax(cell_anchor))
                anchor = anchors[layer_idx][anchor_idx]
                
                # Get class index
                class_idx = int(np.argmax(cell_class))
                if class_idx >= num_classes:
                    continue
                
                # Decode center coordinates:
                # stored_xy is already-activated offset, add cell grid position
                activated_xy = cell_stored_xy
                cell_grid_val = cell_grid[i, j, :]              # [2] - [x, y] = [j, i]
                box_xy_grid = activated_xy + cell_grid_val      # [2] - grid coordinates
                # Normalize using actual grid size
                box_xy_normalized = box_xy_grid / np.array([grid_w, grid_h], dtype=np.float32)
                
                # Convert to pixel coordinates
                abs_x = float(box_xy_normalized[0] * input_w)
                abs_y = float(box_xy_normalized[1] * input_h)
                
                # Decode width/height: anchors are in pixels, raw_wh is log-space
                box_wh_pixels = anchor * np.exp(cell_raw_wh)    # [2] in pixels
                abs_w = float(box_wh_pixels[0])
                abs_h = float(box_wh_pixels[1])
                
                # Convert from center format to corner format
                x1 = abs_x - abs_w / 2.0
                y1 = abs_y - abs_h / 2.0
                x2 = abs_x + abs_w / 2.0
                y2 = abs_y + abs_h / 2.0
                
                # Clip to image bounds
                x1 = float(np.clip(x1, 0, input_w))
                y1 = float(np.clip(y1, 0, input_h))
                x2 = float(np.clip(x2, 0, input_w))
                y2 = float(np.clip(y2, 0, input_h))
                
                boxes.append([x1, y1, x2, y2])
                classes.append(class_idx)
                anchor_indices.append(anchor_idx)
                grid_positions.append([i, j])
        
        # Optional deduplication: keep only one box per unique rounded pixel box
        if deduplicate and len(boxes) > 0:
            boxes_array = np.array(boxes)
            boxes_rounded = np.round(boxes_array).astype(int)
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




def visualize_assignments(image: np.ndarray,
                          original_boxes: np.ndarray,
                          decoded_boxes_per_layer: List[Dict[str, Any]],
                          grid_shapes: List[Tuple[int, int]],
                          class_names: List[str],
                          input_shape: Tuple[int, int],
                          output_path: str):
    """Visualize original annotation boxes vs decoded y_true boxes."""
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
        x1, y1, x2, y2, obj_cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        obj_cls = int(obj_cls)
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax1.add_patch(rect)
        
        if obj_cls < len(class_names):
            label = class_names[obj_cls]
        else:
            label = f'Class {obj_cls}'
        
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
        
        
        for box, obj_cls, anchor_idx, grid_pos in zip(boxes, classes, anchors, grid_positions):
            x1, y1, x2, y2 = box
            # Store original float values for debugging
            x1_float, y1_float, x2_float, y2_float = x1, y1, x2, y2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            obj_cls = int(obj_cls)
            anchor_idx = int(anchor_idx)
            grid_i, grid_j = int(grid_pos[0]), int(grid_pos[1])
            
            # Draw box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color_normalized, facecolor='none'
            )
            ax.add_patch(rect)
            
            if obj_cls < len(class_names):
                label = f'{class_names[obj_cls]} (A{anchor_idx}, G{grid_i},{grid_j})'
            else:
                label = f'Class{obj_cls} (A{anchor_idx}, G{grid_i},{grid_j})'
            
            ax.text(x1, y1 - 5, label, fontsize=8, color=color_normalized,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Draw original boxes for comparison (lighter)
        for box in original_boxes:
            x1, y1, x2, y2, obj_cls = box
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
    
    print("Loading configuration...")
    config = ConfigLoader.load_config(args.config)
    model_config_path = config.get('model_config')
    if not model_config_path:
        raise ValueError("model_config not found in config")
    
    model_config = ConfigLoader.load_config(model_config_path)
    
    data_config = config.get('data', {})
    classes_path = data_config.get('classes_path') or model_config['model']['preset'].get('classes_path')
    anchors_path = model_config['model']['preset']['anchors_path']
    input_shape = tuple(model_config['model']['preset']['input_shape'][:2])
    
    print("Loading classes and anchors...")
    class_names = get_classes(classes_path)
    anchors = load_anchors(anchors_path)
    num_classes = len(class_names)
    
    print(f"   Classes: {num_classes}")
    print(f"   Anchors: {len(anchors)} scales")
    print(f"   Input shape: {input_shape}")
    
    print("\nLoading annotation...")
    if Path(args.annotation).is_file():
        lines = load_annotation_lines(args.annotation)
        if not lines:
            raise ValueError(f"No annotations found in {args.annotation}")
        annotation_line = lines[0]
        print(f"   Using first annotation from file: {args.annotation}")
    else:
        annotation_line = args.annotation
    
    parts = annotation_line.strip().split()
    if not parts:
        raise ValueError("Empty annotation line")
    
    image_path = parts[0]
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"   Image: {image_path}")
    
    boxes_list = []
    for box_str in parts[1:]:
        coords = [float(x) for x in box_str.split(',')]
        if len(coords) >= 5:
            boxes_list.append(coords[:5])
    
    original_boxes = np.array(boxes_list, dtype=np.float32) if boxes_list else np.zeros((0, 5), dtype=np.float32)
    
    print(f"   Original boxes: {len(original_boxes)}")
    
    pil_image = Image.open(image_path)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    image_array = np.array(pil_image, dtype=np.uint8)
    original_h, original_w = image_array.shape[:2]
    print(f"   Original image shape: ({original_w}, {original_h})")
    
    from multigriddet.data.augmentation import letterbox_resize
    target_size = (input_shape[1], input_shape[0])
    resized_image, padding_size, offset = letterbox_resize(
        pil_image, target_size=target_size, return_padding_info=True
    )
    image_resized_array = np.array(resized_image, dtype=np.uint8)
    
    pad_left_val, pad_top_val = offset
    new_w_val, new_h_val = padding_size
    
    # Match letterbox geometry used in training pipeline (get_ground_truth_data / reshape_boxes):
    # scale = min(new_w / original_w, new_h / original_h)
    # x = x * scale + pad_left,  y = y * scale + pad_top  (only for coordinates)
    scale = min(new_w_val / original_w, new_h_val / original_h)
    boxes_scaled = original_boxes.copy()
    if boxes_scaled.size > 0:
        boxes_scaled[:, [0, 2]] = boxes_scaled[:, [0, 2]] * scale + pad_left_val
        boxes_scaled[:, [1, 3]] = boxes_scaled[:, [1, 3]] * scale + pad_top_val
    
    num_layers = len(anchors)
    grid_shapes = [(input_shape[0] // (32 >> l), input_shape[1] // (32 >> l)) for l in range(num_layers)]
    
    print("\nProcessing boxes with tf_preprocess_true_boxes...")
    boxes_batch = np.expand_dims(boxes_scaled, axis=0)
    boxes_tf = tf.constant(boxes_batch, dtype=tf.float32)
    anchors_tf = [tf.constant(anchor, dtype=tf.float32) for anchor in anchors]
    
    with tf.device('/CPU:0'):
        y_true_list = tf_preprocess_true_boxes(
            boxes_tf, input_shape, anchors_tf, num_classes,
            multi_anchor_assign=False, grid_shapes=grid_shapes
        )
        y_true_list = [y.numpy() for y in y_true_list]
    
    print(f"   Number of y_true layers: {len(y_true_list)}")
    for i, y_true in enumerate(y_true_list):
        objectness = y_true[0, ..., 4:5]
        num_objects = np.sum(objectness > 0.5)
        print(f"   Layer {i} y_true shape: {y_true.shape}, objects: {num_objects}")
    
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
    
    print("\nCreating visualization...")
    visualize_assignments(
        image_resized_array,
        boxes_scaled,
        decoded_boxes_per_layer,
        grid_shapes,
        class_names,
        input_shape,
        args.output
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
