#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script to show exactly what the model sees after all training augmentations.

This script:
1. Loads the training config and builds a MultiGridDataGenerator using real augmentation settings
2. Runs the same tf.data pipeline (including Mosaic/MixUp, fixed-capacity padding, etc.)
3. For each batch, draws every sample's bounding boxes (with class labels) on the augmented images
4. Saves visualizations and metadata to an output directory

The goal is to visually confirm that augmentations and annotations are correct without running a full training job.
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.config import ConfigLoader
from multigriddet.data.generators import MultiGridDataGenerator
from multigriddet.data.utils import load_annotation_lines, get_classes, get_colors, draw_boxes
from multigriddet.utils.anchors import load_anchors




def filter_valid_boxes(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out invalid (padded) boxes.
    
    Args:
        boxes: Box array of shape [N, 5] with format [x1, y1, x2, y2, class]
        
    Returns:
        Tuple of (valid_boxes, valid_indices)
    """
    if len(boxes) == 0:
        return np.zeros((0, 5)), np.array([], dtype=int)
    
    # Check for valid boxes (non-zero area and valid coordinates)
    # CRITICAL: Padded boxes have all zeros (x1=0, y1=0, x2=0, y2=0, class=0)
    # We need to exclude these by checking that boxes have non-zero area AND non-zero coordinates
    # Simply checking class >= 0 is insufficient because class 0 might be a valid class!
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    has_nonzero_coords = (boxes[:, 0] != 0) | (boxes[:, 1] != 0) | (boxes[:, 2] != 0) | (boxes[:, 3] != 0)
    
    valid_mask = (
        (boxes[:, 0] < boxes[:, 2]) &  # x1 < x2
        (boxes[:, 1] < boxes[:, 3]) &  # y1 < y2
        (boxes[:, 0] >= 0) &  # x1 >= 0
        (boxes[:, 1] >= 0) &  # y1 >= 0
        (box_areas > 0)
    )
    
    valid_indices = np.where(valid_mask)[0]
    valid_boxes = boxes[valid_indices]
    
    return valid_boxes, valid_indices


def visualize_batch_sample(image: np.ndarray,
                           boxes: np.ndarray,
                           class_names: List[str],
                           colors: List[Tuple[int, int, int]],
                           batch_idx: int,
                           sample_idx: int,
                           output_dir: Path) -> Dict[str, Any]:
    """
    Visualize a single sample from a batch.
    
    Args:
        image: Image array in normalized float32 format [0, 1]
        boxes: Box array of shape [N, 5] with format [x1, y1, x2, y2, class]
        class_names: List of class names
        colors: List of RGB color tuples for each class
        batch_idx: Batch index
        sample_idx: Sample index within batch
        output_dir: Output directory for saving files
        
    Returns:
        Dictionary with metadata about the visualization
    """
    # Filter valid boxes
    valid_boxes, valid_indices = filter_valid_boxes(boxes)
    
    # CRITICAL: Convert image from normalized float32 [0, 1] to uint8 [0, 255] BEFORE drawing boxes
    # Images from the dataset are now normalized to [0, 1] range at the end of preprocessing
    # (after all augmentations), so they should always be in [0, 1] range.
    # OpenCV drawing functions require uint8 images. Drawing on normalized images will result in dark/black images.
    
    # First, ensure we have a copy to avoid modifying the original
    image_to_draw = image.copy()
    
    # Convert to uint8 - images should be in [0, 1] range after normalization
    if image_to_draw.dtype == np.float32 or image_to_draw.dtype == np.float64:
        # Images are normalized to [0, 1] - clip to ensure valid range and convert
        image_clipped = np.clip(image_to_draw, 0.0, 1.0)
        image_uint8 = (image_clipped * 255.0).astype(np.uint8)
        
        # Debug check: warn if values are outside expected range (shouldn't happen now)
        img_max = float(image_to_draw.max())
        img_min = float(image_to_draw.min())
        if img_min < -0.01 or img_max > 1.01:
            print(f"   WARNING: Image values outside [0, 1] range (min={img_min:.6f}, max={img_max:.6f})")
            print(f"   This should not happen with the new normalization approach!")
    else:
        # Already integer type, but ensure it's in valid range and uint8
        image_clipped = np.clip(image_to_draw, 0, 255)
        image_uint8 = image_clipped.astype(np.uint8)
    
    # Safety check: Verify image is not all black or has very low values before proceeding
    image_max_uint8 = image_uint8.max()
    image_mean_uint8 = image_uint8.mean()
    
    if image_max_uint8 == 0:
        print(f"   ERROR: Image is completely black (all zeros)!")
        print(f"   Original image stats: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}")
        print(f"   After conversion: max={image_max_uint8}, mean={image_mean_uint8:.2f}")
        # Create a placeholder image to avoid saving black image
        image_uint8 = np.ones_like(image_uint8) * 128  # Gray placeholder
    elif image_mean_uint8 < 5.0:
        # Very dark image - might indicate conversion issue
        print(f"   WARNING: Image is very dark (mean={image_mean_uint8:.2f}, max={image_max_uint8})")
        print(f"   Original image stats: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}")
    
    # Ensure image is in BGR format for OpenCV (draw_boxes expects BGR)
    # TensorFlow loads images as RGB, so we need to convert to BGR for OpenCV
    if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_uint8.copy()
    
    # Prepare data for draw_boxes
    # CRITICAL: Use class names (not indices) to catch mislabeling issues
    if len(valid_boxes) > 0:
        boxes_list = valid_boxes[:, :4].tolist()  # [x1, y1, x2, y2]
        # Extract class indices - ensure they're integers and within valid range
        class_indices_raw = valid_boxes[:, 4]
        class_indices = np.clip(class_indices_raw, 0, len(class_names) - 1).astype(np.int32)
        
        # Convert to class names for visualization (helps catch mislabeling)
        # draw_boxes expects indices, but we validate them first
        classes_list = class_indices.tolist()
        scores_list = [1.0] * len(valid_boxes)  # Dummy scores for visualization
        
        # Debug: Check for invalid or suspicious class indices
        invalid_classes = class_indices_raw >= len(class_names)
        clipped_classes = class_indices != class_indices_raw.astype(np.int32)
        if np.any(invalid_classes) or np.any(clipped_classes):
            if np.any(invalid_classes):
                invalid_indices = class_indices_raw[invalid_classes]
                print(f"   ERROR: Sample {sample_idx + 1} has invalid class indices: {invalid_indices}")
                print(f"   Valid class range: [0, {len(class_names) - 1}]")
            if np.any(clipped_classes):
                clipped_indices = class_indices_raw[clipped_classes]
                print(f"   WARNING: Sample {sample_idx + 1} has clipped class indices: {clipped_indices}")
                print(f"   These were clipped to valid range [0, {len(class_names) - 1}]")
            
            # Show which class names are being used
            unique_classes = np.unique(class_indices)
            used_class_names = [class_names[int(c)] for c in unique_classes]
            print(f"   Used class names in this sample: {used_class_names}")
    else:
        boxes_list = []
        classes_list = []
        scores_list = []
    
    # Draw boxes on the uint8 BGR image (NOT on the normalized image)
    # image_bgr is already converted to uint8 [0, 255] and BGR format
    image_with_boxes = image_bgr.copy()
    if len(boxes_list) > 0:
        # draw_boxes expects indices, looks up class names internally
        # This ensures class names are displayed correctly in visualization
        image_with_boxes = draw_boxes(
            image_with_boxes,
            boxes_list,
            classes_list,  # Indices - draw_boxes will look up names
            scores_list,
            class_names,  # Pass class_names so draw_boxes can display them
            colors,
            show_score=False
        )
    
    # Save visualization (image_with_boxes is uint8 BGR, ready for cv2.imwrite)
    output_filename = f"batch_{batch_idx:04d}_sample_{sample_idx:04d}.png"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), image_with_boxes)
    
    # Prepare metadata
    metadata = {
        "batch_idx": batch_idx,
        "sample_idx": sample_idx,
        "image_shape": list(image.shape),
        "num_boxes_total": len(boxes),
        "num_boxes_valid": len(valid_boxes),
        "boxes": []
    }
    
    # Add box information
    for i, box in enumerate(valid_boxes):
        x1, y1, x2, y2, cls = box
        # Ensure class index is valid integer
        cls_int = int(np.round(cls))
        cls_int = max(0, min(cls_int, len(class_names) - 1))  # Clip to valid range
        class_name = class_names[cls_int] if cls_int < len(class_names) else f"INVALID_CLASS_{cls_int}"
        metadata["boxes"].append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "class_id": cls_int,
            "class_name": class_name,
            "width": float(x2 - x1),
            "height": float(y2 - y1)
        })
    
    # Save metadata
    metadata_filename = f"batch_{batch_idx:04d}_sample_{sample_idx:04d}.json"
    metadata_path = output_dir / metadata_filename
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Visualize augmented training batches to verify augmentations and annotations"
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=5,
        help='Number of batches to visualize (default: 5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training config file (default: configs/train_config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/visualizations',
        help='Output directory for visualizations (default: tests/visualizations)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("=" * 80)
    print("MultiGridDet Augmented Batch Visualization")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of batches: {args.num_batches}")
    print()
    
    # Load configuration
    try:
        config = ConfigLoader.load_config(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] Config file not found: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Error loading config: {e}")
        return 1
    
    # Load model configuration
    model_config_path = config.get('model_config')
    if not model_config_path:
        print("[ERROR] model_config not specified in config")
        return 1
    
    try:
        model_config = ConfigLoader.load_config(model_config_path)
    except Exception as e:
        print(f"[ERROR] Error loading model config: {e}")
        return 1
    
    # Merge configs
    full_config = ConfigLoader.merge_configs(model_config, config)
    
    # Get paths and parameters
    data_config = full_config.get('data', {})
    training_config = full_config.get('training', {})
    augment_config = training_config.get('augmentation', {})
    
    train_annotation = data_config.get('train_annotation')
    classes_path = data_config.get('classes_path')
    
    if not train_annotation:
        print("[ERROR] data.train_annotation not specified in config")
        return 1
    if not classes_path:
        print("[ERROR] data.classes_path not specified in config")
        return 1
    
    # Load annotation lines
    print("Loading annotation lines...")
    annotation_lines = load_annotation_lines(train_annotation)
    print(f"   Loaded {len(annotation_lines)} annotation lines")
    
    # Get model parameters
    model_preset = model_config.get('model', {}).get('preset', {})
    input_shape = tuple(model_preset.get('input_shape', [608, 608, 3])[:2])
    anchors_path = model_preset.get('anchors_path')
    
    if not anchors_path:
        print("[ERROR] anchors_path not found in model config")
        return 1
    
    # Load anchors and classes
    print("Loading anchors and classes...")
    anchors = load_anchors(anchors_path)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    colors_rgb = get_colors(num_classes)
    # Convert RGB colors to BGR for OpenCV (draw_boxes uses cv2 which expects BGR)
    colors = [(b, g, r) for (r, g, b) in colors_rgb]
    
    print(f"   Classes: {num_classes}")
    print(f"   Anchors: {len(anchors)} scales")
    print(f"   Input shape: {input_shape}")
    
    # Get training parameters
    batch_size = training_config.get('batch_size', 4)
    num_workers = full_config.get('data_loader', {}).get('num_workers', 8)
    
    # Get augmentation parameters
    augment_enabled = augment_config.get('enabled', True)
    enhance_type = augment_config.get('enhance_type')
    mosaic_prob = augment_config.get('mosaic_prob', 0.3)
    mixup_prob = augment_config.get('mixup_prob', 0.1)
    rescale_interval = augment_config.get('rescale_interval', -1)
    max_boxes_per_image = augment_config.get('max_boxes_per_image', 100)
    
    print()
    print("Augmentation settings:")
    print(f"   Enabled: {augment_enabled}")
    print(f"   Enhance type: {enhance_type}")
    print(f"   Mosaic probability: {mosaic_prob}")
    print(f"   MixUp probability: {mixup_prob}")
    print(f"   Rescale interval: {rescale_interval}")
    print(f"   Max boxes per image: {max_boxes_per_image}")
    print()
    
    # Create data generator
    print("Creating data generator...")
    generator = MultiGridDataGenerator(
        annotation_lines=annotation_lines,
        batch_size=batch_size,
        input_shape=input_shape,
        anchors=anchors,
        num_classes=num_classes,
        augment=augment_enabled,
        enhance_augment=enhance_type,
        rescale_interval=rescale_interval,
        multi_anchor_assign=training_config.get('multi_anchor_assign', False),
        shuffle=True,
        num_workers=num_workers,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        max_boxes_per_image=max_boxes_per_image
    )
    print("   Data generator created")
    
    # Build visualization dataset using the actual training pipeline
    print("Building visualization dataset...")
    dataset = generator.build_visualization_dataset(
        prefetch_buffer_size=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        shuffle_buffer_size=4096,
        interleave_cycle_length=None
    )
    print("   Dataset built")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    print()
    
    # Iterate through batches
    print("Processing batches...")
    batch_count = 0
    total_samples = 0
    
    try:
        for batch_images, batch_boxes in dataset:
            if batch_count >= args.num_batches:
                break
            
            batch_size_actual = batch_images.shape[0]
            print(f"Batch {batch_count + 1}/{args.num_batches} (size: {batch_size_actual})")
            
            # Convert to numpy
            batch_images_np = batch_images.numpy()
            batch_boxes_np = batch_boxes.numpy()
            
            # Process each sample in the batch
            for sample_idx in range(batch_size_actual):
                image = batch_images_np[sample_idx]
                boxes = batch_boxes_np[sample_idx]
                
                # Debug: Check image statistics to catch issues early
                image_min = float(image.min())
                image_max = float(image.max())
                image_mean = float(image.mean())
                image_dtype = image.dtype
                
                # Check for problematic images before conversion
                if image_max < 0.01 or image_mean < 0.001:
                    print(f"   WARNING: Sample {sample_idx + 1} appears to be black/dark before conversion")
                    print(f"   Stats: dtype={image_dtype}, min={image_min:.6f}, max={image_max:.6f}, mean={image_mean:.6f}")
                elif image_min < -0.5 or image_max > 2.0:
                    print(f"   INFO: Sample {sample_idx + 1} has values outside [0, 1] (likely from augmentations)")
                    print(f"   Stats: dtype={image_dtype}, min={image_min:.6f}, max={image_max:.6f}, mean={image_mean:.6f}")
                
                # Visualize sample
                metadata = visualize_batch_sample(
                    image=image,
                    boxes=boxes,
                    class_names=class_names,
                    colors=colors,
                    batch_idx=batch_count,
                    sample_idx=sample_idx,
                    output_dir=output_dir
                )
                
                total_samples += 1
                print(f"   Sample {sample_idx + 1}: {metadata['num_boxes_valid']} valid boxes")
            
            batch_count += 1
            print()
    
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Error processing batches: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 80)
    print("Visualization complete")
    print(f"   Processed {batch_count} batches")
    print(f"   Total samples: {total_samples}")
    print(f"   Output directory: {output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

