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
from multigriddet.data.generators import (
    MultiGridDataGenerator,
    tf_parse_annotation_line,
    tf_load_and_decode_image,
    tf_parse_boxes,
    tf_letterbox_resize,
    tf_random_resize_crop_pad,
    tf_random_horizontal_flip,
    tf_random_brightness,
    tf_random_contrast,
    tf_random_saturation,
    tf_random_hue,
    tf_random_grayscale,
    tf_random_rotate,
    tf_random_gridmask,
    tf_random_mosaic,
    tf_random_mixup
)
from multigriddet.data.utils import load_annotation_lines, get_classes, get_colors, draw_boxes
from multigriddet.utils.anchors import load_anchors


def build_visualization_dataset(generator: MultiGridDataGenerator,
                                prefetch_buffer_size=tf.data.AUTOTUNE,
                                num_parallel_calls=tf.data.AUTOTUNE,
                                shuffle_buffer_size: int = 4096,
                                interleave_cycle_length: int = None):
    """
    Build a tf.data.Dataset that returns (images, boxes_dense) for visualization.
    
    This replicates the pipeline from build_tf_dataset() but stops before _process_batch_wrapper()
    to capture boxes in their original [x1, y1, x2, y2, class] format.
    
    Args:
        generator: MultiGridDataGenerator instance
        prefetch_buffer_size: Prefetch buffer size
        num_parallel_calls: Number of parallel calls
        shuffle_buffer_size: Shuffle buffer size
        interleave_cycle_length: Interleave cycle length
        
    Returns:
        tf.data.Dataset yielding (images, boxes_dense) tuples
    """
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Create dataset from annotation lines
    annotation_paths = tf.constant(generator.annotation_lines)
    dataset = tf.data.Dataset.from_tensor_slices(annotation_paths)
    
    # Shuffle dataset
    if generator.shuffle:
        buffer_size = min(shuffle_buffer_size, len(generator.annotation_lines))
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    # Load and parse annotations
    if interleave_cycle_length is not None and interleave_cycle_length > 1:
        def _load_and_parse(annotation_line):
            image_path, boxes_string = tf_parse_annotation_line(annotation_line)
            image = tf_load_and_decode_image(image_path)
            boxes = tf_parse_boxes(boxes_string)
            return image, boxes, image_path
        
        interleave_parallel_calls = tf.data.AUTOTUNE
        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.from_tensors(x).map(_load_and_parse),
            cycle_length=interleave_cycle_length,
            block_length=1,
            num_parallel_calls=interleave_parallel_calls,
            deterministic=False
        )
    else:
        def _load_and_parse(annotation_line):
            image_path, boxes_string = tf_parse_annotation_line(annotation_line)
            image = tf_load_and_decode_image(image_path)
            boxes = tf_parse_boxes(boxes_string)
            return image, boxes, image_path
        
        dataset = dataset.map(_load_and_parse, num_parallel_calls=num_parallel_calls)
    
    # Preprocess and augment
    if hasattr(generator, 'input_shape_list') and len(generator.input_shape_list) > 0:
        input_shape_list_tf = tf.constant(generator.input_shape_list, dtype=tf.int32)
        input_shape_base_tf = tf.constant(generator.input_shape, dtype=tf.int32)
        has_multiscale = True
    else:
        input_shape_list_tf = None
        input_shape_base_tf = tf.constant(generator.input_shape, dtype=tf.int32)
        has_multiscale = False
    
    def _preprocess_image_and_boxes(image, boxes, image_path):
        # Get original image size BEFORE normalization (boxes are in original coordinates)
        image_shape = tf.shape(image)
        src_h = tf.cast(image_shape[0], tf.float32)
        src_w = tf.cast(image_shape[1], tf.float32)
        
        # Convert image to float32 and normalize
        image = tf.cast(image, tf.float32) / 255.0
        
        # Multi-scale handling
        if has_multiscale and generator.rescale_interval > 0:
            num_scales = tf.shape(input_shape_list_tf)[0]
            scale_idx = tf.random.uniform([], 0, num_scales, dtype=tf.int32)
            target_shape_sample = tf.gather(input_shape_list_tf, scale_idx)
            scale_h = tf.cast(target_shape_sample[0], tf.float32) / tf.cast(input_shape_base_tf[0], tf.float32)
            scale_w = tf.cast(target_shape_sample[1], tf.float32) / tf.cast(input_shape_base_tf[1], tf.float32)
            scaled_h = tf.cast(src_h * scale_h, tf.int32)
            scaled_w = tf.cast(src_w * scale_w, tf.int32)
            image_scaled = tf.image.resize(image, [scaled_h, scaled_w], method='bicubic')
            # Letterbox resize with padding info
            image_resized, (new_w, new_h), (pad_left, pad_top) = tf_letterbox_resize(
                image_scaled, generator.input_shape, return_padding_info=True
            )
            # Transform boxes: first scale, then add padding offset
            scale = tf.minimum(
                tf.cast(new_w, tf.float32) / tf.cast(scaled_w, tf.float32),
                tf.cast(new_h, tf.float32) / tf.cast(scaled_h, tf.float32)
            )
            boxes = boxes * tf.stack([scale_w * scale, scale_h * scale, scale_w * scale, scale_h * scale, 1.0])
            boxes = boxes + tf.stack([
                tf.cast(pad_left, tf.float32),
                tf.cast(pad_top, tf.float32),
                tf.cast(pad_left, tf.float32),
                tf.cast(pad_top, tf.float32),
                0.0
            ])
        else:
            # Letterbox resize with padding info to transform boxes correctly
            image_resized, (new_w, new_h), (pad_left, pad_top) = tf_letterbox_resize(
                image, generator.input_shape, return_padding_info=True
            )
            # Transform boxes: scale first, then add padding offset
            scale = tf.minimum(
                tf.cast(new_w, tf.float32) / src_w,
                tf.cast(new_h, tf.float32) / src_h
            )
            pad_left_f = tf.cast(pad_left, tf.float32)
            pad_top_f = tf.cast(pad_top, tf.float32)
            boxes = boxes * tf.stack([scale, scale, scale, scale, 1.0])
            boxes = boxes + tf.stack([pad_left_f, pad_top_f, pad_left_f, pad_top_f, 0.0])
        
        # Apply augmentations if enabled
        if generator.augment:
            image_resized, boxes, _, _ = tf_random_resize_crop_pad(
                image_resized, generator.input_shape, boxes,
                aspect_ratio_jitter=0.3, scale_jitter=0.5
            )
            image_resized, boxes = tf_random_horizontal_flip(image_resized, boxes)
            image_resized = tf_random_brightness(image_resized, max_delta=0.2)
            image_resized = tf_random_contrast(image_resized, lower=0.8, upper=1.2)
            image_resized = tf_random_saturation(image_resized, lower=0.8, upper=1.2)
            image_resized = tf_random_hue(image_resized, max_delta=0.1)
            image_resized = tf_random_grayscale(image_resized, probability=0.1)
            image_resized, boxes = tf_random_rotate(image_resized, boxes, rotate_range=20.0, prob=0.05)
            image_resized, boxes = tf_random_gridmask(image_resized, boxes, prob=0.1)
        
        return image_resized, boxes
    
    dataset = dataset.map(_preprocess_image_and_boxes, num_parallel_calls=num_parallel_calls)
    
    # Batch the dataset with fixed capacity padding
    padded_shapes = (
        [generator.input_shape[0], generator.input_shape[1], 3],
        [generator.max_boxes_per_image, 5]
    )
    padding_values = (0.0, 0.0)
    dataset = dataset.padded_batch(
        generator.batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=False
    )
    
    # Expand box capacity before batch augmentations
    if generator.augment:
        def _expand_box_capacity(images, boxes_dense):
            current_max_boxes = tf.shape(boxes_dense)[1]
            
            mosaic_enabled = (generator.enhance_augment == 'mosaic') and (generator.mosaic_prob > 0.0)
            mixup_enabled = generator.mixup_prob > 0.0
            
            if mosaic_enabled and mixup_enabled:
                expansion_factor = 8
            elif mosaic_enabled:
                expansion_factor = 4
            elif mixup_enabled:
                expansion_factor = 2
            else:
                expansion_factor = 1
            
            target_max_boxes = current_max_boxes * expansion_factor
            batch_size = tf.shape(boxes_dense)[0]
            padding_needed = target_max_boxes - current_max_boxes
            
            if padding_needed > 0:
                padding = tf.zeros([batch_size, padding_needed, 5], dtype=tf.float32)
                boxes_expanded = tf.concat([boxes_dense, padding], axis=1)
            else:
                boxes_expanded = boxes_dense
            
            return images, boxes_expanded
        
        dataset = dataset.map(_expand_box_capacity, num_parallel_calls=num_parallel_calls)
    
    # Apply batch-level augmentations (Mosaic, MixUp)
    if generator.augment:
        mosaic_prob = getattr(generator, 'mosaic_prob', 0.3)
        mixup_prob = getattr(generator, 'mixup_prob', 0.1)
        
        def _apply_batch_augmentations(images, boxes_dense):
            if generator.enhance_augment == 'mosaic' and mosaic_prob > 0.0:
                images, boxes_dense = tf_random_mosaic(images, boxes_dense, prob=mosaic_prob)
            if mixup_prob > 0.0:
                images, boxes_dense = tf_random_mixup(images, boxes_dense, prob=mixup_prob, alpha=0.2)
            return images, boxes_dense
        
        dataset = dataset.map(_apply_batch_augmentations, num_parallel_calls=num_parallel_calls)
    
    # Prefetch for performance
    dataset = dataset.prefetch(prefetch_buffer_size)
    
    return dataset


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
    valid_mask = (
        (boxes[:, 0] < boxes[:, 2]) &  # x1 < x2
        (boxes[:, 1] < boxes[:, 3]) &  # y1 < y2
        (boxes[:, 0] >= 0) &  # x1 >= 0
        (boxes[:, 1] >= 0) &  # y1 >= 0
        (boxes[:, 4] >= 0)  # class >= 0
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
    
    # Convert image from normalized float32 [0, 1] to uint8 [0, 255]
    # CRITICAL: Images from the dataset are normalized to [0, 1] range
    # We need to clip and convert properly to avoid black images
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Check if image is in [0, 1] range (normalized) or [0, 255] range
        if image.max() <= 1.0:
            # Normalized [0, 1] range - clip and convert
            image_clipped = np.clip(image, 0.0, 1.0)
            image_uint8 = (image_clipped * 255.0).astype(np.uint8)
        else:
            # Already in [0, 255] range (shouldn't happen, but handle it)
            image_clipped = np.clip(image, 0.0, 255.0)
            image_uint8 = image_clipped.astype(np.uint8)
    else:
        # Already integer type, but ensure it's in valid range
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    
    # Safety check: Verify image is not all black before proceeding
    if image_uint8.max() == 0:
        print(f"   ERROR: Image is completely black (all zeros)! This should not happen.")
        # Create a placeholder image to avoid saving black image
        image_uint8 = np.ones_like(image_uint8) * 128  # Gray placeholder
    
    # Ensure image is in BGR format for OpenCV (draw_boxes expects BGR)
    if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
        # Assume RGB, convert to BGR
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_uint8.copy()
    
    # Prepare data for draw_boxes
    if len(valid_boxes) > 0:
        boxes_list = valid_boxes[:, :4].tolist()  # [x1, y1, x2, y2]
        classes_list = valid_boxes[:, 4].astype(int).tolist()
        scores_list = [1.0] * len(valid_boxes)  # Dummy scores for visualization
    else:
        boxes_list = []
        classes_list = []
        scores_list = []
    
    # Draw boxes
    image_with_boxes = image_bgr.copy()
    if len(boxes_list) > 0:
        image_with_boxes = draw_boxes(
            image_with_boxes,
            boxes_list,
            classes_list,
            scores_list,
            class_names,
            colors,
            show_score=False
        )
    
    # Save visualization
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
        class_name = class_names[int(cls)] if int(cls) < len(class_names) else f"class_{int(cls)}"
        metadata["boxes"].append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "class_id": int(cls),
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
    
    # Build visualization dataset
    print("Building visualization dataset...")
    dataset = build_visualization_dataset(
        generator,
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
                
                # Debug: Check image statistics to catch black images early
                image_min = image.min()
                image_max = image.max()
                image_mean = image.mean()
                if image_max < 0.01 or image_mean < 0.001:
                    print(f"   WARNING: Sample {sample_idx + 1} appears to be black (min={image_min:.4f}, max={image_max:.4f}, mean={image_mean:.4f})")
                
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

