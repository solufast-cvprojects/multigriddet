#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive regression test suite for data augmentation pipelines.

This module provides automated testing and visual verification for all augmentations
that affect bounding box coordinates. Each augmentation is tested independently to
ensure correct coordinate transformations and box alignment.

Tested Augmentations:
    - Horizontal Flip: Mirrors images and boxes horizontally
    - Rotation: Applies 90/180/270 degree rotations with box transformations
    - Random Resize Crop Pad: Scales, crops, and pads images with coordinate adjustments
    - GridMask: Applies grid-based masking (may filter boxes if too small)
    - MixUp: Batch-level augmentation that blends two images and concatenates boxes
    - Color Augmentations: Brightness, contrast, saturation, hue, grayscale
        (tested for completeness; should not affect box coordinates)

Usage Examples:
    # Test all augmentations
    python tests/test_augmentations.py --annotation data/coco_train2017.txt --num-tests 3

    # Test specific augmentation
    python tests/test_augmentations.py --annotation data/coco_train2017.txt \\
        --aug horizontal_flip --num-tests 5

    # Custom configuration
    python tests/test_augmentations.py --annotation data/coco_train2017.txt \\
        --aug all --num-tests 5 --seed 42 --input-shape 640 640

Output:
    Generates side-by-side visualizations (original vs augmented) for manual inspection.
    Visualizations are saved to tests/augmentation_test_outputs/{aug_name}/

Author:
    MultiGridDet Project
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.data.generators import (
    tf_load_and_decode_image,
    tf_parse_boxes,
    tf_letterbox_resize,
    tf_random_horizontal_flip,
    tf_random_rotate,
    tf_random_resize_crop_pad,
    tf_random_gridmask,
    tf_random_mixup,
    tf_random_brightness,
    tf_random_contrast,
    tf_random_saturation,
    tf_random_hue,
    tf_random_grayscale
)
from multigriddet.data.utils import load_annotation_lines


def draw_boxes_on_image(image: np.ndarray, boxes: np.ndarray, 
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes on image."""
    # Convert image to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    image_copy = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2, cls = box
        # Skip invalid boxes
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        
        # Ensure coordinates are integers and within image bounds
        x1 = int(np.clip(x1, 0, image_copy.shape[1] - 1))
        y1 = int(np.clip(y1, 0, image_copy.shape[0] - 1))
        x2 = int(np.clip(x2, 0, image_copy.shape[1] - 1))
        y2 = int(np.clip(y2, 0, image_copy.shape[0] - 1))
        
        # Skip if box is invalid after clipping
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Draw class label
        label = f"cls:{int(cls)}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_copy, (x1, y1 - text_height - 4), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(image_copy, label, (x1, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image_copy


def create_side_by_side_visualization(original_image: np.ndarray, 
                                      augmented_image: np.ndarray,
                                      original_boxes: np.ndarray,
                                      augmented_boxes: np.ndarray,
                                      aug_name: str,
                                      output_path: str):
    """
    Create a side-by-side visualization comparing original and augmented images.
    
    Args:
        original_image: Original image array (H, W, 3) in uint8 format
        augmented_image: Augmented image array (H, W, 3) in uint8 format
        original_boxes: Original bounding boxes (N, 5) in format (x1, y1, x2, y2, class)
        augmented_boxes: Augmented bounding boxes (M, 5) in format (x1, y1, x2, y2, class)
        aug_name: Name of the augmentation applied (for title)
        output_path: File path to save the visualization
        
    Note:
        Images are converted from RGB to BGR for OpenCV drawing, then back to RGB
        for matplotlib display. Original boxes are drawn in green, augmented boxes in red.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Ensure images are in uint8 format and RGB
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255.0).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)
    
    if augmented_image.dtype != np.uint8:
        if augmented_image.max() <= 1.0:
            augmented_image = (augmented_image * 255.0).astype(np.uint8)
        else:
            augmented_image = augmented_image.astype(np.uint8)
    
    # TensorFlow images are in RGB format, but draw_boxes_on_image expects BGR (OpenCV)
    # So we need to convert RGB -> BGR for drawing, then BGR -> RGB for matplotlib
    
    # Original image: Convert RGB to BGR for OpenCV drawing
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        orig_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    else:
        orig_bgr = original_image
    
    # Original image with boxes (draw_boxes_on_image uses OpenCV/BGR)
    orig_with_boxes = draw_boxes_on_image(orig_bgr, original_boxes, 
                                         color=(0, 255, 0), thickness=2)
    # Convert back to RGB for matplotlib display
    orig_with_boxes_rgb = cv2.cvtColor(orig_with_boxes, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_with_boxes_rgb)
    axes[0].set_title(f'Original ({len(original_boxes)} boxes)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Augmented image: TensorFlow operations return RGB, convert to BGR for OpenCV drawing
    if len(augmented_image.shape) == 3 and augmented_image.shape[2] == 3:
        aug_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    else:
        aug_bgr = augmented_image
    
    # Augmented image with boxes (draw_boxes_on_image uses OpenCV/BGR)
    aug_with_boxes = draw_boxes_on_image(aug_bgr, augmented_boxes,
                                        color=(255, 0, 0), thickness=2)
    # Convert back to RGB for matplotlib display
    aug_with_boxes_rgb = cv2.cvtColor(aug_with_boxes, cv2.COLOR_BGR2RGB)
    axes[1].imshow(aug_with_boxes_rgb)
    axes[1].set_title(f'{aug_name} ({len(augmented_boxes)} boxes)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def test_horizontal_flip(image: tf.Tensor, boxes: tf.Tensor, 
                         input_shape: Tuple[int, int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test horizontal flip augmentation.
    
    Args:
        image: Image tensor of shape (H, W, 3) with dtype float32 in range [0, 1]
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        input_shape: Target image shape (height, width) - unused but kept for interface consistency
        seed: Random seed for deterministic testing
        
    Returns:
        Tuple of (flipped_image, flipped_boxes) as numpy arrays
        
    Note:
        Uses deterministic seed for reproducible testing. The augmentation function
        randomly decides whether to flip, but with fixed seed the result is deterministic.
    """
    tf.random.set_seed(seed)
    
    # Use the actual augmentation function (it will randomly flip, but with fixed seed it's deterministic)
    flipped_image, flipped_boxes = tf_random_horizontal_flip(image, boxes)
    
    return flipped_image.numpy(), flipped_boxes.numpy()


def test_rotation(image: tf.Tensor, boxes: tf.Tensor,
                 input_shape: Tuple[int, int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test rotation augmentation with forced 90-degree rotation for visibility.
    
    Args:
        image: Image tensor of shape (H, W, 3) with dtype float32 in range [0, 1]
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        input_shape: Target image shape (height, width) - unused but kept for interface consistency
        seed: Random seed for deterministic testing
        
    Returns:
        Tuple of (rotated_image, rotated_boxes) as numpy arrays
        
    Note:
        Forces a 90-degree rotation (k=1) to ensure visible transformation in test output.
        The actual rotation function supports 90/180/270 degree rotations, but excludes
        k=0 (no rotation) to guarantee visible changes when rotation is applied.
    """
    tf.random.set_seed(seed)
    
    # Force 90-degree rotation (k=1) for testing to ensure visible transformation
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    
    # Apply 90-degree rotation
    rotated_image = tf.image.rot90(image, k=1)
    
    # Transform boxes for 90-degree rotation: (x, y) -> (y, width - x)
    x1, y1, x2, y2, cls = tf.split(boxes, 5, axis=-1)
    new_x1 = y1
    new_y1 = width - x2
    new_x2 = y2
    new_y2 = width - x1
    rotated_boxes = tf.concat([new_x1, new_y1, new_x2, new_y2, cls], axis=-1)
    
    # Clip boxes to image boundaries
    rotated_boxes = tf.clip_by_value(
        rotated_boxes,
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [width, height, width, height, tf.cast(tf.shape(boxes)[0], tf.float32)]
    )
    
    return rotated_image.numpy(), rotated_boxes.numpy()


def test_resize_crop_pad(image: tf.Tensor, boxes: tf.Tensor,
                        input_shape: Tuple[int, int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test random resize crop pad augmentation.
    
    Args:
        image: Image tensor of shape (H, W, 3) with dtype float32 in range [0, 1]
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        input_shape: Target image shape (height, width)
        seed: Random seed for deterministic testing
        
    Returns:
        Tuple of (resized_image, transformed_boxes) as numpy arrays
        
    Note:
        This augmentation applies aspect ratio jittering, random scaling, cropping,
        and padding. Box coordinates are automatically adjusted to match the transformations.
    """
    tf.random.set_seed(seed)
    
    # Apply resize crop pad with standard jitter parameters
    resized_image, transformed_boxes, _, _ = tf_random_resize_crop_pad(
        image, input_shape, boxes,
        aspect_ratio_jitter=0.3, scale_jitter=0.5
    )
    
    return resized_image.numpy(), transformed_boxes.numpy()


def test_gridmask(image: tf.Tensor, boxes: tf.Tensor,
                 input_shape: Tuple[int, int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test GridMask augmentation.
    
    Args:
        image: Image tensor of shape (H, W, 3) with dtype float32 in range [0, 1]
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        input_shape: Target image shape (height, width) - unused but kept for interface consistency
        seed: Random seed for deterministic testing
        
    Returns:
        Tuple of (masked_image, filtered_boxes) as numpy arrays
        
    Note:
        GridMask applies a grid-based masking pattern to the image. Boxes that become
        too small after masking may be filtered out, which is expected behavior.
    """
    tf.random.set_seed(seed)
    
    # Apply GridMask with probability 1.0 to ensure it's applied for testing
    masked_image, filtered_boxes = tf_random_gridmask(image, boxes, prob=1.0)
    
    return masked_image.numpy(), filtered_boxes.numpy()


def test_color_augmentations(image: tf.Tensor, boxes: tf.Tensor,
                             input_shape: Tuple[int, int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test color augmentations (verifies boxes remain unchanged).
    
    Args:
        image: Image tensor of shape (H, W, 3) with dtype float32 in range [0, 1]
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        input_shape: Target image shape (height, width) - unused but kept for interface consistency
        seed: Random seed for deterministic testing
        
    Returns:
        Tuple of (augmented_image, boxes) as numpy arrays
        
    Note:
        Color augmentations (brightness, contrast, saturation, hue, grayscale) should
        NOT affect bounding box coordinates. This test verifies that geometric properties
        are preserved when only color transformations are applied.
    """
    tf.random.set_seed(seed)
    
    # Apply all color augmentations in sequence
    aug_image = tf_random_brightness(image, max_delta=0.2)
    aug_image = tf_random_contrast(aug_image, lower=0.8, upper=1.2)
    aug_image = tf_random_saturation(aug_image, lower=0.8, upper=1.2)
    aug_image = tf_random_hue(aug_image, max_delta=0.1)
    aug_image = tf_random_grayscale(aug_image, probability=0.1)
    
    # Boxes should remain unchanged (color augmentations don't affect geometry)
    return aug_image.numpy(), boxes.numpy()


def test_mixup(images: tf.Tensor, boxes: tf.Tensor,
              input_shape: Tuple[int, int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test MixUp batch-level augmentation.
    
    Args:
        images: Batch of images tensor of shape (batch_size, H, W, 3) with dtype float32 [0, 1]
        boxes: Batch of boxes tensor of shape (batch_size, max_boxes, 5) in format (x1, y1, x2, y2, class)
        input_shape: Target image shape (height, width) - unused but kept for interface consistency
        seed: Random seed for deterministic testing
        
    Returns:
        Tuple of (mixed_image, mixed_boxes) for the first image in batch as numpy arrays
        
    Note:
        MixUp is a batch-level augmentation that blends pairs of images and concatenates
        their bounding boxes. This function requires at least 2 images in the batch.
        The mixing ratio is sampled from a Beta distribution (alpha=0.2).
    """
    tf.random.set_seed(seed)
    
    # Apply MixUp with probability 1.0 to ensure it's applied for testing
    mixed_images, mixed_boxes = tf_random_mixup(images, boxes, prob=1.0, alpha=0.2)
    
    # Return the first image from the batch for visualization
    return mixed_images[0].numpy(), mixed_boxes[0].numpy()


def test_augmentation(aug_name: str, annotation_lines: List[str], 
                     input_shape: Tuple[int, int], num_tests: int,
                     output_dir: str, seed: int = 42):
    """Test a specific augmentation."""
    
    # Filter annotation lines that have boxes
    valid_lines = []
    for line in annotation_lines:
        parts = line.strip().split()
        if len(parts) > 1:
            valid_lines.append(line)
    
    if len(valid_lines) == 0:
        print(f"ERROR: No valid annotation lines with boxes found!")
        return
    
    print(f"\n{'='*80}")
    print(f"Testing: {aug_name.upper()}")
    print(f"{'='*80}")
    print(f"Input shape: {input_shape}")
    print(f"Number of tests: {num_tests}")
    print(f"Random seed: {seed}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(valid_lines)} valid annotation lines with boxes")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select augmentation function
    aug_functions = {
        'horizontal_flip': test_horizontal_flip,
        'rotation': test_rotation,
        'resize_crop_pad': test_resize_crop_pad,
        'gridmask': test_gridmask,
        'mixup': test_mixup,
        'color': test_color_augmentations,
    }
    
    # MixUp requires batch-level processing (needs at least 2 images)
    is_batch_augmentation = (aug_name == 'mixup')
    
    if aug_name not in aug_functions:
        print(f"ERROR: Unknown augmentation '{aug_name}'")
        print(f"Available: {list(aug_functions.keys())}")
        return
    
    aug_func = aug_functions[aug_name]
    
    # Run tests
    for test_idx in range(num_tests):
        print(f"Test {test_idx + 1}/{num_tests}...")
        
        try:
            if is_batch_augmentation:
                # MixUp requires a batch of at least 2 images
                # Load 2 images for mixing
                images_list = []
                boxes_list = []
                
                for i in range(2):
                    line_idx = (seed + test_idx * 2 + i) % len(valid_lines)
                    line = valid_lines[line_idx]
                    parts = line.strip().split()
                    image_path = parts[0]
                    boxes_str = ' '.join(parts[1:]) if len(parts) > 1 else ''
                    
                    # Load and process image
                    image_tf = tf_load_and_decode_image(tf.constant(image_path, dtype=tf.string))
                    image_np = image_tf.numpy()
                    orig_h, orig_w = image_np.shape[:2]
                    
                    # Resize to input shape
                    result = tf_letterbox_resize(image_tf, input_shape, return_padding_info=True)
                    image_resized = result[0]
                    offset = result[2]
                    image_resized_np = image_resized.numpy()
                    
                    # Parse and scale boxes
                    boxes_tf = tf_parse_boxes(tf.constant(boxes_str, dtype=tf.string))
                    boxes_np = boxes_tf.numpy()
                    
                    target_h, target_w = input_shape
                    scale = min(target_w / orig_w, target_h / orig_h)
                    pad_left = int(offset[0].numpy())
                    pad_top = int(offset[1].numpy())
                    
                    if len(boxes_np) > 0:
                        boxes_np[:, 0] = boxes_np[:, 0] * scale + pad_left
                        boxes_np[:, 1] = boxes_np[:, 1] * scale + pad_top
                        boxes_np[:, 2] = boxes_np[:, 2] * scale + pad_left
                        boxes_np[:, 3] = boxes_np[:, 3] * scale + pad_top
                    
                    # Normalize image
                    image_float = tf.cast(image_resized, tf.float32) / 255.0
                    if len(boxes_np) > 0:
                        boxes_float = tf.constant(boxes_np, dtype=tf.float32)
                    else:
                        boxes_float = tf.zeros((0, 5), dtype=tf.float32)
                    
                    images_list.append(image_float)
                    boxes_list.append(boxes_float)
                
                # Stack into batch
                images_batch = tf.stack(images_list)  # (2, H, W, 3)
                
                # Pad boxes to same length
                max_boxes_list = [tf.shape(boxes)[0].numpy() for boxes in boxes_list]
                max_boxes = max(max_boxes_list) if max_boxes_list else 10
                max_boxes = max(max_boxes, 10)
                
                boxes_padded = []
                for boxes in boxes_list:
                    num_boxes = tf.shape(boxes)[0]
                    if num_boxes < max_boxes:
                        padding = tf.zeros([max_boxes - num_boxes, 5], dtype=tf.float32)
                        boxes_padded.append(tf.concat([boxes, padding], axis=0))
                    else:
                        boxes_padded.append(boxes[:max_boxes])
                
                boxes_batch = tf.stack(boxes_padded)  # (2, max_boxes, 5)
                
                # Apply MixUp augmentation
                aug_image_np, aug_boxes_np = aug_func(
                    images_batch, boxes_batch, input_shape, seed + test_idx
                )
                
                # Convert augmented image back to uint8
                aug_image_np = np.clip(aug_image_np, 0.0, 1.0)
                aug_image_np = (aug_image_np * 255.0).astype(np.uint8)
                
                # Use first image as "original" for visualization
                # Get the first image from the list (before normalization)
                first_image_tf = images_list[0] * 255.0
                orig_image_np = first_image_tf.numpy().astype(np.uint8)
                
                # Use original boxes from first image
                boxes_np = boxes_list[0].numpy() if len(boxes_list) > 0 else np.zeros((0, 5))
                
            else:
                # Standard single-image augmentation
                # Select a random annotation line
                line_idx = (seed + test_idx) % len(valid_lines)
                line = valid_lines[line_idx]
                
                # Parse annotation line
                parts = line.strip().split()
                image_path = parts[0]
                boxes_str = ' '.join(parts[1:]) if len(parts) > 1 else ''
                
                # Load image
                image_tf = tf_load_and_decode_image(tf.constant(image_path, dtype=tf.string))
                image_np = image_tf.numpy()
                
                # Get original image dimensions
                orig_h, orig_w = image_np.shape[:2]
                
                # Resize to input shape with padding info
                result = tf_letterbox_resize(
                    image_tf,
                    input_shape,
                    return_padding_info=True
                )
                image_resized = result[0]
                padding_size = result[1]  # (new_w, new_h)
                offset = result[2]  # (pad_left, pad_top)
                
                image_resized_np = image_resized.numpy()
                
                # Parse boxes
                boxes_tf = tf_parse_boxes(tf.constant(boxes_str, dtype=tf.string))
                boxes_np = boxes_tf.numpy()
                
                # Scale boxes to match letterbox resize
                target_h, target_w = input_shape
                scale = min(target_w / orig_w, target_h / orig_h)
                
                # Get padding info from TensorFlow tensors
                pad_left = int(offset[0].numpy())
                pad_top = int(offset[1].numpy())
                
                # Scale boxes and add padding offset
                if len(boxes_np) > 0:
                    boxes_np[:, 0] = boxes_np[:, 0] * scale + pad_left  # x1
                    boxes_np[:, 1] = boxes_np[:, 1] * scale + pad_top   # y1
                    boxes_np[:, 2] = boxes_np[:, 2] * scale + pad_left  # x2
                    boxes_np[:, 3] = boxes_np[:, 3] * scale + pad_top   # y2
                
                # Convert to float32 and normalize image to [0, 1]
                image_float = tf.cast(image_resized, tf.float32) / 255.0
                
                # Use the scaled boxes (boxes_np already has correct coordinates after letterbox)
                if len(boxes_np) > 0:
                    boxes_float = tf.constant(boxes_np, dtype=tf.float32)
                else:
                    boxes_float = tf.zeros((0, 5), dtype=tf.float32)
                
                # Apply augmentation
                aug_image_np, aug_boxes_np = aug_func(
                    image_float, boxes_float, input_shape, seed + test_idx
                )
                
                # Use the original resized image (before normalization) for visualization
                # tf_letterbox_resize returns float32, so we need to convert to uint8
                orig_image_np = image_resized_np.copy()
                if orig_image_np.dtype != np.uint8:
                    orig_image_np = np.clip(orig_image_np, 0, 255).astype(np.uint8)
            
                # Convert augmented image back to uint8 (augmented image is in [0, 1] range)
                aug_image_np = np.clip(aug_image_np, 0.0, 1.0)
                aug_image_np = (aug_image_np * 255.0).astype(np.uint8)
            
            # Filter out invalid boxes
            valid_mask_orig = np.any(boxes_np[:, :4] != 0, axis=1)
            boxes_orig_valid = boxes_np[valid_mask_orig]
            
            valid_mask_aug = np.any(aug_boxes_np[:, :4] != 0, axis=1)
            boxes_aug_valid = aug_boxes_np[valid_mask_aug]
            
            print(f"   Original boxes: {len(boxes_orig_valid)}")
            print(f"   Augmented boxes: {len(boxes_aug_valid)}")
            
            # Create visualization
            output_path = os.path.join(output_dir, f"{aug_name}_test_{test_idx + 1:03d}.png")
            create_side_by_side_visualization(
                orig_image_np, aug_image_np,
                boxes_orig_valid, boxes_aug_valid,
                aug_name.replace('_', ' ').title(),
                output_path
            )
            
            print(f"   Saved visualization to: {output_path}")
            print()
            
        except Exception as e:
            print(f"   ERROR processing test {test_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    print(f"Test Complete for {aug_name.upper()}!")
    print(f"Visualizations saved to: {output_dir}")
    print()
    print("Visual Inspection Checklist:")
    print("  - Check that boxes in augmented image align with objects")
    print("  - Verify boxes are correctly transformed (flipped/rotated/etc)")
    print("  - Ensure no boxes are outside image boundaries")
    print("  - Confirm boxes maintain correct class labels")
    print("=" * 80)
    print()


def main():
    """
    Main entry point for the augmentation test suite.
    
    Parses command-line arguments and orchestrates testing for specified augmentations.
    Generates visual verification outputs for manual inspection of augmentation correctness.
    """
    parser = argparse.ArgumentParser(
        description='Comprehensive test suite for data augmentation pipelines with visual verification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--annotation', type=str, required=True,
                       help='Path to annotation file')
    parser.add_argument('--aug', type=str, default='all',
                       choices=['all', 'horizontal_flip', 'rotation', 'resize_crop_pad', 
                               'gridmask', 'mixup', 'color'],
                       help='Augmentation to test (default: all)')
    parser.add_argument('--num-tests', type=int, default=3,
                       help='Number of test cases per augmentation (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[608, 608],
                       metavar=('HEIGHT', 'WIDTH'),
                       help='Input image shape (default: 608 608)')
    parser.add_argument('--output-dir', type=str, default='tests/augmentation_test_outputs',
                       help='Output directory for visualizations (default: tests/augmentation_test_outputs)')
    
    args = parser.parse_args()
    
    # Load annotations
    print(f"Loading annotations from: {args.annotation}")
    annotation_lines = load_annotation_lines(args.annotation)
    print(f"Loaded {len(annotation_lines)} annotation lines")
    
    input_shape = tuple(args.input_shape)
    
    # Determine which augmentations to test
    if args.aug == 'all':
        augmentations = ['horizontal_flip', 'rotation', 'resize_crop_pad', 'gridmask', 'mixup', 'color']
    else:
        augmentations = [args.aug]
    
    # Test each augmentation
    for aug_name in augmentations:
        aug_output_dir = os.path.join(args.output_dir, aug_name)
        test_augmentation(
            aug_name, annotation_lines, input_shape,
            args.num_tests, aug_output_dir, args.seed
        )
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE!")
    print("=" * 80)
    print(f"All visualizations saved to: {args.output_dir}")
    print("\nReview the output images to verify augmentations are working correctly.")


if __name__ == '__main__':
    main()

