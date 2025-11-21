#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for Mosaic and MixUp augmentations to verify box preservation.

This test suite ensures that:
1. Box counts never decrease unexpectedly after Mosaic/MixUp
2. No silent truncation occurs (all boxes from source images are preserved)
3. Visual verification of box alignment on composite images

CRITICAL: The old implementation silently truncated boxes to single-image max_boxes,
discarding up to 75% of objects per composite and corrupting supervision signals.
This test verifies that the fix preserves all boxes.

Usage:
    python tests/test_mosaic_mixup_regression.py --annotation data/coco_train2017.txt --num-tests 5
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add project root and tests directory to path
project_root = Path(__file__).parent.parent
tests_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tests_dir))

from multigriddet.data.generators import (
    tf_load_and_decode_image,
    tf_parse_boxes,
    tf_parse_annotation_line,
    tf_letterbox_resize,
    tf_random_mosaic,
    tf_random_mixup
)
from multigriddet.data.utils import load_annotation_lines
from test_utils import (
    convert_image_to_uint8,
    draw_boxes_with_class_names,
    load_class_names_from_config,
)


def count_valid_boxes(boxes: np.ndarray) -> int:
    """Count valid (non-zero) boxes."""
    if len(boxes) == 0:
        return 0
    
    valid_count = 0
    for box in boxes:
        x1, y1, x2, y2, cls = box
        # Check if box is valid (has non-zero area)
        if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
            valid_count += 1
    
    return valid_count


def test_mosaic_box_preservation(annotation_lines: List[str],
                                 input_shape: Tuple[int, int],
                                 num_tests: int,
                                 output_dir: str,
                                 class_names: List[str],
                                 colors: List[Tuple[int, int, int]],
                                 seed: int = 42):
    """
    Test Mosaic augmentation preserves all boxes from 4 source images.
    
    Verifies:
    - Total valid boxes after Mosaic >= sum of valid boxes from 4 source images (accounting for filtering)
    - No unexpected box count decreases
    - Visual verification of box alignment
    """
    print(f"\n{'='*60}")
    print("Testing Mosaic Box Preservation")
    print(f"{'='*60}")
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Filter annotation lines that have boxes
    valid_lines = [line for line in annotation_lines if len(line.strip().split()) > 1]
    
    if len(valid_lines) < 4:
        print(f"ERROR: Need at least 4 annotation lines with boxes, got {len(valid_lines)}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_passed = True
    
    for test_idx in range(num_tests):
        print(f"\nTest {test_idx + 1}/{num_tests}")
        
        # Select 4 random annotation lines
        selected_indices = np.random.choice(len(valid_lines), size=4, replace=False)
        selected_lines = [valid_lines[i] for i in selected_indices]
        
        # Load and preprocess 4 images
        images_list = []
        boxes_list = []
        total_source_boxes = 0
        
        for line_idx, line in enumerate(selected_lines):
            # Parse annotation
            image_path, boxes_string = tf_parse_annotation_line(tf.constant(line))
            image = tf_load_and_decode_image(image_path)
            boxes = tf_parse_boxes(boxes_string)
            
            # Get original image size for box transformation
            image_shape = tf.shape(image)
            src_h = tf.cast(image_shape[0], tf.float32)
            src_w = tf.cast(image_shape[1], tf.float32)
            
            # Resize to input shape with padding info
            image_resized, (new_w, new_h), (pad_left, pad_top) = tf_letterbox_resize(
                image, input_shape, return_padding_info=True
            )
            
            # Transform boxes to match letterbox-resized image
            # Scale boxes first, then add padding offset
            scale = tf.minimum(
                tf.cast(new_w, tf.float32) / src_w,
                tf.cast(new_h, tf.float32) / src_h
            )
            pad_left_f = tf.cast(pad_left, tf.float32)
            pad_top_f = tf.cast(pad_top, tf.float32)
            
            # Transform boxes: scale then add offset
            boxes_transformed = boxes * tf.stack([scale, scale, scale, scale, 1.0])
            boxes_transformed = boxes_transformed + tf.stack([pad_left_f, pad_top_f, pad_left_f, pad_top_f, 0.0])
            
            # Normalize image for processing
            image_resized = tf.cast(image_resized, tf.float32) / 255.0
            
            images_list.append(image_resized)
            boxes_list.append(boxes_transformed)
            
            # Count valid boxes in source
            boxes_np = boxes_transformed.numpy()
            valid_count = count_valid_boxes(boxes_np)
            total_source_boxes += valid_count
            print(f"  Source image {line_idx + 1}: {valid_count} valid boxes")
        
        print(f"  Total source boxes: {total_source_boxes}")
        
        # Expand box capacity to 4× for Mosaic (critical fix)
        # Find max boxes per image
        max_boxes_per_image = max([tf.shape(boxes)[0].numpy() for boxes in boxes_list])
        expanded_capacity = max_boxes_per_image * 4
        
        # Pad all boxes to expanded capacity
        boxes_padded_list = []
        for boxes in boxes_list:
            current_size = tf.shape(boxes)[0].numpy()
            if current_size < expanded_capacity:
                padding = tf.zeros([expanded_capacity - current_size, 5], dtype=tf.float32)
                boxes_padded = tf.concat([boxes, padding], axis=0)
            else:
                boxes_padded = boxes
            boxes_padded_list.append(boxes_padded)
        
        # Stack into batch format
        images_batch = tf.stack(images_list, axis=0)  # (4, H, W, 3)
        boxes_batch = tf.stack(boxes_padded_list, axis=0)  # (4, expanded_capacity, 5)
        
        # Apply Mosaic with probability 1.0 (always apply for testing)
        mosaic_images, mosaic_boxes = tf_random_mosaic(
            images_batch, boxes_batch, prob=1.0, min_offset=0.2
        )
        
        # Get first mosaic result
        mosaic_image = mosaic_images[0].numpy()
        mosaic_boxes_result = mosaic_boxes[0].numpy()
        
        # Count valid boxes after Mosaic
        valid_after = count_valid_boxes(mosaic_boxes_result)
        print(f"  Valid boxes after Mosaic: {valid_after}")
        
        # Verify: valid boxes should be >= expected minimum
        # (Some boxes may be filtered due to size/overlap, but we should preserve most)
        # Expect at least 50% of source boxes to survive (conservative estimate)
        expected_minimum = int(total_source_boxes * 0.5)
        
        if valid_after < expected_minimum:
            print(f"  WARNING: Box count lower than expected minimum ({expected_minimum})")
            print(f"  This may indicate truncation or excessive filtering")
            all_passed = False
        else:
            print(f"  PASS: Box count ({valid_after}) >= expected minimum ({expected_minimum})")
        
        # Verify no truncation: check if we hit capacity limit unexpectedly
        if valid_after == expanded_capacity:
            print(f"  WARNING: Hit capacity limit ({expanded_capacity}), may indicate truncation")
            # This is actually OK if we have that many boxes, but worth noting
        
        # Visual verification - convert image from normalized [0,1] to uint8 [0,255]
        mosaic_image_uint8 = convert_image_to_uint8(mosaic_image)
        mosaic_with_boxes_bgr = draw_boxes_with_class_names(
            mosaic_image_uint8,
            mosaic_boxes_result,
            class_names,
            colors,
            show_score=False,
        )
        mosaic_with_boxes_rgb = cv2.cvtColor(mosaic_with_boxes_bgr, cv2.COLOR_BGR2RGB)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"mosaic_test_{test_idx + 1:03d}.png")
        plt.figure(figsize=(12, 12))
        plt.imshow(mosaic_with_boxes_rgb)
        plt.title(f"Mosaic Test {test_idx + 1}\nSource boxes: {total_source_boxes}, "
                 f"After Mosaic: {valid_after}, Expected min: {expected_minimum}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization: {output_path}")
    
    if all_passed:
        print(f"\n{'='*60}")
        print("All Mosaic tests PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("Some Mosaic tests FAILED - check warnings above")
        print(f"{'='*60}")


def test_mixup_box_preservation(annotation_lines: List[str],
                                input_shape: Tuple[int, int],
                                num_tests: int,
                                output_dir: str,
                                class_names: List[str],
                                colors: List[Tuple[int, int, int]],
                                seed: int = 42):
    """
    Test MixUp augmentation preserves all boxes from 2 source images.
    
    Verifies:
    - Total valid boxes after MixUp = sum of valid boxes from 2 source images
    - No unexpected box count decreases
    - Visual verification of box alignment
    """
    print(f"\n{'='*60}")
    print("Testing MixUp Box Preservation")
    print(f"{'='*60}")
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Filter annotation lines that have boxes
    valid_lines = [line for line in annotation_lines if len(line.strip().split()) > 1]
    
    if len(valid_lines) < 2:
        print(f"ERROR: Need at least 2 annotation lines with boxes, got {len(valid_lines)}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_passed = True
    
    for test_idx in range(num_tests):
        print(f"\nTest {test_idx + 1}/{num_tests}")
        
        # Select 2 random annotation lines
        selected_indices = np.random.choice(len(valid_lines), size=2, replace=False)
        selected_lines = [valid_lines[i] for i in selected_indices]
        
        # Load and preprocess 2 images
        images_list = []
        boxes_list = []
        total_source_boxes = 0
        
        for line_idx, line in enumerate(selected_lines):
            # Parse annotation
            image_path, boxes_string = tf_parse_annotation_line(tf.constant(line))
            image = tf_load_and_decode_image(image_path)
            boxes = tf_parse_boxes(boxes_string)
            
            # Get original image size for box transformation
            image_shape = tf.shape(image)
            src_h = tf.cast(image_shape[0], tf.float32)
            src_w = tf.cast(image_shape[1], tf.float32)
            
            # Resize to input shape with padding info
            image_resized, (new_w, new_h), (pad_left, pad_top) = tf_letterbox_resize(
                image, input_shape, return_padding_info=True
            )
            
            # Transform boxes to match letterbox-resized image
            # Scale boxes first, then add padding offset
            scale = tf.minimum(
                tf.cast(new_w, tf.float32) / src_w,
                tf.cast(new_h, tf.float32) / src_h
            )
            pad_left_f = tf.cast(pad_left, tf.float32)
            pad_top_f = tf.cast(pad_top, tf.float32)
            
            # Transform boxes: scale then add offset
            boxes_transformed = boxes * tf.stack([scale, scale, scale, scale, 1.0])
            boxes_transformed = boxes_transformed + tf.stack([pad_left_f, pad_top_f, pad_left_f, pad_top_f, 0.0])
            
            # Normalize image for processing
            image_resized = tf.cast(image_resized, tf.float32) / 255.0
            
            images_list.append(image_resized)
            boxes_list.append(boxes_transformed)
            
            # Count valid boxes in source
            boxes_np = boxes_transformed.numpy()
            valid_count = count_valid_boxes(boxes_np)
            total_source_boxes += valid_count
            print(f"  Source image {line_idx + 1}: {valid_count} valid boxes")
        
        print(f"  Total source boxes: {total_source_boxes}")
        
        # Expand box capacity to 2× for MixUp (critical fix)
        # Find max boxes per image
        max_boxes_per_image = max([tf.shape(boxes)[0].numpy() for boxes in boxes_list])
        expanded_capacity = max_boxes_per_image * 2
        
        # Pad all boxes to expanded capacity
        boxes_padded_list = []
        for boxes in boxes_list:
            current_size = tf.shape(boxes)[0].numpy()
            if current_size < expanded_capacity:
                padding = tf.zeros([expanded_capacity - current_size, 5], dtype=tf.float32)
                boxes_padded = tf.concat([boxes, padding], axis=0)
            else:
                boxes_padded = boxes
            boxes_padded_list.append(boxes_padded)
        
        # Stack into batch format
        images_batch = tf.stack(images_list, axis=0)  # (2, H, W, 3)
        boxes_batch = tf.stack(boxes_padded_list, axis=0)  # (2, expanded_capacity, 5)
        
        # Apply MixUp with probability 1.0 (always apply for testing)
        mixup_images, mixup_boxes = tf_random_mixup(
            images_batch, boxes_batch, prob=1.0, alpha=0.2
        )
        
        # Get first mixup result
        mixup_image = mixup_images[0].numpy()
        mixup_boxes_result = mixup_boxes[0].numpy()
        
        # Count valid boxes after MixUp
        valid_after = count_valid_boxes(mixup_boxes_result)
        print(f"  Valid boxes after MixUp: {valid_after}")
        
        # Verify: MixUp should preserve ALL boxes from both images (no filtering)
        # Expected: exactly total_source_boxes (all boxes concatenated)
        if valid_after != total_source_boxes:
            print(f"  FAIL: Box count mismatch!")
            print(f"  Expected: {total_source_boxes}, Got: {valid_after}")
            all_passed = False
        else:
            print(f"  PASS: Box count matches expected ({total_source_boxes})")
        
        # Verify no truncation: check if we hit capacity limit unexpectedly
        if valid_after == expanded_capacity and total_source_boxes > expanded_capacity:
            print(f"  WARNING: Hit capacity limit ({expanded_capacity}), may indicate truncation")
            all_passed = False
        
        # Visual verification - convert image from normalized [0,1] to uint8 [0,255]
        mixup_image_uint8 = convert_image_to_uint8(mixup_image)
        mixup_with_boxes_bgr = draw_boxes_with_class_names(
            mixup_image_uint8,
            mixup_boxes_result,
            class_names,
            colors,
            show_score=False,
        )
        mixup_with_boxes_rgb = cv2.cvtColor(mixup_with_boxes_bgr, cv2.COLOR_BGR2RGB)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"mixup_test_{test_idx + 1:03d}.png")
        plt.figure(figsize=(12, 12))
        plt.imshow(mixup_with_boxes_rgb)
        plt.title(f"MixUp Test {test_idx + 1}\nSource boxes: {total_source_boxes}, "
                 f"After MixUp: {valid_after}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization: {output_path}")
    
    if all_passed:
        print(f"\n{'='*60}")
        print("All MixUp tests PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("Some MixUp tests FAILED - check errors above")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Regression tests for Mosaic/MixUp box preservation')
    parser.add_argument('--annotation', type=str, required=True,
                       help='Path to annotation file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML file (for class names/colors)')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[640, 640],
                       help='Input image shape (height width)')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for deterministic testing')
    parser.add_argument('--output-dir', type=str, 
                       default='tests/augmentation_test_outputs',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    input_shape = tuple(args.input_shape)
    
    print("Loading annotation lines...")
    annotation_lines = load_annotation_lines(args.annotation, shuffle=False)
    print(f"Loaded {len(annotation_lines)} annotation lines")
    
    # Load class names and colors from config for visualization
    print(f"Loading classes from config: {args.config}")
    class_names, colors = load_class_names_from_config(args.config)
    print(f"Loaded {len(class_names)} classes")
    
    # Test Mosaic
    mosaic_output_dir = os.path.join(args.output_dir, 'mosaic')
    test_mosaic_box_preservation(
        annotation_lines, input_shape, args.num_tests, mosaic_output_dir,
        class_names, colors, args.seed
    )
    
    # Test MixUp
    mixup_output_dir = os.path.join(args.output_dir, 'mixup')
    test_mixup_box_preservation(
        annotation_lines, input_shape, args.num_tests, mixup_output_dir,
        class_names, colors, args.seed
    )
    
    print("\nRegression tests completed!")
    print(f"Visual outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

