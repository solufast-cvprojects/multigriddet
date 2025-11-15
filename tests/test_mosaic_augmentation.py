#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test suite for Mosaic augmentation with visual verification.

This module provides automated testing and visual validation for the Mosaic augmentation
pipeline. Mosaic augmentation combines four training images into a single composite image,
effectively increasing the effective batch size and introducing diverse contextual information.

The test verifies:
    1. Box coordinates are correctly transformed when quadrants are pasted
    2. Boxes align properly with objects in the combined mosaic image
    3. No boxes are lost or incorrectly positioned during the combination process
    4. Coordinate transformations maintain geometric consistency

Usage Examples:
    # Basic usage with default settings
    python tests/test_mosaic_augmentation.py --annotation data/coco_train2017.txt --num-tests 5

    # With custom seed for reproducibility
    python tests/test_mosaic_augmentation.py --annotation data/coco_train2017.txt \\
        --num-tests 5 --seed 42

    # Custom input shape and output directory
    python tests/test_mosaic_augmentation.py --annotation data/coco_train2017.txt \\
        --num-tests 10 --input-shape 640 640 --output-dir tests/my_mosaic_tests

Output:
    Generates grid visualizations showing all 4 original images plus the resulting mosaic.
    Visualizations are saved to tests/mosaic_test_outputs/ by default.

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
import matplotlib.patches as patches
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.data.generators import (
    tf_load_and_decode_image,
    tf_parse_annotation_line,
    tf_parse_boxes,
    tf_letterbox_resize,
    tf_random_mosaic
)
from multigriddet.data.utils import load_annotation_lines


def draw_boxes_on_image(image: np.ndarray, boxes: np.ndarray, 
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Image array (H, W, 3) in range [0, 1] or [0, 255]
        boxes: Boxes array (N, 5) in format (x1, y1, x2, y2, class)
        color: RGB color for boxes
        thickness: Line thickness
        
    Returns:
        Image with boxes drawn
    """
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


def create_visualization_grid(images: List[np.ndarray], titles: List[str],
                             output_path: str, figsize: Tuple[int, int] = (20, 5)):
    """
    Create a grid visualization of multiple images.
    
    Args:
        images: List of image arrays
        titles: List of titles for each image
        output_path: Path to save the visualization
        figsize: Figure size (width, height)
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    if num_images == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype == np.uint8 else img
        else:
            img_rgb = img
        
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_path}")
    plt.close()


def test_mosaic_augmentation(annotation_lines: List[str], 
                            input_shape: Tuple[int, int] = (608, 608),
                            num_tests: int = 5,
                            seed: int = 42,
                            output_dir: str = "tests/mosaic_test_outputs"):
    """
    Test Mosaic augmentation with visual verification.
    
    Args:
        annotation_lines: List of annotation lines
        input_shape: Input image shape (height, width)
        num_tests: Number of test cases to run
        seed: Random seed for reproducibility
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print("=" * 80)
    print("Mosaic Augmentation Regression Test")
    print("=" * 80)
    print(f"Input shape: {input_shape}")
    print(f"Number of tests: {num_tests}")
    print(f"Random seed: {seed}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Filter annotation lines that have boxes
    valid_lines = []
    for line in annotation_lines:
        parts = line.strip().split()
        if len(parts) > 1:  # Has boxes
            valid_lines.append(line)
    
    if len(valid_lines) < 4:
        print(f"[ERROR] Need at least 4 annotation lines with boxes, found {len(valid_lines)}")
        return
    
    print(f"Found {len(valid_lines)} valid annotation lines with boxes")
    print()
    
    # Run tests
    for test_idx in range(num_tests):
        print(f"Test {test_idx + 1}/{num_tests}...")
        
        # Select 4 random annotation lines
        selected_indices = np.random.choice(len(valid_lines), size=4, replace=False)
        selected_lines = [valid_lines[i] for i in selected_indices]
        
        # Load and preprocess images and boxes
        images_list = []
        boxes_list = []
        image_paths = []
        
        for line in selected_lines:
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
            # Letterbox resize: scale image, then pad to target size
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
            
            images_list.append(image_resized_np)
            boxes_list.append(boxes_np)
            image_paths.append(image_path)
        
        # Prepare batch for Mosaic augmentation
        # Stack images: (4, H, W, 3)
        images_batch = tf.stack([tf.constant(img, dtype=tf.float32) for img in images_list])
        
        # Pad boxes to same length and stack: (4, max_boxes, 5)
        max_boxes = max(len(boxes) for boxes in boxes_list) if boxes_list else 1
        max_boxes = max(max_boxes, 10)  # Ensure minimum size
        
        boxes_padded = []
        for boxes in boxes_list:
            if len(boxes) == 0:
                padded = np.zeros((max_boxes, 5), dtype=np.float32)
            else:
                padded = np.zeros((max_boxes, 5), dtype=np.float32)
                padded[:len(boxes)] = boxes
            boxes_padded.append(padded)
        
        boxes_batch = tf.stack([tf.constant(boxes, dtype=tf.float32) for boxes in boxes_padded])
        
        # Apply Mosaic augmentation
        # Note: tf_random_mosaic expects images in [0, 1] range
        images_normalized = images_batch / 255.0
        
        # Set seed for this specific augmentation
        tf.random.set_seed(seed + test_idx)
        
        mosaic_image, mosaic_boxes = tf_random_mosaic(
            images_normalized,
            boxes_batch,
            prob=1.0,  # Always apply
            min_offset=0.2
        )
        
        # Convert back to numpy
        mosaic_image_np = (mosaic_image[0].numpy() * 255.0).astype(np.uint8)
        mosaic_boxes_np = mosaic_boxes[0].numpy()
        
        # Filter out invalid boxes (all zeros)
        valid_mask = np.any(mosaic_boxes_np[:, :4] != 0, axis=1)
        mosaic_boxes_valid = mosaic_boxes_np[valid_mask]
        
        print(f"   Original boxes: {[len(boxes) for boxes in boxes_list]}")
        print(f"   Mosaic boxes: {len(mosaic_boxes_valid)}")
        
        # Debug: Print sample box coordinates to verify transformation
        if len(mosaic_boxes_valid) > 0:
            sample_box = mosaic_boxes_valid[0]
            print(f"   Sample mosaic box: x1={sample_box[0]:.1f}, y1={sample_box[1]:.1f}, x2={sample_box[2]:.1f}, y2={sample_box[3]:.1f}, cls={sample_box[4]}")
        if len(boxes_list) > 0 and len(boxes_list[0]) > 0:
            sample_orig = boxes_list[0][0]
            print(f"   Sample original box (img0): x1={sample_orig[0]:.1f}, y1={sample_orig[1]:.1f}, x2={sample_orig[2]:.1f}, y2={sample_orig[3]:.1f}, cls={sample_orig[4]}")
        
        # Create visualizations
        # 1. Original images with boxes
        original_images_with_boxes = []
        for img, boxes in zip(images_list, boxes_list):
            img_uint8 = (img * 255.0).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            img_with_boxes = draw_boxes_on_image(img_uint8, boxes, color=(0, 255, 0), thickness=2)
            original_images_with_boxes.append(img_with_boxes)
        
        # 2. Mosaic image with boxes
        mosaic_with_boxes = draw_boxes_on_image(mosaic_image_np, mosaic_boxes_valid, 
                                                color=(255, 0, 0), thickness=3)
        
        # Create grid visualization
        all_images = original_images_with_boxes + [mosaic_with_boxes]
        all_titles = [f"Original {i+1}" for i in range(4)] + ["Mosaic Result"]
        
        output_path = os.path.join(output_dir, f"mosaic_test_{test_idx + 1:03d}.png")
        create_visualization_grid(all_images, all_titles, output_path, figsize=(25, 5))
        
        # Also save individual mosaic result
        mosaic_output_path = os.path.join(output_dir, f"mosaic_only_{test_idx + 1:03d}.png")
        cv2.imwrite(mosaic_output_path, cv2.cvtColor(mosaic_with_boxes, cv2.COLOR_RGB2BGR))
        
        print(f"   Saved test visualization")
        print()
    
    print("=" * 80)
    print("Test Complete!")
    print(f"Visualizations saved to: {output_dir}")
    print()
    print("Visual Inspection Checklist:")
    print("  - Check that boxes in Mosaic result align with objects")
    print("  - Verify boxes from different quadrants are correctly positioned")
    print("  - Ensure no boxes are outside image boundaries")
    print("  - Confirm boxes maintain correct class labels")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Test Mosaic augmentation with visual verification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--annotation',
        type=str,
        required=True,
        help='Path to annotation file'
    )
    parser.add_argument(
        '--num-tests',
        type=int,
        default=5,
        help='Number of test cases to run'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=2,
        default=[608, 608],
        help='Input image shape (height width)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/mosaic_test_outputs',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    # Load annotation lines
    print(f"Loading annotations from: {args.annotation}")
    annotation_lines = load_annotation_lines(args.annotation, shuffle=False)
    print(f"Loaded {len(annotation_lines)} annotation lines")
    print()
    
    # Run tests
    test_mosaic_augmentation(
        annotation_lines=annotation_lines,
        input_shape=tuple(args.input_shape),
        num_tests=args.num_tests,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

