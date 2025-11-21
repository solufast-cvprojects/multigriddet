#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test suite for Mosaic augmentation using the actual training pipeline.

This script:
    1. Loads the training config and builds a MultiGridDataGenerator using real augmentation settings
    2. Uses build_visualization_dataset() so Mosaic/MixUp and all geometry come from generators.py
    3. Draws bounding boxes with **class names** (not indices) on the augmented images
    4. Saves grid visualizations for manual inspection of Mosaic behavior

The goal is to verify Mosaic as the model actually sees it during training, instead of
re-implementing augmentation logic in this test.
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

from test_utils import (
    load_class_names_from_config,
    convert_image_to_uint8,
    draw_boxes_with_class_names,
    get_generator_from_config,
)


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


def test_mosaic_augmentation(config_path: str,
                             num_batches: int = 3,
                             batch_size: int = 4,
                             seed: int = 42,
                             output_dir: str = "tests/mosaic_test_outputs") -> None:
    """
    Test Mosaic augmentation using the real training pipeline (MultiGridDataGenerator).
    
    Args:
        config_path: Path to training config YAML file
        num_batches: Number of batches to visualize
        batch_size: Batch size for the generator
        seed: Random seed for reproducibility
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print("=" * 80)
    print("Mosaic Augmentation Regression Test (Pipeline-accurate)")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Number of batches: {num_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Random seed: {seed}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load class names and colors from config
    class_names, colors = load_class_names_from_config(config_path)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes from config")
    
    # Build generator using the same settings as training
    print("Creating generator from config (including Mosaic/MixUp settings)...")
    generator = get_generator_from_config(
        config_path=config_path,
        augment=True,
        batch_size=batch_size
    )
    # Force Mosaic augmentation for this regression test, regardless of current training config.
    # This keeps the test focused on verifying Mosaic behavior while still using the exact
    # implementation from the training pipeline.
    generator.enhance_augment = 'mosaic'
    generator.mosaic_prob = 1.0
    generator.mixup_prob = 0.0
    print("   Overriding augmentation for test: enhance_augment='mosaic', mosaic_prob=1.0, mixup_prob=0.0")
    
    # Build visualization dataset that returns (images, boxes_dense)
    # This uses the exact same preprocessing and batch augmentations as training.
    print("Building visualization dataset...")
    dataset = generator.build_visualization_dataset(
        prefetch_buffer_size=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        shuffle_buffer_size=4096,
        interleave_cycle_length=None
    )
    print("   Dataset built")
    print()
    
    batch_count = 0
    for images, boxes in dataset:
        if batch_count >= num_batches:
            break
        
        batch_size_actual = int(images.shape[0])
        print(f"Batch {batch_count + 1}/{num_batches} (size: {batch_size_actual})")
        
        images_np = images.numpy()
        boxes_np = boxes.numpy()
        
        # Collect BGR images with class-name overlays for this batch
        batch_images_with_boxes = []
        titles = []
        
        for sample_idx in range(batch_size_actual):
            img = images_np[sample_idx]
            bxs = boxes_np[sample_idx]
            
            # Count valid boxes (exclude padded zeros)
            valid_mask = np.any(bxs[:, :4] != 0, axis=1)
            bxs_valid = bxs[valid_mask]
            
            # Convert image to uint8 and draw boxes with class names
            img_uint8 = convert_image_to_uint8(img)
            img_with_boxes_bgr = draw_boxes_with_class_names(
                img_uint8,
                bxs_valid,
                class_names,
                colors,
                show_score=False
            )
            
            batch_images_with_boxes.append(img_with_boxes_bgr)
            titles.append(f"Batch {batch_count+1} Sample {sample_idx+1} ({len(bxs_valid)} boxes)")
        
        # Create a grid visualization for this batch
        output_path = os.path.join(output_dir, f"mosaic_batch_{batch_count + 1:03d}.png")
        create_visualization_grid(batch_images_with_boxes, titles, output_path, figsize=(5 * batch_size_actual, 5))
        
        batch_count += 1
        print()
    
    print("=" * 80)
    print("Test Complete!")
    print(f"Visualizations saved to: {output_dir}")
    print()
    print("Visual Inspection Checklist:")
    print("  - Check that boxes in each composite image align with objects")
    print("  - Verify images exhibit Mosaic behavior (multiple source images per sample)")
    print("  - Ensure no boxes are outside image boundaries")
    print("  - Confirm boxes maintain correct class labels (via class names)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Test Mosaic augmentation using the real training pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config YAML file'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=3,
        help='Number of batches to visualize'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for the generator used in this test'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/mosaic_test_outputs',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    test_mosaic_augmentation(
        config_path=args.config,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

