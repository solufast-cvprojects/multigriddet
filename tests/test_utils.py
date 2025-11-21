#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for test scripts.

This module provides common functions used across all test scripts to:
1. Load data using the actual training pipeline from generators.py
2. Visualize with class names instead of indices
3. Avoid code duplication across test scripts
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.data.generators import MultiGridDataGenerator
from multigriddet.data.utils import get_classes, get_colors, draw_boxes, load_annotation_lines
from multigriddet.config import ConfigLoader


def load_class_names_from_config(config_path: str) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    """
    Load class names and colors from config file.
    
    Args:
        config_path: Path to training config YAML file
        
    Returns:
        Tuple of (class_names, colors)
    """
    config = ConfigLoader.load_config(config_path)
    
    # Get classes path from config
    classes_path = config.get('data', {}).get('classes_path')
    if not classes_path:
        # Try model config
        model_config_path = config.get('model_config')
        if model_config_path:
            model_config = ConfigLoader.load_config(model_config_path)
            classes_path = model_config.get('model', {}).get('preset', {}).get('classes_path')
    
    if not classes_path:
        raise ValueError("classes_path not found in config")
    
    class_names = get_classes(classes_path)
    colors = get_colors(len(class_names))
    
    return class_names, colors


def convert_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 format for visualization.
    
    Args:
        image: Image array in any format (float32 [0,1], float32 [0,255], or uint8)
        
    Returns:
        Image array in uint8 format [0, 255]
    """
    if image.dtype == np.uint8:
        return image.copy()
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            # Normalized [0, 1] range
            image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            # Already in [0, 255] range
            image_uint8 = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    return image_uint8


def draw_boxes_with_class_names(image: np.ndarray,
                               boxes: np.ndarray,
                               class_names: List[str],
                               colors: List[Tuple[int, int, int]],
                               show_score: bool = False,
                               scores: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draw bounding boxes on image with class names (not indices).
    
    Args:
        image: Image array (H, W, 3) in any format
        boxes: Boxes array (N, 5) in format (x1, y1, x2, y2, class_index)
        class_names: List of class names
        colors: List of RGB color tuples for each class
        show_score: Whether to show scores
        scores: Optional scores array (N,)
        
    Returns:
        Image with boxes drawn (uint8 BGR format for OpenCV)
    """
    # Convert image to uint8
    image_uint8 = convert_image_to_uint8(image)
    
    # Ensure image is in BGR format for OpenCV
    if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
        # Check if it's RGB (TensorFlow format) and convert to BGR
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_uint8.copy()
    
    # Filter valid boxes
    valid_mask = (
        (boxes[:, 0] < boxes[:, 2]) &
        (boxes[:, 1] < boxes[:, 3]) &
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0)
    )
    valid_boxes = boxes[valid_mask]
    
    if len(valid_boxes) == 0:
        return image_bgr
    
    # Prepare data for draw_boxes
    boxes_list = valid_boxes[:, :4].tolist()  # [x1, y1, x2, y2]
    
    # Convert class indices to class names
    class_indices = valid_boxes[:, 4].astype(np.int32)
    class_indices = np.clip(class_indices, 0, len(class_names) - 1)
    classes_list = [class_names[int(cls)] for cls in class_indices]
    
    if scores is not None and show_score:
        valid_scores = scores[valid_mask]
        scores_list = valid_scores.tolist()
    else:
        scores_list = [1.0] * len(valid_boxes)
    
    # Use the standard draw_boxes function
    image_with_boxes = draw_boxes(
        image_bgr,
        boxes_list,
        class_indices, 
        scores_list,
        class_names,
        colors,
        show_score=show_score
    )
    
    return image_with_boxes


def get_generator_from_config(config_path: str, 
                             augment: bool = True,
                             batch_size: int = 4) -> MultiGridDataGenerator:
    """
    Create a MultiGridDataGenerator from config file.
    
    This ensures test scripts use the EXACT same pipeline as training.
    
    Args:
        config_path: Path to training config YAML file
        augment: Whether to enable augmentations
        batch_size: Batch size for the generator
        
    Returns:
        MultiGridDataGenerator instance
    """
    config = ConfigLoader.load_config(config_path)
    
    # Get model config
    model_config_path = config.get('model_config')
    if not model_config_path:
        raise ValueError("model_config not found in config")
    
    model_config = ConfigLoader.load_config(model_config_path)
    
    # Get data config
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Get paths
    train_annotation = data_config.get('train_annotation')
    classes_path = data_config.get('classes_path')
    anchors_path = model_config.get('model', {}).get('preset', {}).get('anchors_path')
    
    if not all([train_annotation, classes_path, anchors_path]):
        raise ValueError("Missing required paths in config")
    
    # Get input shape
    input_shape = tuple(model_config.get('model', {}).get('preset', {}).get('input_shape', [608, 608, 3])[:2])
    
    # Get augmentation config
    augment_config = training_config.get('augmentation', {})
    enhance_type = augment_config.get('enhance_type', 'mosaic')
    mosaic_prob = augment_config.get('mosaic_prob', 0.3)
    mixup_prob = augment_config.get('mixup_prob', 0.1)
    
    # Get other config
    max_boxes_per_image = data_config.get('max_boxes_per_image', 100)
    num_workers = config.get('data_loader', {}).get('num_workers', 8)
    rescale_interval = training_config.get('rescale_interval', -1)
    
    # Load classes to get num_classes
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    
    # Load anchors
    from multigriddet.utils.anchors import load_anchors
    anchors = load_anchors(anchors_path)
    
    # Create generator
    generator = MultiGridDataGenerator(
        annotation_lines=load_annotation_lines(train_annotation, shuffle=False)[:100],  # Limit for testing
        input_shape=input_shape,
        batch_size=batch_size,
        num_classes=num_classes,
        anchors=anchors,
        max_boxes_per_image=max_boxes_per_image,
        augment=augment,
        enhance_augment=enhance_type,
        rescale_interval=rescale_interval,
        multi_anchor_assign=training_config.get('multi_anchor_assign', False),
        shuffle=False,  # Disable shuffle for reproducible tests
        num_workers=num_workers,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob
    )
    
    return generator
