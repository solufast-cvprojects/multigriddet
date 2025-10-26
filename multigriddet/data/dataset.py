#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Dataset class with tf.data.Dataset integration.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional, Union, Dict, Any
import cv2
import os
import sys

# Add multigriddet to path for importing the proven generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# Import from the new structure
from .data_tf217_optimized import OptimizedDenseYOLO2DataGenerator

from .augmentation import ModernAugmentation, AugmentationConfig
from .preprocessing import ImagePreprocessor
from .target_encoding import MultiGridTargetEncoder, MultiGridConfig
from .keras3_generator_wrapper import Keras3CompatibleGenerator


class MultiGridDataset:
    """
    MultiGridDet dataset wrapper around proven OptimizedDataGenerator.
    
    This provides a modern config-based API while using the reliable working generator internally.
    """
    
    def __init__(self, 
                 annotation_lines: List[str],
                 config: MultiGridConfig,
                 augmentation_config: Optional[AugmentationConfig] = None,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 cache: bool = False,
                 rescale_interval: int = -1,
                 input_shape_list: Optional[List[Tuple[int, int]]] = None,
                 **kwargs):
        """
        Initialize MultiGridDet dataset wrapper.
         
        Args:
            annotation_lines: List of annotation lines (image_path box1 box2 ...)
            config: MultiGridDet configuration
            augmentation_config: Augmentation configuration
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            cache: Whether to cache the dataset in memory
            rescale_interval: Interval for multi-scale training
            input_shape_list: List of input shapes for multi-scale training
            **kwargs: Additional arguments passed to OptimizedDataGenerator
        """
        self.annotation_lines = annotation_lines
        self.config = config
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache = cache
        self.rescale_interval = rescale_interval
        self.input_shape_list = input_shape_list or [config.input_shape]
        
        # Extract parameters from config
        input_shape = config.input_shape
        anchors = config.anchors
        num_classes = config.num_classes
        
        # Determine augmentation settings
        augment = True
        if augmentation_config:
            # Check if any augmentation is enabled
            augment = any([
                augmentation_config.horizontal_flip_prob > 0,
                augmentation_config.vertical_flip_prob > 0,
                augmentation_config.rotation_range > 0,
                augmentation_config.brightness_jitter > 0,
                augmentation_config.contrast_jitter > 0,
                augmentation_config.saturation_jitter > 0,
                augmentation_config.hue_jitter > 0,
                augmentation_config.blur_prob > 0,
                augmentation_config.noise_prob > 0,
                augmentation_config.gridmask_prob > 0
            ])
        
        # Create Keras 3.0-compatible generator using proven OptimizedDataGenerator
        self._generator = Keras3CompatibleGenerator(
            annotation_lines=annotation_lines,
            config=config,
            augmentation_config=augmentation_config,
            batch_size=batch_size,
            shuffle=shuffle,
            rescale_interval=rescale_interval,
            input_shape_list=input_shape_list,
            **kwargs
        )
        
        # For maximum GPU efficiency, we use the internal generator directly
        # No need for tf.data.Dataset wrapper as the internal generator is already optimized
    
    def _get_grid_shape(self, scale: int) -> Tuple[int, int]:
        """Get grid shape for given scale."""
        stride = 2 ** (5 + scale)  # 32, 16, 8 for scales 0, 1, 2
        return (self.config.input_shape[0] // stride, self.config.input_shape[1] // stride)
    
    def _get_anchors_per_scale(self, scale: int) -> int:
        """Get number of anchors per scale."""
        return len(self.config.anchors[scale]) if scale < len(self.config.anchors) else 3
    
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self._generator)
    
    def __getitem__(self, index):
        """Get sample at specified index."""
        return self._generator[index]
    
    def __iter__(self):
        """Make dataset iterable."""
        return iter(self._generator)
    
    def __next__(self):
        """Get next sample."""
        return next(self._generator)
    
    @property
    def generator(self):
        """Get the internal Keras Sequence generator for model.fit()"""
        return self._generator
    
    # Delegate any other methods to the internal generator
    def __getattr__(self, name):
        """Delegate unknown attributes to the internal generator."""
        return getattr(self._generator, name)


def create_dataset(annotation_lines: List[str],
                  config: MultiGridConfig,
                  batch_size: int = 16,
                  augmentation_config: Optional[AugmentationConfig] = None,
                  **kwargs) -> MultiGridDataset:
    """
    Convenience function to create a MultiGridDataset.
    
    Args:
        annotation_lines: List of annotation lines
        config: MultiGridConfig configuration
        batch_size: Batch size
        augmentation_config: Augmentation configuration
        **kwargs: Additional arguments
        
    Returns:
        MultiGridDataset instance
    """
    return MultiGridDataset(
        annotation_lines=annotation_lines,
        config=config,
        batch_size=batch_size,
        augmentation_config=augmentation_config,
        **kwargs
    )