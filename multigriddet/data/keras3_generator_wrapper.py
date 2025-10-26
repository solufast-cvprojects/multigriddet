#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3.0-compatible wrapper for OptimizedDataGenerator.
This resolves the output_signature error by properly implementing PyDataset.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional, Union, Dict, Any
from tensorflow.keras.utils import Sequence
import os
import sys

# Add multigriddet to path for importing the proven generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from denseyolo2.data_tf217_optimized import OptimizedDenseYOLO2DataGenerator

from .augmentation import ModernAugmentation, AugmentationConfig
from .preprocessing import ImagePreprocessor
from .target_encoding import MultiGridTargetEncoder, MultiGridConfig


class Keras3CompatibleGenerator(Sequence):
    """
    Keras 3.0-compatible wrapper for OptimizedDataGenerator.
    
    This class properly implements the PyDataset interface expected by Keras 3.0,
    resolving the output_signature error while maintaining all the proven functionality.
    """
    
    def __init__(self, 
                 annotation_lines: List[str],
                 config: MultiGridConfig,
                 augmentation_config: Optional[AugmentationConfig] = None,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 rescale_interval: int = -1,
                 input_shape_list: Optional[List[Tuple[int, int]]] = None,
                 **kwargs):
        """
        Initialize Keras 3.0-compatible generator wrapper.
        
        Args:
            annotation_lines: List of annotation lines
            config: MultiGridConfig configuration
            augmentation_config: Augmentation configuration
            batch_size: Batch size
            shuffle: Whether to shuffle data
            rescale_interval: Interval for multi-scale training
            input_shape_list: List of input shapes for multi-scale training
            **kwargs: Additional arguments passed to OptimizedDataGenerator
        """
        # Call parent constructor with all kwargs for Keras 3.0 compatibility
        super().__init__(**kwargs)
        
        self.annotation_lines = annotation_lines
        self.config = config
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rescale_interval = rescale_interval
        self.input_shape_list = input_shape_list or [config.input_shape]
        
        # Extract parameters from config
        input_shape = config.input_shape
        anchors = config.anchors
        num_classes = config.num_classes
        
        # Determine augmentation settings
        augment = True
        enhance_augment = None
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
                augmentation_config.noise_prob > 0,
                augmentation_config.blur_prob > 0,
                augmentation_config.gridmask_prob > 0
            ])
            
            # Set enhance_augment based on config
            if hasattr(augmentation_config, 'enhance_augment'):
                enhance_augment = augmentation_config.enhance_augment
        
        # Create the internal optimized generator
        self._generator = OptimizedDenseYOLO2DataGenerator(
            annotation_lines=annotation_lines,
            batch_size=batch_size,
            input_shape=input_shape,
            anchors=anchors,
            num_classes=num_classes,
            augment=augment,
            enhance_augment=enhance_augment,
            rescale_interval=rescale_interval,
            multi_anchor_assign=config.multi_anchor_assign,
            shuffle=shuffle,
            **kwargs
        )
    
    def __len__(self):
        """Return the number of batches."""
        return len(self._generator)
    
    def __getitem__(self, index):
        """Get a batch of data."""
        return self._generator[index]
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if hasattr(self._generator, 'on_epoch_end'):
            self._generator.on_epoch_end()
    
    @property
    def anchors(self):
        """Get anchors from the internal generator."""
        return self._generator.anchors
    
    @property
    def num_classes(self):
        """Get number of classes from the internal generator."""
        return self._generator.num_classes
    
    @property
    def input_shape(self):
        """Get input shape from the internal generator."""
        return self._generator.input_shape
