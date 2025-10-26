#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base head class for MultiGridDet.
"""

import tensorflow as tf
import numpy as np
from ..base import BaseHead


class BaseHead(BaseHead):
    """Base class for MultiGridDet heads."""
    
    def __init__(self, config: dict):
        """
        Initialize base head.
        
        Args:
            config: Head configuration dictionary
        """
        super().__init__(config)
        self.num_scales = config.get('num_scales', 3)
        self.anchors_per_scale = config.get('anchors_per_scale', 3)
        self.grid_assignment = config.get('grid_assignment', '3x3')  # MultiGridDet innovation
        self.use_iol = config.get('use_iol', True)  # Use IoL anchor matching
    
    def get_num_scales(self) -> int:
        """
        Get the number of detection scales.
        
        Returns:
            Number of scales
        """
        return self.num_scales
    
    def get_anchors_per_scale(self) -> int:
        """
        Get the number of anchors per scale.
        
        Returns:
            Number of anchors per scale
        """
        return self.anchors_per_scale
    
    def get_grid_assignment(self) -> str:
        """
        Get the grid assignment strategy.
        
        Returns:
            Grid assignment strategy (e.g., '3x3' for MultiGridDet)
        """
        return self.grid_assignment
    
    def uses_iol(self) -> bool:
        """
        Check if this head uses IoL anchor matching.
        
        Returns:
            True if uses IoL, False otherwise
        """
        return self.use_iol
    
    def create_detection_layers(self, 
                               feature_maps: list, 
                               num_classes: int, 
                               anchors: list) -> list:
        """
        Create detection layers for each scale.
        
        Args:
            feature_maps: List of feature map tensors
            num_classes: Number of object classes
            anchors: List of anchor arrays for each scale
            
        Returns:
            List of detection output tensors
        """
        outputs = []
        
        for i, (feature_map, anchor_set) in enumerate(zip(feature_maps, anchors)):
            # Calculate number of anchors for this scale
            num_anchors = len(anchor_set)
            
            # Calculate output channels: 5 (bbox + objectness) + num_anchors + num_classes
            output_channels = 5 + num_anchors + num_classes
            
            # Create detection layer
            detection_layer = tf.keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=1,
                strides=1,
                padding='same',
                name=f'detection_{i}',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'
            )
            
            # Apply detection layer
            output = detection_layer(feature_map)
            outputs.append(output)
        
        return outputs
    
    def reshape_outputs(self, outputs: list, anchors: list) -> list:
        """
        Reshape outputs to proper format for loss calculation.
        
        Args:
            outputs: List of detection output tensors
            anchors: List of anchor arrays
            
        Returns:
            List of reshaped output tensors
        """
        reshaped_outputs = []
        
        for i, (output, anchor_set) in enumerate(zip(outputs, anchors)):
            # Get output shape
            batch_size = tf.shape(output)[0]
            height = tf.shape(output)[1]
            width = tf.shape(output)[2]
            channels = tf.shape(output)[3]
            
            # Reshape to (batch, height, width, anchors, channels_per_anchor)
            num_anchors = len(anchor_set)
            channels_per_anchor = channels // num_anchors
            
            reshaped = tf.reshape(
                output,
                [batch_size, height, width, num_anchors, channels_per_anchor]
            )
            
            reshaped_outputs.append(reshaped)
        
        return reshaped_outputs
