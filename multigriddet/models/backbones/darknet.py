#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Darknet53 backbone for MultiGridDet.
Migrated from original implementation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from .base_backbone import BaseBackbone
from ..registry import register_backbone
from ..layers import DarknetConv2D_BN_Leaky, compose


def darknet53_body(x):
    """Darknet53 body having 52 Convolution2D layers - EXACT replica of original for weight compatibility."""
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body_original(x, 64, 1)
    x = resblock_body_original(x, 128, 2)
    x = resblock_body_original(x, 256, 8)
    x = resblock_body_original(x, 512, 8)
    x = resblock_body_original(x, 1024, 4)
    return x


def resblock_body_original(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D - EXACT original version for weight compatibility."""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x


def darknet_conv2d(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    """Darknet Conv2D layer with batch normalization and LeakyReLU."""
    if padding == 'same' and strides == (2, 2):
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=name + '_conv' if name else None
    )(x)
    
    x = BatchNormalization(name=name + '_bn' if name else None)(x)
    x = LeakyReLU(alpha=0.1, name=name + '_leaky' if name else None)(x)
    
    return x


def resblock_body(x, num_filters, num_blocks, name_prefix):
    """A series of resblocks starting with a downsampling Convolution2D."""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = darknet_conv2d(x, num_filters, (3, 3), strides=(2, 2), name=name_prefix + '_downsample')
    
    for i in range(num_blocks):
        y = darknet_conv2d(x, num_filters // 2, (1, 1), name=f'{name_prefix}_res{i}_conv1')
        y = darknet_conv2d(y, num_filters, (3, 3), name=f'{name_prefix}_res{i}_conv2')
        x = Add(name=f'{name_prefix}_res{i}_add')([x, y])
    
    return x


@register_backbone('darknet53', {
    'pretrained_weights': None,
    'freeze_layers': 0,
    'feature_layer_names': ['conv2d_92', 'conv2d_152', 'conv2d_185']
})
class Darknet53Backbone(BaseBackbone):
    """Darknet53 backbone for MultiGridDet."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Darknet53 backbone.
        
        Args:
            config: Backbone configuration dictionary
        """
        super().__init__(config)
        self.feature_layer_names = [
            'conv2d_92',   # 52x52x256 (scale 3)
            'conv2d_152',  # 26x26x512 (scale 2) 
            'conv2d_185'   # 13x13x1024 (scale 1)
        ]
    
    def build_backbone(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """
        Build Darknet53 backbone.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            
        Returns:
            Built backbone model
        """
        inputs = tf.keras.layers.Input(shape=input_shape, name='input')
        
        # Initial convolution
        x = darknet_conv2d(inputs, 32, (3, 3), name='conv1')
        
        # Resblock bodies
        x = resblock_body(x, 64, 1, 'res1')    # 64 filters, 1 block
        x = resblock_body(x, 128, 2, 'res2')   # 128 filters, 2 blocks
        x = resblock_body(x, 256, 8, 'res3')   # 256 filters, 8 blocks
        x = resblock_body(x, 512, 8, 'res4')   # 512 filters, 8 blocks
        x = resblock_body(x, 1024, 4, 'res5')  # 1024 filters, 4 blocks
        
        # Create model
        self.model = Model(inputs=inputs, outputs=x, name='darknet53')
        
        # Load pretrained weights if specified
        if self.pretrained_weights:
            self.load_pretrained_weights(self.pretrained_weights)
        
        # Freeze layers if specified
        if self.freeze_layers > 0:
            self.freeze_layers(self.freeze_layers)
        
        return self.model
    
    def get_feature_maps(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        """
        Extract feature maps from Darknet53 backbone.
        
        Args:
            inputs: Input tensor
            
        Returns:
            List of feature map tensors at different scales
        """
        if self.model is None:
            raise ValueError("Backbone must be built before extracting feature maps")
        
        # Get feature maps from specific layers
        feature_maps = []
        
        # Scale 3: 52x52x256 (for small objects)
        scale3_output = self.model.get_layer('conv2d_92').output
        feature_maps.append(scale3_output)
        
        # Scale 2: 26x26x512 (for medium objects)
        scale2_output = self.model.get_layer('conv2d_152').output
        feature_maps.append(scale2_output)
        
        # Scale 1: 13x13x1024 (for large objects)
        scale1_output = self.model.get_layer('conv2d_185').output
        feature_maps.append(scale1_output)
        
        return feature_maps
    
    def get_output_shapes(self, input_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Get output shapes for each feature map.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            
        Returns:
            List of output shapes (height, width, channels)
        """
        height, width, channels = input_shape
        
        # Calculate output shapes based on downsampling
        output_shapes = [
            (height // 8, width // 8, 256),   # Scale 3: 8x downsampling
            (height // 16, width // 16, 512), # Scale 2: 16x downsampling
            (height // 32, width // 32, 1024) # Scale 1: 32x downsampling
        ]
        
        return output_shapes


@register_backbone('csp_darknet53', {
    'pretrained_weights': None,
    'freeze_layers': 0,
    'feature_layer_names': ['conv2d_131', 'conv2d_204', 'conv2d_234']
})
class CSPDarknet53Backbone(BaseBackbone):
    """CSPDarknet53 backbone for MultiGridDet (YOLOv4 style)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSPDarknet53 backbone.
        
        Args:
            config: Backbone configuration dictionary
        """
        super().__init__(config)
        self.feature_layer_names = [
            'conv2d_131',  # 52x52x256 (scale 3)
            'conv2d_204',  # 26x26x512 (scale 2)
            'conv2d_234'   # 13x13x1024 (scale 1)
        ]
    
    def build_backbone(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """
        Build CSPDarknet53 backbone.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            
        Returns:
            Built backbone model
        """
        # For now, use regular Darknet53 as base
        # In a full implementation, this would include CSP blocks
        darknet53 = Darknet53Backbone(self.config)
        return darknet53.build_backbone(input_shape)
    
    def get_feature_maps(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        """Extract feature maps from CSPDarknet53 backbone."""
        if self.model is None:
            raise ValueError("Backbone must be built before extracting feature maps")
        
        # Get feature maps from specific layers
        feature_maps = []
        
        # Scale 3: 52x52x256
        scale3_output = self.model.get_layer('conv2d_131').output
        feature_maps.append(scale3_output)
        
        # Scale 2: 26x26x512
        scale2_output = self.model.get_layer('conv2d_204').output
        feature_maps.append(scale2_output)
        
        # Scale 1: 13x13x1024
        scale1_output = self.model.get_layer('conv2d_234').output
        feature_maps.append(scale1_output)
        
        return feature_maps
    
    def get_output_shapes(self, input_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get output shapes for each feature map."""
        height, width, channels = input_shape
        
        # Calculate output shapes based on downsampling
        output_shapes = [
            (height // 8, width // 8, 256),   # Scale 3: 8x downsampling
            (height // 16, width // 16, 512), # Scale 2: 16x downsampling
            (height // 32, width // 32, 1024) # Scale 1: 32x downsampling
        ]
        
        return output_shapes
