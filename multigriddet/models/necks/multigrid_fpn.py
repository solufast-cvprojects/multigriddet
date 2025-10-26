#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGrid FPN (Feature Pyramid Network) neck for MultiGridDet.
Extracted from denseyolo2_predictions() implementation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, UpSampling2D, BatchNormalization, LeakyReLU
from typing import Tuple, List, Optional, Dict, Any

from .base_neck import BaseNeck
from ..registry import register_neck


def darknet_conv2d_bn_leaky(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    """Darknet Conv2D with batch normalization and LeakyReLU."""
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


@register_neck('multigrid_fpn', {
    'feature_channels': (512, 256, 128),
    'upsample_method': 'upsampling2d',
    'fusion_method': 'concatenate'
})
class MultiGridFPN(BaseNeck):
    """
    MultiGrid FPN neck for MultiGridDet.
    
    This neck implements the Feature Pyramid Network logic from denseyolo2_predictions():
    - Takes 3 feature maps from backbone: f3 (52x52), f2 (26x26), f1 (13x13)
    - Applies channel reduction convolutions
    - Performs upsampling and concatenation for FPN fusion
    - Returns enhanced features for each scale
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MultiGrid FPN neck.
        
        Args:
            config: Neck configuration dictionary
        """
        super().__init__(config)
        
        # Feature channel numbers (from denseyolo2_predictions)
        self.f1_channel_num = config.get('f1_channel_num', 512)  # 13x13 scale
        self.f2_channel_num = config.get('f2_channel_num', 256)  # 26x26 scale  
        self.f3_channel_num = config.get('f3_channel_num', 128)  # 52x52 scale
        
        # FPN configuration
        self.upsample_method = config.get('upsample_method', 'upsampling2d')
        self.fusion_method = config.get('fusion_method', 'concatenate')
    
    def build_neck(self, feature_maps: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Build MultiGrid FPN neck.
        
        Args:
            feature_maps: List of feature maps from backbone [f1, f2, f3]
                         f1: 13x13x1024 (large objects)
                         f2: 26x26x512 (medium objects)  
                         f3: 52x52x256 (small objects)
        
        Returns:
            List of enhanced feature maps for each scale
        """
        f1, f2, f3 = feature_maps
        
        # This follows the exact logic from denseyolo2_predictions()
        # Process f1 first (13x13 scale)
        x = darknet_conv2d_bn_leaky(f1, self.f1_channel_num // 2, (1, 1), name='fpn_f1_reduce')
        
        # Upsample and merge with f2 for scale 2 (26x26)
        x = darknet_conv2d_bn_leaky(x, self.f2_channel_num // 2, (1, 1), name='fpn_f1_to_f2')
        x = UpSampling2D(2, name='fpn_upsample_f1_to_f2')(x)
        x = Concatenate(name='fpn_concat_f2')([x, f2])
        
        # Process f2 for scale 2 (26x26)
        x_f2 = darknet_conv2d_bn_leaky(x, self.f2_channel_num // 2, (3, 3), name='fpn_f2_conv1')
        x_f2 = darknet_conv2d_bn_leaky(x_f2, self.f2_channel_num, (3, 3), name='fpn_f2_conv2')
        
        # Upsample and merge with f3 for scale 3 (52x52)
        x = darknet_conv2d_bn_leaky(x, self.f3_channel_num // 2, (1, 1), name='fpn_f2_to_f3')
        x = UpSampling2D(2, name='fpn_upsample_f2_to_f3')(x)
        x = Concatenate(name='fpn_concat_f3')([x, f3])
        
        # Process f3 for scale 3 (52x52)
        x_f3 = darknet_conv2d_bn_leaky(x, self.f3_channel_num // 2, (3, 3), name='fpn_f3_conv1')
        x_f3 = darknet_conv2d_bn_leaky(x_f3, self.f3_channel_num, (3, 3), name='fpn_f3_conv2')
        
        # Return enhanced features in order: [f1, f2, f3] (largest to smallest)
        # This matches the original denseyolo2_predictions return order
        return [f1, x_f2, x_f3]
    
    def get_output_shapes(self, input_shapes: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Get output shapes for the FPN neck.
        
        Args:
            input_shapes: List of input feature map shapes
            
        Returns:
            List of output feature map shapes
        """
        # Output shapes match input shapes but with different channel numbers
        f3_shape, f2_shape, f1_shape = input_shapes
        
        return [
            (f3_shape[0], f3_shape[1], self.f3_channel_num),  # 52x52x128
            (f2_shape[0], f2_shape[1], self.f2_channel_num),  # 26x26x256
            (f1_shape[0], f1_shape[1], self.f1_channel_num // 2)  # 13x13x256
        ]
