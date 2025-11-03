#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet head implementation.
This is the core innovation of MultiGridDet - dense prediction with 3x3 grid assignment.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, UpSampling2D, ZeroPadding2D, BatchNormalization, LeakyReLU
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from .base_head import BaseHead
from ..registry import register_head


def darknet_conv2d_bn_leaky(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    """Darknet Conv2D with batch normalization and LeakyReLU."""
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


def make_last_layers(x, num_filters, out_filters, predict_filters=None, predict_id='1'):
    """
    Make the last layers for MultiGridDet head - EXACT replica of original.
    
    Args:
        x: Input tensor
        num_filters: Number of filters for intermediate layers
        out_filters: Number of output filters
        predict_filters: Number of filters for prediction layer
        predict_id: Prediction layer identifier
        
    Returns:
        Tuple of (processed_tensor, prediction_tensor)
    """
    from ..layers import DarknetConv2D_BN_Leaky, compose
    from tensorflow.keras.layers import Conv2D
    
    # First set of layers (matching original exactly)
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1,1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
        # DarknetConv2D_BN_Leaky(num_filters, (1,1)),  # Commented out in original
        # DarknetConv2D_BN_Leaky(num_filters*2, (3,3)), # Commented out in original
        DarknetConv2D_BN_Leaky(num_filters, (1,1))
    )(x)
    
    # Set predict_filters if not provided
    if predict_filters is None:
        predict_filters = num_filters*2
    
    # Final prediction layers (matching original exactly)
    y = compose(
        DarknetConv2D_BN_Leaky(predict_filters, (3,3)),
        Conv2D(out_filters, (1,1), name='predict_conv_' + predict_id)
    )(x)
    
    return x, y


@register_head('multigrid', {
    'num_scales': 3,
    'anchors_per_scale': 3,
    'grid_assignment': '3x3',
    'use_iol': True
})
class MultiGridHead(BaseHead):
    """
    MultiGridDet head implementation.
    
    This head implements the core MultiGridDet innovation:
    - Dense prediction where multiple grid cells detect each object
    - 3x3 grid assignment per object (configurable)
    - IoL-based anchor matching
    - Feature Pyramid Network (FPN) style upsampling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MultiGridDet head.
        
        Args:
            config: Head configuration dictionary
        """
        super().__init__(config)
        self.num_scales = config.get('num_scales', 3)
        self.anchors_per_scale = config.get('anchors_per_scale', 3)
        self.grid_assignment = config.get('grid_assignment', '3x3')
        self.use_iol = config.get('use_iol', True)
        
        # Feature channel numbers for each scale
        self.feature_channels = config.get('feature_channels', [512, 256, 128])
    
    def build_head(self, feature_maps: List[tf.Tensor], 
                   num_classes: int, 
                   anchors: List[np.ndarray]) -> List[tf.Tensor]:
        """
        Build the MultiGridDet head.
        
        Args:
            feature_maps: List of feature map tensors from backbone
            num_classes: Number of object classes
            anchors: List of anchor arrays for each scale
            
        Returns:
            List of output tensors for each scale
        """
        if len(feature_maps) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} feature maps, got {len(feature_maps)}")
        
        if len(anchors) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} anchor sets, got {len(anchors)}")
        
        # Get feature maps (from largest to smallest scale)
        f1, f2, f3 = feature_maps  # f1: 13x13, f2: 26x26, f3: 52x52
        f1_channels, f2_channels, f3_channels = self.feature_channels
        
        outputs = []
        
        # Scale 1: Large objects (13x13 for 416 input)
        # Output format: 5 (bbox + objectness) + num_anchors + num_classes
        out_filters_1 = 5 + len(anchors[0]) + num_classes
        x1, y1 = make_last_layers(f1, f1_channels // 2, out_filters_1, '1')
        outputs.append(y1)
        
        # Upsample and merge with f2
        x1_upsample = darknet_conv2d_bn_leaky(x1, f2_channels // 2, (1, 1), name='upsample_1_conv')
        x1_upsample = UpSampling2D(2, name='upsample_1')(x1_upsample)
        x2_merged = Concatenate(name='merge_2')([x1_upsample, f2])
        
        # Scale 2: Medium objects (26x26 for 416 input)
        out_filters_2 = 5 + len(anchors[1]) + num_classes
        x2, y2 = make_last_layers(x2_merged, f2_channels // 2, out_filters_2, '2')
        outputs.append(y2)
        
        # Upsample and merge with f3
        x2_upsample = darknet_conv2d_bn_leaky(x2, f3_channels // 2, (1, 1), name='upsample_2_conv')
        x2_upsample = UpSampling2D(2, name='upsample_2')(x2_upsample)
        x3_merged = Concatenate(name='merge_3')([x2_upsample, f3])
        
        # Scale 3: Small objects (52x52 for 416 input)
        out_filters_3 = 5 + len(anchors[2]) + num_classes
        x3, y3 = make_last_layers(x3_merged, f3_channels // 2, out_filters_3, '3')
        outputs.append(y3)
        
        return outputs
    
    def get_output_shapes(self, 
                         input_shapes: List[Tuple[int, int, int]], 
                         num_classes: int, 
                         anchors: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Get output shapes for MultiGridDet head.
        
        Args:
            input_shapes: List of input feature map shapes
            num_classes: Number of object classes
            anchors: List of anchor arrays
            
        Returns:
            List of output shapes (batch, height, width, channels)
        """
        if len(input_shapes) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} input shapes, got {len(input_shapes)}")
        
        output_shapes = []
        
        for i, (input_shape, anchor_set) in enumerate(zip(input_shapes, anchors)):
            height, width, channels = input_shape
            
            # Calculate output channels: 5 (bbox + objectness) + num_anchors + num_classes
            output_channels = 5 + len(anchor_set) + num_classes
            
            # Output shape: (batch, height, width, channels)
            output_shapes.append((None, height, width, output_channels))
        
        return output_shapes
    
    def get_grid_assignment_info(self) -> Dict[str, Any]:
        """
        Get information about the grid assignment strategy.
        
        Returns:
            Dictionary with grid assignment information
        """
        return {
            'strategy': self.grid_assignment,
            'description': f'MultiGridDet uses {self.grid_assignment} grid assignment',
            'innovation': 'Each object is assigned to multiple grid cells for dense prediction',
            'use_iol': self.use_iol,
            'iol_description': 'Uses IoL (Intersection over Largest) for anchor matching'
        }
    
    def explain_multigrid_innovation(self) -> str:
        """
        Explain the MultiGridDet innovation.
        
        Returns:
            String explanation of the innovation
        """
        return """
        MultiGridDet Innovation:
        
        1. Dense Prediction: Unlike traditional YOLO where each object is assigned to a single grid cell,
           MultiGridDet assigns each object to multiple grid cells (3x3 area by default).
        
        2. IoL Anchor Matching: Uses Intersection over Largest (IoL) instead of IoU for anchor matching,
           which provides better handling of objects with extreme aspect ratios.
        
        3. Improved Small Object Detection: The dense prediction strategy improves detection of small objects
           by ensuring multiple grid cells contribute to their detection.
        
        4. Better Localization: Multiple grid cells predicting the same object leads to more accurate
           bounding box regression and better localization.
        
        5. Robust Training: The dense assignment makes training more robust and reduces the impact of
           annotation inconsistencies.
        """


@register_head('multigrid_lite', {
    'num_scales': 3,
    'anchors_per_scale': 3,
    'grid_assignment': '3x3',
    'use_iol': True,
    'feature_channels': [256, 128, 64]
})
class MultiGridLiteHead(MultiGridHead):
    """
    Lightweight version of MultiGridDet head.
    
    Uses fewer filters for faster inference while maintaining
    the MultiGridDet dense prediction innovation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MultiGridLite head.
        
        Args:
            config: Head configuration dictionary
        """
        # Override feature channels for lite version
        config['feature_channels'] = config.get('feature_channels', [256, 128, 64])
        super().__init__(config)
    
    def build_head(self, feature_maps: List[tf.Tensor], 
                   num_classes: int, 
                   anchors: List[np.ndarray]) -> List[tf.Tensor]:
        """
        Build the lightweight MultiGridDet head.
        
        Uses fewer filters and simpler architecture for faster inference.
        """
        # Use the same architecture as MultiGridHead but with fewer filters
        return super().build_head(feature_maps, num_classes, anchors)


def multigriddet_predictions(feature_maps, feature_channel_nums, num_anchors_per_head, num_classes, use_spp=False):
    """
    MultiGridDet predictions function.
    This is the exact function from the original implementation.
    """
    from ..layers import DarknetConv2D_BN_Leaky, compose
    from tensorflow.keras.layers import UpSampling2D, Concatenate
    
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head & output (13x13 for 416 input)
    pred_filter = 8 * (num_anchors_per_head[0] + num_classes + 5)

    if use_spp:
        x, y1 = make_spp_last_layers(f1, f1_channel_num//2, num_anchors_per_head[0] + num_classes + 5, predict_id='1')
    else:
        x, y1 = make_last_layers(f1, f1_channel_num//2, num_anchors_per_head[0] + num_classes + 5, pred_filter, predict_id='1')

    #upsample fpn merge for feature map 1 & 2
    x = compose(
            DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,f2])

    #feature map 2 head & output (26x26 for 416 input)
    pred_filter = 4 * (num_anchors_per_head[0] + num_classes + 5)
    x, y2 = make_last_layers(x, f2_channel_num//2, num_anchors_per_head[1] + num_classes + 5, pred_filter, predict_id='2')

    #upsample fpn merge for feature map 2 & 3
    x = compose(
            DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    pred_filter = 2 * (num_anchors_per_head[0] + num_classes + 5)
    x, y3 = make_last_layers(x, f3_channel_num//2, num_anchors_per_head[2] + num_classes + 5, pred_filter, predict_id='3')
    return y1, y2, y3


def make_spp_last_layers(x, num_filters, out_filters, predict_id):
    """Make SPP last layers for compatibility."""
    from ..layers import DarknetConv2D_BN_Leaky, compose
    from tensorflow.keras.layers import MaxPooling2D, Concatenate
    
    x = darknet_conv2d_bn_leaky(x, num_filters, (1,1))
    x = darknet_conv2d_bn_leaky(x, num_filters*2, (3,3))
    x = darknet_conv2d_bn_leaky(x, num_filters, (1,1))
    
    # SPP layers
    maxpool5 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
    maxpool9 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
    maxpool13 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)
    
    x = Concatenate()([x, maxpool5, maxpool9, maxpool13])
    
    x = darknet_conv2d_bn_leaky(x, num_filters, (1,1))
    x = darknet_conv2d_bn_leaky(x, num_filters*2, (3,3))
    x = darknet_conv2d_bn_leaky(x, num_filters, (1,1))
    
    y = Conv2D(out_filters, (1,1), name='predict_conv_' + predict_id)(x)
    
    return x, y
