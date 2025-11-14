#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet ResNet model builder.
This is the modern ResNet-style Darknet53 implementation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from .backbones.darknet import Darknet53Backbone
from .necks.multigrid_fpn import MultiGridFPN
from .heads.multigrid_head import MultiGridHead

def build_multigriddet_resnet(input_shape: Tuple[int, int, int] = (416, 416, 3),
                              num_anchors_per_head: List[int] = [3, 3, 3],
                              num_classes: int = 80,
                              weights_path: Optional[str] = None,
                              **kwargs) -> Tuple[Model, int]:
    """
    Build MultiGridDet model with ResNet-style Darknet53 backbone.
    
    This is the modern implementation with ResNet-style blocks.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_anchors_per_head: Number of anchors per detection scale
        num_classes: Number of object classes
        weights_path: Path to pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, backbone_length)
    """
    # 1. Build backbone (it will create its own input)
    backbone_config = {
        'input_shape': input_shape,
        'weights_path': weights_path,
        'freeze_layers': 0
    }
    backbone = Darknet53Backbone(backbone_config)
    backbone_model = backbone.build_backbone(input_shape)
    
    # Get the input from the backbone model
    inputs = backbone_model.input
    
    # Extract feature maps at specific layers
    # f1: 13x13x1024 (output), f2: 26x26x512 (layer 152), f3: 52x52x256 (layer 92)
    f1 = backbone_model.output  # 13x13x1024
    f2 = backbone_model.layers[152].output  # 26x26x512  
    f3 = backbone_model.layers[92].output  # 52x52x256
    
    # 3. Build neck (FPN) - pass in order f1, f2, f3 (largest to smallest)
    neck_config = {
        'f1_channel_num': 512,  # From multigriddet_predictions
        'f2_channel_num': 256,  # From multigriddet_predictions
        'f3_channel_num': 128,  # From multigriddet_predictions
        'upsample_method': 'upsampling2d',
        'fusion_method': 'concatenate'
    }
    neck = MultiGridFPN(neck_config)
    enhanced_features = neck.build_neck([f1, f2, f3])
    
    # 4. Build head
    head_config = {
        'num_scales': 3,
        'anchors_per_scale': 3,
        'grid_assignment': '3x3',
        'use_iol': True,
        'feature_channels': [512, 256, 128]
    }
    head = MultiGridHead(head_config)
    
    # Create anchors for the head
    anchors = [
        np.array([[116, 90], [156, 198], [373, 326]]),  # Large objects
        np.array([[30, 61], [62, 45], [59, 119]]),     # Medium objects
        np.array([[10, 13], [16, 30], [33, 23]])       # Small objects
    ]
    
    outputs = head.build_head(enhanced_features, num_classes, anchors)
    
    # 5. Create model
    model = Model(inputs=inputs, outputs=outputs, name='multigriddet_resnet')
    
    # Load pretrained weights if provided
    if weights_path and tf.io.gfile.exists(weights_path):
        try:
            model.load_weights(weights_path, by_name=True)
            print(f'Loaded weights from {weights_path}')
        except Exception as e:
            print(f'Warning: Could not load weights from {weights_path}: {e}')
    
    return model, 185  # Darknet53 backbone length


def build_multigriddet_resnet_train(anchors: List[np.ndarray],
                                    num_classes: int = 80,
                                    input_shape: Tuple[int, int, int] = (416, 416, 3),
                                    weights_path: Optional[str] = None,
                                    backbone_weights_path: Optional[str] = None,
                                    freeze_level: int = 1,
                                    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                                    label_smoothing: float = 0.0,
                                    **kwargs) -> Tuple[Model, int]:
    """
    Build training model with loss.
    
    MultiGridDet ResNet implementation for training.
    
    Args:
        anchors: List of anchor arrays for each scale
        num_classes: Number of object classes
        input_shape: Input image shape
        weights_path: Path to pretrained full model weights
        backbone_weights_path: Path to pretrained backbone weights (e.g., darknet53.h5)
        freeze_level: Freeze level (0=unfreeze all, 1=freeze backbone, 2=freeze all but head)
        optimizer: Optimizer for training
        label_smoothing: Label smoothing factor
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (training_model, backbone_length)
    """
    # Calculate number of anchors per head
    num_feature_layers = len(anchors)
    num_anchors_per_head = [len(anchors[l]) for l in range(num_feature_layers)]
    
    print(f"num_anchors_per_head: {num_anchors_per_head}")
    
    # Build base model with backbone weights (if provided)
    # Backbone weights are loaded during model construction
    model, backbone_len = build_multigriddet_resnet(
        input_shape=input_shape,
        num_anchors_per_head=num_anchors_per_head,
        num_classes=num_classes,
        weights_path=backbone_weights_path,  # Pass backbone weights to load into backbone
        **kwargs
    )
    
    print(f'Create MultiGridDet ResNet model with {sum(num_anchors_per_head)} anchors and {num_classes} classes.')
    print(f'model layer number: {len(model.layers)}')
    
    # Load full model weights if provided (after backbone weights)
    # This allows full model weights to override backbone weights if both are provided
    if weights_path and tf.io.gfile.exists(weights_path):
        try:
            model.load_weights(weights_path, by_name=True)
            print(f'Loaded full model weights from {weights_path}.')
        except Exception as e:
            print(f'Warning: Could not load full model weights from {weights_path}: {e}')
    
    # Apply freezing based on freeze_level
    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers
        num = (backbone_len, len(model.layers) - 3)[freeze_level - 1]
        for i in range(num):
            model.layers[i].trainable = False
        print(f'Freeze the first {num} layers.')
    elif freeze_level == 0:
        # Unfreeze all layers
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        print('Unfreeze all layers.')
    
    # Create training model with loss
    # This would need to be implemented with the actual loss function
    # For now, return the base model
    training_model = model
    
    # Set optimizer
    if optimizer is None:
        optimizer = Adam(learning_rate=1e-3)
    
    # Compile model (loss function would be added here)
    # training_model.compile(optimizer=optimizer, loss=multigrid_loss, ...)
    
    return training_model, backbone_len


# Model configuration for reference
MULTIGRIDDET_RESNET_CONFIG = {
    'builder': build_multigriddet_resnet,
    'train_builder': build_multigriddet_resnet_train,
    'backbone_len': 185,
    'default_weights': 'weights/darknet53.h5',
    'input_shape': (416, 416, 3),
    'description': 'MultiGridDet with ResNet-style Darknet53 backbone'
}
