#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Darknet model builder - EXACT replica of original implementation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import os
import sys

# Import from the new structure
from .backbones.darknet import darknet53_body
from .heads.multigrid_head import multigriddet_predictions

# Import MultiGridLoss for training
from ..losses.multigrid_loss import MultiGridLoss


def build_multigriddet_darknet(input_shape: Tuple[int, int, int] = (416, 416, 3),
                               num_anchors_per_head: List[int] = [3, 3, 3],
                               num_classes: int = 80,
                               weights_path: Optional[str] = None,
                               **kwargs) -> Tuple[Model, int]:
    """
    Build MultiGridDet model with TRUE Darknet53 backbone.
    
    This is an EXACT replica of the original implementation.
    
    Architecture:
    - Backbone: Darknet53
    - Neck: Implicit FPN in predictions
    - Head: MultiGridDet multi-scale output
        
    Args:
        input_shape: Input image shape (height, width, channels)
        num_anchors_per_head: Number of anchors per detection scale (heads)
        num_classes: Number of object classes
        weights_path: Path to pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, backbone_length) - backbone_length is the number of layers in the backbone
    """
    # Build exactly as original implementation does
    inputs = Input(shape=input_shape)
    darknet = Model(inputs, darknet53_body(inputs))
    
    if weights_path and os.path.exists(weights_path):
        darknet.load_weights(weights_path, by_name=True)
        print(f'Loaded backbone weights from {weights_path}')
    
    # Extract feature maps
    f1 = darknet.output  # 13x13x1024
    f2 = darknet.layers[152].output  # 26x26x512
    f3 = darknet.layers[92].output  # 52x52x256
    
    # Feature channels (from original implementation)
    f1_channel_num = 512
    f2_channel_num = 256
    f3_channel_num = 128
    
    # Build predictions exactly as original implementation does
    y1, y2, y3 = multigriddet_predictions(
        (f1, f2, f3), 
        (f1_channel_num, f2_channel_num, f3_channel_num),
        num_anchors_per_head,
        num_classes
    )
    
    model = Model(inputs, [y1, y2, y3], name='multigriddet_darknet')
    
    return model, 185  # Darknet53 backbone length


def build_multigriddet_darknet_train(anchors: List[np.ndarray],
                                    num_classes: int = 80,
                                    input_shape: Tuple[int, int, int] = (416, 416, 3),
                                    weights_path: Optional[str] = None,
                                    backbone_weights_path: Optional[str] = None,
                                    freeze_level: int = 1,
                                    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                                    label_smoothing: float = 0.0,
                                    elim_grid_sense: bool = False,
                                    loss_option: int = 3,  # NEW: Add this parameter
                                    coord_scale: float = 1.0,
                                    object_scale: float = 1.0,
                                    no_object_scale: float = 1.0,
                                    class_scale: float = 1.0,
                                    anchor_scale: float = 1.0,
                                    **kwargs) -> Tuple[Model, int]:
    """
    Build training model for multigriddet_darknet.
    
    Args:
        anchors: List of anchor arrays for each scale (heads)
        num_classes: Number of object classes
        input_shape: Input image shape
        weights_path: Path to pretrained full model weights
        backbone_weights_path: Path to pretrained backbone weights (e.g., darknet53.h5)
        freeze_level: Freeze level (0=unfreeze all, 1=freeze backbone, 2=freeze all but head)
        optimizer: Optimizer for training
        label_smoothing: Label smoothing factor
        elim_grid_sense: Eliminate grid sense loss
        loss_option: Loss option (1=IoU, 2=GIOU, 3=CIoU)
        coord_scale: Scale for localization (coordinate) loss
        object_scale: Scale for objectness loss (positive cells)
        no_object_scale: Scale for objectness loss (negative cells)
        class_scale: Scale for classification loss
        anchor_scale: Scale for anchor prediction loss
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (training_model, backbone_length) - backbone_length is the number of layers in the backbone
    """
    # Calculate number of anchors per head
    num_feature_layers = len(anchors)
    num_anchors_per_head = [len(anchors[l]) for l in range(num_feature_layers)]
    
    print(f"num_anchors_per_head: {num_anchors_per_head}")
    
    # Build base model with backbone weights (if provided)
    # Backbone weights are loaded during model construction
    model, backbone_len = build_multigriddet_darknet(
        input_shape=input_shape,
        num_anchors_per_head=num_anchors_per_head,
        num_classes=num_classes,
        weights_path=backbone_weights_path,  # Pass backbone weights to load into backbone
        **kwargs
    )
    
    print(f'Create MultiGridDet Darknet model with {sum(num_anchors_per_head)} anchors and {num_classes} classes.')
    print(f'model layer number: {len(model.layers)}')
    
    # Load full model weights if provided (after backbone weights)
    # This allows full model weights to override backbone weights if both are provided
    if weights_path and os.path.exists(weights_path):
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
    
    # Create dummy y_true inputs for the loss function
    y_true = [Input(shape=(None, None, num_anchors_per_head[l] + num_classes + 5), name=f'y_true_{l}') for l in range(num_feature_layers)]
    
    # Create the MultiGridLoss function
    multigrid_loss_fn = MultiGridLoss(
        anchors=anchors,
        num_classes=num_classes,
        input_shape=input_shape[:2], # Pass only H, W
        label_smoothing=label_smoothing,
        elim_grid_sense=elim_grid_sense,
        loss_option=loss_option,  # Pass the option
        coord_scale=coord_scale,
        object_scale=object_scale,
        no_object_scale=no_object_scale,
        class_scale=class_scale,
        anchor_scale=anchor_scale
    )
    
    # Create loss layer using Lambda
    multigrid_loss = tf.keras.layers.Lambda(
        lambda x: multigrid_loss_fn(x[3:], x[:3]),  # y_true, y_pred
        output_shape=(None,),  # Loss function returns a scalar
        name='multigrid_loss'
    )([*model.outputs, *y_true]) # Pass model outputs and y_true to loss layer
    
    # Create the training model
    training_model = Model(inputs=[model.input] + y_true, outputs=multigrid_loss)
    
    # Set optimizer
    if optimizer is None:
        optimizer = Adam(learning_rate=1e-3)
    
    # Compile model with loss wrapper (same as original implementation)
    training_model.compile(
        optimizer=optimizer,
        loss={'multigrid_loss': lambda y_true, y_pred: y_pred}
    )
    
    return training_model, backbone_len


## Model configuration for reference
MULTIGRIDDET_DARKNET_CONFIG = {
    'builder': build_multigriddet_darknet,
    'train_builder': build_multigriddet_darknet_train,
    'backbone_len': 185,
    'default_weights': 'model5.h5',
    'input_shape': (416, 416, 3),
    'compatible_with': 'multigriddet_darknet',
    'description': 'TRUE MultiGridDet with Darknet53'
}
