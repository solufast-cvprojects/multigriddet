#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model builder for MultiGridDet.
Builds models from YAML configuration.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam, AdamW, SGD

from ..models import build_multigriddet_darknet, build_multigriddet_resnet
from ..models.multigriddet_darknet import build_multigriddet_darknet_train
from ..models.multigriddet_resnet import build_multigriddet_resnet_train
from ..utils.anchors import compute_class_weights


def create_optimizer_from_config(config: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    """
    Create optimizer from configuration.
    
    Supports:
    - Adam: Basic Adam optimizer
    - AdamW: Adam with decoupled weight decay (recommended for modern training)
    - SGD: Stochastic Gradient Descent with momentum
    
    Args:
        config: Configuration dictionary with 'optimizer' key
        
    Returns:
        Configured optimizer instance
    """
    optimizer_config = config.get('optimizer', {})
    opt_type = optimizer_config.get('type', 'adam').lower()
    learning_rate = optimizer_config.get('learning_rate', 0.001)
    
    if opt_type == 'adamw':
        # AdamW: Modern standard, uses weight_decay parameter
        weight_decay = optimizer_config.get('weight_decay', optimizer_config.get('decay', 0.0005))
        beta_1 = optimizer_config.get('beta_1', 0.9)
        beta_2 = optimizer_config.get('beta_2', 0.999)
        epsilon = optimizer_config.get('epsilon', 1e-7)
        optimizer = AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon
        )
        print(f"✓ Created AdamW optimizer: lr={learning_rate}, weight_decay={weight_decay}")
        
    elif opt_type == 'sgd':
        # SGD: Traditional optimizer with momentum
        momentum = optimizer_config.get('momentum', 0.937)
        decay = optimizer_config.get('decay', 0.0005)
        nesterov = optimizer_config.get('nesterov', False)
        optimizer = SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            decay=decay,
            nesterov=nesterov
        )
        print(f"✓ Created SGD optimizer: lr={learning_rate}, momentum={momentum}, decay={decay}")
        
    else:  # Default to Adam
        # Adam: Basic Adam optimizer (legacy support)
        beta_1 = optimizer_config.get('beta_1', 0.9)
        beta_2 = optimizer_config.get('beta_2', 0.999)
        epsilon = optimizer_config.get('epsilon', 1e-7)
        decay = optimizer_config.get('decay', 0.0)  # Note: Adam uses decay, not weight_decay
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            decay=decay
        )
        print(f"✓ Created Adam optimizer: lr={learning_rate}, decay={decay}")
    
    return optimizer


def build_model_from_config(config: Dict[str, Any], for_training: bool = False, anchors: List = None, 
                           weights_path: Optional[str] = None, backbone_weights_path: Optional[str] = None) -> Model:
    """Build model from YAML configuration."""
    
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == 'preset':
        # Simple preset mode
        preset_config = model_config['preset']
        architecture = preset_config['architecture']
        num_classes = preset_config['num_classes']
        input_shape = tuple(preset_config['input_shape'])
        
        # Get loss option and loss scales if for training
        loss_option = 2  # Default
        loss_scales = {}  # Default empty dict (will use defaults in MultiGridLoss)
        if for_training and 'training' in config:
            training_config = config['training']
            loss_option = training_config.get('loss_option', 2)
            
            # Extract loss scale parameters from config
            loss_config = training_config.get('loss', {})
            if loss_config:
                loss_scales = {
                    'coord_scale': loss_config.get('coord_scale', 1.0),
                    'object_scale': loss_config.get('object_scale', 1.0),
                    'no_object_scale': loss_config.get('no_object_scale', 1.0),
                    'class_scale': loss_config.get('class_scale', 1.0),
                    'anchor_scale': loss_config.get('anchor_scale', 1.0),
                }
            
            # Compute class weights for handling class imbalance
            class_weights_config = training_config.get('class_weights', None)
            if class_weights_config is not None:
                if isinstance(class_weights_config, str) and class_weights_config.lower() == 'auto':
                    # Auto-compute from training data
                    data_config = config.get('data', {})
                    train_annotation = data_config.get('train_annotation')
                    if train_annotation:
                        class_weights_method = training_config.get('class_weights_method', 'balanced')
                        class_weights = compute_class_weights(train_annotation, num_classes, method=class_weights_method)
                        loss_scales['class_weights'] = class_weights
                        print(f"[INFO] Computed class weights using '{class_weights_method}' method")
                        print(f"   Class weights range: [{np.min(class_weights):.3f}, {np.max(class_weights):.3f}]")
                        print(f"   Mean weight: {np.mean(class_weights):.3f}")
                elif isinstance(class_weights_config, (list, np.ndarray)):
                    # Manual class weights provided
                    class_weights = np.array(class_weights_config, dtype=np.float32)
                    if len(class_weights) != num_classes:
                        raise ValueError(f"class_weights length ({len(class_weights)}) must match num_classes ({num_classes})")
                    loss_scales['class_weights'] = class_weights
                    print(f"[INFO] Using manual class weights")
                else:
                    print(f"[WARNING] Invalid class_weights config, using equal weights")
            else:
                # No class weights specified, use equal weights (default)
                pass
        
        # Get optimizer if for training
        optimizer = None
        if for_training:
            optimizer = create_optimizer_from_config(config)
        
        if architecture == 'multigriddet_darknet':
            if for_training:
                # Use training model with loss function
                result = build_multigriddet_darknet_train(
                    anchors=anchors or [[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]],  # Default COCO anchors
                    input_shape=input_shape,
                    num_classes=num_classes,
                    weights_path=weights_path,
                    backbone_weights_path=backbone_weights_path,
                    optimizer=optimizer,
                    loss_option=loss_option,
                    **loss_scales  # Pass loss scale parameters
                )
            else:
                # Use inference model
                result = build_multigriddet_darknet(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    num_anchors_per_head=[3, 3, 3],
                    weights_path=backbone_weights_path or weights_path
                )
            # Handle both tuple (model, backbone_len) and single model return
            model = result[0] if isinstance(result, tuple) else result
        elif architecture == 'multigriddet_resnet':
            if for_training:
                # Use training model with loss function
                result = build_multigriddet_resnet_train(
                    anchors=anchors or [[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]],  # Default COCO anchors
                    input_shape=input_shape,
                    num_classes=num_classes,
                    weights_path=weights_path,
                    backbone_weights_path=backbone_weights_path,
                    optimizer=optimizer,
                    loss_option=loss_option,
                    **loss_scales  # Pass loss scale parameters
                )
            else:
                # Use inference model
                result = build_multigriddet_resnet(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    num_anchors_per_head=[3, 3, 3],
                    weights_path=backbone_weights_path or weights_path
                )
            # Handle both tuple (model, backbone_len) and single model return
            model = result[0] if isinstance(result, tuple) else result
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    elif model_type == 'custom':
        # Advanced custom mode - for future implementation
        raise NotImplementedError("Custom model composition not yet implemented. Use preset models.")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def build_model_for_training(config: Dict[str, Any], anchors: List = None, 
                            weights_path: Optional[str] = None, 
                            backbone_weights_path: Optional[str] = None) -> Model:
    """Build model specifically for training with loss function."""
    model = build_model_from_config(
        config, 
        for_training=True, 
        anchors=anchors,
        weights_path=weights_path,
        backbone_weights_path=backbone_weights_path
    )
    
    # Model is already compiled by build_multigriddet_darknet_train
    # No need to compile again - just return the model as-is
    return model


def build_model_for_inference(config: Dict[str, Any], weights_path: str = None) -> Model:
    """Build model for inference with loaded weights."""
    model = build_model_from_config(config, for_training=False)
    
    # Load weights
    if weights_path is None:
        weights_path = config.get('weights_path')
    
    if weights_path and os.path.exists(weights_path):
        try:
            # Try loading with by_name=True for legacy models
            model.load_weights(weights_path, by_name=True)
            print(f"✓ Loaded weights from: {weights_path}")
        except ValueError:
            # Fallback to standard loading for Keras 3.x models
            model.load_weights(weights_path)
            print(f"✓ Loaded weights from: {weights_path} (Keras 3.x format)")
    elif weights_path:
        print(f"⚠ Warning: Weights file not found: {weights_path}")
    else:
        print("⚠ Warning: No weights path specified")
    
    return model


def get_model_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get model information from config."""
    model_config = config['model']
    
    info = {
        'name': model_config.get('name', 'unknown'),
        'type': model_config.get('type', 'preset'),
        'architecture': None,
        'num_classes': None,
        'input_shape': None,
        'num_anchors_per_head': [3, 3, 3]  # Default
    }
    
    if model_config['type'] == 'preset':
        preset = model_config['preset']
        info.update({
            'architecture': preset['architecture'],
            'num_classes': preset['num_classes'],
            'input_shape': tuple(preset['input_shape'])
        })
    elif model_config['type'] == 'custom':
        custom = model_config['custom']
        info.update({
            'architecture': f"custom_{custom['backbone']['type']}_{custom['neck']['type']}_{custom['head']['type']}",
            'num_classes': custom['head'].get('num_classes', 80),
            'input_shape': tuple(custom.get('input_shape', [608, 608, 3]))
        })
    
    return info

