#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model builder for MultiGridDet.
Builds models from YAML configuration.
"""

import os
from typing import Dict, Any, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from ..models import build_multigriddet_darknet, build_multigriddet_resnet
from ..models.multigriddet_darknet import build_multigriddet_darknet_train
from ..models.multigriddet_resnet import build_multigriddet_resnet_train


def build_model_from_config(config: Dict[str, Any], for_training: bool = False, anchors: List = None) -> Model:
    """Build model from YAML configuration."""
    
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == 'preset':
        # Simple preset mode
        preset_config = model_config['preset']
        architecture = preset_config['architecture']
        num_classes = preset_config['num_classes']
        input_shape = tuple(preset_config['input_shape'])
        
        # Get loss option if for training
        loss_option = 2  # Default
        if for_training and 'training' in config:
            loss_option = config['training'].get('loss_option', 2)
        
        if architecture == 'multigriddet_darknet':
            if for_training:
                # Use training model with loss function
                result = build_multigriddet_darknet_train(
                    anchors=anchors or [[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]],  # Default COCO anchors
                    input_shape=input_shape,
                    num_classes=num_classes,
                    weights_path=None,  # We'll load weights separately
                    loss_option=loss_option
                )
            else:
                # Use inference model
                result = build_multigriddet_darknet(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    num_anchors_per_head=[3, 3, 3],
                    weights_path=None
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
                    weights_path=None,  # We'll load weights separately
                    loss_option=loss_option
                )
            else:
                # Use inference model
                result = build_multigriddet_resnet(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    num_anchors_per_head=[3, 3, 3],
                    weights_path=None
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


def build_model_for_training(config: Dict[str, Any], anchors: List = None) -> Model:
    """Build model specifically for training with loss function."""
    model = build_model_from_config(config, for_training=True, anchors=anchors)
    
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

