#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model initialization and registration.
This file handles the registration of models to avoid circular imports.
"""

from .registry import get_registry

def register_models():
    """Register all available models."""
    registry = get_registry()
    
    # Register multigriddet_resnet model
    from .multigriddet_resnet import build_multigriddet_resnet, build_multigriddet_resnet_train, MULTIGRIDDET_RESNET_CONFIG
    
    registry._models['multigriddet_resnet'] = {
        'class': None,  # This is a function-based model, not a class
        'config': MULTIGRIDDET_RESNET_CONFIG,
        'builder': build_multigriddet_resnet,
        'train_builder': build_multigriddet_resnet_train
    }
    
    # Register TRUE multigriddet_darknet model
    from .multigriddet_darknet import build_multigriddet_darknet, build_multigriddet_darknet_train, MULTIGRIDDET_DARKNET_CONFIG
    
    registry._models['multigriddet_darknet'] = {
        'class': None,  # This is a function-based model, not a class
        'config': MULTIGRIDDET_DARKNET_CONFIG,
        'builder': build_multigriddet_darknet,
        'train_builder': build_multigriddet_darknet_train
    }

# Auto-register models when this module is imported
register_models()
