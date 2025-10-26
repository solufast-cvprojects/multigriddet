#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration helper functions for MultiGridDet.
"""

import yaml
import numpy as np
from typing import Dict, List, Any, Optional
from ..data.utils import load_classes, load_anchors
from ..data.target_encoding import MultiGridConfig
from ..data.augmentation import AugmentationConfig


def load_classes(classes_path: str) -> List[str]:
    """Load class names from file."""
    from ..data.utils import load_classes as _load_classes
    return _load_classes(classes_path)


def load_anchors(anchors_path: str) -> List[np.ndarray]:
    """Load anchors from file."""
    from ..data.utils import load_anchors as _load_anchors
    return _load_anchors(anchors_path)


def create_multigrid_config_from_yaml(yaml_config: Dict[str, Any]) -> MultiGridConfig:
    """
    Create MultiGridConfig from YAML configuration.
    
    Args:
        yaml_config: YAML configuration dictionary
        
    Returns:
        MultiGridConfig object
    """
    data_config = yaml_config.get('data', {})
    
    # Load classes and anchors
    classes = load_classes(data_config['classes_path'])
    anchors = load_anchors(data_config['anchors_path'])
    
    # Create MultiGridConfig
    config = MultiGridConfig(
        input_shape=tuple(data_config['input_shape']),
        num_classes=len(classes),
        anchors=anchors,
        max_boxes=data_config.get('max_boxes', 100),
        multi_anchor_assign=data_config.get('multi_anchor_assign', True)
    )
    
    return config


def create_augmentation_config_from_yaml(yaml_config: Dict[str, Any]) -> AugmentationConfig:
    """
    Create AugmentationConfig from YAML configuration.
    
    Args:
        yaml_config: YAML configuration dictionary
        
    Returns:
        AugmentationConfig object
    """
    data_config = yaml_config.get('data', {})
    augmentation_config = data_config.get('augmentation', {})
    
    # Create AugmentationConfig with defaults
    config = AugmentationConfig(
        horizontal_flip=augmentation_config.get('horizontal_flip', 0.5),
        vertical_flip=augmentation_config.get('vertical_flip', 0.0),
        rotation=augmentation_config.get('rotation', 0.0),
        brightness=augmentation_config.get('brightness', 0.0),
        contrast=augmentation_config.get('contrast', 0.0),
        saturation=augmentation_config.get('saturation', 0.0),
        hue=augmentation_config.get('hue', 0.0),
        blur=augmentation_config.get('blur', 0.0),
        noise=augmentation_config.get('noise', 0.0),
        cutout=augmentation_config.get('cutout', 0.0)
    )
    
    return config


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
