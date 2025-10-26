#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility layer for MultiGridDet models.
Handles migration from denseyolo2_darknet to multigriddet_darknet.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import os

from .multigriddet_darknet import build_multigriddet_darknet, build_multigriddet_darknet_train


def get_model_for_inference(model_path: str, 
                           model_type: str = 'multigriddet_darknet',
                           input_shape: Tuple[int, int, int] = (416, 416, 3),
                           num_classes: int = 80,
                           anchors: Optional[List[np.ndarray]] = None) -> Model:
    """
    Load pre-trained model for inference.
    Handles both old denseyolo2_darknet and new multigriddet_darknet.
    
    Args:
        model_path: Path to the model weights file
        model_type: Type of model to load
        input_shape: Input image shape
        num_classes: Number of object classes
        anchors: Anchor arrays for each scale
        
    Returns:
        Loaded model ready for inference
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Default anchors if not provided (COCO anchors)
    if anchors is None:
        anchors = [
            np.array([[116, 90], [156, 198], [373, 326]]),  # Large objects (13x13)
            np.array([[30, 61], [62, 45], [59, 119]]),     # Medium objects (26x26)
            np.array([[10, 13], [16, 30], [33, 23]])       # Small objects (52x52)
        ]
    
    num_anchors_per_head = [len(anchors[l]) for l in range(len(anchors))]
    
    if model_type == 'multigriddet_darknet':
        # Use new structured model
        model, _ = build_multigriddet_darknet(
            input_shape=input_shape,
            num_anchors_per_head=num_anchors_per_head,
            num_classes=num_classes,
            weights_path=model_path
        )
        return model
    
    elif model_type == 'denseyolo2_darknet':
        # Fallback to old model for compatibility
        try:
            # Try to load with old denseyolo2 structure
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            from denseyolo2.model import get_denseyolo2_model
            
            model, _ = get_denseyolo2_model(
                model_type='denseyolo2_darknet',
                num_feature_layers=len(anchors),
                num_anchors_per_head=num_anchors_per_head,
                num_classes=num_classes,
                input_shape=input_shape,
                model_pruning=False
            )
            
            # Load weights
            model.load_weights(model_path, by_name=True)
            return model
            
        except ImportError:
            raise ImportError("Old denseyolo2 model not available. Use multigriddet_darknet instead.")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_training_model_for_compatibility(model_type: str = 'multigriddet_darknet',
                                       anchors: Optional[List[np.ndarray]] = None,
                                       num_classes: int = 80,
                                       input_shape: Tuple[int, int, int] = (416, 416, 3),
                                       weights_path: Optional[str] = None,
                                       freeze_level: int = 1,
                                       **kwargs) -> Tuple[Model, int]:
    """
    Get training model with backward compatibility.
    
    Args:
        model_type: Type of model to create
        anchors: Anchor arrays for each scale
        num_classes: Number of object classes
        input_shape: Input image shape
        weights_path: Path to pretrained weights
        freeze_level: Freeze level for training
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (training_model, backbone_length)
    """
    if anchors is None:
        # Default COCO anchors
        anchors = [
            np.array([[116, 90], [156, 198], [373, 326]]),  # Large objects
            np.array([[30, 61], [62, 45], [59, 119]]),     # Medium objects
            np.array([[10, 13], [16, 30], [33, 23]])       # Small objects
        ]
    
    if model_type == 'multigriddet_darknet':
        return build_multigriddet_darknet_train(
            anchors=anchors,
            num_classes=num_classes,
            input_shape=input_shape,
            weights_path=weights_path,
            freeze_level=freeze_level,
            **kwargs
        )
    
    elif model_type == 'denseyolo2_darknet':
        # Fallback to old training model
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            from denseyolo2.model import get_denseyolo2_train_model
            
            return get_denseyolo2_train_model(
                model_type='denseyolo2_darknet',
                anchors=anchors,
                num_classes=num_classes,
                weights_path=weights_path,
                freeze_level=freeze_level,
                **kwargs
            )
            
        except ImportError:
            raise ImportError("Old denseyolo2 training model not available. Use multigriddet_darknet instead.")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def migrate_weights(old_model_path: str, 
                   new_model_path: str,
                   model_type: str = 'multigriddet_darknet') -> bool:
    """
    Migrate weights from old denseyolo2_darknet to new multigriddet_darknet.
    
    Args:
        old_model_path: Path to old model weights
        new_model_path: Path to save new model weights
        model_type: Type of new model
        
    Returns:
        True if migration successful, False otherwise
    """
    try:
        # Load old model
        old_model = get_model_for_inference(old_model_path, 'denseyolo2_darknet')
        
        # Create new model with same architecture
        new_model = get_model_for_inference(old_model_path, model_type)
        
        # Save new model
        new_model.save_weights(new_model_path)
        
        print(f"Successfully migrated weights from {old_model_path} to {new_model_path}")
        return True
        
    except Exception as e:
        print(f"Failed to migrate weights: {e}")
        return False


def check_model_compatibility(model_path: str, model_type: str = 'multigriddet_darknet') -> Dict[str, Any]:
    """
    Check if a model is compatible with the new structure.
    
    Args:
        model_path: Path to model weights
        model_type: Expected model type
        
    Returns:
        Dictionary with compatibility information
    """
    result = {
        'compatible': False,
        'model_type': model_type,
        'weights_exist': os.path.exists(model_path),
        'can_load': False,
        'layer_count': 0,
        'input_shape': None,
        'output_shapes': None,
        'error': None
    }
    
    if not result['weights_exist']:
        result['error'] = f"Weights file not found: {model_path}"
        return result
    
    try:
        model = get_model_for_inference(model_path, model_type)
        result['can_load'] = True
        result['layer_count'] = len(model.layers)
        result['input_shape'] = model.input_shape
        result['output_shapes'] = [output.shape for output in model.outputs]
        result['compatible'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result
