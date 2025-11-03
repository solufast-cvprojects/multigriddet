"""
MultiGridDet Models Module.

This module contains the model architecture components:
- Backbones: Feature extractors (Darknet53, MobileNet, EfficientNet, etc.)
- Necks: Feature fusion layers (FPN, PAN, etc.)
- Heads: Detection heads (MultiGridDet head)
- Registry: Model registration system for easy extension
"""

from .base import BaseModel, BaseBackbone, BaseNeck, BaseHead
from .registry import ModelRegistry, register_model, create_model, list_available_models
from . import __init_models__  # This will register all models

# Import specific models
from .backbones import *
from .heads import *
from .necks import *

# Import model builders
from .multigriddet_resnet import build_multigriddet_resnet, build_multigriddet_resnet_train
from .multigriddet_darknet import build_multigriddet_darknet, build_multigriddet_darknet_train

__all__ = [
    # Base classes
    "BaseModel",
    "BaseBackbone", 
    "BaseNeck",
    "BaseHead",
    
    # Registry
    "ModelRegistry",
    "register_model",
    "create_model",
    "list_available_models",
    
    # Model builders
    "build_multigriddet_resnet",
    "build_multigriddet_resnet_train",
    "build_multigriddet_darknet",
    "build_multigriddet_darknet_train",
]
