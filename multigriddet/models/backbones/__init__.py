"""
MultiGridDet Backbones Module.

This module contains backbone architectures for feature extraction.
Each backbone implements the BaseBackbone interface and can be easily
swapped in the model configuration.

Available backbones:
- Darknet53: Original YOLO backbone
- MobileNet: Lightweight mobile-optimized backbone
- EfficientNet: Efficient scaling backbone
- And more...

To add a new backbone:
1. Create a new file in this directory
2. Implement the BaseBackbone interface
3. Register it using @register_backbone decorator
"""

from .base_backbone import BaseBackbone

# Import specific backbones
try:
    from .darknet import Darknet53Backbone
    __all__ = ["BaseBackbone", "Darknet53Backbone"]
except ImportError:
    __all__ = ["BaseBackbone"]
