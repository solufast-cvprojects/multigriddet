"""
MultiGridDet: A modern object detection framework with MultiGrid dense prediction.

MultiGridDet is a TensorFlow 2.17+ compatible object detection framework that implements
the MultiGrid dense prediction strategy, where multiple grid cells are used to detect
each object instead of the traditional single grid cell approach.

Key Features:
- MultiGrid dense prediction strategy (3x3 grid assignment per object)
- IoL (Intersection over Largest) anchor matching
- Modern TensorFlow 2.17+ implementation
- Extensible architecture for easy addition of new backbones and heads
- Advanced data augmentation pipeline
- Multiple backbone support (Darknet53, MobileNet, EfficientNet, etc.)

Example:
    >>> import multigriddet
    >>> from multigriddet.models import create_model
    >>> from multigriddet.data import create_dataset
    >>> 
    >>> # Create model
    >>> model = create_model("multigriddet_darknet53", num_classes=80)
    >>> 
    >>> # Create dataset
    >>> dataset = create_dataset("path/to/annotations.txt", batch_size=16)
    >>> 
    >>> # Train model
    >>> model.fit(dataset, epochs=100)
"""

__version__ = "1.0.0"
__author__ = "MultiGridDet Team"
__email__ = ""

# Core imports
from . import models
# Temporarily comment out data import to fix standalone functionality
# from . import data
from . import losses
from . import postprocess
from . import metrics
from . import utils

# Convenience imports
from .models import create_model, list_available_models
from .utils.visualization import draw_boxes

__all__ = [
    # Core modules
    "models",
    "data", 
    "losses",
    "postprocess",
    "metrics",
    "utils",
    
    # Convenience functions
    "create_model",
    "list_available_models",
    "draw_boxes",
]
