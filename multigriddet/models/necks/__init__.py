"""
MultiGridDet Necks Module.

This module contains neck architectures for feature fusion.
Necks are optional components that can be used to combine
features from different scales before feeding to the detection head.

Available necks:
- FPN: Feature Pyramid Network
- PAN: Path Aggregation Network
- And more...

To add a new neck:
1. Create a new file in this directory
2. Implement the BaseNeck interface
3. Register it using @register_neck decorator
"""

from .base_neck import BaseNeck

# Import specific necks
try:
    from .fpn import FPNNeck
    __all__ = ["BaseNeck", "FPNNeck"]
except ImportError:
    __all__ = ["BaseNeck"]
