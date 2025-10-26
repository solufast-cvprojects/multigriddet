"""
MultiGridDet Heads Module.

This module contains detection head architectures.
The MultiGridDet head is the main innovation, implementing
dense prediction where multiple grid cells detect each object.

Available heads:
- MultiGridHead: The main MultiGridDet head with 3x3 grid assignment
- And more...

To add a new head:
1. Create a new file in this directory
2. Implement the BaseHead interface
3. Register it using @register_head decorator
"""

from .base_head import BaseHead

# Import specific heads
try:
    from .multigrid_head import MultiGridHead
    __all__ = ["BaseHead", "MultiGridHead"]
except ImportError:
    __all__ = ["BaseHead"]
