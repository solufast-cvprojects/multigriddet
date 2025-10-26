"""
MultiGridDet Losses Module.

This module contains loss functions for MultiGridDet training:
- MultiGridLoss: Main loss combining objectness, classification, and localization
- FocalLoss: Focal loss for handling class imbalance
- IoULosses: IoU-based losses (GIoU, DIoU, CIoU)

The MultiGridDet loss uses IoL (Intersection over Largest) anchor matching
instead of traditional IoU matching for better handling of extreme aspect ratios.
"""

from .multigrid_loss import MultiGridLoss
from .focal_loss import FocalLoss, SigmoidFocalLoss, SoftmaxFocalLoss
from .iou_losses import IoULoss, GIoULoss, DIoULoss, CIoULoss

__all__ = [
    "MultiGridLoss",
    "FocalLoss",
    "SigmoidFocalLoss", 
    "SoftmaxFocalLoss",
    "IoULoss",
    "GIoULoss",
    "DIoULoss",
    "CIoULoss",
]
