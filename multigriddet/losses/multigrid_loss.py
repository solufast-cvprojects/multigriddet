#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet loss function implementation.
Clean and correct implementation:
- Location loss: MSE (YOLOv3 style)
- Anchor loss: BCE
- Classification loss: BCE
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from typing import List, Tuple, Optional

from .focal_loss import SigmoidFocalLoss, SoftmaxFocalLoss
from .iou_losses import GIoULoss, DIoULoss, CIoULoss


class MultiGridLoss:
    """
    MultiGridDet loss function - clean and correct implementation.
    
    Loss components:
    - Location loss: MSE (simple, like YOLOv3)
    - Anchor loss: BCE (predicts which anchor fits best)
    - Classification loss: BCE (all classes)
    
    Supports multiple loss options:
    - Option 1: MSE location (no IoL weighting)
    - Option 2: MSE location with anchor prediction
    - Option 3: GIoU/DIoU/CIoU losses for better localization
    """
    
    __name__ = 'MultiGridLoss'
    
    def __init__(self, 
                 anchors: List[np.ndarray],
                 num_classes: int,
                 input_shape: Tuple[int, int] = (608, 608),
                 ignore_thresh: float = 0.5,
                 label_smoothing: float = 0.0,
                 elim_grid_sense: bool = False,
                 use_focal_loss: bool = False,  # Default to BCE for classification
                 use_softmax_loss: bool = False,
                 use_iol: bool = True,  # Keep for compatibility
                 use_giou_loss: bool = False,
                 use_diou_loss: bool = False,
                 use_ciou_loss: bool = False,
                 loss_option: int = 2,  # Option 2 recommended (MSE + anchor)
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 coord_scale: float = 1.0,
                 object_scale: float = 1.0,
                 no_object_scale: float = 1.0,
                 class_scale: float = 1.0,
                 anchor_scale: float = 1.0):
        """
        Initialize MultiGridDet loss.
        
        Args:
            anchors: List of anchor arrays for each scale
            num_classes: Number of object classes
            input_shape: Input image shape (height, width)
            ignore_thresh: IoU threshold for ignoring objectness loss (future use)
            label_smoothing: Label smoothing factor for classification
            elim_grid_sense: Whether to eliminate grid sensitivity (not implemented)
            use_focal_loss: Whether to use focal loss for classification (default: False, use BCE)
            use_softmax_loss: Whether to use softmax loss for classification
            use_iol: Whether to use IoL (kept for compatibility)
            use_giou_loss: Whether to use GIoU loss (Option 3)
            use_diou_loss: Whether to use DIoU loss (Option 3)
            use_ciou_loss: Whether to use CIoU loss (Option 3)
            loss_option: Loss option (1=MSE only, 2=MSE+anchor, 3=GIoU/DIoU/CIoU)
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            coord_scale: Coordinate loss scale
            object_scale: Object loss scale (for anchor predictions)
            no_object_scale: No-object loss scale (for negative anchor predictions)
            class_scale: Classification loss scale
            anchor_scale: Anchor prediction loss scale (used in Option 2)
        """
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.ignore_thresh = ignore_thresh
        self.label_smoothing = label_smoothing
        self.elim_grid_sense = elim_grid_sense
        self.use_focal_loss = use_focal_loss
        self.use_softmax_loss = use_softmax_loss
        self.use_iol = use_iol
        self.use_giou_loss = use_giou_loss
        self.use_diou_loss = use_diou_loss
        self.use_ciou_loss = use_ciou_loss
        self.loss_option = loss_option
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.coord_scale = coord_scale
        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.class_scale = class_scale
        self.anchor_scale = anchor_scale
        
        self.num_layers = len(anchors)
        self.eps = K.epsilon()
        
        # Initialize loss components only if needed
        if use_focal_loss:
            self.focal_loss = SigmoidFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if use_softmax_loss:
            self.softmax_focal_loss = SoftmaxFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if use_giou_loss and loss_option == 3:
            self.giou_loss = GIoULoss()
        if use_diou_loss and loss_option == 3:
            self.diou_loss = DIoULoss()
        if use_ciou_loss and loss_option == 3:
            self.ciou_loss = CIoULoss()
    
    def __call__(self, y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
        """Compute MultiGridDet loss."""
        return self.compute_loss(y_true, y_pred)
    
    def compute_loss(self, y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
        """
        Compute the complete MultiGridDet loss.
        
        Loss Formula (Option 2):
        L_total = coord_scale * L_location + 
                  object_scale * L_anchor + 
                  class_scale * L_classification
        
        Where:
        - L_location = MSE(true_xy - pred_xy) + MSE(true_wh - pred_wh) [only on object cells]
        - L_anchor = BCE(true_anchors, pred_anchors) [on object cells and ignore regions]
        - L_classification = BCE(true_class, pred_class) [only on object cells, all classes]
        
        Args:
            y_true: List of ground truth tensors for each scale
                   Shape: [batch, grid_h, grid_w, 5 + num_anchors + num_classes]
                   - [..., 0:2]: xy (center coordinates)
                   - [..., 2:4]: wh (width, height)
                   - [..., 4:5]: objectness (1.0 if object present)
                   - [..., 5:5+num_anchors]: anchor one-hot (which anchor fits)
                   - [..., 5+num_anchors:]: class one-hot
            y_pred: List of prediction tensors for each scale (same shape as y_true)
            
        Returns:
            Total loss tensor (scalar)
        """
        batch_size = K.shape(y_pred[0])[0]
        batch_size_f = K.cast(batch_size, 'float32')
        
        total_loss_location = 0
        total_loss_classification = 0
        total_loss_anchor = 0
        
        # Process each scale
        for layer_idx in range(self.num_layers):
            y_pred_layer = y_pred[layer_idx]
            y_true_layer = y_true[layer_idx]
            anchor_layer = self.anchors[layer_idx]
            num_anchors = len(anchor_layer)
            
            # Extract prediction components
            pred_xy = y_pred_layer[..., 0:2]  # [batch, grid_h, grid_w, 2]
            pred_wh = y_pred_layer[..., 2:4]  # [batch, grid_h, grid_w, 2]
            pred_anchors = y_pred_layer[..., 5:5+num_anchors]  # [batch, grid_h, grid_w, num_anchors]
            pred_class = y_pred_layer[..., 5+num_anchors:]  # [batch, grid_h, grid_w, num_classes]
            
            # Extract ground truth components
            true_xy = y_true_layer[..., 0:2]
            true_wh = y_true_layer[..., 2:4]
            true_obj = y_true_layer[..., 4:5]  # [batch, grid_h, grid_w, 1]
            true_anchors = y_true_layer[..., 5:5+num_anchors]
            true_class = y_true_layer[..., 5+num_anchors:]
            
            # Object mask: shape [batch, grid_h, grid_w, 1]
            object_mask = K.cast(true_obj > 0.5, 'float32')
            
            # Compute ignore mask (simplified for now, can be enhanced)
            ignore_mask = self._compute_ignore_mask(
                pred_xy, pred_wh, true_xy, true_wh,
                anchor_layer, object_mask
            )
            
            # ========== LOCALIZATION LOSS ==========
            if self.loss_option == 1:
                # Option 1: Simple MSE (YOLOv3 style)
                loc_loss = self._compute_mse_loss(
                    true_xy, true_wh, pred_xy, pred_wh, object_mask
                )
                loc_loss = loc_loss / batch_size_f
                
            elif self.loss_option == 2:
                # Option 2: MSE + Anchor prediction loss
                loc_loss = self._compute_mse_loss(
                    true_xy, true_wh, pred_xy, pred_wh, object_mask
                )
                loc_loss = loc_loss / batch_size_f
                
                # Anchor prediction loss (part of Option 2)
                anchor_loc_loss = self._compute_anchor_loss(
                    true_anchors, pred_anchors, object_mask, ignore_mask, batch_size_f
                )
                total_loss_anchor += self.anchor_scale * anchor_loc_loss
                
            else:  # Option 3
                # Option 3: GIoU/DIoU/CIoU losses
                if self.use_giou_loss:
                    loc_loss = self._compute_giou_loss(
                        true_xy, true_wh, pred_xy, pred_wh, object_mask
                    )
                elif self.use_diou_loss:
                    loc_loss = self._compute_diou_loss(
                        true_xy, true_wh, pred_xy, pred_wh, object_mask
                    )
                elif self.use_ciou_loss:
                    loc_loss = self._compute_ciou_loss(
                        true_xy, true_wh, pred_xy, pred_wh, object_mask
                    )
                else:
                    loc_loss = self._compute_mse_loss(
                        true_xy, true_wh, pred_xy, pred_wh, object_mask
                    )
                loc_loss = loc_loss / batch_size_f
            
            total_loss_location += loc_loss
            
            # ========== ANCHOR LOSS (Objectness) ==========
            # Only compute if NOT Option 2 (Option 2 computes it separately above)
            if self.loss_option != 2:
                anchor_loss = self._compute_anchor_loss(
                    true_anchors, pred_anchors, object_mask, ignore_mask, batch_size_f
                )
                total_loss_anchor += self.object_scale * anchor_loss
            
            # ========== CLASSIFICATION LOSS ==========
            # Use ALL classes, not just top-k!
            if self.use_focal_loss and not self.use_softmax_loss:
                class_loss = self._compute_focal_classification_loss(
                    true_class, pred_class, object_mask, batch_size_f
                )
            elif self.use_softmax_loss:
                class_loss = self._compute_softmax_classification_loss(
                    true_class, pred_class, object_mask, batch_size_f
                )
            else:
                # Standard BCE with label smoothing if enabled
                class_loss = self._compute_bce_classification_loss(
                    true_class, pred_class, object_mask, batch_size_f
                )
            
            total_loss_classification += class_loss
        
        # ========== COMBINE ALL LOSSES ==========
        total_loss = (
            self.coord_scale * total_loss_location +
            total_loss_anchor +
            self.class_scale * total_loss_classification
        )
        
        return total_loss
    
    def _compute_ignore_mask(self, pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                            true_xy: tf.Tensor, true_wh: tf.Tensor,
                            anchors: np.ndarray, object_mask: tf.Tensor) -> tf.Tensor:
        """
        Compute ignore mask for anchor loss.
        Ignore predictions with IoU > ignore_thresh but < object confidence.
        """
        # Simplified version: return zeros (can be enhanced later)
        # A full implementation would compute IoU with all GT boxes in the image
        ignore_mask = tf.zeros_like(object_mask)
        return ignore_mask
    
    def _compute_giou_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                          pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                          object_mask: tf.Tensor) -> tf.Tensor:
        """Compute GIoU loss for localization."""
        if hasattr(self, 'giou_loss'):
            return self.giou_loss.compute_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
        return self._compute_mse_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
    
    def _compute_diou_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                          pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                          object_mask: tf.Tensor) -> tf.Tensor:
        """Compute DIoU loss for localization."""
        if hasattr(self, 'diou_loss'):
            return self.diou_loss.compute_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
        return self._compute_mse_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
    
    def _compute_ciou_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                          pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                          object_mask: tf.Tensor) -> tf.Tensor:
        """Compute CIoU loss for localization."""
        if hasattr(self, 'ciou_loss'):
            return self.ciou_loss.compute_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
        return self._compute_mse_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
    
    def _compute_mse_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                          pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                          object_mask: tf.Tensor) -> tf.Tensor:
        """
        Compute MSE loss for localization (YOLOv3 style).
        
        Formula:
        L_xy = Σ object_mask × (true_xy - pred_xy)²
        L_wh = Σ object_mask × (true_wh - pred_wh)²
        L_location = L_xy + L_wh
        """
        # XY coordinate loss
        xy_diff = true_xy - pred_xy
        xy_loss = K.sum(K.square(xy_diff), axis=-1, keepdims=True)  # [batch, grid_h, grid_w, 1]
        
        # WH size loss
        wh_diff = true_wh - pred_wh
        wh_loss = K.sum(K.square(wh_diff), axis=-1, keepdims=True)  # [batch, grid_h, grid_w, 1]
        
        # Combine and apply object mask
        loc_loss = (xy_loss + wh_loss) * object_mask
        
        return K.sum(loc_loss)
    
    def _compute_anchor_loss(self, true_anchors: tf.Tensor, pred_anchors: tf.Tensor,
                            object_mask: tf.Tensor, ignore_mask: Optional[tf.Tensor],
                            batch_size_f: tf.Tensor) -> tf.Tensor:
        """
        Compute anchor prediction loss using BCE.
        
        Formula:
        L_anchor = Σ (object_mask + ignore_mask) × BCE(true_anchors, pred_anchors) / batch_size
        """
        if ignore_mask is None:
            ignore_mask = tf.zeros_like(object_mask)
        
        # Combined mask: penalize anchor predictions on object cells and ignore regions
        combined_mask = K.cast((object_mask + ignore_mask) > 0, 'float32')
        
        # BCE loss on anchor predictions
        anchor_loss = K.binary_crossentropy(true_anchors, pred_anchors, from_logits=True)
        
        # Apply mask: shape broadcasting [batch, grid_h, grid_w, 1] × [batch, grid_h, grid_w, num_anchors]
        anchor_loss = anchor_loss * combined_mask
        
        return K.sum(anchor_loss) / batch_size_f
    
    def _compute_focal_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                         object_mask: tf.Tensor, batch_size_f: tf.Tensor) -> tf.Tensor:
        """Compute focal loss for classification (all classes)."""
        class_loss = self.focal_loss.compute_loss(true_class, pred_class)
        # Broadcasting: [batch, grid_h, grid_w, 1] * [batch, grid_h, grid_w, num_classes]
        class_loss = class_loss * object_mask
        return K.sum(class_loss) / batch_size_f
    
    def _compute_softmax_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                            object_mask: tf.Tensor, batch_size_f: tf.Tensor) -> tf.Tensor:
        """Compute softmax focal loss for classification (all classes)."""
        class_loss = self.softmax_focal_loss.compute_loss(true_class, pred_class)
        # Broadcasting: [batch, grid_h, grid_w, 1] * [batch, grid_h, grid_w, num_classes]
        class_loss = class_loss * object_mask
        return K.sum(class_loss) / batch_size_f
    
    def _compute_bce_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                        object_mask: tf.Tensor, batch_size_f: tf.Tensor) -> tf.Tensor:
        """
        Compute BCE loss for classification (all classes, not top-k!).
        
        Formula:
        L_class = Σ object_mask × BCE(true_class, pred_class) / batch_size
        
        Applies to ALL classes, ensuring all classes receive gradients.
        """
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            true_class_smooth = true_class * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            true_class_smooth = true_class
        
        # BCE loss on all classes
        class_loss = K.binary_crossentropy(true_class_smooth, pred_class, from_logits=True)
        
        # Apply mask: only penalize classification on object cells
        # Broadcasting: [batch, grid_h, grid_w, 1] * [batch, grid_h, grid_w, num_classes]
        class_loss = class_loss * object_mask
        
        return K.sum(class_loss) / batch_size_f


def multigriddet_loss(args, anchors, num_classes, **kwargs):
    """
    MultiGridDet loss function wrapper for Keras compatibility.
    
    Args:
        args: List containing multigriddet_outputs and y_true
        anchors: List of anchor arrays
        num_classes: Number of classes
        **kwargs: Additional loss parameters
        
    Returns:
        Loss tensor
    """
    num_layers = len(anchors)
    multigriddet_outputs = args[:num_layers]
    y_true = args[num_layers:]
    
    loss_fn = MultiGridLoss(anchors=anchors, num_classes=num_classes, **kwargs)
    return loss_fn.compute_loss(y_true, multigriddet_outputs)
