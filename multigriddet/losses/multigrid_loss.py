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
                 anchor_scale: float = 1.0,
                 class_weights: Optional[np.ndarray] = None,
                 loss_normalization: Optional[List[str]] = None,
                 use_iou_aware_objectness: bool = False,
                 iou_objectness_power: float = 1.0,
                 iou_objectness_ratio: float = 1.0,
                 trainable_nms_weight: float = 0.0,
                 trainable_nms_power: float = 2.0,
                 use_consensus_loss: bool = False,
                 consensus_kernel_size: int = 3,
                 consensus_iou_power: float = 1.5,
                 consensus_min_iou: float = 1e-3,
                 consensus_coord_scale: float = 0.5,
                 consensus_obj_scale: float = 0.5,
                 consensus_class_scale: float = 0.3,
                 consensus_stop_gradient: bool = True,
                 consensus_center_tolerance: float = 1e-4):
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
            class_weights: Optional array of class weights for handling class imbalance (shape: [num_classes])
                          If None, all classes have equal weight (1.0)
            loss_normalization: List of normalization methods to apply. Options:
                              - "positives": Normalize by number of positive cells (varies per batch)
                              - "batch": Normalize by batch size (default, recommended)
                              - "grid": Normalize by batch_size * grid_h * grid_w
                              Default: ["batch"] for consistent loss scaling
            use_iou_aware_objectness: If True, positives use IoU-aware targets for objectness (acts like trainable NMS)
            iou_objectness_power: Exponent applied to IoU target (higher = harsher suppression of low-IoU boxes)
            iou_objectness_ratio: Blend factor between IoU target and binary (1.0 = pure IoU, 0.0 = legacy target)
            trainable_nms_weight: Additional weight applied to ignore regions to push duplicates down
            trainable_nms_power: Exponent for scaling the ignore penalty by IoU (higher = focus on high-overlap dupes)
            use_consensus_loss: Enables variance-based consensus regularization across MultiGrid assignments
            consensus_kernel_size: Spatial window (odd) used to gather cooperative cells (3 = 3x3 neighborhood)
            consensus_iou_power: Exponent applied to IoU weights when computing consensus distributions
            consensus_min_iou: Floor value to keep weights stable when IoU is tiny early in training
            consensus_coord_scale: Weight applied to the consensus coordinate variance term
            consensus_obj_scale: Weight applied to the consensus objectness variance term
            consensus_class_scale: Weight applied to the consensus classification variance term
            consensus_stop_gradient: If True, stop gradients through consensus targets for stability
            consensus_center_tolerance: Tolerance when matching cells to the same GT center in grid space
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
        self.use_iou_aware_objectness = use_iou_aware_objectness
        self.iou_objectness_power = iou_objectness_power
        # Clamp ratio between 0 and 1 to avoid invalid blends
        self.iou_objectness_ratio = float(np.clip(iou_objectness_ratio, 0.0, 1.0))
        self.trainable_nms_weight = float(trainable_nms_weight)
        self.trainable_nms_power = trainable_nms_power
        self.use_consensus_loss = use_consensus_loss
        self.consensus_kernel_size = consensus_kernel_size
        self.consensus_iou_power = consensus_iou_power
        self.consensus_min_iou = consensus_min_iou
        self.consensus_coord_scale = consensus_coord_scale
        self.consensus_obj_scale = consensus_obj_scale
        self.consensus_class_scale = consensus_class_scale
        self.consensus_stop_gradient = consensus_stop_gradient
        self.consensus_center_tolerance = consensus_center_tolerance
        
        # Loss normalization configuration
        if loss_normalization is None:
            loss_normalization = ["batch"]  # Default to batch normalization
        if not isinstance(loss_normalization, list):
            loss_normalization = [loss_normalization]
        self.loss_normalization = loss_normalization
        
        # Handle class weights for class imbalance
        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(f"class_weights length ({len(class_weights)}) must match num_classes ({num_classes})")
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)  # [num_classes]
        else:
            # Default: equal weights for all classes
            self.class_weights = tf.ones(num_classes, dtype=tf.float32)  # [num_classes]
        
        self.num_layers = len(anchors)
        self.eps = K.epsilon()
        if self.use_consensus_loss:
            if self.consensus_kernel_size % 2 == 0 or self.consensus_kernel_size < 1:
                raise ValueError("consensus_kernel_size must be an odd positive integer")
        
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
    
    def _get_normalization_factor(self, batch_size: tf.Tensor, grid_shape: Tuple[tf.Tensor, tf.Tensor],
                                  object_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute normalization factor based on loss_normalization configuration.
        
        Args:
            batch_size: Batch size tensor
            grid_shape: Tuple of (grid_h, grid_w) tensors
            object_mask: Optional object mask for "positives" normalization [batch, grid_h, grid_w, 1]
            
        Returns:
            Normalization factor tensor (scalar)
        """
        batch_size_f = K.cast(batch_size, 'float32')
        grid_h, grid_w = grid_shape
        # Cast grid dimensions to float32 before multiplication to avoid type mismatch
        grid_h_f = K.cast(grid_h, 'float32')
        grid_w_f = K.cast(grid_w, 'float32')
        total_grid_cells = batch_size_f * grid_h_f * grid_w_f
        
        norm_factor = 1.0
        for norm_type in self.loss_normalization:
            if norm_type == "positives":
                if object_mask is not None:
                    num_positives = K.sum(object_mask)
                    norm_factor = norm_factor * K.maximum(num_positives, 1.0)
                else:
                    # Fallback to batch if object_mask not provided
                    norm_factor = norm_factor * batch_size_f
            elif norm_type == "batch":
                norm_factor = norm_factor * batch_size_f
            elif norm_type == "grid":
                norm_factor = norm_factor * total_grid_cells
            else:
                # Unknown normalization type, skip
                pass
        
        return K.maximum(norm_factor, 1.0)  # Avoid division by zero
    
    def compute_loss(self, y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
        """
        Compute the complete MultiGridDet loss.
        
        Loss Formula (Option 2):
        L_total = coord_scale * L_location + 
                  object_scale * L_objectness +
                  anchor_scale * L_anchor + 
                  class_scale * L_classification
        
        Where:
        - L_location = MSE(true_xy - pred_xy) + MSE(true_wh - pred_wh) [only on object cells]
        - L_objectness = BCE(true_obj, sigmoid(pred_obj)) [on all cells]
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
        total_loss_objectness = 0
        total_loss_classification = 0
        total_loss_anchor = 0
        total_consensus_coord = 0.0
        total_consensus_objectness = 0.0
        total_consensus_class = 0.0
        
        # Process each scale
        for layer_idx in range(self.num_layers):
            y_pred_layer = y_pred[layer_idx]
            y_true_layer = y_true[layer_idx]
            anchor_layer = self.anchors[layer_idx]
            num_anchors = len(anchor_layer)
            
            # Debug: Assert input shapes
            tf.debugging.assert_rank(y_pred_layer, 4, message=f"y_pred_layer[{layer_idx}] must be rank 4")
            tf.debugging.assert_rank(y_true_layer, 4, message=f"y_true_layer[{layer_idx}] must be rank 4")
            
            # Extract prediction components
            pred_xy = y_pred_layer[..., 0:2]  # [batch, grid_h, grid_w, 2]
            pred_wh = y_pred_layer[..., 2:4]  # [batch, grid_h, grid_w, 2]
            pred_obj = y_pred_layer[..., 4:5]  # [batch, grid_h, grid_w, 1] - raw logits
            pred_anchors = y_pred_layer[..., 5:5+num_anchors]  # [batch, grid_h, grid_w, num_anchors]
            pred_class = y_pred_layer[..., 5+num_anchors:]  # [batch, grid_h, grid_w, num_classes]
            
            # Extract ground truth components
            true_xy = y_true_layer[..., 0:2]
            true_wh = y_true_layer[..., 2:4]
            true_obj = y_true_layer[..., 4:5]  # [batch, grid_h, grid_w, 1]
            true_anchors = y_true_layer[..., 5:5+num_anchors]
            true_class = y_true_layer[..., 5+num_anchors:]
            
            # Debug: Assert extracted component shapes
            tf.debugging.assert_equal(tf.shape(pred_xy)[-1], 2, message=f"pred_xy last dim must be 2")
            tf.debugging.assert_equal(tf.shape(pred_wh)[-1], 2, message=f"pred_wh last dim must be 2")
            tf.debugging.assert_equal(tf.shape(pred_anchors)[-1], num_anchors, message=f"pred_anchors last dim must be {num_anchors}")
            tf.debugging.assert_equal(tf.shape(true_anchors)[-1], num_anchors, message=f"true_anchors last dim must be {num_anchors}")
            
            # Object mask: shape [batch, grid_h, grid_w, 1]
            object_mask = K.cast(true_obj > 0.5, 'float32')
            
            # Get grid shape from tensor (use dynamic shape for graph mode compatibility)
            grid_shape = (K.shape(y_pred_layer)[1], K.shape(y_pred_layer)[2])
            
            # Debug: Assert grid shape is valid
            tf.debugging.assert_positive(grid_shape[0], message=f"grid_h must be positive")
            tf.debugging.assert_positive(grid_shape[1], message=f"grid_w must be positive")
            
            # Compute ignore mask for all loss options
            # Ignore mask prevents objectness/anchor loss on cells with high IoU but not assigned as positive
            ignore_mask, assigned_anchor_iou, max_iou_map = self._compute_ignore_mask(
                pred_xy, pred_wh, true_xy, true_wh,
                anchor_layer, object_mask, y_true_layer, grid_shape
            )
            # Debug: Assert ignore mask shape matches object mask
            tf.debugging.assert_equal(tf.shape(ignore_mask), tf.shape(object_mask), 
                                    message=f"Ignore mask shape mismatch at layer {layer_idx}")
            
            # ========== LOCALIZATION LOSS ==========
            # Get grid shape for normalization
            grid_shape = (K.shape(y_pred_layer)[1], K.shape(y_pred_layer)[2])
            # Compute normalization factor based on configuration
            loc_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
            
            if self.loss_option == 1:
                # Option 1: Simple MSE (YOLOv3 style)
                loc_loss = self._compute_mse_loss(
                    true_xy, true_wh, pred_xy, pred_wh, object_mask
                )
                loc_loss = loc_loss / loc_norm_factor
                
            elif self.loss_option == 2:
                # Option 2: MSE + Anchor prediction loss
                loc_loss = self._compute_mse_loss(
                    true_xy, true_wh, pred_xy, pred_wh, object_mask
                )
                loc_loss = loc_loss / loc_norm_factor
                
                # Anchor prediction loss (part of Option 2)
                anchor_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
                anchor_loc_loss = self._compute_anchor_loss(
                    true_anchors, pred_anchors, object_mask, ignore_mask, anchor_norm_factor
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
                loc_loss = loc_loss / loc_norm_factor
            
            total_loss_location += loc_loss
            
            # ========== OBJECTNESS LOSS ==========
            # Objectness loss: predicts if object exists in grid cell (sigmoid BCE)
            obj_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
            obj_loss = self._compute_objectness_loss(
                true_obj, pred_obj, object_mask, ignore_mask, obj_norm_factor,
                positive_iou_map=assigned_anchor_iou,
                max_iou_map=max_iou_map
            )
            total_loss_objectness += obj_loss
            
            # ========== ANCHOR LOSS ==========
            # Only compute if NOT Option 2 (Option 2 computes it separately above)
            if self.loss_option != 2:
                anchor_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
                anchor_loss = self._compute_anchor_loss(
                    true_anchors, pred_anchors, object_mask, ignore_mask, anchor_norm_factor
                )
                total_loss_anchor += self.anchor_scale * anchor_loss
            
            # ========== CLASSIFICATION LOSS ==========
            # Use ALL classes, not just top-k!
            # Compute normalization factor based on configuration
            class_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
            if self.use_focal_loss and not self.use_softmax_loss:
                class_loss = self._compute_focal_classification_loss(
                    true_class, pred_class, object_mask, class_norm_factor
                )
            elif self.use_softmax_loss:
                class_loss = self._compute_softmax_classification_loss(
                    true_class, pred_class, object_mask, class_norm_factor
                )
            else:
                # Standard BCE with label smoothing if enabled
                class_loss = self._compute_bce_classification_loss(
                    true_class, pred_class, object_mask, class_norm_factor
                )
            
            total_loss_classification += class_loss
            
            if self.use_consensus_loss:
                consensus_coord_loss, consensus_obj_loss, consensus_class_loss = (
                    self._compute_variance_consensus_loss(
                        pred_xy=pred_xy,
                        pred_wh=pred_wh,
                        pred_obj=pred_obj,
                        pred_class=pred_class,
                        true_xy=true_xy,
                        object_mask=object_mask,
                        assigned_anchor_iou=assigned_anchor_iou,
                        grid_shape=grid_shape
                    )
                )
                total_consensus_coord += consensus_coord_loss
                total_consensus_objectness += consensus_obj_loss
                total_consensus_class += consensus_class_loss
        
        # ========== COMBINE ALL LOSSES ==========
        total_loss = (
            self.coord_scale * total_loss_location +
            self.object_scale * total_loss_objectness +
            self.anchor_scale * total_loss_anchor +
            self.class_scale * total_loss_classification
        )
        if self.use_consensus_loss:
            total_loss += (
                self.consensus_coord_scale * total_consensus_coord +
                self.consensus_obj_scale * total_consensus_objectness +
                self.consensus_class_scale * total_consensus_class
            )
        
        return total_loss
    
    def _compute_iou_batch(self, boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
        """
        Compute IoU between two sets of boxes using vectorized operations.
        
        Args:
            boxes1: First set of boxes in center format [..., 4] (cx, cy, w, h)
                   Shape: [batch, grid_h, grid_w, 4] or [batch, max_boxes, 4]
            boxes2: Second set of boxes in center format [..., 4] (cx, cy, w, h)
                   Shape: [batch, max_boxes, 4]
        
        Returns:
            IoU matrix: [batch, grid_h, grid_w, max_boxes] or [batch, max_boxes, max_boxes]
        """
        # Convert to corner format
        boxes1_mins = boxes1[..., 0:2] - boxes1[..., 2:4] / 2.0
        boxes1_maxes = boxes1[..., 0:2] + boxes1[..., 2:4] / 2.0
        boxes2_mins = boxes2[..., 0:2] - boxes2[..., 2:4] / 2.0
        boxes2_maxes = boxes2[..., 0:2] + boxes2[..., 2:4] / 2.0
        
        # Expand dimensions for broadcasting
        # boxes1: [batch, grid_h, grid_w, 1, 4] or [batch, max_boxes, 1, 4]
        # boxes2: [batch, 1, 1, max_boxes, 4] or [batch, 1, max_boxes, 4]
        boxes1_mins_expanded = tf.expand_dims(boxes1_mins, axis=-2)  # [..., 1, 2]
        boxes1_maxes_expanded = tf.expand_dims(boxes1_maxes, axis=-2)  # [..., 1, 2]
        boxes2_mins_expanded = tf.expand_dims(boxes2_mins, axis=-3)  # [..., 1, ..., 2]
        boxes2_maxes_expanded = tf.expand_dims(boxes2_maxes, axis=-3)  # [..., 1, ..., 2]
        
        # Compute intersection
        intersect_mins = K.maximum(boxes1_mins_expanded, boxes2_mins_expanded)
        intersect_maxes = K.minimum(boxes1_maxes_expanded, boxes2_maxes_expanded)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        # Compute areas
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # [batch, grid_h, grid_w] or [batch, max_boxes]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]  # [batch, max_boxes]
        
        # Expand for broadcasting
        boxes1_area_expanded = tf.expand_dims(boxes1_area, axis=-1)  # [..., 1]
        boxes2_area_expanded = tf.expand_dims(boxes2_area, axis=-3)  # [..., 1, ...]
        
        # Compute union
        union_area = boxes1_area_expanded + boxes2_area_expanded - intersect_area
        
        # Compute IoU
        iou = intersect_area / (union_area + self.eps)
        
        return iou
    
    def _compute_ignore_mask(self, pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                            true_xy: tf.Tensor, true_wh: tf.Tensor,
                            anchors: np.ndarray, object_mask: tf.Tensor,
                            y_true_layer: tf.Tensor, grid_shape: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute ignore mask for objectness and anchor loss.
        Ignore predictions with IoU > ignore_thresh but not assigned as positive.
        
        This implementation computes IoU between predicted boxes (for each anchor) and
        all ground truth boxes in the image, marking cells as ignore if they have high
        IoU with a GT box but are not assigned as positive.
        
        CRITICAL: Applies tanh(0.15*x) + sigmoid(0.15*x) activation to pred_xy before
        computing IoU. This matches the inference pipeline and ensures ignore masks are
        computed using the same decoded coordinates that the model actually predicts.
        Without this activation, IoU uses raw logits which don't match decoded predictions,
        leading to incorrect ignore masks and loss divergence during fine-tuning.
        
        Args:
            pred_xy: Predicted center coordinates [batch, grid_h, grid_w, 2] (grid-relative, raw logits)
            pred_wh: Predicted width/height [batch, grid_h, grid_w, 2] (log-space: log(wh/anchor))
            true_xy: True center coordinates [batch, grid_h, grid_w, 2] (grid-relative, already activated)
            true_wh: True width/height [batch, grid_h, grid_w, 2] (log-space: log(wh/anchor))
            anchors: Anchor array for this layer [num_anchors, 2]
            object_mask: Object presence mask [batch, grid_h, grid_w, 1]
            y_true_layer: Full ground truth tensor [batch, grid_h, grid_w, 5+num_anchors+num_classes]
            grid_shape: Grid shape (grid_h, grid_w)
        
        Returns:
            Tuple:
              - ignore_mask [batch, grid_h, grid_w, 1]: 1.0 where cells should be ignored
              - assigned_anchor_iou [batch, grid_h, grid_w, 1]: IoU of the assigned anchor (0 for negatives)
              - max_iou_map [batch, grid_h, grid_w, 1]: Max IoU over anchors (used for soft suppression)
        """
        batch_size = K.shape(pred_xy)[0]
        grid_h, grid_w = grid_shape
        num_anchors = len(anchors)
        
        # Debug: Assert inputs
        tf.debugging.assert_rank(pred_xy, 4, message="pred_xy must be rank 4")
        tf.debugging.assert_rank(pred_wh, 4, message="pred_wh must be rank 4")
        tf.debugging.assert_rank(true_xy, 4, message="true_xy must be rank 4")
        tf.debugging.assert_rank(true_wh, 4, message="true_wh must be rank 4")
        tf.debugging.assert_rank(object_mask, 4, message="object_mask must be rank 4")
        tf.debugging.assert_positive(grid_h, message="grid_h must be positive")
        tf.debugging.assert_positive(grid_w, message="grid_w must be positive")
        tf.debugging.assert_greater(num_anchors, 0, message="num_anchors must be > 0")
        
        # Get grid coordinates for converting relative to absolute
        grid_h_int = tf.cast(grid_h, tf.int32)
        grid_w_int = tf.cast(grid_w, tf.int32)
        grid_y = tf.range(grid_h_int, dtype=tf.float32)
        grid_x = tf.range(grid_w_int, dtype=tf.float32)
        # Use meshgrid to create coordinate grids
        grid_x_mesh, grid_y_mesh = tf.meshgrid(grid_x, grid_y, indexing='ij')
        # Stack to create [grid_h, grid_w, 2] coordinates
        grid_coords = tf.stack([grid_x_mesh, grid_y_mesh], axis=-1)  # [grid_h, grid_w, 2]
        grid_coords = tf.expand_dims(grid_coords, axis=0)  # [1, grid_h, grid_w, 2]
        
        # Scale factors to convert grid coordinates to image coordinates
        scale_h = tf.cast(self.input_shape[0], tf.float32) / tf.cast(grid_h, tf.float32)
        scale_w = tf.cast(self.input_shape[1], tf.float32) / tf.cast(grid_w, tf.float32)
        # Create scale as a tensor that can be broadcast
        scale = tf.stack([scale_w, scale_h])  # [2]
        scale = tf.reshape(scale, [1, 1, 1, 2])  # [1, 1, 1, 2] for broadcasting
        
        # Convert true boxes from grid-relative to absolute image coordinates
        # true_xy is relative to grid cell, add grid position
        true_xy_abs = (true_xy + grid_coords) * scale  # [batch, grid_h, grid_w, 2]
        # true_wh is in log-space: log(wh/anchor), need to decode
        # For GT, we need to find which anchor was used - use the anchor with highest value in true_anchors
        true_anchors = y_true_layer[..., 5:5+num_anchors]  # [batch, grid_h, grid_w, num_anchors]
        true_anchor_idx = tf.argmax(true_anchors, axis=-1, output_type=tf.int32)  # [batch, grid_h, grid_w]
        anchors_tf = tf.constant(anchors, dtype=tf.float32)  # [num_anchors, 2]
        # Gather selected anchors: [batch, grid_h, grid_w, 2]
        # Use one-hot encoding to select anchors
        true_anchor_one_hot = tf.one_hot(true_anchor_idx, depth=num_anchors, dtype=tf.float32)  # [batch, grid_h, grid_w, num_anchors]
        selected_anchors = tf.tensordot(true_anchor_one_hot, anchors_tf, axes=[[-1], [0]])  # [batch, grid_h, grid_w, 2]
        true_wh_abs = tf.exp(true_wh) * selected_anchors * scale  # [batch, grid_h, grid_w, 2]
        
        # Collect all valid GT boxes (where object_mask > 0)
        # Flatten spatial dimensions
        true_xy_flat = tf.reshape(true_xy_abs, [batch_size, -1, 2])  # [batch, grid_h*grid_w, 2]
        true_wh_flat = tf.reshape(true_wh_abs, [batch_size, -1, 2])  # [batch, grid_h*grid_w, 2]
        object_mask_flat = tf.reshape(object_mask, [batch_size, -1])  # [batch, grid_h*grid_w]
        
        # Apply activation to pred_xy before converting to absolute coordinates
        # This matches the inference pipeline: tanh(0.15*x) + sigmoid(0.15*x) to reach [-1, 2] range
        # CRITICAL: Without this activation, IoU computation uses raw logits which don't match
        # the actual decoded predictions, leading to incorrect ignore masks and loss divergence
        pred_xy_activated = tf.nn.tanh(0.15 * pred_xy) + tf.nn.sigmoid(0.15 * pred_xy)
        
        # Convert predicted boxes to absolute coordinates for each anchor
        pred_xy_abs = (pred_xy_activated + grid_coords) * scale  # [batch, grid_h, grid_w, 2]
        
        # For each anchor, compute predicted boxes and IoU with GT
        anchors_tf_expanded = tf.reshape(anchors_tf, [1, 1, 1, num_anchors, 2])  # [1, 1, 1, num_anchors, 2]
        pred_wh_expanded = tf.expand_dims(tf.exp(pred_wh), axis=-2)  # [batch, grid_h, grid_w, 1, 2]
        pred_wh_abs_all = pred_wh_expanded * anchors_tf_expanded * scale  # [batch, grid_h, grid_w, num_anchors, 2]
        
        # Compute IoU for all anchors simultaneously using vectorized operations
        # pred_wh_abs_all: [batch, grid_h, grid_w, num_anchors, 2]
        # pred_xy_abs: [batch, grid_h, grid_w, 2]
        # Expand pred_xy_abs to match: [batch, grid_h, grid_w, num_anchors, 2]
        pred_xy_abs_expanded = tf.expand_dims(pred_xy_abs, axis=-2)  # [batch, grid_h, grid_w, 1, 2]
        pred_xy_abs_expanded = tf.tile(pred_xy_abs_expanded, [1, 1, 1, num_anchors, 1])  # [batch, grid_h, grid_w, num_anchors, 2]
        
        # Combine xy and wh: [batch, grid_h, grid_w, num_anchors, 4]
        pred_boxes_all = tf.concat([pred_xy_abs_expanded, pred_wh_abs_all], axis=-1)
        
        # Reshape for processing: [batch, grid_h*grid_w*num_anchors, 4]
        pred_boxes_flat = tf.reshape(pred_boxes_all, [batch_size, -1, 4])
        
        # Compute IoU with all GT boxes for each batch item
        def compute_iou_for_batch(batch_idx):
            pred_b = pred_boxes_flat[batch_idx]  # [grid_h*grid_w*num_anchors, 4]
            gt_b = tf.concat([true_xy_flat[batch_idx], true_wh_flat[batch_idx]], axis=-1)  # [grid_h*grid_w, 4]
            valid_mask_b = object_mask_flat[batch_idx] > 0.5  # [grid_h*grid_w]
            
            # Filter to valid GT boxes
            valid_indices = tf.where(valid_mask_b)[:, 0]  # [num_valid]
            num_valid = tf.shape(valid_indices)[0]
            num_cells_anchors = grid_h_int * grid_w_int * num_anchors
            
            # Use tf.cond for graph-mode compatibility
            def compute_with_gt():
                gt_valid = tf.gather(gt_b, valid_indices)  # [num_valid, 4]
                
                # Compute IoU: [grid_h*grid_w*num_anchors, num_valid]
                # pred_b: [grid_h*grid_w*num_anchors, 4]
                # gt_valid: [num_valid, 4]
                # Expand for broadcasting: [grid_h*grid_w*num_anchors, 1, 4] and [1, num_valid, 4]
                pred_b_expanded = tf.expand_dims(pred_b, axis=1)  # [grid_h*grid_w*num_anchors, 1, 4]
                gt_valid_expanded = tf.expand_dims(gt_valid, axis=0)  # [1, num_valid, 4]
                
                iou_matrix = self._compute_iou_batch(pred_b_expanded, gt_valid_expanded)
                # Result shape: [grid_h*grid_w*num_anchors, 1, num_valid] -> squeeze middle dim
                # Only squeeze if dimension is 1, otherwise use reshape
                iou_matrix = tf.reshape(iou_matrix, [tf.shape(pred_b)[0], tf.shape(gt_valid)[0]])  # [grid_h*grid_w*num_anchors, num_valid]
                
                # Get max IoU per prediction
                return tf.reduce_max(iou_matrix, axis=-1)  # [grid_h*grid_w*num_anchors]
            
            def return_zeros():
                return tf.zeros([num_cells_anchors], dtype=tf.float32)
            
            # Use tf.cond instead of Python if
            return tf.cond(
                tf.equal(num_valid, 0),
                return_zeros,
                compute_with_gt
            )
        
        # Use map_fn for batch processing (graph-mode compatible)
        iou_all = tf.map_fn(
            compute_iou_for_batch,
            tf.range(batch_size),
            fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.float32)
        )  # [batch, grid_h*grid_w*num_anchors]
        
        # Reshape back: [batch, grid_h, grid_w, num_anchors]
        # Need to compute total cells dynamically
        total_cells = grid_h_int * grid_w_int
        expected_flat_size = total_cells * num_anchors
        
        # Debug: Assert reshape is valid
        actual_flat_size = tf.shape(iou_all)[1]
        tf.debugging.assert_equal(actual_flat_size, expected_flat_size,
                                 message=f"IoU reshape mismatch: expected {expected_flat_size}, got {actual_flat_size}")
        
        iou_all = tf.reshape(iou_all, [batch_size, grid_h_int, grid_w_int, num_anchors])
        
        # Debug: Assert reshaped shape
        tf.debugging.assert_equal(tf.shape(iou_all)[1], grid_h_int, message="Reshaped grid_h mismatch")
        tf.debugging.assert_equal(tf.shape(iou_all)[2], grid_w_int, message="Reshaped grid_w mismatch")
        tf.debugging.assert_equal(tf.shape(iou_all)[3], num_anchors, message="Reshaped num_anchors mismatch")
        
        # Get max IoU across anchors for each cell: [batch, grid_h, grid_w]
        max_iou_per_cell = tf.reduce_max(iou_all, axis=-1)
        
        # Debug: Assert max_iou shape
        tf.debugging.assert_equal(tf.shape(max_iou_per_cell)[0], batch_size, message="max_iou batch mismatch")
        tf.debugging.assert_equal(tf.shape(max_iou_per_cell)[1], grid_h_int, message="max_iou grid_h mismatch")
        tf.debugging.assert_equal(tf.shape(max_iou_per_cell)[2], grid_w_int, message="max_iou grid_w mismatch")
        
        # Create ignore mask: IoU > threshold AND not assigned as positive
        object_mask_squeezed = tf.squeeze(object_mask, axis=-1)  # [batch, grid_h, grid_w]
        tf.debugging.assert_equal(tf.shape(object_mask_squeezed), tf.shape(max_iou_per_cell),
                                 message="object_mask and max_iou shape mismatch")
        
        ignore_mask = tf.cast(
            (max_iou_per_cell > self.ignore_thresh) & (object_mask_squeezed < 0.5),
            dtype=tf.float32
        )
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)  # [batch, grid_h, grid_w, 1]

        # IoU of the assigned anchor (zeroed for empty cells)
        assigned_anchor_iou = tf.reduce_sum(iou_all * true_anchor_one_hot, axis=-1, keepdims=True)
        assigned_anchor_iou = assigned_anchor_iou * object_mask
        assigned_anchor_iou = tf.stop_gradient(assigned_anchor_iou)

        # Max IoU map for soft suppression
        max_iou_map = tf.expand_dims(max_iou_per_cell, axis=-1)
        max_iou_map = tf.stop_gradient(max_iou_map)
        
        # Debug: Final shape check
        tf.debugging.assert_equal(tf.shape(ignore_mask), tf.shape(object_mask),
                                message="Final ignore_mask shape mismatch")
        
        return ignore_mask, assigned_anchor_iou, max_iou_map
    
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
        
        Applies tanh+sigmoid activation to pred_xy to match inference pipeline behavior.
        
        Formula:
        pred_xy_activated = tanh(0.15 * pred_xy) + sigmoid(0.15 * pred_xy)
        L_xy = Σ object_mask * (true_xy - pred_xy_activated)²
        L_wh = Σ object_mask * (true_wh - pred_wh)²
        L_location = L_xy + L_wh
        """
        # Apply tanh+sigmoid activation to pred_xy to match inference pipeline
        pred_xy_activated = tf.nn.tanh(0.15 * pred_xy) + tf.nn.sigmoid(0.15 * pred_xy)
        
        # XY coordinate loss
        xy_diff = true_xy - pred_xy_activated
        xy_loss = K.sum(K.square(xy_diff), axis=-1, keepdims=True)  # [batch, grid_h, grid_w, 1]
        
        # WH size loss
        wh_diff = true_wh - pred_wh
        wh_loss = K.sum(K.square(wh_diff), axis=-1, keepdims=True)  # [batch, grid_h, grid_w, 1]
        
        # Combine and apply object mask
        loc_loss = (xy_loss + wh_loss) * object_mask
        
        return K.sum(loc_loss)
    
    def _compute_anchor_loss(self, true_anchors: tf.Tensor, pred_anchors: tf.Tensor,
                            object_mask: tf.Tensor, ignore_mask: Optional[tf.Tensor],
                            norm_factor: tf.Tensor) -> tf.Tensor:
        """
        Compute anchor prediction loss using BCE.
        
        Anchor loss is computed ONLY on object cells (like classification loss).
        Anchor prediction answers "which anchor fits this object best", so it should only
        be computed where objects exist.
        
        Formula:
        L_anchor = Σ object_mask * (1 - ignore_mask) * BCE(true_anchors, pred_anchors) / normalization
        
        Note: anchor_scale is applied when composing total loss, not here.
        This decouples anchor scaling from object_scale for better controllability.
        
        This includes:
        - Positive cells (object_mask > 0): included in loss
        - Negative cells: EXCLUDED (anchor prediction doesn't apply to background)
        - Ignore cells (ignore_mask > 0): excluded from loss
        """
        if ignore_mask is None:
            ignore_mask = tf.zeros_like(object_mask)
        
        # BCE loss on anchor predictions
        anchor_loss = K.binary_crossentropy(true_anchors, pred_anchors, from_logits=True)
        
        # Apply object mask: ONLY compute anchor loss on object cells
        # Anchor prediction is about "which anchor fits this object best", so it only
        # makes sense for cells that contain objects
        # Broadcasting: [batch, grid_h, grid_w, 1] × [batch, grid_h, grid_w, num_anchors]
        anchor_loss = anchor_loss * object_mask
        
        # Exclude ignore cells (if any)
        anchor_loss = anchor_loss * (1.0 - ignore_mask)
        
        # Normalize by norm_factor (passed as parameter for consistency)
        anchor_loss_sum = K.sum(anchor_loss)
        tf.debugging.assert_all_finite(anchor_loss_sum, message="anchor_loss_sum must be finite")
        
        return anchor_loss_sum / norm_factor
    
    def _compute_focal_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                         object_mask: tf.Tensor, norm_factor: tf.Tensor) -> tf.Tensor:
        """Compute focal loss for classification (all classes)."""
        class_loss = self.focal_loss.compute_loss(true_class, pred_class)
        # Broadcasting: [batch, grid_h, grid_w, 1] * [batch, grid_h, grid_w, num_classes]
        class_loss = class_loss * object_mask
        
        # Apply class weights for handling class imbalance
        # Expand class_weights: [num_classes] -> [1, 1, 1, num_classes]
        class_weights_expanded = tf.reshape(self.class_weights, [1, 1, 1, self.num_classes])
        class_loss = class_loss * class_weights_expanded
        
        return K.sum(class_loss) / norm_factor
    
    def _compute_softmax_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                            object_mask: tf.Tensor, norm_factor: tf.Tensor) -> tf.Tensor:
        """Compute softmax focal loss for classification (all classes)."""
        class_loss = self.softmax_focal_loss.compute_loss(true_class, pred_class)
        # Broadcasting: [batch, grid_h, grid_w, 1] * [batch, grid_h, grid_w, num_classes]
        class_loss = class_loss * object_mask
        
        # Apply class weights for handling class imbalance
        # Expand class_weights: [num_classes] -> [1, 1, 1, num_classes]
        class_weights_expanded = tf.reshape(self.class_weights, [1, 1, 1, self.num_classes])
        class_loss = class_loss * class_weights_expanded
        
        return K.sum(class_loss) / norm_factor
    
    def _compute_bce_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                        object_mask: tf.Tensor, norm_factor: tf.Tensor) -> tf.Tensor:
        """
        Compute BCE loss for classification (all classes, not top-k!).
        
        Formula:
        L_class = Σ object_mask * class_weights * BCE(true_class, pred_class) / total_grid_cells
        
        Applies to ALL classes, ensuring all classes receive gradients.
        Class weights are applied to handle class imbalance.
        Normalized by total_grid_cells for consistent measurement across batches.
        """
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            true_class_smooth = true_class * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            true_class_smooth = true_class
        
        # BCE loss on all classes
        class_loss = K.binary_crossentropy(true_class_smooth, pred_class, from_logits=True)
        
        # Apply class weights for handling class imbalance
        # Expand class_weights: [num_classes] -> [1, 1, 1, num_classes]
        class_weights_expanded = tf.reshape(self.class_weights, [1, 1, 1, self.num_classes])
        class_loss = class_loss * class_weights_expanded
        
        # Apply mask: only penalize classification on object cells
        # Broadcasting: [batch, grid_h, grid_w, 1] * [batch, grid_h, grid_w, num_classes]
        class_loss = class_loss * object_mask
        
        return K.sum(class_loss) / norm_factor
    
    def _compute_objectness_loss(self, true_obj: tf.Tensor, pred_obj: tf.Tensor,
                                object_mask: tf.Tensor, ignore_mask: Optional[tf.Tensor],
                                norm_factor: tf.Tensor,
                                positive_iou_map: Optional[tf.Tensor] = None,
                                max_iou_map: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute objectness loss using BCE with logits.
        
        Objectness predicts if an object exists in a grid cell.
        
        Formula:
        L_objectness = Σ [object_mask * object_scale + (1-object_mask) * (1-ignore_mask) * no_object_scale]
                       * BCE(target_obj, pred_obj, from_logits=True) / normalization
        
        target_obj defaults to binary labels but, when IoU-aware objectness is enabled,
        follows the predicted IoU (stopped-gradient) to teach the model to down-weight
        boxes with poor localization — effectively acting as trainable NMS.
        
        Uses from_logits=True for numerical stability (BCE handles sigmoid internally).
        
        This includes:
        - Positive cells (object_mask > 0): weighted by object_scale
        - Negative cells (object_mask == 0 and ignore_mask == 0): weighted by no_object_scale
        - Ignore cells (ignore_mask > 0): excluded unless trainable_nms_weight > 0, in which case they receive a soft penalty
        """
        if ignore_mask is None:
            ignore_mask = tf.zeros_like(object_mask)
        
        # Build IoU-aware targets if enabled (acts like trainable, differentiable NMS)
        obj_target = true_obj
        if self.use_iou_aware_objectness:
            if positive_iou_map is None:
                positive_iou_map = object_mask
            positive_iou_map = tf.stop_gradient(tf.clip_by_value(positive_iou_map, 0.0, 1.0))
            iou_target = tf.pow(positive_iou_map + self.eps, self.iou_objectness_power)
            blended_target = (
                self.iou_objectness_ratio * iou_target +
                (1.0 - self.iou_objectness_ratio) * true_obj
            )
            obj_target = object_mask * blended_target + (1.0 - object_mask) * obj_target
        
        # BCE loss on objectness predictions (from_logits=True - feed logits directly to BCE)
        # This is more numerically stable than manually applying sigmoid first
        objectness_loss = K.binary_crossentropy(obj_target, pred_obj, from_logits=True)
        
        # Weight mask: positive cells get object_scale, negative cells get no_object_scale
        # Ignore cells (ignore_mask > 0) are excluded (weight = 0)
        positive_weight = object_mask * self.object_scale  # [batch, grid_h, grid_w, 1]
        negative_mask = (1.0 - object_mask) * (1.0 - ignore_mask)  # [batch, grid_h, grid_w, 1]
        negative_weight = negative_mask * self.no_object_scale  # [batch, grid_h, grid_w, 1]
        combined_weight = positive_weight + negative_weight  # [batch, grid_h, grid_w, 1]

        # Optional: softly penalize ignore regions using IoU as a weight (trainable NMS)
        if self.trainable_nms_weight > 0.0 and max_iou_map is not None:
            max_iou_map = tf.stop_gradient(tf.clip_by_value(max_iou_map, 0.0, 1.0))
            suppression_scale = tf.pow(max_iou_map + self.eps, self.trainable_nms_power)
            soft_ignore_mask = (1.0 - object_mask) * ignore_mask
            soft_ignore_weight = soft_ignore_mask * self.trainable_nms_weight * suppression_scale
            combined_weight = combined_weight + soft_ignore_weight
        
        # Apply weight mask
        objectness_loss = objectness_loss * combined_weight
        
        # Normalize by norm_factor (passed as parameter for consistency)
        objectness_loss_sum = K.sum(objectness_loss)
        tf.debugging.assert_all_finite(objectness_loss_sum, message="objectness_loss_sum must be finite")
        
        return objectness_loss_sum / norm_factor
    
    def _build_grid_coordinates(self, grid_shape: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Create grid coordinate tensor [1, grid_h, grid_w, 2] for converting to absolute centers."""
        grid_h, grid_w = grid_shape
        grid_h_int = tf.cast(grid_h, tf.int32)
        grid_w_int = tf.cast(grid_w, tf.int32)
        grid_y = tf.range(grid_h_int, dtype=tf.float32)
        grid_x = tf.range(grid_w_int, dtype=tf.float32)
        grid_x_mesh, grid_y_mesh = tf.meshgrid(grid_x, grid_y, indexing='ij')
        coords = tf.stack([grid_x_mesh, grid_y_mesh], axis=-1)
        return tf.expand_dims(coords, axis=0)
    
    def _compute_center_mask(self, true_xy: tf.Tensor, object_mask: tf.Tensor) -> tf.Tensor:
        """Identify central grid cells (the ones whose offsets stay within [0,1))."""
        center_x = tf.logical_and(true_xy[..., 0] >= 0.0, true_xy[..., 0] < 1.0)
        center_y = tf.logical_and(true_xy[..., 1] >= 0.0, true_xy[..., 1] < 1.0)
        center_mask = tf.cast(tf.logical_and(center_x, center_y), tf.float32)
        center_mask = tf.expand_dims(center_mask, axis=-1)
        return center_mask * object_mask
    
    def _extract_local_patches(self, tensor: tf.Tensor) -> tf.Tensor:
        """Extract consensus neighborhoods using tf.image.extract_patches."""
        k = self.consensus_kernel_size
        patches = tf.image.extract_patches(
            images=tensor,
            sizes=[1, k, k, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        tensor_shape = tf.shape(tensor)
        patch_area = k * k
        return tf.reshape(
            patches,
            [tensor_shape[0], tensor_shape[1], tensor_shape[2], patch_area, tensor_shape[3]]
        )
    
    def _compute_variance_consensus_loss(self,
                                         pred_xy: tf.Tensor,
                                         pred_wh: tf.Tensor,
                                         pred_obj: tf.Tensor,
                                         pred_class: tf.Tensor,
                                         true_xy: tf.Tensor,
                                         object_mask: tf.Tensor,
                                         assigned_anchor_iou: tf.Tensor,
                                         grid_shape: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute variance-based consensus losses for coordinates, objectness, and classes.
        """
        center_mask = self._compute_center_mask(true_xy, object_mask)
        grid_coords = self._build_grid_coordinates(grid_shape)
        true_centers = true_xy + grid_coords  # [batch, grid_h, grid_w, 2]
        
        # Gather patches for masks, IoU weights, and center coordinates
        mask_patches = self._extract_local_patches(object_mask)
        iou_patches = self._extract_local_patches(assigned_anchor_iou)
        center_patches = self._extract_local_patches(true_centers)
        
        # Identify cells that share the same decoded center as the central cell
        center_ref = tf.expand_dims(true_centers, axis=3)  # [batch, grid_h, grid_w, 1, 2]
        center_diff = tf.abs(center_patches - center_ref)
        same_center = tf.cast(
            tf.reduce_max(center_diff, axis=-1, keepdims=True) < self.consensus_center_tolerance,
            tf.float32
        )
        
        group_mask = mask_patches * same_center
        group_mask = group_mask * tf.expand_dims(center_mask, axis=3)
        
        # IoU-based weighting with minimum floor
        valid_weights = tf.where(
            group_mask > 0.0,
            tf.maximum(iou_patches, self.consensus_min_iou),
            tf.zeros_like(iou_patches)
        )
        raw_weights = tf.pow(valid_weights, self.consensus_iou_power) * group_mask
        weight_sum = tf.reduce_sum(raw_weights, axis=3, keepdims=True) + self.eps
        weights = raw_weights / weight_sum
        weights_scalar = tf.squeeze(weights, axis=-1)
        
        normalizer = K.maximum(tf.reduce_sum(center_mask), 1.0)
        
        # Coordinate consensus (xy + wh stacked)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
        box_patches = self._extract_local_patches(pred_box)
        box_consensus = tf.reduce_sum(weights * box_patches, axis=3)
        if self.consensus_stop_gradient:
            box_consensus = tf.stop_gradient(box_consensus)
        box_diff = box_patches - tf.expand_dims(box_consensus, axis=3)
        box_diff_sq = tf.reduce_sum(tf.square(box_diff), axis=-1)
        coord_variance = tf.reduce_sum(weights_scalar * box_diff_sq) / normalizer
        
        # Objectness consensus (operate on sigmoid probabilities)
        obj_probs = tf.nn.sigmoid(pred_obj)
        obj_patches = self._extract_local_patches(obj_probs)
        obj_consensus = tf.reduce_sum(weights * obj_patches, axis=3)
        if self.consensus_stop_gradient:
            obj_consensus = tf.stop_gradient(obj_consensus)
        obj_diff = obj_patches - tf.expand_dims(obj_consensus, axis=3)
        obj_diff_sq = tf.square(obj_diff)
        obj_variance = tf.reduce_sum(weights_scalar * tf.squeeze(obj_diff_sq, axis=-1)) / normalizer
        
        # Classification consensus (use sigmoid probs for BCE-style heads)
        class_probs = tf.nn.sigmoid(pred_class)
        class_patches = self._extract_local_patches(class_probs)
        class_consensus = tf.reduce_sum(weights * class_patches, axis=3)
        if self.consensus_stop_gradient:
            class_consensus = tf.stop_gradient(class_consensus)
        class_diff = class_patches - tf.expand_dims(class_consensus, axis=3)
        class_diff_sq = tf.square(class_diff)
        class_weights = tf.expand_dims(weights_scalar, axis=-1)
        class_variance = tf.reduce_sum(class_weights * class_diff_sq)
        class_variance = class_variance / (normalizer * tf.cast(self.num_classes, tf.float32))
        
        return coord_variance, obj_variance, class_variance


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
