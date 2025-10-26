#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet loss function implementation.
Migrated from original implementation with IoL anchor matching and dense prediction support.
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math

from .focal_loss import SigmoidFocalLoss, SoftmaxFocalLoss
from .iou_losses import GIoULoss, DIoULoss, CIoULoss


class MultiGridLoss:
    """
    MultiGridDet loss function.
    
    This loss function implements the core MultiGridDet innovations:
    - IoL (Intersection over Largest) anchor matching
    - Dense prediction with 3x3 grid assignment
    - Focal loss for handling class imbalance
    - Advanced IoU losses (GIoU, DIoU, CIoU)
    """
    
    # Add __name__ attribute for Keras model saving compatibility
    __name__ = 'MultiGridLoss'
    
    def __init__(self, 
                 anchors: List[np.ndarray],
                 num_classes: int,
                 input_shape: Tuple[int, int] = (608, 608),
                 ignore_thresh: float = 0.5,
                 label_smoothing: float = 0.25,
                 elim_grid_sense: bool = False,
                 use_focal_loss: bool = True,
                 use_focal_obj_loss: bool = True,
                 use_softmax_loss: bool = False,
                 use_iol: bool = True,
                 use_giou_loss: bool = True,
                 use_diou_loss: bool = False,
                 use_ciou_loss: bool = False,
                 loss_option: int = 3,  # NEW: 1, 2, or 3 (default 3 for GIoU/DIoU)
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 coord_scale: float = 1.0,
                 object_scale: float = 1.0,
                 no_object_scale: float = 1.0,
                 class_scale: float = 0.5,
                 anchor_scale: float = 0.5):
        """
        Initialize MultiGridDet loss.
        
        Args:
            anchors: List of anchor arrays for each scale
            num_classes: Number of object classes
            input_shape: Input image shape (height, width)
            ignore_thresh: IoU threshold for ignoring object confidence loss
            label_smoothing: Label smoothing factor
            elim_grid_sense: Whether to eliminate grid sensitivity
            use_focal_loss: Whether to use focal loss for classification
            use_focal_obj_loss: Whether to use focal loss for objectness
            use_softmax_loss: Whether to use softmax loss for classification
            use_iol: Whether to use IoL anchor matching
            use_giou_loss: Whether to use GIoU loss for localization
            use_diou_loss: Whether to use DIoU loss for localization
            use_ciou_loss: Whether to use CIoU loss for localization
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            coord_scale: Coordinate loss scale
            object_scale: Object loss scale
            no_object_scale: No-object loss scale
            class_scale: Classification loss scale
            anchor_scale: Anchor loss scale
        """
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.ignore_thresh = ignore_thresh
        self.label_smoothing = label_smoothing
        self.elim_grid_sense = elim_grid_sense
        self.use_focal_loss = use_focal_loss
        self.use_focal_obj_loss = use_focal_obj_loss
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
        
        # Initialize loss components
        if use_focal_loss:
            self.focal_loss = SigmoidFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if use_softmax_loss:
            self.softmax_focal_loss = SoftmaxFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if use_giou_loss:
            self.giou_loss = GIoULoss()
        if use_diou_loss:
            self.diou_loss = DIoULoss()
        if use_ciou_loss:
            self.ciou_loss = CIoULoss()
    
    def __call__(self, y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
        """
        Compute MultiGridDet loss.
        
        Args:
            y_true: List of ground truth tensors for each scale
            y_pred: List of prediction tensors for each scale
            
        Returns:
            Total loss tensor
        """
        return self.compute_loss(y_true, y_pred)
    
    def compute_loss(self, y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> tf.Tensor:
        """
        Compute the complete MultiGridDet loss.
        
        Args:
            y_true: List of ground truth tensors for each scale
            y_pred: List of prediction tensors for each scale
            
        Returns:
            Total loss tensor
        """
        # Calculate grid shapes and cell dimensions
        input_shape = K.cast(K.shape(y_pred[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = [K.cast(K.shape(y_pred[i])[1:3], 'float32') for i in range(self.num_layers)]
        cell_wh = [K.cast(input_shape / grid_shapes[i], 'float32') for i in range(self.num_layers)]
        
        batch_size = K.shape(y_pred[0])[0]
        batch_size_f = K.cast(batch_size, 'float32')
        
        total_loss = 0
        total_loss_location = 0
        total_loss_classification = 0
        total_loss_objectness = 0
        
        # Process each scale
        for layer_idx in range(self.num_layers):
            # Get predictions and ground truth for this layer
            y_pred_layer = y_pred[layer_idx]
            y_true_layer = y_true[layer_idx]
            
            # Get anchor information
            anchor_layer = self.anchors[layer_idx]
            num_anchors = len(anchor_layer)
            
            # Reshape predictions
            # y_pred: (batch, grid_h, grid_w, 5 + num_anchors + num_classes)
            grid_h, grid_w = K.shape(y_pred_layer)[1], K.shape(y_pred_layer)[2]
            
            # Extract prediction components
            pred_xy = y_pred_layer[..., 0:2]  # Center coordinates
            pred_wh = y_pred_layer[..., 2:4]  # Width and height
            pred_obj = y_pred_layer[..., 4:5]  # Objectness
            pred_anchors = y_pred_layer[..., 5:5+num_anchors]  # Anchor-specific objectness
            pred_class = y_pred_layer[..., 5+num_anchors:]  # Class predictions
            
            # Extract ground truth components
            true_xy = y_true_layer[..., 0:2]
            true_wh = y_true_layer[..., 2:4]
            true_obj = y_true_layer[..., 4:5]
            true_anchors = y_true_layer[..., 5:5+num_anchors]
            true_class = y_true_layer[..., 5+num_anchors:]
            
            # Create object mask - use the original approach
            object_mask = true_obj
            # object_mask should already have shape [batch, grid_h, grid_w, 1] from y_true_layer[..., 4:5]
            # Ensure object_mask maintains 4D shape for consistent broadcasting
            # Always expand to 4D to avoid shape issues
            object_mask = K.expand_dims(K.squeeze(object_mask, axis=-1), axis=-1)
            
            # Extract assigned anchor index (following original implementation approach)
            anchor_index = K.argmax(true_anchors, axis=-1)
            
            # Create anchor mask based on assigned anchors (simplified approach)
            best_anchor_mask = K.one_hot(anchor_index, num_anchors)
            
            # Compute ignore mask
            ignore_mask = self._compute_ignore_mask(
                pred_xy, pred_wh, true_xy, true_wh, 
                anchor_layer, object_mask, grid_shapes[layer_idx]
            )
            
            # Localization loss - choose based on loss_option
            if self.loss_option == 1:
                # Option 1: IoL-weighted MSE (original paper)
                loc_loss = self._compute_option1_loss(
                    true_xy, true_wh, pred_xy, pred_wh, 
                    object_mask, batch_size_f
                )
            elif self.loss_option == 2:
                # Option 2: IoL-weighted MSE with trainable anchor prediction
                loc_loss = self._compute_option2_loss(
                    true_xy, true_wh, pred_xy, pred_wh,
                    object_mask, batch_size_f, true_anchors, pred_anchors
                )
            else:  # Option 3 (default)
                # Option 3: GIoU/DIoU/CIoU losses
                if self.use_giou_loss:
                    loc_loss = self._compute_giou_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
                elif self.use_diou_loss:
                    loc_loss = self._compute_diou_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
                elif self.use_ciou_loss:
                    loc_loss = self._compute_ciou_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
                else:
                    loc_loss = self._compute_mse_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
                loc_loss = loc_loss / batch_size_f
            
            total_loss_location += loc_loss
            
            # Objectness loss (following original implementation approach exactly)
            # Create combined mask - ensure proper shape for broadcasting
            combined_mask = K.cast((object_mask + ignore_mask) > 0, 'float32')
            # combined_mask should be 4D [batch, grid_h, grid_w, 1] to match anchor_loss [batch, grid_h, grid_w, num_anchors]
            # No need to expand - it should already be 4D from object_mask and ignore_mask
            
            # Compute BCE loss on anchor predictions (following original line 657)
            anchor_loss = K.binary_crossentropy(true_anchors, pred_anchors, from_logits=True)
            anchor_loss = 0.5 * combined_mask * anchor_loss
            
            obj_loss = K.sum(anchor_loss) / batch_size_f
            
            total_loss_objectness += obj_loss
            
            # Classification loss - use top-k selection like original
            pred_cls_values, pred_cls_indices = tf.math.top_k(pred_class, k=5)
            true_cls_values = tf.gather(true_class, pred_cls_indices, batch_dims=-1)
            class_loss = K.binary_crossentropy(true_cls_values, pred_cls_values, from_logits=False)
            class_loss = K.sum(class_loss) / batch_size_f
            
            total_loss_classification += class_loss
        
        # Combine all losses
        total_loss = (
            self.coord_scale * total_loss_location +
            self.object_scale * total_loss_objectness +
            self.class_scale * total_loss_classification
        )
        
        return total_loss
    
    
    
    def _compute_ignore_mask(self, pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                           true_xy: tf.Tensor, true_wh: tf.Tensor,
                           anchors: np.ndarray, object_mask: tf.Tensor,
                           grid_shape: tf.Tensor) -> tf.Tensor:
        """Compute ignore mask for objectness loss."""
        # This is a simplified version - in practice, you'd compute IoU with all ground truth boxes
        # For now, we'll use a simple approach
        ignore_mask = tf.zeros_like(object_mask)
        return ignore_mask
    
    def _compute_giou_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                          pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                          object_mask: tf.Tensor) -> tf.Tensor:
        """Compute GIoU loss for localization."""
        if hasattr(self, 'giou_loss'):
            return self.giou_loss.compute_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
        else:
            # Fallback to MSE loss
            return self._compute_mse_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
    
    def _compute_diou_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                          pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                          object_mask: tf.Tensor) -> tf.Tensor:
        """Compute DIoU loss for localization."""
        if hasattr(self, 'diou_loss'):
            return self.diou_loss.compute_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
        else:
            # Fallback to MSE loss
            return self._compute_mse_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
    
    def _compute_ciou_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                          pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                          object_mask: tf.Tensor) -> tf.Tensor:
        """Compute CIoU loss for localization."""
        if hasattr(self, 'ciou_loss'):
            return self.ciou_loss.compute_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
        else:
            # Fallback to MSE loss
            return self._compute_mse_loss(true_xy, true_wh, pred_xy, pred_wh, object_mask)
    
    def _compute_mse_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                         pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                         object_mask: tf.Tensor) -> tf.Tensor:
        """Compute MSE loss for localization."""
        # Coordinate loss
        xy_loss = K.sum(K.square(true_xy - pred_xy), axis=-1, keepdims=True)
        
        # Size loss
        wh_loss = K.sum(K.square(K.sqrt(true_wh) - K.sqrt(pred_wh)), axis=-1, keepdims=True)
        
        # Combine losses
        loc_loss = xy_loss + wh_loss
        
        # Apply object mask - ensure shapes match
        # loc_loss: [batch, grid_h, grid_w, 1]
        # object_mask: [batch, grid_h, grid_w, 1]
        loc_loss = loc_loss * object_mask
        
        return K.sum(loc_loss)
    
    def _compute_focal_objectness_loss(self, true_obj: tf.Tensor, pred_obj: tf.Tensor,
                                     true_anchors: tf.Tensor, pred_anchors: tf.Tensor,
                                     best_anchor_mask: tf.Tensor, ignore_mask: tf.Tensor) -> tf.Tensor:
        """Compute focal loss for objectness following original implementation approach."""
        # Create combined mask (following original implementation approach)
        # Use object_mask instead of true_obj to match original
        object_mask = true_obj
        combined_mask = K.cast((object_mask + ignore_mask) > 0, 'float32')
        
        # Compute focal loss on anchor predictions (following original approach)
        anchor_loss = self.focal_loss.compute_loss(true_anchors, pred_anchors)
        
        # Apply combined mask
        anchor_loss = anchor_loss * combined_mask
        
        return K.sum(anchor_loss)
    
    def _compute_bce_objectness_loss(self, true_obj: tf.Tensor, pred_obj: tf.Tensor,
                                   true_anchors: tf.Tensor, pred_anchors: tf.Tensor,
                                   best_anchor_mask: tf.Tensor, ignore_mask: tf.Tensor) -> tf.Tensor:
        """Compute BCE loss for objectness following original implementation approach."""
        # Create combined mask (following original implementation approach)
        # Use object_mask instead of true_obj to match original
        object_mask = true_obj
        combined_mask = K.cast((object_mask + ignore_mask) > 0, 'float32')
        
        # Compute BCE loss on anchor predictions (following original approach)
        anchor_loss = K.binary_crossentropy(true_anchors, pred_anchors, from_logits=True)
        
        # Apply combined mask
        anchor_loss = anchor_loss * combined_mask
        
        return K.sum(anchor_loss)
    
    def _compute_focal_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                         object_mask: tf.Tensor) -> tf.Tensor:
        """Compute focal loss for classification."""
        class_loss = self.focal_loss.compute_loss(true_class, pred_class)
        # Ensure object_mask has the right shape for broadcasting
        # object_mask should have shape [batch, grid_h, grid_w, 1]
        # class_loss has shape [batch, grid_h, grid_w, num_classes]
        # We need to expand object_mask to match class_loss dimensions
        object_mask = K.expand_dims(object_mask, axis=-1)
        object_mask = K.tile(object_mask, [1, 1, 1, K.shape(pred_class)[-1]])
        class_loss = class_loss * object_mask
        return K.sum(class_loss)
    
    def _compute_softmax_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                           object_mask: tf.Tensor) -> tf.Tensor:
        """Compute softmax focal loss for classification."""
        class_loss = self.softmax_focal_loss.compute_loss(true_class, pred_class)
        # Ensure object_mask has the right shape for broadcasting
        # object_mask should have shape [batch, grid_h, grid_w, 1]
        # class_loss has shape [batch, grid_h, grid_w, num_classes]
        # We need to expand object_mask to match class_loss dimensions
        object_mask = K.expand_dims(object_mask, axis=-1)
        object_mask = K.tile(object_mask, [1, 1, 1, K.shape(pred_class)[-1]])
        class_loss = class_loss * object_mask
        return K.sum(class_loss)
    
    def _compute_bce_classification_loss(self, true_class: tf.Tensor, pred_class: tf.Tensor,
                                       object_mask: tf.Tensor) -> tf.Tensor:
        """Compute BCE loss for classification."""
        class_loss = K.binary_crossentropy(true_class, pred_class, from_logits=True)
        # Ensure object_mask has the right shape for broadcasting
        # object_mask should have shape [batch, grid_h, grid_w, 1]
        # class_loss has shape [batch, grid_h, grid_w, num_classes]
        # We need to expand object_mask to match class_loss dimensions
        object_mask = K.expand_dims(object_mask, axis=-1)
        object_mask = K.tile(object_mask, [1, 1, 1, K.shape(pred_class)[-1]])
        class_loss = class_loss * object_mask
        return K.sum(class_loss)

    def _compute_option1_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                             pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                             object_mask: tf.Tensor, batch_size_f: tf.Tensor) -> tf.Tensor:
        """
        Option 1: IoL-weighted MSE loss (original paper).
        Uses raw predictions with IoL confidence weighting.
        """
        # Compute IoL confidence from best IoU with predicted boxes
        # For now, use simplified approach (can be enhanced with actual IoL computation)
        iol_conf = object_mask  # Simplified - in full implementation, compute actual IoL
        
        # MSE loss weighted by IoL confidence
        loss_xy = K.exp(-iol_conf/0.8) * K.square(true_xy - pred_xy)
        loss_wh = K.exp(-iol_conf/0.8) * K.square(true_wh - pred_wh)
        
        return (K.sum(loss_xy) + K.sum(loss_wh)) / batch_size_f

    def _compute_option2_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                             pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                             object_mask: tf.Tensor, batch_size_f: tf.Tensor,
                             true_anchors: tf.Tensor, pred_anchors: tf.Tensor) -> tf.Tensor:
        """
        Option 2: IoL-weighted MSE loss with trainable anchor prediction.
        This implements the original MultiGridDet approach where the model learns
        to predict which anchor has the highest IoL with the ground truth object.
        """
        # Compute IoL confidence (simplified - in full implementation, compute actual IoL)
        iol_conf = object_mask
        
        # MSE loss with IoL confidence weighting and object mask
        loss_xy = object_mask * K.exp(-iol_conf/0.8) * K.square(true_xy - pred_xy)
        loss_wh = object_mask * K.exp(-iol_conf/0.8) * K.square(true_wh - pred_wh)
        
        # Trainable anchor prediction loss (following original MultiGridDet paper)
        # The model learns to predict which anchor has highest IoL with ground truth
        loss_anchor = object_mask * K.binary_crossentropy(true_anchors, pred_anchors, from_logits=True)
        
        return (K.sum(loss_xy) + K.sum(loss_wh) + K.sum(loss_anchor)) / batch_size_f


# Backward compatibility function
def denseyolo2_loss(args, anchors, num_classes, **kwargs):
    """
    Backward compatibility function for MultiGridDet loss.
    
    Args:
        args: List containing yolo_outputs and y_true
        anchors: List of anchor arrays
        num_classes: Number of classes
        **kwargs: Additional loss parameters
        
    Returns:
        Loss tensor
    """
    num_layers = len(anchors)
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    
    # Create MultiGridLoss instance
    loss_fn = MultiGridLoss(anchors=anchors, num_classes=num_classes, **kwargs)
    
    # Compute loss
    return loss_fn.compute_loss(y_true, yolo_outputs)
