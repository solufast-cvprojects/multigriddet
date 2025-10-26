#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IoU-based loss functions for MultiGridDet.
Includes IoU, GIoU, DIoU, and CIoU losses.
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import math
from typing import Optional


class IoULoss:
    """Base IoU loss class."""
    
    def compute_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                    pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                    object_mask: tf.Tensor) -> tf.Tensor:
        """
        Compute IoU loss.
        
        Args:
            true_xy: True center coordinates
            true_wh: True width and height
            pred_xy: Predicted center coordinates
            pred_wh: Predicted width and height
            object_mask: Object presence mask
            
        Returns:
            IoU loss tensor
        """
        raise NotImplementedError("Subclasses must implement compute_loss method")


class GIoULoss(IoULoss):
    """
    Generalized IoU (GIoU) loss.
    
    Reference: "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
    https://arxiv.org/abs/1902.09630
    """
    
    def compute_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                    pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                    object_mask: tf.Tensor) -> tf.Tensor:
        """
        Compute GIoU loss.
        
        Args:
            true_xy: True center coordinates
            true_wh: True width and height
            pred_xy: Predicted center coordinates
            pred_wh: Predicted width and height
            object_mask: Object presence mask
            
        Returns:
            GIoU loss tensor
        """
        # Convert to corner coordinates
        true_mins = true_xy - true_wh / 2.0
        true_maxes = true_xy + true_wh / 2.0
        pred_mins = pred_xy - pred_wh / 2.0
        pred_maxes = pred_xy + pred_wh / 2.0
        
        # Compute intersection
        intersect_mins = K.maximum(true_mins, pred_mins)
        intersect_maxes = K.minimum(true_maxes, pred_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        # Compute areas
        true_area = true_wh[..., 0] * true_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        union_area = true_area + pred_area - intersect_area
        
        # Compute IoU
        iou = intersect_area / (union_area + K.epsilon())
        
        # Compute enclosed area
        enclose_mins = K.minimum(true_mins, pred_mins)
        enclose_maxes = K.maximum(true_maxes, pred_maxes)
        enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        
        # Compute GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + K.epsilon())
        
        # Convert to loss (1 - GIoU)
        giou_loss = 1.0 - giou
        
        # Apply object mask
        giou_loss = giou_loss * object_mask
        
        return K.sum(giou_loss)


class DIoULoss(IoULoss):
    """
    Distance IoU (DIoU) loss.
    
    Reference: "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    https://arxiv.org/abs/1911.08287
    """
    
    def compute_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                    pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                    object_mask: tf.Tensor) -> tf.Tensor:
        """
        Compute DIoU loss.
        
        Args:
            true_xy: True center coordinates
            true_wh: True width and height
            pred_xy: Predicted center coordinates
            pred_wh: Predicted width and height
            object_mask: Object presence mask
            
        Returns:
            DIoU loss tensor
        """
        # Convert to corner coordinates
        true_mins = true_xy - true_wh / 2.0
        true_maxes = true_xy + true_wh / 2.0
        pred_mins = pred_xy - pred_wh / 2.0
        pred_maxes = pred_xy + pred_wh / 2.0
        
        # Compute intersection
        intersect_mins = K.maximum(true_mins, pred_mins)
        intersect_maxes = K.minimum(true_maxes, pred_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        # Compute areas
        true_area = true_wh[..., 0] * true_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        union_area = true_area + pred_area - intersect_area
        
        # Compute IoU
        iou = intersect_area / (union_area + K.epsilon())
        
        # Compute center distance
        center_distance = K.sum(K.square(true_xy - pred_xy), axis=-1, keepdims=True)
        
        # Compute enclosed diagonal distance
        enclose_mins = K.minimum(true_mins, pred_mins)
        enclose_maxes = K.maximum(true_maxes, pred_maxes)
        enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
        enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1, keepdims=True)
        
        # Compute DIoU
        diou = iou - center_distance / (enclose_diagonal + K.epsilon())
        
        # Convert to loss (1 - DIoU)
        diou_loss = 1.0 - diou
        
        # Apply object mask
        diou_loss = diou_loss * object_mask
        
        return K.sum(diou_loss)


class CIoULoss(IoULoss):
    """
    Complete IoU (CIoU) loss.
    
    Reference: "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    https://arxiv.org/abs/1911.08287
    """
    
    def compute_loss(self, true_xy: tf.Tensor, true_wh: tf.Tensor,
                    pred_xy: tf.Tensor, pred_wh: tf.Tensor,
                    object_mask: tf.Tensor) -> tf.Tensor:
        """
        Compute CIoU loss.
        
        Args:
            true_xy: True center coordinates
            true_wh: True width and height
            pred_xy: Predicted center coordinates
            pred_wh: Predicted width and height
            object_mask: Object presence mask
            
        Returns:
            CIoU loss tensor
        """
        # Convert to corner coordinates
        true_mins = true_xy - true_wh / 2.0
        true_maxes = true_xy + true_wh / 2.0
        pred_mins = pred_xy - pred_wh / 2.0
        pred_maxes = pred_xy + pred_wh / 2.0
        
        # Compute intersection
        intersect_mins = K.maximum(true_mins, pred_mins)
        intersect_maxes = K.minimum(true_maxes, pred_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        # Compute areas
        true_area = true_wh[..., 0] * true_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        union_area = true_area + pred_area - intersect_area
        
        # Compute IoU
        iou = intersect_area / (union_area + K.epsilon())
        
        # Compute center distance
        center_distance = K.sum(K.square(true_xy - pred_xy), axis=-1, keepdims=True)
        
        # Compute enclosed diagonal distance
        enclose_mins = K.minimum(true_mins, pred_mins)
        enclose_maxes = K.maximum(true_maxes, pred_maxes)
        enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
        enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1, keepdims=True)
        
        # Compute DIoU
        diou = iou - center_distance / (enclose_diagonal + K.epsilon())
        
        # Compute aspect ratio consistency (CIoU extension)
        v = 4.0 * K.square(
            tf.math.atan2(true_wh[..., 0], true_wh[..., 1]) - 
            tf.math.atan2(pred_wh[..., 0], pred_wh[..., 1])
        ) / (math.pi * math.pi)
        
        # Compute alpha parameter
        alpha = v / (1.0 - iou + v + K.epsilon())
        
        # Compute CIoU
        ciou = diou - alpha * v
        
        # Convert to loss (1 - CIoU)
        ciou_loss = 1.0 - ciou
        
        # Apply object mask
        ciou_loss = ciou_loss * object_mask
        
        return K.sum(ciou_loss)


# Backward compatibility functions
def box_giou(b_true, b_pred):
    """Backward compatibility function for GIoU computation."""
    loss_fn = GIoULoss()
    return loss_fn.compute_loss(
        b_true[..., :2], b_true[..., 2:4],
        b_pred[..., :2], b_pred[..., 2:4],
        tf.ones_like(b_true[..., :1])
    )


def box_diou(b_true, b_pred, use_ciou=False):
    """Backward compatibility function for DIoU/CIoU computation."""
    if use_ciou:
        loss_fn = CIoULoss()
    else:
        loss_fn = DIoULoss()
    
    return loss_fn.compute_loss(
        b_true[..., :2], b_true[..., 2:4],
        b_pred[..., :2], b_pred[..., 2:4],
        tf.ones_like(b_true[..., :1])
    )
