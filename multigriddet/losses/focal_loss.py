#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focal loss implementations for MultiGridDet.
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from typing import Optional


class FocalLoss:
    """Base focal loss class."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
        """
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Prediction tensor
            
        Returns:
            Focal loss tensor
        """
        raise NotImplementedError("Subclasses must implement compute_loss method")


class SigmoidFocalLoss(FocalLoss):
    """
    Sigmoid focal loss for binary classification.
    
    Reference: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """
    
    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute sigmoid focal loss.
        
        Args:
            y_true: Ground truth tensor (0 or 1)
            y_pred: Prediction logits tensor
            
        Returns:
            Sigmoid focal loss tensor
        """
        # Convert logits to probabilities
        pred_prob = tf.sigmoid(y_pred)
        
        # Compute binary cross entropy
        bce_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)
        
        # Compute p_t
        p_t = y_true * pred_prob + (1 - y_true) * (1 - pred_prob)
        
        # Compute modulating factor
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Compute alpha weighting factor
        alpha_weight_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Compute focal loss
        focal_loss = modulating_factor * alpha_weight_factor * bce_loss
        
        return focal_loss


class SoftmaxFocalLoss(FocalLoss):
    """
    Softmax focal loss for multi-class classification.
    
    Reference: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """
    
    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute softmax focal loss.
        
        Args:
            y_true: Ground truth tensor (one-hot encoded)
            y_pred: Prediction logits tensor
            
        Returns:
            Softmax focal loss tensor
        """
        # Convert logits to probabilities
        pred_prob = tf.nn.softmax(y_pred, axis=-1)
        
        # Compute cross entropy
        ce_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=True)
        
        # Compute p_t
        p_t = tf.reduce_sum(y_true * pred_prob, axis=-1)
        
        # Compute modulating factor
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Compute focal loss
        focal_loss = modulating_factor * ce_loss
        
        return focal_loss


class InverseFocalLoss(FocalLoss):
    """
    Inverse focal loss for handling class imbalance differently.
    
    This variant uses inverse alpha weighting and can be useful
    for certain types of class imbalance.
    """
    
    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute inverse focal loss.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Prediction logits tensor
            
        Returns:
            Inverse focal loss tensor
        """
        # Convert logits to probabilities
        pred_prob = tf.sigmoid(y_pred)
        
        # Compute binary cross entropy
        bce_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)
        
        # Compute p_t
        p_t = y_true * pred_prob + (1 - y_true) * (1 - pred_prob)
        
        # Compute modulating factor
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Compute inverse alpha weighting factor
        alpha_weight_factor = y_true * (1 - self.alpha) + (1 - y_true) * self.alpha
        
        # Compute inverse focal loss
        focal_loss = modulating_factor * alpha_weight_factor * bce_loss
        
        return focal_loss


class FocalLossWithIoL(FocalLoss):
    """
    Focal loss with IoL (Intersection over Largest) confidence.
    
    This is used in MultiGridDet for anchor-specific objectness loss
    where the confidence is based on IoL scores.
    """
    
    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                    iol_conf: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute focal loss with IoL confidence.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Prediction logits tensor
            iol_conf: IoL confidence tensor (optional)
            
        Returns:
            Focal loss with IoL confidence tensor
        """
        # Convert logits to probabilities
        pred_prob = tf.sigmoid(y_pred)
        
        # Compute binary cross entropy
        bce_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)
        
        # Use IoL confidence if provided, otherwise use standard p_t
        if iol_conf is not None:
            p_t = iol_conf * pred_prob + (1 - y_true) * (1 - pred_prob)
            alpha_weight_factor = iol_conf * (1 - self.alpha) + (1 - y_true) * self.alpha
        else:
            p_t = y_true * pred_prob + (1 - y_true) * (1 - pred_prob)
            alpha_weight_factor = y_true * (1 - self.alpha) + (1 - y_true) * self.alpha
        
        # Compute modulating factor
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Compute focal loss
        focal_loss = modulating_factor * alpha_weight_factor * bce_loss
        
        return focal_loss


# Convenience functions for backward compatibility
def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Backward compatibility function for sigmoid focal loss."""
    loss_fn = SigmoidFocalLoss(alpha=alpha, gamma=gamma)
    return loss_fn.compute_loss(y_true, y_pred)


def sigmoid_inversefocal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Backward compatibility function for inverse focal loss."""
    loss_fn = InverseFocalLoss(alpha=alpha, gamma=gamma)
    return loss_fn.compute_loss(y_true, y_pred)


def sigmoid_inversefocal_loss_objness(y_true, y_pred, iol_conf, gamma=2.0, alpha=0.25):
    """Backward compatibility function for focal loss with IoL confidence."""
    loss_fn = FocalLossWithIoL(alpha=alpha, gamma=gamma)
    return loss_fn.compute_loss(y_true, y_pred, iol_conf)
