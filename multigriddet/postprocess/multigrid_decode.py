#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet decoding and postprocessing utilities.
Implements the MultiGrid dense prediction decoding strategy.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union, Dict, Any
from scipy.special import expit, softmax

from .nms import NMS, DIoUNMS, SoftNMS, ClusterNMS
from .wbf import WeightedBoxesFusion


class MultiGridDecoder:
    """
    MultiGridDet decoder for converting model outputs to bounding boxes.
    
    This decoder handles the MultiGridDet dense prediction outputs and
    converts them to standard bounding box format with proper confidence scoring.
    """
    
    def __init__(self, 
                 anchors: List[np.ndarray],
                 num_classes: int,
                 input_shape: Tuple[int, int] = (608, 608),
                 rescore_confidence: bool = True,
                 use_softmax: bool = True):
        """
        Initialize MultiGridDet decoder.
        
        Args:
            anchors: List of anchor arrays for each scale
            num_classes: Number of object classes
            input_shape: Input image shape (height, width)
            rescore_confidence: Whether to rescore confidence using IoL
            use_softmax: Whether to use softmax for anchor and class probabilities
        """
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.rescore_confidence = rescore_confidence
        self.use_softmax = use_softmax
        self.num_layers = len(anchors)
    
    def decode_predictions(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Decode MultiGridDet predictions to bounding boxes.
        
        Args:
            predictions: List of prediction tensors for each scale
            
        Returns:
            Decoded predictions tensor of shape (batch, num_boxes, 5 + num_classes)
        """
        if len(predictions) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} predictions, got {len(predictions)}")
        
        results = []
        
        for i, prediction in enumerate(predictions):
            decoded = self._decode_single_scale(
                prediction, 
                self.anchors[i], 
                i
            )
            results.append(decoded)
        
        # Concatenate results from all scales
        return np.concatenate(results, axis=1)
    
    def _decode_single_scale(self, prediction: np.ndarray, 
                           anchors: np.ndarray, 
                           scale_idx: int) -> np.ndarray:
        """
        Decode predictions for a single scale.
        
        Args:
            prediction: Prediction tensor for this scale
            anchors: Anchor array for this scale
            scale_idx: Scale index
            
        Returns:
            Decoded predictions for this scale
        """
        batch_size = prediction.shape[0]
        num_anchors = len(anchors)
        grid_size = prediction.shape[1:3]  # (height, width)
        
        # Create grid coordinates
        grid_y = np.arange(grid_size[0])
        grid_x = np.arange(grid_size[1])
        x_offset, y_offset = np.meshgrid(grid_x, grid_y)
        
        x_offset = np.reshape(x_offset, (-1, 1))
        y_offset = np.reshape(y_offset, (-1, 1))
        
        x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
        x_y_offset = np.reshape(x_y_offset, (-1, 2))
        x_y_offset = np.reshape(x_y_offset, (-1, grid_size[0], grid_size[1], 2))
        cell_grid = x_y_offset
        
        # Extract prediction components
        # Format: [tx, ty, tw, th, obj, anchor_probs..., class_probs...]
        raw_xy = prediction[..., 0:2]
        raw_wh = prediction[..., 2:4]
        objectness = prediction[..., 4:5]
        anchor_probs = prediction[..., 5:5+num_anchors]
        class_probs = prediction[..., 5+num_anchors:]
        
        # Apply activation functions
        if self.use_softmax:
            anchor_probs = softmax(anchor_probs, axis=-1)
            class_probs = softmax(class_probs, axis=-1)
        else:
            anchor_probs = expit(anchor_probs)
            class_probs = expit(class_probs)
        
        objectness_probs = expit(objectness)
        
        # MultiGridDet innovation: Use tanh + sigmoid for coordinate prediction
        # This helps with grid sensitivity elimination
        raw_xy = (np.tanh(0.15 * raw_xy) + expit(0.15 * raw_xy))
        
        # Convert to absolute coordinates
        box_xy = raw_xy + cell_grid
        box_xy /= grid_size  # Normalize to [0, 1]
        
        # Get best anchor for each grid cell
        predicted_anchor_index = np.argmax(anchor_probs, axis=-1)
        anchors_per_grid = np.take(anchors, predicted_anchor_index, axis=0)
        
        # Convert width and height
        box_wh = anchors_per_grid * np.exp(raw_wh)
        box_wh /= self.input_shape  # Normalize to [0, 1]
        
        # Rescore confidence using MultiGridDet strategy
        if self.rescore_confidence:
            # Use objectness, best anchor probability, and class probability
            best_anchor_prob = np.max(anchor_probs, axis=-1, keepdims=True)
            class_probs = objectness_probs * best_anchor_prob * class_probs
        
        # Combine all components
        prediction_decoded = np.concatenate([
            box_xy, box_wh, objectness_probs, class_probs
        ], axis=-1)
        
        # Reshape to (batch, num_boxes, features)
        prediction_decoded = np.reshape(
            prediction_decoded, 
            (batch_size, grid_size[0] * grid_size[1], self.num_classes + 5)
        )
        
        return prediction_decoded
    
    def correct_boxes(self, predictions: np.ndarray, 
                     image_shape: Tuple[int, int],
                     model_image_size: Tuple[int, int]) -> np.ndarray:
        """
        Correct bounding boxes back to original image shape.
        
        Args:
            predictions: Decoded predictions
            image_shape: Original image shape (height, width)
            model_image_size: Model input size (height, width)
            
        Returns:
            Corrected predictions
        """
        box_xy = predictions[..., 0:2]
        box_wh = predictions[..., 2:4]
        objectness_prob = np.expand_dims(predictions[..., 4], axis=-1)
        class_probs = predictions[..., 5:]
        
        # Convert to numpy arrays
        model_image_size = np.array(model_image_size, dtype='float32')
        image_shape = np.array(image_shape, dtype='float32')
        height, width = image_shape
        
        # Calculate scaling and offset for letterbox resize
        new_shape = np.round(image_shape * np.min(model_image_size / image_shape))
        offset = (model_image_size - new_shape) / 2.0 / model_image_size
        scale = model_image_size / new_shape
        
        # Reverse offset/scale to match (width, height) order
        offset = offset[..., ::-1]
        scale = scale[..., ::-1]
        
        # Apply correction
        box_xy = (box_xy - offset) * scale
        box_wh *= scale
        
        # Convert from center coordinates to corner coordinates
        box_xy -= box_wh / 2.0
        
        # Scale back to original image shape
        image_wh = image_shape[..., ::-1]  # (width, height)
        box_xy *= image_wh
        box_wh *= image_wh
        
        # Combine corrected components
        prediction_corrected = np.concatenate([
            box_xy, box_wh, objectness_prob, class_probs
        ], axis=-1)
        
        return prediction_corrected
    
    def handle_predictions(self, 
                          predictions: np.ndarray,
                          image_shape: Tuple[int, int],
                          max_boxes: int = 100,
                          confidence: float = 0.1,
                          nms_threshold: float = 0.5,
                          use_iol: bool = True,
                          nms_method: str = 'diou',
                          use_wbf: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle predictions with confidence filtering and NMS.
        
        Args:
            predictions: Corrected predictions
            image_shape: Image shape (height, width)
            max_boxes: Maximum number of boxes to return
            confidence: Confidence threshold
            nms_threshold: NMS IoU threshold
            use_iol: Whether to use IoL for NMS
            nms_method: NMS method ('diou', 'soft', 'cluster')
            use_wbf: Whether to use Weighted Boxes Fusion
            
        Returns:
            Tuple of (boxes, classes, scores)
        """
        boxes = predictions[..., 0:4]
        box_confidences = predictions[..., 4]
        box_class_probs = predictions[..., 5:]
        
        # Get class predictions
        box_scores = box_confidences
        box_classes = np.argmax(box_class_probs, axis=-1)
        
        # Filter by confidence
        pos = np.where(box_scores >= confidence)
        
        if len(pos[0]) == 0:
            return np.array([]), np.array([]), np.array([])
        
        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_scores[pos]
        
        # Apply NMS or WBF
        if use_wbf:
            # Use Weighted Boxes Fusion
            wbf = WeightedBoxesFusion()
            n_boxes, n_classes, n_scores = wbf.fuse_boxes(
                [boxes], [classes], [scores], 
                image_shape, 
                iou_thr=nms_threshold
            )
        else:
            # Use traditional NMS
            if nms_method == 'diou':
                nms = DIoUNMS(use_iol=use_iol)
            elif nms_method == 'soft':
                nms = SoftNMS()
            elif nms_method == 'cluster':
                nms = ClusterNMS(use_iol=use_iol)
            else:
                nms = NMS(use_iol=use_iol)
            
            n_boxes, n_classes, n_scores = nms.apply_nms(
                boxes, classes, scores, 
                nms_threshold, confidence
            )
        
        if n_boxes and len(n_boxes) > 0:
            # Concatenate results if multiple batches
            if isinstance(n_boxes, list):
                boxes = np.concatenate(n_boxes)
                classes = np.concatenate(n_classes).astype('int32')
                scores = np.concatenate(n_scores)
            else:
                boxes = n_boxes
                classes = n_classes.astype('int32')
                scores = n_scores
            
            # Filter to max_boxes
            boxes, classes, scores = self._filter_boxes(boxes, classes, scores, max_boxes)
            
            return boxes, classes, scores
        else:
            return np.array([]), np.array([]), np.array([])
    
    def _filter_boxes(self, boxes: np.ndarray, classes: np.ndarray, 
                     scores: np.ndarray, max_boxes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter boxes to maximum number.
        
        Args:
            boxes: Bounding boxes
            classes: Class labels
            scores: Confidence scores
            max_boxes: Maximum number of boxes
            
        Returns:
            Filtered boxes, classes, and scores
        """
        if len(boxes) <= max_boxes:
            return boxes, classes, scores
        
        # Sort by confidence score
        sorted_indices = np.argsort(scores)[::-1]
        
        # Take top max_boxes
        top_indices = sorted_indices[:max_boxes]
        
        return boxes[top_indices], classes[top_indices], scores[top_indices]
    
    def postprocess(self, 
                   yolo_outputs: List[np.ndarray],
                   image_shape: Tuple[int, int],
                   model_image_size: Tuple[int, int],
                   max_boxes: int = 100,
                   confidence: float = 0.1,
                   nms_threshold: float = 0.5,
                   use_iol: bool = True,
                   nms_method: str = 'diou',
                   use_wbf: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete postprocessing pipeline.
        
        Args:
            yolo_outputs: Raw model outputs
            image_shape: Original image shape
            model_image_size: Model input size
            max_boxes: Maximum number of boxes
            confidence: Confidence threshold
            nms_threshold: NMS threshold
            use_iol: Whether to use IoL
            nms_method: NMS method
            use_wbf: Whether to use WBF
            
        Returns:
            Tuple of (boxes, classes, scores)
        """
        # Decode predictions
        predictions = self.decode_predictions(yolo_outputs)
        
        # Correct boxes to original image shape
        predictions = self.correct_boxes(predictions, image_shape, model_image_size)
        
        # Handle predictions with NMS
        boxes, classes, scores = self.handle_predictions(
            predictions, image_shape, max_boxes, confidence, 
            nms_threshold, use_iol, nms_method, use_wbf
        )
        
        return boxes, classes, scores


# Backward compatibility functions
def denseyolo2_decode(predictions, anchors, num_classes, input_dims, rescore_confidence=True):
    """Backward compatibility function for DenseYOLO2 decode."""
    decoder = MultiGridDecoder(anchors, num_classes, input_dims, rescore_confidence)
    return decoder.decode_predictions(predictions)


def denseyolo2_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, 
                             model_image_size, max_boxes=100, confidence=0.1, 
                             use_iol=True, nms_threshold=0.5, rescore_confidence=True):
    """Backward compatibility function for DenseYOLO2 postprocess."""
    decoder = MultiGridDecoder(anchors, num_classes, model_image_size, rescore_confidence)
    return decoder.postprocess(
        yolo_outputs, image_shape, model_image_size, 
        max_boxes, confidence, nms_threshold, use_iol
    )


# Additional backward compatibility functions for denseyolo_postprocess.py
def denseyolo_decode(predictions, anchors, num_classes, input_dims, output_layer_id=None, rescore_confidence=True):
    """Decode single prediction tensor."""
    # Handle single prediction tensor
    if len(predictions.shape) == 4:  # Single prediction (batch, height, width, channels)
        batch_size = predictions.shape[0]
        grid_size = predictions.shape[1:3]
        num_anchors = len(anchors)
        
        # Create grid coordinates
        grid_y = np.arange(grid_size[0])
        grid_x = np.arange(grid_size[1])
        x_offset, y_offset = np.meshgrid(grid_x, grid_y)
        
        x_offset = np.reshape(x_offset, (-1, 1))
        y_offset = np.reshape(y_offset, (-1, 1))
        
        x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
        x_y_offset = np.reshape(x_y_offset, (-1, 2))
        x_y_offset = np.reshape(x_y_offset, (-1, grid_size[0], grid_size[1], 2))
        cell_grid = x_y_offset
        
        # Extract prediction components
        raw_xy = predictions[..., 0:2]
        raw_wh = predictions[..., 2:4]
        objectness = predictions[..., 4:5]
        anchor_probs = predictions[..., 5:5+num_anchors]
        class_probs = predictions[..., 5+num_anchors:]
        
        # Apply activation functions
        from scipy.special import expit, softmax
        anchor_probs = softmax(anchor_probs, axis=-1)
        class_probs = softmax(class_probs, axis=-1)
        objectness_probs = expit(objectness)
        
        # MultiGridDet innovation: Use tanh + sigmoid for coordinate prediction
        raw_xy = (np.tanh(0.15 * raw_xy) + expit(0.15 * raw_xy))
        
        # Convert to absolute coordinates
        box_xy = raw_xy + cell_grid
        box_xy /= grid_size  # Normalize to [0, 1]
        
        # Get best anchor for each grid cell
        predicted_anchor_index = np.argmax(anchor_probs, axis=-1)
        anchors_per_grid = np.take(anchors, predicted_anchor_index, axis=0)
        
        # Convert width and height
        box_wh = anchors_per_grid * np.exp(raw_wh)
        box_wh /= input_dims  # Normalize to [0, 1]
        
        # Rescore confidence using MultiGridDet strategy
        if rescore_confidence:
            best_anchor_prob = np.max(anchor_probs, axis=-1, keepdims=True)
            class_probs = objectness_probs * best_anchor_prob * class_probs
        
        # Combine all components
        prediction_decoded = np.concatenate([
            box_xy, box_wh, objectness_probs, class_probs
        ], axis=-1)
        
        # Reshape to (batch, num_boxes, features)
        prediction_decoded = np.reshape(
            prediction_decoded, 
            (batch_size, grid_size[0] * grid_size[1], num_classes + 5)
        )
        
        return prediction_decoded
    else:
        # Fallback to original method
        return denseyolo2_decode(predictions, anchors, num_classes, input_dims, rescore_confidence)


def denseyolo_handle_predictions(predictions, image_shape, max_boxes=100, confidence=0.1, 
                                nms_threshold=0.5, use_iol=True, nms_method='diou', use_wbf=False,
                                rescore_confidence=True, use_cluster_nms=False):
    """Handle predictions with confidence filtering and NMS."""
    decoder = MultiGridDecoder([], 0, (608, 608))  # Dummy decoder
    return decoder.handle_predictions(
        predictions, image_shape, max_boxes, confidence, 
        nms_threshold, use_iol, nms_method, use_wbf
    )


def denseyolo_correct_boxes(predictions, image_shape, model_image_size):
    """Correct bounding boxes back to original image shape."""
    decoder = MultiGridDecoder([], 0, model_image_size)  # Dummy decoder
    return decoder.correct_boxes(predictions, image_shape, model_image_size)


def denseyolo_adjust_boxes(boxes, image_shape, model_image_size):
    """Adjust boxes to original image coordinates."""
    # This is a simple wrapper around correct_boxes
    dummy_predictions = np.concatenate([boxes, np.zeros((boxes.shape[0], 1))], axis=-1)
    corrected = denseyolo_correct_boxes(dummy_predictions, image_shape, model_image_size)
    return corrected[..., :4]  # Return only the box coordinates
