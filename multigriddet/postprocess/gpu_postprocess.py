#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated postprocessing for MultiGridDet evaluation.
Uses TensorFlow GPU operations for maximum performance.
"""

import tensorflow as tf
import numpy as np
from typing import List, Tuple, Union


def denseyolo2_decode_tf(predictions: List[tf.Tensor], 
                         anchors: List[np.ndarray], 
                         num_classes: int, 
                         input_dims: Tuple[int, int],
                         rescore_confidence: bool = True) -> tf.Tensor:
    """
    TensorFlow GPU version of denseyolo2_decode.
    
    Args:
        predictions: List of TF tensors (batch, grid_h, grid_w, anchors*(5+classes))
        anchors: List of anchor arrays for each scale
        num_classes: Number of object classes
        input_dims: Model input dimensions (height, width)
        rescore_confidence: Whether to rescore confidence using IoL
        
    Returns:
        decoded_predictions: (batch, num_boxes, 5+num_classes) tensor on GPU
    """
    batch_size = tf.shape(predictions[0])[0]
    num_layers = len(predictions)
    
    # Process each prediction layer
    all_predictions = []
    
    for i, prediction in enumerate(predictions):
        # Get grid size
        grid_size = tf.shape(prediction)[1:3]  # (grid_h, grid_w)
        
        # Create grid coordinates
        grid_y = tf.range(grid_size[0], dtype=tf.float32)
        grid_x = tf.range(grid_size[1], dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(grid_x, grid_y, indexing='ij')
        
        # Reshape to (grid_h*grid_w, 2)
        x_offset = tf.reshape(x_offset, [-1, 1])
        y_offset = tf.reshape(y_offset, [-1, 1])
        x_y_offset = tf.concat([x_offset, y_offset], axis=1)
        
        # Reshape to (1, grid_h, grid_w, 2) for broadcasting
        x_y_offset = tf.reshape(x_y_offset, [1, grid_size[0], grid_size[1], 2])
        
        # Ensure all tensors are float32 to avoid mixed precision issues
        prediction = tf.cast(prediction, tf.float32)
        
        # Extract raw predictions
        raw_xy = prediction[..., 0:2]
        raw_wh = prediction[..., 2:4]
        objectness = prediction[..., 4:5]
        anchor_probs = prediction[..., 5:5+len(anchors[i])]
        class_probs = prediction[..., 5+len(anchors[i]):]
        
        # Apply sigmoid to objectness
        objectness_probs = tf.nn.sigmoid(objectness)
        
        # Apply softmax to anchor and class probabilities
        anchor_probs = tf.nn.softmax(anchor_probs, axis=-1)
        class_probs = tf.nn.softmax(class_probs, axis=-1)
        
        # Decode box coordinates
        # Apply custom activation to xy (same as original)
        raw_xy_processed = (tf.nn.tanh(0.15 * raw_xy) + tf.nn.sigmoid(0.15 * raw_xy))
        box_xy = raw_xy_processed + x_y_offset
        
        # Get best anchor for each grid cell
        predicted_anchor_index = tf.argmax(anchor_probs, axis=-1, output_type=tf.int32)
        
        # Select anchors based on predicted indices
        anchors_tensor = tf.constant(anchors[i], dtype=tf.float32)
        selected_anchors = tf.gather(anchors_tensor, predicted_anchor_index)
        
        # Decode width and height
        box_wh = selected_anchors * tf.exp(raw_wh)
        
        # Normalize coordinates
        grid_size_float = tf.cast(grid_size, tf.float32)
        box_xy = box_xy / grid_size_float
        box_wh = box_wh / tf.cast(input_dims, tf.float32)
        
        # Rescore confidence if requested
        if rescore_confidence:
            max_anchor_probs = tf.reduce_max(anchor_probs, axis=-1, keepdims=True)
            class_probs = objectness_probs * max_anchor_probs * class_probs
        
        # Concatenate all components
        prediction_concat = tf.concat([box_xy, box_wh, objectness_probs, class_probs], axis=-1)
        
        # Reshape to (batch, grid_h*grid_w, 5+num_classes)
        grid_h, grid_w = grid_size[0], grid_size[1]
        prediction_reshaped = tf.reshape(prediction_concat, [batch_size, grid_h * grid_w, 5 + num_classes])
        
        all_predictions.append(prediction_reshaped)
    
    # Concatenate all layers
    decoded_predictions = tf.concat(all_predictions, axis=1)
    
    return decoded_predictions


def correct_boxes_tf(predictions: tf.Tensor, 
                     image_shapes: tf.Tensor, 
                     model_image_size: Tuple[int, int]) -> tf.Tensor:
    """
    Correct box coordinates from model space to original image space.
    
    Args:
        predictions: (batch, num_boxes, 5+num_classes) decoded predictions
        image_shapes: (batch, 2) original image shapes (height, width)
        model_image_size: Model input size (height, width)
        
    Returns:
        corrected_predictions: (batch, num_boxes, 5+num_classes) with corrected boxes
    """
    batch_size = tf.shape(predictions)[0]
    model_h, model_w = model_image_size
    
    # Extract components
    box_xy = predictions[..., 0:2]
    box_wh = predictions[..., 2:4]
    objectness = predictions[..., 4:5]
    class_probs = predictions[..., 5:]
    
    # Convert to (batch, 1, 1, 2) for broadcasting
    model_size = tf.constant([model_h, model_w], dtype=tf.float32)
    model_size = tf.reshape(model_size, [1, 1, 2])
    
    # Calculate scaling factors
    # image_shapes is (batch, 2) where 2 is (height, width)
    # We need to handle letterboxing
    image_h = tf.cast(image_shapes[:, 0:1], tf.float32)  # (batch, 1)
    image_w = tf.cast(image_shapes[:, 1:2], tf.float32)  # (batch, 1)
    
    # Calculate new shape (letterboxed)
    scale = tf.minimum(model_h / image_h, model_w / image_w)
    new_h = image_h * scale
    new_w = image_w * scale
    
    # Calculate offset
    offset_h = (model_h - new_h) / 2.0 / model_h
    offset_w = (model_w - new_w) / 2.0 / model_w
    
    # Scale factors
    scale_h = model_h / new_h
    scale_w = model_w / new_w
    
    # Reshape for broadcasting
    offset = tf.stack([offset_h, offset_w], axis=-1)  # (batch, 2)
    scale = tf.stack([scale_h, scale_w], axis=-1)      # (batch, 2)
    
    # Apply corrections
    box_xy = (box_xy - offset) * scale
    box_wh = box_wh * scale
    
    # Convert to absolute coordinates
    box_xy = box_xy * tf.stack([image_w, image_h], axis=-1)
    box_wh = box_wh * tf.stack([image_w, image_h], axis=-1)
    
    # Convert to x1, y1, x2, y2 format
    box_x1y1 = box_xy - box_wh / 2.0
    box_x2y2 = box_xy + box_wh / 2.0
    box_coords = tf.concat([box_x1y1, box_x2y2], axis=-1)
    
    # Concatenate back
    corrected_predictions = tf.concat([box_coords, objectness, class_probs], axis=-1)
    
    return corrected_predictions


def gpu_nms_boxes(boxes: tf.Tensor, 
                  scores: tf.Tensor, 
                  iou_threshold: float = 0.45,
                  score_threshold: float = 0.001,
                  max_boxes: int = 500) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    GPU-accelerated NMS using tf.image.combined_non_max_suppression.
    
    Args:
        boxes: (batch, num_boxes, 4) in [x1, y1, x2, y2] format
        scores: (batch, num_boxes, num_classes) class scores
        iou_threshold: NMS IoU threshold
        score_threshold: Confidence threshold
        max_boxes: Maximum detections per image
        
    Returns:
        nmsed_boxes: (batch, max_boxes, 4)
        nmsed_scores: (batch, max_boxes)
        nmsed_classes: (batch, max_boxes)
        valid_detections: (batch,) number of valid boxes per image
    """
    # Convert to [y1, x1, y2, x2] format for tf.image.combined_non_max_suppression
    boxes_yxyx = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)
    
    # Apply NMS
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = \
        tf.image.combined_non_max_suppression(
            boxes=tf.expand_dims(boxes_yxyx, axis=2),  # (batch, num_boxes, 1, 4)
            scores=scores,  # (batch, num_boxes, num_classes)
            max_output_size_per_class=max_boxes // tf.shape(scores)[-1],
            max_total_size=max_boxes,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )
    
    # Convert back to [x1, y1, x2, y2] format
    nmsed_boxes_xyxy = tf.stack([nmsed_boxes[..., 1], nmsed_boxes[..., 0], 
                                 nmsed_boxes[..., 3], nmsed_boxes[..., 2]], axis=-1)
    
    return nmsed_boxes_xyxy, nmsed_scores, nmsed_classes, valid_detections


def denseyolo2_postprocess_gpu(yolo_outputs: List[tf.Tensor], 
                               image_shapes: tf.Tensor, 
                               anchors: List[np.ndarray], 
                               num_classes: int,
                               model_image_size: Tuple[int, int], 
                               max_boxes: int = 500, 
                               confidence: float = 0.001, 
                               nms_threshold: float = 0.45,
                               rescore_confidence: bool = True,
                               use_iol: bool = True) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    GPU-accelerated postprocessing for batched evaluation.
    
    Args:
        yolo_outputs: List of TensorFlow tensors (batch, grid_h, grid_w, anchors*(5+classes))
        image_shapes: (batch, 2) original image shapes (height, width)
        anchors: List of anchor arrays for each scale
        num_classes: Number of object classes
        model_image_size: Model input size (height, width)
        max_boxes: Maximum detections per image
        confidence: Score threshold
        nms_threshold: NMS IoU threshold
        rescore_confidence: Whether to rescore confidence
        
    Returns:
        boxes: (batch, max_boxes, 4) in [x1, y1, x2, y2] format
        classes: (batch, max_boxes) class indices
        scores: (batch, max_boxes) confidence scores
        valid_detections: (batch,) number of valid boxes per image
    """
    # 1. Decode predictions on GPU
    predictions = denseyolo2_decode_tf(yolo_outputs, anchors, num_classes, model_image_size, rescore_confidence)
    
    # 2. Correct boxes from model space to image space
    predictions = correct_boxes_tf(predictions, image_shapes, model_image_size)
    
    # 3. Extract components for NMS
    boxes = predictions[..., :4]  # (batch, num_boxes, 4)
    objectness = predictions[..., 4:5]  # (batch, num_boxes, 1)
    class_probs = predictions[..., 5:]  # (batch, num_boxes, num_classes)
    
    # 4. Apply confidence threshold
    max_class_scores = tf.reduce_max(class_probs, axis=-1, keepdims=True)  # (batch, num_boxes, 1)
    combined_scores = objectness * max_class_scores  # (batch, num_boxes, 1)
    
    # Filter by confidence threshold
    confidence_mask = combined_scores >= confidence
    confidence_mask = tf.expand_dims(confidence_mask, axis=-1)  # (batch, num_boxes, 1)
    
    # Apply mask to boxes and scores
    boxes_masked = tf.where(confidence_mask, boxes, tf.zeros_like(boxes))
    scores_masked = tf.where(confidence_mask, class_probs, tf.zeros_like(class_probs))
    
    # 5. GPU NMS
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = gpu_nms_boxes(
        boxes_masked, scores_masked, nms_threshold, confidence, max_boxes
    )
    
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
