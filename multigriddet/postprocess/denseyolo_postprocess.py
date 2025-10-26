#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet postprocessing utilities.
Legacy postprocessing functions for MultiGridDet.
"""

import numpy as np
from .denseyolo_postprocess_standalone import denseyolo_decode, denseyolo_handle_predictions, denseyolo_correct_boxes, \
    denseyolo_adjust_boxes


def denseyolo2_decode(predictions, anchors, num_classes, input_dims, rescore_confidence=True):
    """
    YOLOv3 Head to process predictions from YOLOv3 models

    :param num_classes: Total number of classes
    :param anchors: YOLO style anchor list for bounding box assignment
    :param input_dims: Input dimensions of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """
    assert len(predictions) == len(anchors), 'anchor numbers does not match prediction.'
    num_layers = len(predictions)
    results = []
    for i, prediction in enumerate(predictions):
        #if i == 0:
        results.append(denseyolo_decode(prediction, anchors[i], num_classes, input_dims, output_layer_id=num_layers,
                                        rescore_confidence=rescore_confidence))
    return np.concatenate(results, axis=1)


def denseyolo2_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_image_size, max_boxes=100,
                              confidence=0.1, use_iol=True, nms_threshold=0.5, rescore_confidence=True):
    """
    Post-process YOLO outputs to get final detections.
    
    Args:
        yolo_outputs: List of YOLO prediction tensors
        image_shape: Original image shape (width, height)
        anchors: List of anchor arrays for each scale
        num_classes: Number of object classes
        model_image_size: Input size used by the model
        max_boxes: Maximum number of boxes to return
        confidence: Confidence threshold
        use_iol: Whether to use IoL (Intersection over Largest) for confidence rescoring
        nms_threshold: Non-maximum suppression threshold
        rescore_confidence: Whether to rescore confidence using IoL
        
    Returns:
        boxes: Array of bounding boxes [x1, y1, x2, y2]
        classes: Array of class indices
        scores: Array of confidence scores
    """
    predictions = denseyolo2_decode(yolo_outputs, anchors, num_classes, input_dims=model_image_size,
                                    rescore_confidence=rescore_confidence)
    predictions = denseyolo_correct_boxes(predictions, image_shape, model_image_size)

    boxes, classes, scores = denseyolo_handle_predictions(predictions,
                                                          image_shape,
                                                          max_boxes=max_boxes,
                                                          confidence=confidence,
                                                          nms_threshold=nms_threshold,
                                                          use_iol=use_iol,
                                                          rescore_confidence=rescore_confidence,
                                                          use_cluster_nms=True, use_wbf=False)

    boxes = denseyolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores

