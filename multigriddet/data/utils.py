#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Dataset utility functions for MultiGridDet."""

import os
import numpy as np
import time
import cv2, colorsys
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import tensorflow as tf


def get_multiscale_list():
    """
    Get list of multi-scale input shapes for training.
    
    Returns pre-defined valid shapes (all multiples of 32) centered around base shape (608x608).
    This avoids the 'None' division issue while enabling true multi-scale training.
    """
    # Pre-defined valid shapes (all multiples of 32)
    # Centered around base shape (608x608) for optimal performance
    input_shape_list = [
        (320, 320), (352, 352), (384, 384), (416, 416), (448, 448), 
        (480, 480), (512, 512), (544, 544), (576, 576), (608, 608),
        (640, 640), (672, 672)
    ]
    return input_shape_list


def validate_input_shape(shape):
    """
    Validate that input shape is compatible with model architecture.
    
    Args:
        shape: Tuple of (height, width)
        
    Returns:
        bool: True if shape is valid (multiple of 32)
    """
    if len(shape) != 2:
        return False
    
    height, width = shape
    return height % 32 == 0 and width % 32 == 0




def resize_anchors(base_anchors, target_shape, base_shape=(416,416)):
    '''
    original anchor size is clustered from COCO dataset
    under input shape (416,416). We need to resize it to
    our train input shape for better performance
    '''
    return np.around(base_anchors*target_shape[::-1]/base_shape[::-1])


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path, dynamic_whpair=False):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors_per_head = f.readlines()
    if dynamic_whpair:
        anchors = sorted([float(x) for x in anchors_per_head[0].split(',')])
        return anchors
    else:
        if len(anchors_per_head)==1:
            anchors = [float(x) for x in anchors_per_head[0].split(',')]
            return np.array(anchors).reshape(-1, 2)
        else:
            anchors = []
            for line in anchors_per_head:
                anchors.append(np.array([float(x) for x in line.split(',')]).reshape(-1, 2))
            return anchors       


def get_colors(number, bright=True):
    """
    Generate random colors for drawing bounding boxes.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    if number <= 0:
        return []

    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / number, 1., brightness)
                  for x in range(number)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def load_annotation_lines(annotation_file, shuffle=True):
    """
    Load annotation lines from file.
    
    Args:
        annotation_file: Path to annotation file
        shuffle: Whether to shuffle the lines
        
    Returns:
        List of annotation lines
    """
    with open(annotation_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if shuffle:
        np.random.seed(int(time.time()))
        np.random.shuffle(lines)

    return lines


def draw_label(image, text, color, coords):
    """Draw label on image."""
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image


def draw_boxes(image, boxes, classes, scores, class_names, colors, show_score=True):
    """Draw bounding boxes on image."""
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = map(int, box)

        class_name = class_names[cls]
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)

        # if no color info, use black(0,0,0)
        if colors == None:
            color = (0,0,0)
        else:
            color = colors[cls]
        cv2.circle(image, (int(xmin+(xmax-xmin)/2), int(ymin+(ymax-ymin)/2)), radius=1, color=color, thickness=-1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, (xmin, ymin))
    return image