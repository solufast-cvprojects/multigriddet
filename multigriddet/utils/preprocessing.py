#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image preprocessing utilities for MultiGridDet.
Migrated from common/data_utils.py to be self-contained.
"""

import numpy as np
from PIL import Image


def letterbox_resize(image, target_size, return_padding_info=False):
    """
    Resize image with unchanged aspect ratio using padding

    Args:
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value

    Returns:
        new_image: resized PIL Image object.
        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_w, src_h = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w/src_w, target_h/src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w)//2
    dy = (target_h - padding_h)//2
    offset = (dx, dy)

    # resize image with letterbox
    resized_image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))
    new_image.paste(resized_image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image


def normalize_image(image):
    """
    normalize image array from 0 ~ 255
    to 0.0 ~ 1.0

    Args:
        image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0

    Returns:
        image: numpy image array with dtype=float, 0.0 ~ 1.0
    """
    image = image.astype(np.float32) / 255.0
    return image


def preprocess_image(image, model_image_size):
    """
    Prepare model input image data with letterbox
    resize, normalize and dim expansion

    Args:
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).

    Returns:
        image_data: numpy array of image data for model input.
    """
    #resized_image = cv2.resize(image, tuple(reversed(model_image_size)), cv2.INTER_AREA)
    resized_image = letterbox_resize(image, tuple(reversed(model_image_size)))
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data


def preprocess_image_batch(image, model_image_size):
    """
    Prepare image for batched inference (no batch dim added).
    
    Args:
        image: PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).
        
    Returns:
        image_data: numpy array of shape (height, width, 3) - NO batch dim
    """
    resized_image = letterbox_resize(image, tuple(reversed(model_image_size)))
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    # NO expand_dims here - let the caller handle batching
    return image_data


def denormalize_image(image):
    """
    Denormalize image array from 0.0 ~ 1.0
    to 0 ~ 255

    Args:
        image: normalized image array with dtype=float, 0.0 ~ 1.0

    Returns:
        image: numpy image array with dtype=uint8, 0 ~ 255
    """
    image = (image * 255.0).astype(np.uint8)
    return image





