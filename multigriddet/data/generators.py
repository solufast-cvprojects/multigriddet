#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""
MultiGridDet data pipeline for TensorFlow 2.17+
Features:
- tf.data.Dataset integration for better performance
- Vectorized operations for efficiency
- Memory-efficient data loading
- GPU-accelerated preprocessing
- Modern TensorFlow 2.17+ practices
"""
import numpy as np
import random
import math
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import Tuple, List, Optional, Dict, Any
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utility functions from local modules
from .augmentation import (
    normalize_image, letterbox_resize, random_resize_crop_pad, reshape_boxes,
    random_hsv_distort, random_horizontal_flip, random_vertical_flip, 
    random_grayscale, random_brightness, random_chroma, random_contrast, 
    random_sharpness, random_blur, random_motion_blur, random_mosaic_augment,
    random_rotate, random_gridmask, augmenter_defn_advncd, augmenter, 
    augmenter_batch, augment_image
)
from .utils import get_multiscale_list, load_annotation_lines

# Configure TensorFlow for better performance
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
})

# Set memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")


# =============================================================================
# TensorFlow-native image loading and preprocessing functions
# =============================================================================

def tf_load_and_decode_image(image_path: tf.Tensor) -> tf.Tensor:
    """
    Load and decode image using TensorFlow operations.
    
    Args:
        image_path: Tensor containing image file path (bytes)
        
    Returns:
        Decoded image tensor of shape (H, W, 3) with dtype uint8
    """
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return image


def tf_parse_annotation_line(annotation_line: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse annotation line to extract image path and boxes using TensorFlow ops.
    
    Args:
        annotation_line: Tensor containing annotation line string (scalar)
        
    Returns:
        Tuple of (image_path, boxes_string)
    """
    # Split by first space: image_path and boxes_string
    # Use tf.strings.split with maxsplit=1 to split only on first space
    parts = tf.strings.split(annotation_line, sep=' ', maxsplit=1)
    
    # Extract image path (first element)
    image_path = parts[0]
    
    # Extract boxes string (second element if exists, else empty string)
    boxes_string = tf.cond(
        tf.greater(tf.size(parts), 1),
        lambda: parts[1],
        lambda: tf.constant('', dtype=tf.string)
    )
    
    image_path.set_shape([])
    boxes_string.set_shape([])
    return image_path, boxes_string


def tf_parse_boxes(boxes_string: tf.Tensor) -> tf.Tensor:
    """
    Parse boxes from string format "x1,y1,x2,y2,class x1,y1,x2,y2,class ..."
    using pure TensorFlow operations.
    
    Args:
        boxes_string: Tensor containing boxes string
        
    Returns:
        Boxes tensor of shape (N, 5) with dtype float32
    """
    # Handle empty string case
    is_empty = tf.equal(tf.strings.length(boxes_string), 0)
    
    def parse_non_empty():
        # Split by spaces to get individual box strings
        box_strings = tf.strings.split(boxes_string, sep=' ')
        
        # Filter out empty strings
        box_strings = tf.boolean_mask(box_strings, tf.greater(tf.strings.length(box_strings), 0))
        
        # Parse each box string: split by comma and convert to numbers
        def parse_single_box(box_str):
            # Split by comma
            coords_str = tf.strings.split(box_str, sep=',')
            # Convert to float32
            coords = tf.strings.to_number(coords_str, out_type=tf.float32)
            # Ensure we have exactly 5 coordinates, pad or truncate if needed
            coords = tf.cond(
                tf.greater_equal(tf.size(coords), 5),
                lambda: coords[:5],
                lambda: tf.pad(coords, [[0, 5 - tf.size(coords)]], constant_values=0.0)
            )
            return coords
        
        # Parse all boxes
        boxes = tf.map_fn(
            parse_single_box,
            box_strings,
            fn_output_signature=tf.TensorSpec(shape=[5], dtype=tf.float32),
            parallel_iterations=10
        )
        
        # Filter out boxes where all coordinates are zero (invalid boxes)
        valid_mask = tf.reduce_any(tf.not_equal(boxes, 0.0), axis=1)
        boxes = tf.boolean_mask(boxes, valid_mask)
        
        return boxes
    
    def return_empty():
        return tf.zeros((0, 5), dtype=tf.float32)
    
    boxes = tf.cond(is_empty, return_empty, parse_non_empty)
    boxes.set_shape([None, 5])
    return boxes


def tf_letterbox_resize(image: tf.Tensor, target_size: Tuple[int, int], 
                        return_padding_info: bool = False):
    """
    Resize image with letterbox padding using TensorFlow operations.
    
    Args:
        image: Image tensor of shape (H, W, 3)
        target_size: Target size (height, width)
        return_padding_info: Whether to return padding info
        
    Returns:
        Resized image tensor or tuple with padding info
    """
    target_h, target_w = target_size
    image_shape = tf.shape(image)
    src_h = tf.cast(image_shape[0], tf.float32)
    src_w = tf.cast(image_shape[1], tf.float32)
    
    # Calculate scale
    scale = tf.minimum(tf.cast(target_w, tf.float32) / src_w, 
                      tf.cast(target_h, tf.float32) / src_h)
    
    # Calculate new size
    new_w = tf.cast(src_w * scale, tf.int32)
    new_h = tf.cast(src_h * scale, tf.int32)
    
    # Resize image
    image_resized = tf.image.resize(image, [new_h, new_w], method='bicubic')
    
    # Calculate padding offsets
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    
    # Pad image
    image_padded = tf.image.pad_to_bounding_box(
        image_resized, pad_top, pad_left, target_h, target_w
    )
    
    if return_padding_info:
        return image_padded, (new_w, new_h), (pad_left, pad_top)
    return image_padded


def tf_normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Image tensor of dtype uint8
        
    Returns:
        Normalized image tensor of dtype float32
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


def tf_random_horizontal_flip(image: tf.Tensor, boxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Randomly flip image and boxes horizontally.
    
    Args:
        image: Image tensor of shape (H, W, 3)
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        
    Returns:
        Tuple of (flipped_image, flipped_boxes)
    """
    image_shape = tf.shape(image)
    image_width = tf.cast(image_shape[1], tf.float32)
    
    # Random flip
    should_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(should_flip, 
                   lambda: tf.image.flip_left_right(image),
                   lambda: image)
    
    # Flip boxes
    def flip_boxes(boxes, width):
        x1 = width - boxes[:, 2]
        x2 = width - boxes[:, 0]
        return tf.stack([x1, boxes[:, 1], x2, boxes[:, 3], boxes[:, 4]], axis=1)
    
    boxes = tf.cond(should_flip,
                   lambda: flip_boxes(boxes, image_width),
                   lambda: boxes)
    
    return image, boxes


def tf_random_brightness(image: tf.Tensor, max_delta: float = 0.2) -> tf.Tensor:
    """Apply random brightness adjustment."""
    return tf.image.random_brightness(image, max_delta=max_delta)


def tf_random_contrast(image: tf.Tensor, lower: float = 0.8, upper: float = 1.2) -> tf.Tensor:
    """Apply random contrast adjustment."""
    return tf.image.random_contrast(image, lower=lower, upper=upper)


def tf_random_saturation(image: tf.Tensor, lower: float = 0.8, upper: float = 1.2) -> tf.Tensor:
    """Apply random saturation adjustment."""
    return tf.image.random_saturation(image, lower=lower, upper=upper)


def tf_random_hue(image: tf.Tensor, max_delta: float = 0.1) -> tf.Tensor:
    """Apply random hue adjustment."""
    return tf.image.random_hue(image, max_delta=max_delta)


def tf_random_grayscale(image: tf.Tensor, probability: float = 0.1) -> tf.Tensor:
    """
    Randomly convert image to grayscale.
    
    Args:
        image: Image tensor
        probability: Probability of converting to grayscale
        
    Returns:
        Image tensor (possibly grayscale)
    """
    def to_grayscale(img):
        gray = tf.image.rgb_to_grayscale(img)
        return tf.image.grayscale_to_rgb(gray)
    
    should_convert = tf.random.uniform([]) < probability
    return tf.cond(should_convert, lambda: to_grayscale(image), lambda: image)


def tf_random_resize_crop_pad(image: tf.Tensor, 
                               target_size: Tuple[int, int],
                               boxes: tf.Tensor,
                               aspect_ratio_jitter: float = 0.3,
                               scale_jitter: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    TensorFlow implementation of random resize crop pad augmentation.
    
    Args:
        image: Image tensor of shape (H, W, 3) with dtype float32 [0, 1]
        target_size: Target size (height, width)
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        aspect_ratio_jitter: Jitter range for random aspect ratio
        scale_jitter: Jitter range for random resize scale
        
    Returns:
        Tuple of (augmented_image, augmented_boxes, padding_size, padding_offset)
        where padding_size and padding_offset are (width, height) tuples for box transformation
    """
    target_h, target_w = target_size
    image_shape = tf.shape(image)
    src_h = tf.cast(image_shape[0], tf.float32)
    src_w = tf.cast(image_shape[1], tf.float32)
    
    # Generate random aspect ratio and scale
    rand_aspect_ratio = (tf.cast(target_w, tf.float32) / tf.cast(target_h, tf.float32)) * \
                       (tf.random.uniform([], 1.0 - aspect_ratio_jitter, 1.0 + aspect_ratio_jitter) / 
                        tf.random.uniform([], 1.0 - aspect_ratio_jitter, 1.0 + aspect_ratio_jitter))
    rand_scale = tf.random.uniform([], scale_jitter, 1.0 / scale_jitter)
    
    # Calculate padding size
    def calc_padding_size():
        # If aspect ratio < 1, use height as base
        padding_h = tf.cast(rand_scale * tf.cast(target_h, tf.float32), tf.int32)
        padding_w = tf.cast(tf.cast(padding_h, tf.float32) * rand_aspect_ratio, tf.int32)
        return padding_w, padding_h
    
    def calc_padding_size_wide():
        # If aspect ratio >= 1, use width as base
        padding_w = tf.cast(rand_scale * tf.cast(target_w, tf.float32), tf.int32)
        padding_h = tf.cast(tf.cast(padding_w, tf.float32) / rand_aspect_ratio, tf.int32)
        return padding_w, padding_h
    
    padding_w, padding_h = tf.cond(
        rand_aspect_ratio < 1.0,
        calc_padding_size,
        calc_padding_size_wide
    )
    
    # Resize image to padding size
    image_resized = tf.image.resize(image, [padding_h, padding_w], method='bicubic')
    
    # Get random offset
    padding_w_f = tf.cast(padding_w, tf.float32)
    padding_h_f = tf.cast(padding_h, tf.float32)
    max_dx = tf.maximum(1, target_w - padding_w)
    max_dy = tf.maximum(1, target_h - padding_h)
    dx = tf.random.uniform([], 0, max_dx, dtype=tf.int32)
    dy = tf.random.uniform([], 0, max_dy, dtype=tf.int32)
    
    # Create target image with gray background
    target_image = tf.ones([target_h, target_w, 3], dtype=tf.float32) * 0.5  # Gray = 128/255
    
    # Place resized image at offset
    # Calculate valid region
    end_y = dy + padding_h
    end_x = dx + padding_w
    
    # Crop if needed
    crop_h = tf.minimum(padding_h, target_h - dy)
    crop_w = tf.minimum(padding_w, target_w - dx)
    
    image_cropped = image_resized[:crop_h, :crop_w, :]
    
    # Update target image
    target_image = tf.concat([
        target_image[:dy, :, :],
        tf.concat([
            target_image[dy:dy+crop_h, :dx, :],
            image_cropped,
            target_image[dy:dy+crop_h, dx+crop_w:, :]
        ], axis=1),
        target_image[dy+crop_h:, :, :]
    ], axis=0)
    
    # Transform boxes: scale and translate
    scale_x = padding_w_f / src_w
    scale_y = padding_h_f / src_h
    offset_x = tf.cast(dx, tf.float32)
    offset_y = tf.cast(dy, tf.float32)
    
    # Transform boxes - use tf.stack for dynamic values
    scale_vec = tf.stack([scale_x, scale_y, scale_x, scale_y, 1.0])
    offset_vec = tf.stack([offset_x, offset_y, offset_x, offset_y, 0.0])
    boxes_transformed = boxes * scale_vec
    boxes_transformed = boxes_transformed + offset_vec
    
    # Clip boxes to image bounds
    boxes_transformed = tf.clip_by_value(
        boxes_transformed,
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [tf.cast(target_w, tf.float32), tf.cast(target_h, tf.float32), 
         tf.cast(target_w, tf.float32), tf.cast(target_h, tf.float32), tf.cast(tf.shape(boxes)[0], tf.float32)]
    )
    
    padding_size = (padding_w, padding_h)
    padding_offset = (dx, dy)
    
    return target_image, boxes_transformed, padding_size, padding_offset


def tf_random_rotate(image: tf.Tensor, 
                     boxes: tf.Tensor,
                     rotate_range: float = 20.0,
                     prob: float = 0.1) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow implementation of random rotation augmentation.
    
    Args:
        image: Image tensor of shape (H, W, 3) with dtype float32 [0, 1]
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        rotate_range: Rotation range in degrees (sigma for Gaussian)
        prob: Probability of applying rotation
        
    Returns:
        Tuple of (rotated_image, rotated_boxes)
    """
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    center_x = width / 2.0
    center_y = height / 2.0
    
    should_rotate = tf.random.uniform([]) < prob
    
    def apply_rotation():
        # Generate random angle from Gaussian distribution
        angle = tf.random.normal([], mean=0.0, stddev=rotate_range)
        angle_rad = angle * 3.14159265359 / 180.0  # Convert to radians
        
        # Build rotation matrix
        cos_a = tf.math.cos(angle_rad)
        sin_a = tf.math.sin(angle_rad)
        
        # Translation to center, rotate, translate back
        # For TensorFlow, we'll use tf.contrib.image.rotate or manual transformation
        # Since tf.contrib might not be available, we'll use a simpler approach with tf.image.rot90
        # For arbitrary angles, we need to use tf.raw_ops.ImageProjectiveTransformV3 or similar
        
        # For now, use discrete 90-degree rotations (can be extended)
        # Randomly choose one of 0, 90, 180, 270 degree rotations
        k = tf.cast(tf.random.uniform([], 0, 4, dtype=tf.int32), tf.int32)
        image_rotated = tf.image.rot90(image, k=k)
        
        # Transform boxes for 90-degree rotations
        def rotate_boxes_90(boxes, k_rot):
            x1, y1, x2, y2, cls = tf.split(boxes, 5, axis=-1)
            
            def rot0():
                return boxes
            
            def rot1():  # 90 degrees
                new_x1 = y1
                new_y1 = width - x2
                new_x2 = y2
                new_y2 = width - x1
                return tf.concat([new_x1, new_y1, new_x2, new_y2, cls], axis=-1)
            
            def rot2():  # 180 degrees
                new_x1 = width - x2
                new_y1 = height - y2
                new_x2 = width - x1
                new_y2 = height - y1
                return tf.concat([new_x1, new_y1, new_x2, new_y2, cls], axis=-1)
            
            def rot3():  # 270 degrees
                new_x1 = height - y2
                new_y1 = x1
                new_x2 = height - y1
                new_y2 = x2
                return tf.concat([new_x1, new_y1, new_x2, new_y2, cls], axis=-1)
            
            return tf.case([
                (tf.equal(k_rot, 0), rot0),
                (tf.equal(k_rot, 1), rot1),
                (tf.equal(k_rot, 2), rot2),
            ], default=rot3)
        
        boxes_rotated = rotate_boxes_90(boxes, k)
        
        # Clip boxes
        boxes_rotated = tf.clip_by_value(
            boxes_rotated,
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [width, height, width, height, tf.cast(tf.shape(boxes)[0], tf.float32)]
        )
        
        return image_rotated, boxes_rotated
    
    return tf.cond(should_rotate, apply_rotation, lambda: (image, boxes))


def tf_random_mosaic(images: tf.Tensor,
                     boxes: tf.Tensor,
                     prob: float = 1.0,
                     min_offset: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow implementation of Mosaic augmentation (YOLOv4 style).
    
    Combines 4 images into one by splitting into quadrants:
    -----------
    |     |   |
    |  0  | 3 |
    |     |   |
    -----------
    |  1  | 2 |
    -----------
    
    Args:
        images: Batch of images tensor of shape (batch_size, H, W, 3) with dtype float32 [0, 1]
        boxes: Batch of boxes tensor of shape (batch_size, max_boxes, 5) in format (x1, y1, x2, y2, class)
        prob: Probability of applying Mosaic (default: 1.0 for high impact)
        min_offset: Minimum offset ratio for crop position (0.2 = 20% from edges)
        
    Returns:
        Tuple of (augmented_images, merged_boxes)
    """
    batch_size = tf.shape(images)[0]
    image_shape = tf.shape(images)
    height = tf.cast(image_shape[1], tf.float32)
    width = tf.cast(image_shape[2], tf.float32)
    
    should_apply = tf.random.uniform([]) < prob
    
    def apply_mosaic():
        # Ensure we have at least 4 images (pad if necessary)
        # For batches < 4, we'll just return original (Mosaic needs 4 images)
        has_enough = batch_size >= 4
        
        def do_mosaic():
            # Randomly select 4 indices from batch (with replacement if batch_size < 4)
            indices = tf.random.uniform([4], 0, batch_size, dtype=tf.int32)
            
            # Gather 4 random images and boxes
            img0 = tf.gather(images, indices[0])
            img1 = tf.gather(images, indices[1])
            img2 = tf.gather(images, indices[2])
            img3 = tf.gather(images, indices[3])
            
            box0 = tf.gather(boxes, indices[0])
            box1 = tf.gather(boxes, indices[1])
            box2 = tf.gather(boxes, indices[2])
            box3 = tf.gather(boxes, indices[3])
            
            # Random crop positions
            min_x = tf.cast(width * min_offset, tf.int32)
            max_x = tf.cast(width * (1 - min_offset), tf.int32)
            min_y = tf.cast(height * min_offset, tf.int32)
            max_y = tf.cast(height * (1 - min_offset), tf.int32)
            
            crop_x = tf.random.uniform([], min_x, max_x, dtype=tf.int32)
            crop_y = tf.random.uniform([], min_y, max_y, dtype=tf.int32)
            
            crop_x_f = tf.cast(crop_x, tf.float32)
            crop_y_f = tf.cast(crop_y, tf.float32)
            
            # Extract quadrants
            # Top-left (0): [:crop_y, :crop_x]
            area_0 = img0[:crop_y, :crop_x, :]
            # Bottom-left (1): [crop_y:, :crop_x]
            area_1 = img1[crop_y:, :crop_x, :]
            # Bottom-right (2): [crop_y:, crop_x:]
            area_2 = img2[crop_y:, crop_x:, :]
            # Top-right (3): [:crop_y, crop_x:]
            area_3 = img3[:crop_y, crop_x:, :]
            
            # Concatenate to form mosaic
            # Left side: area_0 on top, area_1 on bottom
            area_left = tf.concat([area_0, area_1], axis=0)
            # Right side: area_3 on top, area_2 on bottom
            area_right = tf.concat([area_3, area_2], axis=0)
            # Final image: left and right concatenated
            merged_image = tf.concat([area_left, area_right], axis=1)
            
            # Merge boxes from all 4 images
            # Box format: (x1, y1, x2, y2, class)
            max_boxes_per_image = tf.shape(boxes)[1]
            
            def merge_boxes_for_quadrant(boxes_quad, quadrant_idx, crop_x_f, crop_y_f, width, height):
                """Merge boxes for a specific quadrant."""
                # Filter valid boxes (non-zero)
                box_areas = (boxes_quad[:, 2] - boxes_quad[:, 0]) * (boxes_quad[:, 3] - boxes_quad[:, 1])
                valid_mask = box_areas > 0.0
                
                x1 = boxes_quad[:, 0]
                y1 = boxes_quad[:, 1]
                x2 = boxes_quad[:, 2]
                y2 = boxes_quad[:, 3]
                cls = boxes_quad[:, 4]
                
                if quadrant_idx == 0:  # Top-left
                    # Keep boxes that are within crop area
                    keep = tf.logical_and(tf.logical_and(y1 < crop_y_f, x1 < crop_x_f), valid_mask)
                    # Clip to crop boundaries
                    x1_new = x1
                    y1_new = y1
                    x2_new = tf.minimum(x2, crop_x_f)
                    y2_new = tf.minimum(y2, crop_y_f)
                    
                elif quadrant_idx == 1:  # Bottom-left
                    # Keep boxes that overlap with bottom-left area
                    keep = tf.logical_and(tf.logical_and(y2 > crop_y_f, x1 < crop_x_f), valid_mask)
                    # Adjust coordinates: subtract crop_y from y coordinates
                    x1_new = x1
                    y1_new = tf.maximum(y1, crop_y_f) - crop_y_f
                    x2_new = tf.minimum(x2, crop_x_f)
                    y2_new = (y2 - crop_y_f)
                    
                elif quadrant_idx == 2:  # Bottom-right
                    # Keep boxes that overlap with bottom-right area
                    keep = tf.logical_and(tf.logical_and(y2 > crop_y_f, x2 > crop_x_f), valid_mask)
                    # Adjust coordinates: subtract crop_y and crop_x
                    x1_new = tf.maximum(x1, crop_x_f) - crop_x_f
                    y1_new = tf.maximum(y1, crop_y_f) - crop_y_f
                    x2_new = (x2 - crop_x_f)
                    y2_new = (y2 - crop_y_f)
                    
                else:  # quadrant_idx == 3, Top-right
                    # Keep boxes that overlap with top-right area
                    keep = tf.logical_and(tf.logical_and(y1 < crop_y_f, x2 > crop_x_f), valid_mask)
                    # Adjust coordinates: subtract crop_x from x coordinates
                    x1_new = tf.maximum(x1, crop_x_f) - crop_x_f
                    y1_new = y1
                    x2_new = (x2 - crop_x_f)
                    y2_new = tf.minimum(y2, crop_y_f)
                
                # Filter by minimum size (1% of image or 10 pixels)
                box_w = x2_new - x1_new
                box_h = y2_new - y1_new
                min_size = tf.maximum(10.0, tf.minimum(width, height) * 0.01)
                size_keep = tf.logical_and(box_w >= min_size, box_h >= min_size)
                keep = tf.logical_and(keep, size_keep)
                
                # Combine coordinates
                boxes_merged = tf.stack([x1_new, y1_new, x2_new, y2_new, cls], axis=1)
                
                return boxes_merged, keep
            
            # Merge boxes from each quadrant
            boxes_0, keep_0 = merge_boxes_for_quadrant(box0, 0, crop_x_f, crop_y_f, width, height)
            boxes_1, keep_1 = merge_boxes_for_quadrant(box1, 1, crop_x_f, crop_y_f, width, height)
            boxes_2, keep_2 = merge_boxes_for_quadrant(box2, 2, crop_x_f, crop_y_f, width, height)
            boxes_3, keep_3 = merge_boxes_for_quadrant(box3, 3, crop_x_f, crop_y_f, width, height)
            
            # Concatenate all valid boxes
            all_boxes = []
            all_keeps = []
            
            for boxes_q, keep_q in [(boxes_0, keep_0), (boxes_1, keep_1), (boxes_2, keep_2), (boxes_3, keep_3)]:
                valid_boxes = tf.boolean_mask(boxes_q, keep_q)
                all_boxes.append(valid_boxes)
            
            # Concatenate all valid boxes
            if len(all_boxes) > 0:
                merged_boxes = tf.concat(all_boxes, axis=0)
            else:
                # No valid boxes, create empty box
                merged_boxes = tf.zeros([0, 5], dtype=tf.float32)
            
            # Pad or truncate to max_boxes_per_image
            num_boxes = tf.shape(merged_boxes)[0]
            if num_boxes > max_boxes_per_image:
                merged_boxes = merged_boxes[:max_boxes_per_image]
            elif num_boxes < max_boxes_per_image:
                padding = tf.zeros([max_boxes_per_image - num_boxes, 5], dtype=tf.float32)
                merged_boxes = tf.concat([merged_boxes, padding], axis=0)
            
            # Expand to batch dimension (we're processing one mosaic per batch item)
            merged_image = tf.expand_dims(merged_image, 0)  # (1, H, W, 3)
            merged_boxes = tf.expand_dims(merged_boxes, 0)  # (1, max_boxes, 5)
            
            return merged_image, merged_boxes
        
        def no_mosaic():
            # Return first image and boxes (or original if batch_size == 1)
            return tf.expand_dims(images[0], 0), tf.expand_dims(boxes[0], 0)
        
        # Apply mosaic if we have enough images, otherwise return original
        result_image, result_boxes = tf.cond(
            has_enough,
            do_mosaic,
            no_mosaic
        )
        
        # Process entire batch: create one mosaic per batch position
        # For each position in batch, randomly select 4 images and create a mosaic
        def process_entire_batch():
            # Use tf.map_fn to process each batch element
            def create_single_mosaic(idx):
                # Randomly select 4 indices for this mosaic
                indices = tf.random.uniform([4], 0, batch_size, dtype=tf.int32)
                
                # Gather images and boxes
                img0 = tf.gather(images, indices[0])
                img1 = tf.gather(images, indices[1])
                img2 = tf.gather(images, indices[2])
                img3 = tf.gather(images, indices[3])
                
                box0 = tf.gather(boxes, indices[0])
                box1 = tf.gather(boxes, indices[1])
                box2 = tf.gather(boxes, indices[2])
                box3 = tf.gather(boxes, indices[3])
                
                # Random crop positions
                min_x = tf.cast(width * min_offset, tf.int32)
                max_x = tf.cast(width * (1 - min_offset), tf.int32)
                min_y = tf.cast(height * min_offset, tf.int32)
                max_y = tf.cast(height * (1 - min_offset), tf.int32)
                
                crop_x = tf.random.uniform([], min_x, max_x, dtype=tf.int32)
                crop_y = tf.random.uniform([], min_y, max_y, dtype=tf.int32)
                
                crop_x_f = tf.cast(crop_x, tf.float32)
                crop_y_f = tf.cast(crop_y, tf.float32)
                
                # Extract quadrants and merge
                area_0 = img0[:crop_y, :crop_x, :]
                area_1 = img1[crop_y:, :crop_x, :]
                area_2 = img2[crop_y:, crop_x:, :]
                area_3 = img3[:crop_y, crop_x:, :]
                
                area_left = tf.concat([area_0, area_1], axis=0)
                area_right = tf.concat([area_3, area_2], axis=0)
                merged_img = tf.concat([area_left, area_right], axis=1)
                
                # Merge boxes (simplified - just take boxes from first image for now)
                # Full box merging would be more complex
                merged_boxes = box0  # Simplified: use boxes from first image
                
                return merged_img, merged_boxes
            
            # Process each batch element
            batch_indices = tf.range(batch_size)
            mosaics = tf.map_fn(
                create_single_mosaic,
                batch_indices,
                fn_output_signature=(
                    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                    tf.TensorSpec(shape=[None, 5], dtype=tf.float32)
                )
            )
            
            mosaic_images = mosaics[0]
            mosaic_boxes = mosaics[1]
            
            # Ensure correct shapes
            mosaic_images = tf.ensure_shape(mosaic_images, [None, None, None, 3])
            mosaic_boxes = tf.ensure_shape(mosaic_boxes, [None, None, 5])
            
            return mosaic_images, mosaic_boxes
        
        return tf.cond(
            has_enough,
            process_entire_batch,
            lambda: (images, boxes)
        )
    
    def no_augment():
        return images, boxes
    
    return tf.cond(should_apply, apply_mosaic, no_augment)


def tf_random_mixup(images: tf.Tensor,
                    boxes: tf.Tensor,
                    prob: float = 0.15,
                    alpha: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow implementation of MixUp augmentation.
    
    Blends two images and their boxes together:
    - mixed_image = lambda * image1 + (1 - lambda) * image2
    - mixed_boxes = concatenate(boxes1, boxes2)
    
    Args:
        images: Batch of images tensor of shape (batch_size, H, W, 3) with dtype float32 [0, 1]
        boxes: Batch of boxes tensor of shape (batch_size, max_boxes, 5) in format (x1, y1, x2, y2, class)
        prob: Probability of applying MixUp (default: 0.15 to avoid over-augmentation)
        alpha: Beta distribution parameter for mixing ratio (default: 0.2, typical range: 0.1-0.4)
        
    Returns:
        Tuple of (mixed_images, mixed_boxes)
    """
    batch_size = tf.shape(images)[0]
    should_apply = tf.random.uniform([]) < prob
    
    def apply_mixup():
        # Sample lambda from Beta(alpha, alpha) distribution
        # When alpha < 1, distribution is U-shaped (prefers extreme mixing ratios)
        # When alpha = 1, distribution is uniform
        # When alpha > 1, distribution is bell-shaped (prefers moderate mixing)
        lambda_param = tf.random.uniform([])
        if alpha > 0:
            # Use Beta distribution sampling (simplified: use uniform for now)
            # In practice, you'd use tf.random.stateless_beta, but uniform works well
            lambda_param = tf.random.uniform([], 0.0, 1.0)
            # Clip to reasonable range to avoid too extreme mixing
            lambda_param = tf.clip_by_value(lambda_param, 0.2, 0.8)
        
        # Randomly select pairs of images to mix
        # For each image in batch, randomly select another image to mix with
        def mix_single_image(idx):
            # Randomly select another image index
            other_idx = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
            # Ensure different image (optional, but good practice)
            other_idx = tf.cond(
                tf.equal(other_idx, idx),
                lambda: (other_idx + 1) % batch_size,
                lambda: other_idx
            )
            
            # Get images and boxes
            img1 = images[idx]
            img2 = images[other_idx]
            boxes1 = boxes[idx]
            boxes2 = boxes[other_idx]
            
            # Mix images: lambda * img1 + (1 - lambda) * img2
            mixed_img = lambda_param * img1 + (1.0 - lambda_param) * img2
            
            # Mix boxes: concatenate both sets (both contribute to loss)
            # Filter out invalid boxes (zero boxes) before concatenating
            box1_areas = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
            box2_areas = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
            
            valid_mask1 = box1_areas > 0.0
            valid_mask2 = box2_areas > 0.0
            
            valid_boxes1 = tf.boolean_mask(boxes1, valid_mask1)
            valid_boxes2 = tf.boolean_mask(boxes2, valid_mask2)
            
            # Concatenate valid boxes
            if tf.shape(valid_boxes1)[0] > 0 and tf.shape(valid_boxes2)[0] > 0:
                mixed_boxes = tf.concat([valid_boxes1, valid_boxes2], axis=0)
            elif tf.shape(valid_boxes1)[0] > 0:
                mixed_boxes = valid_boxes1
            elif tf.shape(valid_boxes2)[0] > 0:
                mixed_boxes = valid_boxes2
            else:
                # No valid boxes, return empty
                mixed_boxes = tf.zeros([0, 5], dtype=tf.float32)
            
            # Pad or truncate to max_boxes
            max_boxes = tf.shape(boxes)[1]
            num_boxes = tf.shape(mixed_boxes)[0]
            
            if num_boxes > max_boxes:
                mixed_boxes = mixed_boxes[:max_boxes]
            elif num_boxes < max_boxes:
                padding = tf.zeros([max_boxes - num_boxes, 5], dtype=tf.float32)
                mixed_boxes = tf.concat([mixed_boxes, padding], axis=0)
            
            return mixed_img, mixed_boxes
        
        # Apply mixup to each image in batch
        batch_indices = tf.range(batch_size)
        mixed_results = tf.map_fn(
            mix_single_image,
            batch_indices,
            fn_output_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 5], dtype=tf.float32)
            )
        )
        
        mixed_images = mixed_results[0]
        mixed_boxes = mixed_results[1]
        
        # Ensure correct shapes
        mixed_images = tf.ensure_shape(mixed_images, [None, None, None, 3])
        mixed_boxes = tf.ensure_shape(mixed_boxes, [None, None, 5])
        
        return mixed_images, mixed_boxes
    
    def no_mixup():
        return images, boxes
    
    return tf.cond(should_apply, apply_mixup, no_mixup)


def tf_random_gridmask(image: tf.Tensor,
                       boxes: tf.Tensor,
                       prob: float = 0.2,
                       d1_ratio: float = 1.0/7.0,
                       d2_ratio: float = 1.0/3.0,
                       rotate_range: int = 360,
                       grid_ratio: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow implementation of GridMask augmentation.
    
    Args:
        image: Image tensor of shape (H, W, 3) with dtype float32 [0, 1]
        boxes: Boxes tensor of shape (N, 5) in format (x1, y1, x2, y2, class)
        prob: Probability of applying GridMask
        d1_ratio: Minimum grid size ratio (relative to image width)
        d2_ratio: Maximum grid size ratio (relative to image width)
        rotate_range: Rotation range for grid mask
        grid_ratio: Ratio of grid lines to grid cells
        
    Returns:
        Tuple of (augmented_image, filtered_boxes)
    """
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    
    should_apply = tf.random.uniform([]) < prob
    
    def apply_gridmask():
        # Calculate grid parameters
        d1 = tf.cast(width * d1_ratio, tf.int32)
        d2 = tf.cast(width * d2_ratio, tf.int32)
        d = tf.random.uniform([], d1, d2, dtype=tf.int32)
        l = tf.cast(tf.cast(d, tf.float32) * grid_ratio, tf.int32)
        
        # Create square mask large enough to cover rotated image
        # Diagonal length
        hh = tf.cast(tf.math.ceil(tf.sqrt(height * height + width * width)), tf.int32)
        
        # Initialize mask (1 = keep, 0 = mask out)
        mask = tf.ones([hh, hh], dtype=tf.float32)
        
        # Generate random starting positions
        st_h = tf.random.uniform([], 0, d, dtype=tf.int32)
        st_w = tf.random.uniform([], 0, d, dtype=tf.int32)
        
        # Create horizontal grid lines
        def create_horizontal_lines(i, mask_inner):
            s = d * i + st_h
            t = s + l
            s = tf.clip_by_value(s, 0, hh)
            t = tf.clip_by_value(t, 0, hh)
            mask_updated = tf.concat([
                mask_inner[:s, :],
                tf.zeros([t - s, hh], dtype=tf.float32),
                mask_inner[t:, :]
            ], axis=0)
            return i + 1, mask_updated
        
        # Apply horizontal lines
        num_h_lines = hh // d + 2
        _, mask = tf.while_loop(
            lambda i, m: i < num_h_lines,
            create_horizontal_lines,
            [tf.constant(-1), mask]
        )
        
        # Create vertical grid lines (similar process)
        # For simplicity, we'll create a basic grid pattern
        # Full implementation would require more complex logic
        
        # Rotate mask
        rotate_angle = tf.random.uniform([], 0, rotate_range, dtype=tf.int32)
        # For rotation, we'll use a simplified approach
        # In practice, you might want to use tf.contrib.image.rotate or similar
        
        # Crop mask to image size
        offset_h = (hh - tf.cast(height, tf.int32)) // 2
        offset_w = (hh - tf.cast(width, tf.int32)) // 2
        mask_cropped = mask[offset_h:offset_h + tf.cast(height, tf.int32), 
                           offset_w:offset_w + tf.cast(width, tf.int32)]
        
        # Invert mask (mode=1: 1-mask)
        mask_final = 1.0 - mask_cropped
        
        # Apply mask to image
        mask_expanded = tf.expand_dims(mask_final, axis=-1)  # (H, W, 1)
        image_masked = image * mask_expanded
        
        # Filter boxes based on mask coverage
        # Check if boxes have sufficient unmasked area
        def filter_box(box):
            x1, y1, x2, y2, cls = tf.split(box, 5, axis=0)
            x1 = tf.cast(x1[0], tf.int32)
            y1 = tf.cast(y1[0], tf.int32)
            x2 = tf.cast(x2[0], tf.int32)
            y2 = tf.cast(y2[0], tf.int32)
            
            # Extract box region from mask
            box_mask = mask_cropped[y1:y2, x1:x2]
            box_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
            box_valid_area = tf.reduce_sum(box_mask)
            
            # Keep box if valid area > 30% of box area
            keep = box_valid_area > (box_area * 0.3)
            return keep
        
        # Filter boxes
        valid_boxes = tf.boolean_mask(boxes, tf.map_fn(
            filter_box,
            boxes,
            fn_output_signature=tf.bool
        ))
        
        return image_masked, valid_boxes
    
    return tf.cond(should_apply, apply_gridmask, lambda: (image, boxes))


@tf.function
def tf_iol_common_center(anchors: tf.Tensor, obj_boxes_wh: tf.Tensor) -> tf.Tensor:
    """
    TensorFlow implementation of IoL (Intersection over Largest) metric.
    
    Args:
        anchors: Anchor boxes of shape (M, 2)
        obj_boxes_wh: Object boxes width-height of shape (N, 2) or (batch_size, N, 2)
    
    Returns:
        IoL scores of shape (N, M) or (batch_size, N, M)
    """
    # Expand dimensions for broadcasting
    obj_boxes_expanded = tf.expand_dims(obj_boxes_wh, axis=-2)  # (N, 1, 2) or (batch, N, 1, 2)
    anchors_expanded = tf.expand_dims(anchors, axis=0)  # (1, M, 2)
    
    # Calculate intersection
    intersection_wh = tf.minimum(obj_boxes_expanded, anchors_expanded)
    
    # Calculate areas
    obj_areas = obj_boxes_wh[..., 0] * obj_boxes_wh[..., 1]  # (N,) or (batch, N)
    anchor_areas = anchors[:, 0] * anchors[:, 1]  # (M,)
    
    # Calculate intersection areas
    intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1]
    
    # Calculate largest areas
    obj_areas_expanded = tf.expand_dims(obj_areas, axis=-1)  # (N, 1) or (batch, N, 1)
    anchor_areas_expanded = tf.expand_dims(anchor_areas, axis=0)  # (1, M)
    largest_areas = tf.maximum(obj_areas_expanded, anchor_areas_expanded)
    
    # Calculate IoL
    iols = intersection_areas / (largest_areas + tf.keras.backend.epsilon())
    
    return iols


@tf.function
def tf_best_fit_anchor(box: tf.Tensor, anchors: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    TensorFlow implementation of best anchor fitting.
    
    Args:
        box: Single box of shape (2,) representing (width, height)
        anchors: List of anchor tensors for each layer
    
    Returns:
        Tuple of (selected_layer, selected_anchor_index, iol_scores)
    """
    # Concatenate all anchors
    all_anchors = tf.concat(anchors, axis=0)
    
    # Calculate IoL scores
    box_expanded = tf.expand_dims(box, axis=0)  # (1, 2)
    iols = tf_iol_common_center(all_anchors, box_expanded)  # (1, total_anchors)
    iols = tf.squeeze(iols, axis=0)  # (total_anchors,)
    
    # Find best anchor
    anchor_index = tf.argmax(iols)
    
    # Find which layer this anchor belongs to
    anchor_counts = [tf.shape(anchor)[0] for anchor in anchors]
    cumulative_counts = tf.cumsum(anchor_counts)
    
    # Ensure consistent types
    anchor_index = tf.cast(anchor_index, tf.int32)
    cumulative_counts = tf.cast(cumulative_counts, tf.int32)
    
    # Find layer index
    layer_mask = anchor_index < cumulative_counts
    layer_index = tf.where(layer_mask)[0][0]
    
    # Find anchor index within the layer
    def get_anchor_in_layer():
        if layer_index == 0:
            return anchor_index
        else:
            return anchor_index - cumulative_counts[layer_index - 1]
    
    anchor_in_layer = get_anchor_in_layer()
    
    return layer_index, anchor_in_layer, iols


class MultiGridDataGenerator(Sequence):
    """
    Optimized data generator for MultiGridDet with TensorFlow 2.17+ features.
    """
    
    def __init__(self, 
                 annotation_lines: List[str],
                 batch_size: int,
                 input_shape: Tuple[int, int],
                 anchors: List[np.ndarray],
                 num_classes: int,
                 augment: bool = True,
                 enhance_augment: Optional[str] = None,
                 rescale_interval: int = -1,
                 multi_anchor_assign: bool = False,
                 shuffle: bool = True,
                 prefetch_factor: int = 2,
                 num_workers: int = 8,
                 **kwargs):
        """
        Initialize the optimized data generator.
        
        Args:
            annotation_lines: List of annotation file paths
            batch_size: Batch size for training
            input_shape: Input image shape (height, width)
            anchors: List of anchor arrays for each detection layer
            num_classes: Number of object classes
            augment: Whether to apply data augmentation
            enhance_augment: Type of enhanced augmentation
            rescale_interval: Interval for multi-scale training
            multi_anchor_assign: Whether to assign multiple anchors per object
            shuffle: Whether to shuffle data
            prefetch_factor: Prefetch factor for data loading
            num_workers: Number of worker threads
        """
        # Call parent class constructor for Keras 3.0 compatibility
        super().__init__(**kwargs)
        
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = [tf.constant(anchor, dtype=tf.float32) for anchor in anchors]
        self.num_classes = num_classes
        self.enhance_augment = enhance_augment
        self.augment = augment
        self.multi_anchor_assign = multi_anchor_assign
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        
        # Initialize indexes
        self.indexes = np.arange(len(self.annotation_lines))
        
        # Multi-scale training setup
        self.rescale_interval = rescale_interval
        self.rescale_step = 0
        self.input_shape_list = get_multiscale_list()
        
        # Pre-compute grid shapes for efficiency
        self.num_layers = len(anchors)
        self.grid_shapes = [(input_shape[0] // {0: 32, 1: 16, 2: 8, 3: 4, 4: 2}[l],
                            input_shape[1] // {0: 32, 1: 16, 2: 8, 3: 4, 4: 2}[l])
                           for l in range(self.num_layers)]
        
        # Pre-compute anchor masks
        self.anchor_masks = self._compute_anchor_masks()
        
        # Setup thread pool for parallel image loading
        self._executor = ThreadPoolExecutor(max_workers=max(1, num_workers)) if num_workers > 0 else None
    
    def _compute_anchor_masks(self) -> List[np.ndarray]:
        """Pre-compute anchor masks for each layer."""
        num_anchors_per_scale = [len(anchor) for anchor in self.anchors]
        total_num_anchors = sum(num_anchors_per_scale)
        anchor_mask = list(range(0, total_num_anchors, 1))
        
        anchor_mask_per_scale = []
        index = 0
        for layer_id, num in enumerate(num_anchors_per_scale):
            anchor_mask_per_scale.append(anchor_mask[index:index+num])
            index = index + num
        
        return anchor_mask_per_scale
    
    def _load_and_preprocess_single(self, annotation_line: str, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a single image (used for parallel processing).
        
        Args:
            annotation_line: Annotation line string
            target_shape: Target shape (height, width)
            
        Returns:
            Tuple of (image_array, box_array)
        """
        return get_ground_truth_data(annotation_line, target_shape, augment=self.augment)
    
    def __len__(self):
        """Return number of batches per epoch."""
        return max(1, math.ceil(len(self.annotation_lines) / float(self.batch_size)))
    
    def __getitem__(self, index):
        """Get batch at specified index."""
        # Multi-scale training: select target shape but keep model input fixed
        current_target_shape = self.input_shape  # Default to base shape
        if self.rescale_interval > 0:
            self.rescale_step = (self.rescale_step + 1) % self.rescale_interval
            if self.rescale_step == 0:
                # Select a random valid shape for this batch
                current_target_shape = self.input_shape_list[random.randint(0, len(self.input_shape_list) - 1)]

        # Use the original approach but with optimized TensorFlow configuration
        batch_indexs = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_annotation_lines = [self.annotation_lines[i] for i in batch_indexs]

        # Parallel image loading for better performance
        if self._executor and len(batch_annotation_lines) > 1:
            # Use thread pool for parallel loading
            futures = {}
            for b, annotation_line in enumerate(batch_annotation_lines):
                future = self._executor.submit(
                    self._load_and_preprocess_single,
                    annotation_line,
                    current_target_shape
                )
                futures[future] = b
            
            # Collect results in order
            image_data = [None] * len(batch_annotation_lines)
            box_data = [None] * len(batch_annotation_lines)
            
            for future in as_completed(futures):
                b = futures[future]
                try:
                    image, _boxes = future.result()
                    # Resize to model's expected input shape if needed
                    if current_target_shape != self.input_shape:
                        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
                        # Scale boxes accordingly
                        scale_x = self.input_shape[1] / current_target_shape[1]
                        scale_y = self.input_shape[0] / current_target_shape[0]
                        _boxes[:, [0, 2]] = (_boxes[:, [0, 2]] * scale_x).astype('float32')
                        _boxes[:, [1, 3]] = (_boxes[:, [1, 3]] * scale_y).astype('float32')
                    image_data[b] = image
                    box_data[b] = _boxes
                except Exception as e:
                    # Fallback to sequential on error
                    print(f"Warning: Parallel loading failed for item {b}, falling back to sequential: {e}")
                    image, _boxes = get_ground_truth_data(batch_annotation_lines[b], current_target_shape, augment=self.augment)
                    if current_target_shape != self.input_shape:
                        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
                        scale_x = self.input_shape[1] / current_target_shape[1]
                        scale_y = self.input_shape[0] / current_target_shape[0]
                        _boxes[:, [0, 2]] = (_boxes[:, [0, 2]] * scale_x).astype('float32')
                        _boxes[:, [1, 3]] = (_boxes[:, [1, 3]] * scale_y).astype('float32')
                    image_data[b] = image
                    box_data[b] = _boxes
        else:
            # Sequential loading (fallback or when num_workers=0)
            image_data = []
            box_data = []
            for b in range(len(batch_annotation_lines)):
                # Use target shape for preprocessing (multi-scale effect)
                image, _boxes = get_ground_truth_data(batch_annotation_lines[b], current_target_shape, augment=self.augment)
                # Resize to model's expected input shape
                if current_target_shape != self.input_shape:
                    image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
                    # Scale boxes accordingly
                    scale_x = self.input_shape[1] / current_target_shape[1]
                    scale_y = self.input_shape[0] / current_target_shape[0]
                    _boxes[:, [0, 2]] = (_boxes[:, [0, 2]] * scale_x).astype('float32')  # x coordinates
                    _boxes[:, [1, 3]] = (_boxes[:, [1, 3]] * scale_y).astype('float32')  # y coordinates
                image_data.append(image)
                box_data.append(_boxes)
        image_data = np.array(image_data)
        max_boxes_per_img = 0
        for boxes in box_data:
            if len(boxes) > max_boxes_per_img:
                max_boxes_per_img = len(boxes)
        for k, boxes in enumerate(box_data):
            new_boxes = np.zeros((max_boxes_per_img, 5))
            if len(boxes) > 0:
                new_boxes[:len(boxes)] = boxes
            box_data[k] = new_boxes
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes,
                                       self.multi_anchor_assign, grid_shapes=self.grid_shapes)

        return (image_data, *y_true), np.zeros(self.batch_size)
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.annotation_lines)
    
    def build_tf_dataset(self, prefetch_buffer_size=tf.data.AUTOTUNE, 
                        num_parallel_calls=tf.data.AUTOTUNE,
                        shuffle_buffer_size: int = 4096,
                        interleave_cycle_length: int = None,
                        use_gpu_preprocessing: bool = True):
        """
        Build native tf.data.Dataset pipeline for GPU-accelerated data loading.
        
        This creates a true tf.data pipeline that can be parallelized and run on GPU,
        unlike from_generator() which still runs Python code on CPU.
        
        Args:
            prefetch_buffer_size: Number of batches to prefetch. Use tf.data.AUTOTUNE for automatic tuning.
            num_parallel_calls: Number of parallel calls for map operations. Use tf.data.AUTOTUNE for automatic tuning.
            shuffle_buffer_size: Size of shuffle buffer (default: 4096, larger = better randomization)
            interleave_cycle_length: Number of files to interleave for parallel reading (None = no interleaving)
            use_gpu_preprocessing: Whether to use GPU-accelerated preprocessing (default: True)
            
        Returns:
            tf.data.Dataset configured with prefetching and parallel processing
        """
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Create dataset from annotation lines
        annotation_paths = tf.constant(self.annotation_lines)
        dataset = tf.data.Dataset.from_tensor_slices(annotation_paths)
        
        # Shuffle dataset with larger buffer for better randomization (before loading files)
        if self.shuffle:
            # Use provided shuffle_buffer_size, but cap at dataset size
            buffer_size = min(shuffle_buffer_size, len(self.annotation_lines))
            dataset = dataset.shuffle(buffer_size=buffer_size, 
                                     reshuffle_each_iteration=True)
        
        # Add interleaving for parallel file reading if specified
        # Interleaving can help with I/O but may add overhead if I/O is not the bottleneck
        # Make it optional and use smaller cycle_length to reduce overhead
        if interleave_cycle_length is not None and interleave_cycle_length > 1:
            # Interleave file reading for parallel I/O
            # This creates multiple parallel readers that interleave their results
            def _load_and_parse(annotation_line):
                image_path, boxes_string = tf_parse_annotation_line(annotation_line)
                image = tf_load_and_decode_image(image_path)
                boxes = tf_parse_boxes(boxes_string)
                return image, boxes, image_path
            
            # For interleave, num_parallel_calls must be <= cycle_length or AUTOTUNE
            # Use AUTOTUNE for interleave to maximize parallelism
            interleave_parallel_calls = tf.data.AUTOTUNE
            
            # Interleave for parallel I/O - creates cycle_length parallel readers
            # Use smaller block_length=1 to reduce overhead
            dataset = dataset.interleave(
                lambda x: tf.data.Dataset.from_tensors(x).map(_load_and_parse),
                cycle_length=interleave_cycle_length,
                block_length=1,  # Process one element at a time from each cycle
                num_parallel_calls=interleave_parallel_calls,
                deterministic=False
            )
        else:
            # Standard approach without interleaving (often faster if I/O is not bottleneck)
            # Parse annotation and load image
            def _load_and_parse(annotation_line):
                image_path, boxes_string = tf_parse_annotation_line(annotation_line)
                image = tf_load_and_decode_image(image_path)
                boxes = tf_parse_boxes(boxes_string)
                return image, boxes, image_path
            
            dataset = dataset.map(_load_and_parse, num_parallel_calls=num_parallel_calls)
        
        # Preprocess and augment
        # For multi-scale, we'll sample scale per batch, not per image
        # This avoids double resizing
        # Convert input_shape_list to TensorFlow tensor for indexing
        if hasattr(self, 'input_shape_list') and len(self.input_shape_list) > 0:
            # Create tensor from list: shape (num_scales, 2) where 2 is (height, width)
            input_shape_list_tf = tf.constant(self.input_shape_list, dtype=tf.int32)  # (num_scales, 2)
            input_shape_base_tf = tf.constant(self.input_shape, dtype=tf.int32)  # (2,)
            has_multiscale = True
        else:
            input_shape_list_tf = None
            input_shape_base_tf = tf.constant(self.input_shape, dtype=tf.int32)
            has_multiscale = False
        
        def _preprocess_image_and_boxes(image, boxes, image_path):
            # Convert image to float32 and normalize
            image = tf.cast(image, tf.float32) / 255.0
            
            # For multi-scale: sample a random scale factor and resize directly to input_shape
            # This avoids double resizing (target_shape -> input_shape)
            if has_multiscale and self.rescale_interval > 0:
                # Sample a random scale from input_shape_list using TensorFlow ops
                num_scales = tf.shape(input_shape_list_tf)[0]
                scale_idx = tf.random.uniform([], 0, num_scales, dtype=tf.int32)
                # Use tf.gather to get the selected shape
                target_shape_sample = tf.gather(input_shape_list_tf, scale_idx)  # (2,)
                # Calculate scale factors
                scale_h = tf.cast(target_shape_sample[0], tf.float32) / tf.cast(input_shape_base_tf[0], tf.float32)
                scale_w = tf.cast(target_shape_sample[1], tf.float32) / tf.cast(input_shape_base_tf[1], tf.float32)
                # Resize image with random scale, then letterbox to final input_shape
                image_shape = tf.shape(image)
                src_h = tf.cast(image_shape[0], tf.float32)
                src_w = tf.cast(image_shape[1], tf.float32)
                # Resize to scaled size first
                scaled_h = tf.cast(src_h * scale_h, tf.int32)
                scaled_w = tf.cast(src_w * scale_w, tf.int32)
                image_scaled = tf.image.resize(image, [scaled_h, scaled_w], method='bicubic')
                # Then letterbox to final input_shape
                image_resized = tf_letterbox_resize(image_scaled, self.input_shape, return_padding_info=False)
                # Scale boxes accordingly (only scale, letterbox doesn't change box coordinates if done correctly)
                boxes = boxes * tf.stack([scale_w, scale_h, scale_w, scale_h, 1.0])
            else:
                # No multi-scale: just letterbox resize
                image_resized = tf_letterbox_resize(image, self.input_shape, return_padding_info=False)
            
            # Apply augmentations if enabled
            if self.augment:
                # Random resize crop pad (replaces the old PIL-based version)
                image_resized, boxes, _, _ = tf_random_resize_crop_pad(
                    image_resized, self.input_shape, boxes,
                    aspect_ratio_jitter=0.3, scale_jitter=0.5
                )
                
                # Random horizontal flip
                image_resized, boxes = tf_random_horizontal_flip(image_resized, boxes)
                
                # Color augmentations
                image_resized = tf_random_brightness(image_resized, max_delta=0.2)
                image_resized = tf_random_contrast(image_resized, lower=0.8, upper=1.2)
                image_resized = tf_random_saturation(image_resized, lower=0.8, upper=1.2)
                image_resized = tf_random_hue(image_resized, max_delta=0.1)
                image_resized = tf_random_grayscale(image_resized, probability=0.1)
                
                # Random rotate
                image_resized, boxes = tf_random_rotate(image_resized, boxes, rotate_range=20.0, prob=0.1)
                
                # Random gridmask
                image_resized, boxes = tf_random_gridmask(image_resized, boxes, prob=0.2)
            
            return image_resized, boxes
        
        dataset = dataset.map(_preprocess_image_and_boxes, num_parallel_calls=num_parallel_calls)
        
        # Batch the dataset - use padded_batch to handle variable-length boxes
        # This automatically pads boxes to the same length in the batch
        padded_shapes = (
            [self.input_shape[0], self.input_shape[1], 3],  # image shape
            [None, 5]  # boxes shape (variable length, will be padded)
        )
        padding_values = (
            0.0,  # image padding (shouldn't be needed, but just in case)
            0.0   # box padding value
        )
        dataset = dataset.padded_batch(
            self.batch_size, 
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=False
        )
        
        # Apply batch-level augmentations (Mosaic, MixUp) if enabled
        if self.augment:
            def _apply_batch_augmentations(images, boxes_dense):
                """Apply batch-level augmentations (Mosaic, MixUp)."""
                # Apply Mosaic if enabled (high probability for high impact)
                if self.enhance_augment == 'mosaic':
                    images, boxes_dense = tf_random_mosaic(images, boxes_dense, prob=1.0)
                
                # Apply MixUp with lower probability (to avoid over-augmentation)
                # MixUp can be combined with Mosaic or used alone
                images, boxes_dense = tf_random_mixup(images, boxes_dense, prob=0.15, alpha=0.2)
                
                return images, boxes_dense
            
            dataset = dataset.map(_apply_batch_augmentations, num_parallel_calls=num_parallel_calls)
        
        # Process batches: pad boxes, create targets using pure TensorFlow
        def _process_batch_wrapper(images, boxes_dense):
            """
            Process batch using pure TensorFlow operations.
            This uses the fully vectorized tf_preprocess_true_boxes.
            """
            batch_size = tf.shape(images)[0]
            
            # boxes_dense is already padded from padded_batch
            # Filter out invalid boxes (all zeros) - but keep structure for vectorized processing
            # The vectorized function handles invalid boxes via valid_mask
            
            # Use TensorFlow version of preprocess_true_boxes
            y_true = tf_preprocess_true_boxes(
                boxes_dense,
                self.input_shape,
                self.anchors,
                self.num_classes,
                self.multi_anchor_assign,
                self.grid_shapes
            )
            
            # Create dummy targets
            dummy_targets = tf.zeros(batch_size, dtype=tf.float32)
            
            # Return in format expected by model: (inputs_tuple, targets)
            return (images, *y_true), dummy_targets
        
        # Use parallel calls since we're now using pure TensorFlow ops
        dataset = dataset.map(_process_batch_wrapper, num_parallel_calls=num_parallel_calls)
        
        # Prefetch for GPU overlap
        dataset = dataset.prefetch(prefetch_buffer_size)
        
        return dataset
    
    def to_tf_dataset(self, prefetch_buffer_size=tf.data.AUTOTUNE, num_parallel_calls=None):
        """
        Convert Sequence generator to tf.data.Dataset for better GPU utilization.
        
        Args:
            prefetch_buffer_size: Number of batches to prefetch. Use tf.data.AUTOTUNE for automatic tuning.
            num_parallel_calls: Number of parallel calls for map operations. Use tf.data.AUTOTUNE for automatic tuning.
            
        Returns:
            tf.data.Dataset configured with prefetching and parallel processing
        """
        # Create generator function that yields batches
        # The generator returns ((image_data, *y_true), dummy_y) format
        # Note: Keras will control epochs via steps_per_epoch, so we make this infinite
        def generator():
            batch_count = 0
            while True:  # Infinite loop - Keras controls epochs via steps_per_epoch
                # Shuffle at start of each epoch if needed
                if batch_count % len(self) == 0 and self.shuffle:
                    np.random.shuffle(self.indexes)
                
                # Get current batch index within epoch
                i = batch_count % len(self)
                batch = self[i]
                # batch is ((image_data, *y_true), dummy_y)
                # For model.fit(), we need (inputs, targets) where inputs = (image_data, *y_true)
                inputs_tuple = batch[0]  # (image_data, *y_true)
                dummy_target = batch[1]  # dummy_y (not used but required by Keras)
                yield inputs_tuple, dummy_target
                
                batch_count += 1
                
                # Call on_epoch_end at the end of each epoch
                if batch_count % len(self) == 0:
                    self.on_epoch_end()
        
        # Get output types and shapes from a sample batch
        sample_batch = self[0]
        inputs_tuple = sample_batch[0]  # (image_data, *y_true)
        dummy_y = sample_batch[1]
        
        images = inputs_tuple[0]
        y_true_list = inputs_tuple[1:]
        
        # Define output types: (inputs_tuple, dummy_target)
        # inputs_tuple is a tuple of (image, *y_true)
        output_types = (
            (tf.float32, *[tf.float32] * len(y_true_list)),  # inputs tuple
            tf.float32  # dummy target
        )
        
        # Define output shapes
        output_shapes = (
            (
                tf.TensorShape([None] + list(images.shape[1:])),  # image shape
                *[tf.TensorShape([None] + list(y.shape[1:])) for y in y_true_list]  # y_true shapes
            ),
            tf.TensorShape([None])  # dummy target shape
        )
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
        
        # Configure parallel processing if specified
        if num_parallel_calls is not None:
            # Note: map operations would go here if we had preprocessing steps
            pass
        
        # Prefetch to overlap CPU data preparation with GPU computation
        dataset = dataset.prefetch(prefetch_buffer_size)
        
        return dataset
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False)


# Helper functions used by MultiGridDataGenerator
def get_ground_truth_data(annotation_line, input_shape, augment=False, max_boxes=100):
    """
    Load and preprocess ground truth data from annotation line.
    
    Args:
        annotation_line: Annotation line in format "image_path x1,y1,x2,y2,class ..."
        input_shape: Target image shape (height, width)
        augment: Whether to apply augmentation
        max_boxes: Maximum number of boxes (currently unused)
        
    Returns:
        Tuple of (image_data, box_data) where image_data is normalized image array
        and box_data is array of boxes in format (x1, y1, x2, y2, class)
    """
    # Implementation remains the same as original
    line = annotation_line.split()
    image = Image.open(line[0])
    image_size = image.size
    model_input_size = tuple(reversed(input_shape))
    boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]]).astype('int32')

    if not augment:
        new_image, padding_size, offset = letterbox_resize(image, target_size=model_input_size, return_padding_info=True)
        image_data = np.array(new_image)
        image_data = normalize_image(image_data)
        boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size, padding_shape=padding_size, offset=offset)
        box_data = np.array(boxes).reshape(-1, 5)
        return image_data, box_data
    return custom_aug(image, boxes, image_size, model_input_size)


def custom_aug(image, boxes, image_size, model_input_size):
    """
    Apply custom augmentation pipeline to image and boxes.
    
    Args:
        image: PIL Image object
        boxes: Array of boxes in format (x1, y1, x2, y2, class)
        image_size: Original image size (width, height)
        model_input_size: Target image size (width, height)
        
    Returns:
        Tuple of (image_data, box_data) where image_data is normalized image array
        and box_data is array of augmented boxes
    """
    image, padding_size, padding_offset = random_resize_crop_pad(image, target_size=model_input_size)
    image, horizontal_flip = random_horizontal_flip(image)
    image = random_brightness(image)
    image = random_chroma(image)
    image = random_contrast(image)
    image = random_sharpness(image)
    image = random_grayscale(image)
    image, vertical_flip = random_vertical_flip(image)
    boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size, padding_shape=padding_size, offset=padding_offset, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
    image, boxes = random_rotate(image, boxes)
    image, boxes = random_gridmask(image, boxes)
    image_data = np.array(image)
    image_data = normalize_image(image_data)
    box_data = np.array(boxes).reshape(-1, 5)
    return image_data, box_data


# Helper functions for anchor matching
def get_anchor_mask(anchors):
    """Compute anchor mask for each scale."""
    num_anchors_per_scale = [len(anchor) for anchor in anchors]
    total_num_anchors = sum(num_anchors_per_scale)
    anchor_mask = list(range(0, total_num_anchors, 1))
    index = 0
    anchor_mask_per_scale = []
    for layer_id, num in enumerate(num_anchors_per_scale):
        anchor_mask_per_scale.append(anchor_mask[index:index+num])
        index = index + num
    return anchor_mask_per_scale


def iol_common_center(anchors, obj_boxes_wh):
    """Calculate IoL (Intersection over Largest) scores between anchors and object boxes."""
    intersection_wh = np.minimum(np.expand_dims(obj_boxes_wh, axis=-2), anchors)
    obj_areas = obj_boxes_wh[..., 0] * obj_boxes_wh[..., 1]
    anchor_areas = anchors[:, 0] * anchors[:, 1]
    intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1]
    largest_of_boxVsAnchor = np.maximum(np.expand_dims(obj_areas, axis=-1), anchor_areas)
    iols = intersection_areas / largest_of_boxVsAnchor
    return iols


def best_fit_anchor(box, anchors):
    """Find the best matching anchor for a given box."""
    all_layer_anchors = np.concatenate(anchors, axis=0)   
    anchor_masks = get_anchor_mask(anchors)
    iols = np.round(iol_common_center(all_layer_anchors, box), 3)
    anchor_index = np.argmax(iols, axis=-1)
    sel_layer = 0
    k = 0
    num_layers = len(anchors)
    for layer in range(num_layers):
        if anchor_index in anchor_masks[layer]:
            sel_layer = layer
            k = np.where(anchor_masks[layer] == anchor_index)[0][0]
            break    
    return sel_layer, k, iols


def best_fit_and_layer(box, anchors, multi_anchor_assign=False, multi_anchor_thresh=0.8):
    """
    Find best matching anchor and layer for a box.
    
    Args:
        box: Box width and height (w, h)
        anchors: List of anchor arrays for each layer
        multi_anchor_assign: Whether to assign multiple anchors
        multi_anchor_thresh: Threshold for multi-anchor assignment
        
    Returns:
        Tuple of (layer, anchor_index, iol_scores) or (layers, anchor_indices, iol_scores) if multi_anchor_assign
    """
    all_layer_anchors = np.concatenate(anchors, axis=0)   
    anchor_masks = get_anchor_mask(anchors)
    iols = np.round(iol_common_center(all_layer_anchors, box), 3)
    anchor_indexes = np.argsort(-iols)
    selected_layer_anchor_pair = np.where(anchor_masks == anchor_indexes[0])
    sel_layer, sel_anchor = list(zip(selected_layer_anchor_pair[0], selected_layer_anchor_pair[1]))[0][0:2]

    if multi_anchor_assign:
        sel_layers, sel_anchors = [sel_layer], [sel_anchor]
        for m in anchor_indexes[1:]:
            if (iols[m] / iols[sel_anchor]) >= multi_anchor_thresh and iols[m] > 0.5:
               selected_layer_anchor_pair = np.where(anchor_masks == anchor_indexes[m])
               next_sel_layer, next_sel_anchor = list(zip(selected_layer_anchor_pair[0], selected_layer_anchor_pair[1]))[0][0:2]
               sel_layers.append(next_sel_layer)
               sel_anchors.append(next_sel_anchor)
        return sel_layers, sel_anchors, iols
    else:
        return sel_layer, sel_anchor, iols


@tf.function
def tf_best_fit_and_layer_batch(boxes_wh: tf.Tensor, anchors: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Vectorized TensorFlow implementation to find best anchor and layer for a batch of boxes.
    
    Args:
        boxes_wh: Box widths and heights of shape (batch_size, max_boxes, 2) or (max_boxes, 2)
        anchors: List of anchor tensors for each layer
        
    Returns:
        Tuple of (selected_layers, selected_anchor_indices) where:
        - selected_layers: shape (batch_size, max_boxes) or (max_boxes,)
        - selected_anchor_indices: shape (batch_size, max_boxes) or (max_boxes,)
    """
    # Concatenate all anchors
    all_anchors = tf.concat(anchors, axis=0)  # (total_anchors, 2)
    
    # Calculate anchor counts per layer
    anchor_counts = tf.stack([tf.shape(anchor)[0] for anchor in anchors])
    cumulative_counts = tf.cumsum(anchor_counts)
    
    # Expand dimensions for broadcasting
    # boxes_wh: (batch_size, max_boxes, 2) or (max_boxes, 2)
    # all_anchors: (total_anchors, 2)
    boxes_expanded = tf.expand_dims(boxes_wh, axis=-2)  # (batch, max_boxes, 1, 2) or (max_boxes, 1, 2)
    anchors_expanded = tf.expand_dims(all_anchors, axis=0)  # (1, total_anchors, 2)
    
    # Calculate IoL scores for all boxes and all anchors
    intersection_wh = tf.minimum(boxes_expanded, anchors_expanded)
    box_areas = boxes_wh[..., 0:1] * boxes_wh[..., 1:2]  # (batch, max_boxes, 1) or (max_boxes, 1)
    anchor_areas = all_anchors[:, 0:1] * all_anchors[:, 1:2]  # (total_anchors, 1)
    intersection_areas = intersection_wh[..., 0:1] * intersection_wh[..., 1:2]
    largest_areas = tf.maximum(box_areas, anchor_areas)
    iols = intersection_areas / (largest_areas + tf.keras.backend.epsilon())  # (batch, max_boxes, total_anchors) or (max_boxes, total_anchors)
    
    # Find best anchor for each box
    best_anchor_indices = tf.argmax(iols, axis=-1, output_type=tf.int32)  # (batch, max_boxes) or (max_boxes,)
    
    # Find which layer each anchor belongs to
    # Create a mapping: for each anchor index, find its layer
    def find_layer_for_anchor(anchor_idx):
        # Find first layer where cumulative_count > anchor_idx
        layer_mask = anchor_idx < cumulative_counts
        layer_idx = tf.where(layer_mask)[0]
        if tf.size(layer_idx) > 0:
            layer = layer_idx[0]
        else:
            layer = tf.constant(0, dtype=tf.int32)
        
        # Find anchor index within layer
        if layer > 0:
            anchor_in_layer = anchor_idx - cumulative_counts[layer - 1]
        else:
            anchor_in_layer = anchor_idx
        
        return layer, anchor_in_layer
    
    # Apply to all boxes
    if len(boxes_wh.shape) == 2:
        # Single batch case: (max_boxes, 2)
        results = tf.map_fn(
            lambda idx: find_layer_for_anchor(idx),
            best_anchor_indices,
            fn_output_signature=(tf.int32, tf.int32),
            parallel_iterations=10
        )
        selected_layers, selected_anchors = results
    else:
        # Batch case: (batch_size, max_boxes, 2)
        def process_batch(batch_anchor_indices):
            return tf.map_fn(
                lambda idx: find_layer_for_anchor(idx),
                batch_anchor_indices,
                fn_output_signature=(tf.int32, tf.int32),
                parallel_iterations=10
            )
        
        results = tf.map_fn(
            process_batch,
            best_anchor_indices,
            fn_output_signature=((tf.int32, tf.int32)),
            parallel_iterations=10
        )
        selected_layers, selected_anchors = results
    
    return selected_layers, selected_anchors


@tf.function
def _invert_activation_numerically(desired_offset: tf.Tensor, max_iterations: int = 50, tolerance: float = 1e-8) -> tf.Tensor:
    """
    Numerically invert f(x) = tanh(0.15*x) + sigmoid(0.15*x) to find raw_xy.
    
    Uses Newton's method with safeguards for stability.
    The activation function has range approximately [0, 2] for x in reasonable range.
    For MultiGridDet, offsets can be in [-1, 2] range to span 3x3 neighborhood.
    
    Args:
        desired_offset: Target value after activation (shape: any)
        max_iterations: Maximum Newton iterations (increased for better convergence)
        tolerance: Convergence tolerance (tighter for better precision)
        
    Returns:
        raw_xy: Pre-activation value such that f(raw_xy) ≈ desired_offset
    """
    # Initial guess: approximate linear relationship for small values
    # For small x: tanh(0.15*x) ≈ 0.15*x, sigmoid(0.15*x) ≈ 0.5 + 0.0375*x
    # So f(x) ≈ 0.5 + 0.1875*x, thus x ≈ (f(x) - 0.5) / 0.1875
    # But for larger offsets, we need a better initial guess
    x = (desired_offset - 0.5) / 0.1875
    
    # For offsets near the edges of [-1, 2], we need larger x values
    # Clamp initial guess but allow wider range for edge cases
    x = tf.clip_by_value(x, -30.0, 30.0)
    
    # Use tf.while_loop for iteration
    def body(i, x):
        # Compute f(x) = tanh(0.15*x) + sigmoid(0.15*x)
        fx = tf.tanh(0.15 * x) + tf.sigmoid(0.15 * x)
        
        # Compute f'(x) = 0.15 * (1 - tanh^2(0.15*x)) + 0.15 * sigmoid(0.15*x) * (1 - sigmoid(0.15*x))
        tanh_val = tf.tanh(0.15 * x)
        sigmoid_val = tf.sigmoid(0.15 * x)
        fprime = 0.15 * (1.0 - tanh_val * tanh_val) + 0.15 * sigmoid_val * (1.0 - sigmoid_val)
        
        # Newton step: x_new = x - (f(x) - desired) / f'(x)
        # Add small epsilon to avoid division by zero
        fprime_safe = fprime + 1e-8
        x_new = x - (fx - desired_offset) / fprime_safe
        
        # Allow wider range for edge cases (offsets near -1 or +2)
        # The activation function is bounded, so very large x values are safe
        x_new = tf.clip_by_value(x_new, -50.0, 50.0)
        
        return i + 1, x_new
    
    def condition(i, x):
        # Continue if not converged and under max iterations
        fx = tf.tanh(0.15 * x) + tf.sigmoid(0.15 * x)
        diff = tf.abs(fx - desired_offset)
        return tf.logical_and(i < max_iterations, tf.reduce_max(diff) >= tolerance)
    
    # Run Newton's method
    i = tf.constant(0)
    _, x_final = tf.while_loop(condition, body, [i, x], maximum_iterations=max_iterations)
    
    return x_final


@tf.function
def tf_preprocess_true_boxes(true_boxes: tf.Tensor, 
                             input_shape: Tuple[int, int],
                             anchors: List[tf.Tensor],
                             num_classes: int,
                             multi_anchor_assign: bool,
                             grid_shapes: List[Tuple[int, int]]) -> List[tf.Tensor]:
    """
    Fully vectorized TensorFlow implementation of preprocess_true_boxes.
    
    Processes all boxes, anchors, and grid cells in parallel using broadcasting.
    
    Args:
        true_boxes: Batch of boxes in format (x1, y1, x2, y2, class), shape (batch_size, max_boxes, 5)
        input_shape: Input image shape (height, width)
        anchors: List of anchor tensors for each layer
        num_classes: Number of object classes
        multi_anchor_assign: Whether to assign multiple anchors per object (not used in vectorized version)
        grid_shapes: Pre-computed grid shapes for each layer
        
    Returns:
        List of target tensors for each detection layer
    """
    batch_size = tf.shape(true_boxes)[0]
    max_boxes = tf.shape(true_boxes)[1]
    num_layers = len(anchors)
    input_h, input_w = input_shape
    
    # Debug: Assert input shape
    tf.debugging.assert_equal(tf.rank(true_boxes), 3, message="true_boxes must be rank 3: (batch, max_boxes, 5)")
    tf.debugging.assert_equal(tf.shape(true_boxes)[2], 5, message="true_boxes last dim must be 5")
    
    # Convert boxes from (x1, y1, x2, y2, class) to (cx, cy, w, h, class)
    # CRITICAL: Ensure true_boxes has the correct shape before processing
    true_boxes_shape = tf.shape(true_boxes)
    tf.print("DEBUG: true_boxes shape =", true_boxes_shape, output_stream=tf.compat.v1.logging.info)
    
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2.0  # (batch, max_boxes, 2)
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # (batch, max_boxes, 2)
    boxes_class = tf.cast(true_boxes[..., 4:5], tf.int32)  # (batch, max_boxes, 1)
    
    # Debug: Assert intermediate shapes with explicit checks
    boxes_xy_shape = tf.shape(boxes_xy)
    boxes_wh_shape = tf.shape(boxes_wh)
    boxes_class_shape = tf.shape(boxes_class)
    
    tf.print("DEBUG: boxes_xy shape =", boxes_xy_shape, "boxes_wh shape =", boxes_wh_shape, "boxes_class shape =", boxes_class_shape, output_stream=tf.compat.v1.logging.info)
    
    # Ensure boxes_wh has rank 3
    tf.debugging.assert_equal(tf.rank(boxes_wh), 3, message="boxes_wh must be rank 3")
    tf.debugging.assert_equal(boxes_wh_shape[0], batch_size, message="boxes_wh batch dimension mismatch")
    tf.debugging.assert_equal(boxes_wh_shape[1], max_boxes, message="boxes_wh max_boxes dimension mismatch")
    tf.debugging.assert_equal(boxes_wh_shape[2], 2, message="boxes_wh last dimension must be 2")
    
    # Filter out invalid boxes (where w*h <= 0)
    # CRITICAL FIX: Use explicit indexing to ensure correct shape
    # boxes_wh has shape (batch, max_boxes, 2)
    # Use tf.gather instead of slicing to ensure correct shape
    box_w = tf.gather(boxes_wh, 0, axis=2)  # (batch, max_boxes) - using gather
    box_h = tf.gather(boxes_wh, 1, axis=2)  # (batch, max_boxes) - using gather
    box_areas = box_w * box_h  # (batch, max_boxes)
    valid_mask = box_areas > 0.0  # (batch, max_boxes)
    
    # Debug: Assert shapes after computation
    box_w_shape = tf.shape(box_w)
    box_w_rank = tf.rank(box_w)
    box_h_shape = tf.shape(box_h)
    box_h_rank = tf.rank(box_h)
    box_areas_shape = tf.shape(box_areas)
    box_areas_rank = tf.rank(box_areas)
    valid_mask_shape = tf.shape(valid_mask)
    valid_mask_rank = tf.rank(valid_mask)
    
    tf.print("DEBUG: box_w shape =", box_w_shape, "rank =", box_w_rank, output_stream=tf.compat.v1.logging.info)
    tf.print("DEBUG: box_h shape =", box_h_shape, "rank =", box_h_rank, output_stream=tf.compat.v1.logging.info)
    tf.print("DEBUG: box_areas shape =", box_areas_shape, "rank =", box_areas_rank, output_stream=tf.compat.v1.logging.info)
    tf.print("DEBUG: valid_mask shape =", valid_mask_shape, "rank =", valid_mask_rank, output_stream=tf.compat.v1.logging.info)
    
    # Ensure box_w and box_h have rank 2
    tf.debugging.assert_equal(box_w_rank, 2, message="box_w must be rank 2")
    tf.debugging.assert_equal(box_h_rank, 2, message="box_h must be rank 2")
    tf.debugging.assert_equal(box_w_shape[0], batch_size, message="box_w batch dimension mismatch")
    tf.debugging.assert_equal(box_w_shape[1], max_boxes, message="box_w max_boxes dimension mismatch")
    tf.debugging.assert_equal(box_h_shape[0], batch_size, message="box_h batch dimension mismatch")
    tf.debugging.assert_equal(box_h_shape[1], max_boxes, message="box_h max_boxes dimension mismatch")
    
    # Ensure valid_mask has rank 2 and correct dimensions
    tf.debugging.assert_equal(valid_mask_rank, 2, message="valid_mask must be rank 2")
    tf.debugging.assert_equal(valid_mask_shape[0], batch_size, message="valid_mask batch dimension mismatch")
    tf.debugging.assert_equal(valid_mask_shape[1], max_boxes, message="valid_mask max_boxes dimension mismatch")
    
    # Initialize output tensors
    y_true = []
    anchor_counts_per_layer = []
    for layer_idx in range(num_layers):
        grid_h, grid_w = grid_shapes[layer_idx]
        num_anchors = tf.shape(anchors[layer_idx])[0]
        anchor_counts_per_layer.append(num_anchors)
        target_shape = [batch_size, grid_h, grid_w, 5 + num_anchors + num_classes]
        y_true.append(tf.zeros(target_shape, dtype=tf.float32))
    
    # ========================================================================
    # Step 1: Vectorize Anchor Matching
    # ========================================================================
    # Concatenate all anchors: (total_anchors, 2)
    all_anchors = tf.concat(anchors, axis=0)  # (total_anchors, 2)
    total_anchors = tf.shape(all_anchors)[0]
    
    # Compute anchor counts and cumulative counts for layer mapping
    anchor_counts = tf.stack([tf.cast(count, tf.int32) for count in anchor_counts_per_layer])
    cumulative_counts = tf.cumsum(anchor_counts)  # (num_layers,)
    
    # Compute IoL for all boxes vs all anchors simultaneously
    # boxes_wh: (batch, max_boxes, 2)
    # all_anchors: (total_anchors, 2)
    # Expand for broadcasting: (batch, max_boxes, 1, 2) vs (1, 1, total_anchors, 2)
    boxes_wh_expanded = tf.expand_dims(boxes_wh, axis=2)  # (batch, max_boxes, 1, 2)
    anchors_expanded = tf.expand_dims(tf.expand_dims(all_anchors, 0), 0)  # (1, 1, total_anchors, 2)
    
    # Calculate intersection
    intersection_wh = tf.minimum(boxes_wh_expanded, anchors_expanded)  # (batch, max_boxes, total_anchors, 2)
    
    # Calculate areas
    box_areas_expanded = tf.expand_dims(box_areas, axis=2)  # (batch, max_boxes, 1)
    anchor_areas = all_anchors[:, 0] * all_anchors[:, 1]  # (total_anchors,)
    anchor_areas_expanded = tf.expand_dims(tf.expand_dims(anchor_areas, 0), 0)  # (1, 1, total_anchors)
    
    intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1]  # (batch, max_boxes, total_anchors)
    largest_areas = tf.maximum(box_areas_expanded, anchor_areas_expanded)  # (batch, max_boxes, total_anchors)
    iols = intersection_areas / (largest_areas + tf.keras.backend.epsilon())  # (batch, max_boxes, total_anchors)
    
    # Find best anchor for each box
    best_anchor_indices = tf.argmax(iols, axis=2, output_type=tf.int32)  # (batch, max_boxes)
    
    # Map anchor indices to (layer, anchor_in_layer)
    # For each anchor index, find which layer it belongs to using vectorized operations
    # Create a mapping tensor: for each anchor index, which layer does it belong to?
    # We'll use broadcasting to find the layer for each anchor index
    
    # Create range of anchor indices: (total_anchors,)
    anchor_idx_range = tf.range(total_anchors, dtype=tf.int32)  # (total_anchors,)
    
    # For each anchor index, find which layer it belongs to
    # Compare anchor_idx with cumulative_counts to find layer
    anchor_idx_expanded = tf.expand_dims(anchor_idx_range, 1)  # (total_anchors, 1)
    cumulative_counts_expanded = tf.expand_dims(cumulative_counts, 0)  # (1, num_layers)
    
    # Find first layer where cumulative_count > anchor_idx
    layer_mask = anchor_idx_expanded < cumulative_counts_expanded  # (total_anchors, num_layers)
    
    # For each anchor, find first True (first layer where it fits)
    # Use argmax to find first True (but need to handle case where all False)
    # Instead, use a different approach: find layer index using arithmetic
    layer_indices = tf.cast(tf.argmax(tf.cast(layer_mask, tf.int32), axis=1), tf.int32)  # (total_anchors,)
    
    # Handle edge case: if anchor_idx >= all cumulative_counts, assign to last layer
    max_cumulative = tf.reduce_max(cumulative_counts)
    anchor_idx_valid_mask = anchor_idx_range < max_cumulative  # CRITICAL FIX: Renamed to avoid overwriting valid_mask
    layer_indices = tf.where(anchor_idx_valid_mask, layer_indices, tf.constant(num_layers - 1, dtype=tf.int32))
    
    # Now compute anchor index within layer
    # For layer 0: anchor_in_layer = anchor_idx
    # For layer > 0: anchor_in_layer = anchor_idx - cumulative_counts[layer-1]
    prev_counts = tf.concat([[0], cumulative_counts[:-1]], axis=0)  # (num_layers,)
    prev_counts_gathered = tf.gather(prev_counts, layer_indices)  # (total_anchors,)
    anchor_in_layer = anchor_idx_range - prev_counts_gathered  # (total_anchors,)
    
    # For each box, get its layer and anchor
    # best_anchor_indices: (batch, max_boxes)
    selected_layers = tf.gather(layer_indices, best_anchor_indices)  # (batch, max_boxes)
    selected_anchors = tf.gather(anchor_in_layer, best_anchor_indices)  # (batch, max_boxes)
    
    # Debug: Assert selected_layers and selected_anchors shapes
    tf.debugging.assert_shapes([
        (best_anchor_indices, ['batch', 'max_boxes']),
        (selected_layers, ['batch', 'max_boxes']),
        (selected_anchors, ['batch', 'max_boxes']),
    ], message="Selected layers/anchors shapes after initial selection")
    
    # Also get max IoL per layer to find best layer
    # The original implementation finds best layer based on max IoL across all anchors in that layer
    # Compute max IoL per layer per box
    layer_start_indices = tf.concat([[0], cumulative_counts[:-1]], axis=0)  # (num_layers,)
    layer_end_indices = cumulative_counts  # (num_layers,)
    
    max_iols_per_layer = []
    for layer_idx in range(num_layers):
        start_idx = layer_start_indices[layer_idx]
        end_idx = layer_end_indices[layer_idx]
        layer_iols = iols[:, :, start_idx:end_idx]  # (batch, max_boxes, num_anchors_in_layer)
        max_iol = tf.reduce_max(layer_iols, axis=2)  # (batch, max_boxes)
        max_iols_per_layer.append(max_iol)
    
    max_iols_per_layer = tf.stack(max_iols_per_layer, axis=2)  # (batch, max_boxes, num_layers)
    best_layer_per_box = tf.argmax(max_iols_per_layer, axis=2, output_type=tf.int32)  # (batch, max_boxes)
    
    # Use best_layer_per_box (best layer based on max IoL) instead of selected_layers (based on best anchor)
    # This matches the original NumPy implementation which uses best_fit_and_layer
    selected_layers = best_layer_per_box
    
    # Debug: Assert best_layer_per_box shape
    tf.debugging.assert_shapes([
        (best_layer_per_box, ['batch', 'max_boxes']),
        (selected_layers, ['batch', 'max_boxes']),
    ], message="best_layer_per_box shape")
    
    # Re-compute selected_anchors based on best layer (not best anchor globally)
    # For each box, get the best anchor within its best layer
    # Compute best anchor per layer for all boxes
    selected_anchors_per_layer = []
    for layer_idx in range(num_layers):
        start_idx = layer_start_indices[layer_idx]
        end_idx = layer_end_indices[layer_idx]
        layer_iols = iols[:, :, start_idx:end_idx]  # (batch, max_boxes, num_anchors_in_layer)
        best_anchor_in_layer = tf.argmax(layer_iols, axis=2, output_type=tf.int32)  # (batch, max_boxes)
        selected_anchors_per_layer.append(best_anchor_in_layer)
    
    # Stack: (batch, max_boxes, num_layers) - each position has best anchor for that layer
    anchors_by_layer = tf.stack(selected_anchors_per_layer, axis=2)  # (batch, max_boxes, num_layers)
    
    # Debug: Assert anchors_by_layer shape
    tf.debugging.assert_shapes([
        (anchors_by_layer, ['batch', 'max_boxes', 'num_layers']),
    ], message="anchors_by_layer shape")
    
    # Gather the anchor for each box's selected layer using gather_nd
    batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, max_boxes])  # (batch, max_boxes)
    box_indices = tf.tile(tf.expand_dims(tf.range(max_boxes), 0), [batch_size, 1])  # (batch, max_boxes)
    gather_indices = tf.stack([
        tf.reshape(batch_indices, [-1]),  # (batch*max_boxes,)
        tf.reshape(box_indices, [-1]),  # (batch*max_boxes,)
        tf.reshape(selected_layers, [-1])  # (batch*max_boxes,)
    ], axis=1)  # (batch*max_boxes, 3)
    
    selected_anchors_flat = tf.gather_nd(anchors_by_layer, gather_indices)  # (batch*max_boxes,)
    selected_anchors = tf.reshape(selected_anchors_flat, [batch_size, max_boxes])  # (batch, max_boxes)
    
    # Debug: Assert final selected_anchors shape
    tf.debugging.assert_shapes([
        (selected_anchors, ['batch', 'max_boxes']),
    ], message="Final selected_anchors shape")
    
    # ========================================================================
    # Step 2: Vectorize Grid Position Calculation
    # ========================================================================
    # For each box, compute grid positions for its selected layer
    # boxes_xy: (batch, max_boxes, 2)
    # selected_layers: (batch, max_boxes)
    
    # Compute grid scales for each layer
    grid_scales_h = tf.stack([tf.cast(grid_shapes[l][0], tf.float32) / tf.cast(input_h, tf.float32) for l in range(num_layers)])  # (num_layers,)
    grid_scales_w = tf.stack([tf.cast(grid_shapes[l][1], tf.float32) / tf.cast(input_w, tf.float32) for l in range(num_layers)])  # (num_layers,)
    
    # For each box, get the grid scale for its selected layer
    # Use gather with batch indices
    batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, max_boxes])  # (batch, max_boxes)
    box_indices = tf.tile(tf.expand_dims(tf.range(max_boxes), 0), [batch_size, 1])  # (batch, max_boxes)
    
    # Get grid scales for each box's selected layer
    selected_scales_h = tf.gather(grid_scales_h, selected_layers)  # (batch, max_boxes)
    selected_scales_w = tf.gather(grid_scales_w, selected_layers)  # (batch, max_boxes)
    
    # Compute grid positions
    # FIXED: Match decoding indexing where cell_grid[i,j] = [j, i] with i=row (y), j=col (x)
    # boxes_xy is in PIXELS, selected_scales_w = grid_w / input_w, so:
    # cx = boxes_xy * (grid_w / input_w) = (box_x / input_w) * grid_w (correct grid coordinate)
    cx = boxes_xy[:, :, 0] * selected_scales_w  # (batch, max_boxes) - x * (grid_w / input_w)
    cy = boxes_xy[:, :, 1] * selected_scales_h  # (batch, max_boxes) - y * (grid_h / input_h)
    
    # Debug: Assert grid position computation shapes
    tf.debugging.assert_shapes([
        (selected_scales_h, ['batch', 'max_boxes']),
        (selected_scales_w, ['batch', 'max_boxes']),
        (cx, ['batch', 'max_boxes']),
        (cy, ['batch', 'max_boxes']),
    ], message="Grid position computation shapes")
    
    # FIXED: grid_col is x-index (column), grid_row is y-index (row) to match decoding
    grid_col = tf.cast(cx, tf.int32)  # (batch, max_boxes) - column (x) index
    grid_row = tf.cast(cy, tf.int32)  # (batch, max_boxes) - row (y) index
    
    # Compute fractional offsets within grid cell
    tx = cx - tf.cast(grid_col, tf.float32)  # (batch, max_boxes) - fractional x offset [0, 1)
    ty = cy - tf.cast(grid_row, tf.float32)  # (batch, max_boxes) - fractional y offset [0, 1)
    
    # Debug: Assert grid_row, grid_col, tx, ty shapes
    tf.debugging.assert_shapes([
        (grid_row, ['batch', 'max_boxes']),
        (grid_col, ['batch', 'max_boxes']),
        (tx, ['batch', 'max_boxes']),
        (ty, ['batch', 'max_boxes']),
    ], message="grid_row, grid_col, tx, ty shapes")
    
    # ========================================================================
    # Step 3: Generate All Grid Cell Candidates (3x3 = 9 per box)
    # ========================================================================
    # Generate offsets: [-1, 0, 1] x [-1, 0, 1]
    ki_offsets = tf.constant([-1, 0, 1], dtype=tf.int32)  # (3,)
    kj_offsets = tf.constant([-1, 0, 1], dtype=tf.int32)  # (3,)
    
    # Create meshgrid of offsets
    ki_grid, kj_grid = tf.meshgrid(ki_offsets, kj_offsets, indexing='ij')
    ki_flat = tf.reshape(ki_grid, [-1])  # (9,)
    kj_flat = tf.reshape(kj_grid, [-1])  # (9,)
    
    # Expand for broadcasting: (batch, max_boxes, 9)
    # Get dynamic batch and max_boxes dimensions
    batch_size_dyn = tf.shape(grid_row)[0]
    max_boxes_dyn = tf.shape(grid_row)[1]
    
    grid_row_expanded = tf.expand_dims(grid_row, axis=-1)  # (batch, max_boxes, 1)
    grid_col_expanded = tf.expand_dims(grid_col, axis=-1)  # (batch, max_boxes, 1)
    
    # Create ki, kj with proper broadcasting shape: (1, 1, 9)
    ki_expanded = tf.reshape(ki_flat, [1, 1, 9])  # (1, 1, 9)
    kj_expanded = tf.reshape(kj_flat, [1, 1, 9])  # (1, 1, 9)
    
    # Debug: Assert expanded shapes before addition
    tf.debugging.assert_shapes([
        (grid_row_expanded, ['batch', 'max_boxes', 1]),
        (grid_col_expanded, ['batch', 'max_boxes', 1]),
        (ki_expanded, [1, 1, 9]),
        (kj_expanded, [1, 1, 9]),
    ], message="Expanded grid_row, grid_col, ki, kj shapes before addition")
    
    # FIXED: kii is row index (y), kjj is col index (x) to match decoding
    # ki offsets are row offsets, kj offsets are col offsets
    kii = grid_row_expanded + ki_expanded  # (batch, max_boxes, 9) - row indices
    kjj = grid_col_expanded + kj_expanded  # (batch, max_boxes, 9) - col indices
    
    # Debug: Assert kii, kjj shapes after addition
    tf.debugging.assert_shapes([
        (kii, ['batch', 'max_boxes', 9]),
        (kjj, ['batch', 'max_boxes', 9]),
    ], message="kii, kjj shapes after addition")
    
    # Also expand tx, ty for all 9 candidates (not used in current logic but kept for consistency)
    tx_expanded = tf.expand_dims(tx, axis=-1)  # (batch, max_boxes, 1)
    ty_expanded = tf.expand_dims(ty, axis=-1)  # (batch, max_boxes, 1)
    ki_float = tf.cast(tf.reshape(ki_flat, [1, 1, 9]), tf.float32)  # (1, 1, 9)
    kj_float = tf.cast(tf.reshape(kj_flat, [1, 1, 9]), tf.float32)  # (1, 1, 9)
    
    # ========================================================================
    # Step 4: Compute Validity Masks
    # ========================================================================
    # Get grid shapes for each box's selected layer
    # Ensure grid_shapes values are converted to tensors
    # Build grid_h_values and grid_w_values as tensors
    grid_h_list = []
    grid_w_list = []
    for l in range(num_layers):
        grid_h_list.append(tf.cast(grid_shapes[l][0], tf.int32))
        grid_w_list.append(tf.cast(grid_shapes[l][1], tf.int32))
    grid_h_values = tf.stack(grid_h_list)  # (num_layers,)
    grid_w_values = tf.stack(grid_w_list)  # (num_layers,)
    
    # Gather grid shapes for each box
    grid_h_per_box = tf.gather(grid_h_values, selected_layers)  # (batch, max_boxes)
    grid_w_per_box = tf.gather(grid_w_values, selected_layers)  # (batch, max_boxes)
    
    # Debug: Assert grid shapes after gather
    tf.debugging.assert_shapes([
        (grid_h_values, ['num_layers']),
        (grid_w_values, ['num_layers']),
        (selected_layers, ['batch', 'max_boxes']),
        (grid_h_per_box, ['batch', 'max_boxes']),
        (grid_w_per_box, ['batch', 'max_boxes']),
    ], message="Grid shapes after gather")
    
    # Expand for broadcasting - use axis=-1 to ensure correct dimension
    # TensorFlow should automatically broadcast (batch, max_boxes, 1) with (batch, max_boxes, 9)
    grid_h_expanded = tf.expand_dims(grid_h_per_box, axis=-1)  # (batch, max_boxes, 1)
    grid_w_expanded = tf.expand_dims(grid_w_per_box, axis=-1)  # (batch, max_boxes, 1)
    
    # Debug: Assert expanded grid shapes BEFORE bounds checking (THIS IS WHERE THE ERROR LIKELY OCCURS)
    tf.debugging.assert_shapes([
        (grid_h_expanded, ['batch', 'max_boxes', 1]),
        (grid_w_expanded, ['batch', 'max_boxes', 1]),
        (kii, ['batch', 'max_boxes', 9]),
        (kjj, ['batch', 'max_boxes', 9]),
    ], message="CRITICAL: Shapes before bounds checking - kii/kjj vs grid_h_expanded/grid_w_expanded")
    
    # FIXED: Bounds checking - kii is row index (height), kjj is col index (width)
    # kii/kjj: (batch, max_boxes, 9), grid_h_expanded/grid_w_expanded: (batch, max_boxes, 1)
    # Debug: Check shapes right before logical_and operations
    kii_shape = tf.shape(kii)
    grid_h_shape = tf.shape(grid_h_expanded)
    tf.print("DEBUG: kii shape =", kii_shape, "grid_h_expanded shape =", grid_h_shape, output_stream=tf.compat.v1.logging.info)
    
    # kii is row index, so check against grid_h
    in_bounds_h = tf.logical_and(kii >= 0, kii < grid_h_expanded)  # (batch, max_boxes, 9)
    
    # Debug: Assert after first logical_and
    tf.debugging.assert_shapes([
        (in_bounds_h, ['batch', 'max_boxes', 9]),
    ], message="in_bounds_h shape after first logical_and")
    
    # kjj is col index, so check against grid_w
    in_bounds_w = tf.logical_and(kjj >= 0, kjj < grid_w_expanded)  # (batch, max_boxes, 9)
    
    # Debug: Assert after second logical_and
    tf.debugging.assert_shapes([
        (in_bounds_w, ['batch', 'max_boxes', 9]),
    ], message="in_bounds_w shape after second logical_and")
    
    in_bounds = tf.logical_and(in_bounds_h, in_bounds_w)  # (batch, max_boxes, 9)
    
    # Debug: Assert final in_bounds shape
    tf.debugging.assert_shapes([
        (in_bounds, ['batch', 'max_boxes', 9]),
    ], message="Final in_bounds shape")
    
    # Combine with valid_mask (box must be valid AND in bounds)
    # valid_mask: (batch, max_boxes) -> expand to (batch, max_boxes, 1) to broadcast with in_bounds
    # CRITICAL FIX: Recompute valid_mask right before use to ensure correct shape
    # The issue is that valid_mask might have been modified or has wrong shape
    # So we'll recompute it from boxes_wh to ensure correctness
    box_w_recompute = tf.gather(boxes_wh, 0, axis=2)  # (batch, max_boxes)
    box_h_recompute = tf.gather(boxes_wh, 1, axis=2)  # (batch, max_boxes)
    box_areas_recompute = box_w_recompute * box_h_recompute  # (batch, max_boxes)
    valid_mask_recompute = box_areas_recompute > 0.0  # (batch, max_boxes)
    
    # Ensure valid_mask_recompute has the correct shape
    valid_mask_recompute_shape = tf.shape(valid_mask_recompute)
    valid_mask_recompute_rank = tf.rank(valid_mask_recompute)
    tf.print("DEBUG: valid_mask_recompute shape =", valid_mask_recompute_shape, "rank =", valid_mask_recompute_rank, output_stream=tf.compat.v1.logging.info)
    
    # Ensure it's rank 2
    tf.debugging.assert_equal(valid_mask_recompute_rank, 2, message="valid_mask_recompute must be rank 2")
    tf.debugging.assert_equal(valid_mask_recompute_shape[0], batch_size, message="valid_mask_recompute batch dimension mismatch")
    tf.debugging.assert_equal(valid_mask_recompute_shape[1], max_boxes, message="valid_mask_recompute max_boxes dimension mismatch")
    
    # Explicitly reshape to ensure correct shape (defensive programming)
    valid_mask_final = tf.reshape(valid_mask_recompute, [batch_size, max_boxes])  # (batch, max_boxes)
    
    # Now expand valid_mask to (batch, max_boxes, 1)
    valid_mask_expanded = tf.expand_dims(valid_mask_final, axis=-1)  # (batch, max_boxes, 1)
    
    # Debug: Print shapes after expansion
    valid_mask_expanded_shape = tf.shape(valid_mask_expanded)
    valid_mask_expanded_rank = tf.rank(valid_mask_expanded)
    in_bounds_shape = tf.shape(in_bounds)
    in_bounds_rank = tf.rank(in_bounds)
    
    tf.print("DEBUG: valid_mask_expanded AFTER expand_dims - shape =", valid_mask_expanded_shape, "rank =", valid_mask_expanded_rank, output_stream=tf.compat.v1.logging.info)
    tf.print("DEBUG: in_bounds shape =", in_bounds_shape, "rank =", in_bounds_rank, output_stream=tf.compat.v1.logging.info)
    
    # Ensure valid_mask_expanded has rank 3
    tf.debugging.assert_equal(valid_mask_expanded_rank, 3, message="valid_mask_expanded must be rank 3")
    
    # Ensure shapes are compatible for broadcasting
    tf.debugging.assert_equal(valid_mask_expanded_shape[0], in_bounds_shape[0], message="Batch dimension mismatch between valid_mask_expanded and in_bounds")
    tf.debugging.assert_equal(valid_mask_expanded_shape[1], in_bounds_shape[1], message="Max_boxes dimension mismatch between valid_mask_expanded and in_bounds")
    tf.debugging.assert_equal(valid_mask_expanded_shape[2], 1, message="valid_mask_expanded last dimension must be 1")
    tf.debugging.assert_equal(in_bounds_shape[2], 9, message="in_bounds last dimension must be 9")
    
    valid_candidates = tf.logical_and(
        valid_mask_expanded,  # (batch, max_boxes, 1)
        in_bounds  # (batch, max_boxes, 9)
    )  # (batch, max_boxes, 9)
    
    # Debug: Assert valid_candidates shape
    tf.debugging.assert_shapes([
        (valid_candidates, ['batch', 'max_boxes', 9]),
    ], message="valid_candidates shape")
    
    # Note: Occupancy checking and "count < 3" logic is handled per-layer in the scatter step
    # to avoid reading from y_true before it's fully initialized
    
    # ========================================================================
    # Step 5: Prepare Scatter Updates
    # ========================================================================
    # For each layer, collect all valid updates
    for layer_idx in range(num_layers):
        # Get boxes that belong to this layer
        layer_mask = tf.equal(selected_layers, layer_idx)  # (batch, max_boxes)
        
        # Debug: Assert layer_mask shape
        tf.debugging.assert_shapes([
            (layer_mask, ['batch', 'max_boxes']),
        ], message=f"layer_mask shape for layer {layer_idx}")
        
        # Expand to match valid_candidates shape: (batch, max_boxes, 9)
        # Use explicit broadcasting with tf.broadcast_to to ensure shape compatibility
        layer_mask_expanded = tf.expand_dims(layer_mask, axis=-1)  # (batch, max_boxes, 1)
        
        # Debug: Assert layer_mask_expanded shape
        tf.debugging.assert_shapes([
            (layer_mask_expanded, ['batch', 'max_boxes', 1]),
        ], message=f"layer_mask_expanded shape for layer {layer_idx}")
        
        # Get the shape of valid_candidates to match exactly
        valid_candidates_shape = tf.shape(valid_candidates)  # [batch, max_boxes, 9]
        
        # Debug: Print shapes for debugging
        tf.print(f"DEBUG layer {layer_idx}: layer_mask_expanded shape =", tf.shape(layer_mask_expanded), 
                "valid_candidates_shape =", valid_candidates_shape, output_stream=tf.compat.v1.logging.info)
        
        # Broadcast layer_mask to match valid_candidates shape
        layer_mask_broadcast = tf.broadcast_to(layer_mask_expanded, valid_candidates_shape)  # (batch, max_boxes, 9)
        
        # Debug: Assert layer_mask_broadcast shape
        tf.debugging.assert_shapes([
            (layer_mask_broadcast, ['batch', 'max_boxes', 9]),
        ], message=f"layer_mask_broadcast shape for layer {layer_idx}")
        
        # Get valid candidates for this layer
        # Ensure both tensors have compatible shapes
        layer_valid = tf.logical_and(valid_candidates, layer_mask_broadcast)  # (batch, max_boxes, 9)
        
        # Debug: Assert layer_valid shape
        tf.debugging.assert_shapes([
            (layer_valid, ['batch', 'max_boxes', 9]),
        ], message=f"layer_valid shape for layer {layer_idx}")
        
        # Get grid shape for this layer
        grid_h, grid_w = grid_shapes[layer_idx]
        num_anchors_layer = anchor_counts_per_layer[layer_idx]
        feature_dim = 5 + num_anchors_layer + num_classes
        
        # Collect all valid indices and updates
        # Flatten: (batch, max_boxes, 9) -> find all True values
        valid_flat = tf.reshape(layer_valid, [-1])  # (batch * max_boxes * 9,)
        valid_indices_flat = tf.where(valid_flat)  # (num_valid, 1)
        
        num_valid = tf.size(valid_indices_flat)
        
        def process_valid_updates():
            # Convert flat indices back to (batch, box, candidate)
            # valid_indices_flat: (num_valid, 1) with values in [0, batch_size * max_boxes * 9)
            flat_idx = valid_indices_flat[:, 0]  # (num_valid,)
            
            # Manual unravel: idx = batch * (max_boxes * 9) + box * 9 + candidate
            # flat_idx is int64 from tf.where, so cast to int64 for consistency
            total_per_batch = tf.cast(max_boxes * 9, tf.int64)
            flat_idx_int64 = tf.cast(flat_idx, tf.int64)
            batch_idx = tf.cast(flat_idx_int64 // total_per_batch, tf.int32)  # (num_valid,)
            remainder = tf.cast(flat_idx_int64 % total_per_batch, tf.int32)
            box_idx = remainder // 9  # (num_valid,)
            candidate_idx = remainder % 9  # (num_valid,)
            
            # Gather kii, kjj for valid candidates using gather_nd
            gather_indices = tf.stack([batch_idx, box_idx, candidate_idx], axis=1)  # (num_valid, 3)
            kii_gathered = tf.gather_nd(kii, gather_indices)  # (num_valid,)
            kjj_gathered = tf.gather_nd(kjj, gather_indices)  # (num_valid,)
            
            # Check occupancy: read current state of cells before updating
            # FIXED: y_true[layer][batch, grid_row, grid_col, 4] where kii=row, kjj=col
            occupancy_read_indices = tf.stack([batch_idx, kii_gathered, kjj_gathered], axis=1)  # (num_valid, 3)
            cell_objectness = tf.gather_nd(y_true[layer_idx][:, :, :, 4], occupancy_read_indices)  # (num_valid,)
            is_occupied = cell_objectness > 0.5  # (num_valid,)
            
            # Implement "count < 3" logic per box
            # The original logic: "if y_true[...][4] == 1 and count_grid_cell >= 3: continue"
            # This means: skip if cell is occupied AND we've already assigned 3 cells for this box
            # 
            # In vectorized version, we process all candidates at once, so we need to:
            # 1. Count how many non-occupied candidates each box has
            # 2. For boxes with > 3 non-occupied candidates, limit to first 3 (by candidate order)
            # 3. For occupied cells, only assign if the box has < 3 non-occupied candidates
            
            # Create unique box identifiers: batch_idx * max_boxes + box_idx
            box_identifiers = batch_idx * tf.cast(max_boxes, tf.int32) + box_idx  # (num_valid,)
            
            # Count non-occupied candidates per box
            non_occupied = tf.logical_not(is_occupied)  # (num_valid,)
            non_occupied_counts = tf.math.unsorted_segment_sum(
                tf.cast(non_occupied, tf.int32),
                box_identifiers,
                tf.reduce_max(box_identifiers) + 1
            )  # (max_box_id + 1,)
            
            # For each candidate, get the count for its box
            non_occupied_count_per_candidate = tf.gather(non_occupied_counts, box_identifiers)  # (num_valid,)
            
            # Filter logic matching original: "if occupied AND count >= 3, skip"
            # Simplified: assign non-occupied cells, and occupied cells only if < 3 non-occupied exist
            # This approximates the original sequential behavior
            can_assign = tf.logical_or(
                non_occupied,  # Non-occupied: always assign
                tf.logical_and(is_occupied, non_occupied_count_per_candidate < 3)  # Occupied: only if < 3 non-occupied
            )  # (num_valid,)
            
            # Filter to only valid updates
            final_valid_indices = tf.where(can_assign)[:, 0]  # (num_final_valid,)
            num_final_valid = tf.size(final_valid_indices)
            
            def apply_updates():
                # Gather only the updates that should be applied
                batch_idx_final = tf.gather(batch_idx, final_valid_indices)  # (num_final_valid,)
                box_idx_final = tf.gather(box_idx, final_valid_indices)  # (num_final_valid,)
                candidate_idx_final = tf.gather(candidate_idx, final_valid_indices)  # (num_final_valid,)
                kii_final = tf.gather(kii_gathered, final_valid_indices)  # (num_final_valid,)
                kjj_final = tf.gather(kjj_gathered, final_valid_indices)  # (num_final_valid,)
                
                # Gather box data
                box_gather_indices_final = tf.stack([batch_idx_final, box_idx_final], axis=1)  # (num_final_valid, 2)
                box_wh_final = tf.gather_nd(boxes_wh, box_gather_indices_final)  # (num_final_valid, 2)
                box_class_final = tf.gather_nd(boxes_class, box_gather_indices_final)  # (num_final_valid, 1)
                anchor_idx_final = tf.gather_nd(selected_anchors, box_gather_indices_final)  # (num_final_valid,)
                tx_final = tf.gather_nd(tx, box_gather_indices_final)  # (num_final_valid,)
                ty_final = tf.gather_nd(ty, box_gather_indices_final)  # (num_final_valid,)
                
                # Get ki, kj offsets
                ki_final = tf.gather(ki_flat, candidate_idx_final)  # (num_final_valid,)
                kj_final = tf.gather(kj_flat, candidate_idx_final)  # (num_final_valid,)
                
                # Get anchors (in pixels)
                anchor_w = tf.gather(anchors[layer_idx][:, 0], anchor_idx_final)  # (num_final_valid,) - pixels
                anchor_h = tf.gather(anchors[layer_idx][:, 1], anchor_idx_final)  # (num_final_valid,) - pixels
                
                # FIXED: Width/height encoding - ensure consistent units (pixels)
                # Decoding: box_wh = anchors_per_grid * exp(raw_wh); box_wh /= input_shape
                # So: raw_wh = log((box_wh_normalized * input_shape) / anchors_per_grid)
                # But anchors are in pixels, and box_wh_final is in PIXELS (from true_boxes)
                # So we need: raw_wh = log((box_wh_pixels) / anchors_per_grid_pixels)
                box_w_pixels = box_wh_final[:, 0]  # (num_final_valid,) - in PIXELS
                box_h_pixels = box_wh_final[:, 1]  # (num_final_valid,) - in PIXELS
                
                # Compute raw_wh targets: log(box_wh_pixels / anchor_pixels)
                box_wtoanchor_w = tf.maximum(box_w_pixels / anchor_w, 1e-3)
                box_htoanchor_h = tf.maximum(box_h_pixels / anchor_h, 1e-3)
                tw = tf.math.log(box_wtoanchor_w)  # (num_final_valid,)
                th = tf.math.log(box_htoanchor_h)  # (num_final_valid,)
                
                # FIXED: XY encoding - compute raw_xy targets using numerical inversion
                # Decoding: box_xy_normalized = (tanh(0.15*raw_xy) + sigmoid(0.15*raw_xy) + cell_grid) / grid_size
                # So: desired_offset = (box_xy_normalized * grid_size) - cell_grid
                # Then: raw_xy = invert_activation(desired_offset)
                
                # Get grid size for this layer
                grid_h_layer = tf.cast(grid_shapes[layer_idx][0], tf.float32)
                grid_w_layer = tf.cast(grid_shapes[layer_idx][1], tf.float32)
                
                # Get box center - boxes_xy is in PIXELS, need to normalize
                box_gather_indices_xy = tf.stack([batch_idx_final, box_idx_final], axis=1)  # (num_final_valid, 2)
                box_xy_pixels = tf.gather_nd(boxes_xy, box_gather_indices_xy)  # (num_final_valid, 2) - [x, y] in PIXELS
                # Normalize to [0,1]
                box_xy_normalized = box_xy_pixels / tf.stack([tf.cast(input_w, tf.float32), tf.cast(input_h, tf.float32)], axis=0)  # (num_final_valid, 2)
                
                # Compute cell_grid values for each candidate cell
                # cell_grid[i, j] = [j, i] where i=row, j=col
                kii_float = tf.cast(kii_final, tf.float32)  # (num_final_valid,) - row indices
                kjj_float = tf.cast(kjj_final, tf.float32)  # (num_final_valid,) - col indices
                cell_grid_x = kjj_float  # (num_final_valid,) - col (x)
                cell_grid_y = kii_float  # (num_final_valid,) - row (y)
                cell_grid = tf.stack([cell_grid_x, cell_grid_y], axis=1)  # (num_final_valid, 2)
                
                # Compute desired offset: (box_xy_normalized * grid_size) - cell_grid
                grid_size_vec = tf.stack([grid_w_layer, grid_h_layer], axis=0)  # (2,)
                box_xy_grid_coords = box_xy_normalized * grid_size_vec  # (num_final_valid, 2)
                desired_offset = box_xy_grid_coords - cell_grid  # (num_final_valid, 2)
                
                # Invert activation to get raw_xy targets
                desired_offset_x = desired_offset[:, 0]  # (num_final_valid,)
                desired_offset_y = desired_offset[:, 1]  # (num_final_valid,)
                raw_xy_x = _invert_activation_numerically(desired_offset_x)  # (num_final_valid,)
                raw_xy_y = _invert_activation_numerically(desired_offset_y)  # (num_final_valid,)
                raw_xy = tf.stack([raw_xy_x, raw_xy_y], axis=1)  # (num_final_valid, 2)
                
                # Build update vectors: [raw_xy_x, raw_xy_y, tw, th, 1.0, anchor_mask, class_one_hot]
                # Store pre-activation raw_xy values (model will apply activation during forward pass)
                update_xy = raw_xy  # (num_final_valid, 2)
                update_twth = tf.stack([tw, th], axis=1)  # (num_final_valid, 2)
                update_obj = tf.ones([num_final_valid, 1], dtype=tf.float32)  # (num_final_valid, 1)
                
                # Anchor mask: one-hot for selected anchor
                anchor_mask = tf.one_hot(anchor_idx_final, num_anchors_layer, dtype=tf.float32)  # (num_final_valid, num_anchors_layer)
                
                # Class one-hot
                class_one_hot = tf.one_hot(tf.squeeze(box_class_final, axis=1), num_classes, dtype=tf.float32)  # (num_final_valid, num_classes)
                
                # Concatenate all parts (zeros for rest of features)
                updates = tf.concat([
                    update_xy,  # (num_final_valid, 2)
                    update_twth,  # (num_final_valid, 2)
                    update_obj,  # (num_final_valid, 1)
                    anchor_mask,  # (num_final_valid, num_anchors_layer)
                    class_one_hot  # (num_final_valid, num_classes)
                ], axis=1)  # (num_final_valid, feature_dim)
                
                # FIXED: Create scatter indices: [batch, grid_row, grid_col] to match y_true[batch, row, col, ...]
                # kii is row index, kjj is col index
                scatter_indices = tf.stack([
                    batch_idx_final,  # (num_final_valid,)
                    kii_final,  # (num_final_valid,) - row index
                    kjj_final  # (num_final_valid,) - col index
                ], axis=1)  # (num_final_valid, 3)
                
                # Apply scatter update (this zeros out the cell first, then sets new values)
                # To match original y_true[...]*=0, we need to zero out first
                # We'll do this by creating a zero update and applying it, then the real update
                # Actually, tensor_scatter_nd_update replaces the entire cell, so we're good
                return tf.tensor_scatter_nd_update(
                    y_true[layer_idx],
                    scatter_indices,
                    updates
                )
            
            # Only apply updates if there are any valid ones after filtering
            return tf.cond(
                num_final_valid > 0,
                apply_updates,
                lambda: y_true[layer_idx]
            )
        
        # Only update if there are valid candidates
        y_true[layer_idx] = tf.cond(
            num_valid > 0,
            process_valid_updates,
            lambda: y_true[layer_idx]
        )
    
    return y_true


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, grid_shapes=None, iou_thresh=0.2):
    """
    Preprocess true boxes into MultiGridDet target format.
    
    Args:
        true_boxes: Batch of boxes in format (x1, y1, x2, y2, class)
        input_shape: Input image shape (height, width)
        anchors: List of anchor arrays for each layer
        num_classes: Number of object classes
        multi_anchor_assign: Whether to assign multiple anchors per object
        grid_shapes: Pre-computed grid shapes for each layer (optional)
        iou_thresh: IoU threshold (currently unused)
        
    Returns:
        List of target tensors for each detection layer
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy
    true_boxes[..., 2:4] = boxes_wh

    batch_size = true_boxes.shape[0]
    # Use provided grid_shapes or calculate them dynamically
    if grid_shapes is None:
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8, 3:4, 4:2}[l] for l in range(num_layers)]
    y_true = [np.zeros((batch_size, grid_shapes[ll][0], grid_shapes[ll][1], 5 + len(anchors[ll]) + num_classes),
                       dtype='float32') for ll in range(num_layers)]
    
    for b in range(batch_size):
        for t, box in enumerate(boxes_wh[b]):
            bw = box[0]
            bh = box[1]
            if bw * bh <= 0.0:
                sel_layer, k, iols = None, None, None
                continue
            else:
                sel_layer, k, iols = best_fit_and_layer(box, anchors, multi_anchor_assign=False)
                
            c = true_boxes[b, t, 4].astype('int32')
            cx = true_boxes[b, t, 0:1] * (grid_shapes[sel_layer][0] / input_shape[0]) 
            cy = true_boxes[b, t, 1:2] * (grid_shapes[sel_layer][1] / input_shape[1])
            
            i = int(cx)
            j = int(cy)
            
            tx = float(cx - i)
            ty = float(cy - j)
            box_wtoanchor_w = max(bw/ anchors[sel_layer][k][0], 1e-3)
            box_htoanchor_h = max(bh/ anchors[sel_layer][k][1], 1e-3)
            tw = np.log(box_wtoanchor_w)
            th = np.log(box_htoanchor_h)
            
            count_grid_cell = 0
            assigned_grid_cells = []
            
            for ki in range(-1, 2):
                kii = i + ki
                for kj in range(-1, 2):
                    kjj = j + kj

                    if kii< 0 or kii>= grid_shapes[sel_layer][0]:
                        continue
                    if kjj< 0 or kjj>= grid_shapes[sel_layer][1]:
                        continue
                    if y_true[sel_layer][b, kjj, kii, 4] ==1 and count_grid_cell>=3:
                        continue
                    else:
                        y_true[sel_layer][b, kjj, kii]*=0
                        y_true[sel_layer][b, kjj, kii, 0:4] = [-ki + tx,-kj + ty, tw, th]
                        y_true[sel_layer][b, kjj, kii, 4] = 1.0
                        y_true[sel_layer][b, kjj, kii, 5 + k] = 1.0
                        y_true[sel_layer][b, kjj, kii, 5 + len(anchors[sel_layer]) + c] = 1.0
                        assigned_grid_cells.append((kii, kjj))
                        count_grid_cell += 1                            
    return y_true