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


def _parse_annotation_numpy(annotation_line_tensor):
    """Parse annotation line using NumPy (called via tf.py_function)."""
    # Convert tensor to string
    if hasattr(annotation_line_tensor, 'numpy'):
        annotation_line = annotation_line_tensor.numpy().decode('utf-8')
    elif isinstance(annotation_line_tensor, bytes):
        annotation_line = annotation_line_tensor.decode('utf-8')
    else:
        annotation_line = str(annotation_line_tensor)
    
    parts = annotation_line.split(' ', 1)
    image_path = parts[0]
    boxes_string = parts[1] if len(parts) > 1 else ''
    return image_path.encode('utf-8'), boxes_string.encode('utf-8')

def tf_parse_annotation_line(annotation_line: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse annotation line to extract image path and boxes.
    
    Args:
        annotation_line: Tensor containing annotation line string (scalar)
        
    Returns:
        Tuple of (image_path, boxes_string)
    """
    # Use py_function for safe parsing
    image_path_bytes, boxes_string_bytes = tf.py_function(
        func=_parse_annotation_numpy,
        inp=[annotation_line],
        Tout=[tf.string, tf.string]
    )
    image_path_bytes.set_shape([])
    boxes_string_bytes.set_shape([])
    return image_path_bytes, boxes_string_bytes


def tf_parse_boxes(boxes_string: tf.Tensor) -> tf.Tensor:
    """
    Parse boxes from string format "x1,y1,x2,y2,class x1,y1,x2,y2,class ..."
    
    Args:
        boxes_string: Tensor containing boxes string
        
    Returns:
        Boxes tensor of shape (N, 5) with dtype float32
    """
    def _parse_boxes_numpy(box_str):
        if box_str == b'':
            return np.zeros((0, 5), dtype=np.float32)
        box_list = []
        for box in box_str.decode('utf-8').split():
            coords = [float(x) for x in box.split(',')]
            if len(coords) == 5:
                box_list.append(coords)
        if len(box_list) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(box_list, dtype=np.float32)
    
    boxes = tf.numpy_function(_parse_boxes_numpy, [boxes_string], tf.float32)
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
                        use_gpu_preprocessing: bool = True):
        """
        Build native tf.data.Dataset pipeline for GPU-accelerated data loading.
        
        This creates a true tf.data pipeline that can be parallelized and run on GPU,
        unlike from_generator() which still runs Python code on CPU.
        
        Args:
            prefetch_buffer_size: Number of batches to prefetch. Use tf.data.AUTOTUNE for automatic tuning.
            num_parallel_calls: Number of parallel calls for map operations. Use tf.data.AUTOTUNE for automatic tuning.
            use_gpu_preprocessing: Whether to use GPU-accelerated preprocessing (default: True)
            
        Returns:
            tf.data.Dataset configured with prefetching and parallel processing
        """
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Create dataset from annotation lines
        annotation_paths = tf.constant(self.annotation_lines)
        dataset = tf.data.Dataset.from_tensor_slices(annotation_paths)
        
        # Shuffle dataset
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, len(self.annotation_lines)), 
                                     reshuffle_each_iteration=True)
        
        # Parse annotation and load image
        def _load_and_parse(annotation_line):
            image_path, boxes_string = tf_parse_annotation_line(annotation_line)
            image = tf_load_and_decode_image(image_path)
            boxes = tf_parse_boxes(boxes_string)
            return image, boxes, image_path
        
        dataset = dataset.map(_load_and_parse, num_parallel_calls=num_parallel_calls)
        
        # Preprocess and augment
        def _preprocess_image_and_boxes(image, boxes, image_path):
            # Get target shape (handle multi-scale if needed)
            target_shape = self.input_shape
            
            # Convert image to float32 and normalize
            image = tf.cast(image, tf.float32) / 255.0
            
            # Apply letterbox resize
            image_resized = tf_letterbox_resize(image, target_shape, return_padding_info=False)
            
            # Apply augmentations if enabled
            if self.augment:
                # Random horizontal flip
                image_resized, boxes = tf_random_horizontal_flip(image_resized, boxes)
                
                # Color augmentations
                image_resized = tf_random_brightness(image_resized, max_delta=0.2)
                image_resized = tf_random_contrast(image_resized, lower=0.8, upper=1.2)
                image_resized = tf_random_saturation(image_resized, lower=0.8, upper=1.2)
                image_resized = tf_random_hue(image_resized, max_delta=0.1)
                image_resized = tf_random_grayscale(image_resized, probability=0.1)
            
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
        
        # Process batches: pad boxes, create targets
        # Capture anchors and other attributes for use in py_function
        anchors_np = [np.array(anchor) if isinstance(anchor, tf.Tensor) else np.array(anchor) 
                     for anchor in self.anchors]
        input_shape_np = self.input_shape
        num_classes_np = self.num_classes
        multi_anchor_assign_np = self.multi_anchor_assign
        grid_shapes_np = self.grid_shapes
        
        def _process_batch_numpy(images_tensor, boxes_tensor):
            """
            Process batch using NumPy operations (for preprocess_true_boxes).
            This will be called via tf.py_function.
            """
            # Convert tensors to numpy
            if hasattr(images_tensor, 'numpy'):
                images_np = images_tensor.numpy()
            else:
                images_np = np.array(images_tensor)
            
            if hasattr(boxes_tensor, 'numpy'):
                boxes_dense = boxes_tensor.numpy()
            else:
                boxes_dense = np.array(boxes_tensor)
            
            batch_size = images_np.shape[0]
            
            # boxes_dense is a dense numpy array of shape (batch_size, max_boxes, 5)
            # Remove padding (boxes with all zeros are padding)
            boxes_padded = boxes_dense.copy()
            
            # Find actual number of boxes per image (non-zero boxes)
            for i in range(batch_size):
                # Find last non-zero box
                non_zero_mask = np.any(boxes_padded[i] != 0, axis=1)
                if np.any(non_zero_mask):
                    # Keep only non-zero boxes, but maintain max_boxes shape
                    # The padding is already there, just ensure we don't process zero boxes
                    pass
                else:
                    # No boxes for this image, ensure at least one zero box
                    boxes_padded[i, 0] = [0, 0, 0, 0, 0]
            
            # Use existing preprocess_true_boxes function
            y_true = preprocess_true_boxes(
                boxes_padded, input_shape_np, 
                anchors_np,
                num_classes_np, multi_anchor_assign_np, 
                grid_shapes=grid_shapes_np
            )
            
            return images_np, *y_true, np.zeros(batch_size, dtype=np.float32)
        
        # Use py_function for complex batch processing
        def _process_batch_wrapper(images, boxes_dense):
            # boxes_dense is already converted to dense tensor in previous map operation
            
            # Get output types - need to return tuple structure
            num_y_true = len(self.anchors)
            num_outputs = 1 + num_y_true + 1  # images + y_true layers + dummy
            
            # Process batch - pass boxes as dense tensor
            results = tf.py_function(
                func=_process_batch_numpy,
                inp=[images, boxes_dense],
                Tout=[tf.float32] * num_outputs
            )
            
            # Unpack results
            images_processed = results[0]
            y_true_list = results[1:1+num_y_true]
            dummy_targets = results[-1]
            
            # Set shapes - critical for tf.py_function outputs
            images_processed.set_shape([None, self.input_shape[0], self.input_shape[1], 3])
            dummy_targets.set_shape([None])
            
            # Set shapes for y_true tensors based on grid_shapes
            for i, y_true in enumerate(y_true_list):
                grid_h, grid_w = self.grid_shapes[i]
                num_anchors = len(self.anchors[i])
                # Shape: (batch, grid_h, grid_w, 5 + num_anchors + num_classes)
                y_true.set_shape([None, grid_h, grid_w, 5 + num_anchors + self.num_classes])
            
            # Return in format expected by model: (inputs_tuple, targets)
            return (images_processed, *y_true_list), dummy_targets
        
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