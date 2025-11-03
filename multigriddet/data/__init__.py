"""
MultiGridDet Data Module.

This module contains data processing components:
- Generators: Keras Sequence generators for training
- Augmentation: Data augmentation utilities
- Preprocessing: Image preprocessing utilities
- Utils: Dataset utility functions

Example:
    >>> from multigriddet.data import MultiGridDataGenerator
    >>> 
    >>> # Create data generator
    >>> generator = MultiGridDataGenerator(
    ...     annotation_lines=lines,
    ...     batch_size=16,
    ...     input_shape=(608, 608),
    ...     anchors=anchors,
    ...     num_classes=80
    ... )
"""

# Import data generators
from .generators import MultiGridDataGenerator

# Import augmentation functions
from .augmentation import (
    letterbox_resize, random_resize_crop_pad, reshape_boxes,
    random_hsv_distort, random_horizontal_flip, random_vertical_flip,
    random_grayscale, random_brightness, random_chroma, random_contrast,
    random_sharpness, random_blur, random_motion_blur, random_mosaic_augment,
    random_rotate, random_gridmask, normalize_image, denormalize_image,
    preprocess_image, augmenter_defn, augmenter_defn_advncd, augmenter,
    augmenter_batch, augment_image
)

# Import utility functions
from .utils import (
    get_multiscale_list, resize_anchors, get_classes, get_anchors,
    get_colors, load_annotation_lines, draw_label, draw_boxes
)

# Import preprocessing
from .preprocessing import ImagePreprocessor

__all__ = [
    # Data generators
    'MultiGridDataGenerator',
    
    # Augmentation functions
    'letterbox_resize', 'random_resize_crop_pad', 'reshape_boxes',
    'random_hsv_distort', 'random_horizontal_flip', 'random_vertical_flip',
    'random_grayscale', 'random_brightness', 'random_chroma', 'random_contrast',
    'random_sharpness', 'random_blur', 'random_motion_blur', 'random_mosaic_augment',
    'random_rotate', 'random_gridmask', 'normalize_image', 'denormalize_image',
    'preprocess_image', 'augmenter_defn', 'augmenter_defn_advncd', 'augmenter',
    'augmenter_batch', 'augment_image',
    
    # Utility functions
    'get_multiscale_list', 'resize_anchors', 'get_classes', 'get_anchors',
    'get_colors', 'load_annotation_lines', 'draw_label', 'draw_boxes',
    
    # Preprocessing
    'ImagePreprocessor'
]
