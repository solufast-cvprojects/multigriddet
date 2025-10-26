#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image preprocessing utilities for MultiGridDet.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
from PIL import Image


class ImagePreprocessor:
    """Image preprocessing utilities for MultiGridDet."""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    interpolation: int = cv2.INTER_AREA) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image as numpy array
            target_size: Target size (width, height)
            interpolation: OpenCV interpolation method
            
        Returns:
            resized_image: Resized image
        """
        return cv2.resize(image, target_size, interpolation=interpolation)
    
    @staticmethod
    def letterbox_resize(image: np.ndarray, target_size: Tuple[int, int], 
                        color: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Resize image with unchanged aspect ratio using padding.
        
        Args:
            image: Input image as numpy array
            target_size: Target size (width, height)
            color: Padding color (R, G, B)
            
        Returns:
            resized_image: Resized image with padding
            padding_size: Original image size after scaling
            offset: Top-left offset of the image in the padded result
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
        
        # Calculate offset
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        
        # Place resized image
        padded[dy:dy + new_h, dx:dx + new_w] = resized
        
        return padded, (new_w, new_h), (dx, dy)
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean: Optional[np.ndarray] = None, 
                       std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize image.
        
        Args:
            image: Input image as numpy array
            mean: Mean values for normalization (default: [0.485, 0.456, 0.406])
            std: Std values for normalization (default: [0.229, 0.224, 0.225])
            
        Returns:
            normalized_image: Normalized image
        """
        if mean is None:
            mean = np.array([0.485, 0.456, 0.406])
        if std is None:
            std = np.array([0.229, 0.224, 0.225])
        
        # Convert to float
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        normalized = (image - mean) / std
        
        return normalized
    
    @staticmethod
    def denormalize_image(image: np.ndarray, mean: Optional[np.ndarray] = None, 
                         std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Denormalize image.
        
        Args:
            image: Normalized image as numpy array
            mean: Mean values used for normalization
            std: Std values used for normalization
            
        Returns:
            denormalized_image: Denormalized image
        """
        if mean is None:
            mean = np.array([0.485, 0.456, 0.406])
        if std is None:
            std = np.array([0.229, 0.224, 0.225])
        
        # Denormalize
        denormalized = image * std + mean
        
        # Convert back to uint8
        denormalized = np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)
        
        return denormalized
    
    @staticmethod
    def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to BGR."""
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def adjust_boxes_for_resize(boxes: np.ndarray, original_size: Tuple[int, int], 
                               new_size: Tuple[int, int], 
                               padding_offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Adjust bounding boxes for image resize with padding.
        
        Args:
            boxes: Bounding boxes in format (x_min, y_min, x_max, y_max)
            original_size: Original image size (width, height)
            new_size: New image size (width, height)
            padding_offset: Padding offset (dx, dy)
            
        Returns:
            adjusted_boxes: Adjusted bounding boxes
        """
        if len(boxes) == 0:
            return boxes
        
        orig_w, orig_h = original_size
        new_w, new_h = new_size
        dx, dy = padding_offset
        
        # Calculate scale factors
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        # Adjust boxes
        adjusted_boxes = boxes.copy()
        adjusted_boxes[:, [0, 2]] = adjusted_boxes[:, [0, 2]] * scale_x + dx
        adjusted_boxes[:, [1, 3]] = adjusted_boxes[:, [1, 3]] * scale_y + dy
        
        return adjusted_boxes
    
    @staticmethod
    def preprocess_for_inference(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array
            target_size: Target size (width, height)
            
        Returns:
            preprocessed_image: Preprocessed image ready for model
            metadata: Metadata about preprocessing (for postprocessing)
        """
        # Store original size
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Letterbox resize
        resized, padding_size, offset = ImagePreprocessor.letterbox_resize(image, target_size)
        
        # Normalize
        normalized = ImagePreprocessor.normalize_image(resized)
        
        # Add batch dimension
        preprocessed = np.expand_dims(normalized, axis=0)
        
        # Create metadata
        metadata = {
            'original_size': original_size,
            'padding_size': padding_size,
            'offset': offset,
            'scale': (padding_size[0] / original_size[0], padding_size[1] / original_size[1])
        }
        
        return preprocessed, metadata
