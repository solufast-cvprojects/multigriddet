#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anchor utilities for MultiGridDet.
Includes IoL (Intersection over Largest) calculation and anchor management.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union, Dict, Any
import os


class AnchorUtils:
    """Utility class for anchor operations in MultiGridDet."""
    
    @staticmethod
    def calculate_iol(anchors: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Calculate IoL (Intersection over Largest) between anchors and boxes.
        
        This is the key innovation in MultiGridDet for better anchor matching,
        especially for objects with extreme aspect ratios.
        
        Args:
            anchors: Anchor boxes of shape (M, 2) - width and height
            boxes: Object boxes of shape (N, 2) - width and height
            
        Returns:
            IoL scores of shape (N, M)
        """
        # Expand dimensions for broadcasting
        boxes_expanded = np.expand_dims(boxes, axis=-2)  # (N, 1, 2)
        anchors_expanded = np.expand_dims(anchors, axis=0)  # (1, M, 2)
        
        # Calculate intersection
        intersection_wh = np.minimum(boxes_expanded, anchors_expanded)
        
        # Calculate areas
        boxes_areas = boxes[..., 0] * boxes[..., 1]  # (N,)
        anchors_areas = anchors[:, 0] * anchors[:, 1]  # (M,)
        
        # Calculate intersection areas
        intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1]
        
        # Calculate largest areas
        boxes_areas_expanded = np.expand_dims(boxes_areas, axis=-1)  # (N, 1)
        anchors_areas_expanded = np.expand_dims(anchors_areas, axis=0)  # (1, M)
        largest_areas = np.maximum(boxes_areas_expanded, anchors_areas_expanded)
        
        # Calculate IoL
        iol = intersection_areas / (largest_areas + 1e-8)
        
        return iol
    
    @staticmethod
    def calculate_iou(anchors: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Calculate traditional IoU between anchors and boxes.
        
        Args:
            anchors: Anchor boxes of shape (M, 2) - width and height
            boxes: Object boxes of shape (N, 2) - width and height
            
        Returns:
            IoU scores of shape (N, M)
        """
        # Expand dimensions for broadcasting
        boxes_expanded = np.expand_dims(boxes, axis=-2)  # (N, 1, 2)
        anchors_expanded = np.expand_dims(anchors, axis=0)  # (1, M, 2)
        
        # Calculate intersection
        intersection_wh = np.minimum(boxes_expanded, anchors_expanded)
        
        # Calculate areas
        boxes_areas = boxes[..., 0] * boxes[..., 1]  # (N,)
        anchors_areas = anchors[:, 0] * anchors[:, 1]  # (M,)
        
        # Calculate intersection areas
        intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1]
        
        # Calculate union areas
        boxes_areas_expanded = np.expand_dims(boxes_areas, axis=-1)  # (N, 1)
        anchors_areas_expanded = np.expand_dims(anchors_areas, axis=0)  # (1, M)
        union_areas = boxes_areas_expanded + anchors_areas_expanded - intersection_areas
        
        # Calculate IoU
        iou = intersection_areas / (union_areas + 1e-8)
        
        return iou
    
    @staticmethod
    def find_best_anchor(anchors: np.ndarray, box: np.ndarray, use_iol: bool = True) -> Tuple[int, float]:
        """
        Find the best anchor for a given box.
        
        Args:
            anchors: Anchor array of shape (M, 2)
            box: Box of shape (2,) - width and height
            use_iol: Whether to use IoL instead of IoU
            
        Returns:
            Tuple of (best_anchor_index, best_score)
        """
        if use_iol:
            scores = AnchorUtils.calculate_iol(anchors, box.reshape(1, 2))
        else:
            scores = AnchorUtils.calculate_iou(anchors, box.reshape(1, 2))
        
        best_idx = np.argmax(scores[0])
        best_score = scores[0, best_idx]
        
        return best_idx, best_score
    
    @staticmethod
    def generate_default_anchors(dataset: str = 'coco') -> List[np.ndarray]:
        """
        Generate default anchors for common datasets.
        
        Args:
            dataset: Dataset name ('coco', 'voc', 'custom')
            
        Returns:
            List of anchor arrays for each scale
        """
        if dataset == 'coco':
            # COCO anchors (from YOLOv3)
            anchors = [
                np.array([[10, 13], [16, 30], [33, 23]]),  # Scale 1 (32x downsampling)
                np.array([[30, 61], [62, 45], [59, 119]]), # Scale 2 (16x downsampling)
                np.array([[116, 90], [156, 198], [373, 326]]) # Scale 3 (8x downsampling)
            ]
        elif dataset == 'voc':
            # VOC anchors (from YOLOv3)
            anchors = [
                np.array([[10, 13], [16, 30], [33, 23]]),  # Scale 1
                np.array([[30, 61], [62, 45], [59, 119]]), # Scale 2
                np.array([[116, 90], [156, 198], [373, 326]]) # Scale 3
            ]
        else:
            # Default anchors
            anchors = [
                np.array([[10, 13], [16, 30], [33, 23]]),
                np.array([[30, 61], [62, 45], [59, 119]]),
                np.array([[116, 90], [156, 198], [373, 326]])
            ]
        
        return anchors
    
    @staticmethod
    def load_anchors_from_file(filepath: str) -> List[np.ndarray]:
        """
        Load anchors from a text file.
        
        Args:
            filepath: Path to anchor file
            
        Returns:
            List of anchor arrays
        """
        anchors = []
        current_scale = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_scale:
                        anchors.append(np.array(current_scale))
                        current_scale = []
                else:
                    # Parse anchor (width, height)
                    parts = line.split(',')
                    if len(parts) == 2:
                        width, height = float(parts[0]), float(parts[1])
                        current_scale.append([width, height])
        
        # Add last scale if exists
        if current_scale:
            anchors.append(np.array(current_scale))
        
        return anchors
    
    @staticmethod
    def save_anchors_to_file(anchors: List[np.ndarray], filepath: str):
        """
        Save anchors to a text file.
        
        Args:
            anchors: List of anchor arrays
            filepath: Path to save anchors
        """
        with open(filepath, 'w') as f:
            for scale_idx, scale_anchors in enumerate(anchors):
                if scale_idx > 0:
                    f.write('\n')  # Empty line between scales
                
                for anchor in scale_anchors:
                    f.write(f'{anchor[0]},{anchor[1]}\n')
    
    @staticmethod
    def validate_anchors(anchors: List[np.ndarray]) -> bool:
        """
        Validate anchor format and values.
        
        Args:
            anchors: List of anchor arrays
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(anchors, list):
            return False
        
        if len(anchors) == 0:
            return False
        
        for scale_idx, scale_anchors in enumerate(anchors):
            if not isinstance(scale_anchors, np.ndarray):
                return False
            
            if scale_anchors.ndim != 2 or scale_anchors.shape[1] != 2:
                return False
            
            # Check for positive values
            if np.any(scale_anchors <= 0):
                return False
        
        return True
    
    @staticmethod
    def get_anchor_info(anchors: List[np.ndarray]) -> Dict[str, Any]:
        """
        Get information about anchors.
        
        Args:
            anchors: List of anchor arrays
            
        Returns:
            Dictionary with anchor information
        """
        if not AnchorUtils.validate_anchors(anchors):
            return {'valid': False, 'error': 'Invalid anchor format'}
        
        info = {
            'valid': True,
            'num_scales': len(anchors),
            'anchors_per_scale': [len(scale) for scale in anchors],
            'total_anchors': sum(len(scale) for scale in anchors),
            'aspect_ratios': [],
            'areas': []
        }
        
        for scale_idx, scale_anchors in enumerate(anchors):
            # Calculate aspect ratios
            aspect_ratios = scale_anchors[:, 0] / scale_anchors[:, 1]
            info['aspect_ratios'].append(aspect_ratios.tolist())
            
            # Calculate areas
            areas = scale_anchors[:, 0] * scale_anchors[:, 1]
            info['areas'].append(areas.tolist())
        
        return info


# Convenience functions
def calculate_iol(anchors: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Convenience function for IoL calculation."""
    return AnchorUtils.calculate_iol(anchors, boxes)


def calculate_iou(anchors: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Convenience function for IoU calculation."""
    return AnchorUtils.calculate_iou(anchors, boxes)


def find_best_anchor(anchors: np.ndarray, box: np.ndarray, use_iol: bool = True) -> Tuple[int, float]:
    """Convenience function for finding best anchor."""
    return AnchorUtils.find_best_anchor(anchors, box, use_iol)


def load_anchors(anchors_path: str) -> List[np.ndarray]:
    """
    Load anchors from file.
    
    Args:
        anchors_path: Path to anchors file
        
    Returns:
        List of anchor arrays for each scale
    """
    anchors = []
    with open(anchors_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Parse anchor line (format: w1,h1 w2,h2 w3,h3)
                anchor_pairs = line.split()
                anchor_array = []
                for pair in anchor_pairs:
                    if pair and ',' in pair:  # Skip empty pairs and ensure comma exists
                        # Remove trailing comma if present
                        pair = pair.rstrip(',')
                        try:
                            w, h = map(float, pair.split(','))
                            anchor_array.append([w, h])
                        except ValueError:
                            # Skip invalid pairs
                            continue
                if anchor_array:  # Only add if we have valid anchors
                    anchors.append(np.array(anchor_array))
    return anchors


def load_classes(classes_path: str) -> List[str]:
    """
    Load class names from file.
    
    Args:
        classes_path: Path to classes file
        
    Returns:
        List of class names
    """
    classes = []
    with open(classes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)
    return classes
