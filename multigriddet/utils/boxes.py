#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bounding box utilities for MultiGridDet.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union, Dict, Any


class BoxUtils:
    """Utility class for bounding box operations."""
    
    @staticmethod
    def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two boxes.
        
        Args:
            box1: First box [x, y, w, h] or [x1, y1, x2, y2]
            box2: Second box [x, y, w, h] or [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Convert to corner format if needed
        if len(box1) == 4 and len(box2) == 4:
            # Assume center format [x, y, w, h]
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Convert to corner format
            x1_min, y1_min = x1 - w1/2, y1 - h1/2
            x1_max, y1_max = x1 + w1/2, y1 + h1/2
            x2_min, y2_min = x2 - w2/2, y2 - h2/2
            x2_max, y2_max = x2 + w2/2, y2 + h2/2
        else:
            raise ValueError("Boxes must have 4 coordinates")
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def box_giou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate GIoU between two boxes.
        
        Args:
            box1: First box [x, y, w, h]
            box2: Second box [x, y, w, h]
            
        Returns:
            GIoU value
        """
        # Convert to corner format
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            inter_area = 0.0
        else:
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        # Calculate enclosed area
        enclose_xmin = min(x1_min, x2_min)
        enclose_ymin = min(y1_min, y2_min)
        enclose_xmax = max(x1_max, x2_max)
        enclose_ymax = max(y1_max, y2_max)
        enclose_area = (enclose_xmax - enclose_xmin) * (enclose_ymax - enclose_ymin)
        
        # Calculate GIoU
        giou = iou - (enclose_area - union_area) / enclose_area if enclose_area > 0 else 0.0
        
        return giou
    
    @staticmethod
    def box_diou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate DIoU between two boxes.
        
        Args:
            box1: First box [x, y, w, h]
            box2: Second box [x, y, w, h]
            
        Returns:
            DIoU value
        """
        # Convert to corner format
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            inter_area = 0.0
        else:
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        # Calculate center distance
        center1_x, center1_y = x1, y1
        center2_x, center2_y = x2, y2
        center_distance = (center1_x - center2_x)**2 + (center1_y - center2_y)**2
        
        # Calculate diagonal distance of enclosing box
        enclose_xmin = min(x1_min, x2_min)
        enclose_ymin = min(y1_min, y2_min)
        enclose_xmax = max(x1_max, x2_max)
        enclose_ymax = max(y1_max, y2_max)
        enclose_diagonal = (enclose_xmax - enclose_xmin)**2 + (enclose_ymax - enclose_ymin)**2
        
        # Calculate DIoU
        diou = iou - center_distance / enclose_diagonal if enclose_diagonal > 0 else 0.0
        
        return diou
    
    @staticmethod
    def box_ciou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate CIoU between two boxes.
        
        Args:
            box1: First box [x, y, w, h]
            box2: Second box [x, y, w, h]
            
        Returns:
            CIoU value
        """
        # Calculate DIoU first
        diou = BoxUtils.box_diou(box1, box2)
        
        # Calculate aspect ratio consistency
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate v (aspect ratio consistency)
        v = 4.0 * (np.arctan2(w1, h1) - np.arctan2(w2, h2))**2 / (np.pi**2)
        
        # Calculate alpha
        iou = BoxUtils.box_iou(box1, box2)
        alpha = v / (1.0 - iou + v) if (1.0 - iou + v) > 0 else 0.0
        
        # Calculate CIoU
        ciou = diou - alpha * v
        
        return ciou
    
    @staticmethod
    def convert_boxes_format(boxes: np.ndarray, 
                           from_format: str = 'xywh', 
                           to_format: str = 'xyxy') -> np.ndarray:
        """
        Convert boxes between different formats.
        
        Args:
            boxes: Box array of shape (N, 4)
            from_format: Source format ('xywh', 'xyxy', 'cxcywh')
            to_format: Target format ('xywh', 'xyxy', 'cxcywh')
            
        Returns:
            Converted box array
        """
        if from_format == to_format:
            return boxes.copy()
        
        converted = boxes.copy()
        
        if from_format == 'xywh' and to_format == 'xyxy':
            # [x, y, w, h] -> [x1, y1, x2, y2]
            converted[:, 2] = converted[:, 0] + converted[:, 2]
            converted[:, 3] = converted[:, 1] + converted[:, 3]
        elif from_format == 'xyxy' and to_format == 'xywh':
            # [x1, y1, x2, y2] -> [x, y, w, h]
            converted[:, 2] = converted[:, 2] - converted[:, 0]
            converted[:, 3] = converted[:, 3] - converted[:, 1]
        elif from_format == 'cxcywh' and to_format == 'xyxy':
            # [cx, cy, w, h] -> [x1, y1, x2, y2]
            converted[:, 0] = converted[:, 0] - converted[:, 2] / 2
            converted[:, 1] = converted[:, 1] - converted[:, 3] / 2
            converted[:, 2] = converted[:, 0] + converted[:, 2]
            converted[:, 3] = converted[:, 1] + converted[:, 3]
        elif from_format == 'xyxy' and to_format == 'cxcywh':
            # [x1, y1, x2, y2] -> [cx, cy, w, h]
            converted[:, 2] = converted[:, 2] - converted[:, 0]
            converted[:, 3] = converted[:, 3] - converted[:, 1]
            converted[:, 0] = converted[:, 0] + converted[:, 2] / 2
            converted[:, 1] = converted[:, 1] + converted[:, 3] / 2
        elif from_format == 'xywh' and to_format == 'cxcywh':
            # [x, y, w, h] -> [cx, cy, w, h]
            converted[:, 0] = converted[:, 0] + converted[:, 2] / 2
            converted[:, 1] = converted[:, 1] + converted[:, 3] / 2
        elif from_format == 'cxcywh' and to_format == 'xywh':
            # [cx, cy, w, h] -> [x, y, w, h]
            converted[:, 0] = converted[:, 0] - converted[:, 2] / 2
            converted[:, 1] = converted[:, 1] - converted[:, 3] / 2
        else:
            raise ValueError(f"Unsupported conversion from {from_format} to {to_format}")
        
        return converted
    
    @staticmethod
    def clip_boxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Clip boxes to image boundaries.
        
        Args:
            boxes: Box array of shape (N, 4) in xyxy format
            image_shape: Image shape (height, width)
            
        Returns:
            Clipped box array
        """
        height, width = image_shape
        clipped = boxes.copy()
        
        # Clip x coordinates
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
        clipped[:, 2] = np.clip(clipped[:, 2], 0, width)
        
        # Clip y coordinates
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
        clipped[:, 3] = np.clip(clipped[:, 3], 0, height)
        
        return clipped
    
    @staticmethod
    def filter_boxes_by_area(boxes: np.ndarray, 
                           min_area: float = 0.0, 
                           max_area: float = float('inf')) -> np.ndarray:
        """
        Filter boxes by area.
        
        Args:
            boxes: Box array of shape (N, 4) in xywh format
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            
        Returns:
            Boolean mask for valid boxes
        """
        areas = boxes[:, 2] * boxes[:, 3]
        return (areas >= min_area) & (areas <= max_area)
    
    @staticmethod
    def filter_boxes_by_aspect_ratio(boxes: np.ndarray, 
                                   min_ratio: float = 0.0, 
                                   max_ratio: float = float('inf')) -> np.ndarray:
        """
        Filter boxes by aspect ratio.
        
        Args:
            boxes: Box array of shape (N, 4) in xywh format
            min_ratio: Minimum aspect ratio (w/h)
            max_ratio: Maximum aspect ratio (w/h)
            
        Returns:
            Boolean mask for valid boxes
        """
        aspect_ratios = boxes[:, 2] / (boxes[:, 3] + 1e-8)
        return (aspect_ratios >= min_ratio) & (aspect_ratios <= max_ratio)


# Convenience functions
def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Convenience function for IoU calculation."""
    return BoxUtils.box_iou(box1, box2)


def box_giou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Convenience function for GIoU calculation."""
    return BoxUtils.box_giou(box1, box2)


def box_diou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Convenience function for DIoU calculation."""
    return BoxUtils.box_diou(box1, box2)


def box_ciou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Convenience function for CIoU calculation."""
    return BoxUtils.box_ciou(box1, box2)
