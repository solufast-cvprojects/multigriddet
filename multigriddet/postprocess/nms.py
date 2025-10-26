#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Maximum Suppression (NMS) implementations for MultiGridDet.
Includes standard NMS, DIoU-NMS, SoftNMS, and ClusterNMS.
"""

import numpy as np
from typing import List, Tuple, Optional, Union


class NMS:
    """Base NMS class."""
    
    def __init__(self, use_iol: bool = False):
        """
        Initialize NMS.
        
        Args:
            use_iol: Whether to use IoL instead of IoU
        """
        self.use_iol = use_iol
    
    def apply_nms(self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
                  nms_threshold: float, confidence: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Apply Non-Maximum Suppression.
        
        Args:
            boxes: Bounding boxes
            classes: Class labels
            scores: Confidence scores
            nms_threshold: IoU threshold for NMS
            confidence: Confidence threshold
            
        Returns:
            Tuple of (filtered_boxes, filtered_classes, filtered_scores)
        """
        raise NotImplementedError("Subclasses must implement apply_nms method")
    
    def _compute_iou(self, boxes: np.ndarray, use_iol: bool = False) -> np.ndarray:
        """
        Compute IoU or IoL between boxes.
        
        Args:
            boxes: Bounding boxes
            use_iol: Whether to use IoL instead of IoU
            
        Returns:
            IoU/IoL matrix
        """
        if len(boxes) == 0:
            return np.array([])
        
        # Get box coordinates and areas
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        areas = w * h
        
        # Compute intersection
        inter_xmin = np.maximum(x[1:], x[0])
        inter_ymin = np.maximum(y[1:], y[0])
        inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
        inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])
        
        inter_w = np.maximum(0.0, inter_xmax - inter_xmin)
        inter_h = np.maximum(0.0, inter_ymax - inter_ymin)
        inter = inter_w * inter_h
        
        if use_iol:
            # IoL: Intersection over Largest
            iou = inter / np.maximum(areas[1:], areas[0])
        else:
            # IoU: Intersection over Union
            union = areas[1:] + areas[0] - inter
            iou = inter / (union + 1e-8)
        
        return iou


class StandardNMS(NMS):
    """Standard Non-Maximum Suppression."""
    
    def apply_nms(self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
                  nms_threshold: float, confidence: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Apply standard NMS."""
        if len(boxes) == 0:
            return [], [], []
        
        # Sort by confidence score
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Take the box with highest confidence
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Compute IoU with remaining boxes
            current_box = boxes[current_idx:current_idx+1]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self._compute_iou_between_boxes(current_box, remaining_boxes)
            
            # Keep boxes with IoU below threshold
            keep_mask = ious < nms_threshold
            sorted_indices = sorted_indices[1:][keep_mask]
        
        # Return filtered results
        if keep_indices:
            return [boxes[keep_indices]], [classes[keep_indices]], [scores[keep_indices]]
        else:
            return [], [], []
    
    def _compute_iou_between_boxes(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        if len(boxes2) == 0:
            return np.array([])
        
        # Get coordinates
        x1, y1, w1, h1 = box1[0]
        x2, y2, w2, h2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
        # Compute intersection
        inter_xmin = np.maximum(x1, x2)
        inter_ymin = np.maximum(y1, y2)
        inter_xmax = np.minimum(x1 + w1, x2 + w2)
        inter_ymax = np.minimum(y1 + h1, y2 + h2)
        
        inter_w = np.maximum(0.0, inter_xmax - inter_xmin)
        inter_h = np.maximum(0.0, inter_ymax - inter_ymin)
        inter = inter_w * inter_h
        
        # Compute areas
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        
        # Compute IoU
        iou = inter / (union + 1e-8)
        
        return iou


class DIoUNMS(NMS):
    """DIoU-based Non-Maximum Suppression."""
    
    def apply_nms(self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
                  nms_threshold: float, confidence: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Apply DIoU-based NMS."""
        if len(boxes) == 0:
            return [], [], []
        
        # Sort by confidence score
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Take the box with highest confidence
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Compute DIoU with remaining boxes
            current_box = boxes[current_idx:current_idx+1]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            dious = self._compute_diou_between_boxes(current_box, remaining_boxes)
            
            # Keep boxes with DIoU below threshold
            keep_mask = dious < nms_threshold
            sorted_indices = sorted_indices[1:][keep_mask]
        
        # Return filtered results
        if keep_indices:
            return [boxes[keep_indices]], [classes[keep_indices]], [scores[keep_indices]]
        else:
            return [], [], []
    
    def _compute_diou_between_boxes(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute DIoU between one box and multiple boxes."""
        if len(boxes2) == 0:
            return np.array([])
        
        # Get coordinates
        x1, y1, w1, h1 = box1[0]
        x2, y2, w2, h2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
        # Compute intersection
        inter_xmin = np.maximum(x1, x2)
        inter_ymin = np.maximum(y1, y2)
        inter_xmax = np.minimum(x1 + w1, x2 + w2)
        inter_ymax = np.minimum(y1 + h1, y2 + h2)
        
        inter_w = np.maximum(0.0, inter_xmax - inter_xmin)
        inter_h = np.maximum(0.0, inter_ymax - inter_ymin)
        inter = inter_w * inter_h
        
        # Compute areas
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        
        # Compute IoU
        iou = inter / (union + 1e-8)
        
        # Compute center distance
        center1_x, center1_y = x1 + w1/2, y1 + h1/2
        center2_x, center2_y = x2 + w2/2, y2 + h2/2
        center_distance = (center1_x - center2_x)**2 + (center1_y - center2_y)**2
        
        # Compute diagonal distance of enclosing box
        enclose_xmin = np.minimum(x1, x2)
        enclose_ymin = np.minimum(y1, y2)
        enclose_xmax = np.maximum(x1 + w1, x2 + w2)
        enclose_ymax = np.maximum(y1 + h1, y2 + h2)
        enclose_diagonal = (enclose_xmax - enclose_xmin)**2 + (enclose_ymax - enclose_ymin)**2
        
        # Compute DIoU
        diou = iou - center_distance / (enclose_diagonal + 1e-8)
        
        return diou


class SoftNMS(NMS):
    """Soft Non-Maximum Suppression."""
    
    def __init__(self, sigma: float = 0.5, score_threshold: float = 0.001):
        """
        Initialize SoftNMS.
        
        Args:
            sigma: SoftNMS sigma parameter
            score_threshold: Score threshold for final filtering
        """
        super().__init__()
        self.sigma = sigma
        self.score_threshold = score_threshold
    
    def apply_nms(self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
                  nms_threshold: float, confidence: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Apply SoftNMS."""
        if len(boxes) == 0:
            return [], [], []
        
        # Sort by confidence score
        sorted_indices = np.argsort(scores)[::-1]
        
        # Apply soft suppression
        soft_scores = scores.copy()
        
        for i in range(len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_score = soft_scores[current_idx]
            
            if current_score < self.score_threshold:
                soft_scores[current_idx] = 0
                continue
            
            # Compute IoU with remaining boxes
            remaining_indices = sorted_indices[i+1:]
            if len(remaining_indices) == 0:
                break
            
            current_box = boxes[current_idx:current_idx+1]
            remaining_boxes = boxes[remaining_indices]
            
            ious = self._compute_iou_between_boxes(current_box, remaining_boxes)
            
            # Apply soft suppression
            soft_scores[remaining_indices] *= np.exp(-ious**2 / self.sigma)
        
        # Filter by final score threshold
        keep_mask = soft_scores >= self.score_threshold
        
        if np.any(keep_mask):
            return [boxes[keep_mask]], [classes[keep_mask]], [soft_scores[keep_mask]]
        else:
            return [], [], []
    
    def _compute_iou_between_boxes(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        if len(boxes2) == 0:
            return np.array([])
        
        # Get coordinates
        x1, y1, w1, h1 = box1[0]
        x2, y2, w2, h2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
        # Compute intersection
        inter_xmin = np.maximum(x1, x2)
        inter_ymin = np.maximum(y1, y2)
        inter_xmax = np.minimum(x1 + w1, x2 + w2)
        inter_ymax = np.minimum(y1 + h1, y2 + h2)
        
        inter_w = np.maximum(0.0, inter_xmax - inter_xmin)
        inter_h = np.maximum(0.0, inter_ymax - inter_ymin)
        inter = inter_w * inter_h
        
        # Compute areas
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        
        # Compute IoU
        iou = inter / (union + 1e-8)
        
        return iou


class ClusterNMS(NMS):
    """Cluster-based Non-Maximum Suppression for faster processing."""
    
    def apply_nms(self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray,
                  nms_threshold: float, confidence: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Apply ClusterNMS."""
        if len(boxes) == 0:
            return [], [], []
        
        # Sort by confidence score
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Take the box with highest confidence
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Compute IoU with remaining boxes
            current_box = boxes[current_idx:current_idx+1]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self._compute_iou_between_boxes(current_box, remaining_boxes)
            
            # Keep boxes with IoU below threshold
            keep_mask = ious < nms_threshold
            sorted_indices = sorted_indices[1:][keep_mask]
        
        # Return filtered results
        if keep_indices:
            return [boxes[keep_indices]], [classes[keep_indices]], [scores[keep_indices]]
        else:
            return [], [], []
    
    def _compute_iou_between_boxes(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        if len(boxes2) == 0:
            return np.array([])
        
        # Get coordinates
        x1, y1, w1, h1 = box1[0]
        x2, y2, w2, h2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
        # Compute intersection
        inter_xmin = np.maximum(x1, x2)
        inter_ymin = np.maximum(y1, y2)
        inter_xmax = np.minimum(x1 + w1, x2 + w2)
        inter_ymax = np.minimum(y1 + h1, y2 + h2)
        
        inter_w = np.maximum(0.0, inter_xmax - inter_xmin)
        inter_h = np.maximum(0.0, inter_ymax - inter_ymin)
        inter = inter_w * inter_h
        
        # Compute areas
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        
        # Compute IoU
        iou = inter / (union + 1e-8)
        
        return iou


# Backward compatibility functions
def nms_boxes(boxes, classes, scores, nms_threshold, use_iol=True, use_diou=False, 
              confidence=0.5, is_soft=False, use_exp=False):
    """Backward compatibility function for NMS."""
    if is_soft:
        nms = SoftNMS()
    elif use_diou:
        nms = DIoUNMS(use_iol=use_iol)
    else:
        nms = StandardNMS(use_iol=use_iol)
    
    return nms.apply_nms(boxes, classes, scores, nms_threshold, confidence)


def fast_cluster_nms_boxes(boxes, classes, scores, nms_threshold, use_iol=True, confidence=0.5):
    """Backward compatibility function for ClusterNMS."""
    nms = ClusterNMS(use_iol=use_iol)
    return nms.apply_nms(boxes, classes, scores, nms_threshold, confidence)
