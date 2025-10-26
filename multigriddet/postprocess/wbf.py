#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Boxes Fusion (WBF) implementation for MultiGridDet.
"""

import numpy as np
from typing import List, Tuple, Optional, Union


class WeightedBoxesFusion:
    """
    Weighted Boxes Fusion for ensemble methods.
    
    Reference: "Weighted boxes fusion: ensembling boxes from different object detection models"
    https://arxiv.org/abs/1910.13302
    """
    
    def __init__(self, 
                 iou_thr: float = 0.55,
                 skip_box_thr: float = 0.0,
                 conf_type: str = 'avg',
                 allows_overflow: bool = False):
        """
        Initialize Weighted Boxes Fusion.
        
        Args:
            iou_thr: IoU threshold for clustering
            skip_box_thr: Skip boxes with confidence below this threshold
            conf_type: Confidence calculation type ('avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg')
            allows_overflow: Whether to allow overflow in confidence calculation
        """
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.conf_type = conf_type
        self.allows_overflow = allows_overflow
    
    def fuse_boxes(self, 
                   boxes_list: List[np.ndarray],
                   classes_list: List[np.ndarray], 
                   scores_list: List[np.ndarray],
                   image_shape: Tuple[int, int],
                   weights: Optional[List[float]] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Fuse boxes from multiple models.
        
        Args:
            boxes_list: List of box arrays from different models
            classes_list: List of class arrays from different models
            scores_list: List of score arrays from different models
            image_shape: Image shape (height, width)
            weights: Optional weights for each model
            
        Returns:
            Tuple of (fused_boxes, fused_classes, fused_scores)
        """
        if len(boxes_list) == 0:
            return [], [], []
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0] * len(boxes_list)
        
        # Combine all boxes with model information
        all_boxes = []
        all_classes = []
        all_scores = []
        all_models = []
        
        for model_idx, (boxes, classes, scores) in enumerate(zip(boxes_list, classes_list, scores_list)):
            if len(boxes) == 0:
                continue
            
            # Filter by confidence threshold
            valid_mask = scores >= self.skip_box_thr
            if not np.any(valid_mask):
                continue
            
            boxes = boxes[valid_mask]
            classes = classes[valid_mask]
            scores = scores[valid_mask]
            
            all_boxes.append(boxes)
            all_classes.append(classes)
            all_scores.append(scores)
            all_models.extend([model_idx] * len(boxes))
        
        if len(all_boxes) == 0:
            return [], [], []
        
        # Concatenate all boxes
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_models = np.array(all_models)
        
        # Group by class
        unique_classes = np.unique(all_classes)
        fused_boxes = []
        fused_classes = []
        fused_scores = []
        
        for class_id in unique_classes:
            class_mask = all_classes == class_id
            class_boxes = all_boxes[class_mask]
            class_scores = all_scores[class_mask]
            class_models = all_models[class_mask]
            
            # Fuse boxes for this class
            f_boxes, f_scores = self._fuse_boxes_for_class(
                class_boxes, class_scores, class_models, weights
            )
            
            if len(f_boxes) > 0:
                fused_boxes.append(f_boxes)
                fused_classes.append(np.full(len(f_boxes), class_id))
                fused_scores.append(f_scores)
        
        if len(fused_boxes) == 0:
            return [], [], []
        
        # Concatenate results
        final_boxes = np.concatenate(fused_boxes, axis=0)
        final_classes = np.concatenate(fused_classes, axis=0)
        final_scores = np.concatenate(fused_scores, axis=0)
        
        return [final_boxes], [final_classes], [final_scores]
    
    def _fuse_boxes_for_class(self, 
                             boxes: np.ndarray, 
                             scores: np.ndarray, 
                             models: np.ndarray,
                             weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse boxes for a single class.
        
        Args:
            boxes: Boxes for this class
            scores: Scores for this class
            models: Model indices for this class
            weights: Model weights
            
        Returns:
            Tuple of (fused_boxes, fused_scores)
        """
        if len(boxes) == 0:
            return np.array([]), np.array([])
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        models = models[sorted_indices]
        
        # Create clusters
        clusters = []
        used_boxes = set()
        
        for i, box in enumerate(boxes):
            if i in used_boxes:
                continue
            
            # Create new cluster
            cluster = {
                'boxes': [box],
                'scores': [scores[i]],
                'models': [models[i]],
                'weights': [weights[models[i]]]
            }
            
            # Find boxes that belong to this cluster
            for j in range(i + 1, len(boxes)):
                if j in used_boxes:
                    continue
                
                # Check IoU
                iou = self._compute_iou(box, boxes[j])
                if iou >= self.iou_thr:
                    cluster['boxes'].append(boxes[j])
                    cluster['scores'].append(scores[j])
                    cluster['models'].append(models[j])
                    cluster['weights'].append(weights[models[j]])
                    used_boxes.add(j)
            
            clusters.append(cluster)
        
        # Fuse each cluster
        fused_boxes = []
        fused_scores = []
        
        for cluster in clusters:
            if len(cluster['boxes']) == 0:
                continue
            
            # Calculate weighted average box
            boxes_array = np.array(cluster['boxes'])
            scores_array = np.array(cluster['scores'])
            weights_array = np.array(cluster['weights'])
            
            # Weight by confidence and model weight
            total_weights = scores_array * weights_array
            total_weights = total_weights / np.sum(total_weights)
            
            # Weighted average
            fused_box = np.average(boxes_array, axis=0, weights=total_weights)
            
            # Calculate fused confidence
            fused_score = self._calculate_fused_confidence(
                scores_array, weights_array, len(boxes_list)
            )
            
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
        
        if len(fused_boxes) == 0:
            return np.array([]), np.array([])
        
        return np.array(fused_boxes), np.array(fused_scores)
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes.
        
        Args:
            box1: First box [x, y, w, h]
            box2: Second box [x, y, w, h]
            
        Returns:
            IoU value
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Compute intersection
        inter_xmin = max(x1, x2)
        inter_ymin = max(y1, y2)
        inter_xmax = min(x1 + w1, x2 + w2)
        inter_ymax = min(y1 + h1, y2 + h2)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_fused_confidence(self, 
                                   scores: np.ndarray, 
                                   weights: np.ndarray, 
                                   num_models: int) -> float:
        """
        Calculate fused confidence score.
        
        Args:
            scores: Confidence scores
            weights: Model weights
            num_models: Total number of models
            
        Returns:
            Fused confidence score
        """
        if self.conf_type == 'avg':
            return np.mean(scores)
        elif self.conf_type == 'max':
            return np.max(scores)
        elif self.conf_type == 'box_and_model_avg':
            return np.mean(scores * weights)
        elif self.conf_type == 'absent_model_aware_avg':
            # This is a simplified version
            return np.mean(scores * weights)
        else:
            return np.mean(scores)


# Backward compatibility function
def weighted_boxes_fusion(boxes_list, classes_list, scores_list, image_shape, 
                         weights=None, iou_thr=0.55, skip_box_thr=0.0, 
                         conf_type='avg', allows_overflow=False):
    """Backward compatibility function for WBF."""
    wbf = WeightedBoxesFusion(
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type=conf_type,
        allows_overflow=allows_overflow
    )
    
    return wbf.fuse_boxes(boxes_list, classes_list, scores_list, image_shape, weights)
