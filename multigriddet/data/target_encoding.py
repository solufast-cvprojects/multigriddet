#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet target encoding utilities.
Implements the MultiGrid dense prediction strategy with IoL anchor matching.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class MultiGridConfig:
    """Configuration for MultiGridDet target encoding."""
    input_shape: Tuple[int, int] = (608, 608)
    num_classes: int = 80
    anchors: Optional[List[np.ndarray]] = None
    num_layers: int = 3
    grid_assignment: str = "3x3"  # MultiGridDet innovation: 3x3 grid assignment
    iou_threshold: float = 0.2
    multi_anchor_assign: bool = False
    max_boxes: int = 100


class MultiGridTargetEncoder:
    """MultiGridDet target encoder with IoL anchor matching and dense grid assignment."""
    
    def __init__(self, config: MultiGridConfig):
        self.config = config
        self._setup_anchors()
        self._setup_grid_shapes()
    
    def _setup_anchors(self):
        """Setup anchors for MultiGridDet."""
        if self.config.anchors is None:
            # Default anchors for COCO dataset (from original implementation)
            self.config.anchors = [
                np.array([[10, 13], [16, 30], [33, 23]]),  # Scale 1 (32x downsampling)
                np.array([[30, 61], [62, 45], [59, 119]]), # Scale 2 (16x downsampling)
                np.array([[116, 90], [156, 198], [373, 326]]) # Scale 3 (8x downsampling)
            ]
        
        self.anchors = self.config.anchors
        self.num_layers = len(self.anchors)
    
    def _setup_grid_shapes(self):
        """Setup grid shapes for each detection layer."""
        self.grid_shapes = [
            (self.config.input_shape[0] // {0: 32, 1: 16, 2: 8, 3: 4, 4: 2}[l],
             self.config.input_shape[1] // {0: 32, 1: 16, 2: 8, 3: 4, 4: 2}[l])
            for l in range(self.num_layers)
        ]
    
    def calculate_iol(self, anchors: np.ndarray, obj_boxes_wh: np.ndarray) -> np.ndarray:
        """
        Calculate IoL (Intersection over Largest) between anchors and object boxes.
        
        This is the key innovation in MultiGridDet - using IoL instead of IoU
        for better handling of objects with extreme aspect ratios.
        
        Args:
            anchors: Anchor boxes of shape (M, 2)
            obj_boxes_wh: Object boxes width-height of shape (N, 2)
        
        Returns:
            IoL scores of shape (N, M)
        """
        # Expand dimensions for broadcasting
        obj_boxes_expanded = np.expand_dims(obj_boxes_wh, axis=-2)  # (N, 1, 2)
        anchors_expanded = np.expand_dims(anchors, axis=0)  # (1, M, 2)
        
        # Calculate intersection
        intersection_wh = np.minimum(obj_boxes_expanded, anchors_expanded)
        
        # Calculate areas
        obj_areas = obj_boxes_wh[..., 0] * obj_boxes_wh[..., 1]  # (N,)
        anchor_areas = anchors[:, 0] * anchors[:, 1]  # (M,)
        
        # Calculate intersection areas
        intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1]
        
        # Calculate largest areas
        obj_areas_expanded = np.expand_dims(obj_areas, axis=-1)  # (N, 1)
        anchor_areas_expanded = np.expand_dims(anchor_areas, axis=0)  # (1, M)
        largest_areas = np.maximum(obj_areas_expanded, anchor_areas_expanded)
        
        # Calculate IoL
        iols = intersection_areas / (largest_areas + 1e-8)
        
        return iols
    
    def find_best_anchor(self, box_wh: np.ndarray) -> Tuple[int, int, float]:
        """
        Find the best anchor for a given box using IoL.
        
        Args:
            box_wh: Box width and height (2,)
            
        Returns:
            (layer_idx, anchor_idx, iol_score)
        """
        best_iol = 0
        best_layer = 0
        best_anchor = 0
        
        for layer_idx, anchors in enumerate(self.anchors):
            iols = self.calculate_iol(anchors, box_wh.reshape(1, 2))
            max_iol_idx = np.argmax(iols[0])
            max_iol = iols[0, max_iol_idx]
            
            if max_iol > best_iol:
                best_iol = max_iol
                best_layer = layer_idx
                best_anchor = max_iol_idx
        
        return best_layer, best_anchor, best_iol
    
    def encode_targets(self, boxes: np.ndarray) -> List[np.ndarray]:
        """
        Encode bounding boxes into MultiGridDet targets.
        
        This implements the core MultiGridDet innovation:
        - Each object is assigned to a 3x3 grid area around its center
        - Uses IoL-based anchor matching instead of IoU
        - Creates dense prediction targets
        
        Args:
            boxes: Bounding boxes in format (x_min, y_min, x_max, y_max, class_id)
            
        Returns:
            List of target tensors for each detection layer
        """
        batch_size = 1  # For single image
        height, width = self.config.input_shape
        
        # Initialize target arrays
        y_true = []
        for layer_idx in range(self.num_layers):
            grid_h, grid_w = self.grid_shapes[layer_idx]
            num_anchors = len(self.anchors[layer_idx])
            
            target = np.zeros((
                batch_size,
                grid_h, 
                grid_w, 
                5 + num_anchors + self.config.num_classes
            ), dtype=np.float32)
            y_true.append(target)
        
        if len(boxes) == 0:
            return y_true
        
        # Convert boxes to center format
        boxes_xy = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
        boxes_wh = boxes[:, 2:4] - boxes[:, :2]
        
        # Normalize to image coordinates
        boxes_xy = boxes_xy / np.array([width, height])
        boxes_wh = boxes_wh / np.array([width, height])
        
        for box_idx, (box_xy, box_wh, class_id) in enumerate(zip(boxes_xy, boxes_wh, boxes[:, 4])):
            # Skip invalid boxes
            if box_wh[0] <= 0 or box_wh[1] <= 0:
                continue
            
            # Find best anchor using IoL
            sel_layer, k, iol_score = self.find_best_anchor(box_wh)
            
            # Skip if IoL is below threshold
            if iol_score < self.config.iou_threshold:
                continue
            
            # Calculate grid coordinates
            grid_h, grid_w = self.grid_shapes[sel_layer]
            cx = box_xy[0] * grid_w
            cy = box_xy[1] * grid_h
            
            # Grid cell indices
            i = int(cx)
            j = int(cy)
            
            # Relative coordinates within grid cell
            tx = cx - i
            ty = cy - j
            
            # Box size relative to anchor
            anchor = self.anchors[sel_layer][k]
            tw = np.log(max(box_wh[0] / anchor[0], 1e-3))
            th = np.log(max(box_wh[1] / anchor[1], 1e-3))
            
            # MultiGridDet innovation: Assign to 3x3 grid area
            count_grid_cell = 0
            assigned_grid_cells = []
            
            for ki in range(-1, 2):  # 3x3 grid area
                for kj in range(-1, 2):
                    kii = i + ki
                    kjj = j + kj
                    
                    # Check bounds
                    if 0 <= kii < grid_w and 0 <= kjj < grid_h:
                        # Assign target values
                        y_true[sel_layer][0, kjj, kii, 0:4] = [-ki + tx, -kj + ty, tw, th]
                        y_true[sel_layer][0, kjj, kii, 4] = 1.0  # objectness
                        y_true[sel_layer][0, kjj, kii, 5 + k] = 1.0  # anchor-specific objectness
                        y_true[sel_layer][0, kjj, kii, 5 + len(self.anchors[sel_layer]) + int(class_id)] = 1.0  # class
                        
                        count_grid_cell += 1
                        assigned_grid_cells.append((kjj, kii))
            
            # Ensure minimum number of grid cells are assigned (MultiGridDet requirement)
            if count_grid_cell < 3:
                # If less than 3 cells assigned, assign to additional neighboring cells
                additional_cells = []
                for ki in range(-2, 3):
                    for kj in range(-2, 3):
                        kii = i + ki
                        kjj = j + kj
                        if (0 <= kii < grid_w and 0 <= kjj < grid_h and 
                            (kjj, kii) not in assigned_grid_cells):
                            additional_cells.append((kjj, kii))
                
                # Assign to additional cells to reach minimum of 3
                for kjj, kii in additional_cells[:max(0, 3 - count_grid_cell)]:
                    y_true[sel_layer][0, kjj, kii, 0:4] = [-ki + tx, -kj + ty, tw, th]
                    y_true[sel_layer][0, kjj, kii, 4] = 1.0
                    y_true[sel_layer][0, kjj, kii, 5 + k] = 1.0
                    y_true[sel_layer][0, kjj, kii, 5 + len(self.anchors[sel_layer]) + int(class_id)] = 1.0
        
        return y_true
    
    def encode_batch_targets(self, batch_boxes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Encode a batch of bounding boxes into MultiGridDet targets.
        
        Args:
            batch_boxes: List of bounding boxes for each image in batch
            
        Returns:
            List of target tensors for each detection layer
        """
        batch_size = len(batch_boxes)
        height, width = self.config.input_shape
        
        # Initialize target arrays
        y_true = []
        for layer_idx in range(self.num_layers):
            grid_h, grid_w = self.grid_shapes[layer_idx]
            num_anchors = len(self.anchors[layer_idx])
            
            target = np.zeros((
                batch_size,
                grid_h, 
                grid_w, 
                5 + num_anchors + self.config.num_classes
            ), dtype=np.float32)
            y_true.append(target)
        
        # Process each image in the batch
        for b, boxes in enumerate(batch_boxes):
            if len(boxes) == 0:
                continue
            
            # Convert boxes to center format
            boxes_xy = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
            boxes_wh = boxes[:, 2:4] - boxes[:, :2]
            
            # Normalize to image coordinates
            boxes_xy = boxes_xy / np.array([width, height])
            boxes_wh = boxes_wh / np.array([width, height])
            
            for box_idx, (box_xy, box_wh, class_id) in enumerate(zip(boxes_xy, boxes_wh, boxes[:, 4])):
                # Skip invalid boxes
                if box_wh[0] <= 0 or box_wh[1] <= 0:
                    continue
                
                # Find best anchor using IoL
                sel_layer, k, iol_score = self.find_best_anchor(box_wh)
                
                # Skip if IoL is below threshold
                if iol_score < self.config.iou_threshold:
                    continue
                
                # Calculate grid coordinates
                grid_h, grid_w = self.grid_shapes[sel_layer]
                cx = box_xy[0] * grid_w
                cy = box_xy[1] * grid_h
                
                # Grid cell indices
                i = int(cx)
                j = int(cy)
                
                # Relative coordinates within grid cell
                tx = cx - i
                ty = cy - j
                
                # Box size relative to anchor
                anchor = self.anchors[sel_layer][k]
                tw = np.log(max(box_wh[0] / anchor[0], 1e-3))
                th = np.log(max(box_wh[1] / anchor[1], 1e-3))
                
                # MultiGridDet innovation: Assign to 3x3 grid area
                count_grid_cell = 0
                assigned_grid_cells = []
                
                for ki in range(-1, 2):  # 3x3 grid area
                    for kj in range(-1, 2):
                        kii = i + ki
                        kjj = j + kj
                        
                        # Check bounds
                        if 0 <= kii < grid_w and 0 <= kjj < grid_h:
                            # Assign target values
                            y_true[sel_layer][b, kjj, kii, 0:4] = [-ki + tx, -kj + ty, tw, th]
                            y_true[sel_layer][b, kjj, kii, 4] = 1.0  # objectness
                            y_true[sel_layer][b, kjj, kii, 5 + k] = 1.0  # anchor-specific objectness
                            y_true[sel_layer][b, kjj, kii, 5 + len(self.anchors[sel_layer]) + int(class_id)] = 1.0  # class
                            
                            count_grid_cell += 1
                            assigned_grid_cells.append((kjj, kii))
                
                # Ensure minimum number of grid cells are assigned
                if count_grid_cell < 3:
                    # Assign to additional neighboring cells
                    additional_cells = []
                    for ki in range(-2, 3):
                        for kj in range(-2, 3):
                            kii = i + ki
                            kjj = j + kj
                            if (0 <= kii < grid_w and 0 <= kjj < grid_h and 
                                (kjj, kii) not in assigned_grid_cells):
                                additional_cells.append((kjj, kii))
                    
                    # Assign to additional cells to reach minimum of 3
                    for kjj, kii in additional_cells[:max(0, 3 - count_grid_cell)]:
                        y_true[sel_layer][b, kjj, kii, 0:4] = [-ki + tx, -kj + ty, tw, th]
                        y_true[sel_layer][b, kjj, kii, 4] = 1.0
                        y_true[sel_layer][b, kjj, kii, 5 + k] = 1.0
                        y_true[sel_layer][b, kjj, kii, 5 + len(self.anchors[sel_layer]) + int(class_id)] = 1.0
        
        return y_true


# Backward compatibility functions
def preprocess_true_boxes(true_boxes: np.ndarray, 
                         input_shape: Tuple[int, int],
                         anchors: List[np.ndarray],
                         num_classes: int,
                         multi_anchor_assign: bool = False,
                         iou_threshold: float = 0.2) -> List[np.ndarray]:
    """
    Backward compatibility function for preprocess_true_boxes.
    
    This function maintains compatibility with the old implementation code
    while using the new MultiGridDet target encoding.
    """
    config = MultiGridConfig(
        input_shape=input_shape,
        num_classes=num_classes,
        anchors=anchors,
        multi_anchor_assign=multi_anchor_assign,
        iou_threshold=iou_threshold
    )
    
    encoder = MultiGridTargetEncoder(config)
    
    # Convert batch format to list format
    batch_boxes = []
    for b in range(true_boxes.shape[0]):
        boxes = true_boxes[b]
        # Filter out zero boxes
        valid_boxes = boxes[np.any(boxes != 0, axis=1)]
        batch_boxes.append(valid_boxes)
    
    return encoder.encode_batch_targets(batch_boxes)
