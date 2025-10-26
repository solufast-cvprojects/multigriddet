#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for MultiGridDet.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Union, Dict, Any
import colorsys
import random


class VisualizationUtils:
    """Utility class for visualization operations."""
    
    @staticmethod
    def create_color_palette(num_classes: int, seed: int = 42) -> List[Tuple[int, int, int]]:
        """
        Create a color palette for different classes.
        
        Args:
            num_classes: Number of classes
            seed: Random seed for reproducible colors
            
        Returns:
            List of RGB color tuples
        """
        random.seed(seed)
        colors = []
        
        for i in range(num_classes):
            # Generate distinct colors using HSV color space
            hue = i / num_classes
            saturation = 0.7 + random.random() * 0.3  # 0.7 to 1.0
            value = 0.8 + random.random() * 0.2  # 0.8 to 1.0
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Convert to 0-255 range
            color = tuple(int(c * 255) for c in rgb)
            colors.append(color)
        
        return colors
    
    @staticmethod
    def draw_boxes(image: np.ndarray, 
                   boxes: np.ndarray, 
                   classes: np.ndarray, 
                   scores: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   colors: Optional[List[Tuple[int, int, int]]] = None,
                   thickness: int = 2,
                   font_scale: float = 0.5,
                   font_thickness: int = 1) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            boxes: Bounding boxes in format [x, y, w, h]
            classes: Class labels
            scores: Confidence scores
            class_names: List of class names
            colors: List of colors for each class
            thickness: Box line thickness
            font_scale: Font scale for labels
            font_thickness: Font thickness for labels
            
        Returns:
            Image with drawn boxes
        """
        if len(boxes) == 0:
            return image.copy()
        
        # Create color palette if not provided
        if colors is None:
            num_classes = len(np.unique(classes)) if len(classes) > 0 else 1
            colors = VisualizationUtils.create_color_palette(num_classes)
        
        # Convert image to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR format from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.copy()
        
        # Draw boxes
        for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
            x, y, w, h = box.astype(int)
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, thickness)
            
            # Prepare label
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {score:.2f}"
            else:
                label = f"Class {class_id}: {score:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                image_rgb, 
                (x, y - text_height - baseline), 
                (x + text_width, y), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                image_rgb, 
                label, 
                (x, y - baseline), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                font_thickness
            )
        
        return image_rgb
    
    @staticmethod
    def visualize_predictions(image: np.ndarray,
                            boxes: np.ndarray,
                            classes: np.ndarray,
                            scores: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            title: str = "MultiGridDet Predictions",
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> None:
        """
        Visualize predictions using matplotlib.
        
        Args:
            image: Input image
            boxes: Bounding boxes
            classes: Class labels
            scores: Confidence scores
            class_names: List of class names
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        # Create color palette
        num_classes = len(np.unique(classes)) if len(classes) > 0 else 1
        colors = VisualizationUtils.create_color_palette(num_classes)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
        
        # Draw boxes
        for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
            x, y, w, h = box
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            color_normalized = [c / 255.0 for c in color]
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=2, 
                edgecolor=color_normalized, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Prepare label
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {score:.2f}"
            else:
                label = f"Class {class_id}: {score:.2f}"
            
            # Add label
            ax.text(
                x, y - 5, label, 
                fontsize=10, 
                color=color_normalized,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def create_detection_grid(image: np.ndarray,
                            grid_size: Tuple[int, int],
                            boxes: np.ndarray,
                            classes: np.ndarray,
                            scores: np.ndarray,
                            class_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Create a grid visualization showing detections at different scales.
        
        Args:
            image: Input image
            grid_size: Grid size (rows, cols)
            boxes: Bounding boxes
            classes: Class labels
            scores: Confidence scores
            class_names: List of class names
            
        Returns:
            Grid image
        """
        rows, cols = grid_size
        h, w = image.shape[:2]
        
        # Create grid
        grid_h = h // rows
        grid_w = w // cols
        
        # Create color palette
        num_classes = len(np.unique(classes)) if len(classes) > 0 else 1
        colors = VisualizationUtils.create_color_palette(num_classes)
        
        # Create grid image
        grid_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw grid lines
        for i in range(rows + 1):
            y = i * grid_h
            cv2.line(grid_image, (0, y), (w, y), (128, 128, 128), 1)
        
        for j in range(cols + 1):
            x = j * grid_w
            cv2.line(grid_image, (x, 0), (x, h), (128, 128, 128), 1)
        
        # Draw detections
        for box, class_id, score in zip(boxes, classes, scores):
            x, y, w_box, h_box = box.astype(int)
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(grid_image, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Draw grid cells that contain this detection
            grid_x = x // grid_w
            grid_y = y // grid_h
            
            # Highlight grid cells
            for dy in range(max(0, grid_y - 1), min(rows, grid_y + 2)):
                for dx in range(max(0, grid_x - 1), min(cols, grid_x + 2)):
                    cell_x = dx * grid_w
                    cell_y = dy * grid_h
                    cv2.rectangle(
                        grid_image, 
                        (cell_x, cell_y), 
                        (cell_x + grid_w, cell_y + grid_h), 
                        color, 
                        1
                    )
        
        return grid_image
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]],
                            metrics: List[str] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            figsize: Figure size
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = list(history.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Plot up to 4 metrics
            if metric in history:
                ax = axes[i]
                ax.plot(history[metric])
                ax.set_title(metric)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.grid(True)
        
        # Hide unused subplots
        for i in range(len(metrics), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def create_anchor_visualization(anchors: List[np.ndarray],
                                  image_size: Tuple[int, int] = (416, 416),
                                  figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize anchor boxes.
        
        Args:
            anchors: List of anchor arrays
            image_size: Image size for visualization
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, len(anchors), figsize=figsize)
        if len(anchors) == 1:
            axes = [axes]
        
        for scale_idx, scale_anchors in enumerate(anchors):
            ax = axes[scale_idx]
            
            # Create a grid
            grid_size = 32 // (2 ** scale_idx)  # 32, 16, 8 for scales 0, 1, 2
            cell_size = image_size[0] // grid_size
            
            # Draw grid
            for i in range(grid_size + 1):
                y = i * cell_size
                ax.axhline(y=y, color='gray', linewidth=0.5)
            for j in range(grid_size + 1):
                x = j * cell_size
                ax.axvline(x=x, color='gray', linewidth=0.5)
            
            # Draw anchor boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(scale_anchors)))
            
            for anchor_idx, anchor in enumerate(scale_anchors):
                w, h = anchor
                
                # Draw anchor at center of grid
                center_x = image_size[0] // 2
                center_y = image_size[1] // 2
                
                rect = patches.Rectangle(
                    (center_x - w/2, center_y - h/2), 
                    w, h, 
                    linewidth=2, 
                    edgecolor=colors[anchor_idx], 
                    facecolor='none',
                    label=f'Anchor {anchor_idx}: {w}x{h}'
                )
                ax.add_patch(rect)
            
            ax.set_xlim(0, image_size[0])
            ax.set_ylim(0, image_size[1])
            ax.set_aspect('equal')
            ax.set_title(f'Scale {scale_idx} Anchors')
            ax.legend()
            ax.invert_yaxis()  # Invert y-axis to match image coordinates
        
        plt.tight_layout()
        plt.show()


# Convenience functions
def draw_boxes(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, 
               scores: np.ndarray, **kwargs) -> np.ndarray:
    """Convenience function for drawing boxes."""
    return VisualizationUtils.draw_boxes(image, boxes, classes, scores, **kwargs)


def visualize_predictions(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, 
                         scores: np.ndarray, **kwargs) -> None:
    """Convenience function for visualizing predictions."""
    return VisualizationUtils.visualize_predictions(image, boxes, classes, scores, **kwargs)


def create_color_palette(num_classes: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """Convenience function for creating color palette."""
    return VisualizationUtils.create_color_palette(num_classes, seed)


def get_colors(number, bright=True):
    """
    Generate random colors for drawing bounding boxes.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    if number <= 0:
        return []

    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / number, 1., brightness)
                  for x in range(number)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def draw_label(image, text, color, coords):
    """Draw a label with background on the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image


def draw_boxes(image, boxes, classes, scores, class_names, colors, show_score=True):
    """Draw bounding boxes on the image."""
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = map(int, box)

        class_name = class_names[cls]
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)
        #print(label, (xmin, ymin), (xmax, ymax))

        # if no color info, use black(0,0,0)
        if colors == None:
            color = (0,0,0)
        else:
            color = colors[cls]
        cv2.circle(image, (int(xmin+(xmax-xmin)/2), int(ymin+(ymax-ymin)/2)), radius=1, color=color, thickness=-1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, (xmin, ymin))
    return image
