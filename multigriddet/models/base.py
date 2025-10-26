#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes for MultiGridDet model components.
"""

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np


class BaseModel(ABC):
    """Abstract base class for MultiGridDet models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.backbone = None
        self.neck = None
        self.head = None
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """
        Build the complete model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            
        Returns:
            Built Keras model
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        pass
    
    def summary(self, line_length: int = 100, positions: Optional[List[float]] = None):
        """Print model summary."""
        if self.model is not None:
            self.model.summary(line_length=line_length, positions=positions)
    
    def save_weights(self, filepath: str, overwrite: bool = True, save_format: Optional[str] = None):
        """Save model weights."""
        if self.model is not None:
            self.model.save_weights(filepath, overwrite=overwrite, save_format=save_format)
    
    def load_weights(self, filepath: str, by_name: bool = False, skip_mismatch: bool = False):
        """Load model weights."""
        if self.model is not None:
            self.model.load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch)


class BaseBackbone(ABC):
    """Abstract base class for backbone networks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backbone.
        
        Args:
            config: Backbone configuration dictionary
        """
        self.config = config
        self.model = None
        self.feature_layers = []
        self.output_shapes = []
    
    @abstractmethod
    def build_backbone(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """
        Build the backbone network.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            
        Returns:
            Built backbone model
        """
        pass
    
    @abstractmethod
    def get_feature_maps(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        """
        Extract feature maps from backbone.
        
        Args:
            inputs: Input tensor
            
        Returns:
            List of feature map tensors
        """
        pass
    
    @abstractmethod
    def get_output_shapes(self, input_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Get output shapes for each feature map.
        
        Args:
            input_shape: Input image shape
            
        Returns:
            List of output shapes
        """
        pass
    
    def freeze_layers(self, num_layers: int):
        """
        Freeze the first num_layers of the backbone.
        
        Args:
            num_layers: Number of layers to freeze
        """
        if self.model is not None:
            for i, layer in enumerate(self.model.layers):
                if i < num_layers:
                    layer.trainable = False
                else:
                    layer.trainable = True


class BaseNeck(ABC):
    """Abstract base class for neck networks (feature fusion)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize neck.
        
        Args:
            config: Neck configuration dictionary
        """
        self.config = config
        self.model = None
    
    @abstractmethod
    def build_neck(self, feature_maps: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Build the neck network.
        
        Args:
            feature_maps: List of feature map tensors from backbone
            
        Returns:
            List of processed feature map tensors
        """
        pass
    
    @abstractmethod
    def get_output_shapes(self, input_shapes: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Get output shapes for neck.
        
        Args:
            input_shapes: List of input feature map shapes
            
        Returns:
            List of output shapes
        """
        pass


class BaseHead(ABC):
    """Abstract base class for detection heads."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize head.
        
        Args:
            config: Head configuration dictionary
        """
        self.config = config
        self.model = None
    
    @abstractmethod
    def build_head(self, feature_maps: List[tf.Tensor], 
                   num_classes: int, 
                   anchors: List[np.ndarray]) -> List[tf.Tensor]:
        """
        Build the detection head.
        
        Args:
            feature_maps: List of feature map tensors
            num_classes: Number of object classes
            anchors: List of anchor arrays for each scale
            
        Returns:
            List of output tensors for each scale
        """
        pass
    
    @abstractmethod
    def get_output_shapes(self, 
                         input_shapes: List[Tuple[int, int, int]], 
                         num_classes: int, 
                         anchors: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Get output shapes for head.
        
        Args:
            input_shapes: List of input feature map shapes
            num_classes: Number of object classes
            anchors: List of anchor arrays
            
        Returns:
            List of output shapes (batch, height, width, channels)
        """
        pass


class MultiGridDetModel(BaseModel):
    """Concrete implementation of MultiGridDet model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MultiGridDet model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.num_classes = config.get('num_classes', 80)
        self.anchors = config.get('anchors', None)
        self.input_shape = config.get('input_shape', (608, 608, 3))
    
    def build_model(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """
        Build the complete MultiGridDet model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            
        Returns:
            Built Keras model
        """
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape, name='input')
        
        # Build backbone
        if self.backbone is None:
            raise ValueError("Backbone must be set before building model")
        
        backbone_model = self.backbone.build_backbone(input_shape)
        feature_maps = self.backbone.get_feature_maps(inputs)
        
        # Build neck (optional)
        if self.neck is not None:
            feature_maps = self.neck.build_neck(feature_maps)
        
        # Build head
        if self.head is None:
            raise ValueError("Head must be set before building model")
        
        outputs = self.head.build_head(feature_maps, self.num_classes, self.anchors)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='multigriddet')
        
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'name': 'MultiGridDet',
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'backbone': self.backbone.__class__.__name__ if self.backbone else None,
            'neck': self.neck.__class__.__name__ if self.neck else None,
            'head': self.head.__class__.__name__ if self.head else None,
            'anchors': self.anchors,
        }
        
        if self.model is not None:
            info['total_params'] = self.model.count_params()
            info['trainable_params'] = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        return info
    
    def set_backbone(self, backbone: BaseBackbone):
        """Set the backbone network."""
        self.backbone = backbone
    
    def set_neck(self, neck: BaseNeck):
        """Set the neck network."""
        self.neck = neck
    
    def set_head(self, head: BaseHead):
        """Set the detection head."""
        self.head = head
