#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base backbone class for MultiGridDet.
"""

from ..base import BaseBackbone


class BaseBackbone(BaseBackbone):
    """Base class for MultiGridDet backbones."""
    
    def __init__(self, config: dict):
        """
        Initialize base backbone.
        
        Args:
            config: Backbone configuration dictionary
        """
        super().__init__(config)
        self.pretrained_weights = config.get('pretrained_weights', None)
        self.freeze_layers = config.get('freeze_layers', 0)
        self.feature_layer_names = config.get('feature_layer_names', [])
    
    def load_pretrained_weights(self, weights_path: str):
        """
        Load pretrained weights.
        
        Args:
            weights_path: Path to pretrained weights file
        """
        if self.model is not None:
            self.model.load_weights(weights_path)
    
    def get_feature_layer_names(self) -> list:
        """
        Get names of feature layers.
        
        Returns:
            List of layer names that output feature maps
        """
        return self.feature_layer_names
    
    def get_feature_layer_indices(self) -> list:
        """
        Get indices of feature layers.
        
        Returns:
            List of layer indices that output feature maps
        """
        if self.model is None:
            return []
        
        indices = []
        for name in self.feature_layer_names:
            for i, layer in enumerate(self.model.layers):
                if layer.name == name:
                    indices.append(i)
                    break
        return indices
