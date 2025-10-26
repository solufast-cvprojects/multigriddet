#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base neck class for MultiGridDet.
"""

from ..base import BaseNeck


class BaseNeck(BaseNeck):
    """Base class for MultiGridDet necks (feature fusion)."""
    
    def __init__(self, config: dict):
        """
        Initialize base neck.
        
        Args:
            config: Neck configuration dictionary
        """
        super().__init__(config)
        self.num_scales = config.get('num_scales', 3)
        self.feature_channels = config.get('feature_channels', [256, 512, 1024])
        self.fusion_method = config.get('fusion_method', 'concat')  # 'concat', 'add', 'multiply'
    
    def get_fusion_method(self) -> str:
        """
        Get the fusion method used by this neck.
        
        Returns:
            Fusion method string
        """
        return self.fusion_method
    
    def get_num_scales(self) -> int:
        """
        Get the number of scales processed by this neck.
        
        Returns:
            Number of scales
        """
        return self.num_scales
    
    def get_feature_channels(self) -> list:
        """
        Get the feature channels for each scale.
        
        Returns:
            List of feature channels
        """
        return self.feature_channels
