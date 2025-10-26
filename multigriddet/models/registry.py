#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model registry system for MultiGridDet.
Provides decorator-based registration for easy extension of models.
"""

import functools
from typing import Dict, Any, Callable, Optional, List
import inspect


class ModelRegistry:
    """Registry for MultiGridDet model components."""
    
    def __init__(self):
        self._backbones = {}
        self._necks = {}
        self._heads = {}
        self._models = {}
    
    def register_backbone(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Decorator to register a backbone.
        
        Args:
            name: Name of the backbone
            config: Default configuration for the backbone
        """
        def decorator(cls):
            self._backbones[name] = {
                'class': cls,
                'config': config or {}
            }
            return cls
        return decorator
    
    def register_neck(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Decorator to register a neck.
        
        Args:
            name: Name of the neck
            config: Default configuration for the neck
        """
        def decorator(cls):
            self._necks[name] = {
                'class': cls,
                'config': config or {}
            }
            return cls
        return decorator
    
    def register_head(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Decorator to register a head.
        
        Args:
            name: Name of the head
            config: Default configuration for the head
        """
        def decorator(cls):
            self._heads[name] = {
                'class': cls,
                'config': config or {}
            }
            return cls
        return decorator
    
    def register_model(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Decorator to register a complete model.
        
        Args:
            name: Name of the model
            config: Default configuration for the model
        """
        def decorator(cls):
            self._models[name] = {
                'class': cls,
                'config': config or {}
            }
            return cls
        return decorator
    
    def get_backbone(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Get a backbone by name.
        
        Args:
            name: Name of the backbone
            config: Configuration to override defaults
            
        Returns:
            Backbone instance
        """
        if name not in self._backbones:
            raise ValueError(f"Backbone '{name}' not found. Available: {list(self._backbones.keys())}")
        
        backbone_info = self._backbones[name]
        backbone_config = backbone_info['config'].copy()
        if config:
            backbone_config.update(config)
        
        return backbone_info['class'](backbone_config)
    
    def get_neck(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Get a neck by name.
        
        Args:
            name: Name of the neck
            config: Configuration to override defaults
            
        Returns:
            Neck instance
        """
        if name not in self._necks:
            raise ValueError(f"Neck '{name}' not found. Available: {list(self._necks.keys())}")
        
        neck_info = self._necks[name]
        neck_config = neck_info['config'].copy()
        if config:
            neck_config.update(config)
        
        return neck_info['class'](neck_config)
    
    def get_head(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Get a head by name.
        
        Args:
            name: Name of the head
            config: Configuration to override defaults
            
        Returns:
            Head instance
        """
        if name not in self._heads:
            raise ValueError(f"Head '{name}' not found. Available: {list(self._heads.keys())}")
        
        head_info = self._heads[name]
        head_config = head_info['config'].copy()
        if config:
            head_config.update(head_config)
        
        return head_info['class'](head_config)
    
    def get_model(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Get a complete model by name.
        
        Args:
            name: Name of the model
            config: Configuration to override defaults
            
        Returns:
            Model instance
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        
        model_info = self._models[name]
        model_config = model_info['config'].copy()
        if config:
            model_config.update(config)
        
        return model_info['class'](model_config)
    
    def list_backbones(self) -> List[str]:
        """List all registered backbones."""
        return list(self._backbones.keys())
    
    def list_necks(self) -> List[str]:
        """List all registered necks."""
        return list(self._necks.keys())
    
    def list_heads(self) -> List[str]:
        """List all registered heads."""
        return list(self._heads.keys())
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def get_backbone_info(self, name: str) -> Dict[str, Any]:
        """Get information about a backbone."""
        if name not in self._backbones:
            raise ValueError(f"Backbone '{name}' not found")
        
        backbone_info = self._backbones[name]
        return {
            'name': name,
            'class': backbone_info['class'].__name__,
            'config': backbone_info['config'],
            'docstring': backbone_info['class'].__doc__
        }
    
    def get_neck_info(self, name: str) -> Dict[str, Any]:
        """Get information about a neck."""
        if name not in self._necks:
            raise ValueError(f"Neck '{name}' not found")
        
        neck_info = self._necks[name]
        return {
            'name': name,
            'class': neck_info['class'].__name__,
            'config': neck_info['config'],
            'docstring': neck_info['class'].__doc__
        }
    
    def get_head_info(self, name: str) -> Dict[str, Any]:
        """Get information about a head."""
        if name not in self._heads:
            raise ValueError(f"Head '{name}' not found")
        
        head_info = self._heads[name]
        return {
            'name': name,
            'class': head_info['class'].__name__,
            'config': head_info['config'],
            'docstring': head_info['class'].__doc__
        }
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        
        model_info = self._models[name]
        return {
            'name': name,
            'class': model_info['class'].__name__,
            'config': model_info['config'],
            'docstring': model_info['class'].__doc__
        }


# Global registry instance
_registry = ModelRegistry()

# Model registration will be done in a separate initialization file to avoid circular imports


# Convenience functions
def register_backbone(name: str, config: Optional[Dict[str, Any]] = None):
    """Register a backbone."""
    return _registry.register_backbone(name, config)


def register_neck(name: str, config: Optional[Dict[str, Any]] = None):
    """Register a neck."""
    return _registry.register_neck(name, config)


def register_head(name: str, config: Optional[Dict[str, Any]] = None):
    """Register a head."""
    return _registry.register_head(name, config)


def register_model(name: str, config: Optional[Dict[str, Any]] = None):
    """Register a complete model."""
    return _registry.register_model(name, config)


def create_model(model_name: str, 
                backbone_name: Optional[str] = None,
                neck_name: Optional[str] = None,
                head_name: Optional[str] = None,
                config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create a model using the registry.
    
    Args:
        model_name: Name of the model (if registered as complete model)
        backbone_name: Name of the backbone (if building from components)
        neck_name: Name of the neck (optional)
        head_name: Name of the head
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    if model_name in _registry.list_models():
        # Create complete model
        return _registry.get_model(model_name, config)
    else:
        # Build model from components
        if not backbone_name or not head_name:
            raise ValueError("Either model_name must be registered or backbone_name and head_name must be provided")
        
        # Create components
        backbone = _registry.get_backbone(backbone_name, config.get('backbone', {}) if config else {})
        neck = _registry.get_neck(neck_name, config.get('neck', {}) if config and neck_name else None)
        head = _registry.get_head(head_name, config.get('head', {}) if config else {})
        
        # Create model
        from .base import MultiGridDetModel
        model = MultiGridDetModel(config or {})
        model.set_backbone(backbone)
        if neck:
            model.set_neck(neck)
        model.set_head(head)
        
        return model


def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models and components.
    
    Returns:
        Dictionary with lists of available models, backbones, necks, and heads
    """
    return {
        'models': _registry.list_models(),
        'backbones': _registry.list_backbones(),
        'necks': _registry.list_necks(),
        'heads': _registry.list_heads()
    }


def get_registry() -> ModelRegistry:
    """Get the global registry instance."""
    return _registry
