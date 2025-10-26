#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loader for MultiGridDet.
Handles YAML configuration loading and validation.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigLoader:
    """Load and validate YAML configurations."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load YAML config file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def load_model_config(model_config_path: str) -> Dict[str, Any]:
        """Load model configuration."""
        return ConfigLoader.load_config(model_config_path)
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """Merge configurations with override priority."""
        def deep_merge(dict1, dict2):
            """Deep merge two dictionaries."""
            result = copy.deepcopy(dict1)
            
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        return deep_merge(base_config, override_config)
    
    @staticmethod
    def validate_config(config: Dict[str, Any], config_type: str = "general") -> bool:
        """Validate configuration structure."""
        if config_type == "model":
            required_keys = ["model"]
            if "model" in config:
                model_config = config["model"]
                if model_config.get("type") == "preset":
                    required_keys.extend(["preset"])
                elif model_config.get("type") == "custom":
                    required_keys.extend(["custom"])
        
        elif config_type == "training":
            required_keys = ["model_config", "data", "training"]
            if "training" in config:
                training_config = config["training"]
                if "loss_option" in training_config:
                    loss_option = training_config["loss_option"]
                    if loss_option not in [1, 2, 3]:
                        raise ValueError(f"Invalid loss_option: {loss_option}. Must be 1, 2, or 3.")
        
        elif config_type == "inference":
            required_keys = ["model_config", "input", "detection"]
        
        elif config_type == "evaluation":
            required_keys = ["model_config", "data", "evaluation"]
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key '{key}' in {config_type} config")
        
        return True
    
    @staticmethod
    def resolve_paths(config: Dict[str, Any], base_dir: str = ".") -> Dict[str, Any]:
        """Resolve relative paths in configuration."""
        base_path = Path(base_dir).resolve()
        
        def resolve_path(value, key_path=""):
            if isinstance(value, str) and (value.endswith('.yaml') or value.endswith('.txt') or value.endswith('.h5')):
                # Check if it's a relative path
                if not os.path.isabs(value):
                    return str(base_path / value)
            elif isinstance(value, dict):
                return {k: resolve_path(v, f"{key_path}.{k}" if key_path else k) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_path(item, f"{key_path}[{i}]") for i, item in enumerate(value)]
            return value
        
        return resolve_path(config)
    
    @staticmethod
    def load_and_validate(config_path: str, config_type: str = "general") -> Dict[str, Any]:
        """Load, resolve paths, and validate configuration."""
        config = ConfigLoader.load_config(config_path)
        config = ConfigLoader.resolve_paths(config, os.path.dirname(config_path))
        ConfigLoader.validate_config(config, config_type)
        return config





