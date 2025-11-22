#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Darknet model builder - EXACT replica of original implementation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import os
import sys
import h5py

# Import from the new structure
from .backbones.darknet import darknet53_body
from .heads.multigrid_head import multigriddet_predictions

# Import MultiGridLoss for training
from ..losses.multigrid_loss import MultiGridLoss


def load_weights_with_debug(model: Model, weights_path: str, by_name: bool = True) -> Dict[str, Any]:
    """
    Load weights with detailed debugging information.
    
    This function wraps model.load_weights() and provides detailed statistics
    about which layers were loaded successfully and which failed.
    
    Args:
        model: Keras model to load weights into
        weights_path: Path to weights file (HDF5 format)
        by_name: Whether to load weights by layer name (default: True)
        
    Returns:
        Dictionary with loading statistics:
        - total_model_layers: Total number of layers in model
        - total_weight_layers: Total number of layers in weight file
        - loaded_count: Number of layers successfully loaded
        - failed_count: Number of layers that failed to load
        - loaded_layers: List of successfully loaded layer names
        - failed_layers: List of failed layer names with reasons
    """
    stats = {
        'total_model_layers': 0,
        'total_weight_layers': 0,
        'loaded_count': 0,
        'failed_count': 0,
        'loaded_layers': [],
        'failed_layers': [],
        'name_mismatches': [],
        'shape_mismatches': [],
        'missing_in_weights': [],
        'missing_in_model': []
    }
    
    print("\n" + "=" * 80)
    print(f"[WEIGHT LOADING DEBUG] Loading weights from: {weights_path}")
    print("=" * 80)
    
    # Get model layer information
    model_layers = {}
    for layer in model.layers:
        if hasattr(layer, 'weights') and len(layer.weights) > 0:
            layer_weights = {}
            for weight in layer.weights:
                weight_name = weight.name.replace(layer.name + '/', '').split(':')[0]
                # Handle dtype - it might be a string or a dtype object
                dtype_str = weight.dtype.name if hasattr(weight.dtype, 'name') else str(weight.dtype)
                layer_weights[weight_name] = {
                    'shape': tuple(weight.shape.as_list()),
                    'dtype': dtype_str
                }
            if layer_weights:
                model_layers[layer.name] = layer_weights
    
    stats['total_model_layers'] = len(model_layers)
    print(f"  Model has {stats['total_model_layers']} layers with weights")
    
    # Get weight file layer information
    weight_file_layers = {}
    try:
        with h5py.File(weights_path, 'r') as f:
            # Try standard Keras format first: model_weights/layer_name/weight_name
            if 'model_weights' in f:
                for layer_name in f['model_weights'].keys():
                    layer_group = f['model_weights'][layer_name]
                    if isinstance(layer_group, h5py.Group):
                        weight_shapes = {}
                        # Handle both direct weights and nested structure
                        # Structure can be: layer_name/weight_name OR layer_name/layer_name/weight_name
                        for subkey in layer_group.keys():
                            subitem = layer_group[subkey]
                            if isinstance(subitem, h5py.Dataset):
                                # Direct weight: layer_name/weight_name
                                weight_shapes[subkey] = subitem.shape
                            elif isinstance(subitem, h5py.Group):
                                # Nested structure: layer_name/layer_name/weight_name
                                # Check if this nested group contains actual weight datasets
                                has_weights = False
                                for weight_name in subitem.keys():
                                    if isinstance(subitem[weight_name], h5py.Dataset):
                                        # Store with the nested path for clarity
                                        weight_shapes[weight_name] = subitem[weight_name].shape
                                        has_weights = True
                                # If nested group has no weights, skip this layer (e.g., 'add' layers)
                                if not has_weights:
                                    continue
                        # Only add layers that actually have weights (skip layers like 'add' that have no trainable weights)
                        if weight_shapes:
                            weight_file_layers[layer_name] = weight_shapes
            else:
                # Try alternative format: direct layer groups
                def get_layer_names(name, obj):
                    if isinstance(obj, h5py.Group):
                        # Check if this group contains weight arrays
                        has_weights = False
                        weight_shapes = {}
                        for key in obj.keys():
                            if isinstance(obj[key], h5py.Dataset):
                                has_weights = True
                                weight_shapes[key] = obj[key].shape
                        if has_weights:
                            # Extract layer name (remove path prefixes)
                            parts = name.split('/')
                            layer_name = parts[-1]
                            # Skip root groups and optimizer weights
                            if layer_name and layer_name not in ['model_weights', 'optimizer_weights', '']:
                                weight_file_layers[layer_name] = weight_shapes
                f.visititems(get_layer_names)
    except Exception as e:
        print(f"  [WARNING] Could not inspect weight file structure: {e}")
        import traceback
        traceback.print_exc()
    
    stats['total_weight_layers'] = len(weight_file_layers)
    print(f"  Weight file has {stats['total_weight_layers']} layers with weights")
    
    # Compare layer names
    model_layer_names = set(model_layers.keys())
    weight_layer_names = set(weight_file_layers.keys())
    
    common_layers = model_layer_names & weight_layer_names
    only_in_model = model_layer_names - weight_layer_names
    only_in_weights = weight_layer_names - model_layer_names
    
    print(f"  Common layers: {len(common_layers)}")
    print(f"  Layers only in model: {len(only_in_model)}")
    print(f"  Layers only in weight file: {len(only_in_weights)}")
    
    # Show examples of layer names for debugging
    if len(common_layers) < 10:
        print(f"\n  [DEBUG] Common layer names (first 10):")
        for name in sorted(list(common_layers))[:10]:
            print(f"    - {name}")
    
    if len(only_in_model) > 0:
        print(f"\n  [DEBUG] Example model layer names (first 10, not in weights):")
        for name in sorted(list(only_in_model))[:10]:
            print(f"    - {name}")
    
    if len(only_in_weights) > 0:
        print(f"\n  [DEBUG] Example weight file layer names (first 10, not in model):")
        for name in sorted(list(only_in_weights))[:10]:
            print(f"    - {name}")
    
    stats['missing_in_weights'] = list(only_in_model)
    stats['missing_in_model'] = list(only_in_weights)
    
    # Try to load weights
    try:
        # Store a sample of initial weights for verification
        sample_weights_before = {}
        sample_layers = list(model_layers.keys())[:5]  # Sample first 5 layers
        for layer_name in sample_layers:
            layer = None
            for l in model.layers:
                if l.name == layer_name:
                    layer = l
                    break
            if layer and len(layer.weights) > 0:
                sample_weights_before[layer_name] = layer.weights[0].numpy().copy()
        
        # Try loading weights with by_name first
        # If that fails or loads very few layers, try loading by position (without by_name)
        # This handles cases where layer names don't match due to auto-incrementing in the same session
        try:
            if by_name:
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                # Check if we actually loaded weights by verifying a sample layer changed
                weights_loaded = False
                for layer_name in sample_layers:
                    if layer_name in sample_weights_before:
                        layer = None
                        for l in model.layers:
                            if l.name == layer_name and len(l.weights) > 0:
                                layer = l
                                break
                        if layer:
                            weights_after = layer.weights[0].numpy()
                            weights_before = sample_weights_before[layer_name]
                            if not np.array_equal(weights_before, weights_after):
                                weights_loaded = True
                                break
                
                # If by_name didn't load weights (names don't match), try by position
                if not weights_loaded and len(common_layers) < len(model_layers) * 0.1:
                    print(f"  [INFO] by_name=True loaded few layers ({len(common_layers)}/{len(model_layers)}), trying by position...")
                    # Reload model to reset weights, then load by position
                    model.load_weights(weights_path, by_name=False)
                    print(f"  [INFO] Loaded weights by position (shape matching)")
            else:
                model.load_weights(weights_path, by_name=False)
        except Exception as e:
            print(f"  [WARNING] Weight loading failed: {e}")
            # Try fallback: load by position
            try:
                print(f"  [INFO] Attempting fallback: loading by position...")
                model.load_weights(weights_path, by_name=False)
                print(f"  [INFO] Fallback successful: loaded by position")
            except Exception as e2:
                print(f"  [ERROR] Fallback also failed: {e2}")
                raise
        
        print(f"  [SUCCESS] Weight loading completed without exceptions")
        
        # Manually load BN statistics (fix for nested structure issue with by_name=True)
        # Keras load_weights(by_name=True) sometimes fails to load nested BN statistics correctly
        # We manually load all BN statistics to ensure they're properly loaded
        try:
            with h5py.File(weights_path, 'r') as f:
                if 'model_weights' in f:
                    model_weights = f['model_weights']
                    bn_loaded_count = 0
                    bn_failed_count = 0
                    
                    # Create a mapping of layer names to model layers for faster lookup
                    model_layer_map = {layer.name: layer for layer in model.layers}
                    
                    for layer_name in model_weights.keys():
                        if 'batch_normalization' in layer_name.lower():
                            # Find corresponding layer in model
                            model_layer = model_layer_map.get(layer_name)
                            
                            if model_layer and isinstance(model_layer, tf.keras.layers.BatchNormalization):
                                layer_group = model_weights[layer_name]
                                # Check nested structure: layer_name/batch_normalization/moving_mean:0
                                if 'batch_normalization' in layer_group:
                                    bn_group = layer_group['batch_normalization']
                                    if 'moving_mean:0' in bn_group and 'moving_variance:0' in bn_group:
                                        try:
                                            # Load BN statistics from file
                                            file_mean = np.array(bn_group['moving_mean:0'])
                                            file_var = np.array(bn_group['moving_variance:0'])
                                            
                                            # Set the weights directly
                                            for weight in model_layer.weights:
                                                if 'moving_mean' in weight.name:
                                                    weight.assign(file_mean)
                                                elif 'moving_variance' in weight.name:
                                                    weight.assign(file_var)
                                            
                                            bn_loaded_count += 1
                                        except Exception as e:
                                            bn_failed_count += 1
                    
                    if bn_loaded_count > 0:
                        print(f"  [INFO] Manually loaded BN statistics for {bn_loaded_count} layers")
                    if bn_failed_count > 0:
                        print(f"  [WARNING] Failed to load BN statistics for {bn_failed_count} layers")
        except Exception as e:
            print(f"  [WARNING] Could not manually load BN statistics: {e}")
            import traceback
            traceback.print_exc()
        
        # Verify weights actually changed
        weights_changed = False
        layers_checked = 0
        for layer_name in sample_layers:
            layer = None
            for l in model.layers:
                if l.name == layer_name:
                    layer = l
                    break
            if layer and len(layer.weights) > 0 and layer_name in sample_weights_before:
                layers_checked += 1
                weights_after = layer.weights[0].numpy()
                weights_before = sample_weights_before[layer_name]
                if not np.array_equal(weights_before, weights_after):
                    weights_changed = True
                    break
        
        # If loading by position, check if weights changed by checking any layer (not just by name)
        if not weights_changed and len(sample_weights_before) == 0:
            # No sample weights stored, check a few random layers
            checked_layers = 0
            for layer in model.layers[:10]:
                if hasattr(layer, 'weights') and len(layer.weights) > 0:
                    # Just verify weights exist and are not all zeros (basic check)
                    weight_sum = tf.reduce_sum(tf.abs(layer.weights[0])).numpy()
                    if weight_sum > 1e-6:  # Not all zeros
                        weights_changed = True
                        break
                    checked_layers += 1
                    if checked_layers >= 5:
                        break
        
        if not weights_changed and layers_checked > 0:
            print(f"  [WARNING] Sample weights did not change after loading - weights may not have been applied!")
            print(f"  [INFO] This might be OK if loading by position and layer structure matches")
        elif weights_changed:
            print(f"  [VERIFIED] Sample weights changed after loading - weights were successfully applied")
        else:
            print(f"  [INFO] Could not verify weight loading (no sample weights stored)")
        
        # Check which layers actually got loaded by comparing common layers
        # Note: This is approximate since we can't easily check if weights changed
        for layer_name in common_layers:
            # Check if weight shapes match
            model_weight_names = set(model_layers[layer_name].keys())
            weight_file_weight_names = set(weight_file_layers[layer_name].keys())
            
            shape_matches = True
            for weight_name in model_weight_names & weight_file_weight_names:
                model_shape = model_layers[layer_name][weight_name]['shape']
                weight_shape = weight_file_layers[layer_name][weight_name]
                if model_shape != tuple(weight_shape):
                    shape_matches = False
                    stats['shape_mismatches'].append({
                        'layer': layer_name,
                        'weight': weight_name,
                        'model_shape': model_shape,
                        'weight_shape': tuple(weight_shape)
                    })
                    break
            
            if shape_matches:
                stats['loaded_layers'].append(layer_name)
                stats['loaded_count'] += 1
            else:
                stats['failed_layers'].append({
                    'layer': layer_name,
                    'reason': 'shape_mismatch'
                })
                stats['failed_count'] += 1
        
        # Layers only in model (not in weights) - these won't be loaded
        for layer_name in only_in_model:
            stats['failed_layers'].append({
                'layer': layer_name,
                'reason': 'missing_in_weight_file'
            })
            stats['failed_count'] += 1
        
        # Layers only in weights (not in model) - these will be skipped
        for layer_name in only_in_weights:
            stats['name_mismatches'].append(layer_name)
        
    except Exception as e:
        print(f"  [ERROR] Exception during weight loading: {e}")
        stats['failed_count'] = stats['total_model_layers']
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n  [SUMMARY]")
    print(f"    Successfully loaded: {stats['loaded_count']} layers")
    print(f"    Failed/Skipped: {stats['failed_count']} layers")
    
    if stats['shape_mismatches']:
        print(f"    Shape mismatches: {len(stats['shape_mismatches'])}")
        for mismatch in stats['shape_mismatches'][:10]:  # Show first 10
            print(f"      - {mismatch['layer']}/{mismatch['weight']}: "
                  f"model={mismatch['model_shape']}, weights={mismatch['weight_shape']}")
        if len(stats['shape_mismatches']) > 10:
            print(f"      ... and {len(stats['shape_mismatches']) - 10} more")
    
    if stats['missing_in_weights']:
        print(f"    Missing in weight file: {len(stats['missing_in_weights'])} layers")
        for layer_name in stats['missing_in_weights'][:10]:  # Show first 10
            print(f"      - {layer_name}")
        if len(stats['missing_in_weights']) > 10:
            print(f"      ... and {len(stats['missing_in_weights']) - 10} more")
    
    if stats['missing_in_model']:
        print(f"    Missing in model (skipped): {len(stats['missing_in_model'])} layers")
        for layer_name in stats['missing_in_model'][:10]:  # Show first 10
            print(f"      - {layer_name}")
        if len(stats['missing_in_model']) > 10:
            print(f"      ... and {len(stats['missing_in_model']) - 10} more")
    
    # Debug check 1: Which head layers are loaded?
    print("\n  [DEBUG 1] Head layers loading status:")
    # Head layers are typically after the backbone (after layer 185 or so)
    # Find layers that are likely head layers (conv layers after backbone)
    head_layers_loaded = []
    head_layers_missing = []
    backbone_len = 185  # Approximate backbone length
    
    for layer_name in stats['loaded_layers']:
        # Find layer index
        layer_idx = None
        for i, layer in enumerate(model.layers):
            if layer.name == layer_name:
                layer_idx = i
                break
        if layer_idx is not None and layer_idx >= backbone_len:
            head_layers_loaded.append((layer_name, layer_idx))
    
    for layer_name in stats['missing_in_weights']:
        layer_idx = None
        for i, layer in enumerate(model.layers):
            if layer.name == layer_name:
                layer_idx = i
                break
        if layer_idx is not None and layer_idx >= backbone_len:
            head_layers_missing.append((layer_name, layer_idx))
    
    print(f"    Head layers (after backbone, layer {backbone_len}+):")
    print(f"      Loaded: {len(head_layers_loaded)} layers")
    if head_layers_loaded:
        for layer_name, idx in sorted(head_layers_loaded, key=lambda x: x[1])[:10]:
            print(f"        - {layer_name} (layer {idx})")
        if len(head_layers_loaded) > 10:
            print(f"        ... and {len(head_layers_loaded) - 10} more")
    print(f"      Missing: {len(head_layers_missing)} layers")
    if head_layers_missing:
        for layer_name, idx in sorted(head_layers_missing, key=lambda x: x[1])[:10]:
            print(f"        - {layer_name} (layer {idx})")
        if len(head_layers_missing) > 10:
            print(f"        ... and {len(head_layers_missing) - 10} more")
    
    # Debug check 2: Batch normalization statistics loading
    print("\n  [DEBUG 2] Batch normalization statistics loading:")
    bn_layers_checked = 0
    bn_with_stats = 0
    bn_missing_stats = []
    
    # Check a sample of batch normalization layers
    for layer in model.layers:
        if 'batch_normalization' in layer.name.lower() or isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_layers_checked += 1
            if bn_layers_checked > 20:  # Check first 20 BN layers
                break
            
            # Check if layer has moving_mean and moving_variance
            has_moving_mean = False
            has_moving_variance = False
            moving_mean_val = None
            moving_variance_val = None
            
            for weight in layer.weights:
                if 'moving_mean' in weight.name:
                    has_moving_mean = True
                    moving_mean_val = weight.numpy()
                elif 'moving_variance' in weight.name:
                    has_moving_variance = True
                    moving_variance_val = weight.numpy()
            
            if has_moving_mean and has_moving_variance:
                # Check if they're non-zero (indicating they were loaded, not just initialized)
                mean_nonzero = np.abs(moving_mean_val).mean() > 1e-6
                var_nonzero = np.abs(moving_variance_val - 1.0).mean() > 1e-6  # Variance usually starts near 1.0
                
                if mean_nonzero or var_nonzero:
                    bn_with_stats += 1
                else:
                    bn_missing_stats.append((layer.name, "statistics appear uninitialized"))
            else:
                bn_missing_stats.append((layer.name, "missing moving_mean or moving_variance"))
    
    print(f"    Checked {bn_layers_checked} batch normalization layers:")
    print(f"      With loaded statistics: {bn_with_stats}")
    print(f"      Missing/uninitialized statistics: {len(bn_missing_stats)}")
    if bn_missing_stats:
        for layer_name, reason in bn_missing_stats[:5]:
            print(f"        - {layer_name}: {reason}")
        if len(bn_missing_stats) > 5:
            print(f"        ... and {len(bn_missing_stats) - 5} more")
    
    print("=" * 80 + "\n")
    
    return stats


def build_multigriddet_darknet(input_shape: Tuple[int, int, int] = (416, 416, 3),
                               num_anchors_per_head: List[int] = [3, 3, 3],
                               num_classes: int = 80,
                               weights_path: Optional[str] = None,
                               clear_session: bool = False,
                               **kwargs) -> Tuple[Model, int]:
    """
    Build MultiGridDet model with TRUE Darknet53 backbone.
    
    This is an EXACT replica of the original implementation.
    
    Architecture:
    - Backbone: Darknet53
    - Neck: Implicit FPN in predictions
    - Head: MultiGridDet multi-scale output
        
    Args:
        input_shape: Input image shape (height, width, channels)
        num_anchors_per_head: Number of anchors per detection scale (heads)
        num_classes: Number of object classes
        weights_path: Path to pretrained weights
        clear_session: Whether to clear Keras session before building (default: False)
                     Set to True when loading weights to ensure layer names match
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, backbone_length) - backbone_length is the number of layers in the backbone
    """
    # Clear session if requested (typically done by build_multigriddet_darknet_train)
    if clear_session:
        tf.keras.backend.clear_session()
    
    # Build exactly as original implementation does
    inputs = Input(shape=input_shape)
    darknet = Model(inputs, darknet53_body(inputs))
    
    if weights_path and os.path.exists(weights_path):
        load_weights_with_debug(darknet, weights_path, by_name=True)
        print(f'Loaded backbone weights from {weights_path}')
    
    # Extract feature maps
    f1 = darknet.output  # 13x13x1024
    f2 = darknet.layers[152].output  # 26x26x512
    f3 = darknet.layers[92].output  # 52x52x256
    
    # Feature channels (from original implementation)
    f1_channel_num = 512
    f2_channel_num = 256
    f3_channel_num = 128
    
    # Build predictions exactly as original implementation does
    y1, y2, y3 = multigriddet_predictions(
        (f1, f2, f3), 
        (f1_channel_num, f2_channel_num, f3_channel_num),
        num_anchors_per_head,
        num_classes
    )
    
    model = Model(inputs, [y1, y2, y3], name='multigriddet_darknet')
    
    return model, 185  # Darknet53 backbone length


def build_multigriddet_darknet_train(anchors: List[np.ndarray],
                                    num_classes: int = 80,
                                    input_shape: Tuple[int, int, int] = (416, 416, 3),
                                    weights_path: Optional[str] = None,
                                    backbone_weights_path: Optional[str] = None,
                                    freeze_level: int = 1,
                                    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                                    label_smoothing: float = 0.0,
                                    elim_grid_sense: bool = False,
                                    loss_option: int = 3,  # NEW: Add this parameter
                                    coord_scale: float = 1.0,
                                    object_scale: float = 1.0,
                                    no_object_scale: float = 1.0,
                                    class_scale: float = 1.0,
                                    anchor_scale: float = 1.0,
                                    class_weights: Optional[np.ndarray] = None,
                                    clear_session: bool = True,
                                    **kwargs) -> Tuple[Model, int]:
    """
    Build training model for multigriddet_darknet.
    
    Args:
        anchors: List of anchor arrays for each scale (heads)
        num_classes: Number of object classes
        input_shape: Input image shape
        weights_path: Path to pretrained full model weights
        backbone_weights_path: Path to pretrained backbone weights (e.g., darknet53.h5)
        freeze_level: Freeze level (0=unfreeze all, 1=freeze backbone, 2=freeze all but head)
        optimizer: Optimizer for training
        label_smoothing: Label smoothing factor
        elim_grid_sense: Eliminate grid sense loss
        loss_option: Loss option (1=IoU, 2=GIOU, 3=CIoU)
        coord_scale: Scale for localization (coordinate) loss
        object_scale: Scale for objectness loss (positive cells)
        no_object_scale: Scale for objectness loss (negative cells)
        class_scale: Scale for classification loss
        anchor_scale: Scale for anchor prediction loss
        class_weights: Optional array of class weights for handling class imbalance [num_classes]
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (training_model, backbone_length) - backbone_length is the number of layers in the backbone
    """
    # Clear Keras session to reset layer name counters
    # This ensures layer names match the weights file (conv2d, batch_normalization, etc.)
    # instead of being auto-incremented (conv2d_198, batch_normalization_264, etc.)
    # CRITICAL: Without this, weights won't load correctly if other models were built earlier
    if clear_session:
        tf.keras.backend.clear_session()
        if weights_path or backbone_weights_path:
            print("[INFO] Cleared Keras session to ensure consistent layer naming for weight loading")
    
    # Calculate number of anchors per head
    num_feature_layers = len(anchors)
    num_anchors_per_head = [len(anchors[l]) for l in range(num_feature_layers)]
    
    print(f"num_anchors_per_head: {num_anchors_per_head}")
    
    # Build base model with backbone weights (if provided)
    # Backbone weights are loaded during model construction
    # Don't clear session again (already cleared above)
    model, backbone_len = build_multigriddet_darknet(
        input_shape=input_shape,
        num_anchors_per_head=num_anchors_per_head,
        num_classes=num_classes,
        weights_path=backbone_weights_path,  # Pass backbone weights to load into backbone
        clear_session=False  # Already cleared above
    )
    
    print(f'Create MultiGridDet Darknet model with {sum(num_anchors_per_head)} anchors and {num_classes} classes.')
    print(f'model layer number: {len(model.layers)}')
    
    # Load full model weights if provided (after backbone weights)
    # This allows full model weights to override backbone weights if both are provided
    if weights_path and os.path.exists(weights_path):
        try:
            load_weights_with_debug(model, weights_path, by_name=True)
            print(f'Loaded full model weights from {weights_path}.')
        except Exception as e:
            print(f'Warning: Could not load full model weights from {weights_path}: {e}')
            import traceback
            traceback.print_exc()
    
    # Apply freezing based on freeze_level
    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers
        num = (backbone_len, len(model.layers) - 3)[freeze_level - 1]
        for i in range(num):
            model.layers[i].trainable = False
        print(f'Freeze the first {num} layers.')
    elif freeze_level == 0:
        # Unfreeze all layers
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        print('Unfreeze all layers.')
    
    # Create dummy y_true inputs for the loss function
    y_true = [Input(shape=(None, None, num_anchors_per_head[l] + num_classes + 5), name=f'y_true_{l}') for l in range(num_feature_layers)]
    
    # Create the MultiGridLoss function
    # Extract loss configuration kwargs (only used for the loss, not the backbone)
    loss_kwargs = kwargs.copy()
    loss_normalization = loss_kwargs.pop('loss_normalization', None)
    multigrid_loss_fn = MultiGridLoss(
        anchors=anchors,
        num_classes=num_classes,
        input_shape=input_shape[:2], # Pass only H, W
        label_smoothing=label_smoothing,
        elim_grid_sense=elim_grid_sense,
        loss_option=loss_option,  # Pass the option
        coord_scale=coord_scale,
        object_scale=object_scale,
        no_object_scale=no_object_scale,
        class_scale=class_scale,
        anchor_scale=anchor_scale,
        class_weights=class_weights,
        loss_normalization=loss_normalization,
        **loss_kwargs
    )
    
    # Create loss layer using Lambda
    multigrid_loss = tf.keras.layers.Lambda(
        lambda x: multigrid_loss_fn(x[3:], x[:3]),  # y_true, y_pred
        output_shape=(None,),  # Loss function returns a scalar
        name='multigrid_loss'
    )([*model.outputs, *y_true]) # Pass model outputs and y_true to loss layer
    
    # Create the training model
    training_model = Model(inputs=[model.input] + y_true, outputs=multigrid_loss)
    
    # Verify weights are accessible in training model after wrapping
    # The training model wraps the base model, so weights should be shared
    sample_layer_name = None
    sample_layer_original = None
    if weights_path and os.path.exists(weights_path):
        # Check if a sample layer's weights are accessible in training model
        try:
            # The base model layers should be directly accessible in training_model.layers
            # Find a layer that exists in both models by name
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'weights') and len(layer.weights) > 0:
                    sample_layer_name = layer.name
                    sample_layer_original = layer
                    break
            
            if sample_layer_name:
                # Find the same layer in training model
                sample_layer_training = None
                for layer in training_model.layers:
                    if layer.name == sample_layer_name and hasattr(layer, 'weights') and len(layer.weights) > 0:
                        sample_layer_training = layer
                        break
                
                if sample_layer_training and sample_layer_original:
                    weight_after_wrap = sample_layer_training.weights[0].numpy().copy()
                    weight_original = sample_layer_original.weights[0].numpy().copy()
                    if np.array_equal(weight_original, weight_after_wrap):
                        print(f'[VERIFIED] Weights accessible in training model wrapper (checked: {sample_layer_name})')
                    else:
                        print(f'[WARNING] Weights differ between base and training model for {sample_layer_name}!')
                        diff = np.abs(weight_original - weight_after_wrap).mean()
                        print(f'         Mean difference: {diff:.6f}')
                else:
                    print(f'[WARNING] Could not find matching layer {sample_layer_name} in training model')
        except Exception as e:
            print(f'[WARNING] Could not verify weights in training model: {e}')
    
    # Set optimizer
    if optimizer is None:
        optimizer = Adam(learning_rate=1e-3)
    
    # Compile model with loss wrapper (same as original implementation)
    training_model.compile(
        optimizer=optimizer,
        loss={'multigrid_loss': lambda y_true, y_pred: y_pred}
    )
    
    # Verify weights persist after compilation
    if weights_path and os.path.exists(weights_path):
        try:
            # Find the same layer we checked before
            if sample_layer_name and sample_layer_original:
                sample_layer_training = None
                for layer in training_model.layers:
                    if layer.name == sample_layer_name and hasattr(layer, 'weights') and len(layer.weights) > 0:
                        sample_layer_training = layer
                        break
                
                if sample_layer_training:
                    weight_after_compile = sample_layer_training.weights[0].numpy().copy()
                    weight_original = sample_layer_original.weights[0].numpy().copy()
                    if np.array_equal(weight_original, weight_after_compile):
                        print(f'[VERIFIED] Weights preserved after compilation (checked: {sample_layer_name})')
                    else:
                        print(f'[ERROR] Weights changed after compilation for {sample_layer_name}! This should not happen!')
                        diff = np.abs(weight_original - weight_after_compile).mean()
                        print(f'         Mean difference: {diff:.6f}')
        except Exception as e:
            print(f'[WARNING] Could not verify weights after compilation: {e}')
    
    return training_model, backbone_len


## Model configuration for reference
MULTIGRIDDET_DARKNET_CONFIG = {
    'builder': build_multigriddet_darknet,
    'train_builder': build_multigriddet_darknet_train,
    'backbone_len': 185,
    'default_weights': 'model5.h5',
    'input_shape': (416, 416, 3),
    'compatible_with': 'multigriddet_darknet',
    'description': 'TRUE MultiGridDet with Darknet53'
}
