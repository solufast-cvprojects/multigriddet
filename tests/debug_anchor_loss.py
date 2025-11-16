#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to analyze anchor loss computation in detail.
"""

import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.config import ConfigLoader
from multigriddet.trainers import MultiGridTrainer

def analyze_anchor_loss():
    """Analyze anchor loss computation in detail."""
    
    # Load config and setup
    config = ConfigLoader.load_config('configs/train_config.yaml')
    trainer = MultiGridTrainer(config)
    trainer.setup_data()
    trainer.build_model()
    
    # Get a batch
    batch_data, _ = trainer.train_generator[0]
    images = batch_data[0]
    y_true_list = list(batch_data[1:])
    
    # Get predictions (using fallback method)
    from multigriddet.models.multigriddet_darknet import build_multigriddet_darknet
    model_config = trainer.model_config['model']['preset']
    input_shape_full = tuple(model_config['input_shape'])
    anchors_from_gen = trainer.train_generator.anchors
    anchors_np = [a.numpy() if isinstance(a, tf.Tensor) else a for a in anchors_from_gen]
    num_anchors_per_head = [len(anchors_np[i]) for i in range(len(anchors_np))]
    num_classes_fallback = trainer.train_generator.num_classes
    
    base_model, _ = build_multigriddet_darknet(
        input_shape=input_shape_full,
        num_anchors_per_head=num_anchors_per_head,
        num_classes=num_classes_fallback,
        weights_path=None
    )
    
    y_pred_list = base_model(images, training=False)
    if not isinstance(y_pred_list, list):
        y_pred_list = [y_pred_list]
    
    print("="*80)
    print("ANCHOR LOSS ANALYSIS")
    print("="*80)
    
    # Analyze each scale
    for layer_idx in range(len(y_true_list)):
        y_true_layer = y_true_list[layer_idx]
        y_pred_layer = y_pred_list[layer_idx]
        
        # Extract components
        true_obj = y_true_layer[..., 4:5]
        true_anchors = y_true_layer[..., 5:5+num_anchors_per_head[layer_idx]]
        pred_anchors = y_pred_layer[..., 5:5+num_anchors_per_head[layer_idx]]
        
        object_mask = tf.cast(true_obj > 0.5, tf.float32)
        num_objects = tf.reduce_sum(object_mask).numpy()
        
        grid_h, grid_w = y_true_layer.shape[1], y_true_layer.shape[2]
        total_cells = grid_h * grid_w * y_true_layer.shape[0]
        num_negative_cells = total_cells - num_objects
        
        print(f"\nScale {layer_idx}:")
        print(f"  Grid shape: {grid_h} x {grid_w}")
        print(f"  Total cells: {total_cells}")
        print(f"  Object cells: {num_objects}")
        print(f"  Negative cells: {num_negative_cells}")
        print(f"  Number of anchors: {num_anchors_per_head[layer_idx]}")
        
        # Compute anchor loss manually (using K.binary_crossentropy like the actual loss function)
        import tensorflow.keras.backend as K
        anchor_loss_per_cell = K.binary_crossentropy(
            true_anchors, pred_anchors, from_logits=True
        )  # [batch, grid_h, grid_w, num_anchors]
        
        # Check anchor loss values
        anchor_loss_mean = tf.reduce_mean(anchor_loss_per_cell).numpy()
        anchor_loss_max = tf.reduce_max(anchor_loss_per_cell).numpy()
        anchor_loss_min = tf.reduce_min(anchor_loss_per_cell).numpy()
        
        print(f"\n  Anchor Loss Statistics (per anchor, per cell):")
        print(f"    Mean: {anchor_loss_mean:.6f}")
        print(f"    Max: {anchor_loss_max:.6f}")
        print(f"    Min: {anchor_loss_min:.6f}")
        
        # Compute on positive cells only
        # object_mask is [batch, grid_h, grid_w, 1], anchor_loss is [batch, grid_h, grid_w, num_anchors]
        # TensorFlow should broadcast automatically, but let's ensure shapes match
        # Check shapes first
        print(f"    object_mask shape: {object_mask.shape}")
        print(f"    anchor_loss_per_cell shape: {anchor_loss_per_cell.shape}")
        
        # Broadcasting should work: [batch, grid_h, grid_w, 1] * [batch, grid_h, grid_w, num_anchors]
        anchor_loss_positive = anchor_loss_per_cell * object_mask
        anchor_loss_positive_sum = tf.reduce_sum(anchor_loss_positive).numpy()
        anchor_loss_positive_mean = anchor_loss_positive_sum / (num_objects * num_anchors_per_head[layer_idx]) if num_objects > 0 else 0
        
        print(f"\n  On Object Cells Only:")
        print(f"    Total loss: {anchor_loss_positive_sum:.6f}")
        print(f"    Mean per anchor per object: {anchor_loss_positive_mean:.6f}")
        
        # Compute on negative cells only
        negative_mask = 1.0 - object_mask  # [batch, grid_h, grid_w, 1]
        anchor_loss_negative = anchor_loss_per_cell * negative_mask  # Broadcasting: [batch, grid_h, grid_w, num_anchors] * [batch, grid_h, grid_w, 1]
        anchor_loss_negative_sum = tf.reduce_sum(anchor_loss_negative).numpy()
        anchor_loss_negative_mean = anchor_loss_negative_sum / (num_negative_cells * num_anchors_per_head[layer_idx]) if num_negative_cells > 0 else 0
        
        print(f"\n  On Negative Cells Only:")
        print(f"    Total loss: {anchor_loss_negative_sum:.6f}")
        print(f"    Mean per anchor per negative cell: {anchor_loss_negative_mean:.6f}")
        
        # Apply weights (like in the actual loss function)
        training_config = trainer.config.get('training', {})
        loss_config = training_config.get('loss', {})
        object_scale = loss_config.get('object_scale', 1.0)
        no_object_scale = loss_config.get('no_object_scale', 1.0)
        
        positive_weight = object_mask * object_scale
        negative_weight = negative_mask * no_object_scale
        combined_weight = positive_weight + negative_weight
        
        anchor_loss_weighted = anchor_loss_per_cell * combined_weight
        anchor_loss_weighted_sum = tf.reduce_sum(anchor_loss_weighted).numpy()
        
        print(f"\n  With Weights (object_scale={object_scale}, no_object_scale={no_object_scale}):")
        print(f"    Weighted total loss: {anchor_loss_weighted_sum:.6f}")
        
        # Normalization
        batch_size = y_true_layer.shape[0]
        norm_factor = float(batch_size)  # Using batch normalization
        anchor_loss_normalized = anchor_loss_weighted_sum / norm_factor
        
        print(f"\n  After Normalization (by batch_size={batch_size}):")
        print(f"    Normalized loss: {anchor_loss_normalized:.6f}")
        
        # Compare: what if we only computed on object cells?
        if num_objects > 0:
            anchor_loss_object_only = anchor_loss_positive_sum * object_scale / norm_factor
            print(f"\n  If Only Object Cells (no negative cells):")
            print(f"    Normalized loss: {anchor_loss_object_only:.6f}")
            print(f"    Ratio (current/object_only): {anchor_loss_normalized / anchor_loss_object_only:.2f}x")
        
        # Show contribution breakdown
        positive_contribution = tf.reduce_sum(anchor_loss_per_cell * positive_weight).numpy() / norm_factor
        negative_contribution = tf.reduce_sum(anchor_loss_per_cell * negative_weight).numpy() / norm_factor
        
        print(f"\n  Contribution Breakdown:")
        print(f"    From object cells: {positive_contribution:.6f} ({100*positive_contribution/anchor_loss_normalized:.1f}%)")
        print(f"    From negative cells: {negative_contribution:.6f} ({100*negative_contribution/anchor_loss_normalized:.1f}%)")
        
        # Compare to classification
        true_class = y_true_layer[..., 5+num_anchors_per_head[layer_idx]:]
        pred_class = y_pred_layer[..., 5+num_anchors_per_head[layer_idx]:]
        
        class_loss_per_cell = tf.keras.losses.binary_crossentropy(
            true_class, pred_class, from_logits=True
        )  # [batch, grid_h, grid_w, num_classes]
        
        class_loss_object_only = class_loss_per_cell * object_mask
        class_loss_sum = tf.reduce_sum(class_loss_object_only).numpy()
        class_loss_normalized = class_loss_sum / norm_factor
        
        print(f"\n  Comparison with Classification Loss:")
        print(f"    Classification loss (object cells only): {class_loss_normalized:.6f}")
        print(f"    Anchor loss (all cells): {anchor_loss_normalized:.6f}")
        print(f"    Ratio (anchor/classification): {anchor_loss_normalized / class_loss_normalized:.2f}x")
        print(f"    Note: Classification has {num_classes_fallback} classes, Anchor has {num_anchors_per_head[layer_idx]} anchors")

if __name__ == '__main__':
    analyze_anchor_loss()

