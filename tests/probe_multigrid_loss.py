#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridLoss Component Diagnostic Tool

Probes individual loss components (localization, objectness, anchor, classification)
during training to diagnose validation loss divergence.

Usage:
    python tests/probe_multigrid_loss.py --num-batches 2
    python tests/probe_multigrid_loss.py --num-batches 5 --config configs/train_config.yaml --output-dir loss_stats/
"""

import argparse
import json
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multigriddet.config import ConfigLoader, build_model_for_training
from multigriddet.trainers import MultiGridTrainer
from multigriddet.losses.multigrid_loss import MultiGridLoss


class DiagnosticMultiGridLoss(MultiGridLoss):
    """
    Diagnostic version of MultiGridLoss that returns individual components.
    
    Extends MultiGridLoss to capture and return all loss components separately
    for diagnostic purposes.
    """
    
    def compute_loss_diagnostic(self, y_true: List[tf.Tensor], y_pred: List[tf.Tensor]) -> Dict[str, Any]:
        """
        Compute loss and return individual components for diagnostics.
        
        Returns:
            Dictionary containing:
            - total_loss: Weighted sum of all losses
            - localization_loss: Total localization loss (sum across scales)
            - objectness_loss: Total objectness loss (sum across scales)
            - anchor_loss: Total anchor loss (sum across scales)
            - classification_loss: Total classification loss (sum across scales)
            - per_scale_losses: List of dicts with per-scale breakdown
            - scaling_coefficients: Dict of scaling factors
            - normalization_factors: Dict of normalization factors per component
        """
        batch_size = K.shape(y_pred[0])[0]
        batch_size_f = K.cast(batch_size, 'float32')
        
        total_loss_location = 0
        total_loss_objectness = 0
        total_loss_classification = 0
        total_loss_anchor = 0
        
        per_scale_losses = []
        normalization_factors = {
            'localization': [],
            'objectness': [],
            'anchor': [],
            'classification': []
        }
        
        # Process each scale
        for layer_idx in range(self.num_layers):
            y_pred_layer = y_pred[layer_idx]
            y_true_layer = y_true[layer_idx]
            anchor_layer = self.anchors[layer_idx]
            num_anchors = len(anchor_layer)
            
            # Extract prediction components
            pred_xy = y_pred_layer[..., 0:2]
            pred_wh = y_pred_layer[..., 2:4]
            pred_obj = y_pred_layer[..., 4:5]
            pred_anchors = y_pred_layer[..., 5:5+num_anchors]
            pred_class = y_pred_layer[..., 5+num_anchors:]
            
            # Extract ground truth components
            true_xy = y_true_layer[..., 0:2]
            true_wh = y_true_layer[..., 2:4]
            true_obj = y_true_layer[..., 4:5]
            true_anchors = y_true_layer[..., 5:5+num_anchors]
            true_class = y_true_layer[..., 5+num_anchors:]
            
            # Object mask
            object_mask = K.cast(true_obj > 0.5, 'float32')
            
            # Get grid shape
            grid_shape = (K.shape(y_pred_layer)[1], K.shape(y_pred_layer)[2])
            
            # Compute ignore mask for all loss options
            ignore_mask = self._compute_ignore_mask(
                pred_xy, pred_wh, true_xy, true_wh,
                anchor_layer, object_mask, y_true_layer, grid_shape
            )
            
            # ========== LOCALIZATION LOSS ==========
            loc_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
            normalization_factors['localization'].append(loc_norm_factor)
            
            if self.loss_option == 1:
                loc_loss = self._compute_mse_loss(
                    true_xy, true_wh, pred_xy, pred_wh, object_mask
                )
                loc_loss = loc_loss / loc_norm_factor
            elif self.loss_option == 2:
                loc_loss = self._compute_mse_loss(
                    true_xy, true_wh, pred_xy, pred_wh, object_mask
                )
                loc_loss = loc_loss / loc_norm_factor
                
                # Anchor prediction loss (part of Option 2)
                anchor_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
                normalization_factors['anchor'].append(anchor_norm_factor)
                anchor_loc_loss = self._compute_anchor_loss(
                    true_anchors, pred_anchors, object_mask, ignore_mask, anchor_norm_factor
                )
                total_loss_anchor += self.anchor_scale * anchor_loc_loss
            else:  # Option 3
                if self.use_giou_loss:
                    loc_loss = self._compute_giou_loss(
                        true_xy, true_wh, pred_xy, pred_wh, object_mask
                    )
                elif self.use_diou_loss:
                    loc_loss = self._compute_diou_loss(
                        true_xy, true_wh, pred_xy, pred_wh, object_mask
                    )
                elif self.use_ciou_loss:
                    loc_loss = self._compute_ciou_loss(
                        true_xy, true_wh, pred_xy, pred_wh, object_mask
                    )
                else:
                    loc_loss = self._compute_mse_loss(
                        true_xy, true_wh, pred_xy, pred_wh, object_mask
                    )
                loc_loss = loc_loss / loc_norm_factor
            
            total_loss_location += loc_loss
            
            # ========== OBJECTNESS LOSS ==========
            obj_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
            normalization_factors['objectness'].append(obj_norm_factor)
            obj_loss = self._compute_objectness_loss(
                true_obj, pred_obj, object_mask, ignore_mask, obj_norm_factor
            )
            total_loss_objectness += obj_loss
            
            # ========== ANCHOR LOSS ==========
            # Only compute if NOT Option 2 (Option 2 computes it separately above)
            anchor_loss_this_scale = None
            if self.loss_option != 2:
                anchor_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
                normalization_factors['anchor'].append(anchor_norm_factor)
                anchor_loss_this_scale = self._compute_anchor_loss(
                    true_anchors, pred_anchors, object_mask, ignore_mask, anchor_norm_factor
                )
                total_loss_anchor += self.anchor_scale * anchor_loss_this_scale
            
            # ========== CLASSIFICATION LOSS ==========
            class_norm_factor = self._get_normalization_factor(batch_size, grid_shape, object_mask)
            normalization_factors['classification'].append(class_norm_factor)
            
            if self.use_focal_loss and not self.use_softmax_loss:
                class_loss = self._compute_focal_classification_loss(
                    true_class, pred_class, object_mask, class_norm_factor
                )
            elif self.use_softmax_loss:
                class_loss = self._compute_softmax_classification_loss(
                    true_class, pred_class, object_mask, class_norm_factor
                )
            else:
                class_loss = self._compute_bce_classification_loss(
                    true_class, pred_class, object_mask, class_norm_factor
                )
            
            total_loss_classification += class_loss
            
            # Store per-scale losses
            scale_loss_dict = {
                'scale': layer_idx,
                'localization': loc_loss,
                'objectness': obj_loss,
                'classification': class_loss,
                'localization_norm_factor': loc_norm_factor,
                'objectness_norm_factor': obj_norm_factor,
                'classification_norm_factor': class_norm_factor
            }
            
            if self.loss_option == 2:
                # Anchor loss was computed in Option 2 block
                scale_loss_dict['anchor'] = anchor_loc_loss
                scale_loss_dict['anchor_norm_factor'] = normalization_factors['anchor'][-1]
            elif anchor_loss_this_scale is not None:
                # Anchor loss was computed in the anchor loss block (Option 1 or 3)
                scale_loss_dict['anchor'] = anchor_loss_this_scale
                scale_loss_dict['anchor_norm_factor'] = normalization_factors['anchor'][-1]
            
            per_scale_losses.append(scale_loss_dict)
        
        # Combine all losses
        total_loss = (
            self.coord_scale * total_loss_location +
            self.object_scale * total_loss_objectness +
            self.anchor_scale * total_loss_anchor +
            self.class_scale * total_loss_classification
        )
        
        return {
            'total_loss': total_loss,
            'localization_loss': total_loss_location,
            'objectness_loss': total_loss_objectness,
            'anchor_loss': total_loss_anchor,
            'classification_loss': total_loss_classification,
            'per_scale_losses': per_scale_losses,
            'scaling_coefficients': {
                'coord_scale': self.coord_scale,
                'object_scale': self.object_scale,
                'no_object_scale': self.no_object_scale,
                'class_scale': self.class_scale,
                'anchor_scale': self.anchor_scale
            },
            'normalization_factors': normalization_factors
        }


def extract_loss_components(model, y_true_list, y_pred_list, loss_fn: DiagnosticMultiGridLoss) -> Dict[str, Any]:
    """
    Extract loss components using diagnostic loss function.
    
    Args:
        model: Compiled model
        y_true_list: List of ground truth tensors
        y_pred_list: List of prediction tensors
        loss_fn: DiagnosticMultiGridLoss instance
        
    Returns:
        Dictionary with loss components
    """
    # Get predictions from model (forward pass without gradients)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape._watch_accessed_variables = False
        # Compute loss components
        loss_dict = loss_fn.compute_loss_diagnostic(y_true_list, y_pred_list)
    
    return loss_dict


def compute_batch_statistics(y_true_list: List[tf.Tensor]) -> Dict[str, Any]:
    """
    Compute batch statistics (number of objects, grid cells, etc.).
    
    Args:
        y_true_list: List of ground truth tensors
        
    Returns:
        Dictionary with batch statistics
    """
    stats = {
        'num_scales': len(y_true_list),
        'batch_size': int(y_true_list[0].shape[0]),
        'per_scale_stats': []
    }
    
    total_objects = 0
    for layer_idx, y_true_layer in enumerate(y_true_list):
        # Extract objectness mask
        true_obj = y_true_layer[..., 4:5]
        object_mask = tf.cast(true_obj > 0.5, tf.float32)
        num_objects = tf.reduce_sum(object_mask).numpy()
        total_objects += num_objects
        
        grid_h, grid_w = y_true_layer.shape[1], y_true_layer.shape[2]
        total_cells = grid_h * grid_w * stats['batch_size']
        
        stats['per_scale_stats'].append({
            'scale': layer_idx,
            'grid_shape': [int(grid_h), int(grid_w)],
            'total_cells': int(total_cells),
            'num_objects': int(num_objects),
            'object_density': float(num_objects / total_cells) if total_cells > 0 else 0.0
        })
    
    stats['total_objects'] = int(total_objects)
    return stats


def log_batch_results(batch_idx: int, loss_dict: Dict[str, Any], batch_stats: Dict[str, Any], 
                      verbose: bool = True):
    """
    Log batch results to console.
    
    Args:
        batch_idx: Batch index
        loss_dict: Dictionary with loss components
        batch_stats: Dictionary with batch statistics
        verbose: Whether to print detailed output
    """
    print(f"\n{'='*80}")
    print(f"Batch {batch_idx}")
    print(f"{'='*80}")
    
    print(f"\nBatch Statistics:")
    print(f"  Batch size: {batch_stats['batch_size']}")
    print(f"  Total objects: {batch_stats['total_objects']}")
    print(f"  Number of scales: {batch_stats['num_scales']}")
    
    for scale_stat in batch_stats['per_scale_stats']:
        print(f"  Scale {scale_stat['scale']}: {scale_stat['grid_shape']} grid, "
              f"{scale_stat['num_objects']} objects, "
              f"density={scale_stat['object_density']:.4f}")
    
    print(f"\nLoss Components (Raw Values):")
    print(f"  Localization:  {float(loss_dict['localization_loss']):.6f}")
    print(f"  Objectness:    {float(loss_dict['objectness_loss']):.6f}")
    print(f"  Anchor:        {float(loss_dict['anchor_loss']):.6f}")
    print(f"  Classification: {float(loss_dict['classification_loss']):.6f}")
    print(f"  Total:         {float(loss_dict['total_loss']):.6f}")
    
    print(f"\nScaling Coefficients:")
    scales = loss_dict['scaling_coefficients']
    print(f"  coord_scale:     {scales['coord_scale']:.3f}")
    print(f"  object_scale:    {scales['object_scale']:.3f}")
    print(f"  no_object_scale: {scales['no_object_scale']:.3f}")
    print(f"  class_scale:     {scales['class_scale']:.3f}")
    print(f"  anchor_scale:    {scales['anchor_scale']:.3f}")
    
    print(f"\nWeighted Loss Components:")
    scales = loss_dict['scaling_coefficients']
    print(f"  Localization (weighted):  {float(loss_dict['localization_loss'] * scales['coord_scale']):.6f}")
    print(f"  Objectness (weighted):    {float(loss_dict['objectness_loss'] * scales['object_scale']):.6f}")
    print(f"  Anchor (weighted):       {float(loss_dict['anchor_loss'] * scales['anchor_scale']):.6f}")
    print(f"  Classification (weighted): {float(loss_dict['classification_loss'] * scales['class_scale']):.6f}")
    
    if verbose:
        print(f"\nPer-Scale Breakdown:")
        for scale_loss in loss_dict['per_scale_losses']:
            print(f"  Scale {scale_loss['scale']}:")
            print(f"    Localization:  {float(scale_loss['localization']):.6f} "
                  f"(norm: {float(scale_loss['localization_norm_factor']):.2f})")
            print(f"    Objectness:    {float(scale_loss['objectness']):.6f} "
                  f"(norm: {float(scale_loss['objectness_norm_factor']):.2f})")
            if 'anchor' in scale_loss:
                print(f"    Anchor:        {float(scale_loss['anchor']):.6f} "
                      f"(norm: {float(scale_loss['anchor_norm_factor']):.2f})")
            print(f"    Classification: {float(scale_loss['classification']):.6f} "
                  f"(norm: {float(scale_loss['classification_norm_factor']):.2f})")


def save_results_json(all_results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    # Convert tensors to floats for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {
            'batch_idx': result['batch_idx'],
            'batch_stats': result['batch_stats'],
            'loss_components': {
                'total_loss': float(result['loss_dict']['total_loss']),
                'localization_loss': float(result['loss_dict']['localization_loss']),
                'objectness_loss': float(result['loss_dict']['objectness_loss']),
                'anchor_loss': float(result['loss_dict']['anchor_loss']),
                'classification_loss': float(result['loss_dict']['classification_loss'])
            },
            'scaling_coefficients': result['loss_dict']['scaling_coefficients'],
            'per_scale_losses': []
        }
        
        for scale_loss in result['loss_dict']['per_scale_losses']:
            scale_dict = {
                'scale': scale_loss['scale'],
                'localization': float(scale_loss['localization']),
                'objectness': float(scale_loss['objectness']),
                'classification': float(scale_loss['classification']),
                'localization_norm_factor': float(scale_loss['localization_norm_factor']),
                'objectness_norm_factor': float(scale_loss['objectness_norm_factor']),
                'classification_norm_factor': float(scale_loss['classification_norm_factor'])
            }
            if 'anchor' in scale_loss:
                scale_dict['anchor'] = float(scale_loss['anchor'])
                scale_dict['anchor_norm_factor'] = float(scale_loss['anchor_norm_factor'])
            json_result['per_scale_losses'].append(scale_dict)
        
        json_results.append(json_result)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def save_results_csv(all_results: List[Dict[str, Any]], output_path: str):
    """Save results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'batch_idx', 'batch_size', 'total_objects',
            'total_loss', 'localization_loss', 'objectness_loss', 
            'anchor_loss', 'classification_loss',
            'coord_scale', 'object_scale', 'no_object_scale', 
            'class_scale', 'anchor_scale'
        ])
        
        # Data rows
        for result in all_results:
            loss_dict = result['loss_dict']
            scales = loss_dict['scaling_coefficients']
            writer.writerow([
                result['batch_idx'],
                result['batch_stats']['batch_size'],
                result['batch_stats']['total_objects'],
                float(loss_dict['total_loss']),
                float(loss_dict['localization_loss']),
                float(loss_dict['objectness_loss']),
                float(loss_dict['anchor_loss']),
                float(loss_dict['classification_loss']),
                scales['coord_scale'],
                scales['object_scale'],
                scales['no_object_scale'],
                scales['class_scale'],
                scales['anchor_scale']
            ])
    
    print(f"Results saved to: {output_path}")


def create_diagnostic_loss(trainer: MultiGridTrainer) -> DiagnosticMultiGridLoss:
    """
    Create a diagnostic loss function with the same configuration as the training model.
    
    Args:
        trainer: MultiGridTrainer instance with configured model
        
    Returns:
        DiagnosticMultiGridLoss instance
    """
    # Get loss configuration from training config
    training_config = trainer.config.get('training', {})
    loss_config = training_config.get('loss', {})
    
    # Get anchors (convert from tf.Tensor to numpy if needed)
    anchors_tf = trainer.train_generator.anchors
    anchors = [anchor.numpy() if isinstance(anchor, tf.Tensor) else anchor for anchor in anchors_tf]
    
    # Get model config
    num_classes = trainer.train_generator.num_classes
    input_shape = tuple(trainer.model_config['model']['preset']['input_shape'][:2])
    
    # Create diagnostic loss with same configuration
    diagnostic_loss = DiagnosticMultiGridLoss(
        anchors=anchors,
        num_classes=num_classes,
        input_shape=input_shape,
        label_smoothing=training_config.get('label_smoothing', 0.0),
        elim_grid_sense=training_config.get('elim_grid_sense', False),
        loss_option=training_config.get('loss_option', 2),
        coord_scale=loss_config.get('coord_scale', 1.0),
        object_scale=loss_config.get('object_scale', 1.0),
        no_object_scale=loss_config.get('no_object_scale', 1.0),
        class_scale=loss_config.get('class_scale', 1.0),
        anchor_scale=loss_config.get('anchor_scale', 1.0),
        loss_normalization=training_config.get('loss_normalization', ['batch'])
    )
    
    return diagnostic_loss


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Probe MultiGridLoss components for diagnostic purposes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/probe_multigrid_loss.py --num-batches 2
  python tests/probe_multigrid_loss.py --num-batches 5 --output-dir loss_stats/
  python tests/probe_multigrid_loss.py --config configs/custom_config.yaml
  python tests/probe_multigrid_loss.py --weights weights/model.h5 --num-batches 2
  python tests/probe_multigrid_loss.py --backbone-weights weights/darknet53.h5 --num-batches 2
        """
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=2,
        help='Number of batches to process (default: 2)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training config file (default: configs/train_config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/loss_probe_outputs',
        help='Output directory for results (default: tests/loss_probe_outputs)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format: json, csv, or both (default: both)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed per-scale breakdown'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to pretrained full model weights (overrides config)'
    )
    parser.add_argument(
        '--backbone-weights',
        type=str,
        default=None,
        help='Path to pretrained backbone weights (e.g., darknet53.h5)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("MultiGridLoss Component Diagnostic Tool")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Output directory: {args.output_dir}")
    if args.weights:
        print(f"Full model weights: {args.weights}")
    if args.backbone_weights:
        print(f"Backbone weights: {args.backbone_weights}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    try:
        config = ConfigLoader.load_config(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] Config file not found: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Override weights paths from command-line arguments (similar to train.py)
    if 'resume' not in config:
        config['resume'] = {}
    
    if args.weights:
        config['resume']['weights_path'] = args.weights
        print(f"[INFO] Using full model weights: {args.weights}")
    
    if args.backbone_weights:
        config['resume']['backbone_weights_path'] = args.backbone_weights
        print(f"[INFO] Using backbone weights: {args.backbone_weights}")
    
    print()
    
    # Create trainer (reuses existing infrastructure)
    try:
        trainer = MultiGridTrainer(config)
        trainer.setup_data()
        trainer.build_model()
    except Exception as e:
        print(f"[ERROR] Error setting up trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create diagnostic loss function
    diagnostic_loss = create_diagnostic_loss(trainer)
    
    print(f"\n[INFO] Processing {args.num_batches} batches...")
    print(f"[INFO] Model has {trainer.model.count_params():,} parameters")
    print()
    
    # Process batches
    all_results = []
    train_generator = trainer.train_generator
    
    for batch_idx in range(min(args.num_batches, len(train_generator))):
        try:
            # Get batch from generator
            # Generator returns: (image_data, *y_true), dummy_target
            batch_data, _ = train_generator[batch_idx]
            
            # Extract images and y_true from batch_data
            # batch_data is a tuple: (image_data, y_true_0, y_true_1, y_true_2)
            if isinstance(batch_data, tuple):
                images = batch_data[0]
                y_true_list = list(batch_data[1:])
            else:
                # Fallback if format is different
                images = batch_data
                y_true_list = []
            
            # Get predictions from base model
            # The training model wraps the base model. We need to extract predictions.
            # The training model structure: inputs=[model.input] + y_true, outputs=loss
            # We can build a model that extracts the base model outputs
            
            # Find the multigrid_loss layer to access its inputs
            multigrid_loss_layer = None
            for layer in trainer.model.layers:
                if layer.name == 'multigrid_loss':
                    multigrid_loss_layer = layer
                    break
            
            # Try to extract base model outputs
            # The multigrid_loss Lambda layer receives [*model.outputs, *y_true]
            # We need to get the model.outputs part
            image_input = trainer.model.inputs[0]
            num_scales = len(y_true_list)
            
            if multigrid_loss_layer is not None:
                # Try to extract base model outputs by building a model from the training model's structure
                # The multigrid_loss Lambda layer receives [*model.outputs, *y_true]
                # We need to get the model.outputs part
                try:
                    # Method 1: Try to find the base model as a sub-layer
                    base_model = None
                    for layer in trainer.model.layers:
                        if isinstance(layer, tf.keras.Model) and layer != trainer.model:
                            try:
                                # Check if this model takes image input (shape: [batch, H, W, 3])
                                if hasattr(layer, 'input') and len(layer.input.shape) == 4 and layer.input.shape[-1] == 3:
                                    base_model = layer
                                    break
                            except:
                                pass
                    
                    if base_model is not None:
                        # Use the base model directly
                        y_pred_list = base_model(images, training=False)
                        if not isinstance(y_pred_list, list):
                            y_pred_list = [y_pred_list]
                    else:
                        # Method 2: Build a model that extracts outputs from the training model
                        # The training model's first input is the image, followed by y_true inputs
                        # We can call the model's internal computation to get base outputs
                        # Actually, simpler: access the layer that produces the outputs
                        # The multigrid_loss layer's call method receives the inputs
                        # We can trace the computation graph
                        raise ValueError("Base model not found as sub-layer")
                except Exception as e:
                    print(f"[WARNING] Could not extract base model: {e}")
                    # Fallback: Rebuild base model from config (slower but reliable)
                    from multigriddet.models.multigriddet_darknet import build_multigriddet_darknet
                    model_config = trainer.model_config['model']['preset']
                    input_shape_full = tuple(model_config['input_shape'])
                    # Get anchors and num_classes from generator
                    anchors_from_gen = trainer.train_generator.anchors
                    anchors_np = [a.numpy() if isinstance(a, tf.Tensor) else a for a in anchors_from_gen]
                    num_anchors_per_head = [len(anchors_np[i]) for i in range(len(anchors_np))]
                    num_classes_fallback = trainer.train_generator.num_classes
                    
                    base_model, _ = build_multigriddet_darknet(
                        input_shape=input_shape_full,
                        num_anchors_per_head=num_anchors_per_head,
                        num_classes=num_classes_fallback,
                        weights_path=None  # Don't load weights for diagnostic
                    )
                    
                    # Copy weights from training model if possible
                    try:
                        # Try to find matching layers and copy weights
                        for train_layer in trainer.model.layers:
                            if isinstance(train_layer, tf.keras.Model) and train_layer != trainer.model:
                                try:
                                    base_model.set_weights(train_layer.get_weights())
                                    break
                                except:
                                    pass
                    except:
                        pass  # If weight copying fails, continue with uninitialized model
                    
                    y_pred_list = base_model(images, training=False)
                    if not isinstance(y_pred_list, list):
                        y_pred_list = [y_pred_list]
            else:
                # Fallback: try to find the base model as a sub-model
                base_model = None
                for layer in trainer.model.layers:
                    if isinstance(layer, tf.keras.Model) and layer != trainer.model:
                        try:
                            if hasattr(layer, 'input') and len(layer.input.shape) == 4:
                                base_model = layer
                                break
                        except:
                            pass
                
                if base_model is None:
                    raise ValueError("Could not extract base model from training model. "
                                   "Training model structure may have changed.")
                
                y_pred_list = base_model(images, training=False)
                if not isinstance(y_pred_list, list):
                    y_pred_list = [y_pred_list]
            
            # Compute batch statistics
            batch_stats = compute_batch_statistics(y_true_list)
            
            # Extract loss components
            loss_dict = extract_loss_components(
                trainer.model, y_true_list, y_pred_list, diagnostic_loss
            )
            
            # Store results
            result = {
                'batch_idx': batch_idx,
                'batch_stats': batch_stats,
                'loss_dict': loss_dict
            }
            all_results.append(result)
            
            # Log results
            log_batch_results(batch_idx, loss_dict, batch_stats, verbose=args.verbose)
            
        except Exception as e:
            print(f"\n[ERROR] Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if args.format in ['json', 'both']:
        json_path = os.path.join(args.output_dir, 'loss_components.json')
        save_results_json(all_results, json_path)
    
    if args.format in ['csv', 'both']:
        csv_path = os.path.join(args.output_dir, 'loss_components.csv')
        save_results_csv(all_results, csv_path)
    
    print(f"\n{'='*80}")
    print("Diagnostic complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

