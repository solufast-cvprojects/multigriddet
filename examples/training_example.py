#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Training Example
Demonstrates how to train a MultiGridDet model with real data.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from multigriddet package
from multigriddet.models import build_multigriddet_darknet_train
from multigriddet.utils.anchors import load_anchors, load_classes
from multigriddet.utils.tf_optimization import optimize_tf_gpu
from multigriddet.losses.multigrid_loss import MultiGridLoss
from multigriddet.data import MultiGridDataGenerator, load_annotation_lines


def create_training_config():
    """Create a training configuration dictionary."""
    
    config = {
        'model': {
            'weights_path': None,  # Start from scratch
        },
        'data': {
            'classes_path': 'configs/coco_classes.txt',
            'anchors_path': 'configs/yolov3_coco_anchor.txt',
            'input_shape': (608, 608, 3),
            'train_file': 'data/coco_train2017.txt',  # Real annotation file
            'val_file': 'data/coco_val2017.txt',      # Real validation file
        },
        'training': {
            'batch_size': 1,  # Small batch for testing
            'learning_rate': 1e-3,
            'transfer_epochs': 1,  # Just 1 epoch for testing
            'total_epochs': 1,     # Just 1 epoch for testing
            'initial_epoch': 0,
            'label_smoothing': 0.0,
            'elim_grid_sense': False,
            'freeze_level': 1,  # Freeze backbone initially
            'loss_option': 2,  # Options: 1 (IoL-weighted MSE), 2 (IoL-weighted MSE with mask), 3 (GIoU/DIoU)
            'log_dir': 'logs/training_example',
            'augment': True,  # Enable data augmentation
            'multi_anchor_assign': False,  # Single anchor assignment
            'rescale_interval': -1,  # No multi-scale training for testing
        }
    }
    
    return config


def setup_data_generators(config, anchors, num_classes):
    """Setup training and validation data generators."""
    
    # Check if annotation files exist
    train_file = config['data']['train_file']
    val_file = config['data']['val_file']
    
    if not os.path.exists(train_file):
        print(f"[WARNING] Training file not found: {train_file}")
        print("Creating dummy annotation file for testing...")
        create_dummy_annotation_file(train_file)
    
    if not os.path.exists(val_file):
        print(f"[WARNING] Validation file not found: {val_file}")
        print("Creating dummy annotation file for testing...")
        create_dummy_annotation_file(val_file)
    
    # Load annotation lines
    train_lines = load_annotation_lines(train_file, shuffle=True)
    val_lines = load_annotation_lines(val_file, shuffle=False)
    
    print(f"Loaded {len(train_lines)} training samples")
    print(f"Loaded {len(val_lines)} validation samples")
    
    # Create data generators
    train_gen = MultiGridDataGenerator(
        annotation_lines=train_lines,
        batch_size=config['training']['batch_size'],
        input_shape=config['data']['input_shape'][:2],  # (height, width)
        anchors=anchors,
        num_classes=num_classes,
        augment=config['training']['augment'],
        multi_anchor_assign=config['training']['multi_anchor_assign'],
        rescale_interval=config['training']['rescale_interval'],
        shuffle=True
    )
    
    val_gen = MultiGridDataGenerator(
        annotation_lines=val_lines,
        batch_size=config['training']['batch_size'],
        input_shape=config['data']['input_shape'][:2],  # (height, width)
        anchors=anchors,
        num_classes=num_classes,
        augment=False,  # No augmentation for validation
        multi_anchor_assign=config['training']['multi_anchor_assign'],
        rescale_interval=-1,  # No multi-scale for validation
        shuffle=False
    )
    
    return train_gen, val_gen


def create_dummy_annotation_file(file_path):
    """Create a dummy annotation file for testing."""
    
    # Create dummy annotation lines
    dummy_lines = [
        "examples/images/dog.jpg 125,196,498,553,16",
        "examples/images/person.jpg 59,51,561,442,0",
        "examples/images/car.jpg 289,70,316,97,2",
    ]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write dummy file
    with open(file_path, 'w') as f:
        for line in dummy_lines:
            f.write(line + '\n')
    
    print(f"Created dummy annotation file: {file_path}")


def main():
    """Run MultiGridDet training example."""
    
    print("MultiGridDet Training Example")
    print("=" * 40)
    
    # Get configuration
    config = create_training_config()
    
    # Optimize TensorFlow
    optimize_tf_gpu()
    
    # Load classes and anchors
    print("Loading classes and anchors...")
    class_names = load_classes(config['data']['classes_path'])
    anchors = load_anchors(config['data']['anchors_path'])
    
    print(f"Loaded {len(class_names)} classes")
    print(f"Loaded {len(anchors)} anchor sets")
    
    # Create model
    print("Creating multigriddet_darknet model...")
    num_anchors_per_head = [len(anchors[l]) for l in range(len(anchors))]
    num_classes = len(class_names)
    input_shape = config['data']['input_shape']
    
    model, backbone_len = build_multigriddet_darknet_train(
        anchors=anchors,
        num_classes=num_classes,
        input_shape=input_shape,
        weights_path=config['model']['weights_path'],
        freeze_level=config['training']['freeze_level'],
        optimizer=None,  # Will be set below
        label_smoothing=config['training']['label_smoothing'],
        elim_grid_sense=config['training']['elim_grid_sense'],
        loss_option=config['training']['loss_option']
    )
    
    print(f"Model created with {model.count_params()} parameters")
    print(f"Backbone length: {backbone_len}")
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    model.compile(optimizer=optimizer, loss={'multigrid_loss': lambda y_true, y_pred: y_pred})
    
    # Setup data generators
    print("Setting up data generators...")
    try:
        train_gen, val_gen = setup_data_generators(config, anchors, num_classes)
        print("[OK] Data generators created successfully")
    except Exception as e:
        print(f"[ERROR] Data generator setup failed: {e}")
        return False
    
    # Test data loading
    print("Testing data loading...")
    try:
        # Get a batch from training generator
        batch_data, batch_labels = train_gen[0]
        images = batch_data[0]
        targets = batch_data[1:]
        
        print(f"[OK] Data loading successful!")
        print(f"Image shape: {images.shape}")
        print(f"Target shapes: {[t.shape for t in targets]}")
        
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return False
    
    # Test training step
    print("Testing training step...")
    try:
        # Debug: Check model inputs
        print(f"Model expects {len(model.input)} inputs")
        print(f"Model input shapes: {[inp.shape for inp in model.input]}")
        print(f"Images shape: {images.shape}")
        print(f"Targets shapes: {[t.shape for t in targets]}")
        
        # Manual training step - provide all inputs (image + targets)
        model_inputs = [images] + list(targets)
        print(f"Model inputs length: {len(model_inputs)}")
        print(f"Model inputs shapes: {[inp.shape for inp in model_inputs]}")
        
        # The training model computes loss internally and returns it
        with tf.GradientTape() as tape:
            loss = model(model_inputs)  # This returns the loss directly
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"[OK] Training step successful! Loss: {loss.numpy():.4f}")
        
    except Exception as e:
        print(f"[ERROR] Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model saving
    print("Testing model saving...")
    try:
        model_path = 'examples/test_model.h5'
        model.save(model_path)
        print(f"[OK] Model saved to: {model_path}")
        
        # Clean up test file
        if os.path.exists(model_path):
            os.remove(model_path)
            print("[OK] Test file cleaned up")
            
    except Exception as e:
        print(f"[ERROR] Model saving failed: {e}")
        return False
    
    print("[OK] MultiGridDet training example completed!")
    return True


if __name__ == '__main__':
    success = main()
    if success:
        print("\n[SUCCESS] MultiGridDet training is working!")
    else:
        print("\n[ERROR] MultiGridDet training example failed.")

