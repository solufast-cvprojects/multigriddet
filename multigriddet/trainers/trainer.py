#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer class for MultiGridDet models.
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import ConfigLoader, build_model_for_training
from ..utils.anchors import load_anchors, load_classes
from ..utils.tf_optimization import optimize_tf_gpu
from ..data import MultiGridDataGenerator, load_annotation_lines


class MultiGridTrainer:
    """Trainer class for MultiGridDet models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.train_dataset = None
        self.val_dataset = None
        self.use_tf_dataset = True  # Default to using tf.data.Dataset
        self.callbacks = []
        
        # Initialize TensorFlow optimizations
        optimize_tf_gpu()
        
        # Load model configuration
        model_config_path = config['model_config']
        self.model_config = ConfigLoader.load_config(model_config_path)
        
        # Merge model config with main config
        self.full_config = ConfigLoader.merge_configs(self.model_config, config)
        
        print("=" * 80)
        print("MultiGridDet Trainer Initialized")
        print("=" * 80)
        
    def setup_data(self):
        """Setup training and validation data generators."""
        data_config = self.config['data']
        training_config = self.config['training']
        
        # Load classes and anchors
        classes_path = data_config['classes_path']
        anchors_path = self.model_config['model']['preset']['anchors_path']
        
        class_names = load_classes(classes_path)
        num_classes = len(class_names)
        anchors = load_anchors(anchors_path)
        
        print(f"\n[INFO] Dataset Configuration:")
        print(f"   Classes: {num_classes}")
        print(f"   Anchors: {len(anchors)} total ({len(anchors[0])}, {len(anchors[1])}, {len(anchors[2])} per scale)")
        
        # Load annotation lines
        train_lines = load_annotation_lines(data_config['train_annotation'], shuffle=True)
        val_lines = load_annotation_lines(data_config['val_annotation'], shuffle=False)
        
        print(f"   Training samples: {len(train_lines)}")
        print(f"   Validation samples: {len(val_lines)}")
        
        # Get input shape
        input_shape = tuple(self.model_config['model']['preset']['input_shape'][:2])
        
        # Get data loader config for num_workers
        data_loader_config = self.config.get('data_loader', {})
        num_workers = data_loader_config.get('num_workers', 8)
        
        # Create training generator
        augment_config = training_config.get('augmentation', {})
        self.train_generator = MultiGridDataGenerator(
            annotation_lines=train_lines,
            batch_size=training_config['batch_size'],
            input_shape=input_shape,
            anchors=anchors,
            num_classes=num_classes,
            augment=augment_config.get('enabled', True),
            enhance_augment=augment_config.get('enhance_type'),
            rescale_interval=augment_config.get('rescale_interval', -1),
            multi_anchor_assign=training_config.get('multi_anchor_assign', False),
            shuffle=True,
            num_workers=num_workers
        )
        
        # Create validation generator
        self.val_generator = MultiGridDataGenerator(
            annotation_lines=val_lines,
            batch_size=training_config['batch_size'],
            input_shape=input_shape,
            anchors=anchors,
            num_classes=num_classes,
            augment=False,
            multi_anchor_assign=training_config.get('multi_anchor_assign', False),
            shuffle=False,
            num_workers=num_workers
        )
        
        # Check if we should use tf.data.Dataset (default: True)
        # data_loader_config already loaded above
        self.use_tf_dataset = data_loader_config.get('use_tf_dataset', True)
        
        if self.use_tf_dataset:
            # Build native tf.data.Dataset pipeline for GPU-accelerated data loading
            use_gpu_preprocessing = data_loader_config.get('use_gpu_preprocessing', True)
            prefetch_buffer_config = data_loader_config.get('prefetch_buffer', 'auto')
            
            # Handle prefetch buffer: can be 'auto', integer (batches), or None
            if prefetch_buffer_config == 'auto' or prefetch_buffer_config is None:
                prefetch_batches = 6  # Default: 6 batches for good GPU-CPU overlap
            elif isinstance(prefetch_buffer_config, str) and prefetch_buffer_config.lower() == 'auto':
                prefetch_batches = 6
            elif isinstance(prefetch_buffer_config, (int, float)):
                prefetch_batches = int(prefetch_buffer_config)
            else:
                prefetch_batches = 6
            
            num_parallel_calls_config = data_loader_config.get('num_parallel_calls', 'auto')
            if num_parallel_calls_config == 'auto' or num_parallel_calls_config is None:
                num_parallel_calls = tf.data.AUTOTUNE
            elif isinstance(num_parallel_calls_config, str) and num_parallel_calls_config.lower() == 'auto':
                num_parallel_calls = tf.data.AUTOTUNE
            elif isinstance(num_parallel_calls_config, (int, float)):
                num_parallel_calls = int(num_parallel_calls_config)
            else:
                num_parallel_calls = tf.data.AUTOTUNE
            
            if use_gpu_preprocessing:
                # Get additional optimization parameters
                shuffle_buffer_size = data_loader_config.get('shuffle_buffer_size', 4096)
                interleave_cycle_length = data_loader_config.get('interleave_cycle_length', None)
                
                print("[INFO] Building native tf.data.Dataset pipeline with GPU-accelerated preprocessing...")
                print(f"   Prefetch buffer: {prefetch_batches} batches")
                print(f"   Parallel calls: {num_parallel_calls if num_parallel_calls != tf.data.AUTOTUNE else 'AUTOTUNE'}")
                print(f"   Shuffle buffer: {shuffle_buffer_size}")
                if interleave_cycle_length:
                    print(f"   Interleave cycle length: {interleave_cycle_length}")
                
                self.train_dataset = self.train_generator.build_tf_dataset(
                    prefetch_buffer_size=prefetch_batches,
                    num_parallel_calls=num_parallel_calls,
                    shuffle_buffer_size=shuffle_buffer_size,
                    interleave_cycle_length=interleave_cycle_length,
                    use_gpu_preprocessing=True
                )
                self.val_dataset = self.val_generator.build_tf_dataset(
                    prefetch_buffer_size=prefetch_batches,
                    num_parallel_calls=num_parallel_calls,
                    shuffle_buffer_size=shuffle_buffer_size,
                    interleave_cycle_length=None,  # No interleaving for validation
                    use_gpu_preprocessing=True
                )
                print("✓ Native tf.data.Dataset pipeline created with GPU preprocessing")
            else:
                print("[INFO] Converting generators to tf.data.Dataset (legacy mode)...")
                self.train_dataset = self.train_generator.to_tf_dataset(
                    prefetch_buffer_size=prefetch_buffer,
                    num_parallel_calls=num_parallel_calls
                )
                self.val_dataset = self.val_generator.to_tf_dataset(
                    prefetch_buffer_size=prefetch_buffer,
                    num_parallel_calls=num_parallel_calls
                )
                print("✓ tf.data.Dataset created with prefetching enabled")
        else:
            print("✓ Using Sequence generators (tf.data.Dataset disabled)")
        
        print("✓ Data generators created successfully\n")
        
    def build_model(self):
        """Build and compile the model."""
        print("🔨 Building model...")
        
        # Get anchors from config
        anchors_path = self.model_config['model']['preset']['anchors_path']
        anchors = load_anchors(anchors_path)
        
        self.model = build_model_for_training(self.full_config, anchors=anchors)
        
        # Load weights if resume is enabled
        resume_config = self.config.get('resume', {})
        if resume_config.get('enabled', False):
            weights_path = resume_config.get('weights_path')
            if weights_path and os.path.exists(weights_path):
                print(f"   Loading weights from: {weights_path}")
                self.model.load_weights(weights_path, by_name=True)
                print("   ✓ Weights loaded successfully")
        
        # Model is already compiled by build_multigriddet_darknet_train
        # No need to compile again - just use the model as-is
        
        print(f"✓ Model built successfully")
        print(f"   Total parameters: {self.model.count_params():,}\n")
        
    def setup_callbacks(self):
        """Setup training callbacks."""
        callback_config = self.config.get('callbacks', {})
        output_config = self.config.get('output', {})
        
        print("[INFO] Setting up callbacks...")
        
        # TensorBoard
        if 'tensorboard' in callback_config:
            log_dir = callback_config['tensorboard'].get('log_dir', 'logs/tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            tensorboard = TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=False
            )
            self.callbacks.append(tensorboard)
            print(f"   ✓ TensorBoard: {log_dir}")
        
        # Model Checkpoint
        if 'checkpoint' in callback_config:
            checkpoint_config = callback_config['checkpoint']
            save_dir = checkpoint_config.get('save_dir', 'logs/checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(
                save_dir,
                'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.weights.h5'
            )
            
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                verbose=1,
                save_best_only=checkpoint_config.get('save_best_only', True),
                save_weights_only=True,
                mode='min'
            )
            self.callbacks.append(checkpoint)
            print(f"   ✓ ModelCheckpoint: {save_dir}")
        
        # Learning Rate Scheduler
        if 'lr_schedule' in self.config:
            lr_config = self.config['lr_schedule']
            if lr_config.get('type') == 'reduce_on_plateau':
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=lr_config.get('factor', 0.5),
                    patience=lr_config.get('patience', 3),
                    min_lr=lr_config.get('min_lr', 1e-7),
                    verbose=1
                )
                self.callbacks.append(reduce_lr)
                print(f"   ✓ ReduceLROnPlateau")
        
        # Early Stopping
        if 'early_stopping' in callback_config:
            es_config = callback_config['early_stopping']
            early_stop = EarlyStopping(
                monitor=es_config.get('monitor', 'val_loss'),
                patience=es_config.get('patience', 10),
                verbose=1,
                mode='min'
            )
            self.callbacks.append(early_stop)
            print(f"   ✓ EarlyStopping (patience={es_config.get('patience', 10)})")
        
        print()
        
    def train(self):
        """Run the training process."""
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        # Setup components
        self.setup_data()
        self.build_model()
        self.setup_callbacks()
        
        # Get training parameters
        training_config = self.config['training']
        epochs = training_config.get('epochs', 100)
        initial_epoch = training_config.get('initial_epoch', 0)
        
        print(f"[INFO] Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Initial Epoch: {initial_epoch}")
        print(f"   Batch Size: {training_config['batch_size']}")
        print(f"   Learning Rate: {training_config.get('learning_rate', 0.001)}")
        print(f"   Loss Option: {training_config.get('loss_option', 2)}")
        print()
        
        # Two-stage training (optional)
        transfer_epochs = training_config.get('transfer_epochs', 0)
        if transfer_epochs > 0 and initial_epoch < transfer_epochs:
            print(f"🔒 Stage 1: Transfer Learning ({transfer_epochs} epochs with frozen backbone)")
            
            # Freeze backbone layers
            freeze_level = training_config.get('freeze_level', 1)
            if freeze_level > 0:
                # Freeze backbone (first 185 layers for darknet53)
                for layer in self.model.layers[:185]:
                    layer.trainable = False
                print(f"   Frozen first 185 layers")
            
            # No recompilation needed in Keras 3.x - model is already compiled
            
            # Train stage 1
            train_data = self.train_dataset if self.use_tf_dataset else self.train_generator
            val_data = self.val_dataset if self.use_tf_dataset else self.val_generator
            steps_per_epoch = max(1, len(self.train_generator))
            validation_steps = max(1, len(self.val_generator))
            
            self.model.fit(
                train_data,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_data,
                validation_steps=validation_steps,
                epochs=transfer_epochs,
                initial_epoch=initial_epoch,
                callbacks=self.callbacks,
                verbose=1
            )
            
            # Unfreeze all layers for stage 2
            print(f"\n🔓 Stage 2: Fine-tuning (all layers trainable)")
            for layer in self.model.layers:
                layer.trainable = True
            
            # Model is already compiled with custom loss, no need to recompile
            
            initial_epoch = transfer_epochs
        
        # Train full model
        print(f"[INFO] Training full model...")
        train_data = self.train_dataset if self.use_tf_dataset else self.train_generator
        val_data = self.val_dataset if self.use_tf_dataset else self.val_generator
        steps_per_epoch = max(1, len(self.train_generator))
        validation_steps = max(1, len(self.val_generator))
        
        history = self.model.fit(
            train_data,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data,
            validation_steps=validation_steps,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=self.callbacks,
            verbose=1
        )
        
        # Save final model
        output_dir = self.config.get('output', {}).get('model_dir', 'trained_models')
        os.makedirs(output_dir, exist_ok=True)
        final_model_path = os.path.join(output_dir, 'final_model.weights.h5')
        self.model.save_weights(final_model_path)
        
        print("\n" + "=" * 80)
        print(f"✓ Training Complete!")
        print(f"   Final model saved to: {final_model_path}")
        print("=" * 80)
        
        return history





