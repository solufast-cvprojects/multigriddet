#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer class for MultiGridDet models.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
)
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import ConfigLoader, build_model_for_training
from ..utils.anchors import load_anchors, load_classes
from ..utils.tf_optimization import optimize_tf_gpu
from ..data import MultiGridDataGenerator, load_annotation_lines


class CosineAnnealingWithWarmup(Callback):
    """
    Cosine annealing learning rate schedule with warmup.
    
    Modern LR schedule used in YOLOv8/YOLOv9:
    - Warmup: Linear increase from warmup_lr to initial_lr over warmup_epochs
    - Cosine Annealing: Smooth decay from initial_lr to min_lr following cosine curve
    """
    
    def __init__(self, initial_lr: float, min_lr: float = 1e-7, 
                 warmup_epochs: int = 3, total_epochs: int = 100,
                 warmup_lr_factor: float = 0.01, verbose: int = 1):
        """
        Initialize cosine annealing with warmup.
        
        Args:
            initial_lr: Initial learning rate after warmup
            min_lr: Minimum learning rate (end of cosine decay)
            warmup_epochs: Number of epochs for warmup
            total_epochs: Total number of training epochs
            warmup_lr_factor: Factor to multiply initial_lr for warmup start (default: 0.01 = 1% of initial_lr)
            verbose: Verbosity level
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = initial_lr * warmup_lr_factor
        self.verbose = verbose
        self.epoch_count = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        """Update learning rate at the beginning of each epoch."""
        self.epoch_count = epoch + 1  # epoch is 0-indexed
        
        if self.epoch_count <= self.warmup_epochs:
            # Warmup phase: linear increase from warmup_lr to initial_lr
            lr = self.warmup_lr + (self.initial_lr - self.warmup_lr) * (self.epoch_count / self.warmup_epochs)
        else:
            # Cosine annealing phase
            # Calculate progress through cosine decay (0 to 1)
            progress = (self.epoch_count - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            # Cosine decay: lr = min_lr + (initial_lr - min_lr) * (1 + cos(π * progress)) / 2
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Set learning rate (compatible with both TF 2.x and Keras 3.x)
        # Use assign() method which works in both eager and graph mode without numpy conversion
        if hasattr(self.model.optimizer.learning_rate, 'assign'):
            self.model.optimizer.learning_rate.assign(lr)
        elif isinstance(self.model.optimizer.learning_rate, (int, float)):
            # If it's a simple float, we can set it directly
            self.model.optimizer.learning_rate = lr
        else:
            # Fallback: use set_value (should work in graph mode)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        
        if self.verbose > 0:
            print(f'\nEpoch {self.epoch_count}/{self.total_epochs} - Learning rate: {lr:.2e}')
    
    def on_train_begin(self, logs=None):
        """Initialize learning rate at the start of training."""
        # Set initial learning rate (compatible with both TF 2.x and Keras 3.x)
        # Use assign() method which works in both eager and graph mode without numpy conversion
        if hasattr(self.model.optimizer.learning_rate, 'assign'):
            self.model.optimizer.learning_rate.assign(self.warmup_lr)
        elif isinstance(self.model.optimizer.learning_rate, (int, float)):
            # If it's a simple float, we can set it directly
            self.model.optimizer.learning_rate = self.warmup_lr
        else:
            # Fallback: use set_value (should work in graph mode)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.warmup_lr)
        if self.verbose > 0:
            print(f'\nCosine Annealing with Warmup initialized:')
            print(f'  Warmup epochs: {self.warmup_epochs}')
            print(f'  Initial LR: {self.initial_lr:.2e}')
            print(f'  Min LR: {self.min_lr:.2e}')
            print(f'  Warmup start LR: {self.warmup_lr:.2e}')


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
            num_workers=num_workers,
            mosaic_prob=augment_config.get('mosaic_prob', 0.3),
            mixup_prob=augment_config.get('mixup_prob', 0.1),
            max_boxes_per_image=augment_config.get('max_boxes_per_image', 100)
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
                
                # Check if we're in frozen backbone stage (transfer learning)
                # In this stage, model compute is lighter, so reduce data pipeline overhead
                training_config = self.config.get('training', {})
                transfer_epochs = training_config.get('transfer_epochs', 0)
                initial_epoch = training_config.get('initial_epoch', 0)
                is_frozen_stage = transfer_epochs > 0 and initial_epoch < transfer_epochs
                
                if is_frozen_stage:
                    # Optimize for frozen backbone: reduce overhead, focus on throughput
                    # Interleaving adds overhead when model compute is light
                    if interleave_cycle_length is None or interleave_cycle_length > 8:
                        interleave_cycle_length = None  # Disable or reduce interleaving
                    # Reduce parallel calls slightly to reduce contention
                    if num_parallel_calls != tf.data.AUTOTUNE and num_parallel_calls > 16:
                        num_parallel_calls = 16
                    print("[INFO] Frozen backbone stage detected - optimizing data pipeline for lighter compute")
                
                print("[INFO] Building native tf.data.Dataset pipeline with GPU-accelerated preprocessing...")
                print(f"   Prefetch buffer: {prefetch_batches} batches")
                print(f"   Parallel calls: {num_parallel_calls if num_parallel_calls != tf.data.AUTOTUNE else 'AUTOTUNE'}")
                print(f"   Shuffle buffer: {shuffle_buffer_size}")
                if interleave_cycle_length:
                    print(f"   Interleave cycle length: {interleave_cycle_length}")
                else:
                    print(f"   Interleaving: disabled (optimized for current stage)")
                
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
        
        # Get weights paths from resume config (for pretrained weights)
        # These are loaded during model building, before freezing
        resume_config = self.config.get('resume', {})
        weights_path = resume_config.get('weights_path')
        backbone_weights_path = resume_config.get('backbone_weights_path')
        
        # Build model with weights (weights are loaded during model construction)
        self.model = build_model_for_training(
            self.full_config, 
            anchors=anchors,
            weights_path=weights_path,
            backbone_weights_path=backbone_weights_path
        )
        
        # Note: Weights are now loaded during model building (before freezing)
        # resume.enabled is only for actual resume (optimizer state, epoch number, etc.)
        
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
            schedule_type = lr_config.get('type', 'reduce_on_plateau')
            
            if schedule_type == 'cosine_annealing':
                # Modern cosine annealing with warmup (recommended)
                training_config = self.config.get('training', {})
                optimizer_config = self.config.get('optimizer', {})
                total_epochs = training_config.get('epochs', 100)
                # Learning rate priority: training.learning_rate > optimizer.learning_rate > default
                # This matches the priority used in create_optimizer_from_config
                if 'learning_rate' in training_config:
                    initial_lr = training_config['learning_rate']
                elif 'learning_rate' in optimizer_config:
                    initial_lr = optimizer_config['learning_rate']
                else:
                    initial_lr = 0.001
                
                cosine_lr = CosineAnnealingWithWarmup(
                    initial_lr=initial_lr,
                    min_lr=lr_config.get('min_lr', 1e-7),
                    warmup_epochs=lr_config.get('warmup_epochs', 3),
                    total_epochs=total_epochs,
                    warmup_lr_factor=lr_config.get('warmup_lr_factor', 0.01),
                    verbose=1
                )
                self.callbacks.append(cosine_lr)
                print(f"   ✓ CosineAnnealingWithWarmup (warmup={lr_config.get('warmup_epochs', 3)} epochs)")
                
            elif schedule_type == 'reduce_on_plateau':
                # Legacy reduce on plateau (reactive)
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=lr_config.get('factor', 0.5),
                    patience=lr_config.get('patience', 3),
                    min_lr=lr_config.get('min_lr', 1e-7),
                    verbose=1
                )
                self.callbacks.append(reduce_lr)
                print(f"   ✓ ReduceLROnPlateau (legacy)")
            else:
                print(f"   ⚠ Unknown LR schedule type: {schedule_type}, skipping...")
        
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
        
        # Multi-stage training (optional)
        # The model is already frozen during building based on freeze_level in config
        # This section handles the transfer learning stage and transition to full fine-tuning
        transfer_epochs = training_config.get('transfer_epochs', 0)
        if transfer_epochs > 0 and initial_epoch < transfer_epochs:
            freeze_level = training_config.get('freeze_level', 1)
            
            # Count frozen layers to report current state
            # Model was already frozen during building, so we just report the state
            frozen_count = sum(1 for layer in self.model.layers if not layer.trainable)
            total_count = len(self.model.layers)
            
            if freeze_level == 2:
                print(f"🔒 Stage 1: Transfer Learning ({transfer_epochs} epochs with frozen backbone+neck, head trainable)")
            elif freeze_level == 1:
                print(f"🔒 Stage 1: Transfer Learning ({transfer_epochs} epochs with frozen backbone only)")
            else:
                print(f"🔒 Stage 1: Transfer Learning ({transfer_epochs} epochs)")
            
            print(f"   Frozen layers: {frozen_count}/{total_count} (freeze_level: {freeze_level})")
            
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
            
            # Transition to stage 2: Unfreeze based on next freeze_level
            # If freeze_level was 2, we might want to go to 1 (freeze backbone only)
            # If freeze_level was 1, we go to 0 (unfreeze all)
            # For now, we always unfreeze all for stage 2
            # For three-stage training (2→1→0), user should manually update config and resume
            next_freeze_level = training_config.get('next_freeze_level', 0)
            
            if next_freeze_level == 1:
                # Transition: freeze_level 2 → 1 (unfreeze neck, keep backbone frozen)
                # Find backbone length (typically 185 for darknet53)
                # We'll unfreeze layers after backbone
                backbone_len = 185  # Default for darknet53
                for i in range(backbone_len, len(self.model.layers)):
                    self.model.layers[i].trainable = True
                print(f"\n🔓 Stage 2: Fine-tuning (backbone frozen, neck+head trainable)")
                print(f"   Unfroze layers {backbone_len} to {len(self.model.layers)-1}")
            else:
                # Transition: unfreeze all (freeze_level 1 → 0 or 2 → 0)
                for layer in self.model.layers:
                    layer.trainable = True
                print(f"\n🔓 Stage 2: Fine-tuning (all layers trainable)")
            
            # Recompile model to refresh trainable variables after unfreezing
            # Get optimizer and loss from current model
            old_optimizer = self.model.optimizer
            loss = self.model.loss
            
            # Extract learning rate as Python float to avoid tensor-to-numpy conversion issues
            # This prevents the "numpy() is only available when eager execution is enabled" error
            # We avoid calling .numpy() on tensors by using config values instead
            # Learning rate priority: training.learning_rate > optimizer.learning_rate > default
            training_config = self.config.get('training', {})
            optimizer_config = self.config.get('optimizer', {})
            if 'learning_rate' in training_config:
                current_lr = training_config['learning_rate']
            elif 'learning_rate' in optimizer_config:
                current_lr = optimizer_config['learning_rate']
            else:
                current_lr = 0.001
            
            # Try to get the current learning rate if it's a simple float (not a tensor)
            # This avoids tensor-to-numpy conversion that requires eager execution
            try:
                if isinstance(old_optimizer.learning_rate, (int, float)):
                    current_lr = float(old_optimizer.learning_rate)
                elif hasattr(old_optimizer.learning_rate, '__call__'):
                    # It's a schedule - use config value (callback will adjust it anyway)
                    pass
                # If it's a tensor, we skip extraction and use config value
                # The callback will set the correct LR in the next epoch
            except (AttributeError, TypeError):
                # Fallback: use config value
                pass
            
            # Create a new optimizer instance with the learning rate from config
            # This ensures we have a clean optimizer without tensor dependencies
            from ..config.model_builder import create_optimizer_from_config
            optimizer = create_optimizer_from_config(self.config)
            # Set learning rate to the extracted/config value
            if hasattr(optimizer.learning_rate, 'assign'):
                optimizer.learning_rate.assign(current_lr)
            elif not hasattr(optimizer.learning_rate, '__call__'):
                # Only set if it's not a schedule
                optimizer.learning_rate = current_lr
            
            # Recompile with the new optimizer
            self.model.compile(optimizer=optimizer, loss=loss)
            print(f"   Model recompiled to refresh trainable variables (LR: {current_lr:.2e})")
            
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





