#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow optimization utilities for MultiGridDet.
"""

import tensorflow as tf
import os
from typing import Optional, Dict, Any


# =============================================================================
# Modern Activation Functions - Self-contained implementations
# =============================================================================

def swish(x):
    """
    Swish activation function.
    
    Args:
        x: Input tensor.
    
    Returns:
        The Swish activation: `x * sigmoid(x)`.
    
    References:
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    try:
        # Use native TensorFlow implementation if available (more memory efficient)
        return tf.nn.swish(x)
    except AttributeError:
        # Fallback to manual implementation
        return x * tf.nn.sigmoid(x)


def hard_sigmoid(x):
    """
    Hard sigmoid activation function.
    
    Args:
        x: Input tensor.
    
    Returns:
        Hard sigmoid activation: `relu6(x + 3) / 6`.
    """
    return tf.nn.relu6(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    """
    Hard swish activation function.
    
    Args:
        x: Input tensor.
    
    Returns:
        Hard swish activation: `x * hard_sigmoid(x)`.
    """
    return x * hard_sigmoid(x)


def mish(x):
    """
    Mish activation function.
    
    Args:
        x: Input tensor.
    
    Returns:
        Mish activation: `x * tanh(softplus(x))`.
    
    References:
        [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    """
    return x * tf.nn.tanh(tf.nn.softplus(x))


def get_custom_objects():
    """
    Get custom objects for Keras model loading.
    
    Returns:
        Dictionary containing custom objects for Keras model loading.
    """
    return {
        'tf': tf,
        'swish': swish,
        'hard_sigmoid': hard_sigmoid,
        'hard_swish': hard_swish,
        'mish': mish
    }


class TFOptimizationUtils:
    """Utility class for TensorFlow optimizations."""
    
    @staticmethod
    def optimize_tf_gpu(memory_growth: bool = True,
                       allow_memory_growth: bool = True,
                       per_process_gpu_memory_fraction: Optional[float] = None,
                       visible_devices: Optional[str] = None) -> None:
        """
        Optimize TensorFlow GPU settings.
        
        Args:
            memory_growth: Whether to enable memory growth
            allow_memory_growth: Whether to allow memory growth
            per_process_gpu_memory_fraction: Fraction of GPU memory to use
            visible_devices: GPU devices to make visible
        """
        # Set visible devices
        if visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
        
        # Configure GPU memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    if memory_growth:
                        tf.config.experimental.set_memory_growth(gpu, allow_memory_growth)
                    
                    if per_process_gpu_memory_fraction is not None:
                        tf.config.experimental.set_memory_growth(gpu, False)
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=int(per_process_gpu_memory_fraction * 1024)
                            )]
                        )
                
                print(f"GPU optimization configured for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU optimization failed: {e}")
        else:
            print("No GPU devices found")
    
    @staticmethod
    def configure_mixed_precision(policy: str = 'mixed_float16',
                                loss_scale: str = 'dynamic') -> None:
        """
        Configure mixed precision training.
        
        Args:
            policy: Mixed precision policy ('mixed_float16', 'mixed_bfloat16')
            loss_scale: Loss scaling strategy ('dynamic', 'fixed')
        """
        try:
            # Set mixed precision policy
            if policy == 'mixed_float16':
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            elif policy == 'mixed_bfloat16':
                tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
            else:
                raise ValueError(f"Unsupported policy: {policy}")
            
            print(f"Mixed precision policy set to: {policy}")
            
            # Configure loss scaling
            if loss_scale == 'dynamic':
                # Dynamic loss scaling is the default
                pass
            elif loss_scale == 'fixed':
                # Fixed loss scaling would need custom implementation
                pass
            
        except Exception as e:
            print(f"Mixed precision configuration failed: {e}")
    
    @staticmethod
    def enable_xla_compilation() -> None:
        """Enable XLA (Accelerated Linear Algebra) compilation."""
        try:
            # Enable XLA compilation
            tf.config.optimizer.set_jit(True)
            print("XLA compilation enabled")
        except Exception as e:
            print(f"XLA compilation setup failed: {e}")
    
    @staticmethod
    def configure_tf_optimizations(enable_xla: bool = True,
                                 enable_grappler: bool = True,
                                 enable_remapping: bool = True) -> None:
        """
        Configure TensorFlow optimizations.
        
        Args:
            enable_xla: Whether to enable XLA compilation
            enable_grappler: Whether to enable Grappler optimizations
            enable_remapping: Whether to enable remapping optimizations
        """
        try:
            # Configure optimizer options
            optimizer_options = tf.config.optimizer.get_jit()
            
            if enable_xla:
                tf.config.optimizer.set_jit(True)
            
            # Configure Grappler optimizations
            if enable_grappler:
                tf.config.optimizer.set_experimental_options({
                    'layout_optimizer': True,
                    'constant_folding': True,
                    'shape_optimization': True,
                    'remapping': enable_remapping,
                    'arithmetic_optimization': True,
                    'dependency_optimization': True,
                    'loop_optimization': True,
                    'function_optimization': True,
                    'debug_stripper': True,
                    'scoped_allocator_optimization': True,
                    'pin_to_host_optimization': True,
                    'implementation_selector': True,
                    'auto_mixed_precision': True,
                })
            
            print("TensorFlow optimizations configured")
        except Exception as e:
            print(f"TensorFlow optimization configuration failed: {e}")
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """
        Get GPU information.
        
        Returns:
            Dictionary with GPU information
        """
        gpu_info = {
            'gpus_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory': []
        }
        
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_info['gpus_available'] = True
                gpu_info['gpu_count'] = len(gpus)
                
                for gpu in gpus:
                    gpu_info['gpu_names'].append(gpu.name)
                    
                    # Get memory info
                    memory_info = tf.config.experimental.get_memory_info(gpu.name)
                    gpu_info['gpu_memory'].append({
                        'current': memory_info['current'] / 1024**3,  # Convert to GB
                        'peak': memory_info['peak'] / 1024**3  # Convert to GB
                    })
        except Exception as e:
            print(f"Failed to get GPU info: {e}")
        
        return gpu_info
    
    @staticmethod
    def print_tf_info() -> None:
        """Print TensorFlow and system information."""
        print("=" * 50)
        print("TensorFlow Information")
        print("=" * 50)
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {tf.keras.__version__}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"Built with GPU: {tf.test.is_built_with_gpu_support()}")
        
        # GPU information
        gpu_info = TFOptimizationUtils.get_gpu_info()
        if gpu_info['gpus_available']:
            print(f"GPU count: {gpu_info['gpu_count']}")
            for i, (name, memory) in enumerate(zip(gpu_info['gpu_names'], gpu_info['gpu_memory'])):
                print(f"GPU {i}: {name}")
                print(f"  Memory: {memory['current']:.2f}GB / {memory['peak']:.2f}GB")
        else:
            print("No GPU devices available")
        
        print("=" * 50)
    
    @staticmethod
    def create_optimized_dataset(dataset: tf.data.Dataset,
                               batch_size: int,
                               prefetch_buffer: int = tf.data.AUTOTUNE,
                               num_parallel_calls: int = tf.data.AUTOTUNE,
                               cache: bool = False) -> tf.data.Dataset:
        """
        Create an optimized dataset for training.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size
            prefetch_buffer: Prefetch buffer size
            num_parallel_calls: Number of parallel calls
            cache: Whether to cache the dataset
            
        Returns:
            Optimized dataset
        """
        # Cache if requested
        if cache:
            dataset = dataset.cache()
        
        # Shuffle
        dataset = dataset.shuffle(buffer_size=1000)
        
        # Batch
        dataset = dataset.batch(batch_size)
        
        # Prefetch
        dataset = dataset.prefetch(prefetch_buffer)
        
        return dataset
    
    @staticmethod
    def setup_training_environment(gpu_memory_growth: bool = True,
                                 mixed_precision: bool = True,
                                 xla_compilation: bool = True) -> None:
        """
        Setup optimal training environment.
        
        Args:
            gpu_memory_growth: Whether to enable GPU memory growth
            mixed_precision: Whether to enable mixed precision
            xla_compilation: Whether to enable XLA compilation
        """
        print("Setting up training environment...")
        
        # Print system info
        TFOptimizationUtils.print_tf_info()
        
        # Configure GPU
        TFOptimizationUtils.optimize_tf_gpu(memory_growth=gpu_memory_growth)
        
        # Configure mixed precision
        if mixed_precision:
            TFOptimizationUtils.configure_mixed_precision()
        
        # Configure XLA
        if xla_compilation:
            TFOptimizationUtils.enable_xla_compilation()
        
        # Configure other optimizations
        TFOptimizationUtils.configure_tf_optimizations()
        
        print("Training environment setup complete!")


# Convenience functions
def optimize_tf_gpu(**kwargs) -> None:
    """Convenience function for GPU optimization."""
    return TFOptimizationUtils.optimize_tf_gpu(**kwargs)


def configure_mixed_precision(**kwargs) -> None:
    """Convenience function for mixed precision configuration."""
    return TFOptimizationUtils.configure_mixed_precision(**kwargs)


def setup_training_environment(**kwargs) -> None:
    """Convenience function for training environment setup."""
    return TFOptimizationUtils.setup_training_environment(**kwargs)
