#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Training Script
Train object detection models with YAML configuration.

Usage:
    python train.py --config configs/train_config.yaml
    python train.py --config configs/train_config.yaml --weights weights/model5.h5
    python train.py --config configs/train_config.yaml --resume
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multigriddet.config import ConfigLoader
from multigriddet.trainers import MultiGridTrainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MultiGridDet model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/train_config.yaml',
        help='Path to training config file'
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default=None,
        help='Path to pretrained weights (overrides config)'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 80)
    print("MultiGridDet Training")
    print("=" * 80)
    print(f"Config file: {args.config}")
    
    # Load configuration
    try:
        config = ConfigLoader.load_config(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"   Please check that the config file exists")
        return 1
    except Exception as e:
        print(f"[ERROR] Error loading config: {e}")
        return 1
    
    # Override with command-line args
    if args.weights:
        if 'resume' not in config:
            config['resume'] = {}
        config['resume']['weights_path'] = args.weights
        print(f"   Using weights: {args.weights}")
    
    if args.resume:
        if 'resume' not in config:
            config['resume'] = {}
        config['resume']['enabled'] = True
        print(f"   Resume mode enabled")
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
        print(f"   Epochs: {args.epochs}")
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        print(f"   Batch size: {args.batch_size}")
    
    print()
    
    # Create trainer
    try:
        trainer = MultiGridTrainer(config)
    except Exception as e:
        print(f"[ERROR] Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Start training
    try:
        trainer.train()
        return 0
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())





