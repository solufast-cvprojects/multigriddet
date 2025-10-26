#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Evaluation Script
Evaluate model performance on validation dataset.

Usage:
    python eval.py --config configs/eval_config.yaml
    python eval.py --config configs/eval_config.yaml --weights weights/model5.h5
    python eval.py --config configs/eval_config.yaml --data data/coco_val2017.txt
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multigriddet.config import ConfigLoader
from multigriddet.evaluation import MultiGridEvaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate MultiGridDet model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/eval_config.yaml',
        help='Path to evaluation config file'
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default=None,
        help='Model weights path (overrides config)'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default=None,
        help='Annotation file path (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold (overrides config)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to evaluate (for testing)'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 80)
    print("MultiGridDet Evaluation")
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
        config['weights_path'] = args.weights
        print(f"   Weights: {args.weights}")
    
    if args.data:
        config['data']['annotation'] = args.data
        print(f"   Data: {args.data}")
    
    if args.batch_size:
        config['evaluation']['batch_size'] = args.batch_size
        print(f"   Batch size: {args.batch_size}")
    
    if args.conf is not None:
        config['evaluation']['confidence_threshold'] = args.conf
        print(f"   Confidence threshold: {args.conf}")
    
    if args.max_images is not None:
        config['evaluation']['max_images'] = args.max_images
        print(f"   Max images: {args.max_images}")
    
    print()
    
    # Create evaluator
    try:
        evaluator = MultiGridEvaluator(config)
    except Exception as e:
        print(f"[ERROR] Error creating evaluator: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run evaluation
    try:
        results = evaluator.evaluate()
        
        # Print results
        evaluator.print_results(results)
        
        return 0
    except KeyboardInterrupt:
        print("\n\n[WARNING] Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())





