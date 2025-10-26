#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Inference Script
Run inference on images, videos, or camera with YAML configuration.

Usage:
    # Image inference
    python infer.py --config configs/infer_config.yaml --input examples/images/dog.jpg
    
    # Video inference
    python infer.py --config configs/infer_config.yaml --input video.mp4 --type video
    
    # Camera inference
    python infer.py --config configs/infer_config.yaml --type camera
    
    # Directory inference
    python infer.py --config configs/infer_config.yaml --input images/ --type directory
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multigriddet.config import ConfigLoader
from multigriddet.inference import MultiGridInference


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='MultiGridDet inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/infer_config.yaml',
        help='Path to inference config file'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default=None,
        help='Input source (overrides config)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default=None,
        help='Model weights path (overrides config)'
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['image', 'video', 'camera', 'directory'],
        default=None,
        help='Input type (overrides config)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold (overrides config)'
    )
    parser.add_argument(
        '--nms',
        type=float,
        default=None,
        help='NMS threshold (overrides config)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not show output'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 80)
    print("MultiGridDet Inference")
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
    if args.input:
        config['input']['source'] = args.input
        print(f"   Input: {args.input}")
    
    if args.output:
        config['output']['output_dir'] = args.output
        print(f"   Output: {args.output}")
    
    if args.weights:
        config['weights_path'] = args.weights
        print(f"   Weights: {args.weights}")
    
    if args.type:
        config['input']['type'] = args.type
        print(f"   Type: {args.type}")
    
    if args.conf is not None:
        config['detection']['confidence_threshold'] = args.conf
        print(f"   Confidence threshold: {args.conf}")
    
    if args.nms is not None:
        config['detection']['nms_threshold'] = args.nms
        print(f"   NMS threshold: {args.nms}")
    
    if args.no_save:
        config['output']['save_result'] = False
    
    if args.no_show:
        config['output']['show_result'] = False
    
    print()
    
    # Create inference engine
    try:
        inference = MultiGridInference(config)
    except Exception as e:
        print(f"[ERROR] Error creating inference engine: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run inference
    try:
        inference.run()
        return 0
    except KeyboardInterrupt:
        print("\n\n[WARNING] Inference interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Inference error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())





