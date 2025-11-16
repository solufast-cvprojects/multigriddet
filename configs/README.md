# Configuration Files

This directory contains configuration templates for MultiGridDet. The actual configuration files are **not tracked in git** to allow for local customization without conflicts.

## Setup

1. **Copy the example files to create your local configs:**
   ```bash
   cp configs/train_config.yaml.example configs/train_config.yaml
   cp configs/eval_config.yaml.example configs/eval_config.yaml
   cp configs/infer_config.yaml.example configs/infer_config.yaml
   # ... and so on for other configs you need
   ```

2. **Edit the local config files** (`train_config.yaml`, `eval_config.yaml`, etc.) to match your environment and preferences.

3. **The local config files are ignored by git**, so you can modify them freely without worrying about conflicts when pulling/pushing.

## Available Configuration Files

### Training
- **`train_config.yaml.example`** - Main training configuration
  - Optimized for fine-tuning from pretrained weights (model5.h5)
  - Can be adjusted for from-scratch training
  - Uses Adam optimizer (simpler than AdamW)
  - Includes cosine annealing with warmup

### Evaluation
- **`eval_config.yaml.example`** - Full evaluation configuration
- **`eval_config_fast.yaml.example`** - Fast evaluation (fewer metrics)
- **`eval_config_full.yaml.example`** - Complete evaluation with all metrics
- **`eval_config_test.yaml.example`** - Testing configuration
- **`eval_config_test_viz.yaml.example`** - Testing with visualizations

### Inference
- **`infer_config.yaml.example`** - Inference/prediction configuration

## Quick Start

For training:
```bash
# Copy the example
cp configs/train_config.yaml.example configs/train_config.yaml

# Edit as needed (especially for RunPod)
# - Set weights_path if using pretrained model
# - Adjust learning_rate (0.0001 for fine-tuning, 0.001 for from-scratch)
# - Set freeze_level (1 for fine-tuning, 0 for from-scratch)
```

## Notes

- All `.example` files are tracked in git and contain recommended defaults
- All actual config files (without `.example`) are ignored by git
- This allows you to customize configs on RunPod or other environments without git conflicts
- When pulling updates, your local configs won't be overwritten

