# Staged Training Guide: How to Use Multi-Stage Fine-Tuning

## Current Trainer Support

The trainer supports **two-stage training** automatically:
- **Stage 1**: Frozen layers (based on `freeze_level` in config)
- **Stage 2**: Unfreeze all layers

For **three-stage training** (freeze_level: 2 → 1 → 0), you need to run training **multiple times** with different configs, resuming from checkpoints.

## How Staged Training Works

### Automatic Two-Stage Training

The trainer automatically handles:
1. **Stage 1**: Trains with frozen layers (for `transfer_epochs` epochs)
2. **Stage 2**: Automatically unfreezes all layers and continues training

**Config for two-stage:**
```yaml
training:
  transfer_epochs: 5      # Stage 1 runs for 5 epochs
  freeze_level: 1        # Freeze backbone only (or 2 for freeze all but head)
  learning_rate: 0.0001
  # ... rest of config
```

**What happens:**
- Model is built with `freeze_level: 1` (backbone frozen)
- Stage 1: Trains for 5 epochs with frozen backbone
- Stage 2: Automatically unfreezes all layers and continues to `epochs: 100`

### Manual Three-Stage Training

For three-stage training (recommended for model5.h5), you need to run training **three times**:

#### Stage 1: Freeze All But Head (5 epochs)
```yaml
training:
  transfer_epochs: 5
  freeze_level: 2        # Freeze all but head
  learning_rate: 0.0001
  initial_epoch: 0
  epochs: 5
augmentation:
  mosaic_prob: 0.0       # DISABLED
  mixup_prob: 0.0        # DISABLED
```

**Run:**
```bash
python train.py --config configs/train_config.yaml
```

#### Stage 2: Freeze Backbone Only (5-10 epochs)
```yaml
training:
  transfer_epochs: 10     # Total including Stage 1
  freeze_level: 1         # Freeze backbone only
  learning_rate: 0.00005  # Lower LR
  initial_epoch: 5        # Resume from Stage 1
  epochs: 10
augmentation:
  mosaic_prob: 0.3        # ENABLED
  mixup_prob: 0.1         # ENABLED
resume:
  enabled: true
  weights_path: logs/checkpoints/best_model.h5  # Resume from Stage 1
```

**Run:**
```bash
python train.py --config configs/train_config.yaml
```

#### Stage 3: Unfreeze All (remaining epochs)
```yaml
training:
  transfer_epochs: 0      # Disable transfer stage
  freeze_level: 0         # Unfreeze all
  learning_rate: 0.00003  # Lowest LR
  initial_epoch: 10       # Resume from Stage 2
  epochs: 100
augmentation:
  mosaic_prob: 0.3
  mixup_prob: 0.1
resume:
  enabled: true
  weights_path: logs/checkpoints/best_model.h5  # Resume from Stage 2
```

**Run:**
```bash
python train.py --config configs/train_config.yaml
```

## Important Notes

1. **Model is frozen during building**: The `freeze_level` is applied when the model is built, not in the trainer. The trainer respects this state.

2. **Resume between stages**: You must resume from the checkpoint of the previous stage. Set `resume.enabled: true` and `resume.weights_path` to the best model from the previous stage.

3. **Initial epoch**: Set `initial_epoch` to the last epoch of the previous stage (e.g., if Stage 1 ended at epoch 5, Stage 2 starts with `initial_epoch: 5`).

4. **Transfer epochs**: In Stage 2, set `transfer_epochs` to the total epochs including Stage 1 (e.g., if Stage 1 was 5 epochs, Stage 2 should have `transfer_epochs: 10`). In Stage 3, set `transfer_epochs: 0` to disable the transfer stage.

5. **Learning rate**: Update `learning_rate` for each stage (1e-4 → 5e-5 → 3e-5).

6. **Augmentation**: Disable during Stage 1 (frozen layers), enable during Stages 2-3.

## Example Workflow Script

You can create a script to automate the three-stage process:

```python
#!/usr/bin/env python3
"""Three-stage training workflow."""

import yaml
import subprocess
import os

config_path = "configs/train_config.yaml"

# Stage 1: Freeze all but head
print("=" * 80)
print("Stage 1: Freeze all but head (5 epochs)")
print("=" * 80)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config['training']['transfer_epochs'] = 5
config['training']['freeze_level'] = 2
config['training']['learning_rate'] = 0.0001
config['training']['initial_epoch'] = 0
config['training']['epochs'] = 5
config['training']['augmentation']['mosaic_prob'] = 0.0
config['training']['augmentation']['mixup_prob'] = 0.0
config['resume']['enabled'] = False
with open(config_path, 'w') as f:
    yaml.dump(config, f)
subprocess.run(["python", "train.py", "--config", config_path])

# Stage 2: Freeze backbone only
print("=" * 80)
print("Stage 2: Freeze backbone only (5 more epochs)")
print("=" * 80)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config['training']['transfer_epochs'] = 10
config['training']['freeze_level'] = 1
config['training']['learning_rate'] = 0.00005
config['training']['initial_epoch'] = 5
config['training']['epochs'] = 10
config['training']['augmentation']['mosaic_prob'] = 0.3
config['training']['augmentation']['mixup_prob'] = 0.1
config['resume']['enabled'] = True
config['resume']['weights_path'] = "logs/checkpoints/best_model.h5"
with open(config_path, 'w') as f:
    yaml.dump(config, f)
subprocess.run(["python", "train.py", "--config", config_path])

# Stage 3: Unfreeze all
print("=" * 80)
print("Stage 3: Unfreeze all (remaining epochs)")
print("=" * 80)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config['training']['transfer_epochs'] = 0
config['training']['freeze_level'] = 0
config['training']['learning_rate'] = 0.00003
config['training']['initial_epoch'] = 10
config['training']['epochs'] = 100
config['resume']['enabled'] = True
config['resume']['weights_path'] = "logs/checkpoints/best_model.h5"
with open(config_path, 'w') as f:
    yaml.dump(config, f)
subprocess.run(["python", "train.py", "--config", config_path])
```

## Summary

- **Two-stage**: Automatic (set `transfer_epochs` and `freeze_level`, trainer handles the rest)
- **Three-stage**: Manual (run training three times with different configs, resuming from checkpoints)

The trainer respects the `freeze_level` set during model building and automatically transitions from Stage 1 to Stage 2. For Stage 3 (or intermediate stages), you need to manually update the config and resume training.

