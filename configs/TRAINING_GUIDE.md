# Training Guide: Fine-Tuning Without "Untraining" Pretrained Weights

## Overview

This guide provides best practices for fine-tuning MultiGridDet models on A100 80GB GPUs while preserving pretrained weights. The key principle: **gradual adaptation with conservative learning rates and staged unfreezing**.

## Critical Fix: Layer Name Matching

**IMPORTANT**: The codebase now automatically calls `tf.keras.backend.clear_session()` before building models when loading pretrained weights. This ensures layer names match the weights file (prevents silent loading failures). This is handled automatically - no action needed.

## Two Training Scenarios

### Scenario 1: Fine-Tuning from Full Model Weights (model5.h5)

**Use case**: You have a fully trained model (e.g., `model5.h5`) and want to fine-tune on your dataset.

**Strategy**: Three-stage progressive unfreezing with conservative learning rates.

#### Stage 1: Freeze All But Head (5 epochs)
```yaml
transfer_epochs: 5
freeze_level: 2          # Freeze all but head
learning_rate: 0.0001    # 1e-4 (conservative)
mosaic_prob: 0.0         # DISABLED - frozen trunk needs clean boxes
mixup_prob: 0.0          # DISABLED
```

**Rationale**: 
- Head adapts to new data distribution without touching backbone/neck
- No augmentation prevents "untraining" frozen layers
- Conservative LR prevents drift

#### Stage 2: Freeze Backbone Only (5-10 epochs)
```yaml
# After Stage 1, manually update config:
transfer_epochs: 10       # Total epochs including Stage 1
freeze_level: 1           # Freeze backbone only
learning_rate: 0.00005    # 5e-5 (slightly lower)
mosaic_prob: 0.3          # ENABLED - moderate augmentation
mixup_prob: 0.1           # ENABLED - conservative mixup
```

**Rationale**:
- Neck/head adapt while backbone stays stable
- Moderate augmentation helps generalization
- Lower LR prevents backbone drift

#### Stage 3: Unfreeze All (remaining epochs)
```yaml
# After Stage 2, manually update config:
transfer_epochs: 0        # Disable transfer stage
freeze_level: 0           # Unfreeze all
learning_rate: 0.00003    # 3e-5 (lowest, prevents untraining)
mosaic_prob: 0.3          # ENABLED
mixup_prob: 0.1           # ENABLED
```

**Rationale**:
- Full fine-tuning with very low LR
- Augmentation helps with generalization
- Low LR ensures pretrained weights don't drift significantly

### Scenario 2: Training from Backbone-Only Weights

**Use case**: You have backbone weights (e.g., `darknet53.h5`) and need to train the head from scratch.

**Strategy**: Two-stage training with slightly higher learning rates.

#### Stage 1: Freeze Backbone (10 epochs)
```yaml
transfer_epochs: 10
freeze_level: 1           # Freeze backbone
learning_rate: 0.0001     # 1e-4
mosaic_prob: 0.3           # ENABLED - head needs diversity
mixup_prob: 0.1           # ENABLED
```

**Rationale**:
- Randomly initialized head needs training data diversity
- Backbone stays stable while head learns
- Standard LR for head training

#### Stage 2: Unfreeze All (remaining epochs)
```yaml
# After Stage 1, manually update config:
transfer_epochs: 0
freeze_level: 0
learning_rate: 0.0002     # 2e-4 (brief boost for head catch-up)
# After 1-2 epochs, cosine schedule will decay naturally
mosaic_prob: 0.3
mixup_prob: 0.1
```

**Rationale**:
- Brief LR boost helps head catch up
- Cosine schedule handles decay automatically
- Full training with stable backbone

## Key Recommendations

### Learning Rates
- **Never exceed 1e-4 for full model fine-tuning** - Higher LRs cause "untraining"
- **Use 3e-5 for final stage** - Lowest safe LR for full fine-tuning
- **For backbone-only**: 1e-4 → 2e-4 briefly, then cosine decay

### Batch Size
- **A100 80GB with 608x608**: 48-64 batch size (48 is safer)
- Larger batches = more stable gradients = less risk of untraining
- Monitor GPU memory usage and adjust accordingly

### Loss Scales
**DO NOT CHANGE** these values - they match model5.h5 training recipe:
```yaml
coord_scale: 5.0
object_scale: 1.0
no_object_scale: 0.5
class_scale: 1.0
anchor_scale: 1.0
```
Large deviations risk "untraining" because the model was trained with these specific scales.

### Loss Normalization
- **Start with `["batch"]` only** - Prevents pretrained head from being shocked
- **Add `["positives"]` later** if Mosaic injects many objects and gradients become unstable
- **Never use `["grid"]` during fine-tuning** - Changes loss scaling too dramatically

### Augmentation Strategy
- **Frozen stages**: Disable Mosaic/MixUp (frozen layers need clean boxes)
- **Unfrozen stages**: Enable moderate augmentation (0.3 Mosaic, 0.1 MixUp)
- **Never use aggressive augmentation** during fine-tuning (risks untraining)

### Monitoring
- **Watch TensorBoard logs** for objectness/class losses
- **They should stay within ±20% of pretrained values** initially
- **If losses spike or diverge**: Stop, reduce LR, or add more frozen epochs

### Stability Tips
1. **One knob at a time**: Change only LR, batch size, or augmentation - not all at once
2. **Gradual unfreezing**: Never jump from freeze_level: 2 → 0 directly
3. **Conservative LRs**: Better to train longer with low LR than risk untraining
4. **Monitor validation loss**: Should decrease smoothly, not spike
5. **Early stopping**: Use patience=10 to catch divergence early

## Example Training Workflow

### For model5.h5:
```bash
# Stage 1: Freeze all but head (5 epochs)
# Edit config: transfer_epochs=5, freeze_level=2, LR=1e-4, mosaic=0, mixup=0
python train.py --config configs/train_config.yaml

# Stage 2: Freeze backbone only (5-10 epochs)
# Edit config: transfer_epochs=10, freeze_level=1, LR=5e-5, mosaic=0.3, mixup=0.1
# Resume from Stage 1 checkpoint
python train.py --config configs/train_config.yaml --resume logs/checkpoints/best_model.h5

# Stage 3: Unfreeze all (remaining epochs)
# Edit config: transfer_epochs=0, freeze_level=0, LR=3e-5, mosaic=0.3, mixup=0.1
# Resume from Stage 2 checkpoint
python train.py --config configs/train_config.yaml --resume logs/checkpoints/best_model.h5
```

### For backbone-only:
```bash
# Stage 1: Freeze backbone (10 epochs)
# Edit config: transfer_epochs=10, freeze_level=1, LR=1e-4, mosaic=0.3, mixup=0.1
python train.py --config configs/train_config.yaml

# Stage 2: Unfreeze all (remaining epochs)
# Edit config: transfer_epochs=0, freeze_level=0, LR=2e-4, mosaic=0.3, mixup=0.1
# Resume from Stage 1 checkpoint
python train.py --config configs/train_config.yaml --resume logs/checkpoints/best_model.h5
```

## Common Pitfalls

1. **Too high LR**: Causes immediate "untraining" - pretrained weights drift
2. **Skipping stages**: Jumping from freeze_level: 2 → 0 causes instability
3. **Aggressive augmentation during frozen stages**: "Untrains" frozen layers
4. **Changing loss scales**: Breaks the training recipe that produced pretrained weights
5. **Not monitoring**: Loss divergence goes unnoticed until too late

## Validation

After training, verify:
- Validation loss is lower than random initialization
- Objectness/class losses are within ±20% of pretrained values
- Model performance matches or exceeds pretrained baseline
- No NaN or infinite losses during training

## References

- Loss fixes: See `multigriddet/losses/multigrid_loss.py` for ignore-mask and anchor scaling fixes
- Weight loading: `clear_session()` is automatically called to ensure layer name matching
- Training pipeline: `multigriddet/trainers/trainer.py` handles staged training automatically

