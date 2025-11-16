# Augmentation Pipeline Capacity and Box Preservation

This document explains the critical fixes to the augmentation pipeline that prevent fine-tuning from "untraining" pretrained weights.

## Problem Summary

The previous implementation had several critical issues:

1. **Variable-length box tensors**: `padded_batch` used `[None, 5]` for boxes, causing inconsistent batch shapes
2. **Insufficient capacity expansion**: Only accounted for Mosaic (4×) or MixUp (2×) separately, not both (8×)
3. **Unconditional MixUp**: MixUp always ran even when `mixup_prob=0`, changing data distribution
4. **Silent label truncation**: Boxes exceeding capacity were silently truncated, dropping up to 75% of labels
5. **No fixed capacity**: No configurable `max_boxes_per_image` to ensure consistent shapes

These issues caused `val_loss` to diverge during fine-tuning, effectively "untraining" pretrained weights.

## Solution Overview

### 1. Fixed-Capacity Padding

**Configuration**: `training.augmentation.max_boxes_per_image` (default: 100)

**Implementation**: `padded_batch` now uses fixed shape `[max_boxes_per_image, 5]` instead of `[None, 5]`

**Why it matters**: Every batch has the same box tensor shape `[batch_size, max_boxes_per_image, 5]`, preventing shape inconsistencies that corrupt training.

**Code location**: `multigriddet/data/generators.py:1617-1629`

### 2. Capacity Expansion Logic

**Implementation**: `_expand_box_capacity` function detects enabled augmentations and expands capacity accordingly:

- **Mosaic only**: 4× capacity (merges 4 images)
- **MixUp only**: 2× capacity (merges 2 images)
- **Mosaic + MixUp**: 8× capacity (can combine up to 8 source images: 4 from Mosaic × 2 from MixUp)
- **Neither**: 1× capacity (no expansion needed)

**Why it matters**: Explicit expansion ensures sufficient capacity for all merged boxes. The previous implementation only expanded to 4× when Mosaic was enabled, causing truncation when MixUp was also applied.

**Code location**: `multigriddet/data/generators.py:1640-1689`

### 3. Augmentation Toggles

**Implementation**: `_apply_batch_augmentations` only applies augmentations when their probabilities are > 0:

- Mosaic: Only applies if `enhance_augment == 'mosaic'` AND `mosaic_prob > 0`
- MixUp: Only applies if `mixup_prob > 0`

**Why it matters**: Fine-tuning can disable augmentations (e.g., `mixup_prob=0`) to match the data distribution of pretrained checkpoints, preventing "untraining" of weights.

**Code location**: `multigriddet/data/generators.py:1699-1720`

### 4. No Label Truncation

**Implementation**: `tf_random_mosaic` and `tf_random_mixup` use explicit padding, never truncation. If capacity is exceeded, a warning is logged and boxes are truncated as a last resort (this should never happen with proper expansion).

**Why it matters**: All ground-truth boxes are preserved unless physically cropped out of the image bounds. The previous implementation silently dropped up to 75% of labels per composite.

**Code locations**: 
- `multigriddet/data/generators.py:677-708` (Mosaic)
- `multigriddet/data/generators.py:885-916` (Mosaic batch processing)
- `multigriddet/data/generators.py:1040-1072` (MixUp)

## Configuration Examples

### Fine-Tuning (Minimal Augmentation)

```yaml
training:
  augmentation:
    enabled: true
    enhance_type: mosaic
    mosaic_prob: 0.3      # Conservative for fine-tuning
    mixup_prob: 0.0       # Disabled to match pretrained checkpoint distribution
    max_boxes_per_image: 100
```

### Full Training (Aggressive Augmentation)

```yaml
training:
  augmentation:
    enabled: true
    enhance_type: mosaic
    mosaic_prob: 0.9      # High probability
    mixup_prob: 0.15      # Moderate probability
    max_boxes_per_image: 100  # Will expand to 800 (8×) when both apply
```

### No Batch Augmentations

```yaml
training:
  augmentation:
    enabled: true
    enhance_type: null
    mosaic_prob: 0.0
    mixup_prob: 0.0
    max_boxes_per_image: 100  # No expansion needed
```

## Testing

Run the regression tests to verify the fixes:

```bash
python tests/test_augmentation_capacity.py
```

The test suite verifies:
1. Fixed-capacity padding: All batches have shape `[batch_size, max_boxes_per_image, 5]`
2. Capacity expansion: 4× for Mosaic, 2× for MixUp, 8× for both
3. Augmentation toggles: MixUp and Mosaic only apply when probabilities > 0
4. Box preservation: Box counts never decrease due to truncation

## Key Code Locations

1. **Fixed-capacity padding**: `multigriddet/data/generators.py:1617-1629`
2. **Capacity expansion**: `multigriddet/data/generators.py:1640-1689`
3. **Augmentation toggles**: `multigriddet/data/generators.py:1699-1720`
4. **Mosaic function**: `multigriddet/data/generators.py:499-920`
5. **MixUp function**: `multigriddet/data/generators.py:922-1075`
6. **Generator initialization**: `multigriddet/data/generators.py:1265-1318`
7. **Trainer setup**: `multigriddet/trainers/trainer.py:167-184`

## Impact

These fixes ensure:
- **Consistent batch shapes**: Every batch has the same tensor shapes, preventing shape-related training issues
- **No label loss**: All ground-truth boxes are preserved unless physically cropped out
- **Proper fine-tuning**: Augmentations can be disabled to match pretrained checkpoint distributions
- **Production-grade pipeline**: Vectorized, GPU-friendly implementation with proper error handling

The previous implementation dropped up to 75% of labels per composite image, causing `val_loss` to diverge during fine-tuning. These fixes eliminate that issue entirely.

