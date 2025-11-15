# Augmentation Probability Configuration

This document describes the augmentation probabilities used in the training pipeline and the rationale behind each choice.

## Overview

Augmentation probabilities have been carefully tuned to balance data diversity with data quality. Aggressive augmentations that can make objects hard to distinguish have been reduced in frequency, while beneficial augmentations are applied more frequently.

## Augmentation Probabilities

### Image-Level Augmentations (Applied Per Image)

| Augmentation | Probability | Rationale |
|-------------|-------------|-----------|
| **Random Resize Crop Pad** | Always (100%) | Essential for scale and aspect ratio diversity. Applied to all images. |
| **Horizontal Flip** | 50% | Standard practice. Preserves object appearance while providing left-right invariance. |
| **Color Augmentations** | Always (100%) | Brightness, contrast, saturation, hue, grayscale. Non-destructive, always applied. |
| **Rotation** | **5%** (reduced from 10%) | Reduced to avoid excessive geometric distortion. Rotation can significantly alter object appearance and context, making objects harder to recognize. |
| **GridMask** | **10%** (reduced from 20%) | Reduced to balance regularization with data quality. GridMask can obscure important object features if applied too frequently. |

### Batch-Level Augmentations (Applied Per Batch)

| Augmentation | Probability | Rationale |
|-------------|-------------|-----------|
| **Mosaic** | **90%** (reduced from 100%) | Highly effective for object detection. Combines 4 images, increasing effective batch size and context diversity. Slightly reduced from 100% to allow some batches without Mosaic for training stability. |
| **MixUp** | **5%** (reduced from 15%) | Significantly reduced due to aggressive nature. MixUp blends two images, which can make objects hard to distinguish. Reduced from 15% to 5% to maintain data quality while still providing regularization benefits. |

## Changes Summary

### Reduced Probabilities (Less Aggressive)

1. **MixUp: 15% → 5%** (67% reduction)
   - **Reason**: Very aggressive augmentation that blends images, making objects hard to distinguish
   - **Impact**: Maintains regularization benefits while preserving data quality

2. **GridMask: 20% → 10%** (50% reduction)
   - **Reason**: Can obscure important object features if applied too frequently
   - **Impact**: Better balance between regularization and data quality

3. **Rotation: 10% → 5%** (50% reduction)
   - **Reason**: Can significantly alter object appearance and context
   - **Impact**: Reduces geometric distortion while maintaining some rotation invariance

4. **Mosaic: 100% → 90%** (10% reduction)
   - **Reason**: Allows some batches without Mosaic for training stability
   - **Impact**: Maintains high effectiveness while providing occasional variation

### Unchanged Probabilities

- **Horizontal Flip: 50%** - Standard practice, not aggressive
- **Color Augmentations: 100%** - Non-destructive, always beneficial
- **Random Resize Crop Pad: 100%** - Essential for scale diversity

## Expected Training Impact

### Positive Effects

1. **Better Data Quality**: Reduced frequency of aggressive augmentations means more training samples with clearly visible, recognizable objects
2. **Improved Learning**: Model can learn from cleaner examples while still benefiting from augmentation diversity
3. **Training Stability**: Less aggressive augmentations reduce the risk of confusing the model with overly distorted samples

### Trade-offs

1. **Slightly Less Regularization**: Lower probabilities mean less aggressive regularization, but this is balanced by maintaining high-quality training data
2. **Reduced Augmentation Diversity**: Some augmentations occur less frequently, but the most beneficial ones (Mosaic, horizontal flip) remain at high frequencies

## Recommendations

### For Standard Training
- Use the current probabilities as configured
- These settings balance augmentation benefits with data quality

### For Challenging Datasets
- If overfitting occurs, consider:
  - Increasing GridMask to 15%
  - Increasing Rotation to 8%
  - Increasing MixUp to 8%

### For High-Quality Datasets
- If underfitting occurs, consider:
  - Reducing GridMask to 5%
  - Reducing Rotation to 3%
  - Reducing MixUp to 3% or disabling it

## Implementation Details

Probabilities are configured in:
- `multigriddet/data/generators.py`:
  - Line ~1529: `tf_random_rotate(..., prob=0.05)`
  - Line ~1534: `tf_random_gridmask(..., prob=0.1)`
  - Line ~1565: `tf_random_mosaic(..., prob=0.9)`
  - Line ~1571: `tf_random_mixup(..., prob=0.05, alpha=0.2)`

## Monitoring

During training, monitor:
1. **Training loss**: Should decrease smoothly without excessive noise
2. **Validation metrics**: Should improve steadily
3. **Visual inspection**: Sample augmented images should show recognizable objects

If objects become too hard to distinguish in augmented samples, further reduce aggressive augmentation probabilities.

