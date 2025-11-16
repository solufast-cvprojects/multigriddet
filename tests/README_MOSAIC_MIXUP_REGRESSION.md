# Mosaic/MixUp Augmentation Regression Test

This test script provides automated verification and visual validation for the Mosaic and MixUp augmentation pipelines. These batch-level augmentations combine multiple training images, and this test ensures that all bounding boxes are preserved without truncation and that coordinate transformations are correct.

## Purpose

This regression test was created to verify a critical fix: the previous implementation silently truncated boxes when composite images contained more boxes than a single source image, discarding up to 75% of objects per composite and corrupting supervision signals. This test verifies that:

1. **No box truncation**: All valid boxes from source images are preserved in composite images
2. **Correct coordinate transformations**: Boxes are correctly positioned in Mosaic and MixUp outputs
3. **Box count preservation**: MixUp preserves all boxes (exact count match), Mosaic preserves boxes after filtering (accounting for geometric filtering)

## Critical Background

The old implementation truncated annotations to `max_boxes_per_image = tf.shape(boxes)[1]`, which caused:
- Mosaic (4 images): Up to 75% of objects silently dropped when composites exceeded single-image capacity
- MixUp (2 images): Up to 50% of objects silently dropped
- Training corruption: Even strong pretrained checkpoints were "untrained" due to missing supervision signals
- Validation loss divergence: Missing boxes caused loss to diverge during training

The fix allocates dynamic capacity (4× for Mosaic, 2× for MixUp) before applying augmentations, ensuring all boxes are preserved.

## Usage

### Basic Usage

```bash
python tests/test_mosaic_mixup_regression.py \
    --annotation data/coco_train2017.txt \
    --num-tests 5
```

### With Custom Settings

```bash
python tests/test_mosaic_mixup_regression.py \
    --annotation data/coco_train2017.txt \
    --num-tests 10 \
    --seed 42 \
    --input-shape 640 640 \
    --output-dir tests/augmentation_test_outputs
```

### Using Conda Environment

```bash
conda run -n cv_modern python tests/test_mosaic_mixup_regression.py \
    --annotation data/coco_train2017.txt \
    --num-tests 5 \
    --seed 42
```

## Arguments

- `--annotation`: Path to annotation file (required)
- `--num-tests`: Number of test cases to run for each augmentation (default: 5)
- `--seed`: Random seed for deterministic testing (default: 42)
- `--input-shape`: Input image shape as height width (default: 640 640)
- `--output-dir`: Directory to save visualizations (default: tests/augmentation_test_outputs)

## Output

The script generates visualizations for both augmentations:

### Mosaic Test Outputs
- Location: `{output_dir}/mosaic/mosaic_test_XXX.png`
- Shows: Composite mosaic image with all boxes drawn (green boxes with class labels)
- Verifies: Box alignment with objects, correct coordinate transformations

### MixUp Test Outputs
- Location: `{output_dir}/mixup/mixup_test_XXX.png`
- Shows: Blended image with all boxes from both source images (green boxes with class labels)
- Verifies: All boxes preserved, correct class labels maintained

## Test Validation

### Mosaic Tests

The test verifies:
- Box count after Mosaic >= expected minimum (50% of source boxes, accounting for geometric filtering)
- Boxes are correctly positioned in their respective quadrants
- No unexpected truncation (box count should not hit capacity limit unexpectedly)

**Note**: Mosaic filters boxes based on crop overlap and minimum size, so some reduction in box count is expected and normal. The critical check is that boxes are not truncated due to capacity limits.

### MixUp Tests

The test verifies:
- Box count after MixUp == total source boxes (exact match required)
- All boxes from both source images are preserved
- Boxes are correctly positioned (no coordinate transformation needed for MixUp)

**Note**: MixUp concatenates boxes from both images without filtering, so the count must match exactly.

## Visual Inspection Checklist

When reviewing the output images, verify:

### Mosaic Images
- [ ] **Box alignment**: Boxes align correctly with objects in their respective quadrants
- [ ] **Quadrant positioning**: Boxes from each quadrant are in the correct positions
- [ ] **Boundary checks**: No boxes are outside image boundaries
- [ ] **Class labels**: Boxes maintain correct class labels
- [ ] **Coordinate transformations**: Boxes from quadrants 1-3 are correctly positioned

### MixUp Images
- [ ] **Box alignment**: Boxes align correctly with objects (both source images' boxes visible)
- [ ] **Box count**: All boxes from both source images are present
- [ ] **Class labels**: Class labels are correct and not mixed up
- [ ] **No offset issues**: Boxes are positioned correctly (no systematic offset)

## Example Output Structure

```
tests/augmentation_test_outputs/
├── mosaic/
│   ├── mosaic_test_001.png
│   ├── mosaic_test_002.png
│   ├── mosaic_test_003.png
│   ├── mosaic_test_004.png
│   └── mosaic_test_005.png
└── mixup/
    ├── mixup_test_001.png
    ├── mixup_test_002.png
    ├── mixup_test_003.png
    ├── mixup_test_004.png
    └── mixup_test_005.png
```

## Interpreting Results

### Good Results
- **Mosaic**: Boxes align with objects, box count is reasonable (some filtering expected)
- **MixUp**: All boxes preserved (exact count match), boxes align with objects, class labels correct
- Test script reports "PASS" for all checks

### Bad Results (Need Investigation)
- **Mosaic**: Boxes misaligned with objects, boxes in wrong quadrants, unexpected box count drops
- **MixUp**: Box count mismatch, boxes misaligned, class labels incorrect
- Test script reports "FAIL" or "WARNING" messages

## Integration with CI/CD

For automated testing, add this to your test suite:

```bash
# Run regression tests and check exit code
python tests/test_mosaic_mixup_regression.py \
    --annotation data/coco_train2017.txt \
    --num-tests 3 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "Mosaic/MixUp regression tests passed"
else
    echo "Mosaic/MixUp regression tests failed"
    exit 1
fi
```

## Troubleshooting

### No boxes visible in visualizations
- Check that annotation file has valid boxes
- Verify image paths in annotation file are correct
- Ensure boxes are in correct format: `x1,y1,x2,y2,class`
- Verify images are loaded correctly (check for file path errors)

### Boxes misaligned in Mosaic
- This indicates a bug in coordinate transformation
- Check that crop offsets are correctly applied to quadrants 1-3
- Verify that boxes maintain absolute coordinates in final image

### Box count mismatch in MixUp
- MixUp must preserve all boxes - any mismatch indicates truncation
- Check that box capacity is expanded to 2× before MixUp
- Verify that no truncation logic remains in MixUp implementation

### Mosaic box count lower than expected
- Some reduction is normal due to geometric filtering (crop overlap, minimum size)
- If reduction is excessive (>50%), check filtering thresholds
- Verify that boxes are not being truncated due to capacity limits

## Technical Details

### Box Capacity Expansion

Before applying batch augmentations, box capacity is expanded:
- **Mosaic**: 4× capacity (merges 4 images)
- **MixUp**: 2× capacity (merges 2 images)
- **Both enabled**: 4× capacity (maximum of both)

This prevents truncation that corrupted supervision signals in the old implementation.

### Coordinate Transformations

**Mosaic**: Boxes maintain their absolute coordinates in the final image coordinate system. When crops are placed using `tf.concat`, they are positioned at their absolute locations, so boxes need only be clipped to boundaries, not shifted.

**MixUp**: No coordinate transformation needed - images are blended and boxes are concatenated directly.

## Related Files

- `multigriddet/data/generators.py`: Contains `tf_random_mosaic()` and `tf_random_mixup()` implementations
- `configs/train_config.yaml`: Configuration for augmentation probabilities (`mosaic_prob`, `mixup_prob`)
- `tests/test_augmentations.py`: General augmentation test suite

