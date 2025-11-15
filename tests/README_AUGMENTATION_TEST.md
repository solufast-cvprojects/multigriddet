# Augmentation Regression Test Suite

This comprehensive test suite verifies that all data augmentation operations correctly
handle bounding box coordinate transformations when applied to images. The tests provide
automated validation and visual verification to ensure augmentations maintain geometric
consistency between images and their associated bounding boxes.

## Purpose

Before training on expensive cloud instances (like RunPod), it's critical to verify that:
1. Box coordinates are correctly transformed by each augmentation
2. Boxes align properly with objects after augmentation
3. No boxes are lost or incorrectly positioned during augmentation
4. Boxes remain within image boundaries

## Tested Augmentations

### Geometric Augmentations (Critical - Affect Box Coordinates)
- **Horizontal Flip**: Mirrors images and bounding boxes along the vertical axis
- **Rotation**: Applies discrete 90/180/270 degree rotations with corresponding box transformations
- **Random Resize Crop Pad**: Performs aspect ratio jittering, scaling, cropping, and padding
    with proper coordinate adjustments
- **GridMask**: Applies grid-based masking pattern (may filter boxes if they become too small
    after augmentation)

### Batch-Level Augmentations
- **MixUp**: Blends two images and concatenates their bounding boxes
    - Requires batch processing (minimum 2 images)
    - Mixing ratio sampled from Beta distribution

### Color Augmentations (Non-Geometric)
- **Color Augmentations**: Brightness, contrast, saturation, hue adjustments, and grayscale
    - These operations should NOT affect box coordinates (tested for completeness)
    - Verifies that color-only augmentations preserve geometric properties

## Usage

### Test All Augmentations

```bash
python tests/test_augmentations.py \
    --annotation data/coco_train2017.txt \
    --aug all \
    --num-tests 3
```

### Test Specific Augmentation

```bash
# Test horizontal flip only
python tests/test_augmentations.py \
    --annotation data/coco_train2017.txt \
    --aug horizontal_flip \
    --num-tests 5

# Test rotation only
python tests/test_augmentations.py \
    --annotation data/coco_train2017.txt \
    --aug rotation \
    --num-tests 5
```

### With Custom Settings

```bash
python tests/test_augmentations.py \
    --annotation data/coco_train2017.txt \
    --aug all \
    --num-tests 5 \
    --seed 42 \
    --input-shape 608 608 \
    --output-dir tests/my_aug_tests
```

## Arguments

- `--annotation`: Path to annotation file (required)
- `--aug`: Augmentation to test: `all`, `horizontal_flip`, `rotation`, `resize_crop_pad`, `gridmask`, `mixup`, `color` (default: `all`)
- `--num-tests`: Number of test cases per augmentation (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)
- `--input-shape`: Input image shape as height width (default: 608 608)
- `--output-dir`: Directory to save visualizations (default: tests/augmentation_test_outputs)

## Output

The script generates side-by-side visualizations showing:
1. **Original image with boxes** (green boxes): The input image with original bounding boxes
2. **Augmented image with boxes** (red boxes): The augmented image with transformed bounding boxes

Each test case produces:
- `{aug_name}_test_XXX.png`: Side-by-side comparison of original and augmented images

### Output Structure

```
tests/augmentation_test_outputs/
├── horizontal_flip/
│   ├── horizontal_flip_test_001.png
│   ├── horizontal_flip_test_002.png
│   └── ...
├── rotation/
│   ├── rotation_test_001.png
│   └── ...
├── resize_crop_pad/
│   └── ...
├── gridmask/
│   └── ...
└── color/
    └── ...
```

## Visual Inspection Checklist

When reviewing the output images, verify:

### For All Augmentations:
- [ ] **Box alignment**: Boxes in augmented image align correctly with objects
- [ ] **Boundary checks**: No boxes are outside image boundaries
- [ ] **Class labels**: Boxes maintain correct class labels
- [ ] **Box count**: Number of boxes is reasonable (some may be filtered if too small)

### For Horizontal Flip:
- [ ] **Mirroring**: Boxes are correctly mirrored horizontally
- [ ] **Position**: Left boxes become right boxes and vice versa

### For Rotation:
- [ ] **Rotation**: Boxes rotate with the image
- [ ] **Shape**: Boxes maintain rectangular shape (may be rotated)
- [ ] **Clipping**: Boxes are correctly clipped to image boundaries

### For Resize Crop Pad:
- [ ] **Scaling**: Boxes scale correctly with image resize
- [ ] **Cropping**: Boxes are correctly clipped to crop boundaries
- [ ] **Padding**: Boxes are correctly offset by padding

### For GridMask:
- [ ] **Filtering**: Boxes that become too small are filtered out
- [ ] **Remaining boxes**: Remaining boxes still align with objects

### For MixUp:
- [ ] **Image blending**: Images are properly blended with visible mixing effect
- [ ] **Box concatenation**: Boxes from both source images are correctly concatenated
- [ ] **Box count**: Total box count should equal sum of valid boxes from both images
- [ ] **Box alignment**: Boxes from both images align with their respective objects in the mixed image

### For Color Augmentations:
- [ ] **No change**: Box coordinates remain exactly the same
- [ ] **Only color changes**: Only image colors change, not box positions

## Interpreting Results

### Good Results
- Boxes align perfectly with objects after augmentation
- All boxes are within image boundaries
- Box coordinates match object positions visually
- Number of boxes is reasonable (some filtering is expected for certain augmentations)

### Bad Results (Need Investigation)
- Boxes misaligned with objects
- Boxes outside image boundaries
- Missing boxes that should be present
- Box coordinates don't match visual object positions
- Box count dramatically different from original (unless expected, e.g., GridMask filtering)

## Integration with CI/CD

For automated testing, you can add this to your test suite:

```bash
# Run tests and check exit code
python tests/test_augmentations.py \
    --annotation data/coco_train2017.txt \
    --aug all \
    --num-tests 2 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "Augmentation tests passed"
else
    echo "Augmentation tests failed"
    exit 1
fi
```

## Troubleshooting

### No boxes visible
- Check that annotation file has valid boxes
- Verify image paths in annotation file are correct
- Ensure boxes are in correct format: `x1,y1,x2,y2,class`

### Boxes misaligned
- This indicates a bug in coordinate transformation - report as issue
- Check that augmentation-specific transformations are correctly applied

### Missing boxes
- Some boxes may be filtered if they're too small after augmentation
- Check minimum box size threshold in augmentation code
- This is expected for GridMask and some rotation cases

### Boxes outside boundaries
- This indicates a bug in coordinate clipping - report as issue
- Boxes should always be clipped to image boundaries

## Next Steps

After verifying augmentations work correctly:
1. Test augmentation combinations (pipeline order matters)
2. Run full training pipeline with augmentations enabled
3. Monitor training metrics to ensure augmentations improve model performance

## Related Tests

- `test_mosaic_augmentation.py`: Comprehensive test for Mosaic augmentation
- All augmentations are now covered in this unified test suite

## Technical Notes

### Batch-Level Augmentations
MixUp requires special handling as it operates on batches of images. The test suite
automatically loads multiple images and prepares batches when testing MixUp.

### Deterministic Testing
All tests use fixed random seeds for reproducibility. This ensures that test results
are consistent across runs, making it easier to identify regressions.

### Coordinate System
All augmentations operate in the same coordinate system (input_shape dimensions).
Boxes are transformed from original image coordinates to the resized coordinate space
before augmentation, ensuring consistency across all transformations.

