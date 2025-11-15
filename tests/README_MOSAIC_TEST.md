# Mosaic Augmentation Regression Test

This test script provides automated verification and visual validation for the Mosaic
augmentation pipeline. Mosaic augmentation combines four training images into a single
composite image, effectively increasing the effective batch size and introducing diverse
contextual information. This test ensures that bounding box coordinates are correctly
transformed when images are combined.

## Purpose

Before training on expensive cloud instances (like RunPod), it's critical to verify that:
1. Box coordinates are correctly shifted when quadrants are pasted
2. Boxes align properly with objects in the combined image
3. No boxes are lost or incorrectly positioned during augmentation

## Usage

### Basic Usage

```bash
python tests/test_mosaic_augmentation.py \
    --annotation data/coco_train2017.txt \
    --num-tests 5
```

### With Custom Settings

```bash
python tests/test_mosaic_augmentation.py \
    --annotation data/coco_train2017.txt \
    --num-tests 10 \
    --seed 42 \
    --input-shape 608 608 \
    --output-dir tests/my_mosaic_tests
```

## Arguments

- `--annotation`: Path to annotation file (required)
- `--num-tests`: Number of test cases to run (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--input-shape`: Input image shape as height width (default: 608 608)
- `--output-dir`: Directory to save visualizations (default: tests/mosaic_test_outputs)

## Output

The script generates visualizations showing:
1. **Original images with boxes** (green boxes): The 4 input images with their original bounding boxes
2. **Mosaic result with boxes** (red boxes): The combined mosaic image with all boxes correctly positioned

Each test case produces:
- `mosaic_test_XXX.png`: Grid showing all 4 original images + mosaic result
- `mosaic_only_XXX.png`: Just the mosaic result for detailed inspection

## Visual Inspection Checklist

When reviewing the output images, verify:

- [ ] **Box alignment**: Boxes in the mosaic result align correctly with objects
- [ ] **Quadrant positioning**: Boxes from different quadrants are in the correct positions
- [ ] **Boundary checks**: No boxes are outside image boundaries
- [ ] **Class labels**: Boxes maintain correct class labels
- [ ] **Box completeness**: All valid boxes from original images appear in mosaic
- [ ] **Coordinate shifting**: Boxes from quadrants 1-3 are correctly shifted by crop offsets

## Example Output Structure

```
tests/mosaic_test_outputs/
├── mosaic_test_001.png
├── mosaic_test_002.png
├── mosaic_test_003.png
├── mosaic_test_004.png
├── mosaic_test_005.png
├── mosaic_only_001.png
├── mosaic_only_002.png
├── mosaic_only_003.png
├── mosaic_only_004.png
└── mosaic_only_005.png
```

## Interpreting Results

### Good Results
- Boxes align perfectly with objects in the mosaic
- All boxes are within image boundaries
- Box coordinates match object positions visually

### Bad Results (Need Investigation)
- Boxes misaligned with objects
- Boxes outside image boundaries
- Missing boxes that should be present
- Boxes in wrong quadrants

## Integration with CI/CD

For automated testing, you can add this to your test suite:

```bash
# Run tests and check exit code
python tests/test_mosaic_augmentation.py \
    --annotation data/coco_train2017.txt \
    --num-tests 3 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "Mosaic augmentation test passed"
else
    echo "Mosaic augmentation test failed"
    exit 1
fi
```

## Troubleshooting

### No boxes visible
- Check that annotation file has valid boxes
- Verify image paths in annotation file are correct
- Ensure boxes are in correct format: `x1,y1,x2,y2,class`

### Boxes misaligned
- This indicates a bug in coordinate shifting - report as issue
- Check that crop offsets are correctly applied to quadrants 1-3

### Missing boxes
- Some boxes may be filtered if they're too small after cropping
- Check minimum box size threshold in augmentation code

