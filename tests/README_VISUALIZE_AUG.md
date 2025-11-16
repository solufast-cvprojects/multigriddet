# Visualize Augmented Training Batches

## Purpose

This diagnostic tool visualizes exactly what the model sees after all training augmentations (Mosaic, MixUp, fixed-capacity padding, etc.) without running a full training job. It helps verify that:

- Augmentations are applied correctly
- Annotations are preserved after augmentations
- Box alignment is correct on augmented images (especially Mosaic/MixUp)
- No boxes are missing after augmentations
- Padding behavior is correct (invalid boxes are filtered out)

## Usage

### Basic Usage

```bash
python tests/visualize_augmented_batches.py --num-batches 5
```

This will:
- Load the training config from `configs/train_config.yaml`
- Process 5 batches through the same augmentation pipeline used in training
- Save visualizations and metadata to `tests/visualizations/`

### Command-Line Arguments

- `--num-batches`: Number of batches to visualize (default: 5)
  - Example: `--num-batches 10` to visualize 10 batches
  
- `--config`: Path to training config file (default: `configs/train_config.yaml`)
  - Example: `--config configs/my_custom_config.yaml`
  
- `--output-dir`: Output directory for visualizations (default: `tests/visualizations/`)
  - Example: `--output-dir my_visualizations/`
  
- `--seed`: Random seed for reproducibility (default: 42)
  - Example: `--seed 123` for different random augmentations

### Examples

```bash
# Visualize 10 batches with custom output directory
python tests/visualize_augmented_batches.py --num-batches 10 --output-dir my_viz/

# Use a different config file
python tests/visualize_augmented_batches.py --config configs/custom_train_config.yaml

# Reproducible visualization with specific seed
python tests/visualize_augmented_batches.py --num-batches 5 --seed 42
```

## Output Format

### Images

For each sample in each batch, an image is saved with bounding boxes drawn:

- **Filename**: `batch_{batch_idx:04d}_sample_{sample_idx:04d}.png`
- **Format**: PNG image with bounding boxes and class labels overlaid
- **Example**: `batch_0000_sample_0000.png`, `batch_0000_sample_0001.png`, etc.

Each image shows:
- The augmented image (after all augmentations including Mosaic/MixUp if enabled)
- Bounding boxes drawn in color-coded rectangles
- Class labels for each box
- Center point markers for each box

### Metadata JSON

For each sample, a JSON file is saved with detailed information:

- **Filename**: `batch_{batch_idx:04d}_sample_{sample_idx:04d}.json`
- **Format**: JSON with structured metadata

Example JSON structure:

```json
{
  "batch_idx": 0,
  "sample_idx": 0,
  "image_shape": [608, 608, 3],
  "num_boxes_total": 200,
  "num_boxes_valid": 15,
  "boxes": [
    {
      "x1": 120.5,
      "y1": 80.3,
      "x2": 250.7,
      "y2": 200.1,
      "class_id": 0,
      "class_name": "person",
      "width": 130.2,
      "height": 119.8
    },
    ...
  ]
}
```

**Fields:**
- `batch_idx`: Index of the batch (0-based)
- `sample_idx`: Index of the sample within the batch (0-based)
- `image_shape`: Shape of the image `[height, width, channels]`
- `num_boxes_total`: Total number of boxes (including padded/invalid boxes)
- `num_boxes_valid`: Number of valid boxes (after filtering padded boxes)
- `boxes`: Array of box objects, each containing:
  - `x1`, `y1`, `x2`, `y2`: Box coordinates in pixel space
  - `class_id`: Class index (0-based)
  - `class_name`: Human-readable class name
  - `width`, `height`: Box dimensions

## What to Look For

When reviewing the visualizations, check:

### 1. Box Alignment

- **Mosaic augmentation**: Boxes should align correctly on the 4-image composite
  - Each quadrant should have boxes from its source image
  - Boxes should not be misaligned or cut off at quadrant boundaries
  
- **MixUp augmentation**: Boxes should be visible on the blended image
  - Both source images' boxes should be present
  - Boxes should align with their respective objects in the blended image

### 2. Class Labels

- Each box should have the correct class label
- Class names should match the expected classes from the dataset
- No boxes should have invalid class IDs

### 3. Box Preservation

- **No missing boxes**: All boxes from source images should be present after augmentations
  - For Mosaic: Should see boxes from all 4 source images
  - For MixUp: Should see boxes from both source images
  - Check that `num_boxes_valid` matches expectations

### 4. Padding Behavior

- Invalid boxes (all zeros) should be filtered out
- `num_boxes_valid` should be less than or equal to `num_boxes_total`
- Padded boxes should not appear in visualizations

### 5. Augmentation Quality

- Images should show realistic augmentations:
  - Color jittering (brightness, contrast, saturation, hue)
  - Geometric transformations (rotation, flip, resize-crop-pad)
  - GridMask (if enabled) should create occlusion patterns
  - Mosaic should create realistic 4-image composites
  - MixUp should create realistic blended images

## Troubleshooting

### No boxes visible

- Check that the annotation file has valid boxes
- Verify that `num_boxes_valid > 0` in the JSON metadata
- Check that boxes are within image bounds

### Boxes misaligned

- This may indicate an issue with augmentation transformations
- Check that box coordinates are correctly transformed during augmentations
- Verify that Mosaic/MixUp are applying transformations correctly

### Missing boxes after augmentation

- Check the `num_boxes_valid` count before and after augmentations
- Verify that capacity expansion is working correctly
- Check that boxes aren't being filtered out incorrectly

### Images look wrong

- Verify that image normalization is correct (images should be in [0, 1] range before visualization)
- Check that color space conversion is correct (RGB vs BGR)
- Verify that augmentations are being applied in the correct order

## Technical Details

### Dataset Pipeline

The script uses the same tf.data pipeline as training:

1. **Load and parse**: Load images and parse annotations
2. **Preprocess**: Letterbox resize, multi-scale (if enabled)
3. **Augment**: Apply individual image augmentations (resize-crop-pad, flip, color jitter, rotation, gridmask)
4. **Batch**: Create batches with fixed-capacity padding
5. **Expand capacity**: Expand box capacity for batch augmentations (8× for Mosaic+MixUp, 4× for Mosaic, 2× for MixUp)
6. **Batch augmentations**: Apply Mosaic and/or MixUp if enabled
7. **Visualize**: Extract boxes before y_true conversion for visualization

### Box Format

Boxes are in format `[x1, y1, x2, y2, class]`:
- `x1, y1`: Top-left corner coordinates
- `x2, y2`: Bottom-right corner coordinates
- `class`: Class index (0-based)

All coordinates are in pixel space relative to the augmented image.

### Image Format

Images are normalized float32 in range [0, 1] during processing, then converted to uint8 [0, 255] for visualization.

## Integration with Training

This script uses the exact same augmentation pipeline as training, so visualizations should match what the model sees during training. The only difference is that this script extracts boxes before they're converted to y_true format, allowing direct visualization.

