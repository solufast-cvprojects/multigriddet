# Visualization Scripts

## visualize_y_true.py

Visualization script to verify y_true assignments from `tf_preprocess_true_boxes`.

### Purpose

This script helps verify that the `tf_preprocess_true_boxes` function in `multigriddet/data/generators.py` is correctly assigning:
- Box coordinates to grid cells
- Anchors to boxes
- Classes to boxes

### Usage

```bash
python scripts/visualize_y_true.py \
    --annotation "path/to/image.jpg x1,y1,x2,y2,class ..." \
    --config configs/train_config.yaml \
    --output y_true_visualization.png
```

Or with an annotation file (uses first line):

```bash
python scripts/visualize_y_true.py \
    --annotation data/coco_val2017.txt \
    --config configs/train_config.yaml \
    --output y_true_visualization.png
```

### Options

- `--annotation`: Annotation line or path to annotation file
- `--config`: Path to training config YAML file (must contain model_config and data paths)
- `--output`: Output path for visualization image (default: `y_true_visualization.png`)
- `--augment`: Enable augmentation (default: False, recommended for testing)

### Output

The script generates a visualization image showing:
1. **Original Annotation Boxes** (green): Boxes from the annotation file
2. **Layer N Decoded Boxes** (red/blue/orange/etc.): Boxes decoded from y_true for each layer
   - Shows grid cells with assignments
   - Displays class, anchor index, and grid position for each box
   - Original boxes shown as dashed green lines for comparison

### Notes

- The script processes a single annotation through the data generator
- Decoded boxes may have slight positional differences due to grid cell quantization
- Multiple boxes may appear per layer (3x3 grid assignment strategy)
- If boxes don't align well, there may be an issue with `tf_preprocess_true_boxes`

