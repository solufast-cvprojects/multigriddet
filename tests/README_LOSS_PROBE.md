# MultiGridLoss Component Diagnostic Tool

## Purpose

This diagnostic tool probes individual loss components (localization, objectness, anchor, classification) during training to diagnose validation loss divergence and identify training instability issues. It helps identify which loss component is causing problems, especially useful when validation loss diverges after the first epoch or when fine-tuning pretrained models.

The tool computes loss components using the exact same infrastructure as training, ensuring diagnostic results match what the model sees during actual training.

## Prerequisites

1. **Conda Environment**: Activate the `cv_modern` conda environment:
   ```bash
   conda activate cv_modern
   ```

2. **Configuration**: Ensure `configs/train_config.yaml` exists and is properly configured, or specify a custom config file.

3. **Weights (Optional)**: If testing with pretrained weights, ensure the weights file exists at the specified path.

## Usage

### Basic Usage

```bash
python tests/probe_multigrid_loss.py --num-batches 2
```

This will:
- Load the training config from `configs/train_config.yaml`
- Build the training model with the same configuration as actual training
- Create a `MultiGridDataGenerator` using the real augmentation pipeline
- Process 2 batches and log all loss components before any weight updates
- Save results to `tests/loss_probe_outputs/` in both JSON and CSV formats

### Command-Line Arguments

- `--num-batches`: Number of batches to process (default: 2)
  - Example: `--num-batches 5` to process 5 batches
  
- `--config`: Path to training config file (default: `configs/train_config.yaml`)
  - Example: `--config configs/my_custom_config.yaml`
  
- `--output-dir`: Output directory for results (default: `tests/loss_probe_outputs/`)
  - Example: `--output-dir loss_stats/`
  
- `--format`: Output format - `json`, `csv`, or `both` (default: `both`)
  - Example: `--format json` to save only JSON format
  
- `--verbose`: Print detailed per-scale breakdown
  - Example: `--verbose` to see per-scale loss components
  
- `--weights`: Path to pretrained full model weights (overrides config)
  - Example: `--weights weights/model.h5` to load full model weights
  
- `--backbone-weights`: Path to pretrained backbone weights (e.g., darknet53.h5)
  - Example: `--backbone-weights weights/darknet53.h5` to load backbone weights

### Examples

```bash
# Process 5 batches with verbose output
python tests/probe_multigrid_loss.py --num-batches 5 --verbose

# Use a different config file
python tests/probe_multigrid_loss.py --config configs/custom_train_config.yaml

# Save only CSV format
python tests/probe_multigrid_loss.py --num-batches 3 --format csv

# Custom output directory
python tests/probe_multigrid_loss.py --num-batches 2 --output-dir my_loss_stats/

# Test with pretrained full model weights
python tests/probe_multigrid_loss.py --weights weights/model.h5 --num-batches 2

# Test with pretrained backbone weights
python tests/probe_multigrid_loss.py --backbone-weights weights/darknet53.h5 --num-batches 2

# Compare random initialization vs pretrained weights
python tests/probe_multigrid_loss.py --num-batches 2 --output-dir loss_random/
python tests/probe_multigrid_loss.py --weights weights/model.h5 --num-batches 2 --output-dir loss_pretrained/
```

## Output Format

### Console Output

The script prints detailed information for each batch:

1. **Batch Statistics**:
   - `Batch size`: Number of images in the batch
   - `Total objects`: Total number of ground truth objects across all scales
   - `Number of scales`: Number of detection scales (typically 3)
   - Per-scale statistics: Grid shape, number of objects, object density

2. **Loss Components (Raw Values)**:
   - `Localization`: Location/coordinate loss (MSE or IoU-based)
   - `Objectness`: Objectness prediction loss (BCE)
   - `Anchor`: Anchor prediction loss (BCE, if applicable)
   - `Classification`: Classification loss (BCE or focal)
   - `Total`: Weighted sum of all losses

3. **Scaling Coefficients**:
   - `coord_scale`: Multiplier for localization loss
   - `object_scale`: Multiplier for positive objectness loss
   - `no_object_scale`: Multiplier for negative objectness loss
   - `class_scale`: Multiplier for classification loss
   - `anchor_scale`: Multiplier for anchor loss

4. **Weighted Loss Components**:
   - Each loss component multiplied by its scaling coefficient
   - Shows the actual contribution to total loss

5. **Per-Scale Breakdown** (with `--verbose`):
   - Loss components for each detection scale separately
   - Normalization factors for each component

### File Outputs

Results are saved to the specified output directory (default: `tests/loss_probe_outputs/`):

#### JSON Format (`loss_components.json`)

Structured data with all loss components, batch statistics, and per-scale breakdowns:

```json
[
  {
    "batch_idx": 0,
    "batch_stats": {
      "batch_size": 4,
      "total_objects": 12,
      "num_scales": 3,
      "per_scale_stats": [...]
    },
    "loss_components": {
      "total_loss": 2.345,
      "localization_loss": 0.123,
      "objectness_loss": 1.456,
      "anchor_loss": 0.234,
      "classification_loss": 0.532
    },
    "scaling_coefficients": {...},
    "per_scale_losses": [...]
  }
]
```

#### CSV Format (`loss_components.csv`)

Tabular format for easy analysis:

| batch_idx | batch_size | total_objects | total_loss | localization_loss | objectness_loss | anchor_loss | classification_loss | coord_scale | object_scale | ... |
|-----------|------------|---------------|------------|-------------------|-----------------|-------------|---------------------|-------------|--------------|-----|
| 0         | 4          | 12            | 2.345      | 0.123            | 1.456          | 0.234       | 0.532              | 1.0         | 1.0          | ... |

## Understanding Loss Components

Each loss component measures a different aspect of the detection task. Understanding their normal ranges and what high values indicate helps diagnose training issues.

### Localization Loss

- **What it measures**: How well the model predicts bounding box coordinates (center x, y and width, height)
- **Normal range**: Typically 0.01-0.5 (depends on input resolution and normalization)
- **High values indicate**: Model struggles to localize objects accurately
- **Common causes**: 
  - Poor anchor matching
  - Incorrect coordinate encoding/decoding
  - Normalization issues

### Objectness Loss

- **What it measures**: How well the model predicts whether an object exists in a grid cell
- **Normal range**: Typically 0.1-2.0 (varies with object density)
- **High values indicate**: Model struggles to detect object presence
- **Common causes**:
  - Class imbalance (too many negative cells)
  - Incorrect `no_object_scale` setting
  - Poor initialization
- **Warning signs**: If objectness loss is 10x higher than other components, it may dominate training

### Anchor Loss

- **What it measures**: How well the model predicts which anchor fits best for each object
- **Normal range**: Typically 0.1-1.0
- **High values indicate**: Model struggles to match objects to anchors
- **Common causes**:
  - Anchor sizes don't match object sizes in dataset
  - Incorrect anchor assignment logic

### Classification Loss

- **What it measures**: How well the model predicts object classes
- **Normal range**: Typically 0.1-1.0 (depends on number of classes)
- **High values indicate**: Model struggles with classification
- **Common causes**:
  - Class imbalance
  - Poor feature representation
  - Incorrect class weights

## What to Look For

### Loss Component Balance

A well-balanced loss typically has:
- **Objectness loss**: Highest component (due to many negative cells)
- **Classification loss**: Second highest (depends on number of classes)
- **Anchor loss**: Moderate (only computed on object cells)
- **Localization loss**: Lowest (typically well-behaved)

If one component dominates (e.g., 5-10x higher than others), it may indicate an issue.

### Objectness Dominating

If objectness loss is much higher than other components (e.g., 5-10x):

1. **Check normalization**: Ensure `loss_normalization` is set correctly
2. **Adjust `no_object_scale`**: Reduce from default 1.0 to 0.5 or lower
3. **Check object density**: If objects are sparse, objectness loss will be high
4. **Verify augmentation**: Ensure augmentations aren't creating too many negative cells

### Localization Too High

If localization loss dominates:

1. **Check coordinate encoding**: Verify coordinate transformation is correct
2. **Verify anchor sizes**: Ensure anchors match object sizes in dataset
3. **Check `coord_scale`**: May need to reduce from 1.0 to 0.5

### Classification Too High

If classification loss is very high:

1. **Check class weights**: Enable `class_weights: auto` if class imbalance exists
2. **Verify number of classes**: Ensure `num_classes` matches dataset
3. **Check label encoding**: Verify class labels are correctly encoded

### Normalization Factor Analysis

The normalization factors show how losses are scaled:

- **"batch" normalization**: Loss divided by batch size (default, recommended)
- **"positives" normalization**: Loss divided by number of objects (adapts to data)
- **"grid" normalization**: Loss divided by total grid cells

If normalization factors vary significantly between batches, it may indicate:
- Inconsistent batch composition
- Variable object density
- Need for different normalization strategy

## Troubleshooting

### Warning: "Could not extract base model: Base model not found as sub-layer"

**Cause**: This is a cosmetic warning in the diagnostic script's internal logic. The training model structure is correct, and weights are shared as intended.

**Solution**: 
- This warning can be safely ignored
- The diagnostic tool will use a fallback method to extract predictions
- Training model functionality is not affected

### Error: "Config file not found"

**Cause**: The config file path is incorrect.

**Solution**:
- Verify the config file exists at the specified path
- Use `--config` to specify the correct path
- Check that you're running from the project root directory

### Error: "Out of memory" or "GPU memory error"

**Cause**: Batch size or model is too large for available memory.

**Solution**:
- Reduce batch size in config file
- Process fewer batches with `--num-batches 1`
- Use CPU mode if GPU memory is limited

### Loss Components Are NaN or Inf

**Cause**: Numerical instability in loss computation.

**Solution**:
- Check for invalid ground truth labels (NaN, Inf, or out-of-range values)
- Verify augmentation pipeline isn't producing invalid boxes
- Check loss normalization factors aren't zero
- Ensure model weights are initialized correctly

### Predictions Are All Zero or Constant

**Cause**: Model may not be initialized or weights are not loaded.

**Solution**:
- Verify model weights are loaded correctly
- Check that the model is in evaluation mode (not training mode)
- Ensure input images are normalized correctly

## Best Practices

1. **Run before training**: Use this tool to verify loss components are reasonable before starting full training
2. **Compare initialization strategies**: Test with random initialization vs pretrained weights to verify weight loading
3. **Compare across epochs**: Save outputs and compare loss component trends across training epochs
4. **Monitor objectness**: Objectness loss is often the culprit for validation divergence - monitor it closely
5. **Check normalization**: Ensure normalization factors are reasonable and consistent
6. **Use verbose mode**: Enable `--verbose` to see per-scale breakdowns for detailed analysis
7. **Save outputs**: Keep JSON/CSV outputs for comparison and analysis

## Technical Details

### Model Building

The diagnostic tool uses the same model building infrastructure as training:
- Same config loading (`ConfigLoader`)
- Same model building (`MultiGridTrainer.build_model()`)
- Same weight loading mechanism (including batch normalization statistics)
- Same loss function configuration

### Data Pipeline

The tool uses the exact same data pipeline as training:
- Same data generator (`MultiGridDataGenerator` with real augmentation pipeline)
- Same augmentation sequence and parameters
- Same batch preparation and padding
- Same target encoding (`y_true` format)

### Loss Computation

Loss components are computed using the actual `MultiGridLoss` class:
- Same loss computation logic as training
- Same normalization factors
- Same scaling coefficients
- Computed before any weight updates (pure forward pass)

## Integration with Training

This diagnostic tool uses the exact same infrastructure as training, ensuring diagnostic results match what the model sees during actual training. The only difference is that no weight updates are performed - it's a pure diagnostic forward pass.

## Example Workflow

1. **Before training**: Run diagnostic to verify loss components are reasonable
   ```bash
   python tests/probe_multigrid_loss.py --num-batches 2 --verbose
   ```

2. **Check outputs**: Review JSON/CSV files to identify any obvious issues

3. **Adjust config**: If objectness is too high, reduce `no_object_scale` in config

4. **Re-run diagnostic**: Verify changes improved loss balance

5. **Start training**: Proceed with training, monitoring validation loss

6. **If validation diverges**: Re-run diagnostic to see which component is causing issues

