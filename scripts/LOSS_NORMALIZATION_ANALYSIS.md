# Loss Normalization Analysis

## Current Implementation

The loss function normalizes by variable factors that depend on the number of objects in each batch:

1. **Location Loss**: Normalized by `num_positive_cells` (varies per batch)
   - Line 211-222 in `multigrid_loss.py`
   - `num_positive_cells = K.maximum(1.0, K.sum(object_mask))`

2. **Anchor Loss**: Normalized by `num_active_cells` (varies per batch)
   - Line 619-629 in `multigrid_loss.py`
   - `num_active_cells = K.maximum(1.0, K.sum(combined_weight))`
   - Includes both positive and negative cells (excluding ignore regions)

3. **Classification Loss**: Normalized by `num_positive_cells` (varies per batch)
   - Line 270-282 in `multigrid_loss.py`
   - Uses the same `num_positive_cells` as location loss

## Problem

### Issue 1: Inconsistent Normalization Between Training and Validation

- Training batches may have different numbers of objects than validation batches
- This causes loss values to be on different scales
- Validation loss can appear higher even when model performance is good
- Makes it difficult to compare losses across epochs

### Issue 2: Variable Normalization Factor

- `num_positive_cells` can vary significantly between batches
- A batch with 1 object will have much higher per-cell loss than a batch with 100 objects
- This can cause unstable gradients and training dynamics

### Issue 3: Potential Division by Zero

- Code uses `K.maximum(1.0, ...)` to prevent division by zero
- But if a batch has no objects, loss becomes 0, which may not be desired

## Proposed Solution

### Option 1: Normalize by Batch Size (Recommended)

Normalize all losses by `batch_size` for consistency:

```python
# Location loss
loc_loss = loc_loss / batch_size_f

# Anchor loss  
anchor_loss = anchor_loss / batch_size_f

# Classification loss
class_loss = class_loss / batch_size_f
```

**Pros:**
- Consistent normalization across all batches
- Training and validation losses are directly comparable
- Stable gradient scale

**Cons:**
- Loss values may be smaller (but still comparable)
- Doesn't account for varying number of objects per batch

### Option 2: Normalize by Fixed Factor

Use a fixed normalization factor (e.g., expected number of objects per batch):

```python
expected_objects_per_batch = 10.0  # Or calculate from dataset
loc_loss = loc_loss / (num_positive_cells / expected_objects_per_batch * batch_size_f)
```

**Pros:**
- Accounts for object density
- More stable than pure per-cell normalization

**Cons:**
- Requires dataset statistics
- Still variable if object density varies

### Option 3: Hybrid Approach

Normalize by `batch_size` but weight by object density:

```python
# Normalize by batch_size, but scale by object density
object_density = num_positive_cells / (batch_size_f * grid_h * grid_w)
loc_loss = loc_loss / batch_size_f * (1.0 + object_density)
```

## Recommendation

**Use Option 1 (normalize by batch_size)** for the following reasons:

1. **Consistency**: All batches normalized the same way
2. **Comparability**: Training and validation losses are directly comparable
3. **Simplicity**: Easy to implement and understand
4. **Standard Practice**: Many object detection frameworks normalize by batch size

### Implementation

Modify `multigrid_loss.py`:

```python
# In compute_loss method, use batch_size_f for all normalizations
batch_size_f = K.cast(batch_size, 'float32')

# Location loss
loc_loss = self._compute_mse_loss(...)
loc_loss = loc_loss / batch_size_f  # Instead of num_positive_cells

# Anchor loss (already receives batch_size_f, but uses num_active_cells)
# Change to:
anchor_loss = anchor_loss_sum / batch_size_f  # Instead of num_active_cells

# Classification loss
class_loss = K.sum(class_loss) / batch_size_f  # Instead of num_positive_cells
```

## Testing

After implementing the change:

1. Run the visualization script to verify y_true assignments are correct
2. If y_true is correct but loss still diverges, apply normalization fix
3. Monitor training/validation loss curves - they should be more comparable
4. Check that gradients are stable and training converges

## Notes

- The current normalization by `num_positive_cells` is not necessarily wrong, but it makes loss values incomparable between batches
- The key issue is that validation batches may have different object densities than training batches
- Normalizing by `batch_size` ensures consistent loss scale regardless of object count

