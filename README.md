# MultiGridDet

A modern implementation of Multi-Grid Object Detection, featuring efficient multi-grid cell annotation and trainable anchor prediction.

## Overview

MultiGridDet is an enhanced one-stage object detector derived from YOLOv3, introducing **multi-grid redundant bounding-box assignment** for tighter and more stable object localization. Unlike conventional YOLO approaches where only the grid cell containing the object's center is responsible for prediction, MultiGridDet extends this responsibility to the **3×3 grid neighborhood** (center cell plus its eight adjacent cells).

This redundant supervision allows multiple grid cells to view and predict the same object from slightly different spatial perspectives, reducing prediction noise, balancing positive-negative grid ratios, and accelerating bounding-box convergence during training.

### Key Technical Contributions

- **Multi-Grid Object Annotation**: Each object is annotated across 9 neighboring grid cells instead of one, enabling multi-view consensus prediction
- **Trainable Anchors**: In contrast to YOLO's fixed, pre-clustered anchors, MultiGridDet treats anchors as predictable and learnable parameters, similar to class logits
- **Compact Detection Head**: Based on DenseYOLO's lightweight detection head, reducing parameters without sacrificing precision
- **Expanded Coordinate Range**: Neighbors regress coordinates in an expanded `[-1,2]` range relative to their cell, allowing fine-tuning from offset spatial references

## Features

### Core Algorithm Features
- **Multi-Grid Redundant Assignment**: 3×3 neighborhood cells collaborate to predict the same object, reducing label sparsity and improving positive/negative balance
- **Trainable Anchor Prediction**: Dynamic anchor selection learned end-to-end, eliminating manual anchor tuning and reducing parameters
- **Expanded Coordinate Range**: Neighbors regress coordinates in `[-1,2]` range, enabling fine-tuning from offset spatial references
- **Multi-View Consensus**: Nine local "views" per object provide redundant supervision for smoother localization and faster convergence

### Technical Features
- **High Performance**: Optimized for GPU training and inference with XLA compilation
- **Modular Design**: Separate backbones (Darknet53/CSPDarknet53), necks (FPN), and heads (DenseYOLO-style)
- **Multiple Loss Functions**: IoL-weighted MSE, GIoU/DIoU losses, and trainable anchor prediction
- **Compact Detection Head**: DenseYOLO-inspired lightweight head with reduced parameters
- **Easy Deployment**: Simple inference API with multiple input types (image, video, camera, directory)
- **Research Ready**: Full training pipeline with academic-quality visualizations
- **YAML-based Configuration**: No code changes needed for different experiments
- **CLI Interface**: Professional command-line tools for training, inference, and evaluation
- **Two-Stage Training**: Transfer learning support with backbone freezing
- **Data Augmentation**: Mosaic, GridMask, multi-scale training
- **Production-Ready**: Error handling, progress tracking, and comprehensive logging

## Core Algorithm Summary

### Multi-Grid Assignment Mechanism

In standard YOLO-style detectors, the grid cell containing an object's center `(cx, cy)` is **exclusively responsible** for predicting that object's bounding box and class.

In **MultiGridDet**, this responsibility is expanded to the **3×3 neighborhood** around that center grid cell:

```
(c'x, c'y) = (cx + dx, cy + dy), where dx, dy ∈ {-1, 0, 1}
```

Every such neighboring cell (9 in total) simultaneously predicts the same object's class and bounding box parameters.

### Coordinate Encoding

Bounding-box regression is computed relative to each participating grid cell:

```
x = (c'x + t'x) × gw
y = (c'y + t'y) × gh
```

where:
```
t'x = ∓dx + tx
t'y = ∓dy + ty
```

and `(gw, gh)` are the grid cell's width and height.

This formulation expands the effective coordinate range from `[0,1]` to `[-1,2]`, allowing each nearby cell to fine-tune the same object from slightly offset spatial references — effectively giving the detector **nine local "views" per object**.

### Visual Comparison

**YOLO (single responsible cell):**
```
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
|    | XX |    |    |   ← only the cell holding the object center (XX)
+----+----+----+----+      predicts {bbox, class, obj}
|    |    |    |    |
+----+----+----+----+
```

**MultiGridDet (3×3 neighborhood responsibility):**
```
+----+----+----+----+----+
|    |    |    |    |    |
+----+----+----+----+----+
|    | [1]| [2]| [3]|    |
+----+----+----+----+----+
|    | [4]|[XX]| [5]|    |   ← 9 cells (XX + its 8 neighbors) are supervised
+----+----+----+----+----+      to predict the same object {bbox, class, obj}
|    | [6]| [7]| [8]|    |
+----+----+----+----+----+
|    |    |    |    |    |
+----+----+----+----+----+
```

### Trainable Anchors

Instead of pre-assigning a *fixed* anchor per object (classic YOLO), MultiGridDet **predicts the anchor choice** jointly with class/objectness/bbox in a lighter output head:

```
head output: [class logits | bbox (t'x,t'y,tw,th) | anchor logits | obj]
```

This cuts parameters (≈k× fewer vs. k replicated heads) and learns anchor selection end-to-end.

### Training Target Generation

```python
# Pseudocode for a single GT box (x, y, bw, bh) at stride gw, gh
cx, cy = floor(x / gw), floor(y / gh)

for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        cpx, cpy = cx + dx, cy + dy           # neighbor cell (participating)
        if not in_bounds(cpx, cpy): 
            continue

        # Coordinate targets for this participating cell
        tx  = (x / gw) - cx                    # classic fractional offset wrt center cell
        ty  = (y / gh) - cy
        tpx = (-dx) + tx                       # shift into neighbor's frame
        tpy = (-dy) + ty                       # => range [-1, 2]

        tw  = log(bw / aw)                     # aw, ah = anchor dims
        th  = log(bh / ah)

        # Supervise this neighbor cell with the SAME object:
        targets[cpx, cpy] = {
            "bbox": (tpx, tpy, tw, th),
            "class": one_hot(cls),
            "anchor": best_anchor_id,         # or anchor logits as learnable choice
            "obj": 1
        }
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.17+
- CUDA-capable GPU (recommended)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/MultiGridDet.git
cd MultiGridDet

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Quick Test

```bash
# Test inference with a sample image
python examples/inference_example.py
```

## Quick Start

### 1. Inference (Detecting Objects)

#### Single Image
```bash
python infer.py --input examples/images/dog.jpg
```

#### Video File
```bash
python infer.py --input video.mp4 --type video
```

#### Real-time Camera
```bash
python infer.py --type camera
```

#### Batch Processing (Directory)
```bash
python infer.py --input images/ --type directory
```

#### Custom Configuration
```bash
# Override confidence and NMS thresholds
python infer.py --input dog.jpg --conf 0.6 --nms 0.4

# Use custom weights
python infer.py --input dog.jpg --weights weights/custom_model.h5

# Don't show result window (headless mode)
python infer.py --input dog.jpg --no-show

# Don't save result
python infer.py --input dog.jpg --no-save
```

### 2. Training

#### Basic Training
```bash
python train.py --config configs/train_config.yaml
```

#### With Pretrained Weights (Transfer Learning)
```bash
python train.py --config configs/train_config.yaml --weights weights/model5.h5
```

#### Resume Training from Checkpoint
```bash
python train.py --config configs/train_config.yaml --resume
```

#### Override Parameters
```bash
# Custom epochs and batch size
python train.py --config configs/train_config.yaml --epochs 50 --batch-size 8
```

### 3. Evaluation

#### Basic Evaluation
```bash
python eval.py --config configs/eval_config.yaml
```

#### Custom Weights and Dataset
```bash
python eval.py --weights weights/custom_model.h5 --data data/custom_val.txt
```

#### Override Parameters
```bash
# Custom batch size and confidence threshold
python eval.py --batch-size 16 --conf 0.5
```

## Directory Structure

```
MultiGridDet/
├── train.py                    # Training script
├── infer.py                    # Inference script
├── eval.py                     # Evaluation script
├── configs/
│   ├── models/
│   │   └── multigriddet_darknet.yaml    # Model architecture config
│   ├── train_config.yaml                 # Training configuration
│   ├── infer_config.yaml                 # Inference configuration
│   └── eval_config.yaml                  # Evaluation configuration
├── multigriddet/
│   ├── config/                 # Configuration system
│   ├── trainers/               # Training module
│   ├── inference/              # Inference module
│   └── evaluation/             # Evaluation module
└── weights/
    └── model5.h5               # Pretrained weights
```

## Configuration Files

### Model Configuration (`configs/models/multigriddet_darknet.yaml`)

Defines the model architecture:

```yaml
model:
  name: multigriddet_darknet
  type: preset  # Simple preset mode
  
  preset:
    architecture: multigriddet_darknet
    num_classes: 80
    input_shape: [608, 608, 3]
    anchors_path: configs/yolov3_coco_anchor.txt
    classes_path: configs/coco_classes.txt
```

### Training Configuration (`configs/train_config.yaml`)

Controls all training parameters:

- **Dataset paths**: Train/val annotations, class names
- **Training parameters**: Batch size, epochs, learning rate
- **Two-stage training**: Freeze backbone first, then fine-tune
- **Loss configuration**: Choose from 3 loss options
  - Option 1: IoL-weighted MSE
  - Option 2: IoL-weighted MSE with anchor prediction mask
  - Option 3: GIoU/DIoU localization loss
- **Data augmentation**: Mosaic, GridMask, multi-scale training
- **Optimizer**: Adam/SGD with configurable parameters
- **Learning rate schedule**: ReduceLROnPlateau, step decay, etc.
- **Callbacks**: TensorBoard, ModelCheckpoint, EarlyStopping

### Inference Configuration (`configs/infer_config.yaml`)

Controls inference behavior:

- **Input type**: Image, video, camera, or directory
- **Detection parameters**: Confidence threshold, NMS threshold
- **IoL support**: Use IoL (Intersection over Largest) instead of IoU
- **Output options**: Save results, show visualization
- **Video/Camera settings**: FPS, codec, resolution

### Evaluation Configuration (`configs/eval_config.yaml`)

Controls model evaluation:

- **Metrics**: mAP, mAP50, mAP75, precision, recall, F1-score
- **Evaluation parameters**: Batch size, confidence threshold
- **Output**: Save results to JSON
- **Visualizations**: Academic-quality plots (configurable, see below)

## Advanced Usage

### Creating Custom Configurations

1. **Copy an existing config**:
```bash
cp configs/train_config.yaml configs/my_experiment.yaml
```

2. **Edit parameters** in `my_experiment.yaml`

3. **Run with custom config**:
```bash
python train.py --config configs/my_experiment.yaml
```

### Multi-Scale Training

Enable in training config:

```yaml
training:
  augmentation:
    enabled: true
    rescale_interval: 10  # Change input size every 10 batches
```

### Two-Stage Training

Freeze backbone first, then fine-tune all layers:

```yaml
training:
  transfer_epochs: 50    # Train with frozen backbone for 50 epochs
  freeze_level: 1        # 0=all, 1=backbone, 2=all but head
  epochs: 100            # Then train all layers until epoch 100
```

### Academic Evaluation Visualizations

MultiGridDet includes publication-quality visualization tools for comprehensive model analysis. Enable visualizations in your evaluation config:

```yaml
# In configs/eval_config.yaml
visualizations:
  enabled: true  # Master switch
  
  # Individual plot controls
  plots:
    precision_recall_curves: true    # PR curves per class
    confusion_matrix: true           # Class confusion heatmap
    per_class_map_bar: true         # AP per class bar chart
    iou_distribution: true           # IoU histogram
    confidence_analysis: true        # Threshold tuning curve
  
  # PR curve settings
  pr_curves:
    show_per_class: true   # Individual class curves
    show_averaged: true    # Averaged curve
    top_k: 10             # Top K classes by AP
    style: 'paper'        # 'paper' or 'presentation'
  
  # Confusion matrix settings
  confusion_matrix:
    normalize: true   # Show percentages
    top_k: 20        # Top K classes
    cmap: 'Blues'    # Color scheme
  
  # Output settings
  output:
    format: 'png'   # 'png', 'pdf', 'svg'
    dpi: 300       # Publication quality
    save_dir: 'results/evaluation/plots'
```

**Generated Visualizations**:

1. **Precision-Recall Curves**: Per-class and averaged PR curves
2. **Confusion Matrix**: Heatmap showing class confusion patterns
3. **Per-Class AP Bar Chart**: Horizontal bar chart of AP@0.5 per class
4. **IoU Distribution**: Histogram of IoU values for true positives
5. **Confidence Analysis**: Precision/Recall/F1 vs confidence threshold

**Quick Testing**:

Three pre-configured evaluation modes are available:

```bash
# Fast evaluation (metrics only, no plots)
python eval.py --config configs/eval_config_fast.yaml

# Standard evaluation (default config)
python eval.py --config configs/eval_config.yaml

# Full evaluation (all visualizations, publication-ready)
python eval.py --config configs/eval_config_full.yaml
```

For testing with a limited dataset, set `max_images: 50` in the evaluation config.

**Example Output Structure**:

```
results/evaluation/plots/
├── pr_curves/
│   ├── pr_curve_person.png
│   ├── pr_curve_car.png
│   └── pr_curve_averaged.png
├── confusion_matrix.png
├── per_class_map.png
├── iou_distribution.png
└── confidence_analysis.png
```

**Requirements**: Visualizations require matplotlib and seaborn:

```bash
pip install matplotlib>=3.3.0 seaborn>=0.11.0
```

## Output Structure

### Training Outputs
```
logs/
├── checkpoints/                    # Model checkpoints
│   └── ep050-loss2.345-val_loss2.123.h5
├── tensorboard/                    # TensorBoard logs
└── training/                       # Training logs

trained_models/
└── final_model.h5                  # Final trained model
```

### Inference Outputs
```
output/
├── result_dog.jpg                  # Annotated image
├── result_video.mp4                # Annotated video
└── camera_output.mp4               # Camera recording
```

### Evaluation Outputs
```
results/
└── evaluation/
    ├── evaluation_results.json     # Metrics in JSON format
    └── plots/                       # Academic visualizations (if enabled)
        ├── pr_curves/
        │   ├── pr_curve_person.png
        │   ├── pr_curve_car.png
        │   └── pr_curve_averaged.png
        ├── confusion_matrix.png
        ├── per_class_map.png
        ├── iou_distribution.png
        └── confidence_analysis.png
```

## Model Architecture

MultiGridDet consists of three main components that work together to implement the multi-grid assignment strategy:

### 1. Backbone: Feature Extraction
- **Darknet53**: Standard YOLO backbone with 53 convolutional layers
- **CSPDarknet53**: Cross Stage Partial connections for improved gradient flow
- **Output**: Multi-scale feature maps at different resolutions (e.g., 19×19, 38×38, 76×76)

### 2. Neck: Feature Pyramid Network (FPN)
- **Multi-scale Feature Fusion**: Combines features from different backbone layers
- **Top-down Pathway**: High-level semantic features flow down to lower levels
- **Lateral Connections**: Skip connections preserve fine-grained spatial information
- **Output**: Enhanced feature maps ready for multi-grid detection

### 3. Head: Multi-Grid Detection Head
- **DenseYOLO-inspired Design**: Compact head with reduced parameters
- **Multi-Grid Assignment**: Each detection head processes 3×3 neighborhood cells
- **Trainable Anchors**: Anchor selection learned end-to-end as part of the prediction
- **Output Format**: `[class_logits | bbox_offsets | anchor_logits | objectness]`

### Key Architectural Differences from YOLO

1. **Expanded Responsibility**: Each object is supervised by 9 grid cells instead of 1
2. **Coordinate Range Extension**: Neighbors predict coordinates in `[-1,2]` range
3. **Dynamic Anchor Selection**: Anchors are predicted rather than pre-assigned
4. **Redundant Supervision**: Multiple cells provide consensus for the same object

## Loss Functions

MultiGridDet supports multiple loss configurations, each optimized for different aspects of the multi-grid detection:

### Option 1: IoL-weighted MSE Loss (Original Paper)
- **IoL (Intersection over Largest)**: More stable than IoU for small objects
- **MSE Regression**: Mean Squared Error for bounding box coordinates
- **Weighted by IoL**: Loss is weighted by the IoL between predicted and ground truth boxes
- **Formula**: `L_bbox = IoL_weight × MSE(tx, ty, tw, th)`

### Option 2: IoL-weighted MSE + Trainable Anchor Prediction (Full MultiGridDet)
- **Complete MultiGridDet Implementation**: Includes all paper contributions
- **Anchor Prediction Loss**: Cross-entropy loss for anchor selection
- **Combined Loss**: `L_total = L_class + L_bbox + L_anchor + L_obj`
- **End-to-end Learning**: All components learned jointly

### Option 3: GIoU/DIoU/CIoU Losses (Modern Approach)
- **GIoU Loss**: Generalized IoU for better box regression
- **DIoU Loss**: Distance IoU considering center distance
- **CIoU Loss**: Complete IoU with aspect ratio consideration
- **Modern Standard**: State-of-the-art localization losses

### Loss Configuration in Training

The loss function can be configured in `configs/train_config.yaml`:

```yaml
training:
  loss_option: 2  # 1, 2, or 3
  loss_weights:
    classification: 1.0
    bbox: 1.0
    anchor: 0.5
    objectness: 1.0
```

### Why IoL Instead of IoU?

- **Stability**: IoL is more stable for small objects and edge cases
- **Better Gradients**: Smoother gradients during training
- **Multi-Grid Compatibility**: Works better with the 3×3 neighborhood assignment
- **Reduced Sensitivity**: Less sensitive to box size variations

## Key Advantages of MultiGridDet

### Compared to Standard YOLO

1. **Improved Localization Accuracy**
   - Multiple grid cells provide redundant supervision for the same object
   - Reduces prediction noise through multi-view consensus
   - Smoother bounding box convergence during training

2. **Better Positive/Negative Balance**
   - 9× more positive samples per object compared to single-cell assignment
   - Reduces label sparsity, especially for small objects
   - More stable training dynamics

3. **Faster Convergence**
   - Redundant supervision accelerates learning
   - Multiple perspectives on the same object improve gradient flow
   - Trainable anchors eliminate manual tuning

4. **Parameter Efficiency**
   - DenseYOLO-inspired compact head reduces parameters
   - Dynamic anchor selection instead of fixed anchor assignment
   - End-to-end learning of all components

### Intuitive Summary

> MultiGridDet transforms the way YOLO assigns responsibility. Instead of a **single grid-cell → one object** mapping, it creates a **small neighborhood collaboration**, where each of the 3×3 cells around the object's center helps predict and refine the same bounding box. Combined with **trainable anchors**, this leads to smoother localization, fewer missed detections, and faster, lighter training.

## Performance

- **Model Size**: ~45M parameters
- **Input Resolution**: 608x608
- **Inference Speed**: ~30 FPS on GTX 1060
- **Training**: Supports multi-GPU training

## Troubleshooting

### Input Shape Mismatch
**Error**: `expected shape=(None, 416, 416, 3), found shape=(1, 608, 608, 3)`

**Solution**: Update `input_shape` in model config to match your weights:
```yaml
preset:
  input_shape: [608, 608, 3]  # Match your trained model
```

### Out of Memory
**Solution**: Reduce batch size in training config:
```yaml
training:
  batch_size: 4  # Reduce from 16
```

### No GPU Found
**Note**: MultiGridDet automatically falls back to CPU if no GPU is available.

## Performance Tips

1. **Use TensorBoard** to monitor training:
```bash
tensorboard --logdir logs/tensorboard
```

2. **Enable XLA compilation** (already enabled by default)

3. **Use mixed precision training** (future enhancement)

4. **Distributed training** on multiple GPUs (future enhancement)

## Example Workflows

### Workflow 1: Quick Inference Test
```bash
# Test on a single image
python infer.py --input examples/images/dog.jpg

# Test on all images in a directory
python infer.py --input examples/images/ --type directory
```

### Workflow 2: Transfer Learning
```bash
# Stage 1: Train with frozen backbone
python train.py --config configs/train_config.yaml \
                --weights weights/model5.h5 \
                --epochs 50

# Stage 2: Fine-tune all layers
python train.py --config configs/train_config.yaml \
                --weights logs/checkpoints/ep050-*.h5 \
                --epochs 100
```

### Workflow 3: Model Evaluation
```bash
# Evaluate on validation set
python eval.py --config configs/eval_config.yaml \
               --weights trained_models/final_model.h5

# Evaluate on custom dataset
python eval.py --weights trained_models/final_model.h5 \
               --data data/custom_test.txt
```

## Python API Usage

### Inference

```python
import numpy as np
from PIL import Image
from multigriddet.models import build_multigriddet_darknet
from multigriddet.utils.anchors import load_anchors, load_classes
from multigriddet.postprocess.multigrid_decode import MultiGridDecoder
from multigriddet.utils.preprocessing import preprocess_image

# Load model
model, _ = build_multigriddet_darknet(
    input_shape=(608, 608, 3),
    num_anchors_per_head=[3, 3, 3],
    num_classes=80,
    weights_path='weights/model5.h5'
)

# Load image
image = Image.open('examples/images/dog.jpg')
image_data = preprocess_image(image, (608, 608))

# Run inference
predictions = model.predict(image_data)

# Post-process results
anchors = load_anchors('configs/yolov3_coco_anchor.txt')
class_names = load_classes('configs/coco_classes.txt')

decoder = MultiGridDecoder(
    anchors=anchors,
    num_classes=80,
    input_shape=(608, 608),
    rescore_confidence=True
)

boxes, classes, scores = decoder.postprocess(
    predictions, 
    tuple(reversed(image.size)),  # (height, width)
    (608, 608),
    return_xyxy=True
)

print(f"Found {len(boxes)} objects")
```

### Training

```python
from multigriddet.models import build_multigriddet_darknet_train

# Create training model
model, backbone_len = build_multigriddet_darknet_train(
    anchors=anchors,
    num_classes=80,
    input_shape=(608, 608, 3),
    loss_option=2  # Trainable anchor prediction
)

# Train the model
model.fit(train_data, epochs=100)
```

## Citation

If you use MultiGridDet in your research, please cite:

```bibtex
@INPROCEEDINGS{9730183,
  author={Tesema, Solomon Negussie and Bourennane, El-Bay},
  booktitle={2021 IEEE Intl Conf on Dependable, Autonomic and Secure Computing, Intl Conf on Pervasive Intelligence and Computing, Intl Conf on Cloud and Big Data Computing, Intl Conf on Cyber Science and Technology Congress (DASC/PiCom/CBDCom/CyberSciTech)}, 
  title={Multi-Grid Redundant Bounding Box Annotation for Accurate Object Detection}, 
  year={2021},
  pages={145-152},
  keywords={Training;Image segmentation;Annotations;Object detection;Detectors;Big Data;Object tracking;Object detection;Multi-grid assignment;Copy-Paste Image augmentation},
  doi={10.1109/DASC-PICom-CBDCom-CyberSciTech52372.2021.00036}}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the original MultiGridDet paper
- Built with TensorFlow 2.17+ and Keras 3.0
- Inspired by YOLO and modern object detection frameworks

## Support

For questions and support, please open an issue on GitHub.

---

**MultiGridDet** - Modern Multi-Grid Object Detection