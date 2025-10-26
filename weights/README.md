# Model Weights

This directory contains pre-trained model weights for MultiGridDet.

## Available Weights

### model5.h5
- **Description**: Pre-trained MultiGridDet model on COCO dataset
- **Input Size**: 608x608
- **Classes**: 80 COCO classes
- **Parameters**: ~45M
- **Performance**: mAP@0.5: 0.42, mAP@0.5:0.95: 0.24

## Download Instructions

To download the pre-trained weights:

```bash
# Create weights directory if it doesn't exist
mkdir -p weights

# Download model5.h5 (replace with actual download link)
wget https://your-domain.com/model5.h5 -O weights/model5.h5
```

## Usage

```python
from multigriddet.models import build_multigriddet_darknet

# Load model with pre-trained weights
model, _ = build_multigriddet_darknet(
    input_shape=(608, 608, 3),
    num_anchors_per_head=[3, 3, 3],
    num_classes=80,
    weights_path='weights/model5.h5'
)
```

## Training Your Own Weights

To train your own model:

```python
from multigriddet.models import build_multigriddet_darknet_train

# Create training model
model, _ = build_multigriddet_darknet_train(
    anchors=anchors,
    num_classes=80,
    input_shape=(608, 608, 3),
    loss_option=2  # Trainable anchor prediction
)

# Train and save
model.fit(train_data, epochs=100)
model.save_weights('weights/my_model.h5')
```

## File Sizes

- `model5.h5`: ~180 MB
- `model5.weights` (Darknet format): ~180 MB

## Notes

- All weights are trained on COCO 2017 dataset
- Models use Darknet53 backbone
- Input images should be resized to 608x608
- Weights are compatible with TensorFlow 2.17+





