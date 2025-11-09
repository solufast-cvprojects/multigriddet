---
layout: default
title: MultiGridDet
description: A modern implementation of Multi-Grid Object Detection with 3×3 redundant bounding-box assignment
---

# MultiGridDet

A modern implementation of Multi-Grid Object Detection, featuring efficient multi-grid cell annotation and trainable anchor prediction.

## Quick Links

- [GitHub Repository](https://github.com/solufast-cvprojects/multigriddet)
- [Installation Guide](#installation)
- [Quick Start](#quick-start)
- [Documentation](https://github.com/solufast-cvprojects/multigriddet#readme)
- [IEEE Paper](https://ieeexplore.ieee.org/document/9730183)

## Overview

MultiGridDet is an enhanced one-stage object detector derived from YOLOv3, introducing **multi-grid redundant bounding-box assignment** for tighter and more stable object localization. Unlike conventional YOLO approaches where only the grid cell containing the object's center is responsible for prediction, MultiGridDet extends this responsibility to the **3×3 grid neighborhood** (center cell plus its eight adjacent cells).

## Key Features

- **Multi-Grid Redundant Assignment**: 3×3 neighborhood cells collaborate to predict the same object
- **Trainable Anchors**: Dynamic anchor selection learned end-to-end
- **Multiple Loss Functions**: IoL-weighted MSE, GIoU/DIoU/CIoU losses
- **Easy Deployment**: Simple inference API for images, videos, and camera
- **Research Ready**: Full training pipeline with academic-quality visualizations

## Installation

```bash
# Clone the repository
git clone https://github.com/solufast-cvprojects/multigriddet.git
cd MultiGridDet

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Inference

```bash
python infer.py --input examples/images/dog.jpg
```

### Training

```bash
python train.py --config configs/train_config.yaml
```

### Evaluation

```bash
python eval.py --config configs/eval_config.yaml
```

## Citation

If you use MultiGridDet in your research, please cite:

```bibtex
@INPROCEEDINGS{9730183,
  author={Tesema, Solomon Negussie and Bourennane, El-Bay},
  booktitle={2021 IEEE Intl Conf on Dependable, Autonomic and Secure Computing},
  title={Multi-Grid Redundant Bounding Box Annotation for Accurate Object Detection}, 
  year={2021},
  pages={145-152},
  doi={10.1109/DASC-PICom-CBDCom-CyberSciTech52372.2021.00036}
}
```

## License

This project is licensed under the MIT License.

---

For more information, visit the [GitHub repository](https://github.com/solufast-cvprojects/multigriddet).

