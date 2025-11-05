# CSPS_RetinaNet: An Enhanced RetinaNet for Rodent Burrow Detection in Grassland Ecosystems

## Overview

CSPS_RetinaNet is an improved object detection model based on RetinaNet, specifically designed for detecting rodent burrows in grassland ecosystems using high-resolution satellite imagery. This repository contains the implementation of our novel architecture that integrates CSPDarknet53 backbone with SimAM attention mechanism and GIoU loss for enhanced detection performance.

## Contributors

**Haitao Sun**<sup>a</sup>, **Songbing Zou**<sup>a*</sup>, **Zhenqing Ji**<sup>a</sup>, **Yajie Bai**<sup>a</sup>, **Wenyong Zhang**<sup>a</sup>, **Tenghao Gou**<sup>a</sup>, **Pengxiang Xie**<sup>a</sup>

<sup>a</sup> Key Laboratory of Western China's Environmental Systems (Ministry of Education), College of Earth and Environmental Sciences, Lanzhou University, Lanzhou, 730000, China.

*Corresponding author

## Key Features

- **CSPDarknet53 Backbone**: Efficient feature extraction with cross-stage partial connections
- **SimAM Attention Mechanism**: Parameter-free spatial attention module for enhanced feature representation
- **GIoU Loss**: Improved bounding box regression for better localization accuracy
- **Feature Pyramid Network (FPN)**: Multi-scale feature fusion for detecting burrows of various sizes

## Model Architecture

The CSPS_RetinaNet architecture consists of:
1. **Backbone Network**: CSPDarknet53 for robust feature extraction
2. **Attention Module**: SimAM applied at C3 feature level
3. **Neck**: Feature Pyramid Network (FPN) generating P3-P7 feature levels
4. **Detection Heads**: Separate classification and regression branches

## Installation

### Requirements

```bash
Python >= 3.7
PyTorch >= 1.7.1
CUDA >= 10.2 (for GPU support)
```

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- torchvision
- numpy
- opencv-python
- pillow
- matplotlib
- scipy
- tqdm
- pycocotools

## Dataset

### Data Structure

Organize your dataset in VOC format:
```
VOCdevkit/
└── VOC2007/
    ├── Annotations/     # XML annotation files
    ├── JPEGImages/      # Image files
    └── ImageSets/
        └── Main/
            ├── train.txt
            ├── val.txt
            └── test.txt
```

### Data Preparation

1. Prepare your images and annotations in VOC format
2. Run the annotation script to generate training files:

```bash
python voc_annotation.py
```

This will generate `2007_train.txt` and `2007_val.txt` for training.

## Configuration

### Model Configuration

Edit `model_data/cls_classes.txt` to define your class names (one class per line).

### Training Configuration

Key parameters in `train.py`:
- `input_shape`: [600, 600] - Input image size
- `phi`: 5 - Use CSPDarknet53 backbone
- `Init_lr`: 1e-4 - Initial learning rate (Adam optimizer)
- `Freeze_Epoch`: 60 - Epochs for frozen training
- `UnFreeze_Epoch`: 120 - Total training epochs
- `batch_size`: Adjust based on your GPU memory

## Usage

### Training

1. **Prepare your dataset** following the VOC format structure

2. **Configure training parameters** in `train.py`:
   - Set `classes_path` to your classes file
   - Set `model_path` to pretrained weights (optional)
   - Adjust `batch_size` based on GPU memory
   - Configure training epochs

3. **Start training**:
```bash
python train.py
```

The model will automatically:
- Download pretrained backbone weights
- Perform frozen training (backbone frozen)
- Perform unfrozen training (full model training)
- Save checkpoints in `logs/` directory
- Generate loss curves and evaluation metrics

### Prediction

For single image prediction:
```python
from retinanet import Retinanet

# Initialize model
model = Retinanet()

# Predict on image
from PIL import Image
image = Image.open('your_image.jpg')
result = model.detect_image(image)
result.show()
```

For batch prediction:
```bash
# Set mode to "dir_predict" in predict.py
python predict.py
```

### Evaluation

Calculate mAP on test set:
```bash
python get_map.py
```

This will generate:
- Detection results for each image
- Precision-Recall curves
- mAP scores
- F1 scores, Recall, and Precision metrics

## Model Performance

The model achieves state-of-the-art performance on rodent burrow detection:
- Optimized for high-resolution (3000×3000) Sentinel-2 imagery
- Robust detection of small objects (burrows)
- Improved localization accuracy with GIoU loss

## Advanced Features

### Mixed Precision Training (FP16)

Enable FP16 for faster training with reduced memory:
```python
fp16 = True  # in train.py
```

### Multi-GPU Training

For distributed training:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

Set `distributed = True` in `train.py`.

### Model Summary

View model architecture and parameters:
```bash
python summary.py
```

## Data Availability

- **Rodent Burrow Dataset**: Available at [https://zenodo.org/records/17491180](https://zenodo.org/records/17491180)
- **Trained Model Weights**: Available at [https://zenodo.org/records/17491180](https://zenodo.org/records/17491180)
- **Implementation Code**: Available at [https://github.com/yeshushahua/CSPS_RetinaNet](https://github.com/yeshushahua/CSPS_RetinaNet)

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{sun2025csps,
  title={CSPS_RetinaNet: An Enhanced RetinaNet for Rodent Burrow Detection in Grassland Ecosystems},
  author={Sun, Haitao and Zou, Songbing and Ji, Zhenqing and Bai, Yajie and Zhang, Wenyong and Gou, Tenghao and Xie, Pengxiang},
  journal={},
  year={2025},
  publisher={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please contact:
- **Songbing Zou** (Corresponding Author): [email]
- Open an issue on GitHub: [https://github.com/yeshushahua/CSPS_RetinaNet/issues](https://github.com/yeshushahua/CSPS_RetinaNet/issues)
---

**Last Updated**: November 2025
