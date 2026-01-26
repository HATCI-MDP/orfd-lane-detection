# Off-Road Lane Detection

A deep learning system for detecting traversable road areas in off-road environments using U-Net semantic segmentation. The model identifies drivable paths and overlays them with a green highlight for visualization.

## Features

- **U-Net architecture** with two variants (small and full)
- **Real-time inference** on images, videos, and webcam feeds
- **Green overlay visualization** of traversable road areas
- **Training pipeline** with mixed precision support and early stopping
- **ORFD dataset** support (Off-Road Freespace Detection)

## Model Variants

| Model       | Parameters | Input Size | GPU Memory |
| ----------- | ---------- | ---------- | ---------- |
| U-Net Small | 7.7M       | 256x256    | ~2GB       |
| U-Net Full  | 31M        | 256x256    | ~3GB       |

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/bryansbtian/orfd-lane-detection.git
cd "Off Road"

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Install PyTorch with CUDA support for GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Dataset Setup

Extract the ORFD dataset:

```bash
# Extract the dataset zip file
unzip ORFD.zip -d datasets/
```

This will create the following structure:

```
datasets/
└── ORFD/
    ├── training/
    │   ├── image_data/     # RGB images (*.png)
    │   └── gt_image/       # Ground truth masks (*_fillcolor.png)
    ├── validation/
    │   └── ...
    └── testing/
        └── ...
```

## Usage

### Training

```bash
# Train U-Net Small (recommended)
python train_unet.py --model unet_small --epochs 50 --batch_size 8 --amp

# Train U-Net Full (more parameters)
python train_unet.py --model unet --epochs 50 --batch_size 4 --amp
```

#### Training Options

| Parameter      | Type  | Default         | Description                         |
| -------------- | ----- | --------------- | ----------------------------------- |
| `--model`      | str   | `unet_small`    | `unet_small` (7.7M) or `unet` (31M) |
| `--img_size`   | int   | 256             | Input image size                    |
| `--batch_size` | int   | 8               | Batch size                          |
| `--epochs`     | int   | 50              | Number of epochs                    |
| `--lr`         | float | 1e-4            | Learning rate                       |
| `--patience`   | int   | 15              | Early stopping patience             |
| `--amp`        | flag  | False           | Enable mixed precision training     |
| `--data_root`  | str   | `datasets/ORFD` | Dataset root directory              |

Training creates a timestamped folder in `checkpoints/` containing:

- `best_model.pth` - Model with best validation IoU
- `latest_model.pth` - Most recent checkpoint
- `training_history.png` - Loss and metrics plots

### Demo / Inference

```bash
# Run on test dataset samples
python demo.py --checkpoint checkpoints/unet_XXXXXX/best_model.pth

# Run on single image
python demo.py --checkpoint path/to/model.pth --mode image --input photo.jpg --output result.png

# Run on video
python demo.py --checkpoint path/to/model.pth --mode video --input video.mp4 --output output.mp4

# Run with webcam
python demo.py --checkpoint path/to/model.pth --mode webcam
```

#### Demo Options

| Parameter       | Type  | Default         | Description                           |
| --------------- | ----- | --------------- | ------------------------------------- |
| `--checkpoint`  | str   | **required**    | Path to trained model checkpoint      |
| `--model`       | str   | `unet_small`    | `unet_small` or `unet`                |
| `--mode`        | str   | `dataset`       | `image`, `video`, `webcam`, `dataset` |
| `--input`       | str   | None            | Input path (for image/video modes)    |
| `--output`      | str   | None            | Output path                           |
| `--img_size`    | int   | 256             | Input image size                      |
| `--threshold`   | float | 0.5             | Prediction threshold                  |
| `--data_root`   | str   | `datasets/ORFD` | Dataset root (for dataset mode)       |
| `--num_samples` | int   | 5               | Number of samples (for dataset mode)  |

## Project Structure

```
Off Road/
├── demo.py              # Demo/inference script
├── train.py             # Training script
├── model.py             # U-Net architecture
├── dataset.py           # Dataset loader
├── requirements.txt     # Dependencies
├── checkpoints/         # Saved models (created during training)
└── datasets/ORFD/       # Dataset
```

## Tips

### Memory Issues

- Use `unet_small` instead of `unet`
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--img_size` (try 128)
- Enable `--amp` for mixed precision

### Better Results

- Lower `--threshold` (e.g., 0.3) for more sensitive detection
- Higher `--threshold` (e.g., 0.7) for more confident predictions
- Train longer with more `--epochs`

## Metrics

- **IoU (Intersection over Union)**: Measures overlap between prediction and ground truth
- **Dice Score**: Similar to IoU, emphasizes overlap
- **Loss**: BCE + Dice loss combination
