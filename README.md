# Aircraft Detection in Satellite Images using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **State-of-the-art aircraft detection in high-resolution satellite imagery using YOLOv8 deep learning architecture**

![Aircraft Detection Results](images/Aircraft_predictions.png)

## 🎯 Project Overview

Aircraft detection from Earth observation satellite images is crucial for monitoring airport activities and mapping aircraft locations. While manual digitization is accurate, it becomes impractical for large regions or when fast assessments are required. This project implements an automated aircraft detection system for high-resolution satellite imagery using YOLOv8, addressing the challenge of monitoring airport activities and aircraft locations at scale. The solution processes 2560×2560 pixel satellite images from Airbus' Pleiades twin satellites, achieving reliable detection through advanced deep learning techniques.

### Key Features
- ✈️ **High-Resolution Processing**: Handles 2560×2560 pixel satellite images
- 🎯 **Precision Detection**: YOLOv8-based object detection with configurable confidence thresholds
- 🔄 **Smart Tiling**: Automated image tiling with overlap handling for large-scale imagery
- 🚀 **Data Augmentation**: Advanced augmentation pipeline using Albumentations
- 📊 **Interactive Inference**: Real-time detection with adjustable parameters
- 🏗️ **MLOps Ready**: Modular architecture with configuration management

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Google Colab (recommended) or local GPU environment
- Access to [Airbus Aircraft Dataset](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/aircraft-detection-yolov8.git
   cd aircraft-detection-yolov8
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download the [Airbus Aircraft Dataset](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset)
   - Extract to `data/` directory

### Basic Usage

```python
from config import Config
import utils

# Initialize configuration
config = Config()

# Load and preprocess data
annotations_df = pd.read_csv("data/annotations.csv")

# Generate training tiles
utils.generate_tiles(img_path="path/to/image.jpg", 
                    df=annotations_df, 
                    val_indexes=val_indexes)

# Train YOLOv8 model
model = utils.train_yolov8_obj_detect(
    data_yaml="data.yaml",
    epochs=20,
    imgsz=512
)

# Run inference
results, metrics = utils.load_and_apply_model(
    image_paths="path/to/test_image.jpg",
    best_model_path="runs/detect/train/weights/best.pt",
    conf=0.7,
    iou=0.65
)
```

## 📁 Repository Structure

```
aircraft-detection-yolov8/
├── 📄 airplane_detection.ipynb     # Main Jupyter notebook
├── 🐍 config.py                   # Configuration management
├── 🛠️ utils.py                    # Utility functions and core logic
├── 📋 data.yaml                   # YOLOv8 dataset configuration
├── ⚙️ constants.yaml              # Data augmentation parameters
├── 📊 data/                       # Dataset directory
│   ├── images/                    # Original satellite images
│   ├── annotations.csv            # Aircraft annotations
│   ├── train/                     # Training tiles and labels
│   ├── val/                       # Validation tiles and labels
│   ├── train_aug/                 # Augmented training data
│   └── predictions/               # Model predictions output
├── 🏃 runs/                       # Training runs and model weights
├── 🖼️ images/                     # Documentation images
└── 📜 README.md                   # Project documentation
```

## 🔧 Technical Implementation

### Architecture Overview
- **Model**: YOLOv8s (Small variant for optimal speed/accuracy balance)
- **Input Resolution**: 512×512 pixel tiles
- **Tiling Strategy**: Overlapping tiles with 64-pixel overlap
- **Data Augmentation**: Albumentations library with geometric and photometric transforms

### Key Components

#### 1. Image Preprocessing Pipeline
```python
# Configuration parameters
TILE_WIDTH = 512
TILE_HEIGHT = 512  
TILE_OVERLAP = 64
TRUNCATED_PERCENT = 0.3
```

#### 2. Data Augmentation
- Random cropping and resizing
- Horizontal and vertical flipping
- Brightness and contrast adjustment
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

#### 3. Training Configuration
- **Epochs**: 20
- **Image Size**: 512×512
- **Batch Size**: Auto-determined by YOLOv8
- **Optimizer**: AdamW (YOLOv8 default)

## 📊 Results & Performance

### Model Metrics
| Metric | Value |
|--------|-------|
| mAP@0.5 | TBD* |
| mAP@0.5:0.95 | TBD* |
| Precision | TBD* |
| Recall | TBD* |
| Training Time | ~2-3 hours (GPU) |

*Results available after training completion

### Key Achievements
- ✅ Successfully processes high-resolution satellite imagery
- ✅ Automated tile generation with smart overlap handling
- ✅ Robust data augmentation pipeline
- ✅ Interactive inference system with real-time parameter adjustment
- ✅ Modular, maintainable codebase structure

## 🛠️ Advanced Configuration

### Custom Training Parameters
```python
# Modify config.py for custom settings
class Config:
    def __init__(self):
        self.tile_height = 512          # Tile dimensions
        self.tile_width = 512
        self.image_width = 2560         # Original image size
        self.image_height = 2560
        self.tile_overlap = 64          # Overlap between tiles
        self.truncated_percent = 0.3    # Min object visibility
```

### Environment Setup
```bash
# For Google Colab
!pip install ultralytics albumentations pybboxes imgaug

# Enable GPU runtime: Runtime > Change runtime type > GPU
```

## 🔬 Methodology

### 1. Data Collection
- **Source**: Airbus Intelligence satellite imagery
- **Resolution**: High-resolution Pleiades twin satellite data
- **Annotations**: Pre-labeled aircraft bounding boxes

### 2. Preprocessing Pipeline
- Image tiling with intelligent overlap management
- YOLO format label conversion
- Quality filtering based on truncation thresholds

### 3. Model Training
- Transfer learning from COCO-pretrained YOLOv8s
- Custom single-class detection (Aircraft)
- Automated hyperparameter optimization

### 4. Evaluation & Validation
- 5-fold cross-validation setup
- Comprehensive metrics tracking
- Visual validation through batch predictions

## 🚀 Future Enhancements

- [ ] **Multi-class Detection**: Extend to different aircraft types
- [ ] **Model Optimization**: Implement model quantization for deployment
- [ ] **Real-time Pipeline**: Streaming inference for satellite feeds
- [ ] **Cloud Deployment**: AWS/GCP deployment with API endpoints
- [ ] **Performance Benchmarking**: Comprehensive accuracy vs. speed analysis

## 📚 References & Acknowledgments

- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Dataset**: [Airbus Aircraft Sample Dataset](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset)
- **Inspiration**: Jeff Faudi's [Aircraft Detection with YOLOv5](https://www.kaggle.com/code/jeffaudi/aircraft-detection-with-yolov5)
- **Augmentation**: [Albumentations Library](https://albumentations.ai/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
