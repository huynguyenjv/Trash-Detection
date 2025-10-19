# TACO Dataset Preprocessing và YOLOv8 Training

Hệ thống preprocessing dữ liệu TACO và training YOLOv8 cho bài toán phát hiện rác thải.

## Cấu trúc dữ liệu

```
training-model/
├── data/
│   └── detection/
│       └── raw/
│           └── data/
│               ├── annotations.json          # COCO annotations
│               ├── batch_1/ ... batch_15/   # Thư mục chứa ảnh
│               ├── all_image_urls.csv
│               └── annotations_unofficial.json
└── data/processed/
    └── detection/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── dataset.yaml
        └── validation_results.json
```

## Classes (7 loại rác)

1. **cardboard** - Giấy carton, hộp giấy
2. **glass** - Thủy tinh
3. **metal** - Kim loại (lon, nắp)
4. **organic** - Rác hữu cơ
5. **other** - Khác (pin, giày, thuốc lá...)
6. **paper** - Giấy
7. **plastic** - Nhựa

## 📁 Project Structure

```
training-model/
├── 📂 data/                           # Dataset storage
│   ├── detection/
│   │   ├── raw/                       # TACO dataset raw
│   │   └── processed/                 # YOLO format data
│   └── classification/
│       ├── raw/                       # TrashNet dataset raw
│       └── processed/                 # Processed classification data
├── 📂 models/                         # Trained models
│   ├── detection/                     # Detection model weights
│   └── classification/                # Classification model weights
├── 📂 results/                        # Training results
│   ├── detection/                     # Detection training results
│   ├── classification/                # Classification training results
│   └── evaluation/                    # Evaluation reports & plots
├── 📂 configs/                        # Configuration files
│   └── training_config.yaml          # Main training configuration
├── 📂 logs/                          # Log files
└── 📜 Training Scripts
    ├── main.py                        # 🚀 Main training pipeline
    ├── data_preprocessing_detection.py    # TACO dataset processing
    ├── data_preprocessing_classification.py # TrashNet dataset processing
    ├── train_detection.py             # Detection model training
    ├── train_classification.py        # Classification model training
    ├── detect.py                      # Real-time detection pipeline
    └── evaluate.py                    # Comprehensive evaluation system
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install ultralytics opencv-python matplotlib seaborn pandas scikit-learn tqdm pyyaml
pip install pycocotools pillow requests

# Hoặc sử dụng requirements.txt nếu có
pip install -r requirements.txt
```

### 2. Run Full Training Pipeline

```bash
# Chạy toàn bộ training pipeline
python main.py --config configs/training_config.yaml --full-pipeline
```

### 3. Run Individual Steps

```bash
# Chỉ preprocessing
python main.py --config configs/training_config.yaml --steps preprocessing

# Detection + Classification training
python main.py --config configs/training_config.yaml --steps detection,classification

# Evaluation only
python main.py --config configs/training_config.yaml --steps evaluation
```

## 📊 Training Pipeline Steps

### Step 1: Data Preprocessing
```bash
# TACO Dataset (Detection)
python data_preprocessing_detection.py --base-dir data/detection/raw --output-dir data/detection/processed

# TrashNet Dataset (Classification)  
python data_preprocessing_classification.py --base-dir data/classification/raw --output-dir data/classification/processed
```

**Features:**
- ✅ TACO dataset download & processing
- ✅ COCO → YOLO format conversion
- ✅ TrashNet dataset processing
- ✅ Automatic train/val/test splitting
- ✅ Class mapping & statistics

### Step 2: Detection Model Training
```bash
python train_detection.py --model yolov8n.pt --data data/detection/processed/dataset_detection.yaml --epochs 100
```

**Features:**
- ✅ YOLOv8 detection training
- ✅ Hyperparameter optimization
- ✅ Validation & metrics tracking
- ✅ Training visualization plots
- ✅ Model checkpoint saving

### Step 3: Classification Model Training
```bash
python train_classification.py --model yolov8n-cls.pt --data data/classification/processed/dataset_classification.yaml --epochs 50
```

**Features:**
- ✅ YOLOv8 classification fine-tuning
- ✅ Comprehensive evaluation với confusion matrix
- ✅ Per-class accuracy analysis
- ✅ Training progress visualization

### Step 4: Real-time Detection Pipeline
```bash
# Webcam detection
python detect.py --source 0

# Video file
python detect.py --source video.mp4 --output output.mp4

# Image file
python detect.py --source image.jpg --output result.jpg
```

**Features:**
- ✅ 2-stage pipeline integration
- ✅ Threading optimization for real-time
- ✅ Confidence threshold filtering
- ✅ Multi-worker classification processing
- ✅ Performance monitoring

### Step 5: Comprehensive Evaluation
```bash
python evaluate.py --detection-model models/detection/best.pt --classification-model models/classification/best.pt
```

**Features:**
- ✅ Detection model evaluation (mAP, precision, recall)
- ✅ Classification model evaluation (accuracy, F1-score)
- ✅ Pipeline performance analysis
- ✅ Visualization plots & reports
- ✅ Multi-threshold analysis

## ⚙️ Configuration

Cấu hình toàn bộ system thông qua `configs/training_config.yaml`:

```yaml
# Detection Model Settings
detection:
  model_name: "yolov8n.pt"
  epochs: 100
  batch_size: 16
  img_size: 640
  learning_rate: 0.01

# Classification Model Settings  
classification:
  model_name: "yolov8n-cls.pt"
  epochs: 50
  batch_size: 32
  img_size: 224
  learning_rate: 0.001

# Pipeline Settings
pipeline:
  detection_conf_threshold: 0.25
  classification_conf_threshold: 0.5
  max_workers: 4
  batch_classification: true
```

## 📈 Performance Metrics

### Detection Model
- **mAP@50**: Mean Average Precision tại IoU=0.5
- **mAP@50-95**: Mean Average Precision từ IoU=0.5 đến 0.95
- **Precision/Recall**: Precision và Recall cho từng class
- **F1-Score**: Harmonic mean của precision và recall

### Classification Model
- **Top-1 Accuracy**: Accuracy cho prediction hàng đầu
- **Top-5 Accuracy**: Accuracy trong top-5 predictions
- **Per-class Metrics**: Precision, recall, F1-score cho từng class
- **Confusion Matrix**: Ma trận confusion cho analysis chi tiết

### Pipeline Performance
- **FPS**: Frames per second processing speed
- **Classification Rate**: Tỷ lệ objects được classify thành công
- **Processing Time**: Average processing time per frame/image

## 🔧 Advanced Usage

### Custom Dataset Training

1. **Chuẩn bị dataset**:
   ```bash
   # Detection: COCO format
   data/detection/raw/
   ├── images/
   ├── annotations/
   └── classes.txt
   
   # Classification: Folder structure
   data/classification/raw/
   ├── class1/
   ├── class2/
   └── ...
   ```

2. **Update configuration**:
   ```yaml
   datasets:
     taco:
       base_dir: "path/to/custom/detection/data"
     trashnet:  
       base_dir: "path/to/custom/classification/data"
   ```

### Model Customization

```yaml
# Sử dụng models lớn hơn
detection:
  model_name: "yolov8m.pt"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

classification:
  model_name: "yolov8m-cls.pt"  # yolov8n-cls, yolov8s-cls, yolov8m-cls, etc.
```

### Hardware Optimization

```yaml
hardware:
  gpu_memory_fraction: 0.8
  mixed_precision: true
  num_workers: 8
  auto_batch_size: true
```

## 📊 Results Analysis

### Training Results
- **Detection**: `results/detection/detection_v1/`
  - Weights: `weights/best.pt`, `weights/last.pt`
  - Plots: Training curves, validation metrics
  - Logs: Training logs và configuration

- **Classification**: `results/classification/classification_v1/`
  - Weights: `weights/best.pt`, `weights/last.pt`
  - Confusion Matrix: `confusion_matrix.png`
  - Training plots: Accuracy, loss curves

### Evaluation Results
- **Comprehensive Report**: `results/evaluation/evaluation_v1_results.json`
- **Visualization Plots**: 
  - Detection threshold analysis
  - Classification confusion matrix
  - Pipeline performance metrics
  - Combined model comparison

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```yaml
   # Giảm batch size
   detection:
     batch_size: 8
   classification:
     batch_size: 16
   ```

2. **Dataset Download Failures**:
   ```bash
   # Manual download và extract datasets
   # Cập nhật paths trong config file
   ```

3. **Performance Issues**:
   ```yaml
   # Optimize threading
   pipeline:
     max_workers: 2
     batch_classification: false
   ```

### Debug Mode

```bash
# Enable verbose logging
python main.py --config configs/training_config.yaml --full-pipeline --verbose

# Check logs
tail -f logs/main_pipeline.log
```

## 📝 Examples

### Example 1: Quick Training
```bash
# Small dataset, fast training
python main.py --config configs/training_config.yaml --steps preprocessing,detection --epochs 10
```

### Example 2: High Accuracy Training
```bash
# Full dataset, nhiều epochs
python train_detection.py --model yolov8l.pt --epochs 200 --batch 8
python train_classification.py --model yolov8l-cls.pt --epochs 100 --batch 16
```

### Example 3: Real-time Detection
```bash
# Webcam detection với custom confidence
python detect.py --source 0 --conf-det 0.3 --conf-cls 0.7 --device cuda
```

## 🚀 Production Deployment

### Model Export
```bash
# Export để deployment
from ultralytics import YOLO

# Detection model
model = YOLO('models/detection/best.pt')
model.export(format='onnx')  # hoặc 'tensorrt', 'tflite'

# Classification model  
model = YOLO('models/classification/best.pt')
model.export(format='onnx')
```

### API Integration
```python
from detect import TrashDetectionPipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig(
    detection_model_path="models/detection/best.pt",
    classification_model_path="models/classification/best.pt"
)
pipeline = TrashDetectionPipeline(config)

# Process image
annotated_frame, detections, performance = pipeline.process_frame(image)
```

## 📚 References

- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **TACO Dataset**: [TACO: Trash Annotations in Context](http://tacodataset.org/)
- **TrashNet Dataset**: [TrashNet Dataset](https://github.com/garythung/trashnet)

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Implement changes với proper testing
4. Submit pull request với detailed description

## 📄 License

Tuân theo license của project gốc.

## 📞 Support

Nếu gặp issues hoặc có questions:
1. Check troubleshooting section
2. Review log files trong `logs/`
3. Create issue với detailed description và logs

---

**Happy Training! 🚀**

*Generated by GitHub Copilot Assistant - Implementation of instruction.md specification*