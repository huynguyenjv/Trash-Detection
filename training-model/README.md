# Trash Detection Training System (Integrated Version)# TACO Dataset Preprocessing và YOLOv8 Training



## 📁 Project Structure (Cleaned)Hệ thống preprocessing dữ liệu TACO và training YOLOv8 cho bài toán phát hiện rác thải.

```

training-model/## Cấu trúc dữ liệu

├── main.py                           # 🎯 Main integrated training pipeline

├── data_preprocessing_detection.py   # 📊 TACO dataset preprocessing  ```

├── data_preprocessing_classification.py # 📊 TrashNet dataset preprocessingtraining-model/

├── configs/├── data/

│   └── training_config.yaml          # ⚙️ Configuration file│   └── detection/

├── data/│       └── raw/

│   ├── detection/│           └── data/

│   │   ├── raw/                      # Raw TACO dataset│               ├── annotations.json          # COCO annotations

│   │   └── processed/                # Processed YOLOv8 format│               ├── batch_1/ ... batch_15/   # Thư mục chứa ảnh

│   └── classification/│               ├── all_image_urls.csv

│       ├── raw/                      # Raw TrashNet dataset  │               └── annotations_unofficial.json

│       └── processed/                # Processed classification format└── data/processed/

├── models/    └── detection/

│   ├── detection/                    # Detection model weights        ├── images/

│   └── classification/               # Classification model weights        │   ├── train/

├── results/        │   ├── val/

│   ├── detection/                    # Detection training results        │   └── test/

│   ├── classification/               # Classification training results        ├── labels/

│   └── evaluation/                   # Evaluation results        │   ├── train/

├── yolo*.pt                          # Pre-trained YOLO weights        │   ├── val/

└── README.md                         # This file        │   └── test/

```        ├── dataset.yaml

        └── validation_results.json

## 🚀 Quick Start```



### 1. Data Preprocessing## Classes (7 loại rác)

```bash

# Process TACO dataset for detection1. **cardboard** - Giấy carton, hộp giấy

python data_preprocessing_detection.py2. **glass** - Thủy tinh

3. **metal** - Kim loại (lon, nắp)

# Process TrashNet dataset for classification  4. **organic** - Rác hữu cơ

python data_preprocessing_classification.py5. **other** - Khác (pin, giày, thuốc lá...)

```6. **paper** - Giấy

7. **plastic** - Nhựa

### 2. Training Options

## 📁 Project Structure

#### Full Pipeline (Recommended)

```bash```

# Run complete training pipelinetraining-model/

python main.py --config configs/training_config.yaml --full-pipeline├── 📂 data/                           # Dataset storage

```│   ├── detection/

│   │   ├── raw/                       # TACO dataset raw

#### Individual Steps│   │   └── processed/                 # YOLO format data

```bash│   └── classification/

# Train only detection model│       ├── raw/                       # TrashNet dataset raw

python main.py --steps detection│       └── processed/                 # Processed classification data

├── 📂 models/                         # Trained models

# Train only classification model│   ├── detection/                     # Detection model weights

python main.py --steps classification│   └── classification/                # Classification model weights

├── 📂 results/                        # Training results

# Run evaluation only│   ├── detection/                     # Detection training results

python main.py --steps evaluation│   ├── classification/                # Classification training results

│   └── evaluation/                    # Evaluation reports & plots

# Custom combination├── 📂 configs/                        # Configuration files

python main.py --steps detection,classification,evaluation│   └── training_config.yaml          # Main training configuration

```├── 📂 logs/                          # Log files

└── 📜 Training Scripts

### 3. Standalone Operations    ├── main.py                        # 🚀 Main training pipeline

    ├── data_preprocessing_detection.py    # TACO dataset processing

#### Detection Only    ├── data_preprocessing_classification.py # TrashNet dataset processing

```bash    ├── train_detection.py             # Detection model training

# Detect objects in image    ├── train_classification.py        # Classification model training

python main.py --detect --source path/to/image.jpg    ├── detect.py                      # Real-time detection pipeline

    └── evaluate.py                    # Comprehensive evaluation system

# Detect in video```

python main.py --detect --source path/to/video.mp4 --output results/output.mp4

```## 🚀 Quick Start



#### Evaluation Only### 1. Environment Setup

```bash

# Run comprehensive evaluation```bash

python main.py --evaluate# Install dependencies

```pip install ultralytics opencv-python matplotlib seaborn pandas scikit-learn tqdm pyyaml

pip install pycocotools pillow requests

## 📊 Training Components (All Integrated in main.py)

# Hoặc sử dụng requirements.txt nếu có

### Detection Trainingpip install -r requirements.txt

- **Model**: YOLOv8 variants (n/s/m/l/x)```

- **Dataset**: TACO (Trash Annotations in Context)

- **Format**: COCO → YOLO format conversion### 2. Run Full Training Pipeline

- **Classes**: 7 unified trash categories

```bash

### Classification Training  # Chạy toàn bộ training pipeline

- **Model**: YOLOv8-cls variants (n/s/m/l/x)python main.py --config configs/training_config.yaml --full-pipeline

- **Dataset**: TrashNet (Garbage Classification)```

- **Format**: ImageNet-style classification

- **Classes**: 6 trash categories### 3. Run Individual Steps



### Evaluation System```bash

- **Detection Metrics**: mAP@50, mAP@50-95, Precision, Recall, F1# Chỉ preprocessing

- **Classification Metrics**: Top-1/Top-5 Accuracypython main.py --config configs/training_config.yaml --steps preprocessing

- **Multi-threshold Analysis**: Optimal confidence threshold detection

# Detection + Classification training

### Real-time Pipelinepython main.py --config configs/training_config.yaml --steps detection,classification

- **2-Stage Detection**: YOLOv8 Detection + YOLOv8 Classification

- **Threading Optimization**: Parallel processing for real-time performance# Evaluation only

- **Configurable Thresholds**: Detection and classification confidencepython main.py --config configs/training_config.yaml --steps evaluation

```

## ⚙️ Configuration

## 📊 Training Pipeline Steps

Edit `configs/training_config.yaml`:

### Step 1: Data Preprocessing

```yaml```bash

# Detection settings# TACO Dataset (Detection)

detection:python data_preprocessing_detection.py --base-dir data/detection/raw --output-dir data/detection/processed

  model_name: "yolov8n.pt"

  epochs: 100# TrashNet Dataset (Classification)  

  batch_size: 16python data_preprocessing_classification.py --base-dir data/classification/raw --output-dir data/classification/processed

  device: "auto"```



# Classification settings  **Features:**

classification:- ✅ TACO dataset download & processing

  model_name: "yolov8n-cls.pt"- ✅ COCO → YOLO format conversion

  epochs: 50- ✅ TrashNet dataset processing

  batch_size: 32- ✅ Automatic train/val/test splitting

  device: "auto"- ✅ Class mapping & statistics



# Evaluation settings### Step 2: Detection Model Training

evaluation:```bash

  detection_conf_thresholds: [0.1, 0.25, 0.5, 0.75]python train_detection.py --model yolov8n.pt --data data/detection/processed/dataset_detection.yaml --epochs 100

  save_plots: true```



# Pipeline settings**Features:**

pipeline:- ✅ YOLOv8 detection training

  detection_conf_threshold: 0.25- ✅ Hyperparameter optimization

  classification_conf_threshold: 0.5- ✅ Validation & metrics tracking

```- ✅ Training visualization plots

- ✅ Model checkpoint saving

## 📈 Monitoring Training

### Step 3: Classification Model Training

### Training Logs```bash

```bashpython train_classification.py --model yolov8n-cls.pt --data data/classification/processed/dataset_classification.yaml --epochs 50

# View real-time training logs```

tail -f main_pipeline.log

```**Features:**

- ✅ YOLOv8 classification fine-tuning

### Results Location- ✅ Comprehensive evaluation với confusion matrix

- **Training Results**: `results/pipeline_results.json`- ✅ Per-class accuracy analysis

- **Model Weights**: `models/detection/best.pt`, `models/classification/best.pt`- ✅ Training progress visualization

- **Evaluation Plots**: `results/evaluation/`

- **Training Plots**: `results/detection/`, `results/classification/`### Step 4: Real-time Detection Pipeline

```bash

## 🎯 Performance Optimization# Webcam detection

python detect.py --source 0

### CPU Optimization (Default)

```yaml# Video file

detection:python detect.py --source video.mp4 --output output.mp4

  batch_size: 8      # Smaller batch for CPU

  workers: 4         # Reduced workers# Image file

  device: "cpu"python detect.py --source image.jpg --output result.jpg

```

classification:  

  batch_size: 16     # Optimized for CPU**Features:**

  device: "cpu"- ✅ 2-stage pipeline integration

```- ✅ Threading optimization for real-time

- ✅ Confidence threshold filtering

### GPU Optimization- ✅ Multi-worker classification processing

```yaml- ✅ Performance monitoring

detection:

  batch_size: 32     # Larger batch for GPU### Step 5: Comprehensive Evaluation

  workers: 8         # More workers```bash

  device: "cuda"     # or specific GPU: "cuda:0"python evaluate.py --detection-model models/detection/best.pt --classification-model models/classification/best.pt

```

classification:

  batch_size: 64     # Much larger batch**Features:**

  device: "cuda"- ✅ Detection model evaluation (mAP, precision, recall)

```- ✅ Classification model evaluation (accuracy, F1-score)

- ✅ Pipeline performance analysis

## 🔧 Advanced Usage- ✅ Visualization plots & reports

- ✅ Multi-threshold analysis

### Custom Dataset Paths

```python## ⚙️ Configuration

# In main.py, modify config loading:

config['detection']['data_yaml'] = "path/to/custom/dataset.yaml"Cấu hình toàn bộ system thông qua `configs/training_config.yaml`:

config['classification']['data_yaml'] = "path/to/custom/dataset"

``````yaml

# Detection Model Settings

### Model Selectiondetection:

```yaml  model_name: "yolov8n.pt"

# Available detection models  epochs: 100

detection:  batch_size: 16

  model_name: "yolov8n.pt"  # nano (fastest)  img_size: 640

  model_name: "yolov8s.pt"  # small    learning_rate: 0.01

  model_name: "yolov8m.pt"  # medium

  model_name: "yolov8l.pt"  # large# Classification Model Settings  

  model_name: "yolov8x.pt"  # extra large (best accuracy)classification:

  model_name: "yolov8n-cls.pt"

# Available classification models    epochs: 50

classification:  batch_size: 32

  model_name: "yolov8n-cls.pt"  # nano  img_size: 224

  model_name: "yolov8s-cls.pt"  # small  learning_rate: 0.001

  model_name: "yolov8m-cls.pt"  # medium

  model_name: "yolov8l-cls.pt"  # large# Pipeline Settings

  model_name: "yolov8x-cls.pt"  # extra largepipeline:

```  detection_conf_threshold: 0.25

  classification_conf_threshold: 0.5

### Export Models  max_workers: 4

```python  batch_classification: true

# Models are automatically exported to ONNX and TorchScript```

# Export paths: 

# - models/detection/best.onnx## 📈 Performance Metrics

# - models/detection/best.torchscript

# - models/classification/best.onnx  ### Detection Model

# - models/classification/best.torchscript- **mAP@50**: Mean Average Precision tại IoU=0.5

```- **mAP@50-95**: Mean Average Precision từ IoU=0.5 đến 0.95

- **Precision/Recall**: Precision và Recall cho từng class

## 📋 Troubleshooting- **F1-Score**: Harmonic mean của precision và recall



### Common Issues### Classification Model

- **Top-1 Accuracy**: Accuracy cho prediction hàng đầu

1. **CUDA out of memory**: Reduce batch_size in config- **Top-5 Accuracy**: Accuracy trong top-5 predictions

2. **Dataset not found**: Run preprocessing scripts first- **Per-class Metrics**: Precision, recall, F1-score cho từng class

3. **Model loading error**: Check model paths in config- **Confusion Matrix**: Ma trận confusion cho analysis chi tiết

4. **Low performance**: Increase epochs or use larger model

### Pipeline Performance

### Debug Mode- **FPS**: Frames per second processing speed

```bash- **Classification Rate**: Tỷ lệ objects được classify thành công

# Enable verbose logging- **Processing Time**: Average processing time per frame/image

python main.py --full-pipeline --config configs/training_config.yaml

# Check main_pipeline.log for detailed logs## 🔧 Advanced Usage

```

### Custom Dataset Training

## 📊 Expected Results

1. **Chuẩn bị dataset**:

### Detection Model   ```bash

- **mAP@50**: 0.4-0.7 (depending on dataset size and model)   # Detection: COCO format

- **Training Time**: 2-4 hours (CPU), 30-60 minutes (GPU)   data/detection/raw/

   ├── images/

### Classification Model     ├── annotations/

- **Top-1 Accuracy**: 0.6-0.9 (depending on dataset quality)   └── classes.txt

- **Training Time**: 1-2 hours (CPU), 15-30 minutes (GPU)   

   # Classification: Folder structure

## 🚀 Next Steps After Training   data/classification/raw/

   ├── class1/

1. **Real-time Detection**: Use `--detect` mode for live inference   ├── class2/

2. **Model Deployment**: Export models to ONNX/TensorRT for production   └── ...

3. **Fine-tuning**: Adjust hyperparameters based on evaluation results   ```

4. **Custom Classes**: Modify preprocessing scripts for custom categories

2. **Update configuration**:

## 📝 Key Features of Integrated Version   ```yaml

   datasets:

- ✅ **Single File**: All training logic in `main.py`     taco:

- ✅ **Modular Design**: Clear separation of detection/classification/evaluation       base_dir: "path/to/custom/detection/data"

- ✅ **Flexible Execution**: Run full pipeline or individual steps     trashnet:  

- ✅ **Clean Structure**: Only essential files (main.py + 2 preprocessing scripts)       base_dir: "path/to/custom/classification/data"

- ✅ **Comprehensive Logging**: Detailed progress tracking   ```

- ✅ **Auto Export**: Models exported in multiple formats

- ✅ **Real-time Ready**: Integrated detection pipeline for inference### Model Customization



---```yaml

# Sử dụng models lớn hơn

**Author**: Huy Nguyen  detection:

**Version**: 2.0.0 (Integrated)    model_name: "yolov8m.pt"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

**Date**: October 2025
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