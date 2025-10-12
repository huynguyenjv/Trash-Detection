# Trash Detection Training Pipeline Results

## Pipeline Information
- **Config File**: D:\MasterUIT\Trash-Detection\training-model\configs\training_config.yaml
- **Timestamp**: 2025-10-09T00:19:29.428544
- **Version**: 1.0.0

## Pipeline Status
- ✅ Data Preprocessing: ❌
- ✅ Detection Training: ❌
- ✅ Classification Training: ❌
- ✅ Comprehensive Evaluation: ❌
- ✅ Integration Testing: ❌

## Key Performance Metrics

### Detection Model
- **Training mAP@50**: N/A
- **Final mAP@50**: N/A

### Classification Model  
- **Training Accuracy**: N/A
- **Final Accuracy**: N/A

### Pipeline Integration
- **Integration Success Rate**: N/A

## Model Files
- **Detection Model**: `models/detection/best.pt`
- **Classification Model**: `models/classification/best.pt`

## Usage Examples

### Training Individual Models
```bash
# Train detection model
python train_detection.py --config configs/training_config.yaml

# Train classification model  
python train_classification.py --config configs/training_config.yaml
```

### Run Complete Pipeline
```bash
# Run full training pipeline
python main.py --config configs/training_config.yaml --full-pipeline

# Run specific steps
python main.py --config configs/training_config.yaml --steps preprocessing,detection
```

### Real-time Detection
```bash
# Webcam detection
python detect.py --source 0

# Video file detection
python detect.py --source video.mp4 --output output.mp4

# Image detection
python detect.py --source image.jpg --output result.jpg
```

### Evaluation
```bash
# Comprehensive evaluation
python evaluate.py --detection-model models/detection/best.pt --classification-model models/classification/best.pt
```

## Directory Structure
```
training-model/
├── data/
│   ├── detection/
│   │   ├── raw/
│   │   └── processed/
│   └── classification/
│       ├── raw/
│       └── processed/
├── models/
│   ├── detection/
│   └── classification/
├── results/
│   ├── detection/
│   ├── classification/
│   └── evaluation/
├── configs/
├── logs/
└── *.py files
```

Generated on: 2025-10-09 00:19:29
