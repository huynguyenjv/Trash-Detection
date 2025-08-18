# 📁 Cấu trúc thư mục dự án Trash Detection (Sau refactor)

## 🌟 Tổng quan cấu trúc

```
Trash-Detection/
├── 📄 README.md                      # Tài liệu chính dự án
├── 📄 README_routing.md              # Tài liệu hệ thống routing thông minh
├── 📄 requirements.txt               # Dependencies cho core project
├── 📄 requirements_routing.txt       # Dependencies cho routing system
├── 📄 Makefile                       # Build automation
├── 📄 USAGE.py                       # Hướng dẫn sử dụng nhanh
├── 📄 .gitignore                     # Git ignore rules
│
├── 📂 src/                           # 🔴 CORE SOURCE CODE
│   ├── data_preprocessing.py         # Tiền xử lý dataset
│   ├── train.py                      # Training YOLOv8 model
│   ├── detect.py                     # Real-time detection
│   ├── evaluate.py                   # Đánh giá model performance
│   ├── smart_routing_system.py       # Hệ thống định tuyến A*
│   ├── interactive_map.py            # Giao diện map tương tác
│   ├── position_utils.py             # Utilities quản lý vị trí
│   ├── demo_realtime.py              # Demo detection real-time
│   ├── monitor_training.py           # Monitor quá trình training
│   ├── evaluation_results/           # Kết quả đánh giá chi tiết
│   └── runs/                         # Training runs output
│
├── 📂 system/                        # 🟡 SMART ROUTING SYSTEM
│   ├── smart_routing_system.py       # Core routing engine
│   ├── interactive_map.py            # Map visualization
│   ├── position_utils.py             # Position management
│   ├── demo_realtime.py              # Real-time demo
│   └── *.json                        # Position history files
│
├── 📂 data/                          # 🟢 DATASETS
│   ├── raw/                          # Dataset gốc từ Kaggle
│   │   └── garbage-classification-v2/
│   └── processed/                    # Dataset đã xử lý
│       ├── images/                   # Ảnh train/val/test
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/                   # YOLO format labels
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── dataset.yaml              # Dataset configuration
│
├── 📂 models/                        # 🔵 MODEL WEIGHTS
│   ├── trash_safe_best.pt           # Best trained model
│   ├── best.pt                      # Backup model
│   ├── final.pt                     # Final checkpoint
│   └── .gitkeep
│
├── 📂 notebooks/                     # 📊 JUPYTER NOTEBOOKS
│   └── *.ipynb                      # Analysis & experimentation
│
├── 📂 runs/                          # 🏃 TRAINING OUTPUTS
│   ├── train/                       # Training runs
│   └── detect/                      # Detection results
│
├── 📂 test/                          # 🧪 TESTING
│   └── *.py                         # Test scripts
│
├── 📂 .github/                       # ⚙️ GITHUB WORKFLOWS
│   └── workflows/
│
└── 📂 trash_detection_env/           # 🐍 VIRTUAL ENVIRONMENT
    └── ...                          # Python environment files
```

---

## 🔍 Chi tiết từng thư mục

### 📂 src/ - Core Source Code

**Chức năng chính**: Chứa tất cả source code chính của dự án

#### 📄 Files quan trọng:

1. **`data_preprocessing.py`**
   - Tải dataset từ Kaggle
   - Convert classification → object detection
   - Tạo YOLO format annotations
   - Chia train/val/test split

2. **`train.py`**
   - Training YOLOv8 model
   - Cấu hình hyperparameters
   - Memory optimization
   - Logging và monitoring

3. **`detect.py`**
   - Real-time detection từ camera/video
   - Batch processing images
   - Visualization results
   - Performance metrics

4. **`evaluate.py`**
   - Đánh giá model performance
   - Confusion matrix
   - mAP, Precision, Recall
   - Per-class analysis

5. **Smart Routing Files**:
   - `smart_routing_system.py`: A* pathfinding
   - `interactive_map.py`: GUI map interaction
   - `position_utils.py`: Position management
   - `demo_realtime.py`: Real-time demo

#### 📂 Subfolders:
- `evaluation_results/`: Kết quả đánh giá, plots, reports
- `runs/`: Training và detection outputs

---

### 📂 system/ - Smart Routing System

**Chức năng**: Hệ thống định tuyến thông minh cho xe gom rác

#### 🎯 Tính năng:
- A* pathfinding algorithm
- Real-time waste detection tracking  
- Interactive map visualization
- GPS coordinate management
- Traffic condition updates
- Waste bin status monitoring

#### 📄 Files:
- `smart_routing_system.py`: Core engine
- `interactive_map.py`: GUI map
- `position_utils.py`: Position utilities
- `demo_realtime.py`: Demo integration
- `*.json`: Position history logs

---

### 📂 data/ - Datasets

```
data/
├── raw/                              # Dataset gốc
│   └── garbage-classification-v2/    # Kaggle dataset
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
│
└── processed/                        # Dataset đã xử lý
    ├── images/                       # Ảnh theo format YOLO
    │   ├── train/                    # 80% - Training images
    │   ├── val/                      # 10% - Validation images
    │   └── test/                     # 10% - Test images
    ├── labels/                       # Annotations YOLO format
    │   ├── train/                    # .txt files cho training
    │   ├── val/                      # .txt files cho validation
    │   └── test/                     # .txt files cho testing
    └── dataset.yaml                  # YOLO dataset config
```

---

### 📂 models/ - Model Weights

```
models/
├── trash_safe_best.pt               # Model tốt nhất (main)
├── best.pt                          # Backup model
├── final.pt                         # Final checkpoint
└── .gitkeep                         # Git placeholder
```

**Cách sử dụng**:
```python
from ultralytics import YOLO

# Load best model
model = YOLO('models/trash_safe_best.pt')

# Run inference
results = model('path/to/image.jpg')
```

---

### 📂 runs/ - Training Outputs

```
runs/
├── train/                           # Training runs
│   ├── trash_safe/                  # Run name
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   ├── results.csv              # Training metrics
│   │   ├── confusion_matrix.png
│   │   └── val_batch*.jpg           # Validation samples
│   └── trash_safe2/                 # Another run
│
└── detect/                          # Detection results
    ├── predict/
    │   └── *.jpg                    # Annotated images
    └── val/                         # Validation results
```

---

## 🚀 Workflow sử dụng

### 1. Data Preparation
```bash
cd src/
python data_preprocessing.py
```

### 2. Model Training  
```bash
# Basic training
python train.py

# Memory-safe training
python safe_train.py

# Monitor training
python monitor_training.py
```

### 3. Model Evaluation
```bash
python evaluate.py --model ../models/trash_safe_best.pt
```

### 4. Real-time Detection
```bash
# Camera detection
python detect.py --mode webcam --source 0

# Image detection
python detect.py --mode image --source image.jpg
```

### 5. Smart Routing System
```bash
# Interactive map
cd system/
python interactive_map.py

# Position utilities  
python position_utils.py --interactive

# Real-time demo with routing
python demo_realtime.py --model ../models/trash_safe_best.pt --camera 0
```

---

## 🔧 Configuration Files

### 📄 requirements.txt
```
ultralytics>=8.0.0
opencv-python>=4.5.0
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
kaggle>=1.5.0
```

### 📄 requirements_routing.txt  
```
# Additional for routing system
folium>=0.12.0
geopandas>=0.9.0
```

### 📄 dataset.yaml
```yaml
path: /path/to/data/processed
train: images/train
val: images/val  
test: images/test

nc: 10  # number of classes
names: ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
```

---

## 📊 Monitoring & Logs

### Log Files:
- `src/training.log`: Training progress
- `src/evaluation.log`: Evaluation results
- `pipeline.log`: Overall pipeline logs

### Monitoring Tools:
- `monitor_training.py`: Real-time training monitor
- TensorBoard integration
- Weights & Biases support

---

## 🎯 Quick Start Guide

```bash
# 1. Clone và setup
git clone <repo-url>
cd Trash-Detection
pip install -r requirements.txt

# 2. Setup Kaggle API
# Tạo ~/.kaggle/kaggle.json với credentials

# 3. Prepare data  
cd src/
python data_preprocessing.py

# 4. Train model
python train.py

# 5. Test detection
python detect.py --mode image --source test_image.jpg

# 6. Run smart routing system
cd ../system/
python demo_realtime.py --model ../models/trash_safe_best.pt --camera 0
```

---

## 🚨 Important Notes

### File Paths:
- Tất cả scripts assume working directory tương ứng
- Model paths: `../models/` từ src/
- Data paths: `../data/` từ src/

### Dependencies:
- Core project: `requirements.txt`  
- Routing system: `requirements_routing.txt`
- Install both nếu dùng full features

### GPU Memory:
- YOLOv8n: ~2GB VRAM
- YOLOv8m: ~6GB VRAM
- Adjust batch_size theo hardware

---

## 🔄 Migration Notes

### Từ structure cũ:
1. Tách smart routing thành `system/` folder
2. Consolidate core ML code trong `src/`
3. Separate requirements files
4. Better organization của outputs

### Breaking Changes:
- Import paths changed cho routing system
- Config files moved
- Some script locations changed

---

## 📞 Support

Nếu có vấn đề với cấu trúc mới:

1. Kiểm tra working directory
2. Verify import paths  
3. Check requirements installation
4. Review README files tương ứng

Happy coding! 🎉
