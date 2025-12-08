# 🎯 Module Huấn Luyện Mô Hình (Training Module)

Module này chứa toàn bộ pipeline huấn luyện cho hệ thống phát hiện và phân loại rác thải.

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Cấu Hình](#cấu-hình)
- [Chi Tiết Các Module](#chi-tiết-các-module)

---

## 🔍 Tổng Quan

Module huấn luyện bao gồm 3 file chính:

| File | Mô Tả |
|------|-------|
| `main.py` | Pipeline huấn luyện tích hợp (Detection + Classification) |
| `data_preprocessing_detection.py` | Tiền xử lý dữ liệu cho Object Detection |
| `data_preprocessing_classification.py` | Tiền xử lý dữ liệu cho Image Classification |

---

## 📁 Cấu Trúc Thư Mục

```
training-model/
├── 📄 main.py                              # Pipeline huấn luyện chính
├── 📄 data_preprocessing_detection.py      # Tiền xử lý Detection
├── 📄 data_preprocessing_classification.py # Tiền xử lý Classification
├── 📄 README.md                            # Tài liệu này
│
├── 📂 configs/                             # Cấu hình huấn luyện
│   └── training_config.yaml
│
├── 📂 data/                                # Dữ liệu
│   ├── classification/                     # Dữ liệu phân loại
│   ├── detection/                          # Dữ liệu phát hiện
│   └── processed/                          # Dữ liệu đã xử lý
│
├── 📂 models/                              # Mô hình đã huấn luyện
│   ├── classification/
│   └── detection/
│
├── 📂 results/                             # Kết quả huấn luyện
│
└── 📂 runs/                                # Log huấn luyện YOLO
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Tiền Xử Lý Dữ Liệu

#### Detection (Phát hiện vật thể)

```bash
# Xử lý dataset TACO cho detection
python data_preprocessing_detection.py

# Với tham số tùy chỉnh
python data_preprocessing_detection.py \
    --raw-dir ../data/raw \
    --output-dir data/processed/detection \
    --train-ratio 0.6 \
    --val-ratio 0.1 \
    --test-ratio 0.3
```

#### Classification (Phân loại)

```bash
# Xử lý dataset Garbage cho classification
python data_preprocessing_classification.py

# Với tham số tùy chỉnh
python data_preprocessing_classification.py \
    --raw-dir ../data/raw \
    --output-dir data/processed/classification
```

### 2. Huấn Luyện Mô Hình

```bash
# Hiển thị help
python main.py --help

# Huấn luyện Detection Model
python main.py --train-detection

# Huấn luyện Classification Model
python main.py --train-classification

# Chạy đánh giá
python main.py --evaluate

# Chạy phát hiện trên ảnh
python main.py --detect --source path/to/image.jpg

# Chạy toàn bộ pipeline
python main.py --full-pipeline
```

#### Tham Số Huấn Luyện

| Tham Số | Mặc Định | Mô Tả |
|---------|----------|-------|
| `--epochs` | 100 | Số epoch huấn luyện |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Kích thước ảnh |
| `--lr` | 0.01 | Learning rate |
| `--device` | auto | Device (cpu/cuda/mps) |

### 3. Đánh Giá Mô Hình

```bash
# Đánh giá Detection Model
python main.py --evaluate \
    --detection-model models/detection/best.pt \
    --data-yaml data/processed/detection/dataset.yaml
```

---

## ⚙️ Cấu Hình

### File `configs/training_config.yaml`

```yaml
# Cấu hình Detection
detection:
  model_name: yolov8n.pt        # YOLOv8 nano model
  epochs: 100                    # Số epoch
  batch_size: 16                 # Batch size
  img_size: 640                  # Kích thước ảnh
  learning_rate: 0.01            # Learning rate
  device: auto                   # auto/cpu/cuda
  data_yaml: data/processed/detection/dataset.yaml
  
# Cấu hình Classification
classification:
  model_name: yolov8n-cls.pt    # YOLOv8 classification
  epochs: 50
  batch_size: 32
  img_size: 224
  learning_rate: 0.001
  device: auto
```

### File `dataset.yaml` (Detection)

```yaml
path: /absolute/path/to/data/processed/detection
train: images/train
val: images/val
test: images/test

nc: 10  # Số lượng class

names:
  0: battery
  1: biological
  2: cardboard
  3: clothes
  4: glass
  5: metal
  6: paper
  7: plastic
  8: shoes
  9: trash
```

---

## 📚 Chi Tiết Các Module

### 1. `main.py` - Pipeline Huấn Luyện Chính

Tích hợp toàn bộ pipeline huấn luyện với các class:

```python
# Cấu hình
@dataclass
class DetectionTrainingConfig:
    """Cấu hình cho detection training"""
    model_name: str = "yolov8n.pt"
    epochs: int = 100
    batch_size: int = 16
    ...

# Trainer classes
class DetectionTrainer:
    def setup_model(self) -> YOLO
    def train_model(self) -> Dict
    def validate_model(self) -> Dict
    
class ClassificationTrainer:
    def setup_model(self) -> YOLO
    def train_model(self) -> Dict
    
class ComprehensiveEvaluator:
    def evaluate_detection(self) -> Dict
    def generate_report(self) -> str
```

### 2. `data_preprocessing_detection.py`

Xử lý dataset TACO/COCO format sang YOLO format:

- **Load annotations**: Đọc COCO annotations
- **Convert format**: Chuyển đổi bbox sang YOLO format
- **Split dataset**: Chia train/val/test
- **Create YAML**: Tạo file cấu hình cho YOLO

### 3. `data_preprocessing_classification.py`

Xử lý Garbage dataset cho classification:

- **Check dataset**: Kiểm tra dataset có sẵn
- **Create structure**: Tạo cấu trúc ImageNet-style
- **Split data**: Chia stratified train/val/test
- **Copy images**: Copy và resize ảnh

---

## 📊 Output Files

Sau khi huấn luyện:

```
results/
├── detection/
│   └── detection_v1/
│       ├── weights/
│       │   ├── best.pt      # Mô hình tốt nhất
│       │   └── last.pt      # Mô hình cuối
│       ├── confusion_matrix.png
│       ├── results.csv
│       └── results.png
│
└── classification/
    └── classification_v1/
        └── weights/
            └── best.pt
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
python main.py --train-detection --batch 8
```

### Dataset không tìm thấy
```bash
# Sử dụng đường dẫn tuyệt đối trong dataset.yaml
path: /home/user/Trash-Detection/training-model/data/processed/detection
```

### Import Error
```bash
pip install ultralytics torch torchvision opencv-python
```

---

## 📝 Ghi Chú

- Dataset gốc cần được đặt trong `../data/raw`
- Mô hình pretrained tải tự động từ Ultralytics
- Log file: `main_pipeline.log`

---

*Tác giả: Huy Nguyen | Cập nhật: Tháng 12, 2025*
