# 🎓 Hướng dẫn sử dụng dự án Trash Detection (Sau Refactor)

## 🌟 Tổng quan dự án

Dự án **Trash Detection** bao gồm 2 phần chính:

1. **🔴 Core ML System** (`src/`): YOLOv8 model training và detection
2. **🟡 Smart Routing System** (`system/`): Hệ thống định tuyến thông minh A*

---

## 🚀 Bắt đầu nhanh

### 1. Cài đặt môi trường
```bash
# Clone project
git clone <repo-url>
cd Trash-Detection

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_routing.txt  # Nếu dùng routing system
```

### 2. Setup Kaggle API
```bash
# Tạo file ~/.kaggle/kaggle.json
{
  "username": "your_username",
  "key": "your_key"
}

chmod 600 ~/.kaggle/kaggle.json
```

### 3. Chuẩn bị dữ liệu
```bash
cd src/
python data_preprocessing.py
# ✅ Sẽ tự động download và xử lý dataset từ Kaggle
```

### 4. Training model
```bash
# Option 1: Training cơ bản
python train.py

# Option 2: Training an toàn cho GPU nhỏ
python safe_train.py

# Monitor training
python monitor_training.py
```

### 5. Test detection
```bash
# Camera real-time
python detect.py --mode webcam --source 0

# Test với ảnh
python detect.py --mode image --source image.jpg
```

---

## 📂 Hiểu cấu trúc thư mục

### 🔴 Core System (`src/`)
```
src/
├── data_preprocessing.py    # Tải và xử lý dataset
├── train.py                 # Training YOLOv8 model
├── detect.py               # Real-time detection
├── evaluate.py             # Đánh giá model
└── smart_routing_system.py # Routing engine
```

**Mục đích**: Phát triển và train model AI phân loại rác thải

### 🟡 Smart Routing (`system/`)
```
system/
├── smart_routing_system.py     # A* pathfinding engine
├── interactive_map.py          # GUI map cơ bản
├── enhanced_map_gui.py         # 🌟 Enhanced GUI (giống Google Maps)
├── web_map_interface.py        # 🌐 Web-based map interface
├── position_utils.py           # Quản lý vị trí GPS
├── demo_realtime.py           # Demo tích hợp
└── linux_gui_setup.py         # Setup GUI cho Linux
```

**Mục đích**: Hệ thống định tuyến thông minh cho xe gom rác

**🌟 Tính năng mới**:
- 🔍 **Search & Navigation**: Tìm kiếm địa điểm, chỉ đường từng bước
- 🗺️ **Interactive Maps**: Zoom, pan, click-to-navigate như Google Maps
- 🌐 **Web Interface**: Giao diện web responsive, mobile-friendly
- 🧭 **Turn-by-turn GPS**: Hướng dẫn từng bước bằng tiếng Việt
- 🚦 **Traffic Info**: Hiển thị tình trạng giao thông real-time
- 📱 **Mobile App**: Progressive Web App cho điện thoại

### 🟢 Data (`data/`)
```
data/
├── raw/                    # Dataset gốc từ Kaggle
└── processed/              # Dataset đã xử lý cho YOLO
    ├── images/             # Ảnh train/val/test
    ├── labels/             # YOLO format labels
    └── dataset.yaml        # Config file
```

### 🔵 Models (`models/`)
```
models/
├── trash_safe_best.pt     # Model chính (sử dụng này)
├── best.pt               # Backup
└── final.pt              # Checkpoint cuối
```

---

## ⚡ Các lệnh quan trọng

### 🤖 Machine Learning Workflow

1. **Chuẩn bị data**:
```bash
cd src/
python data_preprocessing.py
```

2. **Training**:
```bash
# GPU mạnh (>8GB VRAM)
python train.py

# GPU yếu (<4GB VRAM)  
python safe_train.py
```

3. **Đánh giá**:
```bash
python evaluate.py --model ../models/trash_safe_best.pt
```

4. **Detection**:
```bash
# Camera
python detect.py --mode webcam --source 0

# Video
python detect.py --mode video --source video.mp4 --output result.mp4

# Batch images
python detect.py --mode batch --source images_folder/
```

### 🗺️ Smart Routing Workflow

1. **Test hệ thống**:
```bash
cd system/
python smart_routing_system.py
```

2. **Giao diện tương tác cơ bản**:
```bash
python interactive_map.py
```

3. **🌟 Giao diện nâng cao (giống Google Maps)**:
```bash
# Enhanced GUI với zoom, pan, search
python enhanced_map_gui.py

# Web-based interface (cần cài folium)
pip install folium
python web_map_interface.py
```

4. **Quản lý vị trí**:
```bash
# Chế độ tương tác
python position_utils.py --interactive

# Hiển thị vị trí hiện tại
python position_utils.py --show

# Set vị trí mới
python position_utils.py --lat 10.77 --lng 106.68
```

5. **Demo tích hợp**:
```bash
python demo_realtime.py --model ../models/trash_safe_best.pt --camera 0 --threshold 10
```

---

## 🎯 Use Cases chính

### 1. Phát triển Model AI
```bash
# Full pipeline
cd src/
python data_preprocessing.py  # Prepare data
python train.py              # Train model  
python evaluate.py            # Test performance
python detect.py --mode webcam --source 0  # Real-time test
```

### 2. Real-time Detection đơn giản  
```bash
cd src/
python detect.py --mode webcam --source 0
```

### 3. Smart Routing System hoàn chỉnh
```bash
cd system/

# Option 1: Enhanced GUI (desktop)
python enhanced_map_gui.py

# Option 2: Web interface (mở browser)  
python web_map_interface.py

# Option 3: Real-time integration
python demo_realtime.py --model ../models/trash_safe_best.pt --camera 0
```

### 4. Research & Analysis
```bash
cd notebooks/
jupyter notebook  # Khám phá data và kết quả
```

---

## 🔧 Tùy chỉnh cấu hình

### Training Configuration (`src/train.py`)
```python
@dataclass
class TrainingConfig:
    epochs: int = 50          # Số epoch
    batch_size: int = 16      # Batch size
    image_size: int = 640     # Kích thước ảnh
    model_name: str = "yolov8n.pt"  # Model size
```

### Detection Settings (`src/detect.py`)
```python
conf_threshold = 0.25     # Confidence threshold
iou_threshold = 0.45      # IoU threshold for NMS
max_detections = 100      # Max objects per image
```

### Routing Settings (`system/smart_routing_system.py`)
```python
threshold = 10            # Số lượng rác trigger routing
w_distance = 1.0          # Trọng số khoảng cách
w_time = 0.5             # Trọng số thời gian
```

---

## 🚨 Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**:
   - Giảm `batch_size` trong training config
   - Dùng `safe_train.py` thay vì `train.py`

2. **Kaggle API error**:
   - Kiểm tra `~/.kaggle/kaggle.json`
   - Verify credentials trên Kaggle

3. **Camera not working**:
   - Thử camera ID khác: `--source 1` hoặc `--source 2`
   - Kiểm tra permissions

4. **Import errors**:
   - Đảm bảo đang ở đúng working directory
   - Install đủ dependencies

5. **Model not found**:
   - Kiểm tra path: `../models/trash_safe_best.pt`
   - Đảm bảo đã train model hoặc download pretrained

### File paths quan trọng:
- Model: `models/trash_safe_best.pt`
- Dataset config: `data/processed/dataset.yaml`  
- Training logs: `src/training.log`
- Evaluation results: `src/evaluation_results/`

---

## 📊 Monitoring & Logs

### Training Progress:
```bash
# Real-time monitor
cd src/
python monitor_training.py

# Check logs
tail -f training.log
```

### Model Performance:
```bash
# Full evaluation
python evaluate.py --model ../models/trash_safe_best.pt

# Quick test
python detect.py --mode test --source ../data/processed/images/test/
```

---

## 🎉 Kết luận

Dự án đã được refactor thành cấu trúc rõ ràng:

- **`src/`**: Focus vào AI/ML development
- **`system/`**: Focus vào smart routing application  
- **`data/`**: Organized dataset storage
- **`models/`**: Centralized model storage

Mỗi phần có thể hoạt động độc lập hoặc tích hợp với nhau tùy nhu cầu sử dụng.

**Happy coding!** 🚀
