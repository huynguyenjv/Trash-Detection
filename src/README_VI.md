# 📊 Module Xử Lý Chính (Source Module)

Module chứa các script xử lý dữ liệu, detection và utility cho hệ thống.

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Danh Sách Files](#danh-sách-files)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)

---

## 🔍 Tổng Quan

Module `src/` chứa các script utility cho hệ thống phát hiện rác:

```
src/
├── data_preprocessing.py       # Tiền xử lý dữ liệu
├── detect.py                   # Phát hiện rác thải
├── evaluate.py                 # Đánh giá mô hình
├── interactive_map.py          # Bản đồ tương tác
└── smart_routing_system.py     # Hệ thống định tuyến thông minh
```

> **Lưu ý:** Để huấn luyện mô hình, sử dụng `training-model/main.py` (pipeline tích hợp đầy đủ).

---

## 📁 Danh Sách Files

### 1. `data_preprocessing.py` - Tiền Xử Lý Dữ Liệu

```python
"""
Tiền xử lý dữ liệu cho training:
- Đọc và validate dataset
- Chuyển đổi format (COCO -> YOLO)
- Chia train/val/test
- Tạo file dataset.yaml
"""
```

**Sử dụng:**
```bash
python src/data_preprocessing.py \
    --input data/raw \
    --output data/processed \
    --split 0.7 0.2 0.1
```

### 2. `detect.py` - Phát Hiện Rác Thải

```python
"""
Phát hiện rác trong ảnh/video:
- Single image detection
- Video stream detection
- Webcam real-time detection
- Batch processing
"""
```

**Sử dụng:**
```bash
# Phát hiện trong ảnh
python src/detect.py --source image.jpg --model models/best.pt

# Phát hiện trong video
python src/detect.py --source video.mp4 --model models/best.pt

# Webcam real-time
python src/detect.py --source 0 --model models/best.pt
```

### 4. `evaluate.py` - Đánh Giá Mô Hình

```python
"""
Đánh giá hiệu suất mô hình:
- Tính mAP, Precision, Recall
- Confusion matrix
- Per-class metrics
- Export báo cáo
"""
```

**Sử dụng:**
```bash
python src/evaluate.py \
    --model models/best.pt \
    --data data/processed/dataset.yaml \
    --output results/evaluation
```

### 5. `interactive_map.py` - Bản Đồ Tương Tác

```python
"""
Tạo bản đồ tương tác với Folium:
- Hiển thị vị trí thùng rác
- Markers với thông tin chi tiết
- Popup hiển thị trạng thái
- Export HTML map
"""
```

**Sử dụng:**
```bash
python src/interactive_map.py \
    --bins bins_data.json \
    --output map.html
```

### 6. `smart_routing_system.py` - Định Tuyến Thông Minh

```python
"""
Hệ thống tối ưu lộ trình thu gom:
- Thuật toán A* pathfinding
- TSP (Traveling Salesman Problem) solver
- Tính toán khoảng cách thực tế
- Visualize route trên bản đồ
"""
```

**Sử dụng:**
```bash
python src/smart_routing_system.py \
    --bins bins_data.json \
    --start "10.762622,106.660172" \
    --output route.html
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Workflow Chuẩn

```bash
# 1. Tiền xử lý dữ liệu
python src/data_preprocessing.py --input data/raw --output data/processed

# 2. Huấn luyện mô hình
python src/train.py --data data/processed/dataset.yaml --epochs 100

# 3. Đánh giá mô hình
python src/evaluate.py --model models/best.pt

# 4. Phát hiện rác
python src/detect.py --source test_image.jpg --model models/best.pt

# 5. Tạo bản đồ
python src/interactive_map.py --output map.html
```

---

## 📊 Output

### Detection Output

```json
{
    "image": "test.jpg",
    "detections": [
        {
            "class": "plastic",
            "confidence": 0.92,
            "bbox": [100, 150, 200, 250],
            "category": "recyclable"
        }
    ],
    "count": 1
}
```

### Evaluation Output

```
=== Evaluation Report ===
Model: models/best.pt
Dataset: 5929 test images

Metrics:
- mAP@50: 85.7%
- mAP@50-95: 72.3%
- Precision: 83.2%
- Recall: 78.5%

Per-class Performance:
- paper: 90.2% mAP
- plastic: 75.9% mAP
- glass: 79.0% mAP
...
```

---

*Tác giả: Huy Nguyen | Cập nhật: Tháng 12, 2025*
