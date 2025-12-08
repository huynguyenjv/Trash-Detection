# 🗑️ Hệ Thống Phát Hiện và Phân Loại Rác Thải Thông Minh

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-red.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 📋 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Tính Năng](#tính-năng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [API Documentation](#api-documentation)
- [Kết Quả Huấn Luyện](#kết-quả-huấn-luyện)
- [Đóng Góp](#đóng-góp)
- [Giấy Phép](#giấy-phép)

---

## 🎯 Giới Thiệu

Hệ thống **Phát Hiện và Phân Loại Rác Thải Thông Minh** sử dụng công nghệ học sâu (Deep Learning) với mô hình YOLOv8 để:

- **Phát hiện rác thải** trong ảnh và video theo thời gian thực
- **Phân loại rác thải** vào 4 nhóm: Tái chế, Hữu cơ, Nguy hại, Khác
- **Tối ưu lộ trình thu gom** sử dụng thuật toán A* Pathfinding
- **Quản lý thùng rác thông minh** với bản đồ tương tác

### Các Loại Rác Được Hỗ Trợ

| STT | Loại Rác | Phân Loại | Mô Tả |
|-----|----------|-----------|-------|
| 1 | Giấy (paper) | ♻️ Tái chế | Giấy báo, giấy văn phòng |
| 2 | Bìa cứng (cardboard) | ♻️ Tái chế | Hộp carton, bìa đựng |
| 3 | Nhựa (plastic) | ♻️ Tái chế | Chai nhựa, túi nhựa |
| 4 | Thủy tinh (glass) | ♻️ Tái chế | Chai thủy tinh, lọ |
| 5 | Kim loại (metal) | ♻️ Tái chế | Lon nhôm, hộp thiếc |
| 6 | Rác hữu cơ (biological) | 🌿 Hữu cơ | Thức ăn thừa, lá cây |
| 7 | Pin (battery) | ⚠️ Nguy hại | Pin, ắc quy |
| 8 | Quần áo (clothes) | 📦 Khác | Vải, quần áo cũ |
| 9 | Giày dép (shoes) | 📦 Khác | Giày, dép cũ |
| 10 | Rác khác (trash) | 📦 Khác | Rác không phân loại |

---

## ✨ Tính Năng

### 🔍 Phát Hiện Rác Thải
- Phát hiện 10 loại rác thải khác nhau
- Độ chính xác mAP@50: **85.7%**
- Xử lý theo thời gian thực với webcam

### 🏷️ Phân Loại Tự Động
- Phân loại vào 4 nhóm chính
- Hỗ trợ đề xuất cách xử lý phù hợp

### 🗺️ Tối Ưu Lộ Trình
- Thuật toán A* Pathfinding
- Bản đồ tương tác với Folium
- Tính toán lộ trình thu gom tối ưu

### 🌐 API Backend
- FastAPI với RESTful API
- WebSocket cho real-time detection
- CORS support cho frontend

---

## 📁 Cấu Trúc Dự Án

```
Trash-Detection/
├── 📂 data/                    # Dữ liệu thô và đã xử lý
│   ├── raw/                    # Dataset gốc
│   └── processed/              # Dataset đã xử lý
│
├── 📂 models/                  # Mô hình đã huấn luyện
│   ├── best.pt                 # Mô hình tốt nhất
│   └── last.pt                 # Mô hình cuối cùng
│
├── 📂 src/                     # Source code chính
│   ├── data_preprocessing.py   # Tiền xử lý dữ liệu
│   ├── train.py                # Huấn luyện mô hình
│   ├── detect.py               # Phát hiện rác thải
│   ├── evaluate.py             # Đánh giá mô hình
│   ├── interactive_map.py      # Bản đồ tương tác
│   └── smart_routing_system.py # Hệ thống định tuyến
│
├── 📂 training-model/          # Module huấn luyện
│   ├── main.py                 # Pipeline huấn luyện chính
│   ├── data_preprocessing_detection.py
│   ├── data_preprocessing_classification.py
│   └── configs/                # Cấu hình huấn luyện
│
├── 📂 waste-system/            # Hệ thống quản lý rác
│   ├── backend/                # API backend (v1)
│   ├── backend-v2/             # API backend (v2)
│   └── frontend/               # Giao diện web
│
├── 📂 notebooks/               # Jupyter notebooks
│   └── trash_detection_tutorial.ipynb
│
├── 📂 paper/                   # Bài báo khoa học
│
├── 📄 requirements.txt         # Dependencies
├── 📄 config.yaml              # Cấu hình hệ thống
├── 📄 test_image_detection.py  # Script test phát hiện
└── 📄 README_VI.md             # Tài liệu tiếng Việt
```

---

## 💻 Yêu Cầu Hệ Thống

### Phần Cứng
- **CPU**: Intel Core i5 trở lên (hoặc tương đương)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị cho training)
- **Disk**: Tối thiểu 10GB dung lượng trống

### Phần Mềm
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8+ (nếu dùng GPU)
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+

---

## 🚀 Cài Đặt

### 1. Clone Repository

```bash
git clone https://github.com/huynguyenjv/Trash-Detection.git
cd Trash-Detection
```

### 2. Tạo Virtual Environment

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Linux/macOS)
source venv/bin/activate

# Kích hoạt (Windows)
.\venv\Scripts\activate
```

### 3. Cài Đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Cài đặt thêm nếu cần
pip install ultralytics torch torchvision
```

### 4. Tải Mô Hình

```bash
# Mô hình đã được huấn luyện nằm trong thư mục models/
# Hoặc tải YOLOv8 pretrained
pip install ultralytics
```

---

## 📖 Hướng Dẫn Sử Dụng

### 1. Phát Hiện Rác Trong Ảnh

```bash
# Sử dụng script test
python test_image_detection.py

# Hoặc sử dụng trực tiếp
python -c "
from waste-system.backend-v2.detector import WasteDetector
detector = WasteDetector('models/best.pt')
results = detector.detect('path/to/image.jpg')
print(results)
"
```

### 2. Huấn Luyện Mô Hình

```bash
cd training-model

# Huấn luyện Detection Model
python main.py --train-detection --epochs 100 --batch 16

# Huấn luyện Classification Model
python main.py --train-classification --epochs 50 --batch 32

# Chạy full pipeline
python main.py --full-pipeline
```

### 3. Tiền Xử Lý Dữ Liệu

```bash
cd training-model

# Tiền xử lý cho Detection
python data_preprocessing_detection.py

# Tiền xử lý cho Classification
python data_preprocessing_classification.py
```

### 4. Khởi Động API Backend

```bash
cd waste-system/backend-v2

# Cài đặt dependencies
pip install -r requirements.txt

# Khởi động server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Truy Cập API Documentation

Sau khi khởi động server, truy cập:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔌 API Documentation

### Endpoints Chính

| Method | Endpoint | Mô Tả |
|--------|----------|-------|
| `GET` | `/` | Health check |
| `POST` | `/detect` | Phát hiện rác trong ảnh |
| `POST` | `/detect/batch` | Phát hiện nhiều ảnh |
| `GET` | `/bins` | Lấy danh sách thùng rác |
| `POST` | `/bins` | Thêm thùng rác mới |
| `GET` | `/route/optimize` | Tối ưu lộ trình thu gom |

### Ví Dụ Request

```python
import requests

# Phát hiện rác trong ảnh
files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8000/detect', files=files)
print(response.json())

# Kết quả:
# {
#     "detections": [
#         {
#             "class": "plastic",
#             "category": "recyclable",
#             "confidence": 0.92,
#             "bbox": [100, 150, 200, 250]
#         }
#     ],
#     "total_count": 1
# }
```

---

## 📊 Kết Quả Huấn Luyện

### Detection Model Performance

| Metric | Giá Trị |
|--------|---------|
| mAP@50 | 85.7% |
| mAP@50-95 | 72.3% |
| Precision | 83.2% |
| Recall | 78.5% |

### Hiệu Suất Theo Loại Rác

| Loại Rác | Precision | Recall | mAP@50 |
|----------|-----------|--------|--------|
| Giấy | 92.3% | 88.1% | 90.2% |
| Nhựa | 77.6% | 74.2% | 75.9% |
| Quần áo | 88.0% | 85.3% | 86.7% |
| Kim loại | 84.5% | 80.1% | 82.3% |
| Thủy tinh | 81.2% | 76.8% | 79.0% |

### Thông Số Huấn Luyện

- **Dataset**: 19,762 ảnh
- **Train/Val/Test**: 60% / 10% / 30%
- **Model**: YOLOv8n (nano)
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640x640
- **Learning Rate**: 0.01

---

## 🤝 Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/TinhNangMoi`)
3. Commit changes (`git commit -m 'Thêm tính năng mới'`)
4. Push to branch (`git push origin feature/TinhNangMoi`)
5. Tạo Pull Request

### Báo Lỗi

Nếu gặp lỗi, vui lòng tạo Issue với:
- Mô tả lỗi chi tiết
- Các bước để tái tạo lỗi
- Log/Screenshot nếu có
- Thông tin môi trường (OS, Python version, ...)

---

## 📜 Giấy Phép

Dự án này được phân phối dưới giấy phép **MIT License**.

---

## 👨‍💻 Tác Giả

**Huy Nguyen**
- GitHub: [@huynguyenjv](https://github.com/huynguyenjv)
- Email: huynguyen@example.com

---

## 🙏 Lời Cảm Ơn

- [Ultralytics](https://ultralytics.com/) - YOLOv8 framework
- [TACO Dataset](http://tacodataset.org/) - Dataset rác thải
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision library

---

<p align="center">
  ⭐ Nếu thấy hữu ích, hãy cho dự án một star! ⭐
</p>
