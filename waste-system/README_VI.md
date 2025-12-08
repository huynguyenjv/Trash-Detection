# 🌐 Hệ Thống Quản Lý Rác Thải (Waste Management System)

Hệ thống backend API để quản lý và phát hiện rác thải thông minh.

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [API Endpoints](#api-endpoints)
- [Services](#services)

---

## 🔍 Tổng Quan

Hệ thống bao gồm:

- **Backend API**: FastAPI server để xử lý detection và quản lý thùng rác
- **Detection Service**: Tích hợp YOLOv8 để phát hiện rác
- **Pathfinding Service**: Thuật toán A* để tối ưu lộ trình thu gom
- **Frontend**: Giao diện web React (trong `frontend/`)

---

## 📁 Kiến Trúc Hệ Thống

```
waste-system/
├── 📂 backend-v2/              # Backend chính (FastAPI)
│   ├── main.py                 # Entry point
│   ├── requirements.txt        # Dependencies
│   │
│   └── 📂 app/                 # Application package
│       ├── config.py           # Cấu hình
│       ├── database.py         # Database connection
│       ├── models.py           # SQLAlchemy models
│       ├── schemas.py          # Pydantic schemas
│       ├── crud.py             # CRUD operations
│       │
│       ├── 📂 api/             # API routes
│       │   └── routes.py
│       │
│       └── 📂 services/        # Business logic
│           ├── detector.py         # Phát hiện rác (YOLOv8)
│           ├── pathfinding.py      # Thuật toán A*
│           ├── waste_manager.py    # Quản lý thùng rác
│           ├── waste_pipeline.py   # Pipeline xử lý
│           └── object_tracker.py   # Theo dõi đối tượng
│
├── 📂 backend/                 # Backend cũ (legacy)
│
└── 📂 frontend/                # Giao diện React
```

---

## 🚀 Cài Đặt

### 1. Cài Đặt Dependencies

```bash
cd waste-system/backend-v2
pip install -r requirements.txt
```

### 2. Cấu Hình Environment

```bash
# Copy file .env.example
cp .env.example .env

# Chỉnh sửa .env
DATABASE_URL=sqlite:///./waste.db
MODEL_PATH=../../models/best.pt
DEBUG=True
```

### 3. Khởi Tạo Database

```bash
python create_db.py
```

### 4. Khởi Động Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Truy Cập API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📡 API Endpoints

### Detection API

| Method | Endpoint | Mô Tả |
|--------|----------|-------|
| `POST` | `/detect` | Phát hiện rác trong ảnh |
| `POST` | `/detect/batch` | Phát hiện nhiều ảnh |
| `POST` | `/detect/stream` | WebSocket real-time |

### Bin Management API

| Method | Endpoint | Mô Tả |
|--------|----------|-------|
| `GET` | `/bins` | Lấy danh sách thùng rác |
| `POST` | `/bins` | Thêm thùng rác mới |
| `GET` | `/bins/{id}` | Chi tiết thùng rác |
| `PUT` | `/bins/{id}` | Cập nhật thùng rác |
| `DELETE` | `/bins/{id}` | Xóa thùng rác |

### Routing API

| Method | Endpoint | Mô Tả |
|--------|----------|-------|
| `GET` | `/route/optimize` | Tối ưu lộ trình thu gom |
| `POST` | `/route/calculate` | Tính toán đường đi |

### Ví Dụ Request

```python
import requests

# Phát hiện rác trong ảnh
files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8000/detect', files=files)

result = response.json()
# {
#     "detections": [
#         {
#             "class": "plastic",
#             "category": "recyclable",
#             "confidence": 0.92,
#             "bbox": [100, 150, 200, 250]
#         }
#     ],
#     "total_count": 1,
#     "processing_time": 0.15
# }
```

---

## 🛠️ Services

### 1. `detector.py` - Phát Hiện Rác

```python
class WasteDetector:
    """
    Phát hiện rác thải sử dụng YOLOv8
    
    Attributes:
        model: YOLOv8 model
        confidence_threshold: Ngưỡng tin cậy (mặc định: 0.25)
        device: Device chạy model (cpu/cuda)
    
    Methods:
        detect(image): Phát hiện rác trong ảnh
        detect_batch(images): Phát hiện nhiều ảnh
    """
    
    def detect(self, image_path: str) -> List[Detection]:
        """
        Phát hiện rác trong ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            List[Detection]: Danh sách các detection
        """
        pass
```

**Class mapping (phân loại rác):**

| Class gốc | Category | Mô tả |
|-----------|----------|-------|
| paper | ♻️ recyclable | Giấy tái chế |
| cardboard | ♻️ recyclable | Bìa cứng |
| plastic | ♻️ recyclable | Nhựa |
| glass | ♻️ recyclable | Thủy tinh |
| metal | ♻️ recyclable | Kim loại |
| biological | 🌿 organic | Rác hữu cơ |
| battery | ⚠️ hazardous | Pin, ắc quy |
| clothes | 📦 other | Quần áo |
| shoes | 📦 other | Giày dép |
| trash | 📦 other | Rác khác |

### 2. `pathfinding.py` - Thuật Toán A*

```python
class AStarPathfinder:
    """
    Thuật toán A* để tìm đường đi tối ưu
    
    Methods:
        find_path(start, end, grid): Tìm đường đi ngắn nhất
        optimize_route(bins): Tối ưu lộ trình qua nhiều điểm
    """
    
    def find_path(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int],
        grid: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Tìm đường đi ngắn nhất từ start đến end
        
        Sử dụng heuristic: Manhattan distance
        f(n) = g(n) + h(n)
        
        Args:
            start: Điểm bắt đầu (x, y)
            end: Điểm kết thúc (x, y)
            grid: Ma trận grid (0: đi được, 1: vật cản)
            
        Returns:
            List path: Danh sách các điểm trên đường đi
        """
        pass
```

### 3. `waste_manager.py` - Quản Lý Thùng Rác

```python
class WasteManager:
    """
    Quản lý thùng rác và trạng thái
    
    Methods:
        get_bins(): Lấy tất cả thùng rác
        add_bin(bin): Thêm thùng rác
        update_bin(id, data): Cập nhật thùng rác
        get_full_bins(): Lấy các thùng đầy
        calculate_collection_route(): Tính lộ trình thu gom
    """
```

### 4. `waste_pipeline.py` - Pipeline Xử Lý

```python
class WastePipeline:
    """
    Pipeline xử lý toàn bộ flow:
    1. Nhận ảnh đầu vào
    2. Phát hiện rác thải
    3. Phân loại rác
    4. Cập nhật database
    5. Tính toán lộ trình (nếu cần)
    """
```

---

## 🗄️ Database Models

### Bin Model

```python
class Bin(Base):
    """Thùng rác"""
    id: int                    # ID
    name: str                  # Tên thùng
    location_lat: float        # Vĩ độ
    location_lng: float        # Kinh độ
    capacity: float            # Dung tích (%)
    waste_type: str            # Loại rác
    last_collection: datetime  # Lần thu gom cuối
```

### Detection Model

```python
class Detection(Base):
    """Lịch sử phát hiện"""
    id: int
    image_path: str
    class_name: str
    category: str
    confidence: float
    bbox: str                  # JSON string
    created_at: datetime
```

---

## 🔧 Cấu Hình

### `app/config.py`

```python
class Settings:
    # Database
    DATABASE_URL: str = "sqlite:///./waste.db"
    
    # Model
    MODEL_PATH: str = "../../models/best.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
```

---

## 📊 Metrics & Monitoring

### Health Check

```bash
GET /health
```

Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "database_connected": true,
    "uptime": "2h 30m"
}
```

### Statistics

```bash
GET /stats
```

Response:
```json
{
    "total_detections": 1234,
    "today_detections": 56,
    "bins_count": 10,
    "full_bins": 3,
    "average_confidence": 0.87
}
```

---

## 🔐 Security

- CORS middleware configured
- Rate limiting (optional)
- API key authentication (optional)
- Input validation với Pydantic

---

## 🐛 Troubleshooting

### Model không load được

```bash
# Kiểm tra đường dẫn model
ls -la ../../models/best.pt

# Cài đặt ultralytics
pip install ultralytics
```

### Database error

```bash
# Xóa và tạo lại database
rm waste.db
python create_db.py
```

### Port đã được sử dụng

```bash
# Tìm process dùng port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Hoặc dùng port khác
uvicorn main:app --port 8001
```

---

*Tác giả: Huy Nguyen | Cập nhật: Tháng 12, 2025*
