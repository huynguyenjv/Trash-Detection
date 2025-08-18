# Smart Waste Detection System - Backend

Python FastAPI backend với YOLOv8 detection, A* pathfinding và WebSocket real-time streaming.

## 🚀 Features

- **YOLOv8 Detection**: Real-time waste detection với custom hoặc pre-trained models
- **A* Pathfinding**: Tìm đường tối ưu đến bãi rác gần nhất
- **WebSocket Streaming**: Real-time detection streaming cho frontend
- **Waste Management**: Thống kê và quản lý các loại rác
- **REST API**: Endpoints đầy đủ cho frontend integration

## 📁 Cấu trúc

```
backend/
├── backend.py          # FastAPI main server
├── detector.py         # YOLOv8 detection engine
├── waste_manager.py    # Waste statistics & bin management
├── pathfinding.py      # A* algorithm implementation
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## 🛠️ Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download YOLOv8 Model (optional)
Hệ thống sẽ tự động tải YOLOv8n nếu không có custom model.
Để dùng custom model, đặt file `.pt` trong thư mục `models/`.

## 🚀 Usage

### Start Server
```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

Hoặc:
```bash
python backend.py
```

Server sẽ chạy tại: `http://localhost:8000`

## 📡 API Endpoints

### REST API

#### `GET /`
Health check và thông tin API

#### `POST /detect`
Detect waste trong ảnh
```json
{
    "image": "base64_encoded_image",
    "confidence_threshold": 0.5
}
```

#### `GET /stats`
Lấy thống kê waste hiện tại và trends

#### `GET /bins`
Lấy danh sách tất cả waste bins

#### `GET /path?lat=10.8231&lon=106.6297&waste_type=recyclable`
Tính đường đi tới bin phù hợp
- `lat`, `lon`: Vị trí hiện tại
- `dest_lat`, `dest_lon`: Đích đến (optional)
- `waste_type`: Loại rác (organic, recyclable, hazardous, other)

### WebSocket

#### `ws://localhost:8000/ws/detect`
Real-time detection streaming

**Send frame:**
```json
{
    "type": "frame",
    "image": "base64_encoded_image"
}
```

**Receive result:**
```json
{
    "type": "detection_result", 
    "detections": [...],
    "timestamp": "2025-08-18T..."
}
```

## 🧠 Modules

### detector.py
- Load YOLOv8 models (custom hoặc pre-trained)
- Base64 image processing
- Object detection và classification
- Waste categorization (organic, recyclable, hazardous, other)

### waste_manager.py
- Real-time statistics tracking
- Waste bin locations management
- Nearest bin finding với distance calculation
- Historical data trends

### pathfinding.py
- A* algorithm implementation
- Grid-based pathfinding
- Lat/lon coordinate conversion
- Route optimization với obstacles

### backend.py
- FastAPI server setup
- CORS configuration
- WebSocket connection management
- API endpoint implementations
- Error handling

## 🎯 Waste Categories

- **Organic**: Food waste, biodegradable materials
- **Recyclable**: Plastic, paper, metal containers
- **Hazardous**: Electronics, batteries, chemicals
- **Other**: General waste không thuộc categories trên

## 🗺️ Default Bin Locations

Hệ thống có sẵn waste bins tại Ho Chi Minh City:
- Central Waste Bin (10.8231, 106.6297)
- Recycling Centers
- Organic Waste Facilities  
- Hazardous Waste Centers

## 🔧 Configuration

### Custom Model
Đặt trained model vào:
```
../../models/trash_safe_best.pt
../models/trash_safe_best.pt
./models/trash_safe_best.pt
```

### Grid Size
Modify pathfinding grid size trong `pathfinding.py`:
```python
pathfinder = AStarPathfinder(grid_size=100)
```

### Waste Categories
Customize waste mapping trong `detector.py`:
```python
self.waste_categories = {
    'bottle': 'recyclable',
    'apple': 'organic',
    # Add more...
}
```

## 🚨 Error Handling

- Tự động fallback sang YOLOv8n nếu custom model fail
- WebSocket auto-reconnect handling
- Direct line routing nếu A* pathfinding fail
- Graceful error responses cho tất cả endpoints

## 📊 Performance

- Detection: ~50-100ms per frame
- A* pathfinding: ~10-50ms per route
- WebSocket: Real-time streaming (10 FPS)
- Memory usage: ~500MB với YOLOv8n

## 🔗 Integration

Frontend cần kết nối tới:
- HTTP API: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws/detect`

CORS đã enable cho development. Production cần configure specific origins.
