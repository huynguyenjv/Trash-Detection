# Waste Detection Backend V2

Clean implementation theo instruction với YOLOv8n default model.

## 🚀 Features

- ✅ Multi-object detection (nhiều objects trong 1 frame)
- ✅ Batch detection (nhiều frames cùng lúc)
- ✅ WebSocket realtime detection
- ✅ Waste classification (organic, recyclable, hazardous, other)
- ✅ A* pathfinding to nearest bins
- ✅ Statistics tracking

## 📦 Installation

```bash
cd waste-system/backend-v2
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 🏃 Run Server

```bash
python backend.py
# hoặc
uvicorn backend:app --reload
```

Server sẽ chạy tại: `http://localhost:8000`

## 🧪 Test Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/
```

### 2. Detect Images (POST)
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 3. Get Statistics
```bash
curl http://localhost:8000/stats
```

### 4. Get Paths (A*)
```bash
curl "http://localhost:8000/path?starts=5,5;10,17"
```

### 5. WebSocket (Realtime)
```javascript
// JavaScript client
const ws = new WebSocket('ws://localhost:8000/ws/detect');

ws.onopen = () => {
  // Send frame (binary JPEG bytes)
  ws.send(frameBytes);
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(result.detections);
};
```

## 📊 API Response Examples

### POST /detect
```json
{
  "count": 2,
  "results": [
    {
      "timestamp": 1739548820.23,
      "detections": [
        {
          "bbox": [120, 45, 210, 170],
          "label": "bottle",
          "confidence": 0.92,
          "category": "recyclable"
        },
        {
          "bbox": [300, 80, 400, 200],
          "label": "banana",
          "confidence": 0.87,
          "category": "organic"
        }
      ]
    }
  ],
  "summaries": [
    {
      "timestamp": 1739548820.23,
      "counts": {"recyclable": 1, "organic": 1}
    }
  ]
}
```

### GET /stats
```json
{
  "totals": {
    "organic": 12,
    "recyclable": 9,
    "hazardous": 1,
    "other": 3
  },
  "recent": [
    {
      "timestamp": "2025-01-15T10:30:00",
      "label": "bottle",
      "category": "recyclable",
      "confidence": 0.92,
      "bbox": [120, 45, 210, 170]
    }
  ]
}
```

### GET /path
```json
{
  "paths": {
    "(5, 5)": {
      "bin": [0, 0],
      "path": [[5, 5], [4, 5], [3, 5], ..., [0, 0]],
      "distance": 10
    },
    "(10, 17)": {
      "bin": [19, 19],
      "path": [[10, 17], [11, 17], ..., [19, 19]],
      "distance": 11
    }
  }
}
```

## 🎯 Modules

### detector.py
- YOLOv8 detection engine
- Single & batch detection
- COCO classes → waste categories mapping

### waste_manager.py
- Statistics tracking
- Recent detections history
- Counter management

### pathfinding.py
- A* algorithm implementation
- Grid-based pathfinding
- Find nearest bin for each waste location

### backend.py
- FastAPI application
- REST endpoints
- WebSocket support
- CORS enabled

## 🔧 Configuration

Model: `yolov8n.pt` (80 COCO classes)
- Confidence threshold: `0.25`
- IOU threshold: `0.45`

Waste Categories:
- **Recyclable**: bottle, cup, fork, knife, spoon, bowl, book
- **Organic**: banana, apple, orange, carrot, pizza, cake
- **Hazardous**: cell phone, laptop, mouse, keyboard, scissors
- **Other**: anything else (not ignored)
- **Ignore**: person, car, truck, bus, bicycle

## 📝 Notes

- Dùng YOLOv8n default (COCO 80 classes) để test system
- Có thể train custom model sau với dataset rác của bạn
- WebSocket support binary frames (JPEG/PNG bytes)
- A* grid size: 20x20 (có thể customize)
