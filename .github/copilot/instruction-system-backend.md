🧠 Project Instruction — Waste Detection & Pathfinding Backend (FastAPI)
🧩 Overview

Xây dựng FastAPI backend cho hệ thống phân loại rác bằng YOLOv8, hỗ trợ:

Detect nhiều đối tượng trong cùng một frame.

Detect nhiều frame cùng lúc (batch detection).

WebSocket realtime detection.

REST API trả về thống kê phân loại rác và đường đi tối ưu (A) đến bãi rác gần nhất*.

⚙️ Tech Stack

Language: Python 3.10+

Framework: FastAPI + Uvicorn

Model: YOLOv8 (Ultralytics)

Libs: OpenCV, NumPy

Pathfinding: A* (grid-based)

📁 Project Structure
waste-backend/
│
├── backend.py           # Main FastAPI app + routes + websocket
├── detector.py          # YOLOv8 inference (single & batch)
├── waste_manager.py     # Counting & classification logic
├── pathfinding.py       # A* pathfinding (multi-start support)
├── requirements.txt     # Dependencies
└── models/
    └── yolov8n.pt       # Pretrained YOLOv8 model

🧩 Modules Description
1. detector.py

Load YOLOv8 model (ultralytics.YOLO).

Detect objects from single frame (detect).

Detect objects from multiple frames (detect_batch).

Trả về list bounding boxes, labels, confidence per frame.

Helper: bytes_to_frame (convert bytes → OpenCV frame).

2. waste_manager.py

Đếm số lượng object theo loại:

organic, recyclable, hazardous, unknown.

Nhận batch detections từ detector.py, lưu vào bộ đếm tổng & recent list.

Cung cấp API:

get_stats() → trả tổng số lượng.

get_recent(limit) → trả danh sách detect gần nhất.

3. pathfinding.py

Cài đặt thuật toán A* trên bản đồ dạng grid.

Hàm find_nearest_bin_for_each:

Nhận danh sách điểm rác (starts) và bãi rác (bins).

Tìm đường đi ngắn nhất từ từng điểm rác đến bãi gần nhất.

4. backend.py

Khởi tạo FastAPI app.

Endpoints:

POST /detect: nhận 1 hoặc nhiều frame, trả về list kết quả detect.

GET /stats: trả thống kê rác (đếm theo loại, recent detections).

GET /path?starts=...: chạy A* cho nhiều điểm rác → bãi rác gần nhất.

WebSocket /ws/detect: stream realtime video detection (frame-by-frame).

Tích hợp các module: detector.py, waste_manager.py, pathfinding.py.

🔌 API Endpoints
1. POST /detect

Description: Nhận 1 hoặc nhiều hình ảnh (frame) để detect nhiều đối tượng.

Request:

Multipart form-data:

files: danh sách ảnh (List[UploadFile])

Response:

{
  "count": 2,
  "results": [
    {
      "timestamp": 1739548820.23,
      "detections": [
        {"bbox": [120, 45, 210, 170], "label": "bottle", "confidence": 0.92},
        {"bbox": [300, 80, 400, 200], "label": "banana", "confidence": 0.87}
      ]
    },
    ...
  ],
  "summaries": [
    {"timestamp": 1739548820.23, "counts": {"recyclable": 1, "organic": 1}}
  ]
}

2. GET /stats

Description: Trả thống kê số lượng rác theo loại.

Response:

{
  "totals": {"organic": 12, "recyclable": 9, "hazardous": 1, "unknown": 3},
  "recent": [
    {"timestamp": 1739548800.12, "label": "bottle", "type": "recyclable", "bbox": [120,45,210,170], "confidence": 0.92},
    ...
  ]
}

3. GET /path

Description: Trả đường đi tối ưu (A*) từ điểm rác đến bãi rác gần nhất.

Params:

starts: Danh sách điểm, ví dụ starts=5,5;10,17

hoặc lat, lon (demo chuyển sang grid)

Response:

{
  "paths": {
    "(5,5)": {"bin": [0,0], "path": [[5,5],[4,5],[3,5],...], "distance": 8},
    "(10,17)": {"bin": [19,19], "path": [[10,17],[11,17],...], "distance": 11}
  }
}

4. WebSocket /ws/detect

Description: Nhận từng frame (binary JPEG/PNG bytes), trả kết quả detect realtime.

Client → Server: gửi binary frame.
Server → Client: trả JSON:

{
  "timestamp": 1739548825.33,
  "detections": [
    {"bbox": [120,45,210,170], "label": "bottle", "confidence": 0.91},
    {"bbox": [300,80,400,200], "label": "banana", "confidence": 0.87}
  ]
}

🧪 Run & Test
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run server
uvicorn backend:app --reload

3️⃣ Test detect API
curl -X POST "http://localhost:8000/detect" \
  -F "files=@frame1.jpg" \
  -F "files=@frame2.jpg"

4️⃣ WebSocket test

Kết nối đến ws://localhost:8000/ws/detect

Gửi binary JPEG frames liên tục (client → server).

Server trả kết quả detect theo thời gian thực.

🧭 Notes & Extensions

Có thể thêm tracker.py (SORT/DeepSORT) để gán ID cho từng object giữa các frame.

Có thể lưu counters & recent detections vào Redis hoặc DB.

Map GPS → grid bằng module mapping riêng.

Giới hạn batch size để tránh GPU overload (max_batch_size trong detector.py).

Hỗ trợ gửi base64 frame nếu không dùng binary WebSocket.