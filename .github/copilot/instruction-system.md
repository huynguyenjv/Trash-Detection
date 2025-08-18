# Instruction cho hệ thống phát hiện và xử lý rác thải

## 🎯 Mục tiêu
Xây dựng một **ứng dụng web hoàn chỉnh** có thể:
1. Phát hiện rác thải realtime bằng YOLOv8.
2. Đếm số lượng và phân loại rác (hữu cơ, tái chế).
3. Tìm bãi rác gần nhất dựa vào kết quả detect.
4. Tìm đường đi ngắn nhất từ điểm phát hiện đến bãi rác bằng A*.
5. Hiển thị kết quả trên web (video detection + bản đồ kiểu Google Maps).

---

## 🏗️ Kiến trúc
- **Backend (Python + FastAPI)**:
  - Chạy YOLOv8 inference realtime.
  - API trả về bounding boxes, số lượng, phân loại rác.
  - API chọn bãi rác gần nhất và tính đường đi bằng A*.
- **Frontend (ReactJS + Tailwind + Leaflet.js)**:
  - Hiển thị video detection realtime với bounding boxes overlay.
  - Hiển thị bảng thống kê số lượng rác theo loại.
  - Hiển thị bản đồ (giống Google Maps) với bãi rác và đường đi tìm được.

---

## 🔄 Flow hoạt động
1. Người dùng mở ứng dụng web.
2. Camera/video stream được gửi tới backend.
3. Backend chạy YOLOv8:
   - Trả về bounding boxes + labels.
   - Đếm số lượng và phân loại rác (hữu cơ, tái chế).
4. Backend chọn bãi rác gần nhất từ danh sách.
5. Backend chạy A* để tìm đường đi ngắn nhất.
6. Frontend hiển thị:
   - Video stream + detection overlay.
   - Số lượng rác theo loại.
   - Bản đồ với bãi rác và đường đi.

---

## 📂 Cấu trúc thư mục
waste-system/
│── backend/
│ ├── detector.py # YOLOv8 inference
│ ├── waste_manager.py # Phân loại & chọn bãi rác phù hợp
│ ├── pathfinding.py # Thuật toán A*
│ ├── backend.py # FastAPI server
│ └── requirements.txt
│
│── frontend/
│ ├── src/
│ │ ├── components/
│ │ │ ├── VideoStream.jsx
│ │ │ ├── WasteStats.jsx
│ │ │ └── MapView.jsx
│ │ ├── App.jsx
│ │ └── main.jsx
│ ├── package.json
│ └── tailwind.config.js
│
└── instruction.md

---

## 📌 Yêu cầu code
- Code phải chạy được ngay sau khi `pip install -r requirements.txt` và `npm install`.
- Backend có endpoint:
  - `POST /detect` → nhận frame video, trả về kết quả YOLOv8.
  - `GET /stats` → trả về số lượng & phân loại rác.
  - `GET /path` → trả về đường đi từ vị trí rác đến bãi rác.
- Frontend:
  - Gọi API để lấy data detection và pathfinding.
  - Render video realtime với bounding boxes.
  - Render bảng thống kê rác.
  - Render bản đồ Leaflet với marker + đường đi.

---

## 🚀 Công nghệ đề xuất
- **Backend**: Python, FastAPI, YOLOv8 (Ultralytics), OpenCV, Numpy.
- **Frontend**: ReactJS, TailwindCSS, Leaflet.js.
- **Realtime**: WebSocket (FastAPI hỗ trợ) để truyền detection data.