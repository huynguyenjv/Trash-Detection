# Instruction Backend

## Yêu cầu
1. Viết bằng **Python + FastAPI**.
2. Module chính:
   - `detector.py`: Load YOLOv8, chạy inference trên frame.
   - `waste_manager.py`: Đếm số lượng object, phân loại hữu cơ/tái chế.
   - `pathfinding.py`: Implement A* trên bản đồ (dạng grid).
   - `backend.py`: Khởi tạo FastAPI, WebSocket cho video stream, REST API cho stats + path.

## API endpoints
- `POST /detect`: nhận frame, trả về bounding boxes + labels.
- `GET /stats`: trả về số lượng rác theo loại.
- `GET /path?lat=..&lon=..`: trả về đường đi từ điểm rác đến bãi rác gần nhất.
- WebSocket `/ws/detect`: truyền detection realtime.

## Yêu cầu code
- Viết code tách module rõ ràng.
- Đảm bảo chạy được bằng `uvicorn backend:app --reload`.
