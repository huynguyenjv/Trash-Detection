# Instruction Frontend

## Yêu cầu
1. Viết bằng **ReactJS + Vite + TailwindCSS**.
2. Sử dụng **Leaflet.js** để hiển thị bản đồ (giống Google Maps).
3. Các component:
   - `VideoStream.jsx`: hiển thị video realtime với detection overlay.
   - `WasteStats.jsx`: bảng số lượng & phân loại rác.
   - `MapView.jsx`: hiển thị bản đồ với vị trí rác, bãi rác, và đường đi.

## Luồng dữ liệu
- Video từ camera stream được gửi tới backend.
- Backend trả về detection data qua WebSocket → frontend overlay bounding boxes.
- Bảng thống kê cập nhật theo API `/stats`.
- Map gọi API `/path` để hiển thị đường đi.

## UI Layout
- Bên trái: Video detection + bảng thống kê.
- Bên phải: Bản đồ tương tác.

## Yêu cầu code
- Dùng WebSocket để nhận dữ liệu realtime từ backend.
- Dùng REST API để lấy stats và path.
- TailwindCSS cho styling, UI hiện đại.
