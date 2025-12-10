# 🚀 Hướng dẫn chạy Frontend với Routing mới

## ✨ Tính năng mới

Frontend đã được tích hợp với **Routing API backend** với các tính năng:

### 1. **5 Thuật toán tìm đường**
- **Weighted Score** (⚖️): Cân bằng khoảng cách 70% + thời gian 30%
- **Dijkstra** (📏): Ưu tiên đường ngắn nhất
- **A*** (🎯): Kết hợp khoảng cách + heuristic
- **Multi-Criteria** (🔬): Xét nhiều yếu tố (khoảng cách, thời gian, giao thông, nhiên liệu)
- **Greedy** (⚡): Chọn nhanh đường gần nhất

### 2. **Hiển thị đường đi thực tế**
- 100+ tọa độ theo đường phố (không phải chim bay)
- Polyline từ Goong Maps API
- Decode coordinates tự động

### 3. **Thông tin chi tiết**
- Khoảng cách (km/m)
- Thời gian ước tính (phút)
- Điểm số thuật toán
- Số thùng rác đã so sánh
- Số đường đi đã xét

---

## 🔧 Cài đặt

### 1. Cài dependencies

```bash
cd waste-system/frontend
npm install
```

### 2. Cấu hình Backend URL

File `.env` đã được tạo:

```env
VITE_API_URL=http://localhost:8000
```

Nếu backend chạy ở port khác, sửa lại URL.

---

## 🚀 Chạy Frontend

```bash
npm run dev
```

Frontend sẽ chạy tại: **http://localhost:5173**

---

## 📖 Cách sử dụng

### 1. **Chuyển giữa Video và Map**

Nhấn nút **🗺️ Bản đồ** ở header để chuyển sang chế độ bản đồ.

### 2. **Chọn thuật toán**

Dropdown menu hiển thị 5 thuật toán:
- ⚖️ Weighted Score
- 📏 Dijkstra
- 🎯 A* Search
- 🔬 Multi-Criteria
- ⚡ Greedy (Nhanh)

### 3. **Tìm đường đi**

Nhấn **🎯 Tìm đường đi** để:
- Tìm thùng rác gần nhất từ vị trí hiện tại
- Hiển thị route với 100+ điểm theo đường thật
- Xem thông tin: khoảng cách, thời gian, điểm số

### 4. **Xem kết quả**

Bảng **Thông tin đường đi** hiển thị:
- Điểm đến
- Khoảng cách (m/km)
- Thời gian (phút)
- Loại thùng rác
- **Điểm số** (do thuật toán tính)
- **Đã so sánh** (số thùng rác)
- Phương thức (Goong Maps / Fallback)

---

## 🎨 UI Components

### MapView Component

**Props:**
- `autoFindRoute`: Tự động tìm đường khi phát hiện rác
- `detectedWaste`: Thông tin rác phát hiện

**State:**
- `currentLocation`: Vị trí hiện tại
- `wasteBins`: Danh sách thùng rác từ backend
- `selectedPath`: Đường đi đang hiển thị
- `selectedAlgorithm`: Thuật toán đang chọn
- `routingServiceStatus`: Trạng thái Goong Maps API

### Routing Service

File: `src/services/routingService.js`

**Functions:**
- `getRoute()`: Lấy route giữa 2 điểm
- `findNearestBin()`: Tìm thùng rác gần nhất
- `optimizeRoute()`: Tối ưu lộ trình nhiều thùng
- `decodePolyline()`: Decode polyline thành coordinates
- `getAllBins()`: Lấy danh sách thùng rác
- `checkRoutingHealth()`: Kiểm tra trạng thái service

---

## 🔍 Testing

### 1. Test với Backend đang chạy

Đảm bảo backend đã start:

```bash
cd waste-system/backend
uvicorn main:app --reload
```

Backend: http://localhost:8000

### 2. Test routing API

Mở browser console (F12) và xem logs:

```
Routing service status: { goong_enabled: true, status: 'ready' }
```

### 3. Test tìm đường

1. Cho phép browser truy cập location
2. Nhấn **🗺️ Bản đồ**
3. Chọn thuật toán (VD: A*)
4. Nhấn **🎯 Tìm đường đi**
5. Xem route hiển thị trên map

---

## 🐛 Troubleshooting

### Lỗi: "Cannot connect to backend"

**Nguyên nhân:** Backend chưa chạy hoặc URL sai

**Giải pháp:**
```bash
# Kiểm tra backend
curl http://localhost:8000/api/routing/health

# Nếu lỗi, start backend
cd waste-system/backend
uvicorn main:app --reload
```

### Lỗi: "Không thể tính đường đi"

**Nguyên nhân:** Goong API key chưa config

**Giải pháp:**
- Hệ thống tự động fallback sang straight-line distance
- Hoặc config Goong API key trong backend `.env`:
  ```env
  GOONG_API_KEY=your_api_key_here
  GOONG_MAPS_ENABLED=true
  ```

### Map không hiển thị

**Nguyên nhân:** Leaflet CSS chưa load

**Giải pháp:**
- Kiểm tra file `index.html` có import Leaflet CSS:
  ```html
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  ```

### Route vẫn là đường thẳng

**Nguyên nhân:** Goong API chưa enable

**Kiểm tra:**
```javascript
// Console sẽ hiển thị
Routing service status: { goong_enabled: false, status: 'fallback_mode' }
```

**Giải pháp:** Config Goong API key trong backend

---

## 📦 Dependencies

Package đã có trong `package.json`:

- **react-leaflet**: Map component
- **leaflet**: Map library
- **react**: UI framework
- **vite**: Build tool

Không cần cài thêm gì!

---

## 🎯 Next Steps

### 1. Thêm tính năng

- [ ] Tối ưu lộ trình nhiều thùng rác
- [ ] Hiển thị hướng dẫn từng bước (turn-by-turn)
- [ ] So sánh nhiều thuật toán cùng lúc
- [ ] Lưu lịch sử đường đi

### 2. UI/UX

- [ ] Animation cho route drawing
- [ ] Loading skeleton
- [ ] Toast notifications
- [ ] Responsive cho mobile

### 3. Performance

- [ ] Cache routing results
- [ ] Debounce location updates
- [ ] Lazy load map tiles

---

## ✅ Checklist

Trước khi demo:

- [ ] Backend đang chạy (`http://localhost:8000`)
- [ ] Frontend đang chạy (`http://localhost:5173`)
- [ ] Browser đã cho phép location access
- [ ] Console không có errors
- [ ] Map hiển thị được
- [ ] Có thể tìm đường đi
- [ ] Route hiển thị trên map
- [ ] Thông tin đường đi hiển thị đầy đủ

---

## 📚 Documentation

- [Backend API Documentation](../backend/API_FOR_FRONTEND.md)
- [Routing Architecture](../backend/docs/HYBRID_ARCHITECTURE_DETAIL.md)
- [Algorithm Guide](../backend/README.md)

---

## 🎉 Done!

Frontend đã sẵn sàng với routing system hoàn chỉnh! 🚀

**Test ngay:**
```bash
npm run dev
```

Nhấn 🗺️ Bản đồ → Chọn thuật toán → 🎯 Tìm đường đi
