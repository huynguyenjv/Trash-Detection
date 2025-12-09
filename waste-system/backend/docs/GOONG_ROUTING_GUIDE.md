# 🗺️ Goong Maps Routing Integration

Hệ thống đã được nâng cấp với tính năng **routing thực tế** sử dụng **Goong Maps API** (tương tự Google Maps cho Việt Nam).

## 📋 Tính năng mới

### 1. **Real-world Routing**
- Tính đường đi thực tế theo đường bộ (không phải đường chim bay)
- Hỗ trợ 3 phương tiện: `car`, `bike`, `foot`
- Trả về khoảng cách, thời gian, và hướng dẫn chi tiết

### 2. **Find Nearest Bin**
- Tìm thùng rác gần nhất dựa trên khoảng cách đường đi thực
- Filter theo loại rác (organic, recyclable, hazardous, other)
- Trả về route đầy đủ để hiển thị trên map

### 3. **Route Optimization**
- Tối ưu lộ trình thu gom rác (cho xe thu gom)
- Ghé thăm nhiều thùng rác với thứ tự tối ưu
- Tiết kiệm thời gian và nhiên liệu

### 4. **Distance Matrix**
- Tính ma trận khoảng cách giữa nhiều điểm
- Hữu ích cho planning và optimization

## 🚀 Cách sử dụng

### Bước 1: Đăng ký Goong Maps API Key

1. Truy cập: https://account.goong.io/
2. Đăng ký tài khoản (miễn phí)
3. Tạo API key mới
4. Copy API key

### Bước 2: Cấu hình Backend

Tạo file `.env` trong thư mục `backend/`:

```env
# Goong Maps Configuration
GOONG_API_KEY=your_api_key_here
GOONG_MAPS_ENABLED=true
```

### Bước 3: Cài đặt dependencies

```bash
pip install requests
# hoặc
pip install -r requirements.txt
```

### Bước 4: Khởi động server

```bash
cd waste-system/backend
python main.py
```

## 📡 API Endpoints

### 1. Check Status
```http
GET /routing/health
```

Response:
```json
{
  "goong_enabled": true,
  "api_key_configured": true,
  "status": "ready"
}
```

### 2. Get Route Between Two Points
```http
POST /routing/route
Content-Type: application/json

{
  "origin_lat": 21.0285,
  "origin_lng": 105.8542,
  "dest_lat": 21.0378,
  "dest_lng": 105.8345,
  "vehicle": "foot"
}
```

Response:
```json
{
  "method": "goong_maps",
  "route": {
    "distance_km": 1.2,
    "distance_text": "1.2 km",
    "duration_minutes": 15.5,
    "duration_text": "16 phút",
    "polyline": "encoded_polyline_string",
    "steps": [
      {
        "instruction": "Đi về hướng đông trên Đường ABC",
        "distance_meters": 200,
        "duration_seconds": 120
      }
    ]
  }
}
```

### 3. Find Nearest Waste Bin
```http
POST /routing/nearest-bin
Content-Type: application/json

{
  "latitude": 21.0285,
  "longitude": 105.8542,
  "category": "recyclable",
  "vehicle": "foot"
}
```

Response:
```json
{
  "method": "goong_maps",
  "nearest_bin": {
    "id": 5,
    "name": "Thùng rác tái chế A",
    "category": "recyclable",
    "address": "123 Đường XYZ",
    "capacity": 75.5
  },
  "route": {
    "distance_km": 0.8,
    "duration_minutes": 10.5,
    "polyline": "...",
    "steps": [...]
  }
}
```

### 4. Optimize Collection Route
```http
POST /routing/optimize-route
Content-Type: application/json

{
  "origin_lat": 21.0285,
  "origin_lng": 105.8542,
  "dest_lat": 21.0378,
  "dest_lng": 105.8345,
  "bin_ids": [1, 3, 5, 7, 9],
  "vehicle": "car"
}
```

Response:
```json
{
  "method": "goong_maps",
  "optimization": {
    "total_distance_km": 8.5,
    "total_duration_minutes": 25.3,
    "waypoint_order": [0, 2, 1, 3, 4],
    "legs": [
      {
        "distance_km": 1.5,
        "duration_minutes": 5.2,
        "start_address": "...",
        "end_address": "..."
      }
    ],
    "polyline": "...",
    "bins": [...]
  }
}
```

## 🎨 Tích hợp Frontend

### Hiển thị route trên map

```javascript
// 1. Gọi API để lấy route
const response = await fetch('/routing/nearest-bin', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    latitude: 21.0285,
    longitude: 105.8542,
    category: 'recyclable',
    vehicle: 'foot'
  })
});

const data = await response.json();

// 2. Decode polyline
const polylineResponse = await fetch(
  `/routing/decode-polyline?encoded=${data.route.polyline}`
);
const { coordinates } = await polylineResponse.json();

// 3. Vẽ route trên Goong Map
const route = new goongjs.Polyline({
  coordinates: coordinates.map(c => [c.lng, c.lat]),
  color: '#3b82f6',
  width: 4
});

map.addLayer(route);

// 4. Hiển thị thông tin
console.log(`Khoảng cách: ${data.route.distance_km}km`);
console.log(`Thời gian: ${data.route.duration_minutes} phút`);
```

## 🔄 Fallback Mode

Nếu không cấu hình Goong API, hệ thống tự động chuyển sang **Fallback Mode**:
- Sử dụng công thức Haversine (khoảng cách đường chim bay)
- Vẫn hoạt động nhưng không chính xác bằng routing thực

```json
{
  "method": "straight_line",
  "distance_km": 0.65,
  "warning": "Goong Maps not configured. Using straight-line distance."
}
```

## 📊 So sánh

| Feature | Straight Line | Goong Maps |
|---------|--------------|------------|
| Khoảng cách | ✅ Có | ✅ Có (chính xác) |
| Thời gian đi | ❌ Không | ✅ Có |
| Đường đi chi tiết | ❌ Không | ✅ Có |
| Hướng dẫn rẽ | ❌ Không | ✅ Có |
| Hiển thị trên map | ❌ Đường thẳng | ✅ Đường thực |
| Chi phí | Miễn phí | Miễn phí (có giới hạn) |

## 📈 Giới hạn API (Free tier)

Goong Maps Free tier:
- **2,500 requests/ngày** cho Directions API
- **2,500 requests/ngày** cho Distance Matrix API
- Phù hợp cho development và testing

Nếu cần nhiều hơn, nâng cấp lên plan trả phí.

## 🛠️ Troubleshooting

### Lỗi: "Goong API request failed"
- Kiểm tra API key có đúng không
- Kiểm tra internet connection
- Kiểm tra tọa độ có hợp lệ không (trong phạm vi Việt Nam)

### Lỗi: "No route found"
- Tọa độ có thể nằm ngoài vùng hỗ trợ
- Thử đổi vehicle type (foot → car)
- Kiểm tra tọa độ có đúng định dạng không

## 💡 Use Cases

### 1. Mobile App - Tìm thùng rác gần nhất
```python
# User ở vị trí (21.0285, 105.8542)
# Muốn vứt chai nhựa (recyclable)

POST /routing/nearest-bin
{
  "latitude": 21.0285,
  "longitude": 105.8542,
  "category": "recyclable",
  "vehicle": "foot"
}

# → App hiển thị thùng gần nhất + đường đi trên map
```

### 2. Waste Collection Truck - Tối ưu lộ trình
```python
# Xe thu gom cần ghé 10 thùng rác

POST /routing/optimize-route
{
  "origin_lat": 21.0285,
  "origin_lng": 105.8542,
  "dest_lat": 21.0378,
  "dest_lng": 105.8345,
  "bin_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "vehicle": "car"
}

# → Hệ thống trả về thứ tự tối ưu để tiết kiệm thời gian
```

### 3. Dashboard - Thống kê khoảng cách trung bình
```python
# Tính khoảng cách từ nhiều điểm phát hiện rác đến thùng

GET /routing/distance-matrix?
  origins=21.02,105.85|21.03,105.86&
  destinations=21.04,105.87|21.05,105.88&
  vehicle=foot

# → Ma trận khoảng cách để phân tích
```

## 🎯 Kết luận

Tích hợp Goong Maps giúp:
- ✅ Chính xác hơn (đường thực thay vì đường chim bay)
- ✅ Có thời gian dự kiến
- ✅ Có hướng dẫn chi tiết
- ✅ Hiển thị đẹp trên map
- ✅ Tối ưu lộ trình thu gom rác

**API đã sẵn sàng để tích hợp vào frontend!** 🚀
