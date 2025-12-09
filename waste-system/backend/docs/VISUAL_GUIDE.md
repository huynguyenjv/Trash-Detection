# 🗺️ GOONG MAPS ROUTING - VISUAL GUIDE

## 📐 KIẾN TRÚC TỔNG QUAN

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  (React App - Goong Map Component)                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ HTTP Request
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API                                   │
│                  (FastAPI - Port 8000)                           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API Router: /routing                                     │  │
│  │  - GET  /routing/health                                   │  │
│  │  - POST /routing/route                                    │  │
│  │  - POST /routing/nearest-bin                              │  │
│  │  - POST /routing/optimize-route                           │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                             │
│                     ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GoongRoutingService                                      │  │
│  │  - get_route()                                            │  │
│  │  - find_nearest_bin_route()                               │  │
│  │  - get_optimized_route()                                  │  │
│  │  - decode_polyline()                                      │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                             │
└─────────────────────┼─────────────────────────────────────────┘
                      │
                      │ HTTPS Request
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              GOONG MAPS API                                      │
│         (https://rsapi.goong.io)                                │
│                                                                   │
│  - /Direction        (Get route between 2 points)               │
│  - /DistanceMatrix   (Get distance matrix)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 FLOW TÌM THÙNG RÁC GẦN NHẤT

```
User vứt chai nhựa
       │
       ▼
┌────────────────────────────────────────┐
│ Frontend gọi API:                      │
│ POST /routing/nearest-bin              │
│ {                                      │
│   latitude: 21.0285,                   │
│   longitude: 105.8542,                 │
│   category: "recyclable",              │
│   vehicle: "foot"                      │
│ }                                      │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Backend:                               │
│ 1. Query database → Lấy danh sách     │
│    thùng rác loại "recyclable"        │
│                                        │
│    bins = [                            │
│      {id: 1, lat: 21.03, lng: 105.85} │
│      {id: 2, lat: 21.04, lng: 105.86} │
│      {id: 3, lat: 21.05, lng: 105.87} │
│    ]                                   │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ GoongRoutingService:                   │
│ Lặp qua từng thùng, gọi Goong API     │
│                                        │
│ For each bin:                          │
│   route = get_route(user → bin)       │
│   if distance < shortest:             │
│     shortest = distance                │
│     best_bin = bin                     │
└───────────────┬────────────────────────┘
                │
                │ (3 API calls to Goong)
                ▼
┌────────────────────────────────────────┐
│ Goong Maps API Response:               │
│                                        │
│ Bin 1: 2.8km, 8min  ◄── SHORTEST     │
│ Bin 2: 4.2km, 12min                   │
│ Bin 3: 6.1km, 18min                   │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Backend Response:                      │
│ {                                      │
│   nearest_bin: {                       │
│     id: 1,                             │
│     name: "Thùng A",                   │
│     address: "123 ABC"                 │
│   },                                   │
│   route: {                             │
│     distance_km: 2.8,                  │
│     duration_minutes: 8,               │
│     polyline: "encoded...",            │
│     steps: [                           │
│       "Rẽ trái vào đường X",          │
│       "Đi thẳng 200m",                │
│       "Rẽ phải..."                    │
│     ]                                  │
│   }                                    │
│ }                                      │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Frontend hiển thị:                     │
│ ┌──────────────────────────────────┐  │
│ │  🗑️  Thùng gần nhất               │  │
│ │  📍 Thùng A - 123 ABC             │  │
│ │  📏 2.8km                          │  │
│ │  ⏱️  8 phút đi bộ                  │  │
│ │                                    │  │
│ │  [Xem đường đi trên bản đồ]      │  │
│ └──────────────────────────────────┘  │
│                                        │
│  ┌─────────────────────────────────┐ │
│  │         GOONG MAP                │ │
│  │                                  │ │
│  │    👤 (User)                     │ │
│  │      │                           │ │
│  │      │ (Route polyline)          │ │
│  │      ▼                           │ │
│  │    🗑️ (Bin A)                    │ │
│  │                                  │ │
│  └─────────────────────────────────┘ │
└────────────────────────────────────────┘
```

---

## 🚛 FLOW TỐI ƯU LỘCH TRÌNH THU GOM

```
Xe thu gom cần ghé 5 thùng
       │
       ▼
┌────────────────────────────────────────┐
│ POST /routing/optimize-route           │
│ {                                      │
│   origin: [21.028, 105.854],          │
│   destination: [21.037, 105.834],     │
│   bin_ids: [1, 2, 3, 4, 5],           │
│   vehicle: "car"                       │
│ }                                      │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Backend query bins:                    │
│                                        │
│ Bin 1: (21.03, 105.85)                │
│ Bin 2: (21.04, 105.86)                │
│ Bin 3: (21.05, 105.87)                │
│ Bin 4: (21.02, 105.84)                │
│ Bin 5: (21.06, 105.88)                │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Goong API: Get optimized route         │
│ With waypoints optimization            │
│                                        │
│ Goong returns optimized order:        │
│ Start → Bin 4 → Bin 1 → Bin 2 →       │
│ → Bin 3 → Bin 5 → End                 │
│                                        │
│ Total: 12.5km, 35 minutes             │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Response with optimized route:        │
│ {                                      │
│   total_distance_km: 12.5,            │
│   total_duration_minutes: 35,         │
│   waypoint_order: [3, 0, 1, 2, 4],   │
│   legs: [                              │
│     {from: "Start", to: "Bin 4",      │
│      distance: 1.5km, time: 5min},    │
│     {from: "Bin 4", to: "Bin 1",      │
│      distance: 2.8km, time: 8min},    │
│     ...                                │
│   ],                                   │
│   polyline: "encoded..."               │
│ }                                      │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│ Driver App shows:                      │
│                                        │
│ 🚛 Lộ trình tối ưu hôm nay            │
│                                        │
│ ✅ 1. Thùng 4 (1.5km, 5min)           │
│ ⏱️  2. Thùng 1 (2.8km, 8min)          │
│ ⏱️  3. Thùng 2 (3.2km, 9min)          │
│ ⏱️  4. Thùng 3 (2.5km, 7min)          │
│ ⏱️  5. Thùng 5 (2.5km, 6min)          │
│                                        │
│ 📊 Tổng: 12.5km, 35 phút              │
│                                        │
│ [Bắt đầu thu gom]                     │
└────────────────────────────────────────┘
```

---

## 🔀 SO SÁNH: OLD vs NEW

### OLD METHOD (Straight-line)
```
User Location: (21.0285, 105.8542)
       │
       ▼
Calculate Haversine distance
       │
       ▼
Bin 1: 1.2km (straight)
Bin 2: 1.8km (straight)
Bin 3: 2.5km (straight)
       │
       ▼
Nearest: Bin 1 (1.2km)
       │
       ▼
❌ Problems:
- Không biết đường đi thực
- Không có thời gian
- Không có hướng dẫn
- Khoảng cách không chính xác
```

### NEW METHOD (Goong Maps)
```
User Location: (21.0285, 105.8542)
       │
       ▼
Goong API: Get real routes
       │
       ▼
Bin 1: 2.8km, 8min (actual road)
Bin 2: 3.5km, 10min (actual road)
Bin 3: 4.2km, 12min (actual road)
       │
       ▼
Nearest: Bin 1 (2.8km, 8min)
       │
       ▼
✅ Benefits:
+ Khoảng cách chính xác (theo đường)
+ Có thời gian dự kiến
+ Có hướng dẫn chi tiết
+ Có polyline để vẽ trên map
+ Hỗ trợ nhiều phương tiện
```

---

## 🎨 FRONTEND INTEGRATION

### React Component Example:
```javascript
// 1. Fetch nearest bin
const response = await fetch('/routing/nearest-bin', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    latitude: userLocation.lat,
    longitude: userLocation.lng,
    category: 'recyclable',
    vehicle: 'foot'
  })
});

const { nearest_bin, route } = await response.json();

// 2. Decode polyline
const polylineResponse = await fetch(
  `/routing/decode-polyline?encoded=${route.polyline}`
);
const { coordinates } = await polylineResponse.json();

// 3. Display on Goong Map
const map = new goongjs.Map({
  container: 'map',
  style: 'https://tiles.goong.io/assets/goong_map_web.json',
  center: [userLocation.lng, userLocation.lat],
  zoom: 14
});

// Add route polyline
map.addSource('route', {
  type: 'geojson',
  data: {
    type: 'Feature',
    geometry: {
      type: 'LineString',
      coordinates: coordinates.map(c => [c.lng, c.lat])
    }
  }
});

map.addLayer({
  id: 'route',
  type: 'line',
  source: 'route',
  paint: {
    'line-color': '#3b82f6',
    'line-width': 4
  }
});

// Add markers
new goongjs.Marker({ color: 'red' })
  .setLngLat([userLocation.lng, userLocation.lat])
  .addTo(map);

new goongjs.Marker({ color: 'green' })
  .setLngLat([nearest_bin.longitude, nearest_bin.latitude])
  .addTo(map);

// Show info
console.log(`Distance: ${route.distance_km}km`);
console.log(`Duration: ${route.duration_minutes} minutes`);
```

---

## 📊 API RATE LIMITS

```
Goong Maps Free Tier:
├── Directions API: 2,500 requests/day
├── Distance Matrix: 2,500 requests/day
└── Geocoding: 2,500 requests/day

Example usage:
- User finds nearest bin: 1 request (if 3 bins = 3 requests)
- Optimize 10 bins route: 1 request
- 100 users/day × 5 searches = 500 requests ✅ OK

⚠️  If exceeding limit:
- Cache routes for popular locations
- Implement rate limiting per user
- Upgrade to paid plan
```

---

## 🔧 CONFIGURATION MATRIX

| Environment | Goong Enabled | API Key | Behavior |
|-------------|---------------|---------|----------|
| Development | ❌ | - | Fallback (Haversine) |
| Development | ✅ | Invalid | Fallback + Error log |
| Development | ✅ | Valid | Real routing ✅ |
| Production | ❌ | - | ⚠️  Not recommended |
| Production | ✅ | Valid | Real routing ✅ |

---

## ✅ CHECKLIST TRIỂN KHAI

### Backend:
- [x] ✅ Tích hợp GoongRoutingService
- [x] ✅ Tạo API endpoints
- [x] ✅ Fallback mode
- [x] ✅ Error handling
- [x] ✅ Documentation

### Frontend (TODO):
- [ ] Tích hợp Goong Map component
- [ ] Gọi routing API
- [ ] Hiển thị route trên map
- [ ] Show turn-by-turn directions
- [ ] Add "Navigate" button

### Production (TODO):
- [ ] Setup environment variables
- [ ] Monitor API usage
- [ ] Cache frequently used routes
- [ ] Add rate limiting
- [ ] Set up alerts

---

## 🎉 KẾT LUẬN

**Backend đã HOÀN THÀNH:**
- ✅ Service: GoongRoutingService
- ✅ API: 6 endpoints
- ✅ Fallback: Haversine distance
- ✅ Documentation: Complete
- ✅ Demo: Scripts ready

**Sẵn sàng cho Frontend tích hợp!** 🚀
