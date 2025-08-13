# Smart Waste Collection Routing System

Hệ thống thông minh theo dõi rác thải và định tuyến tối ưu cho xe gom rác sử dụng YOLOv8 + thuật toán A*.

## 🌟 Tính năng

### 1. Theo dõi rác thải real-time
- Phát hiện và phân loại rác thải bằng YOLOv8
- Đếm số lượng từng loại rác (hữu cơ, nhựa, thủy tinh, kim loại, giấy, v.v.)
- Cảnh báo khi đạt threshold để gom rác

### 2. Định tuyến thông minh A*
- Tìm đường tối ưu đến bãi rác phù hợp
- Tính toán chi phí dựa trên:
  - Khoảng cách Haversine
  - Điều kiện giao thông
  - Chất lượng đường
  - Trạng thái bãi rác (FULL/NEAR_FULL/OK)
- Penalty cho bãi rác gần đầy

### 3. Visualisation 
- Hiển thị đường đi trên bản đồ
- Thông tin chi tiết: khoảng cách, ETA, cost
- Real-time display trên video/camera

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd Trash-Detection
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements_routing.txt
```

### 3. Tải model YOLOv8 đã train
Đảm bảo có file model trong `models/` directory:
```
models/
├── trash_safe_best.pt        # Model đã train
└── trash_detection_best.pt   # Hoặc model khác
```

## 📖 Sử dụng

### 1. Demo Real-time với Camera
```bash
cd src
python demo_realtime.py --model ../models/trash_safe_best.pt --camera 0 --threshold 10
```

### 2. Xử lý Video
```bash
python demo_realtime.py --model ../models/trash_safe_best.pt --video video.mp4 --threshold 5
```

### 3. Xử lý Single Image
```bash
python demo_realtime.py --model ../models/trash_safe_best.pt --image image.jpg --threshold 1
```

### 4. Test hệ thống với dữ liệu mẫu
```bash
python smart_routing_system.py
```

## 🎮 Điều khiển

Khi chạy real-time demo:
- **Q**: Thoát chương trình
- **R**: Reset counters về 0
- **S**: Lưu trạng thái hiện tại

## 🏗️ Kiến trúc hệ thống

### Core Components

1. **WasteType Enum**: Định nghĩa các loại rác
   - ORGANIC, PLASTIC, GLASS, METAL, PAPER, CARDBOARD, BATTERY, CLOTHES, SHOES, GENERAL

2. **GPSCoordinate**: Tọa độ GPS với lat/lng

3. **WasteBin**: Thông tin bãi rác
   - Vị trí, loại rác hỗ trợ, sức chứa, trạng thái

4. **SmartRoutingSystem**: Core routing engine
   - A* pathfinding algorithm
   - Cost calculation với multiple factors
   - Traffic và road condition updates

5. **RealTimeWasteDetector**: YOLOv8 integration
   - Real-time detection từ camera/video
   - Waste counting và threshold monitoring

### Thuật toán A*

**Cost Function:**
```
f(n) = g(n) + h(n)

g(n) = actual_cost_from_start + edge_cost + bin_penalty
h(n) = haversine_distance_to_goal

edge_cost = (distance × w_dist) + (time × w_time) + traffic_penalty + road_penalty
bin_penalty = status_penalty + capacity_penalty
```

**Traffic Penalties:**
- CLEAR: 1.0x
- MODERATE: 1.3x  
- HEAVY: 2.0x
- BLOCKED: ∞

**Bin Penalties:**
- NEAR_FULL: +50
- Capacity ratio: +(ratio × 100)
- FULL: Cost = ∞

## 📊 Dữ liệu mẫu

Hệ thống có sẵn dữ liệu mẫu khu vực TP.HCM:

**Bãi rác:**
- BIN001: Quận 1 (Plastic, Glass, Metal)
- BIN002: Quận 3 (Organic, Paper) - NEAR_FULL
- BIN003: Bình Thạnh (Plastic, Cardboard) - FULL
- BIN004: Quận 4 (Battery, Metal, Clothes)  
- BIN005: Quận 1 (General, Shoes, Clothes)

**Road Network:** Simplified road connections với traffic conditions

## 🔧 Customization

### 1. Thêm bãi rác mới
```python
new_bin = WasteBin(
    id="BIN006",
    location=GPSCoordinate(lat=10.123, lng=106.456),
    supported_types={WasteType.PLASTIC, WasteType.GLASS},
    max_capacity=1000,
    current_capacity=200,
    status=BinStatus.OK
)
system.add_waste_bin(new_bin)
```

### 2. Thêm đoạn đường
```python
segment = RoadSegment(
    start=GPSCoordinate(lat1, lng1),
    end=GPSCoordinate(lat2, lng2),
    distance=2.5,  # km
    travel_time=8.0,  # minutes
    traffic_condition=TrafficCondition.CLEAR,
    road_quality=0.9  # 0-1 scale
)
system.add_road_segment(segment)
```

### 3. Cập nhật traffic real-time
```python
system.update_traffic_condition(
    start_coord, end_coord, 
    TrafficCondition.HEAVY
)
```

### 4. Mapping YOLO classes
```python
class_to_waste_type = {
    'bottle': WasteType.PLASTIC,
    'can': WasteType.METAL,
    'food_waste': WasteType.ORGANIC,
    # Add more mappings...
}
```

## 📈 Output

### Console Output
```
INFO:smart_routing_system:Detected 1 plastic: total = 8
INFO:smart_routing_system:🚨 THRESHOLD REACHED: plastic
INFO:smart_routing_system:📍 Route found to BIN001
INFO:smart_routing_system:📏 Distance: 2.10km
INFO:smart_routing_system:⏱️ ETA: 6.2min
```

### Files Generated
- `route_plastic_1234567890.png`: Map visualization
- `waste_state_1234567890.json`: System state snapshot
- `test_route_plastic.png`: Test route visualization

### Route Visualization
- Blue triangle: Robot position
- Colored squares: Waste bins (green=OK, orange=NEAR_FULL, red=FULL)
- Blue line: Optimal route
- Gray lines: Road network
- Red/orange lines: Traffic congestion

## 🧪 Testing

Run all tests:
```bash
python smart_routing_system.py
```

Test specific components:
```python
# Test routing only
test_routing_system()

# Test detection simulation  
test_real_time_detection()
```

## 🔍 Troubleshooting

### Common Issues

1. **Model not found**
   ```
   ❌ Model file not found: models/trash_safe_best.pt
   ```
   → Đảm bảo có file model trong đúng đường dẫn

2. **Camera không mở được**
   ```
   Cannot open camera 0
   ```
   → Thử camera ID khác (1, 2, ...) hoặc kiểm tra camera permissions

3. **Memory error với video lớn**
   → Giảm resolution hoặc process từng frame thay vì load toàn bộ

4. **No route found**
   → Kiểm tra:
     - Robot position đã set chưa
     - Có bãi rác hỗ trợ loại rác đó không
     - Road network có kết nối đến bãi rác không

### Debug Mode
```bash
python demo_realtime.py --model model.pt --camera 0 --debug
```

## 🚧 Limitations & Future Work

### Current Limitations
- Simplified road network (cần integrate với real map data)
- Static traffic data (cần real-time traffic API)
- Basic heuristic function (có thể optimize thêm)

### Future Enhancements
- Integration với Google Maps API
- Real-time traffic từ traffic APIs
- Multi-vehicle routing optimization
- Machine learning cho traffic prediction
- Mobile app interface
- IoT sensor integration cho bin status

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## 📞 Support

Nếu có vấn đề gì, tạo issue trên GitHub repository hoặc liên hệ qua email.

---

🎯 **Happy waste management!** 🚛♻️
