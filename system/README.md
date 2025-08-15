# Smart Waste Management System

Hệ thống quản lý rác thải thông minh với kiến trúc modular dễ bảo trì và mở rộng.

## 🏗️ Cấu trúc dự án

```
system/
├── core/                    # Core business logic
│   ├── __init__.py         
│   ├── models.py           # Data models (GPSCoordinate, WasteBin, etc.)
│   ├── enums.py            # Enumerations (WasteType, BinStatus, etc.)  
│   ├── routing_engine.py   # Pathfinding và route optimization
│   └── detection_engine.py # YOLO detection engine
├── interfaces/              # User interfaces  
│   ├── __init__.py
│   ├── web_interface.py    # Web map với Folium
│   ├── desktop_interface.py # Desktop GUI với Matplotlib
│   └── mobile_interface.py # Progressive Web App
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── data_generator.py   # Tạo dữ liệu mẫu
│   ├── gui_helper.py       # Helper functions cho GUI
│   └── file_utils.py       # File operations
├── config/                  # Configuration
│   ├── __init__.py
│   └── settings.py         # System configuration
├── __init__.py
├── main.py                 # Main application entry point
└── README.md
```

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies

```bash
pip install ultralytics opencv-python matplotlib folium numpy
```

### 2. Chạy ứng dụng

```bash
# Web interface (mặc định)
python main.py

# Desktop GUI
python main.py --mode desktop

# Real-time detection
python main.py --mode detection

# Tìm đường tối ưu
python main.py --mode route --bin-id BIN001

# Hiển thị trạng thái hệ thống
python main.py --mode status
```

## 📱 Các interface có sẵn

### 1. Web Interface
- Interactive map với Folium
- Tương tác như Google Maps
- Responsive design
- Multi-layer support

### 2. Desktop Interface  
- GUI với Matplotlib
- Real-time interaction
- Click-to-navigate
- Keyboard shortcuts

### 3. Mobile Interface
- Progressive Web App
- Touch-optimized
- GPS location tracking
- Offline capability

## 🧭 Tính năng chính

### Core Features
- ✅ A* pathfinding algorithm
- ✅ Route optimization (TSP)  
- ✅ YOLO object detection
- ✅ Real-time waste monitoring
- ✅ Multi-interface support

### Navigation Features
- 🗺️ Interactive maps
- 🧭 Turn-by-turn directions
- 📍 GPS tracking
- 🚦 Traffic simulation
- ⛽ Fuel estimation

### Data Management
- 📊 Sample data generation
- 💾 Configuration management
- 📁 File utilities
- 🔧 Modular architecture

## 🔧 Configuration

Tạo file `config.json`:

```json
{
  "system": {
    "yolo_model_path": "yolov8n.pt",
    "default_center_lat": 10.77,
    "default_center_lng": 106.68,
    "fuel_consumption_rate": 8.0
  },
  "web": {
    "host": "localhost",
    "port": 8080,
    "enable_caching": true
  }
}
```

## 📦 Modules

### Core Module
- `models.py`: Data structures (GPSCoordinate, WasteBin, Road, etc.)
- `enums.py`: Enumerations (WasteType, BinStatus, TrafficCondition)
- `routing_engine.py`: A* pathfinding, route optimization
- `detection_engine.py`: YOLO-based waste detection

### Interfaces Module  
- `web_interface.py`: Folium-based web mapping
- `desktop_interface.py`: Matplotlib-based GUI
- `mobile_interface.py`: Progressive Web App generator

### Utils Module
- `data_generator.py`: Generate sample data for testing
- `gui_helper.py`: GUI utility functions
- `file_utils.py`: File operations (JSON, CSV, pickle)

### Config Module
- `settings.py`: System configuration management

## 🎯 Ví dụ sử dụng

### Tạo hệ thống cơ bản

```python
from system import SmartWasteManagementSystem

# Initialize system
system = SmartWasteManagementSystem()

# Setup sample data
system.setup_sample_data()

# Run web interface
system.run_web_interface()
```

### Sử dụng từng module

```python
from system.core import RoutingEngine, GPSCoordinate
from system.utils import DataGenerator

# Create routing engine
routing_engine = RoutingEngine()

# Generate sample data
center = GPSCoordinate(10.77, 106.68)
waste_bins = DataGenerator.create_sample_waste_bins(center, 10)

# Find optimal route
route = routing_engine.optimize_collection_route(center, waste_bins)
```

## 🔄 So sánh với version cũ

### Version cũ (system/ - deprecated)
- ❌ Tất cả code trong vài file lớn
- ❌ Logic trộn lẫn với UI
- ❌ Khó test và maintain
- ❌ Duplicate code nhiều

### Version mới (system/ - current)
- ✅ Modular architecture
- ✅ Separation of concerns  
- ✅ Easy to test và extend
- ✅ Reusable components
- ✅ Clean code structure

## 🚧 Development

### Adding new features
1. Core logic → `core/` module
2. UI components → `interfaces/` module  
3. Utilities → `utils/` module
4. Configuration → `config/` module

### Testing
```bash
# Test individual modules
python -c "from system.core import RoutingEngine; print('Core OK')"
python -c "from system.interfaces import WebMapInterface; print('Interfaces OK')"
python -c "from system.utils import DataGenerator; print('Utils OK')"
```

## 📈 Performance

- Modular loading: Chỉ import modules cần thiết
- Lazy initialization: Components khởi tạo khi cần
- Caching: Built-in caching cho web interface
- Memory efficient: Tránh duplicate data

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Follow modular structure
4. Add documentation
5. Submit pull request

## 📄 License

MIT License - See LICENSE file for details
