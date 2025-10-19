# 🎥 Realtime Detection Test Scripts

Scripts để test waste detection và classification realtime từ webcam, không cần web interface.

## 📋 Requirements

```bash
pip install ultralytics opencv-python numpy torch pillow
```

## 🚀 Quick Start

### Option 1: Quick Test (Đơn giản nhất)

```bash
python quick_test.py
```

### Option 2: Full Test (Đầy đủ tính năng)

```bash
python test_realtime_detection.py
```

## 📖 Detailed Usage

### Quick Test Script

Script đơn giản để test nhanh:

```bash
python quick_test.py
```

**Features:**
- ✅ Realtime detection từ webcam
- ✅ Hiển thị bounding boxes với labels
- ✅ Phân loại theo categories (organic, recyclable, hazardous, other)
- ✅ Lưu screenshot (phím 's')
- ✅ Thoát (phím 'q')

### Full Test Script

Script đầy đủ với nhiều tính năng:

```bash
# Basic usage
python test_realtime_detection.py

# With custom model
python test_realtime_detection.py --model models/final.pt

# Custom camera
python test_realtime_detection.py --camera 1

# Custom confidence threshold
python test_realtime_detection.py --confidence 0.6

# All options
python test_realtime_detection.py --model models/final.pt --camera 0 --confidence 0.5
```

**Features:**
- ✅ Realtime detection và classification
- ✅ FPS và detection time tracking
- ✅ Session statistics
- ✅ Bounding boxes với corner markers
- ✅ Category-based colors
- ✅ Information panel
- ✅ Screenshot capture (phím 's')
- ✅ Statistics reset (phím 'r')
- ✅ Confidence threshold adjustment (phím '+' / '-')
- ✅ Session summary khi thoát

## ⌨️ Keyboard Controls

### Quick Test:
- `q` - Quit
- `s` - Save screenshot

### Full Test:
- `q` - Quit
- `s` - Save screenshot
- `r` - Reset statistics
- `+` / `=` - Increase confidence threshold
- `-` / `_` - Decrease confidence threshold

## 🎯 Model Paths

Scripts sẽ tự động tìm model theo thứ tự:

1. Path được chỉ định qua `--model`
2. `models/final.pt`
3. `waste-system/backend/models/final.pt`
4. `./final.pt`
5. Default YOLOv8n (tự động download)

## 📊 Output Examples

### Quick Test Output:
```
🚀 Quick Realtime Detection Test
==================================================
📦 Loading detector...
✅ Found model: models/final.pt
✅ Detector loaded!

🎥 Opening webcam...
✅ Camera opened!

📋 Controls:
  - Press 'q' to quit
  - Press 's' to save screenshot

🎯 Starting detection...

🎯 Detected 2 objects:
  - bottle (recyclable): 87.50%
  - banana (organic): 72.30%
```

### Full Test Output:
```
🚀 Initializing Realtime Waste Detection Test...
📦 Loading custom model: models/final.pt
✅ Detector initialized successfully!

🎥 Starting webcam test (Camera ID: 0)
============================================================
Controls:
  - Press 'q' to quit
  - Press 's' to save screenshot
  - Press 'r' to reset statistics
  - Press '+' to increase confidence threshold
  - Press '-' to decrease confidence threshold
============================================================
📹 Camera: 640x480 @ 30 FPS
🎯 Confidence Threshold: 0.5

🚀 Starting detection... (Press 'q' to quit)

🎯 Frame 30: Detected 2 objects
  - bottle (recyclable): 87.50%
  - banana (organic): 72.30%

============================================================
📊 SESSION SUMMARY
============================================================
Total Frames Processed: 850
Total Objects Detected: 1250

📦 Detection Breakdown:
  🍂 Organic:     320
  ♻️  Recyclable: 580
  ⚠️  Hazardous:  150
  🗑️  Other:      200

⚡ Average FPS: 28.50
⏱️  Average Detection Time: 35.20ms
============================================================
```

## 🎨 Category Colors

- 🍂 **Organic** - Green
- ♻️ **Recyclable** - Orange
- ⚠️ **Hazardous** - Red
- 🗑️ **Other** - Yellow

## 🔧 Troubleshooting

### Camera không mở được:
```bash
# Thử camera ID khác
python test_realtime_detection.py --camera 1
```

### Import error:
```bash
# Cài đặt dependencies
pip install ultralytics opencv-python numpy torch pillow
```

### Model không tìm thấy:
```bash
# Chỉ định path rõ ràng
python test_realtime_detection.py --model path/to/your/model.pt
```

### FPS thấp:
- Giảm resolution camera
- Tăng confidence threshold (ít detections hơn)
- Sử dụng GPU nếu có

## 📝 Notes

1. **Camera ID**: Thường là 0 cho webcam mặc định, 1, 2,... cho camera khác
2. **Confidence Threshold**: 
   - 0.3-0.5: Nhiều detections, có thể có false positives
   - 0.5-0.7: Cân bằng
   - 0.7-0.9: Ít detections, chính xác cao
3. **Screenshots**: Được lưu trong thư mục hiện tại với timestamp
4. **Performance**: FPS phụ thuộc vào CPU/GPU và số lượng objects

## 🚀 Advanced Usage

### Test với video file:

Sửa code trong `quick_test.py`:

```python
# Thay vì:
cap = cv2.VideoCapture(0)

# Dùng:
cap = cv2.VideoCapture('path/to/video.mp4')
```

### Lưu video output:

Thêm vào script:

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Trong loop:
out.write(frame)

# Cleanup:
out.release()
```

## 📧 Support

Nếu có vấn đề, check:
1. Dependencies đã cài đầy đủ
2. Camera đang hoạt động
3. Model path đúng
4. Python version >= 3.8

Happy testing! 🎉
