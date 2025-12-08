# 🎯 Object Tracking System

## Overview

Backend sử dụng **Object Tracking** để đảm bảo **1 vật thể = 1 record trong database**.

### Vấn đề đã giải quyết

**Trước khi có tracking:**
```
30 FPS video stream → 1 bottle xuất hiện 10s → 300 duplicate records! 😱
Analytics SAI: "Hôm nay phát hiện 300 bottles" (thực tế chỉ 1)
```

**Sau khi có tracking:**
```
30 FPS video stream → 1 bottle xuất hiện 10s → 1 UNIQUE record ✅
Analytics ĐÚNG: "Hôm nay phát hiện 1 bottle"
```

---

## Architecture

```
┌────────────────────────────────────────────┐
│  REALTIME DETECTION (30 FPS)               │
│  - Receive video frames via WebSocket      │
│  - YOLO detects objects every frame        │
│  - Send bounding boxes to frontend         │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│  OBJECT TRACKER (In-memory)                │
│  - Match detections across frames (IOU)    │
│  - Assign unique ID to each object         │
│  - Track: first_seen, last_seen, frames    │
│  - NOT saved to DB yet                     │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│  OBJECT LIFECYCLE                          │
│  - Object appears → Track in memory        │
│  - Object exists → Update tracker          │
│  - Object disappears (3s) → Save to DB ✅  │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│  DATABASE (Unique objects only)            │
│  Each record = 1 complete object lifecycle │
│  + Metadata: duration, frames, confidence  │
└────────────────────────────────────────────┘
```

---

## How It Works

### 1. Object Detection (Every Frame)

```python
# YOLO detects objects in frame
detections = [
    {'label': 'bottle', 'bbox': [100, 200, 50, 80], 'confidence': 0.85, ...},
    {'label': 'cup', 'bbox': [300, 150, 40, 70], 'confidence': 0.92, ...}
]
```

### 2. Object Matching (IOU Algorithm)

```python
# For each detection, find matching tracked object
for detection in detections:
    # Calculate IOU (Intersection over Union) with existing objects
    for tracked_obj in active_objects:
        iou = calculate_iou(detection.bbox, tracked_obj.bbox)
        
        if iou > 0.4 and detection.label == tracked_obj.label:
            # MATCH! Update existing object
            tracked_obj.update(detection)
        else:
            # NEW object → Create tracker
            create_new_tracked_object(detection)
```

**IOU (Intersection over Union):**
```
  bbox1: [100, 200, 50, 80]
  bbox2: [105, 200, 50, 80]  (slightly moved)
  
  IOU = 0.85 (85% overlap) → SAME OBJECT ✅
```

### 3. Object Lifecycle

```python
Timeline:
T=0.0s:  bottle detected → Create tracker "bottle_1"
T=0.03s: Same bottle (IOU=0.9) → Update "bottle_1"
T=0.06s: Same bottle (IOU=0.88) → Update "bottle_1"
...
T=10.0s: Bottle not detected anymore
T=13.0s: 3 seconds without detection → Save "bottle_1" to DB ✅

Database Record:
{
    "label": "bottle",
    "first_seen": 0.0,
    "last_seen": 10.0,
    "duration": 10.0,
    "frame_count": 300,
    "avg_confidence": 0.87
}
```

### 4. Force Save on Disconnect

```python
# When WebSocket disconnects (user closes browser)
remaining_objects = tracker.force_save_all()

# Save all objects that are still being tracked
for obj in remaining_objects:
    save_to_database(obj)  # Don't lose data!
```

---

## Configuration

### Tracker Parameters

In `app/api/websocket.py`:

```python
tracker = ObjectTracker(
    disappear_threshold=3.0,  # Object disappears after 3s without detection
    iou_threshold=0.4         # 40% bbox overlap = same object
)
```

**Tuning Guidelines:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `disappear_threshold` | **1.0s** | Fast save, may miss if object temporarily hidden |
| `disappear_threshold` | **3.0s** | Balanced (recommended) |
| `disappear_threshold` | **5.0s** | Slow save, very robust |
| `iou_threshold` | **0.3** | Loose matching, may merge different objects |
| `iou_threshold` | **0.4** | Balanced (recommended) |
| `iou_threshold` | **0.6** | Strict matching, may duplicate same object |

---

## Database Schema

### Detection Table

```sql
CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL,
    
    -- Object info
    label VARCHAR(100) NOT NULL,
    category VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    
    -- Bounding box
    bbox_x FLOAT NOT NULL,
    bbox_y FLOAT NOT NULL,
    bbox_width FLOAT NOT NULL,
    bbox_height FLOAT NOT NULL,
    
    -- Location (optional)
    latitude FLOAT,
    longitude FLOAT,
    
    -- Tracking metadata (NEW!)
    metadata JSONB,  -- {duration, frame_count, avg_confidence, ...}
    
    -- Timestamp
    detected_at TIMESTAMP NOT NULL
);
```

### Metadata Structure

```json
{
    "duration_seconds": 10.5,
    "frame_count": 315,
    "average_confidence": 0.87,
    "first_seen": 1729234567.123,
    "last_seen": 1729234577.623,
    "force_saved": false,
    "saved_on_error": false
}
```

---

## API Response

### WebSocket Response Format

```json
{
    "timestamp": 1729234567.123,
    
    "detections": [
        {
            "label": "bottle",
            "category": "recyclable",
            "bbox": [100, 200, 50, 80],
            "confidence": 0.85
        }
    ],
    
    "tracking": {
        "active_objects": 3,
        "objects_by_label": {
            "bottle": 2,
            "cup": 1
        },
        "saved_this_frame": 1
    }
}
```

**Fields:**
- `detections`: Current frame detections (for display)
- `tracking.active_objects`: How many unique objects being tracked
- `tracking.saved_this_frame`: How many objects just saved to DB (disappeared)

---

## Analytics Queries

### Count Unique Objects

```sql
-- Total unique objects detected today
SELECT COUNT(*) as total_objects
FROM detections
WHERE DATE(detected_at) = CURRENT_DATE;

-- By category
SELECT category, COUNT(*) as count
FROM detections
WHERE DATE(detected_at) = CURRENT_DATE
GROUP BY category;
```

### Average Object Duration

```sql
-- How long objects stay in view
SELECT 
    label,
    AVG((metadata->>'duration_seconds')::FLOAT) as avg_duration,
    AVG((metadata->>'frame_count')::INTEGER) as avg_frames
FROM detections
WHERE metadata IS NOT NULL
GROUP BY label;
```

### Confidence Distribution

```sql
-- Object detection quality
SELECT 
    label,
    AVG((metadata->>'average_confidence')::FLOAT) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence
FROM detections
WHERE metadata IS NOT NULL
GROUP BY label;
```

---

## Testing

### Verify Tracking Works

```bash
cd waste-system/backend-v2
python check_database.py
```

**Expected Output:**
```
🎯 DETECTIONS:
Total detections: 5

Last 5 detection(s):
  ID #5: bottle (recyclable) - Confidence: 0.87
    Session: 7, BBox: [100.5, 200.3, 50.2, 80.1]
    Time: 2025-10-18 10:45:23
    📊 Tracking: Duration=10.5s, Frames=315, AvgConf=0.87

  ID #4: cup (recyclable) - Confidence: 0.92
    Session: 7, BBox: [300.1, 150.7, 40.3, 70.2]
    Time: 2025-10-18 10:45:15
    📊 Tracking: Duration=5.2s, Frames=156, AvgConf=0.91
```

---

## Frontend Integration

### Update WebSocket Handler

```javascript
// Frontend: waste-system/frontend/src/...

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Display detections (current frame)
    displayBoundingBoxes(data.detections);
    
    // Show tracking stats
    updateStats({
        activeObjects: data.tracking.active_objects,
        objectsByLabel: data.tracking.objects_by_label,
        savedThisFrame: data.tracking.saved_this_frame
    });
};
```

---

## Performance

### Memory Usage

```
Assumption: 30 FPS, average 5 objects in frame

In-memory tracking:
- 5 objects × ~500 bytes/object = 2.5 KB
- Negligible memory footprint ✅

Database writes:
- Before: 5 objects × 30 FPS = 150 writes/second 😱
- After: ~0.5 writes/second (objects disappear) ✅
- 99.7% reduction!
```

### Latency

```
Detection latency: <50ms
- YOLO inference: ~30ms
- Tracker update: <5ms
- DB write (async): ~10ms

No noticeable delay! ✅
```

---

## Troubleshooting

### Issue: Objects duplicated in database

**Symptoms:**
- Same object saved multiple times
- IOU threshold too low

**Solution:**
```python
# Increase IOU threshold
tracker = ObjectTracker(
    iou_threshold=0.5  # Was 0.4, now stricter
)
```

### Issue: Objects not saved

**Symptoms:**
- Active objects count increases but no DB saves
- disappear_threshold too high

**Solution:**
```python
# Decrease disappear threshold
tracker = ObjectTracker(
    disappear_threshold=2.0  # Was 3.0, now faster
)
```

### Issue: Objects saved too quickly

**Symptoms:**
- Object temporarily hidden → saved → reappears as new object
- disappear_threshold too low

**Solution:**
```python
# Increase disappear threshold
tracker = ObjectTracker(
    disappear_threshold=5.0  # More forgiving
)
```

---

## Future Enhancements

### 1. Deep SORT Integration

Replace IOU-based matching with deep learning features:
```python
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort()
tracks = tracker.update_tracks(detections, frame=frame)
```

**Benefits:**
- More robust tracking (appearance features)
- Handle occlusions better
- Track objects across camera views

### 2. Trajectory Tracking

Save object movement path:
```python
metadata = {
    "trajectory": [
        {"t": 0.0, "bbox": [100, 200, 50, 80]},
        {"t": 1.0, "bbox": [110, 200, 50, 80]},
        {"t": 2.0, "bbox": [120, 200, 50, 80]}
    ]
}
```

### 3. Event Detection

Detect specific events:
```python
# Object thrown into bin
if object.trajectory.speed > threshold:
    event = "thrown"
    
# Object picked up
if object.disappeared_near_hand:
    event = "picked_up"
```

---

## References

- **IOU Algorithm**: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
- **Object Tracking**: https://learnopencv.com/object-tracking-using-opencv-cpp-python/
- **DeepSORT**: https://github.com/nwojke/deep_sort

---

**🎯 Summary:**
- ✅ 1 physical object = 1 database record
- ✅ Accurate analytics (no duplicates)
- ✅ Rich metadata (duration, frames, confidence)
- ✅ 99.7% reduction in database writes
- ✅ Production-ready!
