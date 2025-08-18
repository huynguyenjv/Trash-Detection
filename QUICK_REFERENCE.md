# 🚀 Quick Reference - Trash Detection Project

## 📁 Cấu trúc thư mục chính

```
Trash-Detection/
├── src/           # 🔴 Core ML code (YOLOv8, training, detection)
├── system/        # 🟡 Smart routing system (A*, GPS, mapping)
├── data/          # 🟢 Datasets (raw + processed)
├── models/        # 🔵 Trained model weights
├── runs/          # 🏃 Training & detection outputs
└── notebooks/     # 📊 Jupyter analysis
```

---

## ⚡ Quick Commands

### 🔄 Data & Training
```bash
# 1. Prepare dataset
cd src/
python data_preprocessing.py

# 2. Train model (memory-safe)
python safe_train.py

# 3. Evaluate model
python evaluate.py --model ../models/trash_safe_best.pt
```

### 🎥 Detection
```bash
cd src/

# Camera real-time
python detect.py --mode webcam --source 0

# Single image
python detect.py --mode image --source image.jpg --output result.jpg

# Video processing
python detect.py --mode video --source video.mp4 --output result.mp4
```

### 🗺️ Smart Routing System
```bash
cd system/

# Interactive map GUI
python interactive_map.py

# Position management
python position_utils.py --interactive

# Real-time detection + routing
python demo_realtime.py --model ../models/trash_safe_best.pt --camera 0 --threshold 10
```

---

## 📂 Key Files

| File | Location | Purpose |
|------|----------|---------|
| `train.py` | `src/` | Train YOLOv8 model |
| `detect.py` | `src/` | Real-time detection |
| `smart_routing_system.py` | `src/` & `system/` | A* pathfinding engine |
| `trash_safe_best.pt` | `models/` | Best trained model |
| `dataset.yaml` | `data/processed/` | Dataset configuration |

---

## 🎯 Workflow Steps

1. **Setup**: `pip install -r requirements.txt`
2. **Data**: `python src/data_preprocessing.py`  
3. **Train**: `python src/safe_train.py`
4. **Test**: `python src/detect.py --mode webcam --source 0`
5. **Route**: `python system/demo_realtime.py --model models/trash_safe_best.pt --camera 0`

---

## 🔧 Configuration

### GPU Memory Settings:
- **< 4GB**: batch_size=4, YOLOv8n
- **4-8GB**: batch_size=8, YOLOv8n  
- **> 8GB**: batch_size=16, YOLOv8m

### Model Files:
- `models/trash_safe_best.pt` - Main model
- `runs/train/trash_safe/weights/best.pt` - Latest training

### Dataset:
- Classes: 10 types (plastic, glass, metal, paper, etc.)
- Format: YOLO (images + .txt labels)
- Split: 80/10/10 (train/val/test)

---

## 🆘 Troubleshooting

**CUDA out of memory**: Reduce batch_size in training scripts
**Camera not found**: Try different camera IDs (0, 1, 2...)
**Import errors**: Check working directory and Python path
**Model not found**: Verify model path `../models/trash_safe_best.pt`

---

## 📚 Documentation

- `README.md` - Main project docs
- `README_routing.md` - Smart routing system docs  
- `PROJECT_STRUCTURE.md` - Detailed folder structure
- `USAGE.py` - Code usage examples
