# 🛠️ Development Guide

Quick reference for developers working on the Smart Waste Detection System.

## 🚀 Quick Start (New Developers)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Trash-Detection

# Copy environment template
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt

# Quick system start
python start_system.py
```

### 2. Access Points
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000  
- **API Documentation**: http://localhost:8000/docs

## 📁 Key Directories

### Backend (`waste-system/backend/`)
- `backend.py` - Main FastAPI server
- `detector.py` - YOLO detection engine
- `waste_manager.py` - Statistics and data management
- `routing.py` - A* pathfinding for collection routes

### Frontend (`waste-system/frontend/`)
- `src/components/VideoStream.jsx` - Main detection interface
- `src/components/DetectionSettings.jsx` - Configuration panel
- `src/components/StatsPanel.jsx` - Analytics dashboard

### Core (`src/`)
- `detect.py` - Standalone detection script
- `train.py` - Model training
- `evaluate.py` - Model evaluation

## 🔧 Common Development Tasks

### Start Individual Services
```bash
# Backend only
cd waste-system/backend
python backend.py

# Frontend only  
cd waste-system/frontend
npm run dev
```

### Model Development
```bash
# Train new model
cd src
python train.py --epochs 100 --batch-size 16

# Evaluate model
python evaluate.py --model ../models/final.pt

# Test detection
python detect.py --source 0  # webcam
python detect.py --source image.jpg  # image file
```

### API Testing
```bash
# Test detection endpoint
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_data", "confidence_threshold": 0.5}'

# Get statistics
curl "http://localhost:8000/stats"
```

## 🧪 Testing

### Run Tests
```bash
# All tests
python -m pytest test/

# Specific test file
python -m pytest test/test_detection.py

# With coverage
python -m pytest test/ --cov=src --cov-report=html
```

### Manual Testing Checklist
- [ ] Start camera in frontend
- [ ] Detection boxes appear around objects
- [ ] Statistics update in real-time  
- [ ] Session analytics work
- [ ] API endpoints respond correctly
- [ ] WebSocket connections stable

## 🐛 Debugging

### Common Issues

1. **No detections appearing**
   - Check backend logs for detection messages
   - Verify model file exists: `models/final.pt`
   - Test detection independently: `python src/detect.py --source 0`

2. **Frontend not connecting**
   - Verify backend is running on port 8000
   - Check WebSocket connection in browser console
   - Ensure CORS is configured properly

3. **Model loading errors**
   - Check PyTorch version compatibility
   - Verify model file integrity
   - Try loading with `weights_only=False`

### Debug Mode
```bash
# Backend with debug logging
cd waste-system/backend
DEBUG=true python backend.py

# Frontend with debug info
cd waste-system/frontend  
NODE_ENV=development npm run dev
```

### Logging
- Backend logs: Console output
- Frontend logs: Browser DevTools Console
- Model logs: `logs/` directory

## 📊 Performance Monitoring

### Check System Performance
```bash
# GPU usage (if available)
nvidia-smi

# CPU/Memory usage
htop  # Linux/Mac
Task Manager  # Windows

# Model inference speed
python src/benchmark.py
```

### Optimization Tips
- Use GPU if available for faster inference
- Optimize image preprocessing pipeline
- Batch process multiple detections
- Use model quantization for edge deployment

## 🔄 Code Organization

### Adding New Features

1. **New Detection Class**
   - Update model training data
   - Retrain model with new class
   - Update class mappings in `detector.py`
   - Add UI support in frontend

2. **New API Endpoint**
   - Add route to `backend.py`
   - Update API documentation
   - Add frontend integration
   - Write tests

3. **New Frontend Component**
   - Create component in `src/components/`
   - Add to routing if needed
   - Update parent components
   - Add styling

### Code Style
- Python: Follow PEP 8
- JavaScript: Use ESLint configuration
- Comments: Use clear, descriptive comments
- Naming: Use descriptive variable names

## 🚀 Deployment

### Development
```bash
python start_system.py  # Local development
```

### Production Preparation
```bash
# Build frontend
cd waste-system/frontend
npm run build

# Configure production settings
cp .env.example .env.production
# Edit .env.production with production values

# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend:app
```

## 📝 Contributing

### Before Submitting PR
1. Run tests: `python -m pytest`
2. Check code style: `flake8 src/`
3. Update documentation if needed
4. Test manually with checklist above

### Commit Messages
```
feat: add new detection class for electronics
fix: resolve WebSocket connection timeout
docs: update API documentation
style: format code according to PEP 8
test: add tests for detection accuracy
```

---
**Need Help?** Check the logs, review the documentation, or create an issue!
