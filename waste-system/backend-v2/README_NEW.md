# Smart Waste Detection System - Backend V2

Complete backend system with database persistence, REST API, and WebSocket support for real-time waste detection.

## 🏗️ Architecture

```
backend-v2/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── database.py        # Database setup & session
│   ├── models.py          # SQLAlchemy ORM models
│   ├── schemas.py         # Pydantic validation schemas
│   ├── crud.py            # Database operations
│   ├── api/               # API routes
│   │   ├── detection.py   # Detection endpoints
│   │   ├── bins.py        # Waste bin management
│   │   ├── stats.py       # Statistics endpoints
│   │   └── websocket.py   # WebSocket endpoints
│   └── services/          # Business logic
│       ├── detector.py    # YOLO detection service
│       ├── waste_manager.py  # Statistics manager
│       └── pathfinding.py    # Route calculation
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
└── yolov8n.pt           # YOLO model weights

## 📊 Database Schema

### Models

1. **Detection** - Individual object detections
   - Detection info (label, category, confidence, bbox)
   - Location (GPS coordinates)
   - Timestamp

2. **DetectionSession** - Groups detections by session
   - Session timing (start/end)
   - Statistics (counts by category)
   - Device information

3. **WasteBin** - Waste bin locations
   - Bin info (name, category, capacity)
   - Location (GPS)
   - Status (active/inactive)

4. **WasteStats** - Aggregated statistics
   - Time period (hourly/daily/weekly/monthly)
   - Counts by category
   - Metrics

5. **Route** - Calculated collection routes
   - Start/end points
   - Path coordinates
   - Distance and time estimates

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```env
DATABASE_URL=sqlite:///./waste_detection.db
HOST=0.0.0.0
PORT=8000
MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
```

### 3. Run Application

The database will be automatically created on first run:

```bash
# Development (with auto-reload)
python main.py

# Production
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### REST API

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger)
- `GET /redoc` - Alternative API documentation

### Detection

- `POST /detection/detect` - Detect waste in uploaded image
- `POST /detection/sessions` - Create detection session
- `GET /detection/sessions` - List recent sessions
- `GET /detection/sessions/{id}` - Get session details
- `POST /detection/sessions/{id}/end` - End session
- `GET /detection/detections` - List all detections
- `GET /detection/detections/{id}` - Get single detection

### Waste Bins

- `POST /bins` - Create waste bin
- `GET /bins` - List waste bins
- `GET /bins/{id}` - Get waste bin
- `GET /bins/category/{category}` - Get bins by category
- `PUT /bins/{id}` - Update waste bin
- `DELETE /bins/{id}` - Delete waste bin

### Statistics

- `GET /stats/current` - Current session statistics
- `GET /stats/history/{period_type}` - Historical statistics
- `GET /stats/summary` - Comprehensive summary
- `POST /stats/reset` - Reset current statistics

### WebSocket

- `WS /ws/detect` - Real-time detection stream
- `WS /ws/stats` - Real-time statistics stream

## 🔌 WebSocket Protocol

### /ws/detect

**Client → Server:**
```json
{
  "type": "frame",
  "image": "base64_encoded_image",
  "dimensions": {
    "streamWidth": 640,
    "streamHeight": 480
  }
}
```

**Server → Client:**
```json
{
  "timestamp": 1234567890.123,
  "detections": [
    {
      "label": "bottle",
      "category": "recyclable",
      "confidence": 0.85,
      "bbox": [x, y, width, height]
    }
  ]
}
```

### /ws/stats

**Server → Client** (every 1 second):
```json
{
  "totals": {
    "organic": 5,
    "recyclable": 3,
    "hazardous": 1,
    "other": 2
  },
  "recent": [...]
}
```

## 🗄️ Database Configuration

### SQLite (Default)
```env
DATABASE_URL=sqlite:///./waste_detection.db
```

### PostgreSQL
```env
DATABASE_URL=postgresql://user:password@localhost:5432/waste_detection
```

### MySQL
```env
DATABASE_URL=mysql://user:password@localhost:3306/waste_detection
```

## 🔧 Configuration

All settings can be configured via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./waste_detection.db` | Database connection string |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `True` | Debug mode |
| `RELOAD` | `True` | Auto-reload on code changes |
| `MODEL_PATH` | `yolov8n.pt` | YOLO model file |
| `CONFIDENCE_THRESHOLD` | `0.25` | Detection confidence threshold |
| `IOU_THRESHOLD` | `0.45` | NMS IOU threshold |
| `WS_HEARTBEAT_INTERVAL` | `30` | WebSocket heartbeat interval |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |

## 📝 Development

### Project Structure

- `app/` - Application package
  - `config.py` - Settings and configuration
  - `database.py` - Database connection and session management
  - `models.py` - SQLAlchemy ORM models (entities)
  - `schemas.py` - Pydantic schemas for validation
  - `crud.py` - Database CRUD operations
  - `api/` - API route handlers
  - `services/` - Business logic services

### Adding New Features

1. **Add Model** - Create entity in `app/models.py`
2. **Add Schema** - Create Pydantic schema in `app/schemas.py`
3. **Add CRUD** - Create database operations in `app/crud.py`
4. **Add Route** - Create API endpoint in `app/api/`
5. **Register Router** - Include in `main.py`

### Database Migrations

The database is automatically initialized on first run. Tables are created based on SQLAlchemy models.

For production, consider using Alembic for migrations:

```bash
# Initialize Alembic
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## 🧪 Testing

```bash
# Test detection endpoint
curl -X POST http://localhost:8000/detection/detect \
  -F "file=@test_image.jpg"

# Test health endpoint
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

## 📊 Monitoring

- **Logs**: Check console output for application logs
- **Database**: Query `waste_detection.db` with SQLite browser
- **API Docs**: Access `/docs` for interactive API testing

## 🐛 Troubleshooting

### Database Issues
- Delete `waste_detection.db` and restart to recreate database
- Check `DATABASE_URL` in `.env` file

### Model Loading Issues
- Ensure `yolov8n.pt` is in the root directory
- Check PyTorch version compatibility

### WebSocket Issues
- Ensure uvicorn is installed with `[standard]` extras
- Check CORS settings in `.env`

## 📄 License

MIT License - see LICENSE file for details
```
