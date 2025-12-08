"""
WebSocket API Routes
Real-time detection and statistics via WebSocket
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import time
import asyncio
import base64
import traceback

from app.database import get_db
from app.services import WasteManager, WastePipeline
from app.services.object_tracker import ObjectTracker
from app.crud import (
    create_detection_session, update_session_stats, 
    end_detection_session, get_active_session, create_detection
)
from app.schemas import DetectionCreate
from app.config import get_settings
from datetime import datetime

router = APIRouter(tags=["WebSocket"])

# Initialize services
settings = get_settings()
# Initialize 2-stage pipeline (detection + optional classification)
pipeline = WastePipeline(
    detection_model_path=getattr(settings, 'detection_model_path', settings.model_path),
    classification_model_path=getattr(settings, 'classification_model_path', None),
    use_classification=getattr(settings, 'use_classification', False)
)

waste_manager = WasteManager()

# Object trackers per session (in-memory tracking)
session_trackers = {}  # {session_id: ObjectTracker}


@router.websocket("/ws/detect")
async def websocket_detection(websocket: WebSocket, db: Session = Depends(get_db)):
    """
    WebSocket endpoint for realtime detection with object tracking
    
    Strategy:
    - Detect objects in each frame
    - Track unique objects over time (in-memory)
    - Only save to DB when object disappears (lifecycle complete)
    - This ensures: 1 physical object = 1 database record
    
    Client sends: JSON with base64 image
    Server sends: JSON detection results + tracking info
    """
    await websocket.accept()
    print("🔌 WebSocket connected")
    
    # Create or get active session
    session = get_active_session(db)
    if not session:
        from app.schemas import DetectionSessionCreate
        session = create_detection_session(db, DetectionSessionCreate())
    
    # Initialize object tracker for this session
    if session.id not in session_trackers:
        session_trackers[session.id] = ObjectTracker(
            disappear_threshold=1.0,  # Object disappears after 1 second without detection (for testing)
            iou_threshold=0.4  # 40% bbox overlap = same object
        )
        print(f"🎯 Tracker initialized for session #{session.id}")
    
    tracker = session_trackers[session.id]
    
    try:
        while True:
            # Receive data (JSON)
            data = await websocket.receive_json()
            
            # Extract base64 image
            image_base64 = data.get('image', '')
            if not image_base64:
                continue
            
            # Convert base64 to bytes
            image_bytes = base64.b64decode(image_base64)
            
            # Convert to frame
            frame = pipeline.bytes_to_frame(image_bytes)
            
            if frame is None:
                print("⚠️ Failed to decode frame")
                continue
            
            # Detect objects in current frame (pipeline may also classify)
            timestamp = time.time()
            detections = pipeline.process_frame(
                frame,
                conf_threshold=settings.confidence_threshold,
                iou_threshold=settings.iou_threshold
            )
            
            # Update object tracker (track unique objects)
            active_objects, completed_objects = tracker.update(detections)
            
            # Update in-memory stats (for real-time display)
            waste_manager.update(detections)
            
            # Save COMPLETED objects to database (objects that just disappeared)
            for obj_record in completed_objects:
                try:
                    detection_data = DetectionCreate(
                        session_id=session.id,
                        label=obj_record['label'],
                        category=obj_record['category'],
                        confidence=obj_record['confidence'],
                        bbox=obj_record['bbox'],
                        latitude=None,
                        longitude=None,
                        detected_at=datetime.fromtimestamp(obj_record['first_seen']),
                        tracking_data={
                            'duration_seconds': obj_record['duration'],
                            'frame_count': obj_record['frame_count'],
                            'average_confidence': obj_record['avg_confidence'],
                            'first_seen': obj_record['first_seen'],
                            'last_seen': obj_record['last_seen']
                        }
                    )
                    
                    # Save unique object to DB
                    db_detection = create_detection(db, detection_data)
                    
                    # Update session stats (only for unique objects)
                    update_session_stats(db, session.id, obj_record['category'])
                    
                    print(f"💾 Saved unique object: {obj_record['label']} "
                          f"(duration={obj_record['duration']:.1f}s, "
                          f"frames={obj_record['frame_count']}, "
                          f"confidence={obj_record['confidence']:.2f})")
                    
                except Exception as e:
                    print(f"⚠️ Failed to save detection: {e}")
                    import traceback
                    print(traceback.format_exc())
            
            # Get tracker stats
            tracker_stats = tracker.get_stats()
            
            # Send result to frontend
            result = {
                'timestamp': timestamp,
                'detections': detections,  # Current frame detections (for display)
                'tracking': {
                    'active_objects': tracker_stats['active_objects'],
                    'objects_by_label': tracker_stats['objects_by_label'],
                    'saved_this_frame': len(completed_objects)
                }
            }
            
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
        
        # Force save all remaining tracked objects before ending session
        if session.id in session_trackers:
            tracker = session_trackers[session.id]
            remaining_objects = tracker.force_save_all()
            
            print(f"💾 Force saving {len(remaining_objects)} remaining objects...")
            
            for obj_record in remaining_objects:
                try:
                    detection_data = DetectionCreate(
                        session_id=session.id,
                        label=obj_record['label'],
                        category=obj_record['category'],
                        confidence=obj_record['confidence'],
                        bbox=obj_record['bbox'],
                        latitude=None,
                        longitude=None,
                        detected_at=datetime.fromtimestamp(obj_record['first_seen']),
                        tracking_data={
                            'duration_seconds': obj_record['duration'],
                            'frame_count': obj_record['frame_count'],
                            'average_confidence': obj_record['avg_confidence'],
                            'first_seen': obj_record['first_seen'],
                            'last_seen': obj_record['last_seen'],
                            'force_saved': True
                        }
                    )
                    
                    create_detection(db, detection_data)
                    update_session_stats(db, session.id, obj_record['category'])
                    
                    print(f"✅ Force saved: {obj_record['label']} "
                          f"(duration={obj_record['duration']:.1f}s)")
                    
                except Exception as e:
                    print(f"⚠️ Failed to force save: {e}")
            
            # Remove tracker from memory
            del session_trackers[session.id]
            print(f"🗑️ Tracker removed for session #{session.id}")
        
        # End session
        end_detection_session(db, session.id)
        
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        print(traceback.format_exc())
        
        # Force save on error too
        if session.id in session_trackers:
            tracker = session_trackers[session.id]
            remaining = tracker.force_save_all()
            
            for obj_record in remaining:
                try:
                    detection_data = DetectionCreate(
                        session_id=session.id,
                        label=obj_record['label'],
                        category=obj_record['category'],
                        confidence=obj_record['confidence'],
                        bbox=obj_record['bbox'],
                        latitude=None,
                        longitude=None,
                        detected_at=datetime.fromtimestamp(obj_record['first_seen']),
                        tracking_data={
                            'duration_seconds': obj_record['duration'],
                            'frame_count': obj_record['frame_count'],
                            'average_confidence': obj_record['avg_confidence'],
                            'force_saved': True,
                            'saved_on_error': True
                        }
                    )
                    create_detection(db, detection_data)
                    update_session_stats(db, session.id, obj_record['category'])
                except:
                    pass
            
            del session_trackers[session.id]
        
        end_detection_session(db, session.id)


@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """
    WebSocket endpoint for realtime statistics
    
    Server sends: JSON stats every second
    """
    await websocket.accept()
    print("📊 Stats WebSocket connected")
    
    try:
        while True:
            # Get current stats
            stats = waste_manager.get_stats()
            
            # Send stats
            await websocket.send_json(stats)
            
            # Wait 1 second before next update
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print("📊 Stats WebSocket disconnected")
    except Exception as e:
        print(f"❌ Stats WebSocket error: {e}")
