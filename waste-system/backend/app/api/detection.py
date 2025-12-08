"""
Detection API Routes
Endpoints for waste detection operations
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import base64

from app.database import get_db
from app.schemas import (
    DetectionResponse, DetectionSessionResponse,
    DetectionSessionDetail, DetectionSessionCreate
)
from app.crud import (
    create_detection, get_detection, get_detections,
    create_detection_session, get_detection_session,
    get_recent_sessions, end_detection_session,
    update_session_stats, get_detections_by_session
)
from app.services import WasteDetector

router = APIRouter(prefix="/detection", tags=["Detection"])

# Initialize detector (will be overridden by dependency injection)
detector = None


def get_detector():
    """Dependency to get detector instance"""
    global detector
    if detector is None:
        from app.config import get_settings
        settings = get_settings()
        detector = WasteDetector(settings.model_path)
    return detector


@router.post("/detect", summary="Detect waste in image")
async def detect_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    det: WasteDetector = Depends(get_detector)
):
    """
    Detect waste objects in uploaded image
    
    - **file**: Image file (JPEG/PNG)
    
    Returns list of detections
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Convert to frame
        frame = det.bytes_to_frame(image_bytes)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detect
        detections = det.detect(frame)
        
        return {
            "filename": file.filename,
            "detections": detections,
            "count": len(detections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=DetectionSessionResponse, summary="Create detection session")
def create_session(
    session_data: DetectionSessionCreate,
    db: Session = Depends(get_db)
):
    """Create a new detection session"""
    return create_detection_session(db, session_data)


@router.get("/sessions/{session_id}", response_model=DetectionSessionDetail, summary="Get detection session")
def get_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get detection session with all detections"""
    session = get_detection_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions", response_model=List[DetectionSessionResponse], summary="List recent sessions")
def list_sessions(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get list of recent detection sessions"""
    return get_recent_sessions(db, limit)


@router.post("/sessions/{session_id}/end", summary="End detection session")
def end_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """End a detection session"""
    end_detection_session(db, session_id)
    return {"message": "Session ended successfully"}


@router.get("/detections/{detection_id}", response_model=DetectionResponse, summary="Get detection")
def get_single_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    """Get single detection by ID"""
    detection = get_detection(db, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection


@router.get("/detections", response_model=List[DetectionResponse], summary="List detections")
def list_detections(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get list of detections"""
    return get_detections(db, skip, limit)
