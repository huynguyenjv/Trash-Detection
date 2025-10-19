"""
CRUD Operations
Database operations for all models
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.models import (
    Detection, DetectionSession, WasteBin, 
    WasteStats, Route, WasteCategory
)
from app.schemas import (
    DetectionCreate, DetectionSessionCreate,
    WasteBinCreate, WasteBinUpdate,
    WasteStatsCreate, RouteCreate
)


# Detection CRUD
def create_detection(db: Session, detection: DetectionCreate) -> Detection:
    """Create new detection"""
    bbox = detection.bbox
    db_detection = Detection(
        session_id=detection.session_id,
        label=detection.label,
        category=detection.category,
        confidence=detection.confidence,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_width=bbox[2],
        bbox_height=bbox[3],
        latitude=detection.latitude,
        longitude=detection.longitude,
        detected_at=detection.detected_at if hasattr(detection, 'detected_at') and detection.detected_at else datetime.utcnow(),
        tracking_data=detection.tracking_data if hasattr(detection, 'tracking_data') else None
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection


def get_detection(db: Session, detection_id: int) -> Optional[Detection]:
    """Get detection by ID"""
    return db.query(Detection).filter(Detection.id == detection_id).first()


def get_detections(db: Session, skip: int = 0, limit: int = 100) -> List[Detection]:
    """Get list of detections"""
    return db.query(Detection).offset(skip).limit(limit).all()


def get_detections_by_session(db: Session, session_id: int) -> List[Detection]:
    """Get all detections for a session"""
    return db.query(Detection).filter(Detection.session_id == session_id).all()


# Detection Session CRUD
def create_detection_session(db: Session, session: DetectionSessionCreate) -> DetectionSession:
    """Create new detection session"""
    db_session = DetectionSession(
        device_id=session.device_id,
        user_agent=session.user_agent
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def get_detection_session(db: Session, session_id: int) -> Optional[DetectionSession]:
    """Get detection session by ID"""
    return db.query(DetectionSession).filter(DetectionSession.id == session_id).first()


def get_active_session(db: Session) -> Optional[DetectionSession]:
    """Get active (not ended) session"""
    return db.query(DetectionSession).filter(DetectionSession.ended_at == None).first()


def update_session_stats(db: Session, session_id: int, category: WasteCategory):
    """Update session statistics when new detection added"""
    session = get_detection_session(db, session_id)
    if session:
        session.total_detections += 1
        if category == WasteCategory.ORGANIC:
            session.organic_count += 1
        elif category == WasteCategory.RECYCLABLE:
            session.recyclable_count += 1
        elif category == WasteCategory.HAZARDOUS:
            session.hazardous_count += 1
        else:
            session.other_count += 1
        db.commit()


def end_detection_session(db: Session, session_id: int):
    """End a detection session"""
    session = get_detection_session(db, session_id)
    if session:
        session.ended_at = datetime.utcnow()
        db.commit()


def get_recent_sessions(db: Session, limit: int = 10) -> List[DetectionSession]:
    """Get recent detection sessions"""
    return db.query(DetectionSession).order_by(DetectionSession.started_at.desc()).limit(limit).all()


# Waste Bin CRUD
def create_waste_bin(db: Session, bin_data: WasteBinCreate) -> WasteBin:
    """Create new waste bin"""
    db_bin = WasteBin(**bin_data.model_dump())
    db.add(db_bin)
    db.commit()
    db.refresh(db_bin)
    return db_bin


def get_waste_bin(db: Session, bin_id: int) -> Optional[WasteBin]:
    """Get waste bin by ID"""
    return db.query(WasteBin).filter(WasteBin.id == bin_id).first()


def get_waste_bins(db: Session, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[WasteBin]:
    """Get list of waste bins"""
    query = db.query(WasteBin)
    if active_only:
        query = query.filter(WasteBin.is_active == 1)
    return query.offset(skip).limit(limit).all()


def get_bins_by_category(db: Session, category: WasteCategory) -> List[WasteBin]:
    """Get waste bins by category"""
    return db.query(WasteBin).filter(
        WasteBin.category == category,
        WasteBin.is_active == 1
    ).all()


def update_waste_bin(db: Session, bin_id: int, bin_update: WasteBinUpdate) -> Optional[WasteBin]:
    """Update waste bin"""
    db_bin = get_waste_bin(db, bin_id)
    if db_bin:
        update_data = bin_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_bin, key, value)
        db_bin.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_bin)
    return db_bin


def delete_waste_bin(db: Session, bin_id: int) -> bool:
    """Delete waste bin (soft delete)"""
    db_bin = get_waste_bin(db, bin_id)
    if db_bin:
        db_bin.is_active = 0
        db.commit()
        return True
    return False


# Waste Stats CRUD
def create_waste_stats(db: Session, stats: WasteStatsCreate) -> WasteStats:
    """Create aggregated waste statistics"""
    db_stats = WasteStats(**stats.model_dump())
    db.add(db_stats)
    db.commit()
    db.refresh(db_stats)
    return db_stats


def get_stats_by_period(db: Session, period_type: str, limit: int = 10) -> List[WasteStats]:
    """Get statistics by period type"""
    return db.query(WasteStats).filter(
        WasteStats.period_type == period_type
    ).order_by(WasteStats.period_start.desc()).limit(limit).all()


# Route CRUD
def create_route(db: Session, route: RouteCreate) -> Route:
    """Create new route"""
    db_route = Route(**route.model_dump())
    db.add(db_route)
    db.commit()
    db.refresh(db_route)
    return db_route


def get_route(db: Session, route_id: int) -> Optional[Route]:
    """Get route by ID"""
    return db.query(Route).filter(Route.id == route_id).first()


def get_routes(db: Session, skip: int = 0, limit: int = 100) -> List[Route]:
    """Get list of routes"""
    return db.query(Route).order_by(Route.created_at.desc()).offset(skip).limit(limit).all()


def update_route_status(db: Session, route_id: int, status: str) -> Optional[Route]:
    """Update route status"""
    db_route = get_route(db, route_id)
    if db_route:
        db_route.status = status
        if status == "completed":
            db_route.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(db_route)
    return db_route
