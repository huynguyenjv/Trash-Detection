"""
Database Models (Entities)
SQLAlchemy ORM models for waste detection system
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.database import Base


class WasteCategory(str, enum.Enum):
    """Waste category enumeration"""
    ORGANIC = "organic"
    RECYCLABLE = "recyclable"
    HAZARDOUS = "hazardous"
    OTHER = "other"


class Detection(Base):
    """
    Detection Model
    Stores individual object detections from YOLO
    """
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=False)
    
    # Detection info
    label = Column(String(100), nullable=False)  # YOLO class name
    category = Column(SQLEnum(WasteCategory), nullable=False)  # Waste category
    confidence = Column(Float, nullable=False)  # Detection confidence
    
    # Bounding box (x, y, width, height)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)
    
    # Location (optional - from GPS)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Tracking metadata (tracking info, duration, frame count, etc.)
    tracking_data = Column(JSON, nullable=True)
    
    # Timestamp
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    session = relationship("DetectionSession", back_populates="detections")
    
    def __repr__(self):
        return f"<Detection(id={self.id}, label={self.label}, category={self.category})>"


class DetectionSession(Base):
    """
    Detection Session Model
    Groups detections from a single detection session
    """
    __tablename__ = "detection_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Session info
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    
    # Statistics
    total_detections = Column(Integer, default=0)
    organic_count = Column(Integer, default=0)
    recyclable_count = Column(Integer, default=0)
    hazardous_count = Column(Integer, default=0)
    other_count = Column(Integer, default=0)
    
    # Device info (optional)
    device_id = Column(String(100), nullable=True)
    user_agent = Column(String(255), nullable=True)
    
    # Relationship
    detections = relationship("Detection", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DetectionSession(id={self.id}, total={self.total_detections})>"


class WasteBin(Base):
    """
    Waste Bin Model
    Stores waste bin locations and information
    """
    __tablename__ = "waste_bins"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Bin info
    name = Column(String(100), nullable=False)
    category = Column(SQLEnum(WasteCategory), nullable=False)
    capacity = Column(Float, default=100.0)  # Percentage
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(String(255), nullable=True)
    
    # Status
    is_active = Column(Integer, default=1)  # SQLite uses Integer for Boolean
    last_emptied = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<WasteBin(id={self.id}, name={self.name}, category={self.category})>"


class WasteStats(Base):
    """
    Waste Statistics Model
    Stores aggregated statistics by time period
    """
    __tablename__ = "waste_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20), nullable=False)  # hourly, daily, weekly, monthly
    
    # Counts by category
    organic_count = Column(Integer, default=0)
    recyclable_count = Column(Integer, default=0)
    hazardous_count = Column(Integer, default=0)
    other_count = Column(Integer, default=0)
    total_count = Column(Integer, default=0)
    
    # Additional metrics
    unique_sessions = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<WasteStats(period={self.period_type}, total={self.total_count})>"


class Route(Base):
    """
    Route Model
    Stores calculated routes for waste collection
    """
    __tablename__ = "routes"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Route info
    name = Column(String(100), nullable=False)
    
    # Start/End points
    start_lat = Column(Float, nullable=False)
    start_lng = Column(Float, nullable=False)
    end_lat = Column(Float, nullable=False)
    end_lng = Column(Float, nullable=False)
    
    # Route data (stored as JSON)
    path_coordinates = Column(JSON, nullable=False)  # List of [lat, lng] points
    waypoints = Column(JSON, nullable=True)  # Bin IDs in order
    
    # Metrics
    distance = Column(Float, nullable=True)  # in meters
    estimated_time = Column(Integer, nullable=True)  # in seconds
    
    # Status
    status = Column(String(20), default="pending")  # pending, active, completed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Route(id={self.id}, name={self.name}, status={self.status})>"
