"""
Pydantic Schemas
Data validation and serialization schemas
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class WasteCategoryEnum(str, Enum):
    """Waste category enumeration"""
    ORGANIC = "organic"
    RECYCLABLE = "recyclable"
    HAZARDOUS = "hazardous"
    OTHER = "other"


# Detection Schemas
class DetectionBase(BaseModel):
    """Base detection schema"""
    label: str
    category: WasteCategoryEnum
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class DetectionCreate(DetectionBase):
    """Schema for creating detection"""
    session_id: int
    detected_at: Optional[datetime] = None
    tracking_data: Optional[Dict[str, Any]] = None


class DetectionResponse(DetectionBase):
    """Schema for detection response"""
    id: int
    session_id: int
    detected_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Detection Session Schemas
class DetectionSessionCreate(BaseModel):
    """Schema for creating detection session"""
    device_id: Optional[str] = None
    user_agent: Optional[str] = None


class DetectionSessionResponse(BaseModel):
    """Schema for detection session response"""
    id: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    total_detections: int
    organic_count: int
    recyclable_count: int
    hazardous_count: int
    other_count: int
    
    model_config = ConfigDict(from_attributes=True)


class DetectionSessionDetail(DetectionSessionResponse):
    """Schema for detailed session with detections"""
    detections: List[DetectionResponse] = []
    
    model_config = ConfigDict(from_attributes=True)


# Waste Bin Schemas
class WasteBinBase(BaseModel):
    """Base waste bin schema"""
    name: str
    category: WasteCategoryEnum
    capacity: float = Field(default=100.0, ge=0.0, le=1000.0)  # Allow up to 1000 liters
    current_fill: float = Field(default=0.0, ge=0.0, le=100.0)
    latitude: float
    longitude: float
    address: Optional[str] = None


class WasteBinCreate(WasteBinBase):
    """Schema for creating waste bin"""
    pass


class WasteBinUpdate(BaseModel):
    """Schema for updating waste bin"""
    name: Optional[str] = None
    capacity: Optional[float] = Field(None, ge=0.0, le=1000.0)  # Allow up to 1000 liters
    is_active: Optional[bool] = None
    last_emptied: Optional[datetime] = None


class WasteBinResponse(WasteBinBase):
    """Schema for waste bin response"""
    id: int
    is_active: bool
    last_emptied: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Statistics Schemas
class StatsResponse(BaseModel):
    """Schema for statistics response"""
    totals: Dict[str, int]
    recent: List[Dict[str, Any]] = []


class WasteStatsCreate(BaseModel):
    """Schema for creating aggregated stats"""
    period_start: datetime
    period_end: datetime
    period_type: str
    organic_count: int = 0
    recyclable_count: int = 0
    hazardous_count: int = 0
    other_count: int = 0
    total_count: int = 0
    unique_sessions: int = 0
    avg_confidence: float = 0.0


class WasteStatsResponse(BaseModel):
    """Schema for waste stats response"""
    id: int
    period_start: datetime
    period_end: datetime
    period_type: str
    organic_count: int
    recyclable_count: int
    hazardous_count: int
    other_count: int
    total_count: int
    unique_sessions: int
    avg_confidence: float
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Route Schemas
class RouteCreate(BaseModel):
    """Schema for creating route"""
    name: str
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    path_coordinates: List[List[float]]
    waypoints: Optional[List[int]] = None
    distance: Optional[float] = None
    estimated_time: Optional[int] = None


class RouteResponse(BaseModel):
    """Schema for route response"""
    id: int
    name: str
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    path_coordinates: List[List[float]]
    waypoints: Optional[List[int]] = None
    distance: Optional[float] = None
    estimated_time: Optional[int] = None
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


# WebSocket Schemas
class FrameRequest(BaseModel):
    """Schema for frame detection request"""
    type: str = "frame"
    image: str  # base64 encoded
    dimensions: Optional[Dict[str, int]] = None


class DetectionResult(BaseModel):
    """Schema for detection result"""
    timestamp: float
    detections: List[Dict[str, Any]]
