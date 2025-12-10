"""
Configuration Module
Loads environment variables and application settings
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application Settings"""
    
    # Database
    database_url: str = "sqlite:///./waste_detection.db"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    reload: bool = True
    
    # YOLO Models (2-stage pipeline)
    detection_model_path: str = "yolov8n.pt"
    classification_model_path: str = "models/classification/best.pt"  # Update when model ready
    use_classification: bool = False  # Set to True when classification model is ready
    
    # Detection thresholds
    confidence_threshold: float = 0.25  # Balanced for real-time detection
    iou_threshold: float = 0.6  # Higher IoU = tighter bounding boxes (training used 0.7)
    
    # Legacy (backward compatibility)
    model_path: str = "yolov8n.pt"
    
    # Goong Maps API (for real-world routing)
    goong_api_key: str = ""  # Get from https://account.goong.io/
    goong_maps_enabled: bool = False  # Enable when API key is set
    
    # WebSocket
    ws_heartbeat_interval: int = 30
    
    # CORS
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
