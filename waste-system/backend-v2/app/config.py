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
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Legacy (backward compatibility)
    model_path: str = "yolov8n.pt"
    
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
