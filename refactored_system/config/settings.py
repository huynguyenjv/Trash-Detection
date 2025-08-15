"""
Configuration Module - Cấu hình hệ thống
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class SystemConfig:
    """Cấu hình chính của hệ thống"""
    
    # Model paths
    yolo_model_path: str = "yolov8n.pt"
    
    # Map settings
    default_center_lat: float = 10.77
    default_center_lng: float = 106.68
    default_zoom: int = 14
    
    # Route optimization
    max_route_distance: float = 50.0  # km
    max_route_time: float = 240.0     # minutes
    fuel_consumption_rate: float = 8.0  # L/100km
    
    # GUI settings
    figure_size: tuple = (14, 10)
    dpi: int = 100
    
    # Detection settings
    detection_confidence_threshold: float = 0.5
    detection_nms_threshold: float = 0.4
    
    # Update intervals
    map_update_interval: int = 30     # seconds
    detection_interval: int = 10      # frames
    
    # File paths
    data_dir: str = "data"
    cache_dir: str = "cache"
    log_dir: str = "logs"


@dataclass  
class WebConfig:
    """Cấu hình web interface"""
    
    # Map tile providers
    default_tile_layer: str = "OpenStreetMap"
    available_tile_layers: list = None
    
    # Server settings
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    
    # Cache settings
    enable_caching: bool = True
    cache_duration: int = 3600  # seconds
    
    def __post_init__(self):
        if self.available_tile_layers is None:
            self.available_tile_layers = [
                "OpenStreetMap",
                "CartoDB positron", 
                "CartoDB dark_matter",
                "Stamen Terrain"
            ]


@dataclass
class DatabaseConfig:
    """Cấu hình database (future use)"""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "waste_management"
    username: str = "admin"
    password: str = ""
    
    # Connection settings
    max_connections: int = 10
    connection_timeout: int = 30


class ConfigManager:
    """Manager cho các cấu hình hệ thống"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.system_config = SystemConfig()
        self.web_config = WebConfig()
        self.database_config = DatabaseConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, file_path: str):
        """Load configuration from file"""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update system config
            if 'system' in config_data:
                for key, value in config_data['system'].items():
                    if hasattr(self.system_config, key):
                        setattr(self.system_config, key, value)
            
            # Update web config  
            if 'web' in config_data:
                for key, value in config_data['web'].items():
                    if hasattr(self.web_config, key):
                        setattr(self.web_config, key, value)
            
            # Update database config
            if 'database' in config_data:
                for key, value in config_data['database'].items():
                    if hasattr(self.database_config, key):
                        setattr(self.database_config, key, value)
                        
        except Exception as e:
            print(f"Warning: Could not load config from {file_path}: {e}")
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        try:
            import json
            
            config_data = {
                'system': self._dataclass_to_dict(self.system_config),
                'web': self._dataclass_to_dict(self.web_config),
                'database': self._dataclass_to_dict(self.database_config)
            }
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error: Could not save config to {file_path}: {e}")
    
    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        return {
            key: value for key, value in obj.__dict__.items()
            if not key.startswith('_')
        }
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.system_config
    
    def get_web_config(self) -> WebConfig:
        """Get web configuration"""
        return self.web_config
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.database_config
    
    def update_system_config(self, **kwargs):
        """Update system configuration"""
        for key, value in kwargs.items():
            if hasattr(self.system_config, key):
                setattr(self.system_config, key, value)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.system_config.data_dir,
            self.system_config.cache_dir,
            self.system_config.log_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Global config instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return get_config_manager().get_system_config()


def get_web_config() -> WebConfig:
    """Get web configuration"""
    return get_config_manager().get_web_config()


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return get_config_manager().get_database_config()
