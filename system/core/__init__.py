"""
Smart Waste Management System - Core Module
Chứa các models, enums và classes chính
"""

from .models import *
from .enums import *
from .routing_engine import *
from .detection_engine import *
from .gps_service import *
from .smart_detection_system import *

__all__ = [
    # Models
    'GPSCoordinate', 'WasteBin', 'Road', 'PathfindingResult',
    
    # Enums  
    'WasteType', 'BinStatus', 'TrafficCondition',
    
    # Engines
    'RoutingEngine', 'DetectionEngine',
    
    # GPS & Location Services
    'GPSLocationService', 'LocationInfo', 'WasteClassificationPoints',
    
    # Smart Detection System
    'SmartWasteDetectionSystem', 'DetectionResult', 'NavigationInfo'
]
