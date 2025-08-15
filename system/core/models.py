"""
Core Models - Các model chính của hệ thống
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import sys
import os

# Add parent directory to path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import WasteType, BinStatus, TrafficCondition


@dataclass
class GPSCoordinate:
    """Tọa độ GPS"""
    lat: float
    lng: float
    
    def distance_to(self, other: 'GPSCoordinate') -> float:
        """Tính khoảng cách Haversine đến điểm khác"""
        R = 6371  # Bán kính Trái Đất (km)
        
        lat1, lng1 = math.radians(self.lat), math.radians(self.lng)
        lat2, lng2 = math.radians(other.lat), math.radians(other.lng)
        
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2)
        
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    def __hash__(self):
        return hash((round(self.lat, 6), round(self.lng, 6)))
    
    def __eq__(self, other):
        if not isinstance(other, GPSCoordinate):
            return False
        return (abs(self.lat - other.lat) < 1e-6 and 
                abs(self.lng - other.lng) < 1e-6)


@dataclass
class WasteBin:
    """Thông tin thùng rác"""
    id: str
    location: GPSCoordinate
    status: BinStatus = BinStatus.OK
    supported_types: List[WasteType] = field(default_factory=lambda: [WasteType.GENERAL])
    max_capacity: float = 100.0  # Lít
    current_capacity: float = 0.0
    last_collection: Optional[str] = None
    priority: int = 1  # 1-5, 5 là ưu tiên cao nhất
    
    @property
    def fill_percentage(self) -> float:
        """Phần trăm đầy"""
        return (self.current_capacity / self.max_capacity) * 100
    
    @property
    def needs_collection(self) -> bool:
        """Cần thu gom không"""
        return self.status in [BinStatus.FULL, BinStatus.NEAR_FULL]


@dataclass
class Road:
    """Thông tin đường"""
    start: GPSCoordinate
    end: GPSCoordinate
    traffic_condition: TrafficCondition = TrafficCondition.CLEAR
    road_type: str = "normal"  # highway, main, residential, etc.
    max_speed: int = 50  # km/h
    length: float = field(init=False)
    
    def __post_init__(self):
        self.length = self.start.distance_to(self.end)
    
    @property
    def travel_time(self) -> float:
        """Thời gian di chuyển (phút)"""
        # Tính toán thời gian dựa trên tình trạng giao thông
        speed_factor = {
            TrafficCondition.CLEAR: 1.0,
            TrafficCondition.LIGHT: 0.9, 
            TrafficCondition.MODERATE: 0.7,
            TrafficCondition.HEAVY: 0.5,
            TrafficCondition.BLOCKED: 0.1
        }
        
        effective_speed = self.max_speed * speed_factor[self.traffic_condition]
        return (self.length / effective_speed) * 60  # Chuyển sang phút


@dataclass
class RouteStep:
    """Một bước trong route"""
    from_point: GPSCoordinate
    to_point: GPSCoordinate
    instruction: str
    distance: float
    duration: float  # phút
    
    
@dataclass
class PathfindingResult:
    """Kết quả tìm đường"""
    path: List[GPSCoordinate]
    total_distance: float
    total_time: float  # phút
    steps: List[RouteStep] = field(default_factory=list)
    fuel_estimate: float = 0.0  # lít
    
    @property
    def is_valid(self) -> bool:
        """Route có hợp lệ không"""
        return len(self.path) >= 2 and self.total_distance > 0


@dataclass
class VehicleInfo:
    """Thông tin xe"""
    id: str
    current_location: GPSCoordinate
    fuel_level: float = 100.0  # %
    capacity: float = 1000.0  # kg
    current_load: float = 0.0
    avg_speed: float = 30.0  # km/h
    fuel_consumption: float = 8.0  # L/100km
    
    @property
    def available_capacity(self) -> float:
        """Sức chứa còn lại"""
        return self.capacity - self.current_load
    
    @property
    def load_percentage(self) -> float:
        """Phần trăm tải trọng"""
        return (self.current_load / self.capacity) * 100
