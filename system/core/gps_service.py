"""
GPS Location Service for Smart Waste Management System

Provides GPS positioning and location tracking capabilities.
Supports multiple positioning methods: browser geolocation, IP-based, and mock GPS.

Author: Smart Waste Management System
Date: August 2025
"""

import json
import time
import requests
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from .models import GPSCoordinate


@dataclass
class LocationInfo:
    """Information about current location"""
    coordinate: GPSCoordinate
    accuracy: float  # meters
    timestamp: float
    source: str  # 'gps', 'ip', 'network', 'mock'
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


class GPSLocationService:
    """GPS Location Service"""
    
    def __init__(self):
        self.current_location: Optional[LocationInfo] = None
        self.location_history = []
        self.mock_mode = False
        self.mock_location = GPSCoordinate(10.762622, 106.660172)  # Default: Ho Chi Minh City
        
    def enable_mock_mode(self, lat: float = None, lng: float = None):
        """Enable mock GPS mode for testing"""
        self.mock_mode = True
        if lat and lng:
            self.mock_location = GPSCoordinate(lat, lng)
        print(f"🎭 Mock GPS enabled: {self.mock_location.lat}, {self.mock_location.lng}")
    
    def disable_mock_mode(self):
        """Disable mock GPS mode"""
        self.mock_mode = False
        print("📱 Real GPS mode enabled")
    
    def get_current_location(self) -> Optional[LocationInfo]:
        """Get current GPS location"""
        if self.mock_mode:
            return self._get_mock_location()
        
        # Try different location methods in order
        location = (
            self._get_ip_location() or
            self._get_default_location()
        )
        
        if location:
            self.current_location = location
            self.location_history.append(location)
            
        return location
    
    def _get_mock_location(self) -> LocationInfo:
        """Generate mock location for testing"""
        return LocationInfo(
            coordinate=self.mock_location,
            accuracy=10.0,
            timestamp=time.time(),
            source='mock',
            address='Mock Address, Ho Chi Minh City',
            city='Ho Chi Minh City',
            country='Vietnam'
        )
    
    def _get_ip_location(self) -> Optional[LocationInfo]:
        """Get location based on IP address"""
        try:
            # Using ipapi.co for IP geolocation
            response = requests.get('http://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                lat = float(data.get('latitude', 10.762622))
                lng = float(data.get('longitude', 106.660172))
                
                return LocationInfo(
                    coordinate=GPSCoordinate(lat, lng),
                    accuracy=1000.0,  # IP location is less accurate
                    timestamp=time.time(),
                    source='ip',
                    address=f"{data.get('city', 'Unknown')}, {data.get('country_name', 'Unknown')}",
                    city=data.get('city', 'Unknown'),
                    country=data.get('country_name', 'Unknown')
                )
        except Exception as e:
            print(f"⚠️ IP location failed: {e}")
            return None
    
    def _get_default_location(self) -> LocationInfo:
        """Get default location (Ho Chi Minh City)"""
        return LocationInfo(
            coordinate=GPSCoordinate(10.762622, 106.660172),
            accuracy=1000.0,
            timestamp=time.time(),
            source='default',
            address='Ho Chi Minh City, Vietnam',
            city='Ho Chi Minh City',
            country='Vietnam'
        )
    
    def update_location_manual(self, lat: float, lng: float, address: str = None) -> LocationInfo:
        """Manually update location"""
        location = LocationInfo(
            coordinate=GPSCoordinate(lat, lng),
            accuracy=5.0,
            timestamp=time.time(),
            source='manual',
            address=address or f"Manual location: {lat:.6f}, {lng:.6f}"
        )
        
        self.current_location = location
        self.location_history.append(location)
        
        print(f"📍 Location updated manually: {lat:.6f}, {lng:.6f}")
        return location
    
    def get_location_history(self, limit: int = 10) -> list:
        """Get recent location history"""
        return self.location_history[-limit:]
    
    def calculate_distance_to(self, target: GPSCoordinate) -> Optional[float]:
        """Calculate distance to target location"""
        if not self.current_location:
            return None
        
        from .routing_engine import HaversineCalculator
        return HaversineCalculator.distance(self.current_location.coordinate, target)
    
    def is_near_location(self, target: GPSCoordinate, threshold_km: float = 0.1) -> bool:
        """Check if current location is near target"""
        distance = self.calculate_distance_to(target)
        return distance is not None and distance <= threshold_km
    
    def get_location_info_dict(self) -> Dict[str, Any]:
        """Get current location as dictionary"""
        if not self.current_location:
            return {
                "status": "NO_LOCATION",
                "message": "Location not available"
            }
        
        loc = self.current_location
        return {
            "status": "ACTIVE",
            "latitude": loc.coordinate.lat,
            "longitude": loc.coordinate.lng,
            "accuracy": loc.accuracy,
            "source": loc.source,
            "address": loc.address,
            "city": loc.city,
            "country": loc.country,
            "timestamp": loc.timestamp,
            "coordinates_string": f"{loc.coordinate.lat:.6f}, {loc.coordinate.lng:.6f}"
        }
    
    def start_continuous_tracking(self, interval_seconds: int = 30):
        """Start continuous location tracking (for future implementation)"""
        print(f"🔄 Continuous tracking would start (interval: {interval_seconds}s)")
        print("Note: This is a placeholder for real GPS tracking implementation")
    
    def stop_continuous_tracking(self):
        """Stop continuous location tracking"""
        print("⏹️ Continuous tracking stopped")


class WasteClassificationPoints:
    """Manages waste classification and collection points"""
    
    def __init__(self):
        self.classification_points = self._load_default_points()
    
    def _load_default_points(self) -> Dict[str, Dict]:
        """Load default waste classification points in Ho Chi Minh City"""
        return {
            "POINT_001": {
                "name": "Trung tâm tái chế Quận 1",
                "location": GPSCoordinate(10.769444, 106.681944),
                "types": ["plastic", "glass", "metal", "paper"],
                "operating_hours": "06:00-22:00",
                "contact": "028-3822-xxxx",
                "capacity": "high"
            },
            "POINT_002": {
                "name": "Điểm thu gom Quận 3", 
                "location": GPSCoordinate(10.786111, 106.692500),
                "types": ["organic", "plastic", "paper"],
                "operating_hours": "05:00-21:00",
                "contact": "028-3930-xxxx", 
                "capacity": "medium"
            },
            "POINT_003": {
                "name": "Trạm phân loại Quận 7",
                "location": GPSCoordinate(10.732222, 106.719722),
                "types": ["electronic", "hazardous", "metal"],
                "operating_hours": "07:00-19:00",
                "contact": "028-5412-xxxx",
                "capacity": "high"
            },
            "POINT_004": {
                "name": "Điểm thu gom Bình Thạnh",
                "location": GPSCoordinate(10.801944, 106.710833),
                "types": ["plastic", "glass", "paper", "organic"],
                "operating_hours": "06:00-20:00",
                "contact": "028-3899-xxxx",
                "capacity": "medium"
            },
            "POINT_005": {
                "name": "Trung tâm xử lý Thủ Đức",
                "location": GPSCoordinate(10.870833, 106.803056),
                "types": ["all"],  # Accepts all waste types
                "operating_hours": "24/7",
                "contact": "028-3724-xxxx",
                "capacity": "very_high"
            }
        }
    
    def find_nearest_classification_point(self, current_location: GPSCoordinate, 
                                        waste_type: str = None) -> Optional[Dict]:
        """Find nearest classification point for specific waste type"""
        from .routing_engine import HaversineCalculator
        
        suitable_points = []
        
        for point_id, point_data in self.classification_points.items():
            # Check if point accepts this waste type
            if waste_type and waste_type not in point_data["types"] and "all" not in point_data["types"]:
                continue
            
            distance = HaversineCalculator.distance(current_location, point_data["location"])
            
            suitable_points.append({
                "id": point_id,
                "distance": distance,
                **point_data
            })
        
        if not suitable_points:
            return None
        
        # Sort by distance and return nearest
        suitable_points.sort(key=lambda x: x["distance"])
        return suitable_points[0]
    
    def get_all_points_info(self) -> Dict[str, Dict]:
        """Get information about all classification points"""
        return self.classification_points.copy()
    
    def add_custom_point(self, point_id: str, name: str, location: GPSCoordinate,
                        waste_types: list, operating_hours: str = "06:00-20:00") -> None:
        """Add custom classification point"""
        self.classification_points[point_id] = {
            "name": name,
            "location": location,
            "types": waste_types,
            "operating_hours": operating_hours,
            "contact": "Custom point",
            "capacity": "medium"
        }
        print(f"✅ Added classification point: {name}")


def demo_gps_service():
    """Demo GPS Location Service"""
    print("🛰️ GPS LOCATION SERVICE DEMO")
    print("=" * 50)
    
    # Initialize GPS service
    gps = GPSLocationService()
    
    # Test 1: Mock mode
    print("\n1️⃣ Testing Mock GPS Mode:")
    gps.enable_mock_mode(10.762622, 106.660172)
    location = gps.get_current_location()
    if location:
        print(f"📍 Mock Location: {location.coordinate.lat:.6f}, {location.coordinate.lng:.6f}")
        print(f"🎯 Source: {location.source}, Accuracy: {location.accuracy}m")
    
    # Test 2: Real location
    print("\n2️⃣ Testing Real Location (IP-based):")
    gps.disable_mock_mode()
    real_location = gps.get_current_location()
    if real_location:
        print(f"📍 Real Location: {real_location.coordinate.lat:.6f}, {real_location.coordinate.lng:.6f}")
        print(f"🎯 Source: {real_location.source}")
        print(f"🏙️ Address: {real_location.address}")
    
    # Test 3: Manual location
    print("\n3️⃣ Testing Manual Location Update:")
    manual_loc = gps.update_location_manual(10.770000, 106.670000, "Manual Test Location")
    
    # Test 4: Classification Points
    print("\n4️⃣ Testing Waste Classification Points:")
    classification = WasteClassificationPoints()
    
    if gps.current_location:
        nearest = classification.find_nearest_classification_point(
            gps.current_location.coordinate, "plastic"
        )
        if nearest:
            print(f"🏢 Nearest point for plastic waste: {nearest['name']}")
            print(f"📍 Location: {nearest['location'].lat:.6f}, {nearest['location'].lng:.6f}")
            print(f"📏 Distance: {nearest['distance']:.2f} km")
            print(f"⏰ Hours: {nearest['operating_hours']}")
    
    # Test 5: Location info
    print("\n5️⃣ Current Location Info:")
    info = gps.get_location_info_dict()
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo_gps_service()
