"""
Smart Waste Detection System - Integrated Detection with GPS Navigation

This module combines waste detection with GPS location services and navigation
to waste classification points.

Author: Smart Waste Management System
Date: August 2025
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from ultralytics import YOLO
import logging

from .models import GPSCoordinate
from .enums import WasteType
from .detection_engine import DetectionEngine
from .gps_service import GPSLocationService, WasteClassificationPoints
from .routing_engine import RoutingEngine

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result with location and navigation info"""
    waste_type: str
    confidence: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    location: Optional[GPSCoordinate] = None
    timestamp: float = None
    classification_point: Optional[Dict] = None
    navigation_route: Optional[Dict] = None


@dataclass
class NavigationInfo:
    """Navigation information to waste classification point"""
    destination: Dict  # Classification point info
    distance: float  # km
    estimated_time: float  # minutes
    route_points: List[GPSCoordinate]
    instructions: List[str]


class SmartWasteDetectionSystem:
    """
    Smart Waste Detection System with GPS and Navigation Integration
    
    Features:
    - Real-time waste detection using YOLO
    - Automatic GPS location tracking
    - Navigation to nearest waste classification points
    - Support for different waste types routing
    """
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize the smart detection system"""
        self.detection_engine = DetectionEngine(model_path)
        self.gps_service = GPSLocationService()
        self.classification_points = WasteClassificationPoints()
        self.routing_engine = RoutingEngine()
        
        # Detection history
        self.detection_history: List[DetectionResult] = []
        
        # System status
        self.is_running = False
        self.auto_navigation = True
        
        print("🤖 Smart Waste Detection System initialized!")
        
    def start_system(self, enable_gps: bool = True, mock_gps: bool = False):
        """Start the detection system"""
        self.is_running = True
        
        if enable_gps:
            if mock_gps:
                self.gps_service.enable_mock_mode(10.762622, 106.660172)
            
            # Get initial location
            location = self.gps_service.get_current_location()
            if location:
                print(f"📍 System started at: {location.coordinate.lat:.6f}, {location.coordinate.lng:.6f}")
                print(f"🎯 Location source: {location.source}")
            else:
                print("⚠️ Could not get GPS location")
        
        print("✅ Smart Waste Detection System is running!")
    
    def stop_system(self):
        """Stop the detection system"""
        self.is_running = False
        print("⏹️ Smart Waste Detection System stopped!")
    
    def detect_and_navigate(self, image: np.ndarray, auto_navigate: bool = True) -> List[DetectionResult]:
        """
        Detect waste in image and provide navigation if needed
        
        Args:
            image: Input image for detection
            auto_navigate: Whether to automatically find navigation to classification points
            
        Returns:
            List of detection results with navigation info
        """
        if not self.is_running:
            print("⚠️ System is not running. Call start_system() first.")
            return []
        
        # 1. Perform waste detection
        raw_detections = self.detection_engine.detect_waste(image)
        
        # 2. Get current location
        current_location = self.gps_service.get_current_location()
        
        # 3. Process detections and add navigation info
        results = []
        
        for detection in raw_detections:
            waste_type = detection.get('waste_type', 'unknown')
            
            # Create detection result
            result = DetectionResult(
                waste_type=waste_type,
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                location=current_location.coordinate if current_location else None,
                timestamp=time.time()
            )
            
            # Add navigation info if requested and location available
            if auto_navigate and current_location:
                nav_info = self._get_navigation_info(current_location.coordinate, waste_type)
                if nav_info:
                    result.classification_point = nav_info.destination
                    result.navigation_route = {
                        'distance': nav_info.distance,
                        'estimated_time': nav_info.estimated_time,
                        'instructions': nav_info.instructions[:3]  # First 3 instructions
                    }
            
            results.append(result)
        
        # Store in history
        self.detection_history.extend(results)
        
        # Print results
        self._print_detection_results(results)
        
        return results
    
    def _get_navigation_info(self, current_location: GPSCoordinate, waste_type: str) -> Optional[NavigationInfo]:
        """Get navigation information for waste type"""
        try:
            # Find nearest classification point
            nearest_point = self.classification_points.find_nearest_classification_point(
                current_location, waste_type
            )
            
            if not nearest_point:
                return None
            
            # Calculate route
            destination = nearest_point['location']
            distance = nearest_point['distance']
            
            # Generate simple navigation instructions
            instructions = self._generate_navigation_instructions(current_location, destination, distance)
            
            # Estimate time (assuming 30 km/h average speed)
            estimated_time = (distance / 30.0) * 60  # minutes
            
            return NavigationInfo(
                destination=nearest_point,
                distance=distance,
                estimated_time=estimated_time,
                route_points=[current_location, destination],
                instructions=instructions
            )
            
        except Exception as e:
            logger.error(f"Navigation calculation failed: {e}")
            return None
    
    def _generate_navigation_instructions(self, start: GPSCoordinate, end: GPSCoordinate, distance: float) -> List[str]:
        """Generate simple navigation instructions"""
        instructions = [
            f"🚀 Bắt đầu từ vị trí hiện tại ({start.lat:.4f}, {start.lng:.4f})",
            f"🧭 Di chuyển về phía {self._get_direction(start, end)}",
            f"📍 Đích đến cách {distance:.2f} km",
            f"🏁 Đến điểm phân loại rác ({end.lat:.4f}, {end.lng:.4f})",
            "✅ Hoàn thành hành trình!"
        ]
        return instructions
    
    def _get_direction(self, start: GPSCoordinate, end: GPSCoordinate) -> str:
        """Get cardinal direction from start to end"""
        lat_diff = end.lat - start.lat
        lng_diff = end.lng - start.lng
        
        if abs(lat_diff) > abs(lng_diff):
            return "Bắc" if lat_diff > 0 else "Nam"
        else:
            return "Đông" if lng_diff > 0 else "Tây"
    
    def _print_detection_results(self, results: List[DetectionResult]):
        """Print detection results to console"""
        if not results:
            print("🔍 No waste detected in image")
            return
        
        print(f"\n🗑️ DETECTED {len(results)} WASTE ITEM(S):")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Waste Type: {result.waste_type}")
            print(f"   Confidence: {result.confidence:.2%}")
            
            if result.location:
                print(f"   Location: {result.location.lat:.6f}, {result.location.lng:.6f}")
            
            if result.classification_point:
                point = result.classification_point
                print(f"   📍 Nearest collection point: {point['name']}")
                print(f"   📏 Distance: {point['distance']:.2f} km")
                print(f"   ⏰ Operating hours: {point['operating_hours']}")
                
            if result.navigation_route:
                route = result.navigation_route
                print(f"   🚗 Travel time: ~{route['estimated_time']:.0f} minutes")
                print(f"   📋 Instructions: {route['instructions'][0]}")
    
    def get_detection_history(self, limit: int = 10) -> List[DetectionResult]:
        """Get recent detection history"""
        return self.detection_history[-limit:]
    
    def get_waste_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        if not self.detection_history:
            return {"total_detections": 0}
        
        waste_counts = {}
        total_confidence = 0
        
        for result in self.detection_history:
            waste_type = result.waste_type
            waste_counts[waste_type] = waste_counts.get(waste_type, 0) + 1
            total_confidence += result.confidence
        
        return {
            "total_detections": len(self.detection_history),
            "waste_type_distribution": waste_counts,
            "average_confidence": total_confidence / len(self.detection_history),
            "unique_waste_types": len(waste_counts),
            "latest_detection": self.detection_history[-1].timestamp if self.detection_history else None
        }
    
    def find_classification_point_for_waste_type(self, waste_type: str) -> Optional[Dict]:
        """Find best classification point for specific waste type"""
        current_location = self.gps_service.get_current_location()
        if not current_location:
            print("⚠️ Current location not available")
            return None
        
        return self.classification_points.find_nearest_classification_point(
            current_location.coordinate, waste_type
        )
    
    def update_location_manual(self, lat: float, lng: float):
        """Manually update GPS location"""
        self.gps_service.update_location_manual(lat, lng)
    
    def demo_detection_with_navigation(self):
        """Demo the complete detection and navigation system"""
        print("🚀 SMART WASTE DETECTION & NAVIGATION DEMO")
        print("=" * 60)
        
        # Start system
        self.start_system(enable_gps=True, mock_gps=True)
        
        # Create demo image (black image for testing)
        demo_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some simulated waste objects
        cv2.rectangle(demo_image, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(demo_image, (300, 200), (400, 300), (255, 0, 0), -1)  # Blue rectangle
        
        print("\n1️⃣ Performing waste detection...")
        results = self.detect_and_navigate(demo_image)
        
        print("\n2️⃣ Detection Statistics:")
        stats = self.get_waste_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n3️⃣ Available Classification Points:")
        points = self.classification_points.get_all_points_info()
        for point_id, point_info in points.items():
            print(f"   {point_id}: {point_info['name']}")
            print(f"      Location: {point_info['location'].lat:.4f}, {point_info['location'].lng:.4f}")
            print(f"      Types: {', '.join(point_info['types'])}")
            print()
        
        # Stop system
        self.stop_system()


def demo_smart_detection_system():
    """Demo function for the smart detection system"""
    system = SmartWasteDetectionSystem()
    system.demo_detection_with_navigation()


if __name__ == "__main__":
    demo_smart_detection_system()
