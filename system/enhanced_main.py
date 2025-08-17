#!/usr/bin/env python3
"""
Enhanced Main Application - Smart Waste Detection with GPS & Navigation

This is the enhanced main application that includes:
1. Automatic GPS location tracking
2. Real-time waste detection
3. Navigation to waste classification points

Usage:
    python enhanced_main.py --mode detection    # Real-time detection with GPS
    python enhanced_main.py --mode gps          # GPS service only
    python enhanced_main.py --mode demo         # Complete demo
    python enhanced_main.py --mode web          # Web interface with GPS

Author: Smart Waste Management System
Date: August 2025
"""

import argparse
import sys
import os
import cv2
import numpy as np
import time
from typing import Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.smart_detection_system import SmartWasteDetectionSystem
from core.gps_service import GPSLocationService, WasteClassificationPoints
from interfaces.web_interface import WebMapInterface


def run_gps_demo():
    """Run GPS service demo"""
    print("🛰️ GPS SERVICE DEMO")
    print("=" * 50)
    
    gps = GPSLocationService()
    
    # Enable mock GPS for demo
    gps.enable_mock_mode(10.762622, 106.660172)
    
    # Get location
    location = gps.get_current_location()
    if location:
        print(f"📍 Current Location: {location.coordinate.lat:.6f}, {location.coordinate.lng:.6f}")
        print(f"🏙️ Address: {location.address}")
        print(f"🎯 Source: {location.source}")
    
    # Test classification points
    classification = WasteClassificationPoints()
    if location:
        nearest = classification.find_nearest_classification_point(location.coordinate, "plastic")
        if nearest:
            print(f"\n🏢 Nearest plastic waste point:")
            print(f"   Name: {nearest['name']}")
            print(f"   Distance: {nearest['distance']:.2f} km")
            print(f"   Hours: {nearest['operating_hours']}")


def run_detection_demo():
    """Run smart detection demo"""
    print("🤖 SMART DETECTION DEMO")
    print("=" * 50)
    
    # Initialize system
    system = SmartWasteDetectionSystem()
    system.start_system(enable_gps=True, mock_gps=True)
    
    # Create demo image
    demo_image = create_demo_image()
    
    # Perform detection
    results = system.detect_and_navigate(demo_image)
    
    print(f"\n📊 DETECTION SUMMARY:")
    stats = system.get_waste_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    system.stop_system()


def run_complete_demo():
    """Run complete system demo"""
    print("🚀 COMPLETE SMART WASTE SYSTEM DEMO")
    print("=" * 60)
    
    # 1. GPS Demo
    print("\n" + "="*20 + " GPS SERVICE " + "="*20)
    run_gps_demo()
    
    # 2. Detection Demo  
    print("\n" + "="*20 + " DETECTION SERVICE " + "="*20)
    run_detection_demo()
    
    # 3. Available Classification Points
    print("\n" + "="*20 + " CLASSIFICATION POINTS " + "="*20)
    classification = WasteClassificationPoints()
    points = classification.get_all_points_info()
    
    for point_id, point_info in points.items():
        print(f"\n🏢 {point_info['name']} ({point_id})")
        print(f"   📍 Location: {point_info['location'].lat:.4f}, {point_info['location'].lng:.4f}")
        print(f"   🗑️ Waste types: {', '.join(point_info['types'])}")
        print(f"   ⏰ Hours: {point_info['operating_hours']}")
        print(f"   📞 Contact: {point_info['contact']}")
    
    print(f"\n✅ Demo completed successfully!")


def run_web_interface_with_gps():
    """Run web interface with GPS integration"""
    print("🌐 STARTING WEB INTERFACE WITH GPS")
    print("=" * 50)
    
    try:
        # Initialize GPS service
        gps = GPSLocationService()
        gps.enable_mock_mode(10.762622, 106.660172)  # Mock GPS for demo
        
        location = gps.get_current_location()
        if location:
            print(f"📍 GPS Location: {location.coordinate.lat:.6f}, {location.coordinate.lng:.6f}")
            
            # Initialize web interface with GPS location
            web_interface = WebMapInterface()
            
            # Add classification points to map
            classification = WasteClassificationPoints()
            points = classification.get_all_points_info()
            
            # Create enhanced web map
            map_html = web_interface.create_enhanced_map_with_gps(
                center_lat=location.coordinate.lat,
                center_lng=location.coordinate.lng,
                classification_points=points
            )
            
            print(f"✅ Web interface created: {map_html}")
            print("🌐 Open the HTML file in your browser to view the map!")
        else:
            print("❌ Could not get GPS location")
            
    except Exception as e:
        print(f"❌ Error running web interface: {e}")


def run_real_time_detection():
    """Run real-time detection with camera (if available)"""
    print("📹 REAL-TIME DETECTION MODE")
    print("=" * 50)
    
    # Initialize system
    system = SmartWasteDetectionSystem()
    system.start_system(enable_gps=True, mock_gps=True)
    
    print("🎥 Attempting to open camera...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠️ Camera not available, using demo images...")
        # Use demo images instead
        for i in range(3):
            print(f"\n📸 Processing demo image {i+1}/3...")
            demo_image = create_demo_image()
            results = system.detect_and_navigate(demo_image)
            time.sleep(2)
    else:
        print("✅ Camera opened successfully!")
        print("Press 'q' to quit, 'space' to detect")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Smart Waste Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar to detect
                print("\n🔍 Performing detection...")
                results = system.detect_and_navigate(frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    system.stop_system()


def create_demo_image() -> np.ndarray:
    """Create a demo image for testing"""
    # Create a black image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate waste objects
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    positions = [
        (50, 50, 150, 150),
        (200, 100, 300, 200),
        (350, 150, 450, 250),
        (100, 300, 200, 400)
    ]
    
    for i, ((x1, y1, x2, y2), color) in enumerate(zip(positions, colors)):
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.putText(image, f"Object {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return image


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Enhanced Smart Waste Management System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_main.py --mode demo       # Complete system demo
  python enhanced_main.py --mode gps        # GPS service only
  python enhanced_main.py --mode detection  # Smart detection demo
  python enhanced_main.py --mode web        # Web interface with GPS
  python enhanced_main.py --mode realtime   # Real-time camera detection
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['demo', 'gps', 'detection', 'web', 'realtime'],
        default='demo',
        help='Operating mode (default: demo)'
    )
    
    args = parser.parse_args()
    
    print("🤖 ENHANCED SMART WASTE MANAGEMENT SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print("=" * 60)
    
    try:
        if args.mode == 'demo':
            run_complete_demo()
        elif args.mode == 'gps':
            run_gps_demo()
        elif args.mode == 'detection':
            run_detection_demo()
        elif args.mode == 'web':
            run_web_interface_with_gps()
        elif args.mode == 'realtime':
            run_real_time_detection()
        else:
            print(f"❌ Unknown mode: {args.mode}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n⏹️ System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
