#!/usr/bin/env python3
"""
Demo script cho refactored Smart Waste Management System
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import GPSCoordinate
from core.routing_engine import RoutingEngine
from core.detection_engine import DetectionEngine
from interfaces.web_interface import WebMapInterface
from interfaces.mobile_interface import MobileInterface
from utils.data_generator import DataGenerator
from utils.gui_helper import GUIHelper
from config.settings import get_system_config

def main():
    print("🚀 SMART WASTE MANAGEMENT SYSTEM - REFACTORED DEMO")
    print("=" * 60)
    
    # 1. Generate sample data
    print("📊 Generating sample data...")
    center = GPSCoordinate(10.77, 106.68)
    routing_engine, waste_bins, vehicles = DataGenerator.create_complete_system(center)
    
    print(f"✅ Created:")
    print(f"   - {len(waste_bins)} waste bins")
    print(f"   - {len(vehicles)} vehicles") 
    print(f"   - {len(routing_engine.roads)} roads")
    
    # 2. System status
    print("\n📋 System Status:")
    bins_by_status = {}
    for bin_data in waste_bins:
        status = bin_data.status.value
        bins_by_status[status] = bins_by_status.get(status, 0) + 1
    
    for status, count in bins_by_status.items():
        icon = GUIHelper.get_status_icon(status)
        print(f"   {icon} {status}: {count} bins")
    
    urgent_bins = [b for b in waste_bins if b.needs_collection]
    print(f"⚠️ Bins needing collection: {len(urgent_bins)}")
    
    # 3. Create web interface
    print("\n🌐 Creating web interface...")
    try:
        web_interface = WebMapInterface(routing_engine)
        web_interface.set_waste_bins(waste_bins)
        web_interface.set_current_position(center)
        
        web_map_path = web_interface.create_map("demo_web_map.html")
        print(f"✅ Web map created: {web_map_path}")
        
    except Exception as e:
        print(f"❌ Web interface failed: {e}")
    
    # 4. Create mobile interface
    print("\n📱 Creating mobile interface...")
    try:
        mobile_interface = MobileInterface()
        mobile_interface.set_waste_bins(waste_bins)
        mobile_interface.set_current_position(center)
        
        mobile_app_path = mobile_interface.create_mobile_app("demo_mobile_app.html")
        print(f"✅ Mobile app created: {mobile_app_path}")
        
    except Exception as e:
        print(f"❌ Mobile interface failed: {e}")
    
    # 5. Detection engine demo
    print("\n🎥 Detection Engine Status:")
    try:
        detection_engine = DetectionEngine()
        status = "✅ Ready" if detection_engine.model_loaded else "⚠️ Demo mode"
        print(f"   {status}")
        
        if detection_engine.model_loaded:
            print("   - YOLOv8 model loaded successfully")
            print("   - Real-time detection available")
        else:
            print("   - Using simulation mode")
            print("   - Install ultralytics for full functionality")
            
    except Exception as e:
        print(f"   ❌ Detection engine error: {e}")
    
    # 6. Configuration info
    print("\n⚙️ Configuration:")
    try:
        config = get_system_config()
        print(f"   - Default center: {config.default_center_lat}, {config.default_center_lng}")
        print(f"   - Figure size: {config.figure_size}")
        print(f"   - Fuel consumption: {config.fuel_consumption_rate}L/100km")
        
    except Exception as e:
        print(f"   ⚠️ Config error: {e}")
    
    # 7. Architecture overview
    print("\n🏗️ Architecture Overview:")
    print("   📦 core/         - Business logic (models, routing, detection)")
    print("   🖥️ interfaces/   - User interfaces (web, desktop, mobile)")
    print("   🔧 utils/        - Utilities (data generation, GUI helpers)")
    print("   ⚙️ config/       - Configuration management")
    
    print("\n🎯 Usage Examples:")
    print("   python main.py --mode web        # Launch web interface")
    print("   python main.py --mode status     # Show system status")
    print("   python main.py --mode detection  # Real-time detection")
    
    print("\n✨ Key Improvements:")
    print("   ✅ Modular architecture")
    print("   ✅ Separation of concerns")
    print("   ✅ Reusable components")
    print("   ✅ Easy to test and maintain")
    print("   ✅ Multiple interface options")
    print("   ✅ Configuration management")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("Files created:")
    
    # List generated files
    generated_files = [
        "demo_web_map.html",
        "demo_mobile_app.html"
    ]
    
    for file_path in generated_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            formatted_size = GUIHelper.format_file_size(size) if hasattr(GUIHelper, 'format_file_size') else f"{size} bytes"
            print(f"   📄 {file_path} ({formatted_size})")

if __name__ == "__main__":
    main()
