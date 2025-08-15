"""
Smart Waste Management System - Refactored Main Application
Hệ thống quản lý rác thải thông minh được tái cấu trúc

Author: Smart Waste Management Team
Date: August 2025
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import GPSCoordinate
from core.routing_engine import RoutingEngine
from core.detection_engine import DetectionEngine
from interfaces.web_interface import WebMapInterface
from interfaces.desktop_interface import DesktopMapInterface
from utils.data_generator import DataGenerator
from utils.gui_helper import GUIHelper
from config.settings import get_config_manager, get_system_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartWasteManagementSystem:
    """Main application class"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the system
        
        Args:
            config_file: Path to configuration file
        """
        # Load configuration
        self.config_manager = get_config_manager()
        if config_file:
            self.config_manager.load_from_file(config_file)
        
        self.system_config = get_system_config()
        
        # Create necessary directories
        self.config_manager.create_directories()
        
        # Initialize components
        self.routing_engine = RoutingEngine()
        self.detection_engine = DetectionEngine(self.system_config.yolo_model_path)
        
        # Initialize data
        self.waste_bins = []
        self.vehicles = []
        self.current_position = GPSCoordinate(
            self.system_config.default_center_lat,
            self.system_config.default_center_lng
        )
        
        logger.info("🚀 Smart Waste Management System initialized")
    
    def setup_sample_data(self):
        """Setup sample data for testing"""
        logger.info("📊 Setting up sample data...")
        
        # Generate complete system data
        routing_engine, waste_bins, vehicles = DataGenerator.create_complete_system(
            self.current_position
        )
        
        self.routing_engine = routing_engine
        self.waste_bins = waste_bins  
        self.vehicles = vehicles
        
        logger.info(f"✅ Sample data created:")
        logger.info(f"   - {len(self.waste_bins)} waste bins")
        logger.info(f"   - {len(vehicles)} vehicles")
        logger.info(f"   - {len(routing_engine.roads)} roads")
    
    def run_web_interface(self, auto_open: bool = True):
        """Run web interface"""
        logger.info("🌐 Starting web interface...")
        
        try:
            web_interface = WebMapInterface(self.routing_engine)
            web_interface.set_waste_bins(self.waste_bins)
            web_interface.set_current_position(self.current_position)
            
            # Create map
            map_path = web_interface.create_map("smart_waste_map.html")
            logger.info(f"✅ Web map created: {map_path}")
            
            # Open in browser
            if auto_open:
                web_interface.open_in_browser(map_path)
                logger.info("🌐 Map opened in browser")
            
            return map_path
            
        except Exception as e:
            logger.error(f"❌ Web interface failed: {e}")
            return None
    
    def run_desktop_interface(self):
        """Run desktop GUI interface"""
        logger.info("🖥️ Starting desktop interface...")
        
        # Check GUI availability
        gui_available, message = GUIHelper.check_gui_availability()
        if not gui_available:
            logger.warning(f"⚠️ Desktop GUI not available: {message}")
            return False
        
        try:
            desktop_interface = DesktopMapInterface(self.routing_engine)
            desktop_interface.set_waste_bins(self.waste_bins)
            desktop_interface.set_current_position(self.current_position)
            
            # Create interface
            desktop_interface.create_interface(self.system_config.figure_size)
            logger.info("✅ Desktop interface created")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Desktop interface failed: {e}")
            return False
    
    def run_detection_demo(self, video_source: int = 0):
        """Run real-time detection demo"""
        logger.info("🎥 Starting detection demo...")
        
        try:
            self.detection_engine.process_realtime_stream(video_source)
            logger.info("✅ Detection demo completed")
            
        except Exception as e:
            logger.error(f"❌ Detection demo failed: {e}")
    
    def find_optimal_route(self, destination_bin_id: str):
        """Find optimal route to specific bin"""
        # Find the bin
        target_bin = None
        for bin_data in self.waste_bins:
            if bin_data.id == destination_bin_id:
                target_bin = bin_data
                break
        
        if not target_bin:
            logger.error(f"❌ Bin {destination_bin_id} not found")
            return None
        
        # Calculate route
        route = self.routing_engine.find_path_astar(
            self.current_position,
            target_bin.location
        )
        
        if route.is_valid:
            logger.info(f"🛣️ Route to bin {destination_bin_id}:")
            logger.info(f"   Distance: {route.total_distance:.2f} km")
            logger.info(f"   Time: {route.total_time:.0f} minutes") 
            logger.info(f"   Fuel: {route.fuel_estimate:.2f} L")
        else:
            logger.error(f"❌ No route found to bin {destination_bin_id}")
        
        return route
    
    def optimize_collection_route(self):
        """Optimize collection route for all bins needing collection"""
        # Filter bins that need collection
        bins_needing_collection = [
            bin_data for bin_data in self.waste_bins 
            if bin_data.needs_collection
        ]
        
        if not bins_needing_collection:
            logger.info("✅ No bins need collection")
            return None
        
        logger.info(f"🗑️ Optimizing route for {len(bins_needing_collection)} bins...")
        
        # Calculate optimal route
        route = self.routing_engine.optimize_collection_route(
            self.current_position,
            bins_needing_collection
        )
        
        if route.is_valid:
            logger.info(f"🛣️ Optimal collection route:")
            logger.info(f"   Distance: {route.total_distance:.2f} km")
            logger.info(f"   Time: {route.total_time:.0f} minutes")
            logger.info(f"   Fuel: {route.fuel_estimate:.2f} L")
            logger.info(f"   Bins: {len(bins_needing_collection)} locations")
        else:
            logger.error("❌ Could not optimize collection route")
        
        return route
    
    def print_system_status(self):
        """Print system status"""
        print("\n" + "="*60)
        print("🗑️  SMART WASTE MANAGEMENT SYSTEM STATUS")
        print("="*60)
        
        print(f"📍 Current Position: {self.current_position.lat:.6f}, {self.current_position.lng:.6f}")
        print(f"🗑️ Total Bins: {len(self.waste_bins)}")
        
        # Bin status breakdown
        from collections import Counter
        status_count = Counter(bin_data.status for bin_data in self.waste_bins)
        
        for status, count in status_count.items():
            icon = GUIHelper.get_status_icon(status.value)
            print(f"   {icon} {status.value}: {count} bins")
        
        # Bins needing collection
        urgent_bins = [b for b in self.waste_bins if b.needs_collection]
        print(f"⚠️ Bins needing collection: {len(urgent_bins)}")
        
        print(f"🚚 Vehicles: {len(self.vehicles)}")
        print(f"🛣️ Roads: {len(self.routing_engine.roads)}")
        
        # Detection engine status
        detection_status = "✅ Ready" if self.detection_engine.model_loaded else "⚠️ Demo mode"
        print(f"🎥 Detection Engine: {detection_status}")
        
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Waste Management System")
    
    parser.add_argument('--mode', choices=['web', 'desktop', 'detection', 'route', 'status'], 
                       default='web', help='Mode to run the application')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--bin-id', type=str, help='Bin ID for routing')
    parser.add_argument('--video-source', type=int, default=0, help='Video source for detection')
    parser.add_argument('--no-sample-data', action='store_true', help='Do not load sample data')
    parser.add_argument('--no-browser', action='store_true', help='Do not auto-open browser')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SmartWasteManagementSystem(args.config)
    
    # Setup sample data if not disabled
    if not args.no_sample_data:
        system.setup_sample_data()
    
    # Run based on mode
    if args.mode == 'web':
        system.run_web_interface(auto_open=not args.no_browser)
        
    elif args.mode == 'desktop':
        system.run_desktop_interface()
        
    elif args.mode == 'detection':
        system.run_detection_demo(args.video_source)
        
    elif args.mode == 'route':
        if args.bin_id:
            system.find_optimal_route(args.bin_id)
        else:
            system.optimize_collection_route()
            
    elif args.mode == 'status':
        system.print_system_status()
    
    logger.info("🏁 Application finished")


if __name__ == "__main__":
    main()
