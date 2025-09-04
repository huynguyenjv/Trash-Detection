#!/bin/bash
# Quick test script for refactored system

echo "🚀 Testing Refactored Smart Waste Management System..."
cd refactored_system

echo "📦 Testing imports..."
python3 -c "
import sys
sys.path.append('.')

try:
    from core.models import GPSCoordinate, WasteBin
    from core.enums import WasteType, BinStatus
    from core.routing_engine import RoutingEngine
    print('✅ Core modules OK')
except ImportError as e:
    print(f'❌ Core import failed: {e}')

try:
    from utils.data_generator import DataGenerator
    from utils.gui_helper import GUIHelper
    print('✅ Utils modules OK')
except ImportError as e:
    print(f'❌ Utils import failed: {e}')

try:
    from config.settings import get_system_config
    print('✅ Config modules OK')
except ImportError as e:
    print(f'❌ Config import failed: {e}')
"

echo "🧪 Testing core functionality..."
python3 -c "
import sys
sys.path.append('.')

from core.models import GPSCoordinate
from utils.data_generator import DataGenerator

# Test data generation
center = GPSCoordinate(10.77, 106.68)
routing_engine, bins, vehicles = DataGenerator.create_complete_system(center)

print(f'✅ Generated {len(bins)} waste bins')
print(f'✅ Generated {len(vehicles)} vehicles')  
print(f'✅ Generated {len(routing_engine.roads)} roads')

# Test routing
route = routing_engine.find_path_astar(center, bins[0].location)
if route.is_valid:
    print(f'✅ Route found: {route.total_distance:.2f}km, {route.total_time:.0f}min')
else:
    print('❌ Route not found')
"

echo "🌐 Testing web interface creation..."
python3 -c "
import sys
sys.path.append('.')

from interfaces.web_interface import WebMapInterface
from utils.data_generator import DataGenerator
from core.models import GPSCoordinate

try:
    center = GPSCoordinate(10.77, 106.68)
    routing_engine, bins, vehicles = DataGenerator.create_complete_system(center)
    
    web_interface = WebMapInterface(routing_engine)
    web_interface.set_waste_bins(bins)
    
    map_path = web_interface.create_map('test_map.html')
    print(f'✅ Web map created: {map_path}')
    
except Exception as e:
    print(f'❌ Web interface failed: {e}')
"

echo "📱 Testing mobile interface..."
python3 -c "
import sys
sys.path.append('.')

from interfaces.mobile_interface import MobileInterface
from utils.data_generator import DataGenerator
from core.models import GPSCoordinate

try:
    center = GPSCoordinate(10.77, 106.68)
    _, bins, _ = DataGenerator.create_complete_system(center)
    
    mobile_interface = MobileInterface()
    mobile_interface.set_waste_bins(bins)
    
    app_path = mobile_interface.create_mobile_app('test_mobile.html')
    print(f'✅ Mobile app created: {app_path}')
    
except Exception as e:
    print(f'❌ Mobile interface failed: {e}')
"

echo "🎯 All tests completed!"
echo ""
echo "📋 Usage examples:"
echo "  python main.py --mode web       # Web interface"
echo "  python main.py --mode status    # System status" 
echo "  python main.py --mode route     # Route optimization"
echo ""
echo "📁 Generated test files:"
ls -la test_*.html 2>/dev/null || echo "  (no test files created)"
