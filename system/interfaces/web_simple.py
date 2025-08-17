"""
Web Map Interface - Enhanced with GPS and Classification Points

Simple web interface for the Smart Waste Management System
"""

import folium
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

class WebMapInterface:
    """Simple Web Map Interface"""
    
    def __init__(self):
        self.center_lat = 10.762622
        self.center_lng = 106.660172
    
    def create_enhanced_map_with_gps(self, center_lat: float, center_lng: float, 
                                   classification_points: dict = None) -> str:
        """Create enhanced map with GPS and classification points"""
        
        # Create map
        enhanced_map = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=14,
            tiles='OpenStreetMap'
        )
        
        # Add current GPS location
        folium.Marker(
            location=[center_lat, center_lng],
            popup=f"""
            <div style='width: 200px; text-align: center;'>
                <h4>📍 Vị trí GPS hiện tại</h4>
                <p><b>Tọa độ:</b> {center_lat:.6f}, {center_lng:.6f}</p>
                <p><b>Thời gian:</b> {datetime.now().strftime('%H:%M:%S')}</p>
            </div>
            """,
            tooltip="📱 Vị trí GPS",
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(enhanced_map)
        
        # Add GPS accuracy circle
        folium.Circle(
            location=[center_lat, center_lng],
            radius=100,
            color='red',
            fillColor='red',
            fillOpacity=0.1,
            popup="📡 Độ chính xác GPS: ±100m"
        ).add_to(enhanced_map)
        
        # Add classification points
        if classification_points:
            for point_id, point_info in classification_points.items():
                # Determine color based on capacity
                capacity = point_info.get('capacity', 'medium')
                color = 'purple' if capacity == 'very_high' else 'blue' if capacity == 'high' else 'green'
                
                # Create popup
                waste_types = ', '.join(point_info.get('types', []))
                popup_html = f"""
                <div style='width: 250px;'>
                    <h4 style='color: {color};'>🏢 {point_info['name']}</h4>
                    <p><b>ID:</b> {point_id}</p>
                    <p><b>Loại rác:</b> {waste_types}</p>
                    <p><b>Giờ hoạt động:</b> {point_info.get('operating_hours', 'N/A')}</p>
                    <p><b>Liên hệ:</b> {point_info.get('contact', 'N/A')}</p>
                </div>
                """
                
                folium.Marker(
                    location=[point_info['location'].lat, point_info['location'].lng],
                    popup=popup_html,
                    tooltip=f"🏢 {point_info['name']}",
                    icon=folium.Icon(color=color, icon='building', prefix='fa')
                ).add_to(enhanced_map)
        
        # Save map
        filename = f'smart_waste_gps_map_{int(time.time())}.html'
        enhanced_map.save(filename)
        
        return filename


def test_web_interface():
    """Test the web interface"""
    print("🌐 Testing Web Interface with GPS...")
    
    interface = WebMapInterface()
    
    # Mock classification points
    from ..core.models import GPSCoordinate
    mock_points = {
        "POINT_001": {
            "name": "Trung tâm tái chế Quận 1",
            "location": GPSCoordinate(10.769444, 106.681944),
            "types": ["plastic", "glass", "metal"],
            "operating_hours": "06:00-22:00",
            "contact": "028-3822-xxxx",
            "capacity": "high"
        }
    }
    
    # Create map
    map_file = interface.create_enhanced_map_with_gps(
        10.762622, 106.660172, mock_points
    )
    
    print(f"✅ Map created: {map_file}")


if __name__ == "__main__":
    test_web_interface()
