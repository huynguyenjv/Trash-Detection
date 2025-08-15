"""
Web Interface - Giao diện web với Folium
"""

import folium
from folium import plugins
import webbrowser
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GPSCoordinate, WasteBin, PathfindingResult
from core.enums import BinStatus
from core.routing_engine import RoutingEngine


class WebMapInterface:
    """Web-based interactive map interface"""
    
    def __init__(self, routing_engine: RoutingEngine):
        self.routing_engine = routing_engine
        self.map = None
        self.current_position = GPSCoordinate(10.77, 106.68)
        self.waste_bins: List[WasteBin] = []
        
    def set_waste_bins(self, bins: List[WasteBin]):
        """Set waste bins for display"""
        self.waste_bins = bins
    
    def set_current_position(self, position: GPSCoordinate):
        """Set current position"""
        self.current_position = position
    
    def create_map(self, save_path: str = "waste_map.html") -> str:
        """
        Create interactive web map
        
        Args:
            save_path: Path to save HTML file
            
        Returns:
            Absolute path to created HTML file
        """
        # Initialize map
        self.map = folium.Map(
            location=[self.current_position.lat, self.current_position.lng],
            zoom_start=14,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        self._add_tile_layers()
        
        # Add current position
        self._add_current_position()
        
        # Add waste bins
        self._add_waste_bins()
        
        # Add controls
        self._add_controls()
        
        # Add layer control
        folium.LayerControl().add_to(self.map)
        
        # Add plugins
        plugins.Fullscreen().add_to(self.map)
        plugins.MeasureControl().add_to(self.map)
        minimap = plugins.MiniMap()
        self.map.add_child(minimap)
        
        # Add JavaScript functionality
        self._add_javascript()
        
        # Save map
        self.map.save(save_path)
        return os.path.abspath(save_path)
    
    def _add_tile_layers(self):
        """Add different tile layers"""
        folium.TileLayer(
            tiles='CartoDB positron',
            name='Light Map'
        ).add_to(self.map)
        
        folium.TileLayer(
            tiles='CartoDB dark_matter', 
            name='Dark Map'
        ).add_to(self.map)
    
    def _add_current_position(self):
        """Add current position marker"""
        folium.Marker(
            location=[self.current_position.lat, self.current_position.lng],
            popup=f"""
            <div style='width: 200px'>
                <h4>📍 Vị trí hiện tại</h4>
                <p><b>Lat:</b> {self.current_position.lat:.6f}</p>
                <p><b>Lng:</b> {self.current_position.lng:.6f}</p>
                <p><b>Thời gian:</b> {datetime.now().strftime('%H:%M:%S')}</p>
            </div>
            """,
            tooltip="Vị trí hiện tại",
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(self.map)
        
        # GPS accuracy circle
        folium.Circle(
            location=[self.current_position.lat, self.current_position.lng],
            radius=50,
            color='red',
            fillColor='red',
            fillOpacity=0.1,
            popup="GPS Accuracy: ±50m"
        ).add_to(self.map)
    
    def _add_waste_bins(self):
        """Add waste bin markers"""
        bin_group = folium.FeatureGroup(name="🗑️ Waste Bins")
        
        for bin_data in self.waste_bins:
            # Determine icon color based on status
            if bin_data.status == BinStatus.FULL:
                color = 'red'
                icon = 'exclamation-triangle'
            elif bin_data.status == BinStatus.NEAR_FULL:
                color = 'orange' 
                icon = 'exclamation'
            else:
                color = 'green'
                icon = 'check'
            
            # Create popup
            popup_html = f"""
            <div style='width: 250px'>
                <h4>🗑️ Thùng rác #{bin_data.id}</h4>
                <p><b>Trạng thái:</b> <span style='color: {color}'>{bin_data.status.value}</span></p>
                <p><b>Dung lượng:</b> {bin_data.fill_percentage:.1f}%</p>
                <p><b>Loại rác:</b> {', '.join([wt.value for wt in bin_data.supported_types])}</p>
                <p><b>Tọa độ:</b> {bin_data.location.lat:.4f}, {bin_data.location.lng:.4f}</p>
                <hr>
                <button onclick="routeToBin('{bin_data.id}', {bin_data.location.lat}, {bin_data.location.lng})" 
                        style='background: #34a853; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;'>
                    🧭 Chỉ đường
                </button>
            </div>
            """
            
            folium.Marker(
                location=[bin_data.location.lat, bin_data.location.lng],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Thùng rác #{bin_data.id} ({bin_data.status.value})",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(bin_group)
        
        bin_group.add_to(self.map)
    
    def _add_controls(self):
        """Add search and routing controls"""
        controls_html = f"""
        <div style='position: fixed; top: 10px; left: 50px; z-index: 1000; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);'>
            <div style='margin-bottom: 10px;'>
                <input type="text" id="searchBox" placeholder="🔍 Tìm địa điểm..." 
                       style='width: 300px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;'>
                <button onclick="searchLocation()" 
                        style='padding: 8px 16px; background: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 5px;'>
                    Tìm
                </button>
            </div>
            
            <div style='margin-bottom: 10px;'>
                <button onclick="findNearestBin()" 
                        style='background: #34a853; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-right: 10px;'>
                    🗑️ Thùng gần nhất
                </button>
                <button onclick="centerOnPosition()" 
                        style='background: #ea4335; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;'>
                    📍 Về vị trí
                </button>
            </div>
            
            <div id="routeInfo" style='font-size: 12px; color: #666; display: none; margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;'>
                <!-- Route information -->
            </div>
        </div>
        """
        
        self.map.get_root().html.add_child(folium.Element(controls_html))
    
    def _add_javascript(self):
        """Add JavaScript functionality"""
        js_code = f"""
        <script>
        let currentRoute = null;
        let destinationMarker = null;
        
        function centerOnPosition() {{
            map.setView([{self.current_position.lat}, {self.current_position.lng}], 16);
        }}
        
        function routeToBin(binId, lat, lng) {{
            // Remove existing route and marker
            if (currentRoute) {{
                map.removeLayer(currentRoute);
            }}
            if (destinationMarker) {{
                map.removeLayer(destinationMarker);
            }}
            
            // Add destination marker
            destinationMarker = L.marker([lat, lng], {{
                icon: L.icon({{
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [1, -34],
                    shadowSize: [41, 41]
                }})
            }}).addTo(map).bindPopup('🏁 Điểm đến').openPopup();
            
            // Create sample route
            const routeCoords = [
                [{self.current_position.lat}, {self.current_position.lng}],
                [(lat + {self.current_position.lat})/2, (lng + {self.current_position.lng})/2],
                [lat, lng]
            ];
            
            currentRoute = L.polyline(routeCoords, {{
                color: '#4285f4',
                weight: 6,
                opacity: 0.8
            }}).addTo(map);
            
            // Fit bounds
            map.fitBounds(currentRoute.getBounds(), {{padding: [20, 20]}});
            
            // Show route info
            const distance = (Math.random() * 3 + 0.5).toFixed(1);
            const time = Math.ceil(distance * 5);
            
            document.getElementById('routeInfo').style.display = 'block';
            document.getElementById('routeInfo').innerHTML = 
                '<b>🧭 Tuyến đường đến thùng #' + binId + '</b><br>' +
                '🛣️ Khoảng cách: ' + distance + ' km<br>' +
                '⏱️ Thời gian: ' + time + ' phút<br>' +
                '⛽ Nhiên liệu: ' + (distance * 0.08).toFixed(1) + 'L';
        }}
        
        function findNearestBin() {{
            // Simple implementation - find first bin
            const bins = [{', '.join([f"{{id: '{bin.id}', lat: {bin.location.lat}, lng: {bin.location.lng}}}" for bin in self.waste_bins])}];
            
            if (bins.length > 0) {{
                const nearest = bins[0];
                routeToBin(nearest.id, nearest.lat, nearest.lng);
            }}
        }}
        
        function searchLocation() {{
            const query = document.getElementById('searchBox').value.toLowerCase();
            const locations = {{
                'bệnh viện': [10.778, 106.672],
                'chợ': [10.765, 106.675],
                'trường học': [10.773, 106.690],
                'công viên': [10.769, 106.683]
            }};
            
            for (let [name, coords] of Object.entries(locations)) {{
                if (name.includes(query) || query.includes(name)) {{
                    map.setView(coords, 16);
                    return;
                }}
            }}
            
            alert('Không tìm thấy: ' + query);
        }}
        
        // Handle Enter key in search
        document.getElementById('searchBox').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                searchLocation();
            }}
        }});
        </script>
        """
        
        self.map.get_root().html.add_child(folium.Element(js_code))
    
    def open_in_browser(self, map_path: str):
        """Open map in browser"""
        webbrowser.open(f"file://{map_path}")
    
    def add_route_to_map(self, route: PathfindingResult, color: str = '#4285f4'):
        """Add route visualization to map"""
        if not route.is_valid or not self.map:
            return
        
        # Create route line
        locations = [[point.lat, point.lng] for point in route.path]
        
        folium.PolyLine(
            locations=locations,
            color=color,
            weight=6,
            opacity=0.8,
            popup=f"🛣️ Khoảng cách: {route.total_distance:.1f}km, Thời gian: {route.total_time:.0f}phút"
        ).add_to(self.map)
        
        # Add route markers
        if len(route.path) > 1:
            # Start marker
            folium.Marker(
                location=[route.path[0].lat, route.path[0].lng],
                popup="🚩 Điểm bắt đầu",
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(self.map)
            
            # End marker
            folium.Marker(
                location=[route.path[-1].lat, route.path[-1].lng],
                popup="🏁 Điểm kết thúc",
                icon=folium.Icon(color='red', icon='stop', prefix='fa')
            ).add_to(self.map)
