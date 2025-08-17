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
            popup=f"<b>Vị trí hiện tại</b><br>Lat: {self.current_position.lat:.6f}<br>Lng: {self.current_position.lng:.6f}",
            tooltip="📍 Vị trí hiện tại",
            icon=folium.Icon(color='blue', icon='user', prefix='fa')
        ).add_to(self.map)
    def create_enhanced_map_with_gps(self, center_lat: float, center_lng: float, 
                                   classification_points: dict = None) -> str:
        """Create enhanced map with GPS and classification points"""
        import folium
        from datetime import datetime
        
        # Create map centered on GPS location
        enhanced_map = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=14,
            tiles=None
        )
        
        # Add different tile layers
        folium.TileLayer('OpenStreetMap', name='Street Map').add_to(enhanced_map)
        folium.TileLayer('Stamen Terrain', name='Terrain').add_to(enhanced_map)
        folium.TileLayer('CartoDB positron', name='Light Mode').add_to(enhanced_map)
        
        # Add current location marker
        folium.Marker(
            location=[center_lat, center_lng],
            popup=f"""
            <div style='width: 200px; text-align: center;'>
                <h4>📍 Vị trí hiện tại</h4>
                <p><b>GPS:</b> {center_lat:.6f}, {center_lng:.6f}</p>
                <p><b>Thời gian:</b> {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}</p>
                <p style='color: #666; font-size: 12px;'>Tự động cập nhật từ GPS</p>
            </div>
            """,
            tooltip="📱 Vị trí GPS hiện tại",
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(enhanced_map)
        
        # GPS accuracy circle  
        folium.Circle(
            location=[center_lat, center_lng],
            radius=100,
            color='red',
            fillColor='red',
            fillOpacity=0.15,
            popup="📡 Độ chính xác GPS: ±100m",
            tooltip="Vùng độ chính xác GPS"
        ).add_to(enhanced_map)
        
        # Add classification points if provided
        if classification_points:
            self._add_classification_points_to_map(enhanced_map, classification_points)
        
        # Add enhanced controls
        self._add_enhanced_controls(enhanced_map, center_lat, center_lng)
        
        # Add enhanced JavaScript
        self._add_enhanced_javascript(enhanced_map)
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(enhanced_map)
        
        # Save map
        filename = f'smart_waste_gps_map_{int(time.time())}.html'
        enhanced_map.save(filename)
        
        return filename
    
    def _add_classification_points_to_map(self, map_obj, classification_points: dict):
        """Add waste classification points to map"""
        import folium
        
        # Create feature group for classification points
        classification_group = folium.FeatureGroup(name="🏢 Điểm phân loại rác")
        
        for point_id, point_info in classification_points.items():
            # Determine icon based on capacity
            capacity = point_info.get('capacity', 'medium')
            if capacity == 'very_high':
                color = 'purple'
                icon = 'industry'
            elif capacity == 'high':
                color = 'blue' 
                icon = 'building'
            else:
                color = 'green'
                icon = 'recycle'
            
            # Create detailed popup
            waste_types = point_info.get('types', [])
            waste_types_html = '<br>'.join([f"• {wtype}" for wtype in waste_types])
            
            popup_html = f"""
            <div style='width: 280px;'>
                <h4 style='color: {color}; margin-bottom: 10px;'>
                    🏢 {point_info['name']}
                </h4>
                <div style='margin-bottom: 8px;'>
                    <b>📍 ID:</b> {point_id}
                </div>
                <div style='margin-bottom: 8px;'>
                    <b>📞 Liên hệ:</b> {point_info.get('contact', 'N/A')}
                </div>
                <div style='margin-bottom: 8px;'>
                    <b>⏰ Giờ hoạt động:</b> {point_info.get('operating_hours', 'N/A')}
                </div>
                <div style='margin-bottom: 8px;'>
                    <b>🏗️ Công suất:</b> {capacity}
                </div>
                <div style='margin-bottom: 10px;'>
                    <b>🗑️ Loại rác:</b><br>
                    {waste_types_html}
                </div>
                <div style='text-align: center; padding-top: 10px; border-top: 1px solid #eee;'>
                    <button onclick="navigateToClassificationPoint('{point_id}', {point_info['location'].lat}, {point_info['location'].lng}, '{point_info['name']}')"
                            style='background: #34a853; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-right: 10px;'>
                        🧭 Chỉ đường
                    </button>
                    <button onclick="callClassificationPoint('{point_info.get('contact', '')}')"
                            style='background: #4285f4; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;'>
                        📞 Gọi
                    </button>
                </div>
            </div>
            """
            
            folium.Marker(
                location=[point_info['location'].lat, point_info['location'].lng],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"🏢 {point_info['name']} - {point_id}",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(classification_group)
        
        classification_group.add_to(map_obj)
    
    def _add_enhanced_controls(self, map_obj, center_lat: float, center_lng: float):
        """Add enhanced control panel"""
        import folium
        
        controls_html = f"""
        <div id="controlPanel" style='position: fixed; top: 10px; left: 10px; z-index: 1000; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); min-width: 350px;'>
            <h3 style='margin: 0 0 15px 0; color: #333; font-size: 18px;'>
                🤖 Smart Waste Management
            </h3>
            
            <!-- GPS Status -->
            <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #28a745;'>
                <div style='font-weight: bold; color: #28a745; margin-bottom: 5px;'>
                    📡 GPS Status: ACTIVE
                </div>
                <div style='font-size: 12px; color: #666;'>
                    Location: {center_lat:.6f}, {center_lng:.6f}
                </div>
            </div>
            
            <!-- Search -->
            <div style='margin-bottom: 15px;'>
                <input type="text" id="searchBox" placeholder="🔍 Tìm điểm phân loại rác..." 
                       style='width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; box-sizing: border-box;'>
            </div>
            
            <!-- Quick Actions -->
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;'>
                <button onclick="findNearestClassificationPoint()" 
                        style='background: linear-gradient(45deg, #28a745, #20c997); color: white; border: none; padding: 12px; border-radius: 5px; cursor: pointer; font-size: 13px; font-weight: bold;'>
                    🏢 Gần nhất
                </button>
                <button onclick="centerOnGPS()" 
                        style='background: linear-gradient(45deg, #dc3545, #fd7e14); color: white; border: none; padding: 12px; border-radius: 5px; cursor: pointer; font-size: 13px; font-weight: bold;'>
                    📍 GPS
                </button>
            </div>
            
            <!-- Waste Type Filter -->
            <div style='margin-bottom: 15px;'>
                <label style='font-weight: bold; color: #333; display: block; margin-bottom: 5px;'>🗑️ Lọc theo loại rác:</label>
                <select id="wasteTypeFilter" onchange="filterByWasteType()" 
                        style='width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;'>
                    <option value="">Tất cả loại rác</option>
                    <option value="plastic">Nhựa</option>
                    <option value="glass">Thủy tinh</option>
                    <option value="metal">Kim loại</option>
                    <option value="paper">Giấy</option>
                    <option value="organic">Hữu cơ</option>
                    <option value="electronic">Điện tử</option>
                    <option value="hazardous">Nguy hiểm</option>
                </select>
            </div>
            
            <!-- Status Display -->
            <div id="statusDisplay" style='background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 12px; color: #666; display: none;'>
                <!-- Status updates will appear here -->
            </div>
            
            <!-- Toggle Button -->
            <button id="togglePanel" onclick="toggleControlPanel()" 
                    style='position: absolute; top: -10px; right: -10px; width: 30px; height: 30px; border-radius: 50%; background: #007bff; color: white; border: none; cursor: pointer; font-size: 16px;'>
                ×
            </button>
        </div>
        """
        
        map_obj.get_root().html.add_child(folium.Element(controls_html))
    
    def _add_enhanced_javascript(self, map_obj):
        """Add enhanced JavaScript functionality"""
        import folium
        
        js_code = f"""
        <script>
        let currentRoute = null;
        let gpsPosition = null;
        let classificationPoints = [];
        
        // Initialize GPS position  
        gpsPosition = {{lat: {self.center_lat}, lng: {self.center_lng}}};
        
        // Control panel functions
        function toggleControlPanel() {{
            const panel = document.getElementById('controlPanel');
            const button = document.getElementById('togglePanel');
            
            if (panel.style.left === '-340px') {{
                panel.style.left = '10px';
                button.innerHTML = '×';
            }} else {{
                panel.style.left = '-340px';  
                button.innerHTML = '☰';
            }}
        }}
        
        // Navigation functions
        function navigateToClassificationPoint(pointId, lat, lng, name) {{
            showStatus(`🧭 Đang tính toán đường đi đến ${{name}}...`);
            
            // Clear existing route
            if (currentRoute) {{
                map.removeLayer(currentRoute);
            }}
            
            // Add simple route line
            const routeCoords = [
                [gpsPosition.lat, gpsPosition.lng],
                [lat, lng]
            ];
            
            currentRoute = L.polyline(routeCoords, {{
                color: '#007bff',
                weight: 4,
                opacity: 0.7,
                dashArray: '10, 10'
            }}).addTo(map);
            
            // Calculate distance
            const distance = calculateDistance(gpsPosition.lat, gpsPosition.lng, lat, lng);
            const estimatedTime = Math.round(distance * 2); // Assume 30km/h speed
            
            showStatus(`
                📍 Đích đến: ${{name}}<br>
                📏 Khoảng cách: ${{distance.toFixed(2)}} km<br>
                ⏱️ Thời gian: ~${{estimatedTime}} phút<br>
                <button onclick="clearRoute()" style="margin-top:5px; padding:5px 10px; background:#dc3545; color:white; border:none; border-radius:3px; cursor:pointer;">Xóa đường đi</button>
            `);
            
            // Fit bounds to show route
            map.fitBounds(routeCoords, {{padding: [50, 50]}});
        }}
        
        function findNearestClassificationPoint() {{
            showStatus('🔍 Đang tìm điểm phân loại gần nhất...');
            
            // This would normally query the backend
            setTimeout(() => {{
                showStatus('✅ Đã tìm thấy điểm gần nhất! Kiểm tra bản đồ.');
                // Simulate finding nearest point
                centerOnGPS();
            }}, 1000);
        }}
        
        function centerOnGPS() {{
            map.setView([gpsPosition.lat, gpsPosition.lng], 15);
            showStatus('📍 Đã chuyển về vị trí GPS hiện tại');
        }}
        
        function filterByWasteType() {{
            const wasteType = document.getElementById('wasteTypeFilter').value;
            if (wasteType) {{
                showStatus(`🗑️ Đang lọc điểm thu gom loại rác: ${{wasteType}}`);
            }} else {{
                showStatus('📋 Hiển thị tất cả điểm thu gom');
            }}
        }}
        
        function callClassificationPoint(phoneNumber) {{
            if (phoneNumber && phoneNumber !== 'N/A' && phoneNumber !== 'Custom point') {{
                window.open(`tel:${{phoneNumber}}`);
                showStatus(`📞 Đang gọi: ${{phoneNumber}}`);
            }} else {{
                showStatus('⚠️ Số điện thoại không khả dụng');
            }}
        }}
        
        function clearRoute() {{
            if (currentRoute) {{
                map.removeLayer(currentRoute);
                currentRoute = null;
                showStatus('✅ Đã xóa đường đi');
            }}
        }}
        
        // Utility functions
        function calculateDistance(lat1, lng1, lat2, lng2) {{
            const R = 6371; // Earth's radius in km
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLng = (lng2 - lng1) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
                      Math.sin(dLng/2) * Math.sin(dLng/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }}
        
        function showStatus(message) {{
            const statusDiv = document.getElementById('statusDisplay');
            statusDiv.innerHTML = message;
            statusDiv.style.display = 'block';
            
            // Auto-hide after 10 seconds
            setTimeout(() => {{
                statusDiv.style.display = 'none';
            }}, 10000);
        }}
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            showStatus('🤖 Smart Waste Management System đã sẵn sàng!');
            
            // Add search functionality
            document.getElementById('searchBox').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    const query = this.value;
                    if (query) {{
                        showStatus(`🔍 Đang tìm kiếm: ${{query}}`);
                    }}
                }}
            }});
        }});
        </script>
        
        <style>
        #controlPanel {{
            transition: left 0.3s ease;
        }}
        
        #controlPanel button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }}
        
        #searchBox:focus {{
            border-color: #007bff;
            box-shadow: 0 0 5px rgba(0,123,255,0.3);
            outline: none;
        }}
        
        .leaflet-popup-content {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        </style>
        """
        
        map_obj.get_root().html.add_child(folium.Element(js_code))
    
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
