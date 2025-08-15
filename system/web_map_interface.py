"""
Web-based Map Interface - Giống Google Maps thực sự
Sử dụng Folium để tạo interactive web map

Author: Smart Waste Management System  
Date: August 2025
"""

import folium
from folium import plugins
import webbrowser
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import time

from smart_routing_system import (
    SmartRoutingSystem, GPSCoordinate, WasteType,
    PathfindingResult, create_sample_data, BinStatus
)


class WebMapInterface:
    """Web-based interactive map như Google Maps"""
    
    def __init__(self, routing_system: SmartRoutingSystem):
        self.routing_system = routing_system
        self.map = None
        self.current_position = GPSCoordinate(10.77, 106.68)
        self.destination = None
        self.current_route = None
        
    def create_web_map(self, save_path: str = "smart_waste_map.html") -> str:
        """Tạo web map interactive"""
        
        # Initialize map centered on current position
        self.map = folium.Map(
            location=[self.current_position.lat, self.current_position.lng],
            zoom_start=14,
            tiles='OpenStreetMap'
        )
        
        # Add multiple tile layers (với attribution cần thiết)
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            name='Terrain',
            overlay=False,
            control=True
        ).add_to(self.map)
        
        folium.TileLayer(
            tiles='CartoDB positron',
            name='Light Map'
        ).add_to(self.map)
        
        folium.TileLayer(
            tiles='CartoDB dark_matter', 
            name='Dark Map'
        ).add_to(self.map)
        
        # Add current position
        self._add_current_position()
        
        # Add waste bins
        self._add_waste_bins()
        
        # Add traffic simulation
        self._add_traffic_simulation()
        
        # Add search functionality
        self._add_search_box()
        
        # Add routing controls
        self._add_routing_controls()
        
        # Add layer control
        folium.LayerControl().add_to(self.map)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(self.map)
        
        # Add measure tool
        plugins.MeasureControl().add_to(self.map)
        
        # Add mini map
        minimap = plugins.MiniMap()
        self.map.add_child(minimap)
        
        # Save map
        self.map.save(save_path)
        return os.path.abspath(save_path)
    
    def _add_current_position(self):
        """Thêm marker vị trí hiện tại"""
        # Current position với GPS accuracy circle
        folium.Marker(
            location=[self.current_position.lat, self.current_position.lng],
            popup=f"""
            <div style='width: 200px'>
                <h4>📍 Vị trí hiện tại</h4>
                <p><b>Lat:</b> {self.current_position.lat:.6f}</p>
                <p><b>Lng:</b> {self.current_position.lng:.6f}</p>
                <p><b>Thời gian:</b> {datetime.now().strftime('%H:%M:%S')}</p>
                <button onclick="centerOnPosition()" style='background: #4285f4; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;'>
                    Căn giữa bản đồ
                </button>
            </div>
            """,
            tooltip="Vị trí hiện tại",
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(self.map)
        
        # GPS accuracy circle
        folium.Circle(
            location=[self.current_position.lat, self.current_position.lng],
            radius=50,  # 50 meters accuracy
            color='red',
            fillColor='red',
            fillOpacity=0.1,
            popup="GPS Accuracy: ±50m"
        ).add_to(self.map)
    
    def _add_waste_bins(self):
        """Thêm markers cho thùng rác"""
        # Create feature group for bins
        bin_group = folium.FeatureGroup(name="🗑️ Waste Bins")
        
        # routing_system.waste_bins là Dict[str, WasteBin], nên cần .values()
        for bin_data in self.routing_system.waste_bins.values():
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
            
            # Create detailed popup
            popup_html = f"""
            <div style='width: 250px'>
                <h4>🗑️ Thùng rác #{bin_data.id}</h4>
                <p><b>Trạng thái:</b> <span style='color: {color}'>{bin_data.status.value}</span></p>
                <p><b>Loại rác:</b> {', '.join([wt.value for wt in bin_data.supported_types])}</p>
                <p><b>Dung lượng:</b> {bin_data.max_capacity}L</p>
                <p><b>Hiện tại:</b> {bin_data.current_capacity}L</p>
                <p><b>Tọa độ:</b> {bin_data.location.lat:.4f}, {bin_data.location.lng:.4f}</p>
                <hr>
                <button onclick="routeToBin('{bin_data.id}', {bin_data.location.lat}, {bin_data.location.lng})" 
                        style='background: #34a853; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; margin-right: 5px;'>
                    🧭 Chỉ đường
                </button>
                <button onclick="reportBin('{bin_data.id}')"
                        style='background: #ea4335; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;'>
                    ⚠️ Báo cáo
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
    
    def _add_traffic_simulation(self):
        """Thêm simulation giao thông"""
        traffic_group = folium.FeatureGroup(name="🚦 Traffic")
        
        # Generate random traffic data
        np.random.seed(42)
        
        # Define some main roads
        roads = [
            [(10.765, 106.675), (10.775, 106.685)],
            [(10.770, 106.670), (10.770, 106.690)],
            [(10.772, 106.678), (10.778, 106.684)],
            [(10.768, 106.682), (10.774, 106.676)]
        ]
        
        traffic_colors = {
            'good': 'green',
            'medium': 'orange', 
            'bad': 'red'
        }
        
        for i, road in enumerate(roads):
            # Random traffic condition
            condition = np.random.choice(['good', 'medium', 'bad'], p=[0.6, 0.3, 0.1])
            
            folium.PolyLine(
                locations=road,
                color=traffic_colors[condition],
                weight=8,
                opacity=0.8,
                popup=f"Đường {i+1}: Giao thông {condition}",
                tooltip=f"Traffic: {condition}"
            ).add_to(traffic_group)
        
        traffic_group.add_to(self.map)
    
    def _add_search_box(self):
        """Thêm search box"""
        # Add custom HTML/JS for search
        search_html = """
        <div style='position: fixed; top: 10px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
            <input type="text" id="searchBox" placeholder="🔍 Tìm địa điểm..." 
                   style='width: 300px; padding: 8px; border: 1px solid #ccc; border-radius: 3px;'>
            <button onclick="searchLocation()" style='padding: 8px 15px; background: #4285f4; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 5px;'>
                Tìm
            </button>
        </div>
        """
        
        self.map.get_root().html.add_child(folium.Element(search_html))
    
    def _add_routing_controls(self):
        """Thêm controls cho routing"""
        controls_html = """
        <div style='position: fixed; top: 70px; left: 50px; z-index: 1000; background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
            <h4 style='margin: 0 0 10px 0;'>🧭 Chỉ đường</h4>
            
            <div style='margin-bottom: 10px;'>
                <label>📍 Từ:</label><br>
                <input type="text" id="startPoint" placeholder="Vị trí hiện tại" disabled 
                       style='width: 250px; padding: 5px; margin-top: 3px; border: 1px solid #ccc; border-radius: 3px;'>
            </div>
            
            <div style='margin-bottom: 10px;'>
                <label>🏁 Đến:</label><br>
                <input type="text" id="endPoint" placeholder="Chọn điểm đến trên bản đồ" 
                       style='width: 250px; padding: 5px; margin-top: 3px; border: 1px solid #ccc; border-radius: 3px;'>
            </div>
            
            <div style='margin-bottom: 10px;'>
                <button onclick="calculateRoute()" style='background: #34a853; color: white; border: none; padding: 8px 15px; border-radius: 3px; cursor: pointer; margin-right: 5px;'>
                    🛣️ Tính đường
                </button>
                <button onclick="clearRoute()" style='background: #ea4335; color: white; border: none; padding: 8px 15px; border-radius: 3px; cursor: pointer;'>
                    🗑️ Xóa
                </button>
            </div>
            
            <div id="routeInfo" style='font-size: 12px; color: #666; display: none;'>
                <!-- Route information will be displayed here -->
            </div>
        </div>
        """
        
        self.map.get_root().html.add_child(folium.Element(controls_html))
    
    def add_javascript_functions(self):
        """Thêm JavaScript functions"""
        js_code = """
        <script>
        let currentRoute = null;
        let destinationMarker = null;
        
        // Search locations database
        const locations = {
            'home': [10.77, 106.68],
            'office': [10.775, 106.685],
            'market': [10.765, 106.675],
            'hospital': [10.778, 106.672],
            'school': [10.773, 106.690],
            'park': [10.769, 106.683],
            'mall': [10.776, 106.679]
        };
        
        function searchLocation() {
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            
            for (let [name, coords] of Object.entries(locations)) {
                if (name.includes(searchTerm) || searchTerm.includes(name)) {
                    map.setView(coords, 16);
                    
                    // Add temporary marker
                    if (window.searchMarker) {
                        map.removeLayer(window.searchMarker);
                    }
                    
                    window.searchMarker = L.marker(coords)
                        .addTo(map)
                        .bindPopup(`📍 ${name.charAt(0).toUpperCase() + name.slice(1)}`)
                        .openPopup();
                    
                    return;
                }
            }
            
            alert('Không tìm thấy địa điểm: ' + searchTerm);
        }
        
        function centerOnPosition() {
            map.setView([""" + str(self.current_position.lat) + """, """ + str(self.current_position.lng) + """], 16);
        }
        
        function routeToBin(binId, lat, lng) {
            document.getElementById('endPoint').value = `Thùng rác #${binId}`;
            
            // Set destination marker
            if (destinationMarker) {
                map.removeLayer(destinationMarker);
            }
            
            destinationMarker = L.marker([lat, lng], {
                icon: L.icon({
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [1, -34],
                    shadowSize: [41, 41]
                })
            }).addTo(map).bindPopup('🏁 Điểm đến').openPopup();
            
            calculateRoute();
        }
        
        function calculateRoute() {
            const endPoint = document.getElementById('endPoint').value;
            if (!endPoint) {
                alert('Vui lòng chọn điểm đến');
                return;
            }
            
            // Simulate route calculation
            document.getElementById('routeInfo').style.display = 'block';
            document.getElementById('routeInfo').innerHTML = `
                <hr>
                <b>📍 Tuyến đường:</b><br>
                🛣️ Khoảng cách: 2.3 km<br>
                ⏱️ Thời gian: 15 phút<br>
                ⛽ Nhiên liệu: 0.2L<br>
                🚦 Giao thông: Tốt<br>
                <small style='color: #666;'>Cập nhật: ${new Date().toLocaleTimeString()}</small>
            `;
            
            // Draw sample route
            if (currentRoute) {
                map.removeLayer(currentRoute);
            }
            
            const routeCoords = [
                [""" + str(self.current_position.lat) + """, """ + str(self.current_position.lng) + """],
                [""" + str(self.current_position.lat + 0.005) + """, """ + str(self.current_position.lng + 0.008) + """],
                [""" + str(self.current_position.lat + 0.008) + """, """ + str(self.current_position.lng + 0.012) + """]
            ];
            
            currentRoute = L.polyline(routeCoords, {
                color: '#4285f4',
                weight: 6,
                opacity: 0.8
            }).addTo(map);
            
            // Add route arrows
            const decorator = L.polylineDecorator(currentRoute, {
                patterns: [
                    {offset: 25, repeat: 100, symbol: L.Symbol.arrowHead({pixelSize: 15, polygon: false, pathOptions: {stroke: true, color: '#4285f4'}})}
                ]
            }).addTo(map);
            
            // Fit bounds to show full route
            map.fitBounds(currentRoute.getBounds(), {padding: [20, 20]});
        }
        
        function clearRoute() {
            if (currentRoute) {
                map.removeLayer(currentRoute);
                currentRoute = null;
            }
            
            if (destinationMarker) {
                map.removeLayer(destinationMarker);
                destinationMarker = null;
            }
            
            document.getElementById('endPoint').value = '';
            document.getElementById('routeInfo').style.display = 'none';
        }
        
        function reportBin(binId) {
            if (confirm(`Báo cáo vấn đề với thùng rác #${binId}?`)) {
                alert('Cảm ơn bạn! Báo cáo đã được gửi đến đội quản lý.');
            }
        }
        
        // Handle Enter key in search box
        document.getElementById('searchBox').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchLocation();
            }
        });
        
        // Handle map clicks for destination selection
        map.on('click', function(e) {
            const lat = e.latlng.lat.toFixed(6);
            const lng = e.latlng.lng.toFixed(6);
            
            document.getElementById('endPoint').value = `${lat}, ${lng}`;
            
            // Set destination marker
            if (destinationMarker) {
                map.removeLayer(destinationMarker);
            }
            
            destinationMarker = L.marker([lat, lng], {
                icon: L.icon({
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [1, -34],
                    shadowSize: [41, 41]
                })
            }).addTo(map).bindPopup('🏁 Điểm đến đã chọn').openPopup();
        });
        </script>
        """
        
        self.map.get_root().html.add_child(folium.Element(js_code))
    
    def create_enhanced_web_map(self, save_path: str = "enhanced_waste_map.html") -> str:
        """Tạo web map với đầy đủ tính năng"""
        map_path = self.create_web_map(save_path)
        
        # Add JavaScript functions
        self.add_javascript_functions()
        
        # Re-save with enhanced features
        self.map.save(save_path)
        
        return os.path.abspath(save_path)
    
    def open_in_browser(self, map_path: str):
        """Mở map trong trình duyệt"""
        webbrowser.open(f"file://{map_path}")


def create_mobile_app_interface():
    """Tạo mobile app interface với HTML/CSS/JS"""
    mobile_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Waste Navigation</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <!-- Custom CSS -->
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            height: 100vh;
            overflow: hidden;
        }
        
        #map {
            height: 100vh;
            width: 100vw;
        }
        
        .mobile-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .search-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .search-input {
            flex: 1;
            padding: 12px 15px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        
        .search-input:focus {
            border-color: #4285f4;
            box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.1);
        }
        
        .menu-btn {
            background: #4285f4;
            color: white;
            border: none;
            padding: 12px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
        }
        
        .mobile-controls {
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            z-index: 1000;
            display: flex;
            gap: 10px;
        }
        
        .control-btn {
            flex: 1;
            padding: 15px;
            background: white;
            border: none;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .control-btn:active {
            transform: scale(0.95);
        }
        
        .control-btn.primary {
            background: #4285f4;
            color: white;
        }
        
        .control-btn.success {
            background: #34a853;
            color: white;
        }
        
        .control-btn.danger {
            background: #ea4335;
            color: white;
        }
        
        .floating-btn {
            position: fixed;
            bottom: 120px;
            right: 20px;
            z-index: 1000;
            background: #4285f4;
            color: white;
            border: none;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            cursor: pointer;
            font-size: 20px;
            transition: all 0.2s;
        }
        
        .floating-btn:active {
            transform: scale(0.9);
        }
        
        .route-panel {
            position: fixed;
            top: 80px;
            left: 20px;
            right: 20px;
            z-index: 1000;
            background: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        
        .route-panel.show {
            max-height: 200px;
        }
        
        .route-step {
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .route-step:last-child {
            border-bottom: none;
        }
        
        .route-icon {
            width: 24px;
            margin-right: 12px;
            text-align: center;
        }
        
        @media (max-width: 480px) {
            .mobile-controls {
                flex-wrap: wrap;
            }
            
            .control-btn {
                min-width: calc(50% - 5px);
            }
        }
    </style>
</head>

<body>
    <!-- Mobile Header -->
    <div class="mobile-header">
        <div class="search-container">
            <input type="text" class="search-input" placeholder="🔍 Tìm địa điểm hoặc thùng rác..." id="mobileSearch">
            <button class="menu-btn" onclick="toggleMenu()">☰</button>
        </div>
    </div>
    
    <!-- Map Container -->
    <div id="map"></div>
    
    <!-- My Location Button -->
    <button class="floating-btn" onclick="centerOnMyLocation()" title="Vị trí của tôi">
        📍
    </button>
    
    <!-- Route Panel -->
    <div class="route-panel" id="routePanel">
        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 10px;">
            <h4>🧭 Chỉ đường</h4>
            <button onclick="hideRoutePanel()" style="background: none; border: none; font-size: 18px; cursor: pointer;">×</button>
        </div>
        <div id="routeSteps">
            <!-- Route steps will be inserted here -->
        </div>
    </div>
    
    <!-- Mobile Controls -->
    <div class="mobile-controls">
        <button class="control-btn success" onclick="findNearestBin()">
            🗑️ Thùng gần nhất
        </button>
        <button class="control-btn primary" onclick="startNavigation()">
            🧭 Bắt đầu
        </button>
        <button class="control-btn danger" onclick="stopNavigation()">
            ⏹️ Dừng
        </button>
    </div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // Initialize map
        const map = L.map('map').setView([10.77, 106.68], 14);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        // Current location marker
        let currentLocationMarker = null;
        let destinationMarker = null;
        let routeLine = null;
        
        // Sample waste bins
        const wasteBins = [
            {id: 'BIN001', lat: 10.7712, lng: 106.6817, status: 'full'},
            {id: 'BIN002', lat: 10.7689, lng: 106.6798, status: 'normal'},
            {id: 'BIN003', lat: 10.7745, lng: 106.6856, status: 'nearly_full'},
        ];
        
        // Add waste bin markers
        wasteBins.forEach(bin => {
            let color = 'green';
            if (bin.status === 'full') color = 'red';
            else if (bin.status === 'nearly_full') color = 'orange';
            
            const marker = L.marker([bin.lat, bin.lng], {
                icon: L.divIcon({
                    html: `<div style="background-color: ${color}; width: 20px; height: 20px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">🗑️</div>`,
                    className: 'waste-bin-marker',
                    iconSize: [24, 24]
                })
            }).addTo(map);
            
            marker.bindPopup(`
                <div style="min-width: 200px;">
                    <h4>🗑️ Thùng rác #${bin.id}</h4>
                    <p><b>Trạng thái:</b> ${bin.status}</p>
                    <button onclick="routeToBin(${bin.lat}, ${bin.lng})" style="background: #4285f4; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 10px;">
                        🧭 Chỉ đường đến đây
                    </button>
                </div>
            `);
        });
        
        // Add current location
        function addCurrentLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(position => {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    
                    if (currentLocationMarker) {
                        map.removeLayer(currentLocationMarker);
                    }
                    
                    currentLocationMarker = L.marker([lat, lng], {
                        icon: L.divIcon({
                            html: '<div style="background-color: #4285f4; width: 16px; height: 16px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.3);"></div>',
                            className: 'current-location-marker',
                            iconSize: [22, 22]
                        })
                    }).addTo(map);
                    
                    currentLocationMarker.bindPopup('📍 Vị trí hiện tại của bạn');
                });
            }
        }
        
        // Center on current location
        function centerOnMyLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(position => {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    map.setView([lat, lng], 16);
                    addCurrentLocation();
                });
            }
        }
        
        // Find nearest bin
        function findNearestBin() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(position => {
                    const userLat = position.coords.latitude;
                    const userLng = position.coords.longitude;
                    
                    let nearestBin = null;
                    let minDistance = Infinity;
                    
                    wasteBins.forEach(bin => {
                        const distance = Math.sqrt(
                            Math.pow(bin.lat - userLat, 2) + Math.pow(bin.lng - userLng, 2)
                        );
                        
                        if (distance < minDistance) {
                            minDistance = distance;
                            nearestBin = bin;
                        }
                    });
                    
                    if (nearestBin) {
                        routeToBin(nearestBin.lat, nearestBin.lng);
                    }
                });
            }
        }
        
        // Route to bin
        function routeToBin(lat, lng) {
            if (destinationMarker) {
                map.removeLayer(destinationMarker);
            }
            
            destinationMarker = L.marker([lat, lng], {
                icon: L.divIcon({
                    html: '<div style="background-color: #34a853; width: 20px; height: 20px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">🏁</div>',
                    className: 'destination-marker',
                    iconSize: [24, 24]
                })
            }).addTo(map);
            
            showRoutePanel();
            
            // Simulate route drawing
            if (routeLine) {
                map.removeLayer(routeLine);
            }
            
            // Get current position and draw route
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(position => {
                    const startLat = position.coords.latitude;
                    const startLng = position.coords.longitude;
                    
                    routeLine = L.polyline([
                        [startLat, startLng],
                        [(startLat + lat) / 2, (startLng + lng) / 2],
                        [lat, lng]
                    ], {
                        color: '#4285f4',
                        weight: 4,
                        opacity: 0.8
                    }).addTo(map);
                    
                    map.fitBounds(routeLine.getBounds(), {padding: [50, 50]});
                });
            }
        }
        
        function showRoutePanel() {
            const panel = document.getElementById('routePanel');
            const steps = document.getElementById('routeSteps');
            
            steps.innerHTML = `
                <div class="route-step">
                    <div class="route-icon">🚶</div>
                    <div>
                        <div style="font-weight: 600;">Bắt đầu di chuyển</div>
                        <div style="font-size: 12px; color: #666;">Đi thẳng 200m</div>
                    </div>
                </div>
                <div class="route-step">
                    <div class="route-icon">↩️</div>
                    <div>
                        <div style="font-weight: 600;">Rẽ trái</div>
                        <div style="font-size: 12px; color: #666;">Rẽ trái và đi 150m</div>
                    </div>
                </div>
                <div class="route-step">
                    <div class="route-icon">🏁</div>
                    <div>
                        <div style="font-weight: 600;">Đến nơi</div>
                        <div style="font-size: 12px; color: #666;">Thùng rác ở bên phải</div>
                    </div>
                </div>
            `;
            
            panel.classList.add('show');
        }
        
        function hideRoutePanel() {
            document.getElementById('routePanel').classList.remove('show');
        }
        
        function startNavigation() {
            if (routeLine) {
                alert('🧭 Bắt đầu chỉ đường! Thực hiện theo hướng dẫn trên màn hình.');
                showRoutePanel();
            } else {
                alert('Vui lòng chọn điểm đến trước!');
            }
        }
        
        function stopNavigation() {
            if (routeLine) {
                map.removeLayer(routeLine);
                routeLine = null;
            }
            
            if (destinationMarker) {
                map.removeLayer(destinationMarker);
                destinationMarker = null;
            }
            
            hideRoutePanel();
        }
        
        function toggleMenu() {
            alert('Menu: Cài đặt, Lịch sử, Báo cáo, Trợ giúp');
        }
        
        // Search functionality
        document.getElementById('mobileSearch').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const query = this.value.toLowerCase();
                
                // Simple search implementation
                const locations = {
                    'home': [10.77, 106.68],
                    'office': [10.775, 106.685],
                    'market': [10.765, 106.675]
                };
                
                for (let [name, coords] of Object.entries(locations)) {
                    if (name.includes(query) || query.includes(name)) {
                        map.setView(coords, 16);
                        return;
                    }
                }
                
                // Search for bin ID
                const bin = wasteBins.find(b => b.id.toLowerCase().includes(query));
                if (bin) {
                    map.setView([bin.lat, bin.lng], 16);
                    return;
                }
                
                alert('Không tìm thấy: ' + query);
            }
        });
        
        // Initialize current location on load
        addCurrentLocation();
        
        // Handle map clicks
        map.on('click', function(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            
            routeToBin(lat, lng);
        });
        
        // Prevent zoom on double tap (iOS Safari)
        let lastTouchEnd = 0;
        document.addEventListener('touchend', function (event) {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
    </script>
</body>
</html>
    """
    
    with open('mobile_waste_app.html', 'w', encoding='utf-8') as f:
        f.write(mobile_html)
    
    return os.path.abspath('mobile_waste_app.html')


def main():
    """Demo Web Map Interface"""
    print("🌐 Creating Web-based Map Interface...")
    
    # Initialize system
    routing_system = create_sample_data()  # Đây trả về SmartRoutingSystem đã có sẵn bins và roads
    
    # Create web map
    web_map = WebMapInterface(routing_system)
    
    print("📍 Creating enhanced web map...")
    map_path = web_map.create_enhanced_web_map()
    
    print(f"✅ Map created: {map_path}")
    
    # Create mobile version
    print("📱 Creating mobile app interface...")
    mobile_path = create_mobile_app_interface()
    print(f"✅ Mobile app created: {mobile_path}")
    
    # Try to open in browser
    try:
        print("🌐 Opening in browser...")
        web_map.open_in_browser(map_path)
        
        print("\n🎯 Features available:")
        print("- 🔍 Search locations")  
        print("- 🗑️ Interactive waste bins")
        print("- 🧭 Turn-by-turn directions")
        print("- 🚦 Traffic information")
        print("- 📱 Mobile-friendly interface")
        print("- 📍 GPS location tracking")
        
    except Exception as e:
        print(f"⚠️ Could not open browser: {e}")
        print(f"📁 Please open manually: {map_path}")
    
    return map_path, mobile_path


if __name__ == "__main__":
    main()
