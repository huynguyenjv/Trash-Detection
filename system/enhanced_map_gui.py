"""
Enhanced Interactive Map GUI - Giống Google Maps
Tính năng:
- Zoom in/out
- Pan (kéo thả bản đồ)  
- Turn-by-turn directions
- Search locations
- Real-time navigation
- Voice guidance (text-to-speech)

Author: Smart Waste Management System
Date: August 2025
"""

import os
import matplotlib
# Auto-detect best backend for Linux
if 'DISPLAY' in os.environ:
    try:
        import tkinter
        matplotlib.use('TkAgg')
    except ImportError:
        try:
            import PyQt5
            matplotlib.use('Qt5Agg')
        except ImportError:
            matplotlib.use('Agg')
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox, Slider
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, asdict
import threading
import math

from smart_routing_system import (
    SmartRoutingSystem, GPSCoordinate, WasteType,
    PathfindingResult, create_sample_data, BinStatus
)


@dataclass
class NavigationStep:
    """Bước chỉ đường"""
    instruction: str
    distance: float  # meters
    direction: str   # "left", "right", "straight", "u-turn"
    coordinates: GPSCoordinate
    estimated_time: int  # seconds


@dataclass
class MapViewport:
    """Khung nhìn bản đồ"""
    center_lat: float
    center_lng: float
    zoom_level: int = 10  # 1-20, như Google Maps
    width_deg: float = 0.1
    height_deg: float = 0.1


class EnhancedMapGUI:
    """Enhanced Map GUI với tính năng như Google Maps"""
    
    def __init__(self, routing_system: SmartRoutingSystem):
        self.routing_system = routing_system
        self.fig = None
        self.ax_map = None
        self.ax_controls = None
        
        # Map state
        self.viewport = MapViewport(10.77, 106.68)
        self.current_position = GPSCoordinate(10.77, 106.68)
        self.destination = None
        self.current_route = None
        self.navigation_steps: List[NavigationStep] = []
        self.current_step_index = 0
        
        # UI elements
        self.widgets = {}
        self.markers = {}
        self.route_line = None
        
        # Navigation state
        self.is_navigating = False
        self.navigation_thread = None
        self.last_position_update = time.time()
        
        # Map layers
        self.show_traffic = True
        self.show_bins = True
        self.show_route = True
        
        # Colors
        self.colors = {
            'road': '#FFFFFF',
            'traffic_good': '#00FF00',
            'traffic_medium': '#FFFF00', 
            'traffic_bad': '#FF0000',
            'water': '#87CEEB',
            'park': '#90EE90',
            'building': '#D3D3D3',
            'route': '#1E90FF',
            'current_pos': '#FF0000',
            'destination': '#00AA00',
            'waste_bin': '#FFA500'
        }
    
    def create_enhanced_map(self) -> plt.Figure:
        """Tạo giao diện bản đồ nâng cao"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('🗺️ Smart Waste Management - Navigation System', fontsize=16, fontweight='bold')
        
        # Layout: Map (70%) + Controls (30%)
        gs = self.fig.add_gridspec(2, 3, 
                                  width_ratios=[2.5, 1, 0.5],
                                  height_ratios=[1, 0.3])
        
        # Main map area
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map.set_title('🌍 Interactive Map', fontweight='bold')
        
        # Controls panel
        self.ax_controls = self.fig.add_subplot(gs[0, 1])
        self.ax_controls.set_title('🎮 Controls', fontweight='bold')
        self.ax_controls.axis('off')
        
        # Navigation panel
        self.ax_nav = self.fig.add_subplot(gs[1, 1])
        self.ax_nav.set_title('🧭 Navigation', fontweight='bold')
        self.ax_nav.axis('off')
        
        # Info panel  
        self.ax_info = self.fig.add_subplot(gs[:, 2])
        self.ax_info.set_title('ℹ️ Info', fontweight='bold')
        self.ax_info.axis('off')
        
        self._setup_map_controls()
        self._draw_base_map()
        self._setup_event_handlers()
        
        return self.fig
    
    def _setup_map_controls(self):
        """Thiết lập các controls"""
        # Zoom controls
        ax_zoom_in = plt.axes([0.02, 0.85, 0.08, 0.05])
        self.widgets['zoom_in'] = Button(ax_zoom_in, '🔍+', color='lightblue')
        self.widgets['zoom_in'].on_clicked(self._zoom_in)
        
        ax_zoom_out = plt.axes([0.02, 0.78, 0.08, 0.05])
        self.widgets['zoom_out'] = Button(ax_zoom_out, '🔍-', color='lightblue')
        self.widgets['zoom_out'].on_clicked(self._zoom_out)
        
        # My Location button
        ax_my_location = plt.axes([0.02, 0.71, 0.08, 0.05])
        self.widgets['my_location'] = Button(ax_my_location, '📍 Me', color='lightgreen')
        self.widgets['my_location'].on_clicked(self._center_on_me)
        
        # Search box
        ax_search = plt.axes([0.12, 0.95, 0.3, 0.04])
        self.widgets['search'] = TextBox(ax_search, '🔍 Search: ', initial='Search location or address...')
        self.widgets['search'].on_submit(self._search_location)
        
        # Layer toggles
        ax_traffic = plt.axes([0.45, 0.95, 0.1, 0.04])
        self.widgets['traffic'] = Button(ax_traffic, '🚦 Traffic', color='yellow' if self.show_traffic else 'lightgray')
        self.widgets['traffic'].on_clicked(self._toggle_traffic)
        
        ax_bins = plt.axes([0.56, 0.95, 0.1, 0.04])
        self.widgets['bins'] = Button(ax_bins, '🗑️ Bins', color='orange' if self.show_bins else 'lightgray')
        self.widgets['bins'].on_clicked(self._toggle_bins)
        
        # Navigation controls
        ax_start_nav = plt.axes([0.68, 0.95, 0.1, 0.04])
        self.widgets['start_nav'] = Button(ax_start_nav, '🧭 Navigate', color='lightcoral')
        self.widgets['start_nav'].on_clicked(self._start_navigation)
        
        ax_stop_nav = plt.axes([0.79, 0.95, 0.1, 0.04])
        self.widgets['stop_nav'] = Button(ax_stop_nav, '⏹️ Stop', color='lightgray')
        self.widgets['stop_nav'].on_clicked(self._stop_navigation)
        
        # Direction panel buttons in controls area
        self._draw_control_buttons()
    
    def _draw_control_buttons(self):
        """Vẽ các nút điều khiển trong panel"""
        self.ax_controls.clear()
        self.ax_controls.set_title('🎮 Controls', fontweight='bold')
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        
        # Quick actions
        buttons_info = [
            (0.1, 0.8, 0.8, 0.15, '🏠 Go Home', 'home'),
            (0.1, 0.6, 0.8, 0.15, '🗑️ Nearest Bin', 'nearest_bin'),
            (0.1, 0.4, 0.8, 0.15, '📊 Route Stats', 'stats'),
            (0.1, 0.2, 0.8, 0.15, '💾 Save Position', 'save_pos'),
        ]
        
        for x, y, w, h, text, action in buttons_info:
            rect = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.02",
                facecolor='lightblue',
                edgecolor='navy',
                linewidth=1
            )
            self.ax_controls.add_patch(rect)
            self.ax_controls.text(x + w/2, y + h/2, text, 
                                ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _draw_base_map(self):
        """Vẽ bản đồ cơ bản"""
        self.ax_map.clear()
        
        # Calculate bounds based on viewport
        lat_min = self.viewport.center_lat - self.viewport.height_deg/2
        lat_max = self.viewport.center_lat + self.viewport.height_deg/2
        lng_min = self.viewport.center_lng - self.viewport.width_deg/2
        lng_max = self.viewport.center_lng + self.viewport.width_deg/2
        
        self.ax_map.set_xlim(lng_min, lng_max)
        self.ax_map.set_ylim(lat_min, lat_max)
        
        # Draw grid (streets)
        self._draw_street_grid()
        
        # Draw POIs
        if self.show_bins:
            self._draw_waste_bins()
            
        if self.show_traffic:
            self._draw_traffic_info()
        
        # Draw current position
        self._draw_current_position()
        
        # Draw destination if set
        if self.destination:
            self._draw_destination()
        
        # Draw route if available
        if self.current_route and self.show_route:
            self._draw_route()
        
        self.ax_map.set_xlabel('Longitude (°E)')
        self.ax_map.set_ylabel('Latitude (°N)')
        self.ax_map.grid(True, alpha=0.3)
    
    def _draw_street_grid(self):
        """Vẽ lưới đường phố"""
        lat_min, lat_max = self.ax_map.get_ylim()
        lng_min, lng_max = self.ax_map.get_xlim()
        
        # Main roads (vertical)
        for lng in np.linspace(lng_min, lng_max, 8):
            self.ax_map.plot([lng, lng], [lat_min, lat_max], 
                           color=self.colors['road'], linewidth=3, alpha=0.8, zorder=1)
        
        # Main roads (horizontal)  
        for lat in np.linspace(lat_min, lat_max, 6):
            self.ax_map.plot([lng_min, lng_max], [lat, lat],
                           color=self.colors['road'], linewidth=3, alpha=0.8, zorder=1)
        
        # Add some curved roads for realism
        theta = np.linspace(0, 2*np.pi, 100)
        center_lat = (lat_min + lat_max) / 2
        center_lng = (lng_min + lng_max) / 2
        
        # Ring road
        radius_lat = (lat_max - lat_min) * 0.3
        radius_lng = (lng_max - lng_min) * 0.3
        
        ring_lat = center_lat + radius_lat * np.sin(theta)
        ring_lng = center_lng + radius_lng * np.cos(theta)
        
        self.ax_map.plot(ring_lng, ring_lat, 
                        color=self.colors['road'], linewidth=2, alpha=0.7, zorder=1)
    
    def _draw_waste_bins(self):
        """Vẽ các thùng rác"""
        # Truy cập chính xác: waste_bins là Dict[str, WasteBin] 
        for bin_data in self.routing_system.waste_bins.values():
            color = self.colors['waste_bin']
            if bin_data.status == BinStatus.FULL:
                color = 'red'
            elif bin_data.status == BinStatus.NEAR_FULL:
                color = 'orange'
            elif bin_data.status == BinStatus.OK:
                color = 'green'
            
            # Draw bin marker
            self.ax_map.scatter(bin_data.location.lng, bin_data.location.lat,
                              c=color, s=100, marker='s', alpha=0.8, zorder=5)
            
            # Add label
            self.ax_map.annotate(f'🗑️{bin_data.id}', 
                               (bin_data.location.lng, bin_data.location.lat),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
    
    def _draw_traffic_info(self):
        """Vẽ thông tin giao thông"""
        # Simulate traffic on roads
        lat_min, lat_max = self.ax_map.get_ylim()
        lng_min, lng_max = self.ax_map.get_xlim()
        
        # Random traffic segments
        np.random.seed(42)  # For consistent display
        
        for _ in range(15):
            start_lng = np.random.uniform(lng_min, lng_max)
            start_lat = np.random.uniform(lat_min, lat_max)
            
            # Random direction and length
            angle = np.random.uniform(0, 2*np.pi)
            length = np.random.uniform(0.01, 0.03)
            
            end_lng = start_lng + length * np.cos(angle)
            end_lat = start_lat + length * np.sin(angle)
            
            # Random traffic condition
            traffic_level = np.random.choice(['good', 'medium', 'bad'], p=[0.6, 0.3, 0.1])
            color = self.colors[f'traffic_{traffic_level}']
            width = 4 if traffic_level == 'bad' else 2
            
            self.ax_map.plot([start_lng, end_lng], [start_lat, end_lat],
                           color=color, linewidth=width, alpha=0.7, zorder=2)
    
    def _draw_current_position(self):
        """Vẽ vị trí hiện tại"""
        # Main position marker
        self.ax_map.scatter(self.current_position.lng, self.current_position.lat,
                          c=self.colors['current_pos'], s=200, marker='o', 
                          edgecolors='white', linewidth=2, zorder=10)
        
        # Direction indicator (if moving)
        # This could show movement direction based on GPS bearing
        
        # Accuracy circle
        accuracy_radius = 0.001  # ~100m at this scale
        circle = Circle((self.current_position.lng, self.current_position.lat),
                       accuracy_radius, fill=False, color=self.colors['current_pos'],
                       alpha=0.3, zorder=9)
        self.ax_map.add_patch(circle)
    
    def _draw_destination(self):
        """Vẽ điểm đích"""
        self.ax_map.scatter(self.destination.lng, self.destination.lat,
                          c=self.colors['destination'], s=150, marker='*',
                          edgecolors='white', linewidth=2, zorder=10)
        
        self.ax_map.annotate('🏁 Destination', 
                           (self.destination.lng, self.destination.lat),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    def _draw_route(self):
        """Vẽ tuyến đường"""
        if not self.current_route or not self.current_route.path:
            return
        
        # Extract coordinates from path
        lats = [coord.lat for coord in self.current_route.path]
        lngs = [coord.lng for coord in self.current_route.path]
        
        # Draw main route line
        self.route_line = self.ax_map.plot(lngs, lats, 
                                          color=self.colors['route'], 
                                          linewidth=4, alpha=0.8, zorder=6)[0]
        
        # Add direction arrows
        self._add_route_arrows(lngs, lats)
        
        # Highlight current navigation segment
        if self.is_navigating and self.current_step_index < len(self.navigation_steps):
            current_step = self.navigation_steps[self.current_step_index]
            self.ax_map.scatter(current_step.coordinates.lng, current_step.coordinates.lat,
                              c='yellow', s=100, marker='>', zorder=11)
    
    def _add_route_arrows(self, lngs: List[float], lats: List[float]):
        """Thêm mũi tên chỉ hướng trên route"""
        if len(lngs) < 2:
            return
            
        # Add arrows every few points
        for i in range(0, len(lngs)-1, max(1, len(lngs)//8)):
            dx = lngs[i+1] - lngs[i]
            dy = lats[i+1] - lats[i]
            
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Avoid zero-length arrows
                self.ax_map.annotate('', xy=(lngs[i+1], lats[i+1]), 
                                   xytext=(lngs[i], lats[i]),
                                   arrowprops=dict(arrowstyle='->', 
                                                 color=self.colors['route'], 
                                                 lw=2))
    
    def _update_navigation_panel(self):
        """Cập nhật panel điều hướng"""
        self.ax_nav.clear()
        self.ax_nav.set_title('🧭 Navigation', fontweight='bold')
        self.ax_nav.set_xlim(0, 1)
        self.ax_nav.set_ylim(0, 1)
        
        if not self.is_navigating or not self.navigation_steps:
            self.ax_nav.text(0.5, 0.5, 'No active navigation', 
                           ha='center', va='center', fontsize=12)
            return
        
        if self.current_step_index >= len(self.navigation_steps):
            self.ax_nav.text(0.5, 0.5, '🎉 Arrived!', 
                           ha='center', va='center', fontsize=14, 
                           fontweight='bold', color='green')
            return
        
        current_step = self.navigation_steps[self.current_step_index]
        
        # Current instruction
        self.ax_nav.text(0.05, 0.8, f"📍 {current_step.instruction}", 
                        fontsize=12, fontweight='bold', wrap=True)
        
        # Distance and time
        self.ax_nav.text(0.05, 0.6, f"📏 {current_step.distance:.0f}m", fontsize=10)
        self.ax_nav.text(0.05, 0.5, f"⏱️ {current_step.estimated_time//60}min {current_step.estimated_time%60}s", 
                        fontsize=10)
        
        # Progress
        progress = (self.current_step_index + 1) / len(self.navigation_steps)
        rect = Rectangle((0.05, 0.3), 0.9, 0.1, facecolor='lightgray')
        self.ax_nav.add_patch(rect)
        
        progress_rect = Rectangle((0.05, 0.3), 0.9 * progress, 0.1, facecolor='green')
        self.ax_nav.add_patch(progress_rect)
        
        self.ax_nav.text(0.5, 0.35, f"{self.current_step_index + 1}/{len(self.navigation_steps)}", 
                        ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _update_info_panel(self):
        """Cập nhật panel thông tin"""
        self.ax_info.clear()
        self.ax_info.set_title('ℹ️ Info', fontweight='bold')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        
        info_lines = [
            f"📍 Lat: {self.current_position.lat:.4f}",
            f"📍 Lng: {self.current_position.lng:.4f}",
            f"🔍 Zoom: {self.viewport.zoom_level}",
            f"⏰ {datetime.now().strftime('%H:%M:%S')}",
        ]
        
        if self.current_route:
            info_lines.extend([
                "",
                f"🛣️ Route Distance: {self.current_route.total_distance:.1f}km",
                f"⏱️ ETA: {self.current_route.estimated_time//60}min",
                f"⛽ Fuel: {self.current_route.fuel_cost:.1f}L"
            ])
        
        # Display info
        for i, line in enumerate(info_lines):
            y_pos = 0.9 - (i * 0.08)
            if y_pos > 0:
                self.ax_info.text(0.05, y_pos, line, fontsize=9, verticalalignment='top')
    
    def _setup_event_handlers(self):
        """Thiết lập event handlers"""
        # Map click for destination
        self.fig.canvas.mpl_connect('button_press_event', self._on_map_click)
        
        # Mouse wheel for zoom
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        # Drag to pan
        self.fig.canvas.mpl_connect('button_press_event', self._on_drag_start)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_drag_move)
        self.fig.canvas.mpl_connect('button_release_event', self._on_drag_end)
        
        self._dragging = False
        self._drag_start = None
    
    def _on_map_click(self, event):
        """Xử lý click trên bản đồ"""
        if event.inaxes == self.ax_map and event.dblclick:
            # Double click to set destination
            self.destination = GPSCoordinate(event.ydata, event.xdata)
            print(f"🎯 Destination set: {self.destination.lat:.4f}, {self.destination.lng:.4f}")
            self._calculate_route()
            self._redraw_map()
    
    def _on_scroll(self, event):
        """Xử lý zoom bằng mouse wheel"""
        if event.inaxes == self.ax_map:
            if event.button == 'up':
                self._zoom_in(None)
            elif event.button == 'down':
                self._zoom_out(None)
    
    def _on_drag_start(self, event):
        """Bắt đầu kéo bản đồ"""
        if event.inaxes == self.ax_map and not event.dblclick:
            self._dragging = True
            self._drag_start = (event.xdata, event.ydata)
    
    def _on_drag_move(self, event):
        """Kéo bản đồ"""
        if self._dragging and event.inaxes == self.ax_map and self._drag_start:
            dx = self._drag_start[0] - event.xdata
            dy = self._drag_start[1] - event.ydata
            
            self.viewport.center_lng += dx
            self.viewport.center_lat += dy
            
            self._redraw_map()
    
    def _on_drag_end(self, event):
        """Kết thúc kéo bản đồ"""
        self._dragging = False
        self._drag_start = None
    
    # Control event handlers
    def _zoom_in(self, event):
        """Phóng to"""
        if self.viewport.zoom_level < 18:
            self.viewport.zoom_level += 1
            self.viewport.width_deg *= 0.7
            self.viewport.height_deg *= 0.7
            self._redraw_map()
    
    def _zoom_out(self, event):
        """Thu nhỏ"""
        if self.viewport.zoom_level > 3:
            self.viewport.zoom_level -= 1
            self.viewport.width_deg *= 1.4
            self.viewport.height_deg *= 1.4
            self._redraw_map()
    
    def _center_on_me(self, event):
        """Căn giữa vào vị trí hiện tại"""
        self.viewport.center_lat = self.current_position.lat
        self.viewport.center_lng = self.current_position.lng
        self._redraw_map()
    
    def _toggle_traffic(self, event):
        """Bật/tắt hiển thị giao thông"""
        self.show_traffic = not self.show_traffic
        self.widgets['traffic'].color = 'yellow' if self.show_traffic else 'lightgray'
        self._redraw_map()
    
    def _toggle_bins(self, event):
        """Bật/tắt hiển thị thùng rác"""
        self.show_bins = not self.show_bins
        self.widgets['bins'].color = 'orange' if self.show_bins else 'lightgray'
        self._redraw_map()
    
    def _search_location(self, text):
        """Tìm kiếm địa điểm"""
        # Simple search simulation - in real app, use geocoding API
        search_locations = {
            'home': GPSCoordinate(10.77, 106.68),
            'office': GPSCoordinate(10.775, 106.685),
            'market': GPSCoordinate(10.765, 106.675),
            'hospital': GPSCoordinate(10.778, 106.672),
            'school': GPSCoordinate(10.773, 106.690)
        }
        
        text_lower = text.lower()
        for name, coord in search_locations.items():
            if name in text_lower:
                self.viewport.center_lat = coord.lat
                self.viewport.center_lng = coord.lng
                print(f"📍 Found: {name} at {coord.lat:.4f}, {coord.lng:.4f}")
                self._redraw_map()
                return
        
        print(f"❌ Location not found: {text}")
    
    def _calculate_route(self):
        """Tính toán tuyến đường"""
        if not self.destination:
            return
            
        print("🔄 Calculating route...")
        result = self.routing_system.find_optimal_route(
            self.current_position, self.destination
        )
        
        if result.success:
            self.current_route = result
            self._generate_navigation_steps()
            print(f"✅ Route calculated: {result.total_distance:.1f}km, {result.estimated_time//60}min")
        else:
            print("❌ Route calculation failed")
    
    def _generate_navigation_steps(self):
        """Tạo các bước chỉ đường"""
        if not self.current_route or not self.current_route.path:
            return
        
        self.navigation_steps = []
        path = self.current_route.path
        
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            # Calculate distance
            distance = self._calculate_distance(current, next_point) * 1000  # Convert to meters
            
            # Determine direction
            direction = self._get_direction(current, next_point, 
                                          path[i - 1] if i > 0 else current)
            
            # Generate instruction
            instruction = self._generate_instruction(direction, distance, i == 0)
            
            step = NavigationStep(
                instruction=instruction,
                distance=distance,
                direction=direction,
                coordinates=next_point,
                estimated_time=int(distance / 1.4)  # ~5 km/h walking speed
            )
            
            self.navigation_steps.append(step)
    
    def _calculate_distance(self, coord1: GPSCoordinate, coord2: GPSCoordinate) -> float:
        """Tính khoảng cách giữa 2 điểm (Haversine formula)"""
        R = 6371  # Earth radius in km
        
        lat1, lng1 = math.radians(coord1.lat), math.radians(coord1.lng)
        lat2, lng2 = math.radians(coord2.lat), math.radians(coord2.lng)
        
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _get_direction(self, current: GPSCoordinate, next_point: GPSCoordinate, 
                      prev_point: GPSCoordinate) -> str:
        """Xác định hướng di chuyển"""
        # Calculate bearings
        bearing1 = self._calculate_bearing(prev_point, current)
        bearing2 = self._calculate_bearing(current, next_point)
        
        # Calculate turn angle
        angle_diff = (bearing2 - bearing1) % 360
        
        if angle_diff < 45 or angle_diff > 315:
            return "straight"
        elif 45 <= angle_diff < 135:
            return "right"
        elif 135 <= angle_diff < 225:
            return "u-turn"
        else:
            return "left"
    
    def _calculate_bearing(self, coord1: GPSCoordinate, coord2: GPSCoordinate) -> float:
        """Tính bearing giữa 2 điểm"""
        lat1, lng1 = math.radians(coord1.lat), math.radians(coord1.lng)
        lat2, lng2 = math.radians(coord2.lat), math.radians(coord2.lng)
        
        dlng = lng2 - lng1
        
        y = math.sin(dlng) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlng)
        
        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360
    
    def _generate_instruction(self, direction: str, distance: float, is_first: bool) -> str:
        """Tạo hướng dẫn bằng tiếng Việt"""
        if is_first:
            return f"Bắt đầu di chuyển {distance:.0f}m về phía trước"
        
        direction_map = {
            "straight": f"Tiếp tục thẳng {distance:.0f}m",
            "left": f"Rẽ trái và đi {distance:.0f}m", 
            "right": f"Rẽ phải và đi {distance:.0f}m",
            "u-turn": f"Quay đầu và đi {distance:.0f}m"
        }
        
        return direction_map.get(direction, f"Di chuyển {distance:.0f}m")
    
    def _start_navigation(self, event):
        """Bắt đầu chỉ đường"""
        if not self.destination or not self.current_route:
            print("❌ Please set destination first")
            return
            
        self.is_navigating = True
        self.current_step_index = 0
        
        # Start navigation thread
        self.navigation_thread = threading.Thread(target=self._navigation_loop, daemon=True)
        self.navigation_thread.start()
        
        print("🧭 Navigation started!")
        self._redraw_map()
    
    def _stop_navigation(self, event):
        """Dừng chỉ đường"""
        self.is_navigating = False
        print("⏹️ Navigation stopped")
        self._redraw_map()
    
    def _navigation_loop(self):
        """Vòng lặp chỉ đường"""
        while self.is_navigating and self.current_step_index < len(self.navigation_steps):
            current_step = self.navigation_steps[self.current_step_index]
            
            # Check if reached current waypoint
            distance_to_waypoint = self._calculate_distance(
                self.current_position, current_step.coordinates
            ) * 1000  # Convert to meters
            
            if distance_to_waypoint < 50:  # Within 50m
                print(f"📍 Reached waypoint: {current_step.instruction}")
                self.current_step_index += 1
                
                # Update display
                self._update_navigation_panel()
                self.fig.canvas.draw()
            
            time.sleep(2)  # Check every 2 seconds
        
        if self.current_step_index >= len(self.navigation_steps):
            print("🎉 Navigation completed! You have arrived at your destination.")
            self.is_navigating = False
    
    def _redraw_map(self):
        """Vẽ lại toàn bộ bản đồ"""
        self._draw_base_map()
        self._update_navigation_panel()
        self._update_info_panel()
        self.fig.canvas.draw()
    
    def update_position(self, new_position: GPSCoordinate):
        """Cập nhật vị trí hiện tại"""
        self.current_position = new_position
        self.last_position_update = time.time()
        
        # Auto-center if navigating
        if self.is_navigating:
            self.viewport.center_lat = new_position.lat
            self.viewport.center_lng = new_position.lng
        
        self._redraw_map()


def main():
    """Demo Enhanced Map GUI"""
    print("🗺️ Starting Enhanced Map GUI Demo...")
    
    # Initialize system 
    routing_system = create_sample_data()  # Trả về SmartRoutingSystem với data
    
    # Create enhanced GUI
    gui = EnhancedMapGUI(routing_system)
    fig = gui.create_enhanced_map()
    
    # Instructions
    print("\n🎮 Controls:")
    print("- Double-click: Set destination")
    print("- Mouse wheel: Zoom in/out") 
    print("- Drag: Pan map")
    print("- Buttons: Various functions")
    print("- Search: Type location name")
    
    # Show map
    backend = matplotlib.get_backend()
    print(f"🖥️ Using backend: {backend}")
    
    try:
        if backend != 'Agg':
            plt.show()
        else:
            fig.savefig('enhanced_map.png', dpi=150, bbox_inches='tight')
            print("💾 Map saved as 'enhanced_map.png'")
    except Exception as e:
        print(f"⚠️ Display error: {e}")
        fig.savefig('enhanced_map.png', dpi=150, bbox_inches='tight')
        print("💾 Map saved as 'enhanced_map.png'")
    
    print("✅ Demo completed!")


if __name__ == "__main__":
    main()
