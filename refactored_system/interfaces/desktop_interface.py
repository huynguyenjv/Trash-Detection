"""
Desktop Interface - Giao diện desktop với Matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
import numpy as np
from typing import List, Optional, Tuple, Callable

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GPSCoordinate, WasteBin, PathfindingResult
from core.enums import BinStatus
from core.routing_engine import RoutingEngine


class DesktopMapInterface:
    """Desktop GUI interface using Matplotlib"""
    
    def __init__(self, routing_engine: RoutingEngine):
        self.routing_engine = routing_engine
        self.fig = None
        self.ax = None
        self.current_position = GPSCoordinate(10.77, 106.68)
        self.waste_bins: List[WasteBin] = []
        self.current_route: Optional[PathfindingResult] = None
        self.selected_bin: Optional[WasteBin] = None
        
        # GUI elements
        self.search_box = None
        self.route_button = None
        self.reset_button = None
        
    def set_waste_bins(self, bins: List[WasteBin]):
        """Set waste bins for display"""
        self.waste_bins = bins
    
    def set_current_position(self, position: GPSCoordinate):
        """Set current position"""
        self.current_position = position
    
    def create_interface(self, figsize: Tuple[int, int] = (14, 10)):
        """Create desktop interface"""
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.suptitle('🗑️ Smart Waste Management System', fontsize=16, fontweight='bold')
        
        # Set up the map area
        self.ax.set_xlim(106.65, 106.71)
        self.ax.set_ylim(10.75, 10.79)
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.grid(True, alpha=0.3)
        
        # Draw components
        self._draw_current_position()
        self._draw_waste_bins()
        self._create_controls()
        self._create_legend()
        
        # Connect events
        self._connect_events()
        
        plt.tight_layout()
        plt.show()
    
    def _draw_current_position(self):
        """Draw current position marker"""
        self.ax.scatter(
            self.current_position.lng, self.current_position.lat,
            c='red', s=200, marker='*', 
            label='Vị trí hiện tại', zorder=10
        )
        
        # Add GPS accuracy circle
        circle = patches.Circle(
            (self.current_position.lng, self.current_position.lat),
            0.0005,  # ~50m
            fill=False, color='red', alpha=0.5, linestyle='--'
        )
        self.ax.add_patch(circle)
    
    def _draw_waste_bins(self):
        """Draw waste bin markers"""
        for bin_data in self.waste_bins:
            # Color based on status
            if bin_data.status == BinStatus.FULL:
                color = 'red'
                marker = 'X'
            elif bin_data.status == BinStatus.NEAR_FULL:
                color = 'orange'
                marker = '^'
            else:
                color = 'green'
                marker = 's'
            
            self.ax.scatter(
                bin_data.location.lng, bin_data.location.lat,
                c=color, s=150, marker=marker, 
                alpha=0.8, edgecolors='black', linewidth=1
            )
            
            # Add bin ID label
            self.ax.annotate(
                f'#{bin_data.id}',
                (bin_data.location.lng, bin_data.location.lat),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold'
            )
    
    def _create_controls(self):
        """Create control buttons and text boxes"""
        # Search box
        search_ax = plt.axes([0.15, 0.02, 0.2, 0.04])
        self.search_box = TextBox(search_ax, 'Tìm kiếm: ', initial='')
        
        # Route button
        route_ax = plt.axes([0.4, 0.02, 0.1, 0.04])
        self.route_button = Button(route_ax, 'Tính đường')
        
        # Reset button
        reset_ax = plt.axes([0.52, 0.02, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Reset')
        
        # Nearest bin button
        nearest_ax = plt.axes([0.64, 0.02, 0.15, 0.04])
        self.nearest_button = Button(nearest_ax, 'Thùng gần nhất')
    
    def _create_legend(self):
        """Create legend"""
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                      markersize=12, label='Vị trí hiện tại'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                      markersize=10, label='Thùng rác bình thường'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', 
                      markersize=10, label='Thùng rác gần đầy'),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                      markersize=10, label='Thùng rác đầy'),
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    def _connect_events(self):
        """Connect event handlers"""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Button callbacks
        self.route_button.on_clicked(self._on_find_route_clicked)
        self.reset_button.on_clicked(self._on_reset_clicked)
        self.nearest_button.on_clicked(self._on_nearest_clicked)
        self.search_box.on_submit(self._on_search_submit)
    
    def _on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
        
        # Find clicked waste bin
        clicked_bin = self._find_bin_at_position(event.xdata, event.ydata)
        
        if clicked_bin:
            self.selected_bin = clicked_bin
            print(f"🗑️ Đã chọn thùng rác #{clicked_bin.id}")
            print(f"   Trạng thái: {clicked_bin.status.value}")
            print(f"   Dung lượng: {clicked_bin.fill_percentage:.1f}%")
            print(f"   Vị trí: {clicked_bin.location.lat:.6f}, {clicked_bin.location.lng:.6f}")
        else:
            # Click on empty space - could be destination selection
            destination = GPSCoordinate(event.ydata, event.xdata)
            print(f"📍 Điểm được chọn: {destination.lat:.6f}, {destination.lng:.6f}")
    
    def _find_bin_at_position(self, x: float, y: float) -> Optional[WasteBin]:
        """Find waste bin at given position"""
        threshold = 0.0005  # ~50m tolerance
        
        for bin_data in self.waste_bins:
            distance = abs(bin_data.location.lng - x) + abs(bin_data.location.lat - y)
            if distance < threshold:
                return bin_data
        
        return None
    
    def _on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'r':
            self._reset_view()
        elif event.key == 'f':
            self._find_nearest_bin()
        elif event.key == 'c':
            self._center_on_position()
    
    def _on_find_route_clicked(self, event):
        """Handle find route button click"""
        if not self.selected_bin:
            print("⚠️ Vui lòng chọn thùng rác trước!")
            return
        
        # Calculate route
        route = self.routing_engine.find_path_astar(
            self.current_position, 
            self.selected_bin.location
        )
        
        if route.is_valid:
            self.current_route = route
            self._draw_route(route)
            
            print(f"🛣️ Tuyến đường đến thùng #{self.selected_bin.id}:")
            print(f"   Khoảng cách: {route.total_distance:.2f} km")
            print(f"   Thời gian: {route.total_time:.0f} phút") 
            print(f"   Nhiên liệu: {route.fuel_estimate:.2f} L")
        else:
            print("❌ Không tìm thấy đường đi!")
    
    def _on_reset_clicked(self, event):
        """Handle reset button click"""
        self._reset_view()
        print("🔄 Đã reset bản đồ")
    
    def _on_nearest_clicked(self, event):
        """Handle nearest bin button click"""
        self._find_nearest_bin()
    
    def _on_search_submit(self, text):
        """Handle search box submit"""
        # Simple search implementation
        text = text.lower().strip()
        
        # Search by bin ID
        for bin_data in self.waste_bins:
            if bin_data.id.lower() == text:
                self._center_on_bin(bin_data)
                self.selected_bin = bin_data
                print(f"🔍 Tìm thấy thùng rác #{bin_data.id}")
                return
        
        # Search by status
        status_map = {
            'full': BinStatus.FULL,
            'đầy': BinStatus.FULL,
            'nearly full': BinStatus.NEAR_FULL,
            'gần đầy': BinStatus.NEAR_FULL
        }
        
        if text in status_map:
            matching_bins = [b for b in self.waste_bins if b.status == status_map[text]]
            if matching_bins:
                self._center_on_bin(matching_bins[0])
                print(f"🔍 Tìm thấy {len(matching_bins)} thùng với trạng thái: {text}")
                return
        
        print(f"❌ Không tìm thấy: {text}")
    
    def _draw_route(self, route: PathfindingResult):
        """Draw route on map"""
        if not route.is_valid:
            return
        
        # Clear existing route
        self._clear_route()
        
        # Draw route line
        x_coords = [point.lng for point in route.path]
        y_coords = [point.lat for point in route.path]
        
        self.current_route_line, = self.ax.plot(
            x_coords, y_coords,
            color='blue', linewidth=3, alpha=0.7,
            label=f'Route ({route.total_distance:.1f}km)'
        )
        
        # Add direction arrows
        self._add_route_arrows(route.path)
        
        # Refresh display
        self.fig.canvas.draw()
    
    def _add_route_arrows(self, path: List[GPSCoordinate]):
        """Add arrows to show route direction"""
        for i in range(0, len(path) - 1, max(1, len(path) // 5)):
            start = path[i]
            end = path[i + 1] if i + 1 < len(path) else path[-1]
            
            dx = end.lng - start.lng
            dy = end.lat - start.lat
            
            self.ax.arrow(
                start.lng, start.lat, dx * 0.3, dy * 0.3,
                head_width=0.0002, head_length=0.0002,
                fc='blue', ec='blue', alpha=0.8
            )
    
    def _clear_route(self):
        """Clear existing route from map"""
        if hasattr(self, 'current_route_line') and self.current_route_line:
            self.current_route_line.remove()
            self.current_route_line = None
            
        # Clear arrows (recreate the plot without them)
        # Simple approach: redraw bins
        self.ax.clear()
        self._setup_axes()
        self._draw_current_position()
        self._draw_waste_bins()
        self.fig.canvas.draw()
    
    def _setup_axes(self):
        """Setup axes properties"""
        self.ax.set_xlim(106.65, 106.71)
        self.ax.set_ylim(10.75, 10.79)
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.grid(True, alpha=0.3)
    
    def _reset_view(self):
        """Reset map view"""
        self.selected_bin = None
        self.current_route = None
        self._clear_route()
        self._create_legend()
    
    def _find_nearest_bin(self):
        """Find and select nearest waste bin"""
        if not self.waste_bins:
            print("❌ Không có thùng rác nào!")
            return
        
        nearest_bin = min(
            self.waste_bins,
            key=lambda b: self.current_position.distance_to(b.location)
        )
        
        self.selected_bin = nearest_bin
        self._center_on_bin(nearest_bin)
        
        distance = self.current_position.distance_to(nearest_bin.location)
        print(f"🎯 Thùng rác gần nhất: #{nearest_bin.id}")
        print(f"   Khoảng cách: {distance:.2f} km")
        print(f"   Trạng thái: {nearest_bin.status.value}")
    
    def _center_on_bin(self, bin_data: WasteBin):
        """Center map on waste bin"""
        margin = 0.002
        self.ax.set_xlim(bin_data.location.lng - margin, bin_data.location.lng + margin)
        self.ax.set_ylim(bin_data.location.lat - margin, bin_data.location.lat + margin)
        self.fig.canvas.draw()
    
    def _center_on_position(self):
        """Center map on current position"""
        margin = 0.003
        self.ax.set_xlim(self.current_position.lng - margin, self.current_position.lng + margin)
        self.ax.set_ylim(self.current_position.lat - margin, self.current_position.lat + margin)
        self.fig.canvas.draw()
