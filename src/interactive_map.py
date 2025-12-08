"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Interactive Map - Giao diện tương tác cho hệ thống định tuyến thông minh

Mô tả:
    Module này cung cấp giao diện đồ họa tương tác cho phép người dùng:
    - Click chọn vị trí trên bản đồ
    - Cập nhật tọa độ robot
    - Visualize lộ trình thu gom rác
    - Theo dõi trạng thái thùng rác real-time

Author: Huy Nguyen
Email: huynguyen@example.com
Date: August 2025
Version: 1.0.0
License: MIT
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
from typing import Optional, Tuple, Callable
import numpy as np

from smart_routing_system import (
    SmartRoutingSystem, GPSCoordinate, WasteType, 
    PathfindingResult, MapVisualizer, create_sample_data,
    BinStatus, TrafficCondition
)


class InteractiveMapGUI:
    """Giao diện tương tác cho bản đồ"""
    
    def __init__(self, routing_system: SmartRoutingSystem):
        self.routing_system = routing_system
        self.fig = None
        self.ax = None
        self.current_position_marker = None
        self.current_route_line = None
        self.selected_waste_type = WasteType.PLASTIC
        
        # Callbacks
        self.on_position_changed: Optional[Callable[[GPSCoordinate], None]] = None
        
        # Status
        self.is_interactive = True
        
    def create_interactive_map(self, figsize: Tuple[int, int] = (14, 10)):
        """Tạo bản đồ tương tác"""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Tìm bounding box từ tất cả điểm
        all_coords = []
        all_coords.extend([bin_obj.location for bin_obj in self.routing_system.waste_bins.values()])
        if self.routing_system.current_position:
            all_coords.append(self.routing_system.current_position)
        
        if all_coords:
            lats = [coord.lat for coord in all_coords]
            lngs = [coord.lng for coord in all_coords]
            
            lat_margin = (max(lats) - min(lats)) * 0.15
            lng_margin = (max(lngs) - min(lngs)) * 0.15
            
            self.ax.set_xlim(min(lngs) - lng_margin, max(lngs) + lng_margin)
            self.ax.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
        
        # Vẽ các thành phần cơ bản
        self._draw_road_network()
        self._draw_waste_bins()
        self._draw_current_position()
        
        # Thiết lập giao diện
        self.ax.set_xlabel('Longitude (Kinh độ)')
        self.ax.set_ylabel('Latitude (Vĩ độ)')
        self.ax.set_title('Smart Waste Collection - Interactive Map\n'
                         'Click để chọn vị trí robot mới')
        self.ax.grid(True, alpha=0.3)
        
        # Thêm legend
        self._create_legend()
        
        # Thêm controls
        self._create_controls()
        
        # Kết nối events
        self._connect_events()
        
        plt.tight_layout()
        return self.fig
    
    def _draw_road_network(self):
        """Vẽ mạng lưới đường"""
        for segment in self.routing_system.road_network:
            color = 'gray'
            alpha = 0.4
            linewidth = 2
            
            if segment.traffic_condition == TrafficCondition.HEAVY:
                color = 'red'
                alpha = 0.8
            elif segment.traffic_condition == TrafficCondition.MODERATE:
                color = 'orange'
                alpha = 0.6
            elif segment.is_blocked:
                color = 'black'
                alpha = 0.9
                linewidth = 4
            
            self.ax.plot([segment.start.lng, segment.end.lng],
                        [segment.start.lat, segment.end.lat],
                        color=color, alpha=alpha, linewidth=linewidth)
    
    def _draw_waste_bins(self):
        """Vẽ các bãi rác"""
        for bin_obj in self.routing_system.waste_bins.values():
            color = 'green'
            if bin_obj.status == BinStatus.NEAR_FULL:
                color = 'orange'
            elif bin_obj.status == BinStatus.FULL:
                color = 'red'
            
            # Kiểm tra hỗ trợ loại rác hiện tại
            marker = 'o'
            size = 120
            if self.selected_waste_type in bin_obj.supported_types:
                marker = 's'  # Square cho bins phù hợp
                size = 150
            
            self.ax.scatter(bin_obj.location.lng, bin_obj.location.lat,
                           c=color, s=size, marker=marker, edgecolors='black',
                           linewidth=2, alpha=0.8)
            
            # Thêm label
            self.ax.annotate(f'{bin_obj.id}\n{bin_obj.current_capacity:.0f}kg', 
                            (bin_obj.location.lng, bin_obj.location.lat),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, ha='left')
    
    def _draw_current_position(self):
        """Vẽ vị trí hiện tại của robot"""
        if self.routing_system.current_position:
            if self.current_position_marker:
                self.current_position_marker.remove()
            
            self.current_position_marker = self.ax.scatter(
                self.routing_system.current_position.lng, 
                self.routing_system.current_position.lat,
                c='blue', s=250, marker='^', edgecolors='white',
                linewidth=3, alpha=0.9, zorder=10
            )
            
            # Thêm label
            self.ax.annotate('ROBOT\nCurrent Position', 
                            (self.routing_system.current_position.lng, 
                             self.routing_system.current_position.lat),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, fontweight='bold', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                            ha='left')
    
    def _create_legend(self):
        """Tạo legend cho bản đồ"""
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                      markersize=12, label='Robot Position'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                      markersize=10, label='Waste Bin (Available)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label='Waste Bin (Near Full)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Waste Bin (Full)'),
            plt.Line2D([0], [0], color='gray', linewidth=3, label='Road (Clear)'),
            plt.Line2D([0], [0], color='orange', linewidth=3, label='Road (Traffic)'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Road (Heavy Traffic)'),
            plt.Line2D([0], [0], color='blue', linewidth=4, label='Optimal Route')
        ]
        
        self.ax.legend(handles=legend_elements, 
                      bbox_to_anchor=(1.02, 1), loc='upper left')
    
    def _create_controls(self):
        """Tạo các điều khiển"""
        # Button để tìm đường
        ax_button = plt.axes([0.02, 0.85, 0.12, 0.04])
        self.find_route_btn = Button(ax_button, 'Find Route')
        self.find_route_btn.on_clicked(self._on_find_route_clicked)
        
        # Button để reset
        ax_reset = plt.axes([0.02, 0.80, 0.12, 0.04])
        self.reset_btn = Button(ax_reset, 'Clear Route')
        self.reset_btn.on_clicked(self._on_reset_clicked)
        
        # Button để lưu vị trí
        ax_save = plt.axes([0.02, 0.75, 0.12, 0.04])
        self.save_btn = Button(ax_save, 'Save Position')
        self.save_btn.on_clicked(self._on_save_position)
        
        # TextBox để hiển thị tọa độ
        ax_coords = plt.axes([0.02, 0.65, 0.12, 0.05])
        self.coord_textbox = TextBox(ax_coords, 'Coordinates:\n', 
                                    initial=self._get_current_coords_text())
        
        # Dropdown cho waste type (simplified)
        ax_info = plt.axes([0.02, 0.45, 0.12, 0.15])
        ax_info.text(0.05, 0.9, 'Current Waste Type:', fontsize=10, fontweight='bold')
        ax_info.text(0.05, 0.7, f'{self.selected_waste_type.value.title()}', fontsize=9)
        ax_info.text(0.05, 0.5, '\nInstructions:', fontsize=10, fontweight='bold')
        ax_info.text(0.05, 0.3, '• Click để chọn vị trí robot', fontsize=8)
        ax_info.text(0.05, 0.1, '• Find Route để tìm đường', fontsize=8)
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
    
    def _connect_events(self):
        """Kết nối các events"""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _on_click(self, event):
        """Xử lý click chuột"""
        if event.inaxes != self.ax or not self.is_interactive:
            return
        
        if event.button == 1:  # Left click
            # Cập nhật vị trí robot
            new_position = GPSCoordinate(event.ydata, event.xdata)
            self.routing_system.update_robot_position(new_position)
            
            # Vẽ lại vị trí
            self._draw_current_position()
            
            # Cập nhật textbox
            self.coord_textbox.set_val(self._get_current_coords_text())
            
            # Callback
            if self.on_position_changed:
                self.on_position_changed(new_position)
            
            self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """Xử lý phím bấm"""
        if event.key == 'r':  # Reset
            self._on_reset_clicked(None)
        elif event.key == 'f':  # Find route
            self._on_find_route_clicked(None)
        elif event.key == 's':  # Save
            self._on_save_position(None)
        elif event.key.isdigit():  # Change waste type
            waste_types = list(WasteType)
            idx = int(event.key) % len(waste_types)
            self.selected_waste_type = waste_types[idx]
            self._refresh_map()
    
    def _on_find_route_clicked(self, event):
        """Xử lý click Find Route"""
        if not self.routing_system.current_position:
            print("❌ Chưa chọn vị trí robot!")
            return
        
        # Clear route cũ
        if self.current_route_line:
            for line in self.current_route_line:
                line.remove()
            self.current_route_line = None
        
        # Tìm đường mới
        result = self.routing_system.find_optimal_route(self.selected_waste_type)
        
        if result:
            # Vẽ đường đi
            path_lngs = [coord.lng for coord in result.path]
            path_lats = [coord.lat for coord in result.path]
            
            self.current_route_line = self.ax.plot(path_lngs, path_lats, 
                                                  color='blue', linewidth=4, 
                                                  alpha=0.8, zorder=5)
            
            # Đánh dấu điểm đích
            target_marker = self.ax.scatter(path_lngs[-1], path_lats[-1], 
                                          c='red', s=200, marker='X', 
                                          edgecolors='white', linewidth=2,
                                          zorder=15)
            
            # Cập nhật title
            self.ax.set_title(f'Route to {result.target_bin.id} - {self.selected_waste_type.value.title()}\n'
                             f'Distance: {result.total_distance:.2f}km, '
                             f'ETA: {result.estimated_time:.1f}min, '
                             f'Cost: {result.total_cost:.2f}')
            
            print(f"✅ Route found to {result.target_bin.id}")
            print(f"📏 Distance: {result.total_distance:.2f}km")
            print(f"⏱️ ETA: {result.estimated_time:.1f}min")
            print(f"💰 Cost: {result.total_cost:.2f}")
        else:
            print(f"❌ No route found for {self.selected_waste_type.value}")
        
        self.fig.canvas.draw()
    
    def _on_reset_clicked(self, event):
        """Xử lý click Reset"""
        # Clear route
        if self.current_route_line:
            for line in self.current_route_line:
                line.remove()
            self.current_route_line = None
        
        # Reset title
        self.ax.set_title('Smart Waste Collection - Interactive Map\n'
                         'Click để chọn vị trí robot mới')
        
        self.fig.canvas.draw()
        print("🔄 Route cleared")
    
    def _on_save_position(self, event):
        """Lưu vị trí hiện tại"""
        if self.routing_system.current_position:
            pos = self.routing_system.current_position
            import json
            import time
            
            position_data = {
                'timestamp': time.time(),
                'latitude': pos.lat,
                'longitude': pos.lng,
                'waste_type': self.selected_waste_type.value
            }
            
            filename = f"robot_position_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(position_data, f, indent=2)
            
            print(f"💾 Position saved: {filename}")
            print(f"📍 Lat: {pos.lat:.6f}, Lng: {pos.lng:.6f}")
    
    def _get_current_coords_text(self) -> str:
        """Lấy text tọa độ hiện tại"""
        if self.routing_system.current_position:
            pos = self.routing_system.current_position
            return f"Lat: {pos.lat:.6f}\nLng: {pos.lng:.6f}"
        return "No position set"
    
    def _refresh_map(self):
        """Refresh toàn bộ bản đồ"""
        self.ax.clear()
        self._draw_road_network()
        self._draw_waste_bins()
        self._draw_current_position()
        self._create_legend()
        
        self.ax.set_xlabel('Longitude (Kinh độ)')
        self.ax.set_ylabel('Latitude (Vĩ độ)')
        self.ax.set_title('Smart Waste Collection - Interactive Map\n'
                         'Click để chọn vị trí robot mới')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.canvas.draw()
    
    def get_current_position(self) -> Optional[GPSCoordinate]:
        """Lấy tọa độ hiện tại của robot"""
        return self.routing_system.current_position
    
    def set_position_change_callback(self, callback: Callable[[GPSCoordinate], None]):
        """Set callback khi vị trí thay đổi"""
        self.on_position_changed = callback


class PositionManager:
    """Manager để quản lý vị trí và lịch sử di chuyển"""
    
    def __init__(self):
        self.position_history = []
        self.current_position: Optional[GPSCoordinate] = None
    
    def update_position(self, new_position: GPSCoordinate):
        """Cập nhật vị trí mới"""
        if self.current_position:
            self.position_history.append({
                'timestamp': time.time(),
                'from': self.current_position,
                'to': new_position
            })
        
        self.current_position = new_position
        print(f"📍 Position updated: Lat {new_position.lat:.6f}, Lng {new_position.lng:.6f}")
    
    def get_position_info(self) -> dict:
        """Lấy thông tin vị trí hiện tại"""
        if not self.current_position:
            return {"status": "No position set"}
        
        return {
            "current_position": {
                "latitude": self.current_position.lat,
                "longitude": self.current_position.lng
            },
            "history_count": len(self.position_history),
            "last_update": self.position_history[-1]['timestamp'] if self.position_history else None
        }
    
    def save_position_history(self, filename: str = None):
        """Lưu lịch sử vị trí"""
        import json
        import time
        
        if not filename:
            filename = f"position_history_{int(time.time())}.json"
        
        history_data = {
            'current_position': {
                'lat': self.current_position.lat,
                'lng': self.current_position.lng
            } if self.current_position else None,
            'history': [
                {
                    'timestamp': entry['timestamp'],
                    'from': {'lat': entry['from'].lat, 'lng': entry['from'].lng},
                    'to': {'lat': entry['to'].lat, 'lng': entry['to'].lng}
                }
                for entry in self.position_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"💾 Position history saved: {filename}")


def demo_interactive_map():
    """Demo giao diện tương tác"""
    print("🗺️ Starting Interactive Map Demo...")
    print("📋 Controls:")
    print("   • Click chuột trái: Chọn vị trí robot")
    print("   • 'F' key hoặc Find Route button: Tìm đường")
    print("   • 'R' key hoặc Clear Route button: Xóa đường")
    print("   • 'S' key hoặc Save Position button: Lưu vị trí")
    print("   • Number keys (0-9): Đổi loại rác")
    
    # Tạo hệ thống với dữ liệu mẫu
    routing_system = create_sample_data()
    
    # Tạo position manager
    position_manager = PositionManager()
    if routing_system.current_position:
        position_manager.update_position(routing_system.current_position)
    
    # Tạo giao diện
    gui = InteractiveMapGUI(routing_system)
    
    # Set callback cho position change
    def on_position_changed(new_pos: GPSCoordinate):
        position_manager.update_position(new_pos)
        # In thông tin vị trí
        info = position_manager.get_position_info()
        print(f"📊 Position Info: {info}")
    
    gui.set_position_change_callback(on_position_changed)
    
    # Tạo và hiển thị map
    fig = gui.create_interactive_map()
    
    print("\n🎯 Map loaded! Start interacting...")
    print(f"📍 Current position: {gui.get_current_position()}")
    
    plt.show()
    
    # Sau khi đóng window, lưu lịch sử
    position_manager.save_position_history()
    print("✅ Demo completed!")


if __name__ == "__main__":
    import time
    demo_interactive_map()
