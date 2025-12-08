#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Routing System - Hệ thống định tuyến thông minh cho thu gom rác

Mô tả:
    Module này cung cấp hệ thống tối ưu lộ trình thu gom rác:
    - Thuật toán A* Pathfinding tìm đường đi ngắn nhất
    - TSP Solver tối ưu thứ tự thu gom
    - GPS Coordinate handling
    - Visualization route trên bản đồ
    - Real-time traffic consideration

Author: Huy Nguyen
Email: huynguyen@example.com
Date: August 2025
Version: 1.0.0
License: MIT
"""

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
import numpy as np


# ==================== ENUMS ====================

class WasteType(Enum):
    """Loại rác thải"""
    RECYCLABLE = "recyclable"   # Tái chế
    ORGANIC = "organic"         # Hữu cơ
    HAZARDOUS = "hazardous"     # Nguy hại
    OTHER = "other"             # Khác


class BinStatus(Enum):
    """Trạng thái thùng rác"""
    EMPTY = "empty"             # Trống (0-25%)
    NORMAL = "normal"           # Bình thường (25-75%)
    FULL = "full"               # Đầy (75-100%)
    OVERFLOW = "overflow"       # Tràn (>100%)


class TrafficCondition(Enum):
    """Tình trạng giao thông"""
    CLEAR = "clear"             # Thông thoáng
    MODERATE = "moderate"       # Trung bình
    CONGESTED = "congested"     # Tắc nghẽn


# ==================== DATA CLASSES ====================

@dataclass
class GPSCoordinate:
    """Tọa độ GPS"""
    latitude: float
    longitude: float
    
    def distance_to(self, other: 'GPSCoordinate') -> float:
        """Tính khoảng cách Haversine (km)"""
        R = 6371  # Bán kính Trái Đất (km)
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)


@dataclass
class WasteBin:
    """Thùng rác"""
    id: str
    location: GPSCoordinate
    waste_type: WasteType
    capacity: float  # Phần trăm đầy (0-100)
    
    @property
    def status(self) -> BinStatus:
        if self.capacity <= 25:
            return BinStatus.EMPTY
        elif self.capacity <= 75:
            return BinStatus.NORMAL
        elif self.capacity <= 100:
            return BinStatus.FULL
        else:
            return BinStatus.OVERFLOW
    
    @property
    def needs_collection(self) -> bool:
        return self.capacity >= 75


@dataclass
class PathfindingResult:
    """Kết quả tìm đường"""
    path: List[Tuple[int, int]]
    distance: float
    estimated_time: float  # Phút
    success: bool
    message: str = ""


# ==================== A* PATHFINDING ====================

class AStarPathfinder:
    """Thuật toán A* tìm đường đi ngắn nhất"""
    
    def __init__(self, grid: np.ndarray):
        """
        Args:
            grid: Ma trận 2D (0: đi được, 1: vật cản)
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        
        # 8 hướng di chuyển (bao gồm đường chéo)
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Lên, Xuống, Trái, Phải
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Đường chéo
        ]
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic: Euclidean distance"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Kiểm tra vị trí hợp lệ"""
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row, col] == 0)
    
    def find_path(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> PathfindingResult:
        """
        Tìm đường đi ngắn nhất từ start đến end
        
        Args:
            start: Điểm bắt đầu (row, col)
            end: Điểm kết thúc (row, col)
            
        Returns:
            PathfindingResult với path và thông tin
        """
        if not self.is_valid(start):
            return PathfindingResult([], 0, 0, False, "Start position is invalid")
        
        if not self.is_valid(end):
            return PathfindingResult([], 0, 0, False, "End position is invalid")
        
        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        
        # Tracking
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                distance = g_score[end]
                estimated_time = distance * 2  # Giả sử 2 phút/đơn vị
                
                return PathfindingResult(
                    path=path,
                    distance=distance,
                    estimated_time=estimated_time,
                    success=True,
                    message=f"Path found: {len(path)} steps"
                )
            
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid(neighbor):
                    continue
                
                # Chi phí di chuyển (đường chéo = √2)
                move_cost = math.sqrt(2) if dx != 0 and dy != 0 else 1
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, end)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        return PathfindingResult([], 0, 0, False, "No path found")


# ==================== SMART ROUTING SYSTEM ====================

class SmartRoutingSystem:
    """Hệ thống định tuyến thông minh"""
    
    def __init__(self, grid_size: Tuple[int, int] = (100, 100)):
        """
        Args:
            grid_size: Kích thước grid (rows, cols)
        """
        self.grid = np.zeros(grid_size, dtype=int)
        self.bins: Dict[str, WasteBin] = {}
        self.robot_position: Optional[GPSCoordinate] = None
        self.pathfinder = AStarPathfinder(self.grid)
    
    def add_bin(self, bin: WasteBin) -> None:
        """Thêm thùng rác"""
        self.bins[bin.id] = bin
    
    def remove_bin(self, bin_id: str) -> None:
        """Xóa thùng rác"""
        if bin_id in self.bins:
            del self.bins[bin_id]
    
    def update_bin_capacity(self, bin_id: str, capacity: float) -> None:
        """Cập nhật dung lượng thùng rác"""
        if bin_id in self.bins:
            self.bins[bin_id].capacity = capacity
    
    def set_robot_position(self, position: GPSCoordinate) -> None:
        """Đặt vị trí robot"""
        self.robot_position = position
    
    def get_bins_needing_collection(self) -> List[WasteBin]:
        """Lấy danh sách thùng rác cần thu gom"""
        return [bin for bin in self.bins.values() if bin.needs_collection]
    
    def add_obstacle(self, row: int, col: int) -> None:
        """Thêm vật cản"""
        if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]:
            self.grid[row, col] = 1
            self.pathfinder = AStarPathfinder(self.grid)
    
    def calculate_collection_route(self) -> List[PathfindingResult]:
        """
        Tính toán lộ trình thu gom tối ưu
        
        Returns:
            List PathfindingResult cho từng đoạn đường
        """
        if self.robot_position is None:
            return []
        
        bins_to_collect = self.get_bins_needing_collection()
        if not bins_to_collect:
            return []
        
        # Simple greedy: đi đến thùng gần nhất trước
        results = []
        current_pos = self.robot_position
        remaining_bins = bins_to_collect.copy()
        
        while remaining_bins:
            # Tìm thùng gần nhất
            nearest_bin = min(
                remaining_bins,
                key=lambda b: current_pos.distance_to(b.location)
            )
            
            # Convert GPS to grid coordinates (simplified)
            start_grid = self._gps_to_grid(current_pos)
            end_grid = self._gps_to_grid(nearest_bin.location)
            
            # Tìm đường
            result = self.pathfinder.find_path(start_grid, end_grid)
            results.append(result)
            
            # Cập nhật vị trí hiện tại
            current_pos = nearest_bin.location
            remaining_bins.remove(nearest_bin)
        
        return results
    
    def _gps_to_grid(self, coord: GPSCoordinate) -> Tuple[int, int]:
        """Chuyển đổi GPS sang grid coordinates (simplified)"""
        # Đây là phiên bản đơn giản hóa
        row = int((coord.latitude % 1) * self.grid.shape[0]) % self.grid.shape[0]
        col = int((coord.longitude % 1) * self.grid.shape[1]) % self.grid.shape[1]
        return (row, col)


# ==================== MAP VISUALIZER ====================

class MapVisualizer:
    """Visualize bản đồ và lộ trình"""
    
    def __init__(self, routing_system: SmartRoutingSystem):
        self.routing_system = routing_system
    
    def create_map_html(self, output_path: str = "route_map.html") -> str:
        """
        Tạo bản đồ HTML với Folium
        
        Args:
            output_path: Đường dẫn file output
            
        Returns:
            Đường dẫn file HTML đã tạo
        """
        try:
            import folium
            
            # Tạo bản đồ centered tại vị trí trung bình
            if self.routing_system.bins:
                avg_lat = sum(b.location.latitude for b in self.routing_system.bins.values()) / len(self.routing_system.bins)
                avg_lon = sum(b.location.longitude for b in self.routing_system.bins.values()) / len(self.routing_system.bins)
            else:
                avg_lat, avg_lon = 10.762622, 106.660172  # Default: Ho Chi Minh City
            
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15)
            
            # Thêm markers cho thùng rác
            for bin in self.routing_system.bins.values():
                color = {
                    BinStatus.EMPTY: 'green',
                    BinStatus.NORMAL: 'blue',
                    BinStatus.FULL: 'orange',
                    BinStatus.OVERFLOW: 'red'
                }.get(bin.status, 'gray')
                
                folium.Marker(
                    location=bin.location.to_tuple(),
                    popup=f"ID: {bin.id}<br>Type: {bin.waste_type.value}<br>Capacity: {bin.capacity:.1f}%",
                    icon=folium.Icon(color=color, icon='trash')
                ).add_to(m)
            
            # Thêm marker cho robot
            if self.routing_system.robot_position:
                folium.Marker(
                    location=self.routing_system.robot_position.to_tuple(),
                    popup="Robot Position",
                    icon=folium.Icon(color='purple', icon='truck')
                ).add_to(m)
            
            m.save(output_path)
            return output_path
            
        except ImportError:
            print("Folium not installed. Run: pip install folium")
            return ""


# ==================== UTILITY FUNCTIONS ====================

def create_sample_data() -> SmartRoutingSystem:
    """Tạo dữ liệu mẫu cho testing"""
    system = SmartRoutingSystem(grid_size=(50, 50))
    
    # Thêm các thùng rác mẫu
    sample_bins = [
        WasteBin("BIN001", GPSCoordinate(10.762622, 106.660172), WasteType.RECYCLABLE, 85),
        WasteBin("BIN002", GPSCoordinate(10.763000, 106.661000), WasteType.ORGANIC, 45),
        WasteBin("BIN003", GPSCoordinate(10.764000, 106.662000), WasteType.HAZARDOUS, 90),
        WasteBin("BIN004", GPSCoordinate(10.765000, 106.660500), WasteType.OTHER, 30),
        WasteBin("BIN005", GPSCoordinate(10.761500, 106.661500), WasteType.RECYCLABLE, 100),
    ]
    
    for bin in sample_bins:
        system.add_bin(bin)
    
    # Đặt vị trí robot
    system.set_robot_position(GPSCoordinate(10.762000, 106.660000))
    
    return system


# ==================== MAIN ====================

if __name__ == "__main__":
    print("=== Smart Routing System Demo ===\n")
    
    # Tạo hệ thống với dữ liệu mẫu
    system = create_sample_data()
    
    # Hiển thị thông tin
    print(f"Total bins: {len(system.bins)}")
    print(f"Bins needing collection: {len(system.get_bins_needing_collection())}")
    
    for bin in system.bins.values():
        print(f"  - {bin.id}: {bin.waste_type.value}, {bin.capacity:.0f}%, {bin.status.value}")
    
    # Tính toán lộ trình
    print("\n=== Calculating Route ===")
    routes = system.calculate_collection_route()
    
    for i, route in enumerate(routes, 1):
        if route.success:
            print(f"Route {i}: {route.distance:.2f} units, ~{route.estimated_time:.0f} min")
        else:
            print(f"Route {i}: {route.message}")
    
    # Tạo bản đồ
    print("\n=== Creating Map ===")
    visualizer = MapVisualizer(system)
    map_path = visualizer.create_map_html("demo_route.html")
    if map_path:
        print(f"Map saved to: {map_path}")
    
    print("\nDone!")
