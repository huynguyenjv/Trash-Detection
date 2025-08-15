"""
Routing Engine - Module xử lý tìm đường và tối ưu hóa
"""

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GPSCoordinate, WasteBin, Road, PathfindingResult, RouteStep
from core.enums import TrafficCondition


class RoutingEngine:
    """Engine xử lý routing và pathfinding"""
    
    def __init__(self):
        self.roads: List[Road] = []
        self.adjacency_graph: Dict[GPSCoordinate, List[Tuple[GPSCoordinate, float]]] = {}
        
    def add_road(self, road: Road):
        """Thêm đường vào graph"""
        self.roads.append(road)
        self._update_adjacency_graph(road)
    
    def _update_adjacency_graph(self, road: Road):
        """Cập nhật adjacency graph"""
        # Thêm cạnh hai chiều
        if road.start not in self.adjacency_graph:
            self.adjacency_graph[road.start] = []
        if road.end not in self.adjacency_graph:
            self.adjacency_graph[road.end] = []
            
        # Trọng số dựa trên thời gian di chuyển
        weight = road.travel_time
        
        self.adjacency_graph[road.start].append((road.end, weight))
        self.adjacency_graph[road.end].append((road.start, weight))
    
    def find_path_astar(self, start: GPSCoordinate, goal: GPSCoordinate) -> PathfindingResult:
        """
        Tìm đường tối ưu bằng thuật toán A*
        """
        if start not in self.adjacency_graph or goal not in self.adjacency_graph:
            return PathfindingResult([], 0, 0)
        
        # Priority queue: (f_score, g_score, current_node, path)
        open_set = [(0, 0, start, [start])]
        visited = set()
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal:
                return self._create_pathfinding_result(path)
            
            # Kiểm tra các node kế bên
            for neighbor, edge_cost in self.adjacency_graph.get(current, []):
                if neighbor in visited:
                    continue
                
                tentative_g_score = g_score + edge_cost
                h_score = self._heuristic(neighbor, goal)
                f_score = tentative_g_score + h_score
                
                new_path = path + [neighbor]
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, new_path))
        
        return PathfindingResult([], 0, 0)  # Không tìm thấy đường
    
    def _heuristic(self, point1: GPSCoordinate, point2: GPSCoordinate) -> float:
        """Heuristic function cho A* (Euclidean distance)"""
        return point1.distance_to(point2)
    
    def _create_pathfinding_result(self, path: List[GPSCoordinate]) -> PathfindingResult:
        """Tạo kết quả pathfinding từ path"""
        if len(path) < 2:
            return PathfindingResult([], 0, 0)
        
        total_distance = 0
        total_time = 0
        steps = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            # Tìm road tương ứng
            road = self._find_road(current, next_point)
            distance = current.distance_to(next_point)
            time = road.travel_time if road else (distance / 30) * 60  # 30 km/h default
            
            total_distance += distance
            total_time += time
            
            # Tạo instruction
            instruction = self._generate_instruction(current, next_point, i == 0)
            
            steps.append(RouteStep(
                from_point=current,
                to_point=next_point,
                instruction=instruction,
                distance=distance,
                duration=time
            ))
        
        # Ước tính nhiên liệu (8L/100km)
        fuel_estimate = (total_distance * 8) / 100
        
        return PathfindingResult(
            path=path,
            total_distance=total_distance,
            total_time=total_time,
            steps=steps,
            fuel_estimate=fuel_estimate
        )
    
    def _find_road(self, start: GPSCoordinate, end: GPSCoordinate) -> Optional[Road]:
        """Tìm đường giữa hai điểm"""
        for road in self.roads:
            if ((road.start == start and road.end == end) or 
                (road.start == end and road.end == start)):
                return road
        return None
    
    def _generate_instruction(self, from_point: GPSCoordinate, to_point: GPSCoordinate, is_first: bool) -> str:
        """Tạo hướng dẫn di chuyển"""
        if is_first:
            return "Bắt đầu di chuyển"
        
        # Tính góc để xác định hướng
        angle = math.atan2(to_point.lng - from_point.lng, to_point.lat - from_point.lat)
        angle_degrees = math.degrees(angle)
        
        if -22.5 <= angle_degrees < 22.5:
            return "Đi thẳng về phía Bắc"
        elif 22.5 <= angle_degrees < 67.5:
            return "Rẽ phải về phía Đông Bắc"
        elif 67.5 <= angle_degrees < 112.5:
            return "Rẽ phải về phía Đông"
        elif 112.5 <= angle_degrees < 157.5:
            return "Rẽ phải về phía Đông Nam"
        elif 157.5 <= angle_degrees or angle_degrees < -157.5:
            return "Quay đầu về phía Nam"
        elif -157.5 <= angle_degrees < -112.5:
            return "Rẽ trái về phía Tây Nam"
        elif -112.5 <= angle_degrees < -67.5:
            return "Rẽ trái về phía Tây"
        else:  # -67.5 <= angle_degrees < -22.5
            return "Rẽ trái về phía Tây Bắc"
    
    def optimize_collection_route(self, start: GPSCoordinate, bins: List[WasteBin]) -> PathfindingResult:
        """
        Tối ưu hóa tuyến đường thu gom rác (Traveling Salesman Problem)
        Sử dụng Nearest Neighbor heuristic
        """
        if not bins:
            return PathfindingResult([], 0, 0)
        
        # Sắp xếp bins theo độ ưu tiên
        sorted_bins = sorted(bins, key=lambda b: b.priority, reverse=True)
        
        current_location = start
        path = [start]
        total_distance = 0
        total_time = 0
        steps = []
        
        unvisited_bins = sorted_bins.copy()
        
        while unvisited_bins:
            # Tìm bin gần nhất
            nearest_bin = min(unvisited_bins, 
                            key=lambda b: current_location.distance_to(b.location))
            
            # Tìm đường đến bin
            segment_result = self.find_path_astar(current_location, nearest_bin.location)
            
            if segment_result.is_valid:
                # Cập nhật path (bỏ điểm đầu để tránh duplicate)
                path.extend(segment_result.path[1:])
                total_distance += segment_result.total_distance
                total_time += segment_result.total_time
                steps.extend(segment_result.steps)
                
                current_location = nearest_bin.location
                unvisited_bins.remove(nearest_bin)
            else:
                # Nếu không tìm thấy đường, bỏ qua bin này
                unvisited_bins.remove(nearest_bin)
        
        # Quay về điểm xuất phát
        return_result = self.find_path_astar(current_location, start)
        if return_result.is_valid:
            path.extend(return_result.path[1:])
            total_distance += return_result.total_distance
            total_time += return_result.total_time
            steps.extend(return_result.steps)
        
        fuel_estimate = (total_distance * 8) / 100
        
        return PathfindingResult(
            path=path,
            total_distance=total_distance,
            total_time=total_time,
            steps=steps,
            fuel_estimate=fuel_estimate
        )
    
    def get_traffic_info(self, road: Road) -> Dict[str, any]:
        """Lấy thông tin giao thông của đường"""
        return {
            'condition': road.traffic_condition.value,
            'speed': road.max_speed,
            'travel_time': road.travel_time,
            'length': road.length
        }
