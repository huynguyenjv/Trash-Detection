"""
Hệ thống theo dõi rác thải và định tuyến thông minh cho xe gom rác
Sử dụng YOLOv8 real-time detection + thuật toán A* pathfinding

Author: Smart Waste Management System
Date: August 2025
"""

import cv2
import numpy as np
import math
import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WasteType(Enum):
    """Enum cho các loại rác thải"""
    ORGANIC = "organic"          # Rác hữu cơ
    PLASTIC = "plastic"          # Nhựa
    GLASS = "glass"              # Thủy tinh
    METAL = "metal"              # Kim loại
    PAPER = "paper"              # Giấy
    CARDBOARD = "cardboard"      # Bìa carton
    BATTERY = "battery"          # Pin
    CLOTHES = "clothes"          # Quần áo
    SHOES = "shoes"              # Giày dép
    GENERAL = "general"          # Rác thải chung


class BinStatus(Enum):
    """Trạng thái bãi rác"""
    OK = "OK"                    # Bình thường
    NEAR_FULL = "NEAR_FULL"      # Gần đầy
    FULL = "FULL"                # Đầy


class TrafficCondition(Enum):
    """Điều kiện giao thông"""
    CLEAR = "clear"              # Thông thoáng
    MODERATE = "moderate"        # Trung bình
    HEAVY = "heavy"              # Kẹt xe
    BLOCKED = "blocked"          # Bị chặn


@dataclass
class GPSCoordinate:
    """Tọa độ GPS"""
    lat: float  # Vĩ độ
    lng: float  # Kinh độ
    
    def __hash__(self):
        return hash((round(self.lat, 6), round(self.lng, 6)))
    
    def __eq__(self, other):
        if not isinstance(other, GPSCoordinate):
            return False
        return (abs(self.lat - other.lat) < 1e-6 and 
                abs(self.lng - other.lng) < 1e-6)


@dataclass
class WasteBin:
    """Thông tin bãi rác"""
    id: str
    location: GPSCoordinate
    supported_types: Set[WasteType]
    max_capacity: float          # Sức chứa tối đa (kg)
    current_capacity: float      # Sức chứa hiện tại (kg)
    status: BinStatus
    last_updated: float = field(default_factory=time.time)
    
    @property
    def capacity_ratio(self) -> float:
        """Tỷ lệ đầy (%/100)"""
        return self.current_capacity / self.max_capacity
    
    @property
    def remaining_capacity(self) -> float:
        """Sức chứa còn lại (kg)"""
        return self.max_capacity - self.current_capacity
    
    def can_accept_waste(self, waste_type: WasteType, amount: float = 0) -> bool:
        """Kiểm tra có thể nhận loại rác này không"""
        return (self.status != BinStatus.FULL and 
                waste_type in self.supported_types and
                self.remaining_capacity >= amount)


@dataclass
class RoadSegment:
    """Đoạn đường giữa 2 điểm"""
    start: GPSCoordinate
    end: GPSCoordinate
    distance: float              # Khoảng cách (km)
    travel_time: float           # Thời gian di chuyển cơ bản (phút)
    traffic_condition: TrafficCondition
    road_quality: float          # Chất lượng đường (0-1, 1=tốt nhất)
    is_blocked: bool = False     # Có bị chặn không
    
    def get_actual_travel_time(self) -> float:
        """Tính thời gian di chuyển thực tế dựa trên điều kiện"""
        base_time = self.travel_time
        
        # Traffic penalty
        traffic_multiplier = {
            TrafficCondition.CLEAR: 1.0,
            TrafficCondition.MODERATE: 1.3,
            TrafficCondition.HEAVY: 2.0,
            TrafficCondition.BLOCKED: float('inf')
        }
        
        # Road quality penalty
        road_penalty = (1 - self.road_quality) * 0.5 + 1  # 1.0 - 1.5x
        
        if self.is_blocked:
            return float('inf')
        
        return base_time * traffic_multiplier[self.traffic_condition] * road_penalty


@dataclass
class WasteCounter:
    """Bộ đếm rác thải"""
    counts: Dict[WasteType, int] = field(default_factory=dict)
    threshold: int = 10
    
    def __post_init__(self):
        # Khởi tạo tất cả loại rác = 0
        for waste_type in WasteType:
            if waste_type not in self.counts:
                self.counts[waste_type] = 0
    
    def add_detection(self, waste_type: WasteType, count: int = 1):
        """Thêm phát hiện rác"""
        self.counts[waste_type] += count
        logger.info(f"Detected {count} {waste_type.value}: total = {self.counts[waste_type]}")
    
    def get_full_types(self) -> List[WasteType]:
        """Lấy danh sách loại rác đã đạt threshold"""
        return [wtype for wtype, count in self.counts.items() 
                if count >= self.threshold]
    
    def reset_type(self, waste_type: WasteType):
        """Reset đếm cho một loại rác"""
        self.counts[waste_type] = 0
        logger.info(f"Reset counter for {waste_type.value}")


class HaversineCalculator:
    """Calculator cho khoảng cách Haversine"""
    
    EARTH_RADIUS_KM = 6371.0
    
    @staticmethod
    def distance(coord1: GPSCoordinate, coord2: GPSCoordinate) -> float:
        """
        Tính khoảng cách Haversine giữa 2 điểm GPS (km)
        
        Args:
            coord1: Tọa độ điểm 1
            coord2: Tọa độ điểm 2
            
        Returns:
            Khoảng cách theo km
        """
        # Chuyển độ sang radian
        lat1, lng1 = math.radians(coord1.lat), math.radians(coord1.lng)
        lat2, lng2 = math.radians(coord2.lat), math.radians(coord2.lng)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return HaversineCalculator.EARTH_RADIUS_KM * c
    
    @staticmethod
    def bearing(coord1: GPSCoordinate, coord2: GPSCoordinate) -> float:
        """Tính hướng từ coord1 đến coord2 (độ)"""
        lat1, lng1 = math.radians(coord1.lat), math.radians(coord1.lng)
        lat2, lng2 = math.radians(coord2.lat), math.radians(coord2.lng)
        
        dlng = lng2 - lng1
        
        y = math.sin(dlng) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlng))
        
        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360


@dataclass
class AStarNode:
    """Node cho thuật toán A*"""
    coordinate: GPSCoordinate
    g_cost: float = float('inf')  # Chi phí thực tế từ start
    h_cost: float = 0.0           # Heuristic cost đến goal
    f_cost: float = float('inf')  # Total cost = g + h
    parent: Optional['AStarNode'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __hash__(self):
        return hash(self.coordinate)
    
    def __eq__(self, other):
        return isinstance(other, AStarNode) and self.coordinate == other.coordinate


@dataclass
class PathfindingResult:
    """Kết quả tìm đường"""
    path: List[GPSCoordinate]
    total_distance: float        # Tổng khoảng cách (km)
    estimated_time: float        # ETA (phút)
    total_cost: float           # Tổng chi phí
    target_bin: Optional[WasteBin] = None
    
    def __bool__(self):
        return len(self.path) > 0


class SmartRoutingSystem:
    """Hệ thống định tuyến thông minh"""
    
    def __init__(self):
        # Trọng số cho cost function
        self.w_distance = 1.0      # Trọng số khoảng cách
        self.w_time = 0.5          # Trọng số thời gian
        self.w_full_penalty = 50   # Penalty cho bãi rác gần đầy
        self.w_capacity_penalty = 100  # Penalty dựa trên tỷ lệ đầy
        
        # Dữ liệu hệ thống
        self.waste_bins: Dict[str, WasteBin] = {}
        self.road_network: List[RoadSegment] = []
        self.current_position: Optional[GPSCoordinate] = None
        
        # Grid cho A* (tạo grid ảo từ road network)
        self.grid_resolution = 0.001  # ~100m resolution
        self.node_cache: Dict[GPSCoordinate, AStarNode] = {}
        
    def add_waste_bin(self, waste_bin: WasteBin):
        """Thêm bãi rác vào hệ thống"""
        self.waste_bins[waste_bin.id] = waste_bin
        logger.info(f"Added waste bin {waste_bin.id} at {waste_bin.location.lat}, {waste_bin.location.lng}")
    
    def add_road_segment(self, segment: RoadSegment):
        """Thêm đoạn đường vào mạng lưới"""
        self.road_network.append(segment)
    
    def update_robot_position(self, position: GPSCoordinate):
        """Cập nhật vị trí robot"""
        self.current_position = position
        logger.info(f"Robot position updated: {position.lat}, {position.lng}")
    
    def update_bin_status(self, bin_id: str, current_capacity: float, status: BinStatus):
        """Cập nhật trạng thái bãi rác"""
        if bin_id in self.waste_bins:
            bin_obj = self.waste_bins[bin_id]
            bin_obj.current_capacity = current_capacity
            bin_obj.status = status
            bin_obj.last_updated = time.time()
            logger.info(f"Updated bin {bin_id}: {current_capacity}kg, {status.value}")
    
    def update_traffic_condition(self, start: GPSCoordinate, end: GPSCoordinate, 
                               condition: TrafficCondition):
        """Cập nhật điều kiện giao thông"""
        for segment in self.road_network:
            if ((segment.start == start and segment.end == end) or
                (segment.start == end and segment.end == start)):
                segment.traffic_condition = condition
                logger.info(f"Updated traffic: {start.lat},{start.lng} -> {end.lat},{end.lng}: {condition.value}")
    
    def find_suitable_bins(self, waste_type: WasteType) -> List[WasteBin]:
        """Tìm các bãi rác phù hợp cho loại rác"""
        suitable_bins = []
        
        for bin_obj in self.waste_bins.values():
            if bin_obj.can_accept_waste(waste_type):
                suitable_bins.append(bin_obj)
        
        # Sắp xếp theo khoảng cách nếu có vị trí robot
        if self.current_position:
            suitable_bins.sort(key=lambda b: HaversineCalculator.distance(
                self.current_position, b.location))
        
        return suitable_bins
    
    def get_neighbors(self, coord: GPSCoordinate) -> List[Tuple[GPSCoordinate, float]]:
        """
        Lấy các neighbor và cost từ một tọa độ
        Trả về: List[(neighbor_coord, edge_cost)]
        """
        neighbors = []
        
        for segment in self.road_network:
            neighbor_coord = None
            
            # Kiểm tra xem coord có khớp với start hoặc end của segment
            if HaversineCalculator.distance(coord, segment.start) < 0.01:  # ~10m tolerance
                neighbor_coord = segment.end
            elif HaversineCalculator.distance(coord, segment.end) < 0.01:
                neighbor_coord = segment.start
            
            if neighbor_coord:
                # Tính cost cho edge này
                edge_cost = self.calculate_edge_cost(segment)
                if edge_cost < float('inf'):
                    neighbors.append((neighbor_coord, edge_cost))
        
        return neighbors
    
    def calculate_edge_cost(self, segment: RoadSegment) -> float:
        """Tính cost cho một edge (đoạn đường)"""
        if segment.is_blocked or segment.traffic_condition == TrafficCondition.BLOCKED:
            return float('inf')
        
        # Base cost từ distance và time
        distance_cost = segment.distance * self.w_distance
        time_cost = segment.get_actual_travel_time() * self.w_time
        
        return distance_cost + time_cost
    
    def calculate_bin_penalty(self, bin_obj: WasteBin) -> float:
        """Tính penalty dựa trên trạng thái bãi rác"""
        penalty = 0.0
        
        # Penalty cho trạng thái NEAR_FULL
        if bin_obj.status == BinStatus.NEAR_FULL:
            penalty += self.w_full_penalty
        
        # Penalty dựa trên tỷ lệ đầy
        capacity_penalty = bin_obj.capacity_ratio * self.w_capacity_penalty
        penalty += capacity_penalty
        
        return penalty
    
    def heuristic(self, current: GPSCoordinate, goal: GPSCoordinate) -> float:
        """Hàm heuristic - khoảng cách Haversine"""
        return HaversineCalculator.distance(current, goal)
    
    def a_star_pathfinding(self, start: GPSCoordinate, goal: GPSCoordinate, 
                          target_bin: Optional[WasteBin] = None) -> PathfindingResult:
        """
        Thuật toán A* tìm đường tối ưu
        
        Args:
            start: Điểm bắt đầu
            goal: Điểm đích
            target_bin: Bãi rác đích (để tính penalty)
            
        Returns:
            PathfindingResult chứa đường đi và thông tin
        """
        logger.info(f"A* pathfinding: {start.lat},{start.lng} -> {goal.lat},{goal.lng}")
        
        # Initialize
        open_set = []
        closed_set: Set[GPSCoordinate] = set()
        
        start_node = AStarNode(start, g_cost=0.0, h_cost=self.heuristic(start, goal))
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        heapq.heappush(open_set, start_node)
        node_map: Dict[GPSCoordinate, AStarNode] = {start: start_node}
        
        while open_set:
            current_node = heapq.heappop(open_set)
            current_coord = current_node.coordinate
            
            # Đã đến đích
            if HaversineCalculator.distance(current_coord, goal) < 0.01:  # ~10m tolerance
                return self._reconstruct_path(current_node, target_bin)
            
            closed_set.add(current_coord)
            
            # Xét các neighbor
            for neighbor_coord, edge_cost in self.get_neighbors(current_coord):
                if neighbor_coord in closed_set:
                    continue
                
                tentative_g = current_node.g_cost + edge_cost
                
                # Thêm bin penalty nếu neighbor là goal và có target_bin
                if (target_bin and 
                    HaversineCalculator.distance(neighbor_coord, goal) < 0.01):
                    tentative_g += self.calculate_bin_penalty(target_bin)
                
                # Tạo hoặc update neighbor node
                if neighbor_coord not in node_map:
                    neighbor_node = AStarNode(
                        coordinate=neighbor_coord,
                        h_cost=self.heuristic(neighbor_coord, goal)
                    )
                    node_map[neighbor_coord] = neighbor_node
                else:
                    neighbor_node = node_map[neighbor_coord]
                
                # Nếu tìm được đường tốt hơn
                if tentative_g < neighbor_node.g_cost:
                    neighbor_node.parent = current_node
                    neighbor_node.g_cost = tentative_g
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                    
                    if neighbor_node not in open_set:
                        heapq.heappush(open_set, neighbor_node)
        
        # Không tìm được đường
        logger.warning(f"No path found from {start.lat},{start.lng} to {goal.lat},{goal.lng}")
        return PathfindingResult([], 0.0, 0.0, float('inf'))
    
    def _reconstruct_path(self, goal_node: AStarNode, 
                         target_bin: Optional[WasteBin] = None) -> PathfindingResult:
        """Xây dựng lại đường đi từ goal node"""
        path = []
        total_distance = 0.0
        total_time = 0.0
        current = goal_node
        
        # Backtrack để tìm path
        while current:
            path.append(current.coordinate)
            if current.parent:
                segment_distance = HaversineCalculator.distance(
                    current.parent.coordinate, current.coordinate)
                total_distance += segment_distance
                
                # Tìm segment tương ứng để lấy travel time
                segment_time = self._get_segment_time(
                    current.parent.coordinate, current.coordinate)
                total_time += segment_time
                
            current = current.parent
        
        path.reverse()
        
        return PathfindingResult(
            path=path,
            total_distance=total_distance,
            estimated_time=total_time,
            total_cost=goal_node.g_cost,
            target_bin=target_bin
        )
    
    def _get_segment_time(self, coord1: GPSCoordinate, coord2: GPSCoordinate) -> float:
        """Lấy thời gian di chuyển giữa 2 tọa độ"""
        for segment in self.road_network:
            if ((HaversineCalculator.distance(coord1, segment.start) < 0.01 and
                 HaversineCalculator.distance(coord2, segment.end) < 0.01) or
                (HaversineCalculator.distance(coord1, segment.end) < 0.01 and
                 HaversineCalculator.distance(coord2, segment.start) < 0.01)):
                return segment.get_actual_travel_time()
        
        # Fallback: ước tính dựa trên khoảng cách (30 km/h average speed)
        distance = HaversineCalculator.distance(coord1, coord2)
        return distance / 30 * 60  # phút
    
    def find_optimal_route(self, waste_type: WasteType) -> Optional[PathfindingResult]:
        """
        Tìm tuyến đường tối ưu đến bãi rác phù hợp
        
        Args:
            waste_type: Loại rác cần đổ
            
        Returns:
            PathfindingResult hoặc None nếu không tìm được
        """
        if not self.current_position:
            logger.error("Robot position not set")
            return None
        
        # Tìm các bãi rác phù hợp
        suitable_bins = self.find_suitable_bins(waste_type)
        if not suitable_bins:
            logger.warning(f"No suitable bins found for {waste_type.value}")
            return None
        
        best_result = None
        best_cost = float('inf')
        
        # Thử tất cả các bãi rác phù hợp
        for bin_obj in suitable_bins:
            result = self.a_star_pathfinding(
                self.current_position, bin_obj.location, bin_obj)
            
            if result and result.total_cost < best_cost:
                best_cost = result.total_cost
                best_result = result
        
        if best_result:
            logger.info(f"Found optimal route to {best_result.target_bin.id}: "
                       f"{best_result.total_distance:.2f}km, "
                       f"{best_result.estimated_time:.1f}min, "
                       f"cost={best_result.total_cost:.2f}")
        
        return best_result


class MapVisualizer:
    """Visualizer cho bản đồ và đường đi"""
    
    @staticmethod
    def plot_route(routing_system: SmartRoutingSystem, 
                  result: PathfindingResult,
                  waste_type: WasteType,
                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Vẽ bản đồ với đường đi tối ưu
        
        Args:
            routing_system: Hệ thống định tuyến
            result: Kết quả tìm đường
            waste_type: Loại rác
            figsize: Kích thước figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tìm bounding box
        all_coords = []
        all_coords.extend([coord for coord in result.path])
        all_coords.extend([bin_obj.location for bin_obj in routing_system.waste_bins.values()])
        if routing_system.current_position:
            all_coords.append(routing_system.current_position)
        
        if not all_coords:
            return fig
        
        lats = [coord.lat for coord in all_coords]
        lngs = [coord.lng for coord in all_coords]
        
        lat_margin = (max(lats) - min(lats)) * 0.1
        lng_margin = (max(lngs) - min(lngs)) * 0.1
        
        ax.set_xlim(min(lngs) - lng_margin, max(lngs) + lng_margin)
        ax.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
        
        # Vẽ road network
        for segment in routing_system.road_network:
            color = 'gray'
            alpha = 0.3
            linewidth = 1
            
            if segment.traffic_condition == TrafficCondition.HEAVY:
                color = 'red'
                alpha = 0.7
            elif segment.traffic_condition == TrafficCondition.MODERATE:
                color = 'orange'
                alpha = 0.5
            elif segment.is_blocked:
                color = 'black'
                alpha = 0.8
                linewidth = 3
            
            ax.plot([segment.start.lng, segment.end.lng],
                   [segment.start.lat, segment.end.lat],
                   color=color, alpha=alpha, linewidth=linewidth)
        
        # Vẽ waste bins
        for bin_obj in routing_system.waste_bins.values():
            color = 'green'
            if bin_obj.status == BinStatus.NEAR_FULL:
                color = 'orange'
            elif bin_obj.status == BinStatus.FULL:
                color = 'red'
            
            # Kiểm tra hỗ trợ loại rác hiện tại
            marker = 'o'
            if waste_type in bin_obj.supported_types:
                marker = 's'  # Square cho bins phù hợp
            
            ax.scatter(bin_obj.location.lng, bin_obj.location.lat,
                      c=color, s=100, marker=marker, edgecolors='black',
                      label=f'Bin {bin_obj.id} ({bin_obj.status.value})')
        
        # Vẽ robot position
        if routing_system.current_position:
            ax.scatter(routing_system.current_position.lng, 
                      routing_system.current_position.lat,
                      c='blue', s=200, marker='^', edgecolors='black',
                      label='Robot Position')
        
        # Vẽ optimal path
        if result.path:
            path_lngs = [coord.lng for coord in result.path]
            path_lats = [coord.lat for coord in result.path]
            
            ax.plot(path_lngs, path_lats, 
                   color='blue', linewidth=3, alpha=0.8,
                   label=f'Optimal Route ({result.total_distance:.2f}km)')
            
            # Đánh dấu điểm đầu và cuối
            ax.scatter(path_lngs[0], path_lats[0], 
                      c='green', s=150, marker='o', edgecolors='black',
                      label='Start')
            ax.scatter(path_lngs[-1], path_lats[-1], 
                      c='red', s=150, marker='X', edgecolors='black',
                      label='Target Bin')
        
        # Thiết lập biểu đồ
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Smart Waste Collection Route - {waste_type.value.title()}\n'
                    f'Distance: {result.total_distance:.2f}km, '
                    f'ETA: {result.estimated_time:.1f}min, '
                    f'Cost: {result.total_cost:.2f}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig


class RealTimeWasteDetector:
    """Hệ thống phát hiện rác thời gian thực"""
    
    def __init__(self, model_path: str, routing_system: SmartRoutingSystem):
        self.model = YOLO(model_path)
        self.routing_system = routing_system
        self.waste_counter = WasteCounter()
        
        # Mapping từ YOLO classes đến WasteType
        self.class_to_waste_type = {
            'organic': WasteType.ORGANIC,
            'biological': WasteType.ORGANIC,
            'plastic': WasteType.PLASTIC,
            'glass': WasteType.GLASS,
            'metal': WasteType.METAL,
            'paper': WasteType.PAPER,
            'cardboard': WasteType.CARDBOARD,
            'battery': WasteType.BATTERY,
            'clothes': WasteType.CLOTHES,
            'shoes': WasteType.SHOES,
            'trash': WasteType.GENERAL
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[WasteType]]:
        """
        Xử lý frame và phát hiện rác
        
        Args:
            frame: Frame từ camera
            
        Returns:
            Tuple[annotated_frame, triggered_waste_types]
        """
        # Chạy detection
        results = self.model(frame)
        
        # Đếm các detections
        frame_detections = {}
        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls)]
                if class_name in self.class_to_waste_type:
                    waste_type = self.class_to_waste_type[class_name]
                    frame_detections[waste_type] = frame_detections.get(waste_type, 0) + 1
        
        # Cập nhật counter
        for waste_type, count in frame_detections.items():
            self.waste_counter.add_detection(waste_type, count)
        
        # Kiểm tra threshold
        triggered_types = self.waste_counter.get_full_types()
        
        # Annotate frame
        annotated_frame = results[0].plot() if results else frame
        
        # Thêm thông tin counter lên frame
        self._draw_counter_info(annotated_frame)
        
        return annotated_frame, triggered_types
    
    def _draw_counter_info(self, frame: np.ndarray):
        """Vẽ thông tin counter lên frame"""
        y_offset = 30
        for waste_type, count in self.waste_counter.counts.items():
            if count > 0:  # Chỉ hiển thị loại có count > 0
                text = f"{waste_type.value}: {count}"
                color = (0, 255, 0) if count < self.waste_counter.threshold else (0, 0, 255)
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
    
    def handle_threshold_reached(self, waste_type: WasteType) -> Optional[PathfindingResult]:
        """
        Xử lý khi một loại rác đạt threshold
        
        Args:
            waste_type: Loại rác đạt threshold
            
        Returns:
            Kết quả tìm đường hoặc None
        """
        logger.info(f"Threshold reached for {waste_type.value}! Finding optimal route...")
        
        # Tìm đường tối ưu
        result = self.routing_system.find_optimal_route(waste_type)
        
        if result:
            # Visualize route
            fig = MapVisualizer.plot_route(self.routing_system, result, waste_type)
            plt.show()
            
            # Reset counter cho loại rác này
            self.waste_counter.reset_type(waste_type)
            
        return result


# ============ EXAMPLE DATA VÀ TEST FUNCTIONS ============

def create_sample_data() -> SmartRoutingSystem:
    """Tạo dữ liệu mẫu để test"""
    system = SmartRoutingSystem()
    
    # Tạo các bãi rác mẫu (khu vực TP.HCM)
    bins_data = [
        {
            "id": "BIN001",
            "lat": 10.762622, "lng": 106.660172,  # Quận 1
            "types": {WasteType.PLASTIC, WasteType.GLASS, WasteType.METAL},
            "max_cap": 1000, "current_cap": 200, "status": BinStatus.OK
        },
        {
            "id": "BIN002", 
            "lat": 10.775831, "lng": 106.700806,  # Quận 3
            "types": {WasteType.ORGANIC, WasteType.PAPER},
            "max_cap": 800, "current_cap": 600, "status": BinStatus.NEAR_FULL
        },
        {
            "id": "BIN003",
            "lat": 10.790123, "lng": 106.680234,  # Quận Bình Thạnh
            "types": {WasteType.PLASTIC, WasteType.CARDBOARD, WasteType.PAPER},
            "max_cap": 1200, "current_cap": 1200, "status": BinStatus.FULL
        },
        {
            "id": "BIN004",
            "lat": 10.745567, "lng": 106.690123,  # Quận 4
            "types": {WasteType.BATTERY, WasteType.METAL, WasteType.CLOTHES},
            "max_cap": 500, "current_cap": 100, "status": BinStatus.OK
        },
        {
            "id": "BIN005",
            "lat": 10.758945, "lng": 106.665432,  # Quận 1 (khác)
            "types": {WasteType.GENERAL, WasteType.SHOES, WasteType.CLOTHES},
            "max_cap": 900, "current_cap": 300, "status": BinStatus.OK
        }
    ]
    
    for bin_data in bins_data:
        bin_obj = WasteBin(
            id=bin_data["id"],
            location=GPSCoordinate(bin_data["lat"], bin_data["lng"]),
            supported_types=bin_data["types"],
            max_capacity=bin_data["max_cap"],
            current_capacity=bin_data["current_cap"],
            status=bin_data["status"]
        )
        system.add_waste_bin(bin_obj)
    
    # Tạo road network mẫu (simplified)
    road_segments = [
        # Từ robot position đến các bins
        RoadSegment(
            GPSCoordinate(10.770000, 106.680000),  # Robot start
            GPSCoordinate(10.762622, 106.660172),  # BIN001
            distance=2.1, travel_time=6.0,
            traffic_condition=TrafficCondition.CLEAR,
            road_quality=0.9
        ),
        RoadSegment(
            GPSCoordinate(10.770000, 106.680000),  # Robot start
            GPSCoordinate(10.775831, 106.700806),  # BIN002
            distance=2.3, travel_time=8.0,
            traffic_condition=TrafficCondition.MODERATE,
            road_quality=0.8
        ),
        RoadSegment(
            GPSCoordinate(10.770000, 106.680000),  # Robot start
            GPSCoordinate(10.790123, 106.680234),  # BIN003
            distance=2.2, travel_time=7.0,
            traffic_condition=TrafficCondition.HEAVY,
            road_quality=0.7
        ),
        RoadSegment(
            GPSCoordinate(10.770000, 106.680000),  # Robot start
            GPSCoordinate(10.745567, 106.690123),  # BIN004
            distance=2.8, travel_time=9.0,
            traffic_condition=TrafficCondition.CLEAR,
            road_quality=0.9
        ),
        RoadSegment(
            GPSCoordinate(10.770000, 106.680000),  # Robot start
            GPSCoordinate(10.758945, 106.665432),  # BIN005
            distance=1.9, travel_time=5.0,
            traffic_condition=TrafficCondition.MODERATE,
            road_quality=0.85
        ),
        # Kết nối giữa các bins
        RoadSegment(
            GPSCoordinate(10.762622, 106.660172),  # BIN001
            GPSCoordinate(10.775831, 106.700806),  # BIN002
            distance=4.2, travel_time=12.0,
            traffic_condition=TrafficCondition.HEAVY,
            road_quality=0.6
        ),
        RoadSegment(
            GPSCoordinate(10.775831, 106.700806),  # BIN002
            GPSCoordinate(10.790123, 106.680234),  # BIN003
            distance=2.5, travel_time=8.0,
            traffic_condition=TrafficCondition.CLEAR,
            road_quality=0.9
        )
    ]
    
    for segment in road_segments:
        system.add_road_segment(segment)
    
    # Set robot position
    robot_pos = GPSCoordinate(10.770000, 106.680000)  # Trung tâm khu vực
    system.update_robot_position(robot_pos)
    
    return system


def test_routing_system():
    """Test cơ bản cho hệ thống định tuyến"""
    logger.info("=== Testing Smart Routing System ===")
    
    # Tạo dữ liệu mẫu
    system = create_sample_data()
    
    # Test 1: Tìm đường cho plastic waste
    logger.info("\n--- Test 1: Plastic Waste Routing ---")
    result = system.find_optimal_route(WasteType.PLASTIC)
    
    if result:
        print(f"✅ Found route for plastic waste:")
        print(f"   Target bin: {result.target_bin.id}")
        print(f"   Distance: {result.total_distance:.2f} km")
        print(f"   ETA: {result.estimated_time:.1f} minutes")
        print(f"   Total cost: {result.total_cost:.2f}")
        print(f"   Path points: {len(result.path)}")
        
        # Visualize
        fig = MapVisualizer.plot_route(system, result, WasteType.PLASTIC)
        plt.savefig("/home/huynguyen/source/Trash-Detection/test_route_plastic.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("❌ No route found for plastic waste")
    
    # Test 2: Tìm đường cho organic waste
    logger.info("\n--- Test 2: Organic Waste Routing ---")
    result2 = system.find_optimal_route(WasteType.ORGANIC)
    
    if result2:
        print(f"✅ Found route for organic waste:")
        print(f"   Target bin: {result2.target_bin.id}")
        print(f"   Distance: {result2.total_distance:.2f} km")
        print(f"   ETA: {result2.estimated_time:.1f} minutes")
        print(f"   Total cost: {result2.total_cost:.2f}")
    else:
        print("❌ No route found for organic waste")
    
    # Test 3: Test update traffic và tìm lại
    logger.info("\n--- Test 3: Traffic Update Impact ---")
    old_cost = result.total_cost if result else float('inf')
    
    # Update traffic condition
    system.update_traffic_condition(
        GPSCoordinate(10.770000, 106.680000),
        GPSCoordinate(10.762622, 106.660172),
        TrafficCondition.HEAVY
    )
    
    result3 = system.find_optimal_route(WasteType.PLASTIC)
    new_cost = result3.total_cost if result3 else float('inf')
    
    print(f"   Cost before traffic update: {old_cost:.2f}")
    print(f"   Cost after traffic update: {new_cost:.2f}")
    print(f"   Impact: {((new_cost - old_cost) / old_cost * 100):.1f}% increase" if old_cost < float('inf') else "No comparison")


def test_real_time_detection():
    """Test hệ thống detection thời gian thực (giả lập)"""
    logger.info("\n=== Testing Real-Time Detection ===")
    
    # Tạo system
    system = create_sample_data()
    
    # Tạo fake detector (không cần model thật)
    class FakeDetector(RealTimeWasteDetector):
        def __init__(self, routing_system):
            self.routing_system = routing_system
            self.waste_counter = WasteCounter(threshold=5)  # Threshold thấp để test
        
        def simulate_detection(self, waste_type: WasteType, count: int = 1):
            """Giả lập detection"""
            self.waste_counter.add_detection(waste_type, count)
            triggered = self.waste_counter.get_full_types()
            
            for wtype in triggered:
                if wtype not in getattr(self, '_handled_types', set()):
                    result = self.handle_threshold_reached(wtype)
                    if not hasattr(self, '_handled_types'):
                        self._handled_types = set()
                    self._handled_types.add(wtype)
                    return result
            return None
    
    detector = FakeDetector(system)
    
    # Giả lập detections
    print("Simulating waste detections...")
    for i in range(7):
        print(f"Frame {i+1}: Detecting 1 plastic item")
        result = detector.simulate_detection(WasteType.PLASTIC, 1)
        
        if result:
            print(f"🚨 Threshold reached! Route found:")
            print(f"   Target: {result.target_bin.id}")
            print(f"   Distance: {result.total_distance:.2f}km")
            break
        
        time.sleep(0.1)  # Simulate frame rate


if __name__ == "__main__":
    # Chạy tests
    test_routing_system()
    test_real_time_detection()
    
    print("\n🎉 All tests completed!")
    print("📁 Check saved route visualization: test_route_plastic.png")
