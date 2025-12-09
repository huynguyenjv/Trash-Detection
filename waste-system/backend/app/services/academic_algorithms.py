"""
Custom Routing Algorithms for Academic Paper
Các thuật toán tự implement (không dùng API) để viết paper/thesis

Bao gồm:
1. Dijkstra's Algorithm - Shortest path
2. A* Algorithm - Heuristic shortest path  
3. K-Nearest Neighbors - Find nearest bins
4. Greedy Algorithm - Route optimization
5. Dynamic Programming - Optimal collection route
"""

import heapq
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class RoadType(Enum):
    """Loại đường"""
    HIGHWAY = 1      # Cao tốc
    MAIN_ROAD = 2    # Đường chính
    STREET = 3       # Đường phố
    ALLEY = 4        # Ngõ/hẻm


@dataclass
class RoadSegment:
    """Đoạn đường giữa 2 node"""
    start_node: int
    end_node: int
    distance: float  # km
    road_type: RoadType
    speed_limit: float  # km/h
    traffic_factor: float = 1.0  # 1.0 = no traffic, 2.0 = heavy traffic
    
    @property
    def travel_time(self) -> float:
        """Thời gian đi (phút)"""
        effective_speed = self.speed_limit / self.traffic_factor
        return (self.distance / effective_speed) * 60
    
    @property
    def weight(self) -> float:
        """Trọng số (có thể là distance hoặc time)"""
        return self.distance


class RoadNetwork:
    """
    Đồ thị mạng lưới đường bộ
    Graph representation of road network
    """
    
    def __init__(self):
        self.nodes: Dict[int, Tuple[float, float]] = {}  # node_id -> (lat, lng)
        self.edges: Dict[int, List[RoadSegment]] = {}  # node_id -> [segments]
        self.node_names: Dict[int, str] = {}  # node_id -> name
    
    def add_node(self, node_id: int, lat: float, lng: float, name: str = ""):
        """Thêm node (giao lộ) vào đồ thị"""
        self.nodes[node_id] = (lat, lng)
        self.edges[node_id] = []
        self.node_names[node_id] = name or f"Node {node_id}"
    
    def add_edge(self, segment: RoadSegment):
        """Thêm cạnh (đoạn đường) vào đồ thị"""
        if segment.start_node not in self.edges:
            self.edges[segment.start_node] = []
        self.edges[segment.start_node].append(segment)
    
    def add_bidirectional_edge(self, node1: int, node2: int, 
                               distance: float, road_type: RoadType, 
                               speed_limit: float = 40.0):
        """Thêm đường 2 chiều"""
        # Forward
        self.add_edge(RoadSegment(node1, node2, distance, road_type, speed_limit))
        # Backward
        self.add_edge(RoadSegment(node2, node1, distance, road_type, speed_limit))
    
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Lấy danh sách neighbor và distance"""
        return [(seg.end_node, seg.weight) for seg in self.edges.get(node_id, [])]
    
    def haversine_distance(self, node1: int, node2: int) -> float:
        """
        Tính khoảng cách đường chim bay (heuristic cho A*)
        Haversine formula
        """
        if node1 not in self.nodes or node2 not in self.nodes:
            return float('inf')
        
        lat1, lng1 = self.nodes[node1]
        lat2, lng2 = self.nodes[node2]
        
        R = 6371.0  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = math.sin(delta_lat / 2)**2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


class DijkstraAlgorithm:
    """
    Dijkstra's Algorithm - Tìm đường đi ngắn nhất
    
    Paper: E. W. Dijkstra, "A note on two problems in connexion with graphs" (1959)
    
    Time Complexity: O((V + E) log V) with binary heap
    Space Complexity: O(V)
    """
    
    @staticmethod
    def shortest_path(
        graph: RoadNetwork,
        start: int,
        goal: int
    ) -> Tuple[List[int], float]:
        """
        Tìm đường đi ngắn nhất từ start đến goal
        
        Returns:
            (path, total_distance)
        """
        # Initialize
        distances = {node: float('inf') for node in graph.nodes}
        distances[start] = 0
        previous = {node: None for node in graph.nodes}
        
        # Priority queue: (distance, node)
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Goal reached
            if current == goal:
                break
            
            # Check all neighbors
            for neighbor, edge_weight in graph.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                new_distance = current_dist + edge_weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_distance, neighbor))
        
        # Reconstruct path
        if distances[goal] == float('inf'):
            return [], float('inf')
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return path, distances[goal]


class AStarAlgorithm:
    """
    A* Algorithm - Tìm đường đi ngắn nhất với heuristic
    
    Paper: P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic 
           Determination of Minimum Cost Paths" (1968)
    
    Ưu điểm so với Dijkstra:
    - Nhanh hơn (sử dụng heuristic)
    - Explore ít node hơn
    - Vẫn guarantee optimal nếu heuristic admissible
    
    Time Complexity: O(b^d) worst case, thực tế nhanh hơn Dijkstra
    Space Complexity: O(b^d)
    """
    
    @staticmethod
    def shortest_path(
        graph: RoadNetwork,
        start: int,
        goal: int,
        heuristic_weight: float = 1.0
    ) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        A* search
        
        Args:
            heuristic_weight: Trọng số cho heuristic (1.0 = optimal, >1 = faster but may not optimal)
        
        Returns:
            (path, total_distance, stats)
        """
        # g(n): Cost from start to n
        g_score = {node: float('inf') for node in graph.nodes}
        g_score[start] = 0
        
        # f(n) = g(n) + h(n): Estimated total cost
        f_score = {node: float('inf') for node in graph.nodes}
        f_score[start] = heuristic_weight * graph.haversine_distance(start, goal)
        
        previous = {node: None for node in graph.nodes}
        
        # Priority queue: (f_score, node)
        open_set = [(f_score[start], start)]
        closed_set = set()
        
        nodes_explored = 0
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            nodes_explored += 1
            closed_set.add(current)
            
            # Goal reached
            if current == goal:
                break
            
            # Explore neighbors
            for neighbor, edge_weight in graph.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + edge_weight
                
                if tentative_g < g_score[neighbor]:
                    previous[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # f(n) = g(n) + h(n)
                    h = heuristic_weight * graph.haversine_distance(neighbor, goal)
                    f_score[neighbor] = tentative_g + h
                    
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Reconstruct path
        if g_score[goal] == float('inf'):
            return [], float('inf'), {"nodes_explored": nodes_explored}
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        stats = {
            "nodes_explored": nodes_explored,
            "total_nodes": len(graph.nodes),
            "exploration_ratio": nodes_explored / len(graph.nodes)
        }
        
        return path, g_score[goal], stats


class NearestBinFinder:
    """
    K-Nearest Neighbors cho waste bins
    
    Algorithms:
    1. Brute Force: O(n)
    2. KD-Tree: O(log n) average
    3. Ball Tree: O(log n) for high dimensions
    """
    
    @staticmethod
    def brute_force_nearest(
        user_location: Tuple[float, float],
        bins: List[Dict[str, Any]],
        k: int = 1
    ) -> List[Tuple[Dict, float]]:
        """
        Brute force search - O(n)
        
        Returns:
            List of (bin, distance) sorted by distance
        """
        distances = []
        
        for bin_data in bins:
            bin_loc = (bin_data['latitude'], bin_data['longitude'])
            dist = NearestBinFinder._haversine(user_location, bin_loc)
            distances.append((bin_data, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        return distances[:k]
    
    @staticmethod
    def nearest_with_routing(
        graph: RoadNetwork,
        user_node: int,
        bin_nodes: List[int],
        algorithm: str = "astar"
    ) -> Tuple[int, List[int], float]:
        """
        Tìm thùng gần nhất dựa trên routing thực tế
        
        Args:
            algorithm: "dijkstra" or "astar"
        
        Returns:
            (nearest_bin_node, path, distance)
        """
        best_bin = None
        best_path = []
        best_distance = float('inf')
        
        for bin_node in bin_nodes:
            if algorithm == "astar":
                path, distance, _ = AStarAlgorithm.shortest_path(graph, user_node, bin_node)
            else:
                path, distance = DijkstraAlgorithm.shortest_path(graph, user_node, bin_node)
            
            if distance < best_distance:
                best_distance = distance
                best_bin = bin_node
                best_path = path
        
        return best_bin, best_path, best_distance
    
    @staticmethod
    def _haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Haversine distance"""
        lat1, lng1 = coord1
        lat2, lng2 = coord2
        
        R = 6371.0
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = math.sin(delta_lat / 2)**2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


class WasteCollectionOptimizer:
    """
    Tối ưu lộ trình thu gom rác
    
    Algorithms:
    1. Greedy Algorithm - O(n²)
    2. Dynamic Programming - O(n² * 2^n)
    3. Nearest Neighbor Heuristic - O(n²)
    """
    
    @staticmethod
    def greedy_route(
        graph: RoadNetwork,
        start_node: int,
        bin_nodes: List[int],
        end_node: Optional[int] = None
    ) -> Tuple[List[int], float]:
        """
        Greedy Algorithm - Chọn thùng gần nhất chưa visit
        
        Time Complexity: O(n²) where n = số thùng
        
        Paper topic: "Greedy Heuristic for Vehicle Routing Problem"
        """
        route = [start_node]
        total_distance = 0
        remaining_bins = set(bin_nodes)
        current = start_node
        
        while remaining_bins:
            # Tìm thùng gần nhất chưa visit
            nearest_bin = None
            nearest_distance = float('inf')
            
            for bin_node in remaining_bins:
                path, distance = DijkstraAlgorithm.shortest_path(graph, current, bin_node)
                
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_bin = bin_node
            
            if nearest_bin is None:
                break
            
            # Visit bin
            route.append(nearest_bin)
            total_distance += nearest_distance
            remaining_bins.remove(nearest_bin)
            current = nearest_bin
        
        # Return to end node
        if end_node and end_node != current:
            path, distance = DijkstraAlgorithm.shortest_path(graph, current, end_node)
            route.append(end_node)
            total_distance += distance
        
        return route, total_distance
    
    @staticmethod
    def nearest_neighbor_tsp(
        distance_matrix: List[List[float]],
        start_idx: int = 0
    ) -> Tuple[List[int], float]:
        """
        Nearest Neighbor Heuristic for TSP
        
        Time Complexity: O(n²)
        
        Suitable for paper: "Heuristic Approaches to Traveling Salesman Problem"
        """
        n = len(distance_matrix)
        visited = {start_idx}
        route = [start_idx]
        total_distance = 0
        current = start_idx
        
        for _ in range(n - 1):
            nearest = None
            nearest_dist = float('inf')
            
            for next_node in range(n):
                if next_node not in visited:
                    dist = distance_matrix[current][next_node]
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = next_node
            
            if nearest is None:
                break
            
            visited.add(nearest)
            route.append(nearest)
            total_distance += nearest_dist
            current = nearest
        
        # Return to start
        total_distance += distance_matrix[current][start_idx]
        route.append(start_idx)
        
        return route, total_distance
    
    @staticmethod
    def dynamic_programming_tsp(
        distance_matrix: List[List[float]],
        start_idx: int = 0
    ) -> Tuple[List[int], float]:
        """
        Dynamic Programming solution for TSP (Held-Karp Algorithm)
        
        Time Complexity: O(n² * 2^n)
        Space Complexity: O(n * 2^n)
        
        Paper: "The Traveling-Salesman Problem and Minimum Spanning Trees" (1970)
        
        Note: Chỉ dùng cho n <= 20 (exponential complexity)
        """
        n = len(distance_matrix)
        
        if n > 20:
            raise ValueError("DP-TSP chỉ phù hợp cho n <= 20. Dùng heuristic cho n lớn.")
        
        # dp[mask][i] = minimum distance to visit all nodes in mask ending at i
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[None] * n for _ in range(1 << n)]
        
        # Base case
        dp[1 << start_idx][start_idx] = 0
        
        # Fill DP table
        for mask in range(1 << n):
            for current in range(n):
                if not (mask & (1 << current)):
                    continue
                
                if dp[mask][current] == float('inf'):
                    continue
                
                # Try all next nodes
                for next_node in range(n):
                    if mask & (1 << next_node):
                        continue
                    
                    new_mask = mask | (1 << next_node)
                    new_dist = dp[mask][current] + distance_matrix[current][next_node]
                    
                    if new_dist < dp[new_mask][next_node]:
                        dp[new_mask][next_node] = new_dist
                        parent[new_mask][next_node] = current
        
        # Find best ending node
        full_mask = (1 << n) - 1
        best_end = min(range(n), key=lambda i: dp[full_mask][i] + distance_matrix[i][start_idx])
        min_distance = dp[full_mask][best_end] + distance_matrix[best_end][start_idx]
        
        # Reconstruct path
        path = []
        mask = full_mask
        current = best_end
        
        while current is not None:
            path.append(current)
            prev = parent[mask][current]
            if prev is not None:
                mask ^= (1 << current)
            current = prev
        
        path.reverse()
        path.append(start_idx)  # Return to start
        
        return path, min_distance


class AlgorithmComparison:
    """
    So sánh performance các thuật toán
    Để viết phần Results & Discussion trong paper
    """
    
    @staticmethod
    def compare_shortest_path_algorithms(
        graph: RoadNetwork,
        start: int,
        goal: int
    ) -> Dict[str, Any]:
        """
        So sánh Dijkstra vs A*
        
        Metrics:
        - Execution time
        - Number of nodes explored
        - Path length
        - Memory usage
        """
        import time
        
        results = {}
        
        # Dijkstra
        start_time = time.time()
        dijkstra_path, dijkstra_dist = DijkstraAlgorithm.shortest_path(graph, start, goal)
        dijkstra_time = time.time() - start_time
        
        results['dijkstra'] = {
            'time': dijkstra_time,
            'distance': dijkstra_dist,
            'path_length': len(dijkstra_path),
            'nodes_explored': len(graph.nodes)  # Dijkstra explores all reachable
        }
        
        # A*
        start_time = time.time()
        astar_path, astar_dist, astar_stats = AStarAlgorithm.shortest_path(graph, start, goal)
        astar_time = time.time() - start_time
        
        results['astar'] = {
            'time': astar_time,
            'distance': astar_dist,
            'path_length': len(astar_path),
            'nodes_explored': astar_stats['nodes_explored'],
            'exploration_ratio': astar_stats['exploration_ratio']
        }
        
        # Speedup
        results['speedup'] = {
            'time_speedup': dijkstra_time / astar_time if astar_time > 0 else 0,
            'exploration_reduction': 1 - astar_stats['exploration_ratio']
        }
        
        return results
    
    @staticmethod
    def compare_tsp_algorithms(
        distance_matrix: List[List[float]]
    ) -> Dict[str, Any]:
        """
        So sánh Greedy vs DP cho TSP
        """
        import time
        
        results = {}
        
        # Nearest Neighbor (Greedy)
        start_time = time.time()
        nn_route, nn_distance = WasteCollectionOptimizer.nearest_neighbor_tsp(distance_matrix)
        nn_time = time.time() - start_time
        
        results['nearest_neighbor'] = {
            'time': nn_time,
            'distance': nn_distance,
            'route': nn_route
        }
        
        # DP (nếu n nhỏ)
        n = len(distance_matrix)
        if n <= 15:
            start_time = time.time()
            dp_route, dp_distance = WasteCollectionOptimizer.dynamic_programming_tsp(distance_matrix)
            dp_time = time.time() - start_time
            
            results['dynamic_programming'] = {
                'time': dp_time,
                'distance': dp_distance,
                'route': dp_route
            }
            
            results['optimality_gap'] = (nn_distance - dp_distance) / dp_distance * 100
        
        return results
