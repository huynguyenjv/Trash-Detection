"""
Custom Route Optimization Algorithms
Thuật toán tự code để chọn best route (không dựa vào API mặc định)

Các thuật toán:
1. Weighted Score (distance + time)
2. Dijkstra-inspired (shortest distance priority)
3. A*-inspired (heuristic-based)
4. Multi-criteria (distance, time, traffic, fuel)
"""

from typing import List, Dict, Any, Optional
import logging
import math

logger = logging.getLogger(__name__)


class RouteOptimizationStrategy:
    """Base class cho các thuật toán chọn route"""
    
    def calculate_score(self, route: Dict[str, Any]) -> float:
        """
        Tính điểm cho route
        Lower score = Better route
        """
        raise NotImplementedError
    
    def select_best_route(self, routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Chọn route tốt nhất từ danh sách"""
        if not routes:
            return None
        
        best_route = None
        best_score = float('inf')
        
        for route in routes:
            score = self.calculate_score(route)
            if score < best_score:
                best_score = score
                best_route = route
        
        return best_route


class WeightedScoreStrategy(RouteOptimizationStrategy):
    """
    Weighted Score Algorithm
    
    Score = (distance × w1) + (time × w2)
    
    Ưu điểm: 
    - Cân bằng giữa khoảng cách và thời gian
    - Tuỳ chỉnh weights theo use case
    
    Use case:
    - w1=0.7, w2=0.3: Ưu tiên khoảng cách (xe thu gom rác)
    - w1=0.3, w2=0.7: Ưu tiên thời gian (xe cấp cứu)
    """
    
    def __init__(self, distance_weight: float = 0.7, time_weight: float = 0.3):
        self.distance_weight = distance_weight
        self.time_weight = time_weight
        
        # Normalize weights
        total = distance_weight + time_weight
        self.distance_weight /= total
        self.time_weight /= total
    
    def calculate_score(self, route: Dict[str, Any]) -> float:
        distance_km = route.get("distance_km", 0)
        duration_min = route.get("duration_minutes", 0)
        
        score = (distance_km * self.distance_weight) + (duration_min * self.time_weight)
        return round(score, 2)


class DijkstraInspiredStrategy(RouteOptimizationStrategy):
    """
    Dijkstra-inspired: Shortest Distance Priority
    
    Algorithm:
    1. Chỉ xét distance_km (giống Dijkstra chỉ xét edge weight)
    2. Bỏ qua time/traffic
    
    Use case: Tiết kiệm nhiên liệu
    """
    
    def calculate_score(self, route: Dict[str, Any]) -> float:
        return route.get("distance_km", float('inf'))


class AStarInspiredStrategy(RouteOptimizationStrategy):
    """
    A*-inspired: Distance + Heuristic
    
    Algorithm:
    Score = g(n) + h(n)
    - g(n) = actual distance
    - h(n) = estimated remaining cost (time penalty)
    
    Use case: Cân bằng distance và time với heuristic
    """
    
    def calculate_score(self, route: Dict[str, Any]) -> float:
        g = route.get("distance_km", 0)  # Actual cost
        h = route.get("duration_minutes", 0) * 0.1  # Heuristic (time penalty)
        
        return g + h


class MultiCriteriaStrategy(RouteOptimizationStrategy):
    """
    Multi-Criteria Decision Making (MCDM)
    
    Score = Σ(criterion_i × weight_i)
    
    Criteria:
    1. Distance (km)
    2. Time (minutes)
    3. Traffic factor (estimated from speed)
    4. Fuel consumption (estimated from distance & speed)
    
    Use case: Real-world optimization với nhiều yếu tố
    """
    
    def calculate_score(self, route: Dict[str, Any]) -> float:
        distance_km = route.get("distance_km", 0)
        duration_min = route.get("duration_minutes", 0)
        
        # Estimate traffic factor: normal speed vs actual speed
        # Normal speed for car in city: ~30 km/h
        # If actual speed < normal → high traffic
        if duration_min > 0:
            actual_speed = (distance_km / duration_min) * 60  # km/h
            normal_speed = 30
            traffic_factor = max(0, normal_speed - actual_speed) / 10
        else:
            traffic_factor = 0
        
        # Estimate fuel consumption (L/100km * distance)
        # Average: 8L/100km in city
        fuel_cost = (distance_km / 100) * 8 * 25000  # VND (gas price ~25k/L)
        fuel_score = fuel_cost / 10000  # Normalize
        
        # Combined score
        score = (
            distance_km * 0.4 +          # 40% distance
            duration_min * 0.3 +         # 30% time
            traffic_factor * 0.2 +       # 20% traffic
            fuel_score * 0.1             # 10% fuel
        )
        
        return round(score, 2)


class GreedyNearestStrategy(RouteOptimizationStrategy):
    """
    Greedy Algorithm: Chọn route gần nhất ngay lập tức
    
    Algorithm:
    - So sánh distance only
    - O(1) complexity (không cần evaluate tất cả)
    
    Use case: Fast decision, real-time
    """
    
    def calculate_score(self, route: Dict[str, Any]) -> float:
        return route.get("distance_km", float('inf'))
    
    def select_best_route(self, routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Override: Chỉ lấy route đầu tiên có distance nhỏ nhất"""
        if not routes:
            return None
        
        return min(routes, key=lambda r: r.get("distance_km", float('inf')))


class RouteOptimizer:
    """
    Main optimizer class
    Chọn strategy và optimize routes
    """
    
    STRATEGIES = {
        "weighted": WeightedScoreStrategy,
        "dijkstra": DijkstraInspiredStrategy,
        "astar": AStarInspiredStrategy,
        "multi_criteria": MultiCriteriaStrategy,
        "greedy": GreedyNearestStrategy
    }
    
    def __init__(self, strategy: str = "weighted"):
        """
        Args:
            strategy: Algorithm name
                - "weighted": Weighted distance + time (default)
                - "dijkstra": Shortest distance only
                - "astar": Distance + heuristic
                - "multi_criteria": Multiple factors
                - "greedy": Fast nearest
        """
        if strategy not in self.STRATEGIES:
            logger.warning(f"Unknown strategy '{strategy}', using 'weighted'")
            strategy = "weighted"
        
        self.strategy_name = strategy
        self.strategy = self.STRATEGIES[strategy]()
        logger.info(f"🧮 Route optimizer initialized with strategy: {strategy}")
    
    def select_best_route(
        self,
        routes: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Chọn best route sử dụng thuật toán đã config
        
        Args:
            routes: List of route data from API
            verbose: Log chi tiết
        
        Returns:
            Best route với thêm field:
            - algorithm_used: Tên thuật toán
            - route_score: Điểm số
            - total_alternatives: Số routes đã evaluate
        """
        if not routes:
            logger.warning("No routes to evaluate")
            return None
        
        if verbose:
            logger.info(f"🔍 Evaluating {len(routes)} routes using {self.strategy_name}...")
        
        # Calculate scores for all routes
        route_scores = []
        for i, route in enumerate(routes):
            score = self.strategy.calculate_score(route)
            route_scores.append((route, score))
            
            if verbose:
                logger.info(
                    f"  Route {i+1}: {route.get('distance_km')}km, "
                    f"{route.get('duration_minutes')}min → Score: {score}"
                )
        
        # Select best
        best_route, best_score = min(route_scores, key=lambda x: x[1])
        
        # Add metadata
        result = best_route.copy()
        result["algorithm_used"] = self.strategy_name
        result["route_score"] = best_score
        result["total_alternatives"] = len(routes)
        
        if verbose:
            logger.info(
                f"✅ Selected best route: {result.get('distance_km')}km, "
                f"score={best_score} (algorithm={self.strategy_name})"
            )
        
        return result
    
    def optimize_multi_destination(
        self,
        origin: tuple,
        destinations: List[Dict[str, Any]],
        route_getter_func,
        vehicle: str = "car"
    ) -> List[Dict[str, Any]]:
        """
        Optimize route visiting multiple destinations (TSP-like)
        
        Args:
            origin: (lat, lng)
            destinations: List of destinations with lat/lng
            route_getter_func: Function to get route between 2 points
            vehicle: Vehicle type
        
        Returns:
            Optimized order of destinations
        """
        if not destinations:
            return []
        
        visited = set()
        current_pos = origin
        optimized_route = []
        
        logger.info(f"🚛 Optimizing route for {len(destinations)} destinations...")
        
        # Greedy nearest neighbor algorithm
        while len(visited) < len(destinations):
            best_dest = None
            best_route = None
            best_score = float('inf')
            
            for i, dest in enumerate(destinations):
                if i in visited:
                    continue
                
                # Get route to this destination
                route = route_getter_func(
                    origin=current_pos,
                    destination=(dest["latitude"], dest["longitude"]),
                    vehicle=vehicle
                )
                
                if route:
                    score = self.strategy.calculate_score(route)
                    if score < best_score:
                        best_score = score
                        best_route = route
                        best_dest = (i, dest)
            
            if best_dest:
                idx, dest = best_dest
                visited.add(idx)
                optimized_route.append({
                    "destination": dest,
                    "route": best_route,
                    "score": best_score
                })
                current_pos = (dest["latitude"], dest["longitude"])
        
        logger.info(f"✅ Optimized route completed: {len(optimized_route)} stops")
        return optimized_route


# Convenience functions

def select_best_route_weighted(routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Quick function: Weighted strategy"""
    optimizer = RouteOptimizer(strategy="weighted")
    return optimizer.select_best_route(routes)


def select_best_route_shortest(routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Quick function: Shortest distance only"""
    optimizer = RouteOptimizer(strategy="dijkstra")
    return optimizer.select_best_route(routes)


def select_best_route_astar(routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Quick function: A* heuristic"""
    optimizer = RouteOptimizer(strategy="astar")
    return optimizer.select_best_route(routes)
