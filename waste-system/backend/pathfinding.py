"""
A* Pathfinding Module
Implements A* algorithm for optimal route calculation
"""
import heapq
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class Node:
    """Node class for A* pathfinding"""
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0, parent=None):
        self.position = position
        self.g_cost = g_cost  # Distance from start
        self.h_cost = h_cost  # Heuristic distance to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class AStarPathfinder:
    def __init__(self, grid_size: int = 100):
        """
        Initialize A* pathfinder
        Args:
            grid_size: Size of the grid for pathfinding (100x100 default)
        """
        self.grid_size = grid_size
        
        # Ho Chi Minh City bounds (approximate)
        self.bounds = {
            'min_lat': 10.70,
            'max_lat': 10.90,
            'min_lon': 106.60,
            'max_lon': 106.80
        }
        
        # Create a simple grid (can be enhanced with real road data)
        self.grid = np.ones((grid_size, grid_size), dtype=int)  # 1 = passable, 0 = blocked
        
        # Add some obstacles (buildings, water, etc.)
        self._add_obstacles()
    
    def _add_obstacles(self):
        """Add some obstacles to the grid to simulate real world"""
        # Add some blocked areas (simplified for demo)
        # In real implementation, this would be based on actual map data
        
        # Add some river/water obstacles
        self.grid[45:55, 20:80] = 0  # Horizontal water body
        
        # Add some building clusters
        for i in range(5, 95, 20):
            for j in range(5, 95, 20):
                # Random small building clusters
                if np.random.random() > 0.7:
                    self.grid[i:i+3, j:j+3] = 0
    
    def lat_lon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert latitude/longitude to grid coordinates"""
        # Normalize to 0-1 range
        lat_norm = (lat - self.bounds['min_lat']) / (self.bounds['max_lat'] - self.bounds['min_lat'])
        lon_norm = (lon - self.bounds['min_lon']) / (self.bounds['max_lon'] - self.bounds['min_lon'])
        
        # Convert to grid coordinates
        grid_x = int(lat_norm * (self.grid_size - 1))
        grid_y = int(lon_norm * (self.grid_size - 1))
        
        # Clamp to grid bounds
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))
        
        return (grid_x, grid_y)
    
    def grid_to_lat_lon(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to latitude/longitude"""
        # Normalize from grid to 0-1 range
        lat_norm = grid_x / (self.grid_size - 1)
        lon_norm = grid_y / (self.grid_size - 1)
        
        # Convert to actual coordinates
        lat = self.bounds['min_lat'] + lat_norm * (self.bounds['max_lat'] - self.bounds['min_lat'])
        lon = self.bounds['min_lon'] + lon_norm * (self.bounds['max_lon'] - self.bounds['min_lon'])
        
        return (lat, lon)
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Manhattan distance)"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        x, y = position
        neighbors = []
        
        # 8-directional movement
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x, new_y = x + dx, y + dy
                
                # Check bounds
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                    # Check if passable
                    if self.grid[new_x, new_y] == 1:
                        neighbors.append((new_x, new_y))
        
        return neighbors
    
    def find_path(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find path using A* algorithm
        Args:
            start_pos: Starting grid position
            goal_pos: Goal grid position
        Returns:
            List of grid positions forming the path
        """
        if self.grid[start_pos[0], start_pos[1]] == 0 or self.grid[goal_pos[0], goal_pos[1]] == 0:
            return []  # Start or goal is blocked
        
        open_set = []
        closed_set = set()
        
        start_node = Node(start_pos, 0, self.heuristic(start_pos, goal_pos))
        heapq.heappush(open_set, start_node)
        
        node_dict = {start_pos: start_node}
        
        while open_set:
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position
            
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            
            # Goal reached
            if current_pos == goal_pos:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]  # Reverse to get path from start to goal
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current_pos):
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate costs
                movement_cost = 1.4 if abs(neighbor_pos[0] - current_pos[0]) + abs(neighbor_pos[1] - current_pos[1]) == 2 else 1.0
                g_cost = current_node.g_cost + movement_cost
                h_cost = self.heuristic(neighbor_pos, goal_pos)
                
                # Check if we found a better path to this neighbor
                if neighbor_pos in node_dict:
                    existing_node = node_dict[neighbor_pos]
                    if g_cost < existing_node.g_cost:
                        existing_node.g_cost = g_cost
                        existing_node.f_cost = g_cost + h_cost
                        existing_node.parent = current_node
                else:
                    neighbor_node = Node(neighbor_pos, g_cost, h_cost, current_node)
                    node_dict[neighbor_pos] = neighbor_node
                    heapq.heappush(open_set, neighbor_node)
        
        return []  # No path found
    
    def calculate_route(self, start_lat: float, start_lon: float, 
                       end_lat: float, end_lon: float) -> Dict[str, Any]:
        """
        Calculate route from start to end coordinates
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            end_lat: Ending latitude
            end_lon: Ending longitude
        Returns:
            Route information with path, distance, and duration
        """
        try:
            # Convert to grid coordinates
            start_grid = self.lat_lon_to_grid(start_lat, start_lon)
            end_grid = self.lat_lon_to_grid(end_lat, end_lon)
            
            # Find path
            grid_path = self.find_path(start_grid, end_grid)
            
            if not grid_path:
                # No path found, return direct line
                return {
                    'path': [[start_lat, start_lon], [end_lat, end_lon]],
                    'distance': self.calculate_distance(start_lat, start_lon, end_lat, end_lon),
                    'duration': 300,  # Default 5 minutes
                    'found_path': False
                }
            
            # Convert grid path back to lat/lon coordinates
            lat_lon_path = []
            total_distance = 0
            
            for i, grid_pos in enumerate(grid_path):
                lat, lon = self.grid_to_lat_lon(grid_pos[0], grid_pos[1])
                lat_lon_path.append([lat, lon])
                
                # Calculate distance to next point
                if i > 0:
                    prev_lat, prev_lon = lat_lon_path[i-1]
                    total_distance += self.calculate_distance(prev_lat, prev_lon, lat, lon)
            
            # Estimate duration (average walking speed ~5 km/h)
            duration_minutes = max(1, int((total_distance / 1000) * 12))  # 12 minutes per km
            
            return {
                'path': lat_lon_path,
                'distance': total_distance,
                'duration': duration_minutes * 60,  # Convert to seconds
                'found_path': True
            }
            
        except Exception as e:
            print(f"Error calculating route: {e}")
            # Return direct line as fallback
            return {
                'path': [[start_lat, start_lon], [end_lat, end_lon]],
                'distance': self.calculate_distance(start_lat, start_lon, end_lat, end_lon),
                'duration': 300,
                'found_path': False
            }
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance


# Global pathfinder instance
pathfinder = None

def get_pathfinder() -> AStarPathfinder:
    """Get global pathfinder instance"""
    global pathfinder
    if pathfinder is None:
        pathfinder = AStarPathfinder()
    return pathfinder
