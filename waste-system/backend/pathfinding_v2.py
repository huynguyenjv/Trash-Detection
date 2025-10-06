"""
A* Pathfinding Module
Implements A* algorithm for optimal route calculation
"""
import heapq
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Node class for A* algorithm
class Node:
    def __init__(self, position: Tuple[int, int], parent: Optional['Node'] = None, g: int = 0, h: int = 0,  f: int = 0):
        self.position = position
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = f  # Total cost

    def __eq__(self, other: Any) -> bool:
        return self.position == other.position

    def __lt__(self, other: 'Node') -> bool:
        return self.f < other.f

class Grid:
    # Considering moving from the centre of one grid cell to another
    # The final path will be adjusted to the actual lat/lon coordinates of the cell centres
    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size

        # Ho Chi Minh City bounds (approximate)
        self.bounds = {
            'min_lat': 10.70,
            'max_lat': 10.90,
            'min_lon': 106.60,
            'max_lon': 106.80
        }

        self.lat_per_slot = (self.bounds['max_lat'] - self.bounds['min_lat'])/self.grid_size
        self.lon_per_slot = (self.bounds['max_lon'] - self.bounds['min_lon'])/self.grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)  # 0 for free cell, 1 for obstacle
        self._add_obstacles()

    def _add_obstacles(self):
        """Add some obstacles to the grid to simulate real world"""
        # Add some blocked areas (simplified for demo)
        # In real implementation, this would be based on actual map data
        
        # Add some river/water obstacles
        self.grid[45:55, 20:80] = 1  # Horizontal water body
        
        # Add some building clusters
        for i in range(5, 95, 20):
            for j in range(5, 95, 20):
                # Random small building clusters
                if np.random.random() > 0.7:
                    self.grid[i:i+3, j:j+3] = 0

    def latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        if not (self.bounds['min_lat'] <= lat <= self.bounds['max_lat']) or not (self.bounds['min_lon'] <= lon <= self.bounds['max_lon']):
            raise ValueError("Latitude or Longitude out of bounds")

        row = int((lat - self.bounds['min_lat']) / self.lat_per_slot) - 1
        col = int((lon - self.bounds['min_lon']) / self.lon_per_slot) - 1 

        return row, col                                                                                                                                   
    
    def grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        if not (0 <= row < self.grid_size) or not (0 <= col < self.grid_size):
            raise ValueError("Grid position out of bounds")

        lat = self.bounds['min_lat'] + (row + 0.5) * self.lat_per_slot
        lon = self.bounds['min_lon'] + (col + 0.5) * self.lon_per_slot

        return lat, lon

class AStarAlgorithm:
    def __init__(self, grid: Grid):
        
        self.open_list:     List[Node] = []
        self.closed_list:   List[Node] = []
        self.grid = grid  

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        # Using Manhattan distance as heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def explore_best_neighbours(self, current_node: Node, goal_node: Node) -> Node:
        neighbours: List[Node] = []

        # 8-directional movement
        directions: List[Tuple[int, int]]= []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                directions.append((dx, dy))
        
        provisional_g = current_node.g + 1
        provisional_h = 0
        provisional_f = 0

        for direction in directions:
            potential_position = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

            # Check if within grid bounds
            if (0 <= potential_position[0] < self.grid.grid_size) and (0 <= potential_position[1] < self.grid.grid_size):
                # Check if not an obstacle
                if self.grid.grid[potential_position[0]][potential_position[1]] == 0:
                    provisional_h = self.heuristic(potential_position, goal_node.position)
                    provisional_f = provisional_g + provisional_h
                    tmp_node = Node(position=potential_position, parent=current_node, g=provisional_g, h=provisional_h, f=provisional_f)
                    if tmp_node not in self.closed_list:
                        heapq.heappush(neighbours, tmp_node)

        return heapq.heappop(neighbours) if neighbours else current_node
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[float, float]]:
        start_node = Node(position=start)
        goal_node = Node(position=goal)

        heapq.heappush(self.open_list, start_node)

        while self.open_list:
            current_node = heapq.heappop(self.open_list)
            self.closed_list.append(current_node)

            # Check if we reached the goal
            if current_node == goal_node:
                return self.reconstruct_path(current_node)
            
            heapq.heappush(self.open_list, self.explore_best_neighbours(current_node, goal_node))

        return []  # No path found
    
    def reconstruct_path(self, current_node: Node) -> List[Tuple[float, float]]:
        path = []
        while current_node.parent:
            path.append(self.grid.grid_to_latlon(current_node.position[0], current_node.position[1]))
            current_node = current_node.parent
        path.reverse()
        return path
    

if __name__ == "__main__":
    grid = Grid(grid_size=10000)
    print(grid.lat_per_slot, grid.lon_per_slot)
    astar = AStarAlgorithm(grid)

    start_lat, start_lon = 10.75, 106.65
    goal_lat, goal_lon = 10.85, 106.75

    start_grid = grid.latlon_to_grid(start_lat, start_lon)
    goal_grid = grid.latlon_to_grid(goal_lat, goal_lon)

    path = astar.find_path(start_grid, goal_grid)
    print("Calculated Path (lat, lon):")
    for coord in path:
        print(coord)