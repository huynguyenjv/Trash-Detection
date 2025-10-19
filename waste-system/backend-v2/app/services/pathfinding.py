"""
A* Pathfinding Module
Find shortest path from waste locations to nearest bins
"""

from typing import List, Tuple, Dict, Any, Optional
import heapq
import math


class AStarPathfinder:
    def __init__(self, grid_size: Tuple[int, int] = (20, 20)):
        """
        Initialize A* pathfinder
        
        Args:
            grid_size: (width, height) of grid
        """
        self.grid_width, self.grid_height = grid_size
        
        # Bin locations (grid coordinates)
        self.bins = {
            'organic': [(0, 0), (19, 19)],
            'recyclable': [(0, 19), (19, 0)],
            'hazardous': [(9, 9)],
            'other': [(10, 10)]
        }
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbor positions (4-directional)"""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to goal using A*
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of positions forming the path, or None if no path found
        """
        frontier = []
        heapq.heappush(frontier, (0, start))
        
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None
    
    def find_nearest_bin_for_each(self, starts: List[Tuple[int, int]], 
                                  waste_type: str = None) -> Dict[str, Any]:
        """
        Find nearest bin and path for each start position
        
        Args:
            starts: List of start positions [(x1, y1), (x2, y2), ...]
            waste_type: Specific waste type, or None for all bins
            
        Returns:
            Dict mapping start position to {bin, path, distance}
        """
        results = {}
        
        # Get bins to search
        if waste_type and waste_type in self.bins:
            all_bins = self.bins[waste_type]
        else:
            # All bins
            all_bins = []
            for bins_list in self.bins.values():
                all_bins.extend(bins_list)
        
        for start in starts:
            best_bin = None
            best_path = None
            best_distance = float('inf')
            
            # Try each bin
            for bin_pos in all_bins:
                path = self.find_path(start, bin_pos)
                if path:
                    distance = len(path) - 1
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
                        best_bin = bin_pos
            
            if best_path:
                results[str(start)] = {
                    'bin': list(best_bin),
                    'path': [list(p) for p in best_path],
                    'distance': best_distance
                }
        
        return results
