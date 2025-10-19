"""
A* Pathfinding Module
Implements A* algorithm for optimal route calculation
"""
import heapq
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import osmnx as ox
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import split
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, graphml_dir: str):
        self.graph = ox.load_graphml(graphml_dir)

    def custom_node_projection(self, location_id: str) -> Tuple[str, Any , Any]:
        """Project lat/lon to nearest edge in the graph"""
        original_node = self.graph.nodes[location_id]
        print(original_node)
        u, v, key = ox.distance.nearest_edges(self.graph, X=original_node['x'], Y=original_node['y'])
        edge_data = self.graph[u][v][key]

        edge_geom = edge_data.get('geometry')
        print(edge_data)
        if edge_geom is None:
            # Create a straight line geometry between nodes if not present
            node_u = self.graph.nodes[u]
            node_v = self.graph.nodes[v]
            edge_geom = LineString([(node_u['x'], node_u['y']), (node_v['x'], node_v['y'])])
        
        projected_point = edge_geom.interpolate(edge_geom.project(Point(original_node['x'], original_node['y'])))
        projected_node_id = max(self.graph.nodes) + 1

        if math.sqrt((projected_point.x - original_node['x'])**2 + (projected_point.y - original_node['y'])**2) > 1e-6:
            self.graph.add_node(projected_node_id, x=projected_point.x, y=projected_point.y)
        else:
            projected_node_id = location_id  # or v, they are the same in this case

        # nearest_proj_node = ox.distance.nearest_nodes(self.graph, X=projected_point.x, Y=projected_point.y)
        # if projected_point.equals(Point(self.graph.nodes[nearest_proj_node]['x'], self.graph.nodes[nearest_proj_node]['y'])):
        #     return self.graph.nodes[nearest_proj_node]['osmid'], None, None  # No need to split edge
        # Get original coordinates
        coords = list(edge_geom.coords)

        # Find the closest segment to insert the point
        min_dist = float('inf')
        insert_index = None

        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            dist = segment.distance(Point(self.graph.nodes[projected_node_id]['x'], self.graph.nodes[projected_node_id]['y']))
            if dist < min_dist:
                min_dist = dist
                insert_index = i + 1  # insert after coords[i]

        # Insert the point
        coords.insert(insert_index, (self.graph.nodes[projected_node_id]['x'], self.graph.nodes[projected_node_id]['y']))

        # Create new LineString
        edge_geom = LineString(coords)

        
        print(edge_geom)
        split_line = split(edge_geom, MultiPoint([projected_point]))  # tiny buffer to force split

        # Check if it's a GeometryCollection and convert to list
        if hasattr(split_line, 'geoms'):
            split_segments = list(split_line.geoms)
            print(split_segments)
            # if len(split_segments) == 2:
            segment1, segment2 = split_segments[0], split_segments[1]
            # else:
            #     print(f"⚠️ Split returned {len(split_segments)} segments — expected 2.")
        else:
            print("❌ Split failed — result is not a GeometryCollection.")

        print("Segment 1:", segment1)
        print("Segment 2:", segment2)
        self.graph.remove_edge(u, v, key)
        # Add edge from u to new node
        max_osmid = -10**18

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            osmid = data.get('osmid')
            if isinstance(osmid, list):
                current_max = max(osmid)
            elif isinstance(osmid, int):
                current_max = osmid
            else:
                continue  # skip if osmid is missing or invalid

            if max_osmid is None or current_max > max_osmid:
                max_osmid = current_max

        print("Maximum osmid:", max_osmid)

        edge_1_id = max_osmid + 1
        edge_2_id = edge_1_id + 1
        self.graph.add_edge(u, projected_node_id, geometry=segment1
                            , length=segment1.length
                            , name=edge_data.get('name')
                            , highway=edge_data.get('highway')
                            , lanes=edge_data.get('lanes')
                            , oneway=edge_data.get('oneway', False)
                            , reversed=edge_data.get('reversed', False)
                            , osmid=0
                            )
        # Add edge from new node to v
        self.graph.add_edge(projected_node_id, v, geometry=segment2, length=segment2.length
                            , name=edge_data.get('name')
                            , highway=edge_data.get('highway')
                            , lanes=edge_data.get('lanes')
                            , oneway=edge_data.get('oneway', False)
                            , reversed=edge_data.get('reversed', False)
                            , osmid=0
                            )
        
        if projected_node_id != location_id:
            self.graph.add_edge(location_id, projected_node_id)

        return projected_node_id, edge_1_id, edge_2_id

class Node:
    def __init__(self, node_id: int, parent: Optional['Node'] = None, g: float = 0.0, h: float = 0.0,  f: float = 0.0):
        self.node_id = node_id
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = f  # Total cost
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.node_id == other.node_id

    def __lt__(self, other: 'Node') -> bool:
        return self.f < other.f


class AStarPathfinder:
    def __init__(self, graph: Graph):
        self.open_list: List[Any] = []
        self.closed_list: List[Any] = []
        self.graph = graph.graph

    def heuristic(self, start: Point, goal: Point) -> float:
        # Using Manhattan distance as heuristic
        return abs(start.x - goal.x) + abs(start.y - goal.y)
    
    def explore_best_neighbours(self, current_node: Node, goal_node: Node) -> Any:
        neighbours: List[Node] = []
        outgoing_edges = list(self.graph.out_edges(current_node.node_id, keys=True, data=True))
        print(outgoing_edges)

        for source, target, key, data in outgoing_edges:
            g = current_node.g + data.get('length', 1.0)
            target_node_data = self.graph.nodes[target]
            h = self.heuristic(Point(target_node_data['x'], target_node_data['y'])
                               ,Point(self.graph.nodes[goal_node.node_id]['x'], self.graph.nodes[goal_node.node_id]['y']))
            f = g + h
            tmp_node= Node(node_id=target, parent=current_node, g=g, h=h, f=f)
            if tmp_node not in self.closed_list:
                heapq.heappush(neighbours, tmp_node)

        return heapq.heappop(neighbours) if neighbours else current_node
    
    def find_path(self, source_id: int, destination_id: int):
        source_node = Node(node_id=source_id)
        destination_node = Node(node_id=destination_id)

        heapq.heappush(self.open_list, source_node)

        while self.open_list:
            current_node = heapq.heappop(self.open_list)
            self.closed_list.append(current_node)

            if current_node == destination_node:
                return self.reconstruct_path(current_node)

            heapq.heappush(self.open_list, self.explore_best_neighbours(current_node, destination_node))
    
    def reconstruct_path(self, current_node: Node) -> List[Tuple[float, float]]:
        path = []
        while current_node.parent:
            map_node = self.graph.nodes[current_node.node_id]
            path.append((map_node['x'], map_node['y']))
            current_node = current_node.parent

        path.reverse()
        return path
#     def reconstruct_path(self, current_node: Node) -> List[Tuple[float, float]]:
#         path = []
#         while current_node.parent:
#             path.append(self.grid.grid_to_latlon(current_node.position[0], current_node.position[1]))
#             current_node = current_node.parent
#         path.reverse()
#         return path


# class AStarAlgorithm:
#     def __init__(self, grid: Grid):
        
#         self.open_list:     List[Node] = []
#         self.closed_list:   List[Node] = []
#         self.grid = grid  

#     def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
#         # Using Manhattan distance as heuristic
#         return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
#     def explore_best_neighbours(self, current_node: Node, goal_node: Node) -> Node:
#         neighbours: List[Node] = []

#         # 8-directional movement
#         directions: List[Tuple[int, int]]= []
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 if dx == 0 and dy == 0:
#                     continue
#                 directions.append((dx, dy))
        
#         provisional_g = current_node.g + 1
#         provisional_h = 0
#         provisional_f = 0

#         for direction in directions:
#             potential_position = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

#             # Check if within grid bounds
#             if (0 <= potential_position[0] < self-10**18.grid.grid_size) and (0 <= potential_position[1] < self.grid.grid_size):
#                 # Check if not an obstacle
#                 if self.grid.grid[potential_position[0]][potential_position[1]] == 0:
#                     provisional_h = self.heuristic(potential_position, goal_node.position)
#                     provisional_f = provisional_g + provisional_h
#                     tmp_node = Node(position=potential_position, parent=current_node, g=provisional_g, h=provisional_h, f=provisional_f)
#                     if tmp_node not in self.closed_list:
#                         heapq.heappush(neighbours, tmp_node)

#         return heapq.heappop(neighbours) if neighbours else current_node
    
#     def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[float, float]]:
#         start_node = Node(position=start)
#         goal_node = Node(position=goal)

#         heapq.heappush(self.open_list, start_node)

#         while self.open_list:
#             current_node = heapq.heappop(self.open_list)
#             self.closed_list.append(current_node)

#             # Check if we reached the goal
#             if current_node == goal_node:
#                 return self.reconstruct_path(current_node)
            
#             heapq.heappush(self.open_list, self.explore_best_neighbours(current_node, goal_node))

#         return []  # No path found
    
#     def reconstruct_path(self, current_node: Node) -> List[Tuple[float, float]]:
#         path = []
#         while current_node.parent:
#             path.append(self.grid.grid_to_latlon(current_node.position[0], current_node.position[1]))
#             current_node = current_node.parent
#         path.reverse()
#         return path
#                
    
#     def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[float, float]]:
#         start_node = Node(position=start)
#         goal_node = Node(position=goal)

#         heapq.heappush(self.open_list, start_node)

#         while self.open_list:
#             current_node = heapq.heappop(self.open_list)
#             self.closed_list.append(current_node)

#             # Check if we reached the goal
#             if current_node == goal_node:
#                 return self.reconstruct_path(current_node)
            
#             heapq.heappush(self.open_list, self.explore_best_neighbours(current_node, goal_node))

#         return []  # No path found
    
#     def reconstruct_path(self, current_node: Node) -> List[Tuple[float, float]]:
#         path = []
#         while current_node.parent:
#             path.append(self.grid.grid_to_latlon(current_node.position[0], current_node.position[1]))
#             current_node = current_node.parent
#         path.reverse()
#         return path
    

# Node class for A* algorithm
# class Node:
#     def __init__(self, position: Tuple[int, int], parent: Optional['Node'] = None, g: int = 0, h: int = 0,  f: int = 0):
#         self.position = position
#         self.parent = parent
#         self.g = g  # Cost from start to current node
#         self.h = h  # Heuristic cost to goal
#         self.f = f  # Total cost

#     def __eq__(self, other: Any) -> bool:
#         return self.position == other.position

#     def __lt__(self, other: 'Node') -> bool:
#         return self.f < other.f

# class Grid:
#     # Considering moving from the centre of one grid cell to another
#     # The final path will be adjusted to the actual lat/lon coordinates of the cell centres
#     def __init__(self, grid_size: int = 100):
#         self.grid_size = grid_size

#         # Ho Chi Minh City bounds (approximate)
#         self.bounds = {
#             'min_lat': 10.70,
#             'max_lat': 10.90,
#             'min_lon': 106.60,
#             'max_lon': 106.80
#         }

#         self.lat_per_slot = (self.bounds['max_lat'] - self.bounds['min_lat'])/self.grid_size
#         self.lon_per_slot = (self.bounds['max_lon'] - self.bounds['min_lon'])/self.grid_size
#         self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)  # 0 for free cell, 1 for obstacle
#         self._add_obstacles()

#     def _add_obstacles(self):
#         """Add some obstacles to the grid to simulate real world"""
#         # Add some blocked areas (simplified for demo)
#         # In real implementation, this would be based on actual map data
        
#         # Add some river/water obstacles
#         self.grid[45:55, 20:80] = 1  # Horizontal water body
        
#         # Add some building clusters
#         for i in range(5, 95, 20):
#             for j in range(5, 95, 20):
#                 # Random small building clusters
#                 if np.random.random() > 0.7:
#                     self.grid[i:i+3, j:j+3] = 0

#     def latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
#         if not (self.bounds['min_lat'] <= lat <= self.bounds['max_lat']) or not (self.bounds['min_lon'] <= lon <= self.bounds['max_lon']):
#             raise ValueError("Latitude or Longitude out of bounds")

#         row = int((lat - self.bounds['min_lat']) / self.lat_per_slot) - 1
#         col = int((lon - self.bounds['min_lon']) / self.lon_per_slot) - 1 

#         return row, col                                                                                                                                   
    
#     def grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
#         if not (0 <= row < self.grid_size) or not (0 <= col < self.grid_size):
#             raise ValueError("Grid position out of bounds")

#         lat = self.bounds['min_lat'] + (row + 0.5) * self.lat_per_slot
#         lon = self.bounds['min_lon'] + (col + 0.5) * self.lon_per_slot

#         return lat, lon

# class AStarAlgorithm:
#     def __init__(self, grid: Grid):
        
#         self.open_list:     List[Node] = []
#         self.closed_list:   List[Node] = []
#         self.grid = grid  

#     def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
#         # Using Manhattan distance as heuristic
#         return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
#     def explore_best_neighbours(self, current_node: Node, goal_node: Node) -> Node:
#         neighbours: List[Node] = []

#         # 8-directional movement
#         directions: List[Tuple[int, int]]= []
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 if dx == 0 and dy == 0:
#                     continue
#                 directions.append((dx, dy))
        
#         provisional_g = current_node.g + 1
#         provisional_h = 0
#         provisional_f = 0

#         for direction in directions:
#             potential_position = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

#             # Check if within grid bounds
#             if (0 <= potential_position[0] < self.grid.grid_size) and (0 <= potential_position[1] < self.grid.grid_size):
#                 # Check if not an obstacle
#                   if self.grid.grid[potential_position[0]][potential_position[1]] == 0:
#                     provisional_h = self.heuristic(potential_position, goal_node.position)
#                     provisional_f = provisional_g + provisional_h
#                     tmp_node = Node(position=potential_position, parent=current_node, g=provisional_g, h=provisional_h, f=provisional_f)
#                     if tmp_node not in self.closed_list:
#                         heapq.heappush(neighbours, tmp_node)

#         return heapq.heappop(neighbours) if neighbours else current_node
# 
# class Grid:
#     # Considering moving from the centre of one grid cell to another
#     # The final path will be adjusted to the actual lat/lon coordinates of the cell centres
#     def __init__(self, grid_size: int = 100):
#         self.grid_size = grid_size

#         # Ho Chi Minh City bounds (approximate)
#         self.bounds = {
#             'min_lat': 10.70,
#             'max_lat': 10.90,
#             'min_lon': 106.60,
#             'max_lon': 106.80
#         }

#         self.lat_per_slot = (self.bounds['max_lat'] - self.bounds['min_lat'])/self.grid_size
#         self.lon_per_slot = (self.bounds['max_lon'] - self.bounds['min_lon'])/self.grid_size
#         self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)  # 0 for free cell, 1 for obstacle
#         self._add_obstacles()

#     def _add_obstacles(self):
#         """Add some obstacles to the grid to simulate real world"""
#         # Add some blocked areas (simplified for demo)
#         # In real implementation, this would be based on actual map data
        
#         # Add some river/water obstacles
#         self.grid[45:55, 20:80] = 1  # Horizontal water body
        
#         # Add some building clusters
#         for i in range(5, 95, 20):
#             for j in range(5, 95, 20):
#                 # Random small building clusters
#                 if np.random.random() > 0.7:
#                     self.grid[i:i+3, j:j+3] = 0

#     def latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
#         if not (self.bounds['min_lat'] <= lat <= self.bounds['max_lat']) or not (self.bounds['min_lon'] <= lon <= self.bounds['max_lon']):
#             raise ValueError("Latitude or Longitude out of bounds")

#         row = int((lat - self.bounds['min_lat']) / self.lat_per_slot) - 1
#         col = int((lon - self.bounds['min_lon']) / self.lon_per_slot) - 1 

#         return row, col                                                                                                                                   
    
#     def grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
#         if not (0 <= row < self.grid_size) or not (0 <= col < self.grid_size):
#             raise ValueError("Grid position out of bounds")

#         lat = self.bounds['min_lat'] + (row + 0.5) * self.lat_per_slot
#         lon = self.bounds['min_lon'] + (col + 0.5) * self.lon_per_slot

#         return lat, lon


if __name__ == "__main__":
    # grid = Grid(grid_size=10000)
    # print(grid.lat_per_slot, grid.lon_per_slot)
    # astar = AStarAlgorithm(grid)
    hcm_graph = Graph(graphml_dir="hcm_road_network.graphml")

    starting_loc = (10.7726866, 106.6959108)
    start_loc_id = max(hcm_graph.graph.nodes) + 1
    ending_loc = (10.7896819, 106.6966359)
    end_loc_id = start_loc_id + 1

    hcm_graph.graph.add_node(start_loc_id, y=starting_loc[0], x=starting_loc[1])
    hcm_graph.graph.add_node(end_loc_id, y=ending_loc[0], x=ending_loc[1])
    print(hcm_graph.graph.nodes[411917825])

    proj_start_node_id, edge1_start_id, edge2_start_id = hcm_graph.custom_node_projection(start_loc_id)
    print(f"Projected node ID: {proj_start_node_id}, Edge1 ID: {edge1_start_id}, Edge2 ID: {edge2_start_id}")
    
    proj_end_node_id, edge1_end_id, edge2_end_id = hcm_graph.custom_node_projection(end_loc_id)
    print(f"Projected node ID: {proj_end_node_id}, Edge1 ID: {edge1_end_id}, Edge2 ID: {edge2_end_id}")
    
    # print(f"Projected node ID: {proj_end_node_id}, Edge1 ID: {edge1_end_id}, Edge2 ID: {edge2_end_id}")
    

    # goal_lat, goal_lon = 10.85, 106.75

    # start_grid = grid.latlon_to_grid(start_lat, start_lon)
    # goal_grid = grid.latlon_to_grid(goal_lat, goal_lon)

    # path = astar.find_path(start_grid, goal_grid)
    # print("Calculated Path (lat, lon):")
    # for coord in path:
    #     print(coord)

    # Define the region (can be a city name, bounding box, or coordinates)
    # hcm_graph.graph = ox.graph_from_place("District 1, Ho Chi Minh City, Vietnam", network_type='drive')

    # Save the graph if needed
    # ox.save_graphml(G, filepath="hcm_road_network.graphml")

    # Explore nodes and edges
    print("Nodes:", hcm_graph.graph.number_of_nodes())
    print("Edges:", hcm_graph.graph.number_of_edges())

#     u, v, key = ox.distance.nearest_edges(hcm_graph.graph, X=starting_loc[1], Y=starting_loc[0])
   
#    # Access edge attributes
#     edge_data = hcm_graph.graph[u][v][key]
#     print(f"Nearest edge is from node {u} to node {v}")
#     print("Edge attributes:", edge_data)
    # ox.plot_graph(G)

    # # Access node attributes
    # for node, data in G.nodes(data=True):
    #     print(node, data)

    # # Access edge attributes
    # for u, v, data in G.edges(data=True):
    #     print(u, v, data)

    # Plot graph and highlight the starting location and the nearest edge

    a_star = AStarPathfinder(hcm_graph)
    path = a_star.find_path(start_loc_id, end_loc_id)

    print(path)

    # draw base graph (suppress immediate show)
    fig, ax = ox.plot_graph(hcm_graph.graph, node_size=0, edge_color='lightgray', show=False, close=False)

    # Plot starting location (lat, lon) -> (x=lon, y=lat)
    start_x, start_y = starting_loc[1], starting_loc[0]
    end_x, end_y = ending_loc[1], ending_loc[0]

    # Try to get the geometry of the edge; fallback to straight line between nodes
    xs, ys = None, None
    geom = edge_data.get('geometry')
    if geom is not None:
        if isinstance(geom, LineString):
            xs, ys = geom.xy
        else:
            # sometimes geometry could be a list-like; convert to LineString
            try:
                geom_ls = LineString(geom)
                xs, ys = geom_ls.xy
            except Exception:
                xs, ys = None, None

    if xs is None or ys is None:
        # fallback: use node coordinates
        node_u = hcm_graph.graph.nodes[u]
        node_v = hcm_graph.graph.nodes[v]
        ux, uy = node_u.get('x'), node_u.get('y')
        vx, vy = node_v.get('x'), node_v.get('y')
        xs, ys = [ux, vx], [uy, vy]

    # Plot the nearest edge with a distinct style
    ax.plot(xs, ys, color='red', linewidth=3.5, alpha=0.9, solid_capstyle='round', zorder=4, label='Nearest edge')
    ax.scatter([start_x], [start_y], c='blue', s=120, marker='*', label='Start', zorder=6)
    ax.scatter([end_x], [end_y], c='green', s=120, marker='*', label='End', zorder=6)
    ax.scatter([hcm_graph.graph.nodes[proj_start_node_id]['x']], [hcm_graph.graph.nodes[proj_start_node_id]['y']], c='brown', s=120, marker='*', label='Start Proj', zorder=6)

    # Optionally mark the edge's end nodes
    ax.scatter([hcm_graph.graph.nodes[u]['x'], hcm_graph.graph.nodes[v]['x']], [hcm_graph.graph.nodes[u]['y'], hcm_graph.graph.nodes[v]['y']],
               c=['orange','orange'], s=60, marker='o', zorder=5, label='Edge nodes')

    ax.legend(loc='upper right')
    plt.title("Road graph with start location (blue star) and nearest edge (red)")
    plt.show()