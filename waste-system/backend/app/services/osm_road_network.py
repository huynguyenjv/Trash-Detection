"""
OpenStreetMap Road Network Loader
Load REAL road network from OpenStreetMap for academic algorithms

Dependencies:
    pip install osmnx networkx folium

Usage:
    loader = OSMRoadNetworkLoader()
    network = loader.load_hanoi_network()
    
    # Run custom algorithms on REAL map
    dijkstra = DijkstraAlgorithm()
    path = dijkstra.shortest_path(network, start_node, end_node)
"""

import osmnx as ox
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import folium
from datetime import datetime


@dataclass
class OSMNode:
    """Node trên road network (giao lộ)"""
    id: int
    lat: float
    lon: float
    street_count: int = 0  # Số đường giao nhau tại node này
    
    def to_dict(self):
        return asdict(self)


@dataclass
class OSMEdge:
    """Edge trên road network (đoạn đường)"""
    from_node: int
    to_node: int
    length: float  # meters
    length_km: float
    highway_type: str  # 'primary', 'secondary', 'residential', etc.
    name: Optional[str] = None
    oneway: bool = False
    maxspeed: Optional[int] = None  # km/h
    
    def to_dict(self):
        return asdict(self)


class OSMRoadNetworkLoader:
    """Load real road network từ OpenStreetMap"""
    
    def __init__(self, cache_dir: str = "osm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_hanoi_network(
        self, 
        network_type: str = 'drive',
        simplify: bool = True,
        use_cache: bool = True
    ) -> Tuple[Dict[int, OSMNode], List[OSMEdge]]:
        """
        Load road network của Hà Nội
        
        Args:
            network_type: 'drive', 'walk', 'bike', 'all'
            simplify: True = merge nodes that are not intersections
            use_cache: Use cached data if available
            
        Returns:
            (nodes_dict, edges_list)
        """
        cache_file = self.cache_dir / f"hanoi_{network_type}_{'simplified' if simplify else 'full'}.pkl"
        
        if use_cache and cache_file.exists():
            print(f"📦 Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("🌍 Downloading Hanoi road network from OpenStreetMap...")
        print("⏳ This may take 2-5 minutes for first time...")
        
        # Download graph từ OSM
        graph = ox.graph_from_place(
            "Hanoi, Vietnam",
            network_type=network_type,
            simplify=simplify
        )
        
        print(f"✅ Downloaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Convert sang format của mình
        nodes_dict = {}
        edges_list = []
        
        # Process nodes
        for node_id, data in graph.nodes(data=True):
            nodes_dict[node_id] = OSMNode(
                id=node_id,
                lat=data['y'],
                lon=data['x'],
                street_count=graph.degree[node_id]
            )
        
        # Process edges
        for u, v, key, data in graph.edges(keys=True, data=True):
            edge = OSMEdge(
                from_node=u,
                to_node=v,
                length=data.get('length', 0),
                length_km=data.get('length', 0) / 1000,
                highway_type=data.get('highway', 'unclassified') if isinstance(data.get('highway'), str) else 'unclassified',
                name=data.get('name'),
                oneway=data.get('oneway', False),
                maxspeed=self._parse_maxspeed(data.get('maxspeed'))
            )
            edges_list.append(edge)
        
        # Cache for next time
        with open(cache_file, 'wb') as f:
            pickle.dump((nodes_dict, edges_list), f)
        
        print(f"💾 Cached to: {cache_file}")
        
        return nodes_dict, edges_list
    
    def load_area_network(
        self,
        center_lat: float,
        center_lon: float,
        radius_meters: int = 5000,
        network_type: str = 'drive'
    ) -> Tuple[Dict[int, OSMNode], List[OSMEdge]]:
        """
        Load road network trong bán kính từ 1 điểm
        
        Example:
            # Load 5km around Hoàn Kiếm Lake
            nodes, edges = loader.load_area_network(21.0285, 105.8542, 5000)
        """
        print(f"🌍 Downloading road network around ({center_lat}, {center_lon})...")
        
        graph = ox.graph_from_point(
            (center_lat, center_lon),
            dist=radius_meters,
            network_type=network_type
        )
        
        print(f"✅ Downloaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Convert
        nodes_dict = {}
        edges_list = []
        
        for node_id, data in graph.nodes(data=True):
            nodes_dict[node_id] = OSMNode(
                id=node_id,
                lat=data['y'],
                lon=data['x'],
                street_count=graph.degree[node_id]
            )
        
        for u, v, key, data in graph.edges(keys=True, data=True):
            edge = OSMEdge(
                from_node=u,
                to_node=v,
                length=data.get('length', 0),
                length_km=data.get('length', 0) / 1000,
                highway_type=data.get('highway', 'unclassified') if isinstance(data.get('highway'), str) else 'unclassified',
                name=data.get('name'),
                oneway=data.get('oneway', False),
                maxspeed=self._parse_maxspeed(data.get('maxspeed'))
            )
            edges_list.append(edge)
        
        return nodes_dict, edges_list
    
    def _parse_maxspeed(self, maxspeed) -> Optional[int]:
        """Parse maxspeed from OSM data"""
        if maxspeed is None:
            return None
        if isinstance(maxspeed, (int, float)):
            return int(maxspeed)
        if isinstance(maxspeed, str):
            try:
                return int(maxspeed.split()[0])
            except (ValueError, TypeError):
                return None
        if isinstance(maxspeed, list):
            try:
                return int(maxspeed[0].split()[0])
            except (ValueError, TypeError):
                return None
        return None
    
    def find_nearest_node(
        self,
        nodes_dict: Dict[int, OSMNode],
        lat: float,
        lon: float
    ) -> int:
        """Tìm node gần nhất với 1 tọa độ"""
        min_dist = float('inf')
        nearest_node = None
        
        for node_id, node in nodes_dict.items():
            dist = self._haversine_distance(lat, lon, node.lat, node.lon)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
        
        return nearest_node
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def export_to_json(
        self,
        nodes_dict: Dict[int, OSMNode],
        edges_list: List[OSMEdge],
        output_file: str = "road_network.json"
    ):
        """Export to JSON for visualization"""
        data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'num_nodes': len(nodes_dict),
                'num_edges': len(edges_list)
            },
            'nodes': [node.to_dict() for node in nodes_dict.values()],
            'edges': [edge.to_dict() for edge in edges_list]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Exported to {output_file}")
    
    def visualize_network(
        self,
        nodes_dict: Dict[int, OSMNode],
        edges_list: List[OSMEdge],
        output_file: str = "road_network_map.html",
        path_nodes: Optional[List[int]] = None,
        poi_nodes: Optional[List[int]] = None,
        show_all_roads: bool = True
    ):
        """
        Visualize road network on interactive map
        
        Args:
            path_nodes: List of node IDs to highlight as path
            poi_nodes: List of node IDs to mark as points of interest (waste bins)
            show_all_roads: If True, show all roads. If False, only show path (cleaner)
        """
        # Calculate center
        if not nodes_dict:
            return
        
        center_lat = sum(n.lat for n in nodes_dict.values()) / len(nodes_dict)
        center_lon = sum(n.lon for n in nodes_dict.values()) / len(nodes_dict)
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Draw all edges (background roads) if enabled
        if show_all_roads:
            print("🎨 Drawing road network...")
            for edge in edges_list:
                from_node = nodes_dict[edge.from_node]
                to_node = nodes_dict[edge.to_node]
                
                # Color by road type
                color = {
                    'motorway': '#e74c3c',
                    'trunk': '#e67e22',
                    'primary': '#f39c12',
                    'secondary': '#f1c40f',
                    'tertiary': '#2ecc71',
                    'residential': '#95a5a6',
                    'service': '#bdc3c7'
                }.get(edge.highway_type.split('_')[0] if '_' in edge.highway_type else edge.highway_type, '#95a5a6')
                
                folium.PolyLine(
                    locations=[[from_node.lat, from_node.lon], [to_node.lat, to_node.lon]],
                    color=color,
                    weight=2,
                    opacity=0.3,
                    popup=f"{edge.name or 'Unnamed'}<br>{edge.length_km:.2f} km<br>{edge.highway_type}"
                ).add_to(m)
        
        # Draw path if provided - SNAP TO ACTUAL ROADS
        if path_nodes and len(path_nodes) >= 2:
            print("🛣️ Drawing path following actual roads...")
            
            # Build edge lookup for fast access
            edge_lookup = {}
            for edge in edges_list:
                edge_lookup[(edge.from_node, edge.to_node)] = edge
                if not edge.oneway:
                    edge_lookup[(edge.to_node, edge.from_node)] = edge
            
            # Draw each segment of the path
            for i in range(len(path_nodes) - 1):
                from_node_id = path_nodes[i]
                to_node_id = path_nodes[i + 1]
                
                if from_node_id not in nodes_dict or to_node_id not in nodes_dict:
                    continue
                
                from_node = nodes_dict[from_node_id]
                to_node = nodes_dict[to_node_id]
                
                # Check if this edge exists in road network
                edge_key = (from_node_id, to_node_id)
                reverse_key = (to_node_id, from_node_id)
                
                if edge_key in edge_lookup or reverse_key in edge_lookup:
                    # Edge exists - draw along the road
                    edge = edge_lookup.get(edge_key, edge_lookup.get(reverse_key))
                    
                    folium.PolyLine(
                        locations=[[from_node.lat, from_node.lon], [to_node.lat, to_node.lon]],
                        color='#2196F3',
                        weight=6,
                        opacity=0.9,
                        popup=f"Path Segment {i+1}<br>{edge.name or 'Unnamed'}<br>{edge.length_km:.2f} km"
                    ).add_to(m)
                else:
                    # Edge doesn't exist - should not happen in valid path
                    print(f"⚠️ Warning: No road between {from_node_id} and {to_node_id}")
                    folium.PolyLine(
                        locations=[[from_node.lat, from_node.lon], [to_node.lat, to_node.lon]],
                        color='red',
                        weight=4,
                        opacity=0.5,
                        dashArray='5, 10',
                        popup="Invalid edge (not in road network)"
                    ).add_to(m)
            
            # Mark start and end
            if path_nodes:
                start_node = nodes_dict[path_nodes[0]]
                end_node = nodes_dict[path_nodes[-1]]
                
                folium.Marker(
                    location=[start_node.lat, start_node.lon],
                    popup=f"START<br>Node: {path_nodes[0]}",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
                folium.Marker(
                    location=[end_node.lat, end_node.lon],
                    popup=f"END<br>Node: {path_nodes[-1]}",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
                
                # Add intermediate nodes for debugging
                for idx, node_id in enumerate(path_nodes[1:-1], 1):
                    node = nodes_dict[node_id]
                    folium.CircleMarker(
                        location=[node.lat, node.lon],
                        radius=4,
                        color='blue',
                        fill=True,
                        fillColor='blue',
                        fillOpacity=0.7,
                        popup=f"Step {idx}<br>Node: {node_id}"
                    ).add_to(m)
        
        # Draw POI nodes (waste bins)
        if poi_nodes:
            print("🗑️ Drawing waste bins...")
            for nid in poi_nodes:
                if nid in nodes_dict:
                    node = nodes_dict[nid]
                    folium.Marker(
                        location=[node.lat, node.lon],
                        popup=f"Waste Bin<br>Node: {nid}",
                        icon=folium.Icon(color='orange', icon='trash')
                    ).add_to(m)
        
        # Save
        m.save(output_file)
        print(f"🗺️ Map saved to {output_file}")
        
        return m


def build_adjacency_graph(
    nodes_dict: Dict[int, OSMNode],
    edges_list: List[OSMEdge]
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build adjacency list for algorithms
    
    Returns:
        {node_id: [(neighbor_id, distance_km), ...]}
    """
    graph = {node_id: [] for node_id in nodes_dict.keys()}
    
    for edge in edges_list:
        # Add forward edge
        graph[edge.from_node].append((edge.to_node, edge.length_km))
        
        # Add reverse edge if not oneway
        if not edge.oneway:
            graph[edge.to_node].append((edge.from_node, edge.length_km))
    
    return graph


# Quick example usage
if __name__ == "__main__":
    print("=" * 60)
    print("OpenStreetMap Road Network Loader")
    print("=" * 60)
    
    loader = OSMRoadNetworkLoader()
    
    # Load small area first (faster for testing)
    print("\n📍 Loading road network around Hoàn Kiếm Lake (2km radius)...")
    nodes, edges = loader.load_area_network(
        center_lat=21.0285,
        center_lon=105.8542,
        radius_meters=2000  # 2km radius
    )
    
    print(f"\n✅ Loaded real road network:")
    print(f"   • Nodes (intersections): {len(nodes)}")
    print(f"   • Edges (road segments): {len(edges)}")
    
    # Export
    loader.export_to_json(nodes, edges, "hanoi_road_network.json")
    
    # Visualize
    loader.visualize_network(nodes, edges, "hanoi_road_network.html")
    
    print("\n✅ Done! Open 'hanoi_road_network.html' to see the map!")
