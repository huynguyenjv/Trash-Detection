"""
Goong Maps Routing Service
Real-world routing using Goong Maps Directions API (like Google Maps for Vietnam)

API Documentation: https://docs.goong.io/rest/directions/
"""

import requests
from typing import List, Dict, Any, Optional, Tuple
import logging

from app.services.route_optimizer import RouteOptimizer

logger = logging.getLogger(__name__)


class GoongRoutingService:
    """
    Service to get real-world routes using Goong Maps API
    """
    
    BASE_URL = "https://rsapi.goong.io"
    
    def __init__(self, api_key: str, optimization_strategy: str = "weighted"):
        if not api_key:
            raise ValueError("Goong API key is required")
        
        self.api_key = api_key
        self.session = requests.Session()
        self.optimizer = RouteOptimizer(strategy=optimization_strategy)
        logger.info(f"✅ Goong Routing Service initialized")
    
    def get_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        vehicle: str = "car",
        alternatives: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get route from origin to destination"""
        valid_vehicles = ["car", "bike", "foot"]
        if vehicle not in valid_vehicles:
            vehicle = "car"
        
        url = f"{self.BASE_URL}/Direction"
        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "vehicle": vehicle,
            "api_key": self.api_key,
            "alternatives": "true" if alternatives else "false"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "OK":
                return None
            
            all_routes = []
            for route in data["routes"]:
                leg = route["legs"][0]
                route_data = {
                    "distance_meters": leg["distance"]["value"],
                    "distance_km": round(leg["distance"]["value"] / 1000, 2),
                    "duration_seconds": leg["duration"]["value"],
                    "duration_minutes": round(leg["duration"]["value"] / 60, 1),
                    "polyline": route["overview_polyline"]["points"],
                    "steps": self._parse_steps(leg.get("steps", [])),
                    "vehicle": vehicle,
                }
                all_routes.append(route_data)
            
            result = self.optimizer.select_best_route(all_routes, verbose=True)
            if not result:
                return None
            
            if alternatives:
                result["alternatives"] = [
                    {"distance_km": r["distance_km"], "duration_minutes": r["duration_minutes"]}
                    for r in all_routes if r["polyline"] != result.get("polyline", "")
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None
    
    def find_nearest_bin_route(
        self,
        origin: Tuple[float, float],
        bins: List[Dict[str, Any]],
        vehicle: str = "foot"
    ) -> Optional[Dict[str, Any]]:
        """Find nearest waste bin with actual route distance"""
        bin_routes = []
        for bin_data in bins:
            destination = (bin_data["latitude"], bin_data["longitude"])
            route = self.get_route(origin=origin, destination=destination, vehicle=vehicle)
            if route:
                bin_routes.append({"bin": bin_data, "route": route})
        
        if not bin_routes:
            return None
        
        best = min(bin_routes, key=lambda x: x["route"].get("route_score", float('inf')))
        return {"bin": best["bin"], "route": best["route"], "evaluated_bins": len(bin_routes)}
    
    def get_optimized_route(
        self,
        origin: Tuple[float, float],
        waypoints: List[Tuple[float, float]],
        destination: Tuple[float, float],
        vehicle: str = "car"
    ) -> Optional[Dict[str, Any]]:
        """Get optimized route visiting multiple waypoints"""
        if len(waypoints) > 23:
            waypoints = waypoints[:23]
        
        url = f"{self.BASE_URL}/Direction"
        waypoints_str = "|".join([f"{lat},{lng}" for lat, lng in waypoints])
        
        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "waypoints": waypoints_str,
            "vehicle": vehicle,
            "api_key": self.api_key,
            "optimize": "true"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "OK":
                return None
            
            route = data["routes"][0]
            total_distance = sum(leg["distance"]["value"] for leg in route["legs"])
            total_duration = sum(leg["duration"]["value"] for leg in route["legs"])
            
            return {
                "total_distance_km": round(total_distance / 1000, 2),
                "total_duration_minutes": round(total_duration / 60, 1),
                "waypoint_order": route.get("waypoint_order", []),
                "polyline": route["overview_polyline"]["points"]
            }
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None
    
    def decode_polyline(self, encoded: str) -> List[Tuple[float, float]]:
        """Decode polyline to coordinates"""
        points = []
        index = lat = lng = 0
        
        while index < len(encoded):
            result = shift = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            lat += ~(result >> 1) if (result & 1) else (result >> 1)
            
            result = shift = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            lng += ~(result >> 1) if (result & 1) else (result >> 1)
            
            points.append((lat / 1e5, lng / 1e5))
        
        return points
    
    @staticmethod
    def _parse_steps(steps: List[Dict]) -> List[Dict]:
        """Parse navigation steps"""
        return [
            {
                "instruction": step.get("html_instructions", ""),
                "distance_meters": step["distance"]["value"],
                "duration_seconds": step["duration"]["value"]
            }
            for step in steps
        ]
    
    def get_distance_matrix(
        self,
        origins: List[Tuple[float, float]],
        destinations: List[Tuple[float, float]],
        vehicle: str = "car"
    ) -> Optional[Dict[str, Any]]:
        """Get distance matrix"""
        url = f"{self.BASE_URL}/DistanceMatrix"
        
        params = {
            "origins": "|".join([f"{lat},{lng}" for lat, lng in origins]),
            "destinations": "|".join([f"{lat},{lng}" for lat, lng in destinations]),
            "vehicle": vehicle,
            "api_key": self.api_key
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "OK":
                return None
            
            return {
                "rows": [
                    {"elements": [
                        {"distance_km": round(e["distance"]["value"]/1000, 2), "duration_minutes": round(e["duration"]["value"]/60, 1)}
                        for e in row["elements"]
                    ]}
                    for row in data["rows"]
                ]
            }
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None


class StraightLineRouter:
    """Fallback router using Haversine formula"""
    
    @staticmethod
    def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        import math
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371.0
        
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return round(R * c, 2)
    
    @classmethod
    def find_nearest_bin(cls, origin: Tuple[float, float], bins: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not bins:
            return None
        
        best_bin = None
        shortest = float('inf')
        
        for bin_data in bins:
            dist = cls.haversine_distance(origin, (bin_data["latitude"], bin_data["longitude"]))
            if dist < shortest:
                shortest = dist
                best_bin = bin_data
        
        if best_bin:
            return {"bin": best_bin, "distance_km": shortest, "method": "straight_line"}
        return None
