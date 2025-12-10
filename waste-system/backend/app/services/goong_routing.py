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
    
    Features:
    - Get actual road directions (not straight line)
    - Support multiple routing profiles (car, bike, foot)
    - Get distance, duration, and polyline
    - Alternative routes
    """
    
    BASE_URL = "https://rsapi.goong.io"
    
    def __init__(self, api_key: str, optimization_strategy: str = "weighted"):
        """
        Initialize Goong routing service
        
        Args:
            api_key: Goong Maps API key (get from https://account.goong.io/)
            optimization_strategy: Algorithm for route selection
                - "weighted": Balance distance + time (default)
                - "dijkstra": Shortest distance only
                - "astar": Distance + heuristic
                - "multi_criteria": Multiple factors
                - "greedy": Fast nearest
        """
        if not api_key:
            raise ValueError("Goong API key is required. Get it from https://account.goong.io/")
        
        self.api_key = api_key
        self.session = requests.Session()
        self.optimizer = RouteOptimizer(strategy=optimization_strategy)
        logger.info(f"✅ Goong Routing Service initialized (strategy={optimization_strategy})")
    
    def get_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        vehicle: str = "car",
        alternatives: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get route from origin to destination
        
        Args:
            origin: (latitude, longitude) of start point
            destination: (latitude, longitude) of end point
            vehicle: Routing profile - "car", "bike", "foot" (default: "car")
            alternatives: Return alternative routes (default: True)
        
        Returns:
            Route data with distance, duration, polyline, and steps
            None if request fails
        
        Example:
            >>> route = service.get_route(
            ...     origin=(21.0285, 105.8542),  # Hà Nội
            ...     destination=(21.0378, 105.8345)
            ... )
            >>> print(route['distance_km'])  # 3.2
            >>> print(route['duration_minutes'])  # 8
        """
        # Validate vehicle type
        valid_vehicles = ["car", "bike", "foot"]
        if vehicle not in valid_vehicles:
            logger.warning(f"Invalid vehicle '{vehicle}', using 'car'")
            vehicle = "car"
        
        # Build API URL
        url = f"{self.BASE_URL}/Direction"
        
        # Parameters
        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "vehicle": vehicle,
            "api_key": self.api_key,
            "alternatives": "true" if alternatives else "false"
        }
        
        try:
            logger.info(f"🗺️ Getting route: {origin} → {destination} (vehicle={vehicle})")
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "OK":
                logger.error(f"Goong API error: {data.get('status')}")
                return None
            
            # Parse ALL routes from API (not just the first one)
            all_routes = []
            for route in data["routes"]:
                leg = route["legs"][0]
                route_data = {
                    "distance_meters": leg["distance"]["value"],
                    "distance_km": round(leg["distance"]["value"] / 1000, 2),
                    "distance_text": leg["distance"]["text"],
                    "duration_seconds": leg["duration"]["value"],
                    "duration_minutes": round(leg["duration"]["value"] / 60, 1),
                    "duration_text": leg["duration"]["text"],
                    "start_address": leg.get("start_address", ""),
                    "end_address": leg.get("end_address", ""),
                    "polyline": route["overview_polyline"]["points"],
                    "steps": self._parse_steps(leg.get("steps", [])),
                    "vehicle": vehicle,
                }
                all_routes.append(route_data)
            
            if not all_routes:
                logger.error("No routes returned from API")
                return None
            
            # ✨ Use CUSTOM ALGORITHM to select BEST route (not API's default)
            # This is OUR algorithm for academic paper
            result = self.optimizer.select_best_route(all_routes, verbose=True)
            
            if not result:
                logger.error("Algorithm failed to select best route")
                return None
            
            # Include other alternatives for comparison
            if alternatives and len(all_routes) > 1:
                result["alternatives"] = [
                    {
                        "distance_km": r["distance_km"],
                        "duration_minutes": r["duration_minutes"],
                        "polyline": r["polyline"]
                    }
                    for r in all_routes if r["polyline"] != result["polyline"]
                ]
            
            logger.info(f"✅ Route found: {result['distance_km']}km, {result['duration_minutes']}min")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Goong API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error processing route: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def find_nearest_bin_route(
        self,
        origin: Tuple[float, float],
        bins: List[Dict[str, Any]],
        vehicle: str = "foot"
    ) -> Optional[Dict[str, Any]]:
        """
        Find nearest waste bin with actual route distance (not straight line)
        
        Args:
            origin: (latitude, longitude) of waste detection location
            bins: List of waste bins with 'latitude', 'longitude', 'id', 'name', 'category'
            vehicle: Transportation mode (default: "foot" for walking)
        
        Returns:
            Best bin with route information
            {
                "bin": {...},  # Bin details
                "route": {...}  # Route details (distance, duration, polyline, steps)
            }
        
        Example:
            >>> bins = [
            ...     {"id": 1, "latitude": 21.03, "longitude": 105.85, "name": "Bin A"},
            ...     {"id": 2, "latitude": 21.04, "longitude": 105.86, "name": "Bin B"}
            ... ]
            >>> result = service.find_nearest_bin_route(
            ...     origin=(21.0285, 105.8542),
            ...     bins=bins
            ... )
        """
        logger.info(f"🔍 Finding nearest bin from {len(bins)} options...")
        
        # Get routes to ALL bins
        bin_routes = []
        for bin_data in bins:
            destination = (bin_data["latitude"], bin_data["longitude"])
            
            route = self.get_route(
                origin=origin,
                destination=destination,
                vehicle=vehicle,
                alternatives=False  # Don't need alternatives for each bin
            )
            
            if route:
                bin_routes.append({
                    "bin": bin_data,
                    "route": route
                })
        
        if not bin_routes:
            logger.warning("❌ No valid routes found to any bin")
            return None
        
        # ✨ Use CUSTOM ALGORITHM to select best bin
        # Compare all bin routes and select the one with best score
        best_bin_route = min(
            bin_routes, 
            key=lambda x: x["route"].get("route_score", float('inf'))
        )
        
        best_result = {
            "bin": best_bin_route["bin"],
            "route": best_bin_route["route"],
            "evaluated_bins": len(bin_routes)
        }
        
        logger.info(
            f"✅ Nearest bin: {best_result['bin']['name']} "
            f"({best_result['route']['distance_km']}km, "
            f"{best_result['route']['duration_minutes']}min)"
        )
        
        return best_result
    
    def get_optimized_route(
        self,
        origin: Tuple[float, float],
        waypoints: List[Tuple[float, float]],
        destination: Tuple[float, float],
        vehicle: str = "car"
    ) -> Optional[Dict[str, Any]]:
        """
        Get optimized route visiting multiple waypoints (for waste collection trucks)
        
        Args:
            origin: Starting point (lat, lng)
            waypoints: List of stops to visit [(lat1, lng1), (lat2, lng2), ...]
            destination: End point (lat, lng)
            vehicle: Vehicle type (default: "car")
        
        Returns:
            Optimized route with total distance, duration, and segments
        
        Note: Goong API supports up to 23 waypoints
        """
        if len(waypoints) > 23:
            logger.warning(f"Too many waypoints ({len(waypoints)}), limiting to 23")
            waypoints = waypoints[:23]
        
        url = f"{self.BASE_URL}/Direction"
        
        # Build waypoints string
        waypoints_str = "|".join([f"{lat},{lng}" for lat, lng in waypoints])
        
        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "waypoints": waypoints_str,
            "vehicle": vehicle,
            "api_key": self.api_key,
            "optimize": "true"  # Optimize waypoint order
        }
        
        try:
            logger.info(f"🚛 Getting optimized route with {len(waypoints)} waypoints...")
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "OK":
                logger.error(f"Route optimization failed: {data.get('status')}")
                return None
            
            route = data["routes"][0]
            
            # Calculate totals
            total_distance = sum(leg["distance"]["value"] for leg in route["legs"])
            total_duration = sum(leg["duration"]["value"] for leg in route["legs"])
            
            result = {
                "total_distance_km": round(total_distance / 1000, 2),
                "total_duration_minutes": round(total_duration / 60, 1),
                "waypoint_order": route.get("waypoint_order", []),
                "legs": [
                    {
                        "distance_km": round(leg["distance"]["value"] / 1000, 2),
                        "duration_minutes": round(leg["duration"]["value"] / 60, 1),
                        "start_address": leg.get("start_address", ""),
                        "end_address": leg.get("end_address", "")
                    }
                    for leg in route["legs"]
                ],
                "polyline": route["overview_polyline"]["points"]
            }
            
            logger.info(f"✅ Optimized route: {result['total_distance_km']}km, {result['total_duration_minutes']}min")
            return result
            
        except Exception as e:
            logger.error(f"❌ Route optimization error: {e}")
            return None
    
    def decode_polyline(self, encoded: str) -> List[Tuple[float, float]]:
        """
        Decode Google/Goong polyline to list of coordinates
        
        Args:
            encoded: Encoded polyline string
        
        Returns:
            List of (latitude, longitude) points
        
        Reference: https://developers.google.com/maps/documentation/utilities/polylinealgorithm
        """
        points = []
        index = 0
        lat = 0
        lng = 0
        
        while index < len(encoded):
            # Decode latitude
            result = 0
            shift = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat
            
            # Decode longitude
            result = 0
            shift = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng
            
            points.append((lat / 1e5, lng / 1e5))
        
        return points
    
    def _parse_steps(self, steps: List[Dict]) -> List[Dict[str, Any]]:
        """Parse route steps/directions"""
        return [
            {
                "instruction": step.get("html_instructions", ""),
                "distance_meters": step["distance"]["value"],
                "duration_seconds": step["duration"]["value"],
                "maneuver": step.get("maneuver", "")
            }
            for step in steps
        ]
    
    def get_distance_matrix(
        self,
        origins: List[Tuple[float, float]],
        destinations: List[Tuple[float, float]],
        vehicle: str = "car"
    ) -> Optional[Dict[str, Any]]:
        """
        Get distance and duration matrix for multiple origins and destinations
        Useful for optimizing waste collection routes
        
        Args:
            origins: List of origin points [(lat1, lng1), ...]
            destinations: List of destination points [(lat2, lng2), ...]
            vehicle: Vehicle type
        
        Returns:
            Matrix of distances and durations
        
        API Endpoint: https://rsapi.goong.io/DistanceMatrix
        """
        url = f"{self.BASE_URL}/DistanceMatrix"
        
        origins_str = "|".join([f"{lat},{lng}" for lat, lng in origins])
        destinations_str = "|".join([f"{lat},{lng}" for lat, lng in destinations])
        
        params = {
            "origins": origins_str,
            "destinations": destinations_str,
            "vehicle": vehicle,
            "api_key": self.api_key
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "OK":
                logger.error(f"Distance matrix failed: {data.get('status')}")
                return None
            
            return {
                "origins": data.get("origin_addresses", []),
                "destinations": data.get("destination_addresses", []),
                "rows": [
                    {
                        "elements": [
                            {
                                "distance_km": round(elem["distance"]["value"] / 1000, 2),
                                "duration_minutes": round(elem["duration"]["value"] / 60, 1),
                                "status": elem["status"]
                            }
                            for elem in row["elements"]
                        ]
                    }
                    for row in data["rows"]
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Distance matrix error: {e}")
            return None


# Fallback: Simple straight-line distance calculator (when Goong API is not available)
class StraightLineRouter:
    """
    Simple fallback router using Haversine formula (straight line distance)
    Used when Goong Maps API is not configured
    """
    
    @staticmethod
    def haversine_distance(
        coord1: Tuple[float, float],
        coord2: Tuple[float, float]
    ) -> float:
        """
        Calculate straight-line distance between two GPS coordinates (Haversine formula)
        
        Args:
            coord1: (latitude, longitude)
            coord2: (latitude, longitude)
        
        Returns:
            Distance in kilometers
        """
        import math
        
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = math.sin(delta_lat / 2)**2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return round(distance, 2)
    
    @classmethod
    def find_nearest_bin(
        cls,
        origin: Tuple[float, float],
        bins: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find nearest bin using straight-line distance
        
        Args:
            origin: (latitude, longitude)
            bins: List of bins with 'latitude', 'longitude'
        
        Returns:
            Nearest bin with distance
        """
        if not bins:
            return None
        
        best_bin = None
        shortest_distance = float('inf')
        
        for bin_data in bins:
            destination = (bin_data["latitude"], bin_data["longitude"])
            distance = cls.haversine_distance(origin, destination)
            
            if distance < shortest_distance:
                shortest_distance = distance
                best_bin = bin_data
        
        if best_bin:
            return {
                "bin": best_bin,
                "distance_km": shortest_distance,
                "method": "straight_line"
            }
        
        return None
