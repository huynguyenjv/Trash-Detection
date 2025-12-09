"""
Routing API Routes
Real-world routing to waste bins using Goong Maps
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field

from app.database import get_db
from app.config import get_settings
from app.crud import get_waste_bins, get_bins_by_category, get_waste_bin
from app.models import WasteCategory
from app.services.goong_routing import GoongRoutingService, StraightLineRouter

router = APIRouter(prefix="/routing", tags=["Routing"])

# Initialize routing service
_routing_service = None


def get_routing_service():
    """Get routing service (Goong or fallback)"""
    global _routing_service
    
    if _routing_service is None:
        settings = get_settings()
        
        if settings.goong_maps_enabled and settings.goong_api_key:
            # Default strategy: weighted (can be overridden per request)
            _routing_service = GoongRoutingService(
                api_key=settings.goong_api_key,
                optimization_strategy="weighted"
            )
        else:
            _routing_service = None  # Will use fallback
    
    return _routing_service


def get_routing_service_with_strategy(strategy: str):
    """Get routing service with specific optimization strategy"""
    settings = get_settings()
    
    if settings.goong_maps_enabled and settings.goong_api_key:
        return GoongRoutingService(
            api_key=settings.goong_api_key,
            optimization_strategy=strategy
        )
    return None


# Request/Response schemas
class RouteRequest(BaseModel):
    """Request schema for route calculation"""
    origin_lat: float = Field(..., description="Origin latitude")
    origin_lng: float = Field(..., description="Origin longitude")
    dest_lat: float = Field(..., description="Destination latitude")
    dest_lng: float = Field(..., description="Destination longitude")
    vehicle: str = Field(default="foot", description="Vehicle type: car, bike, foot")
    algorithm: str = Field(
        default="weighted",
        description="Route optimization algorithm: weighted, dijkstra, astar, multi_criteria, greedy"
    )


class NearestBinRequest(BaseModel):
    """Request schema for finding nearest bin"""
    latitude: float = Field(..., description="Current latitude")
    longitude: float = Field(..., description="Current longitude")
    category: Optional[WasteCategory] = Field(None, description="Waste category filter")
    vehicle: str = Field(default="foot", description="Vehicle type: car, bike, foot")
    algorithm: str = Field(
        default="weighted",
        description="Route optimization algorithm: weighted, dijkstra, astar, multi_criteria, greedy"
    )


class WaypointOptimizationRequest(BaseModel):
    """Request schema for route optimization"""
    origin_lat: float
    origin_lng: float
    dest_lat: float
    dest_lng: float
    bin_ids: List[int] = Field(..., description="List of bin IDs to visit")
    vehicle: str = Field(default="car", description="Vehicle type")


@router.get("/health", summary="Check routing service status")
def routing_health():
    """Check if Goong Maps API is configured"""
    settings = get_settings()
    
    return {
        "goong_enabled": settings.goong_maps_enabled,
        "api_key_configured": bool(settings.goong_api_key),
        "fallback_available": True,
        "status": "ready" if settings.goong_maps_enabled else "fallback_mode"
    }


@router.post("/route", summary="Get route between two points")
def get_route(request: RouteRequest):
    """
    Get route from origin to destination using real roads
    
    - Uses Goong Maps API if configured
    - Falls back to straight-line distance
    - Applies CUSTOM ALGORITHM to select best route from alternatives
    
    Algorithms:
    - weighted: Balance distance (70%) + time (30%) - Default
    - dijkstra: Shortest distance only
    - astar: Distance + heuristic
    - multi_criteria: Distance + time + traffic + fuel
    - greedy: Fast nearest
    
    Returns distance, duration, polyline, and turn-by-turn directions
    """
    # Get service with requested algorithm
    service = get_routing_service_with_strategy(request.algorithm)
    
    origin = (request.origin_lat, request.origin_lng)
    destination = (request.dest_lat, request.dest_lng)
    
    if service:
        # Use Goong Maps + Custom Algorithm
        route = service.get_route(
            origin=origin,
            destination=destination,
            vehicle=request.vehicle,
            alternatives=True  # Get multiple routes for algorithm to compare
        )
        
        if route:
            return {
                "method": "goong_maps",
                "route": route
            }
        else:
            raise HTTPException(status_code=400, detail="Could not find route")
    
    else:
        # Fallback: Straight line
        distance = StraightLineRouter.haversine_distance(origin, destination)
        
        return {
            "method": "straight_line",
            "route": {
                "distance_km": distance,
                "duration_minutes": None,
                "warning": "Goong Maps not configured. Using straight-line distance."
            }
        }


@router.post("/nearest-bin", summary="Find nearest waste bin with route")
def find_nearest_bin(
    request: NearestBinRequest,
    db: Session = Depends(get_db)
):
    """
    Find nearest waste bin with actual route distance
    
    - Filters by waste category (if specified)
    - Returns bin details + route (distance, duration, directions)
    - Uses CUSTOM ALGORITHM to select best bin (not just closest)
    - Uses Goong Maps for real roads or fallback to straight line
    
    Algorithms:
    - weighted: Balance distance + time (default)
    - dijkstra: Shortest distance only
    - astar: Distance + heuristic
    - multi_criteria: Multiple factors
    - greedy: Fast nearest
    """
    # Get service with requested algorithm
    service = get_routing_service_with_strategy(request.algorithm)
    origin = (request.latitude, request.longitude)
    
    # Get bins (filter by category if specified)
    if request.category:
        bins = get_bins_by_category(db, request.category)
    else:
        bins = get_waste_bins(db, active_only=True)
    
    if not bins:
        raise HTTPException(status_code=404, detail="No active bins found")
    
    # Convert to dict format
    bins_data = [
        {
            "id": b.id,
            "name": b.name,
            "category": b.category,
            "latitude": b.latitude,
            "longitude": b.longitude,
            "address": b.address,
            "capacity": b.capacity
        }
        for b in bins
    ]
    
    if service:
        # Use Goong Maps for real routing
        result = service.find_nearest_bin_route(
            origin=origin,
            bins=bins_data,
            vehicle=request.vehicle
        )
        
        if result:
            return {
                "method": "goong_maps",
                "nearest_bin": result["bin"],
                "route": result["route"],
                "evaluated_bins": result.get("evaluated_bins", len(bins_data))
            }
        else:
            raise HTTPException(status_code=400, detail="Could not calculate routes to bins")
    
    else:
        # Fallback: Straight line
        result = StraightLineRouter.find_nearest_bin(origin, bins_data)
        
        if result:
            return {
                "method": "straight_line",
                "nearest_bin": result["bin"],
                "distance_km": result["distance_km"],
                "warning": "Goong Maps not configured. Using straight-line distance."
            }
        else:
            raise HTTPException(status_code=404, detail="No bins available")


@router.post("/optimize-route", summary="Optimize waste collection route")
def optimize_collection_route(
    request: WaypointOptimizationRequest,
    db: Session = Depends(get_db)
):
    """
    Optimize route for waste collection truck visiting multiple bins
    
    - Visits specified bins in optimized order
    - Returns total distance, duration, and waypoint order
    - Requires Goong Maps API
    """
    service = get_routing_service()
    
    if not service:
        raise HTTPException(
            status_code=503,
            detail="Route optimization requires Goong Maps API. Please configure GOONG_API_KEY."
        )
    
    # Get bin locations
    bins = []
    for bin_id in request.bin_ids:
        bin_data = get_waste_bin(db, bin_id)
        if bin_data:
            bins.append(bin_data)
    
    if len(bins) != len(request.bin_ids):
        raise HTTPException(status_code=404, detail="Some bins not found")
    
    # Build waypoints
    waypoints = [(b.latitude, b.longitude) for b in bins]
    
    # Get optimized route
    result = service.get_optimized_route(
        origin=(request.origin_lat, request.origin_lng),
        waypoints=waypoints,
        destination=(request.dest_lat, request.dest_lng),
        vehicle=request.vehicle
    )
    
    if result:
        # Add bin details to result
        result["bins"] = [
            {
                "id": b.id,
                "name": b.name,
                "category": b.category,
                "address": b.address
            }
            for b in bins
        ]
        
        return {
            "method": "goong_maps",
            "optimization": result
        }
    else:
        raise HTTPException(status_code=400, detail="Could not optimize route")


@router.get("/decode-polyline", summary="Decode polyline to coordinates")
def decode_polyline(encoded: str = Query(..., description="Encoded polyline string")):
    """
    Decode Google/Goong polyline to list of GPS coordinates
    
    Useful for rendering route on map
    """
    service = get_routing_service()
    
    if service:
        points = service.decode_polyline(encoded)
    else:
        # Use static method
        points = GoongRoutingService.decode_polyline(None, encoded)
    
    return {
        "coordinates": [
            {"lat": lat, "lng": lng}
            for lat, lng in points
        ],
        "count": len(points)
    }


@router.get("/distance-matrix", summary="Get distance matrix")
def get_distance_matrix(
    origins: str = Query(..., description="Origin coordinates (lat1,lng1|lat2,lng2)"),
    destinations: str = Query(..., description="Destination coordinates (lat1,lng1|lat2,lng2)"),
    vehicle: str = Query(default="car", description="Vehicle type")
):
    """
    Get distance/duration matrix for multiple origins and destinations
    """
    service = get_routing_service()
    
    if not service:
        raise HTTPException(
            status_code=503,
            detail="Distance matrix requires Goong Maps API"
        )
    
    # Parse coordinates
    try:
        origin_coords = [
            tuple(map(float, pair.split(',')))
            for pair in origins.split('|')
        ]
        dest_coords = [
            tuple(map(float, pair.split(',')))
            for pair in destinations.split('|')
        ]
    except:
        raise HTTPException(status_code=400, detail="Invalid coordinate format")
    
    result = service.get_distance_matrix(
        origins=origin_coords,
        destinations=dest_coords,
        vehicle=vehicle
    )
    
    if result:
        return result
    else:
        raise HTTPException(status_code=400, detail="Could not calculate matrix")
