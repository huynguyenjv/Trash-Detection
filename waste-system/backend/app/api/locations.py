# -*- coding: utf-8 -*-
"""
HCM Waste Locations API
Provides detailed information about real waste collection locations in Ho Chi Minh City
"""

from fastapi import APIRouter, Query
from typing import List, Optional
from pydantic import BaseModel
from app.data import (
    HCM_WASTE_LOCATIONS, 
    LOCATION_TYPES,
    get_all_locations,
    get_locations_by_type,
    get_locations_by_district,
    get_location_stats
)

router = APIRouter(prefix="/locations", tags=["HCM Waste Locations"])


# Pydantic models for response
class LocationResponse(BaseModel):
    name: str
    name_en: str
    type: str
    type_display: str
    latitude: float
    longitude: float
    address: str
    district: str
    capacity_tons_per_day: int
    operator: str
    waste_types: List[str]
    status: str
    description: str


class LocationStatsResponse(BaseModel):
    total_locations: int
    by_type: dict
    by_district: dict
    total_capacity: int


class DistrictInfo(BaseModel):
    name: str
    location_count: int
    total_capacity: int


@router.get("/", response_model=List[LocationResponse])
async def get_all_waste_locations(
    location_type: Optional[str] = Query(None, description="Filter by location type"),
    district: Optional[str] = Query(None, description="Filter by district"),
    status: Optional[str] = Query(None, description="Filter by status (active/inactive)"),
    waste_type: Optional[str] = Query(None, description="Filter by waste type (organic, recyclable, hazardous, other)")
):
    """
    Get all waste collection locations in Ho Chi Minh City
    
    Optional filters:
    - location_type: treatment_facility, transfer_station, collection_point, hazardous_facility, recycling_center, public_bin
    - district: Quận 1, Quận 3, Bình Thạnh, Thủ Đức, etc.
    - status: active, inactive
    - waste_type: organic, recyclable, hazardous, other
    """
    locations = HCM_WASTE_LOCATIONS.copy()
    
    # Apply filters
    if location_type:
        locations = [loc for loc in locations if loc["type"] == location_type]
    
    if district:
        locations = [loc for loc in locations if loc["district"].lower() == district.lower()]
    
    if status:
        locations = [loc for loc in locations if loc.get("status", "active") == status]
    
    if waste_type:
        locations = [loc for loc in locations if waste_type in loc.get("waste_types", [])]
    
    # Format response
    result = []
    for loc in locations:
        result.append(LocationResponse(
            name=loc["name"],
            name_en=loc.get("name_en", loc["name"]),
            type=loc["type"],
            type_display=LOCATION_TYPES.get(loc["type"], loc["type"]),
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            address=loc["address"],
            district=loc["district"],
            capacity_tons_per_day=loc.get("capacity_tons_per_day", 0),
            operator=loc.get("operator", "Unknown"),
            waste_types=loc.get("waste_types", []),
            status=loc.get("status", "active"),
            description=loc.get("description", "")
        ))
    
    return result


@router.get("/stats", response_model=LocationStatsResponse)
async def get_location_statistics():
    """Get statistics about waste collection locations"""
    locations = HCM_WASTE_LOCATIONS
    
    # Count by type
    by_type = {}
    for loc in locations:
        loc_type = loc["type"]
        type_display = LOCATION_TYPES.get(loc_type, loc_type)
        if type_display not in by_type:
            by_type[type_display] = 0
        by_type[type_display] += 1
    
    # Count by district
    by_district = {}
    for loc in locations:
        district = loc["district"]
        if district not in by_district:
            by_district[district] = {"count": 0, "capacity": 0}
        by_district[district]["count"] += 1
        by_district[district]["capacity"] += loc.get("capacity_tons_per_day", 0)
    
    # Total capacity
    total_capacity = sum(loc.get("capacity_tons_per_day", 0) for loc in locations)
    
    return LocationStatsResponse(
        total_locations=len(locations),
        by_type=by_type,
        by_district=by_district,
        total_capacity=total_capacity
    )


@router.get("/types")
async def get_location_types():
    """Get all available location types"""
    return {
        "types": [
            {"key": key, "display": value} 
            for key, value in LOCATION_TYPES.items()
        ]
    }


@router.get("/districts")
async def get_districts():
    """Get all districts with waste collection locations"""
    districts = {}
    for loc in HCM_WASTE_LOCATIONS:
        district = loc["district"]
        if district not in districts:
            districts[district] = {
                "name": district,
                "location_count": 0,
                "total_capacity": 0,
                "types": set()
            }
        districts[district]["location_count"] += 1
        districts[district]["total_capacity"] += loc.get("capacity_tons_per_day", 0)
        districts[district]["types"].add(loc["type"])
    
    # Convert sets to lists for JSON serialization
    result = []
    for name, data in districts.items():
        result.append({
            "name": name,
            "location_count": data["location_count"],
            "total_capacity": data["total_capacity"],
            "location_types": list(data["types"])
        })
    
    # Sort by location count descending
    result.sort(key=lambda x: x["location_count"], reverse=True)
    
    return {"districts": result}


@router.get("/nearby")
async def get_nearby_locations(
    latitude: float = Query(..., description="Current latitude"),
    longitude: float = Query(..., description="Current longitude"),
    radius_km: float = Query(5.0, description="Search radius in kilometers"),
    location_type: Optional[str] = Query(None, description="Filter by location type"),
    limit: int = Query(10, description="Maximum number of results")
):
    """
    Get nearby waste collection locations based on coordinates
    """
    import math
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in km"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    # Calculate distance for each location
    locations_with_distance = []
    for loc in HCM_WASTE_LOCATIONS:
        if location_type and loc["type"] != location_type:
            continue
        
        distance = haversine_distance(
            latitude, longitude,
            loc["latitude"], loc["longitude"]
        )
        
        if distance <= radius_km:
            locations_with_distance.append({
                **loc,
                "distance_km": round(distance, 2)
            })
    
    # Sort by distance
    locations_with_distance.sort(key=lambda x: x["distance_km"])
    
    # Limit results
    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km
        },
        "count": len(locations_with_distance[:limit]),
        "locations": locations_with_distance[:limit]
    }


@router.get("/map-data")
async def get_map_data():
    """
    Get all locations formatted for map display with GeoJSON format
    """
    features = []
    for loc in HCM_WASTE_LOCATIONS:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [loc["longitude"], loc["latitude"]]
            },
            "properties": {
                "name": loc["name"],
                "name_en": loc.get("name_en", loc["name"]),
                "type": loc["type"],
                "type_display": LOCATION_TYPES.get(loc["type"], loc["type"]),
                "address": loc["address"],
                "district": loc["district"],
                "capacity": loc.get("capacity_tons_per_day", 0),
                "operator": loc.get("operator", "Unknown"),
                "waste_types": loc.get("waste_types", []),
                "status": loc.get("status", "active"),
                "description": loc.get("description", "")
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total": len(features),
            "city": "Ho Chi Minh City",
            "country": "Vietnam"
        }
    }
