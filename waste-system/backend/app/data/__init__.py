# -*- coding: utf-8 -*-
"""
Data module for HCM City waste management data
"""

from .hcm_waste_locations import (
    HCM_WASTE_LOCATIONS,
    WASTE_TYPE_MAPPING,
    LOCATION_TYPES,
    get_all_locations,
    get_locations_by_type,
    get_locations_by_district,
    get_location_stats
)

__all__ = [
    "HCM_WASTE_LOCATIONS",
    "WASTE_TYPE_MAPPING", 
    "LOCATION_TYPES",
    "get_all_locations",
    "get_locations_by_type",
    "get_locations_by_district",
    "get_location_stats"
]
