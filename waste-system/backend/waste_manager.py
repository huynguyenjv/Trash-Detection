"""
Waste Management Module
Handles waste statistics and bin management
"""
from typing import Dict, List, Tuple, Any
import math
from datetime import datetime, timedelta


class WasteManager:
    def __init__(self):
        """Initialize waste manager with statistics and bin locations"""
        # Current session statistics
        self.session_stats = {
            'total': 0,
            'organic': 0,
            'recyclable': 0,
            'hazardous': 0,
            'other': 0,
            'last_updated': datetime.now()
        }
        
        # Waste bin locations (lat, lon)
        self.waste_bins = {
            'general': [
                {'id': 1, 'name': 'Central Waste Bin', 'lat': 10.8231, 'lon': 106.6297, 'capacity': 100},
                {'id': 5, 'name': 'District 1 General Bin', 'lat': 10.7769, 'lon': 106.7009, 'capacity': 80},
            ],
            'recyclable': [
                {'id': 2, 'name': 'Recycling Center', 'lat': 10.8331, 'lon': 106.6397, 'capacity': 150},
                {'id': 6, 'name': 'District 3 Recycling', 'lat': 10.7886, 'lon': 106.6947, 'capacity': 120},
            ],
            'organic': [
                {'id': 3, 'name': 'Organic Waste Bin', 'lat': 10.8131, 'lon': 106.6197, 'capacity': 90},
                {'id': 7, 'name': 'Binh Thanh Organic', 'lat': 10.8014, 'lon': 106.7108, 'capacity': 100},
            ],
            'hazardous': [
                {'id': 4, 'name': 'Hazardous Waste Facility', 'lat': 10.8431, 'lon': 106.6497, 'capacity': 50},
                {'id': 8, 'name': 'District 7 Hazardous', 'lat': 10.7378, 'lon': 106.7218, 'capacity': 60},
            ]
        }
        
        # Historical data for trends
        self.hourly_stats = []
        
    def update_stats(self, detections: List[Dict[str, Any]]):
        """
        Update statistics from new detections
        Args:
            detections: List of detection results from detector
        """
        # Reset current stats
        new_stats = {
            'total': 0,
            'organic': 0,
            'recyclable': 0,
            'hazardous': 0,
            'other': 0,
            'last_updated': datetime.now()
        }
        
        # Count by category
        for detection in detections:
            category = detection.get('category', 'other')
            if category in new_stats:
                new_stats[category] += 1
            new_stats['total'] += 1
        
        # Update session stats
        self.session_stats = new_stats
        
        # Store hourly data for trends
        current_time = datetime.now()
        self.hourly_stats.append({
            'timestamp': current_time,
            'stats': new_stats.copy()
        })
        
        # Keep only last 24 hours of data
        cutoff_time = current_time - timedelta(hours=24)
        self.hourly_stats = [
            entry for entry in self.hourly_stats 
            if entry['timestamp'] > cutoff_time
        ]
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return self.session_stats.copy()
    
    def get_all_bins(self) -> List[Dict[str, Any]]:
        """Get all waste bins with their information"""
        all_bins = []
        for category, bins in self.waste_bins.items():
            for bin_info in bins:
                bin_data = bin_info.copy()
                bin_data['type'] = category
                all_bins.append(bin_data)
        return all_bins
    
    def find_nearest_bin(self, lat: float, lon: float, waste_category: str = None) -> Dict[str, Any]:
        """
        Find nearest appropriate waste bin
        Args:
            lat: Current latitude
            lon: Current longitude  
            waste_category: Type of waste (organic, recyclable, hazardous, other)
        Returns:
            Nearest bin information with distance
        """
        # Determine compatible bin types
        if waste_category == 'organic':
            compatible_types = ['organic', 'general']
        elif waste_category == 'recyclable':
            compatible_types = ['recyclable', 'general']  
        elif waste_category == 'hazardous':
            compatible_types = ['hazardous']  # Only hazardous bins for hazardous waste
        else:
            compatible_types = ['general']
        
        nearest_bin = None
        min_distance = float('inf')
        
        # Check all compatible bins
        for bin_type in compatible_types:
            if bin_type in self.waste_bins:
                for bin_info in self.waste_bins[bin_type]:
                    distance = self.calculate_distance(lat, lon, bin_info['lat'], bin_info['lon'])
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_bin = bin_info.copy()
                        nearest_bin['type'] = bin_type
                        nearest_bin['distance'] = distance
        
        return nearest_bin
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        Returns distance in meters
        """
        # Earth's radius in meters
        R = 6371000
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def get_trend_data(self) -> Dict[str, Any]:
        """Get hourly trend data for the last 24 hours"""
        if not self.hourly_stats:
            return {'hours': [], 'total': [], 'organic': [], 'recyclable': [], 'hazardous': [], 'other': []}
        
        # Group by hour
        hourly_data = {}
        for entry in self.hourly_stats:
            hour = entry['timestamp'].replace(minute=0, second=0, microsecond=0)
            if hour not in hourly_data:
                hourly_data[hour] = {'total': 0, 'organic': 0, 'recyclable': 0, 'hazardous': 0, 'other': 0}
            
            # Add to hourly totals
            for category in ['total', 'organic', 'recyclable', 'hazardous', 'other']:
                hourly_data[hour][category] += entry['stats'].get(category, 0)
        
        # Sort by hour and format for frontend
        sorted_hours = sorted(hourly_data.keys())
        
        trend_data = {
            'hours': [hour.strftime('%H:%M') for hour in sorted_hours],
            'total': [hourly_data[hour]['total'] for hour in sorted_hours],
            'organic': [hourly_data[hour]['organic'] for hour in sorted_hours],
            'recyclable': [hourly_data[hour]['recyclable'] for hour in sorted_hours],
            'hazardous': [hourly_data[hour]['hazardous'] for hour in sorted_hours],
            'other': [hourly_data[hour]['other'] for hour in sorted_hours]
        }
        
        return trend_data


# Global waste manager instance
waste_manager = None

def get_waste_manager() -> WasteManager:
    """Get global waste manager instance"""
    global waste_manager
    if waste_manager is None:
        waste_manager = WasteManager()
    return waste_manager
