"""
Data Generator - Tạo dữ liệu mẫu cho testing và demo
"""

import random
import numpy as np
from typing import List, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GPSCoordinate, WasteBin, Road, VehicleInfo
from core.enums import WasteType, BinStatus, TrafficCondition
from core.routing_engine import RoutingEngine


class DataGenerator:
    """Class tạo dữ liệu mẫu cho hệ thống"""
    
    @staticmethod
    def create_sample_waste_bins(center: GPSCoordinate, count: int = 10) -> List[WasteBin]:
        """
        Tạo sample waste bins quanh một vị trí trung tâm
        
        Args:
            center: Tọa độ trung tâm
            count: Số lượng thùng rác
            
        Returns:
            List of WasteBin objects
        """
        bins = []
        np.random.seed(42)  # For reproducible results
        
        for i in range(count):
            # Random location around center (within ~2km radius)
            lat_offset = np.random.uniform(-0.01, 0.01)
            lng_offset = np.random.uniform(-0.01, 0.01)
            
            location = GPSCoordinate(
                center.lat + lat_offset,
                center.lng + lng_offset
            )
            
            # Random properties
            status = np.random.choice([
                BinStatus.OK, BinStatus.NEAR_FULL, BinStatus.FULL
            ], p=[0.6, 0.3, 0.1])
            
            supported_types = np.random.choice(
                list(WasteType), 
                size=np.random.randint(1, 4),
                replace=False
            ).tolist()
            
            max_capacity = np.random.uniform(80, 150)
            
            if status == BinStatus.FULL:
                current_capacity = max_capacity * np.random.uniform(0.9, 1.0)
            elif status == BinStatus.NEAR_FULL:
                current_capacity = max_capacity * np.random.uniform(0.7, 0.9)
            else:
                current_capacity = max_capacity * np.random.uniform(0.1, 0.7)
            
            priority = np.random.randint(1, 6)
            
            waste_bin = WasteBin(
                id=f"BIN{i+1:03d}",
                location=location,
                status=status,
                supported_types=supported_types,
                max_capacity=max_capacity,
                current_capacity=current_capacity,
                priority=priority
            )
            
            bins.append(waste_bin)
        
        return bins
    
    @staticmethod
    def create_sample_road_network(center: GPSCoordinate, complexity: str = "medium") -> List[Road]:
        """
        Tạo sample road network
        
        Args:
            center: Tọa độ trung tâm
            complexity: "simple", "medium", "complex"
            
        Returns:
            List of Road objects
        """
        roads = []
        np.random.seed(42)
        
        # Define grid points based on complexity
        if complexity == "simple":
            grid_size = 3
            step = 0.005
        elif complexity == "medium":
            grid_size = 5
            step = 0.003
        else:  # complex
            grid_size = 7
            step = 0.002
        
        # Create grid points
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                lat = center.lat + (i - grid_size//2) * step
                lng = center.lng + (j - grid_size//2) * step
                grid_points.append(GPSCoordinate(lat, lng))
        
        # Connect adjacent points horizontally and vertically
        for i in range(grid_size):
            for j in range(grid_size):
                current_idx = i * grid_size + j
                
                # Horizontal connection
                if j < grid_size - 1:
                    next_idx = i * grid_size + (j + 1)
                    road = Road(
                        start=grid_points[current_idx],
                        end=grid_points[next_idx],
                        traffic_condition=np.random.choice(list(TrafficCondition)),
                        max_speed=np.random.choice([30, 40, 50, 60])
                    )
                    roads.append(road)
                
                # Vertical connection
                if i < grid_size - 1:
                    next_idx = (i + 1) * grid_size + j
                    road = Road(
                        start=grid_points[current_idx],
                        end=grid_points[next_idx],
                        traffic_condition=np.random.choice(list(TrafficCondition)),
                        max_speed=np.random.choice([30, 40, 50, 60])
                    )
                    roads.append(road)
        
        # Add some diagonal roads for complexity
        if complexity in ["medium", "complex"]:
            for _ in range(grid_size):
                i = np.random.randint(0, grid_size - 1)
                j = np.random.randint(0, grid_size - 1)
                
                current_idx = i * grid_size + j
                diag_idx = (i + 1) * grid_size + (j + 1)
                
                road = Road(
                    start=grid_points[current_idx],
                    end=grid_points[diag_idx],
                    traffic_condition=np.random.choice(list(TrafficCondition)),
                    max_speed=np.random.choice([25, 35, 45])
                )
                roads.append(road)
        
        return roads
    
    @staticmethod
    def create_sample_vehicles(center: GPSCoordinate, count: int = 3) -> List[VehicleInfo]:
        """
        Tạo sample vehicles
        
        Args:
            center: Tọa độ trung tâm
            count: Số lượng xe
            
        Returns:
            List of VehicleInfo objects
        """
        vehicles = []
        np.random.seed(42)
        
        vehicle_types = [
            {"capacity": 800, "fuel_consumption": 12, "speed": 25},    # Small truck
            {"capacity": 1200, "fuel_consumption": 15, "speed": 22},  # Medium truck  
            {"capacity": 1800, "fuel_consumption": 20, "speed": 20}   # Large truck
        ]
        
        for i in range(count):
            # Random location near center
            lat_offset = np.random.uniform(-0.005, 0.005)
            lng_offset = np.random.uniform(-0.005, 0.005)
            
            location = GPSCoordinate(
                center.lat + lat_offset,
                center.lng + lng_offset
            )
            
            vehicle_type = np.random.choice(vehicle_types)
            
            vehicle = VehicleInfo(
                id=f"TRUCK{i+1:02d}",
                current_location=location,
                fuel_level=np.random.uniform(30, 100),
                capacity=vehicle_type["capacity"],
                current_load=np.random.uniform(0, vehicle_type["capacity"] * 0.3),
                avg_speed=vehicle_type["speed"],
                fuel_consumption=vehicle_type["fuel_consumption"]
            )
            
            vehicles.append(vehicle)
        
        return vehicles
    
    @staticmethod
    def create_complete_system(center: GPSCoordinate = None) -> tuple:
        """
        Tạo hệ thống hoàn chỉnh với routing engine và dữ liệu mẫu
        
        Args:
            center: Tọa độ trung tâm (mặc định: TP.HCM)
            
        Returns:
            Tuple (routing_engine, waste_bins, vehicles)
        """
        if center is None:
            center = GPSCoordinate(10.77, 106.68)  # TP.HCM
        
        # Create routing engine
        routing_engine = RoutingEngine()
        
        # Create road network
        roads = DataGenerator.create_sample_road_network(center, "medium")
        for road in roads:
            routing_engine.add_road(road)
        
        # Create waste bins
        waste_bins = DataGenerator.create_sample_waste_bins(center, 12)
        
        # Create vehicles
        vehicles = DataGenerator.create_sample_vehicles(center, 3)
        
        return routing_engine, waste_bins, vehicles
    
    @staticmethod
    def generate_traffic_data() -> Dict[str, float]:
        """Tạo dữ liệu giao thông mẫu"""
        return {
            "avg_speed": np.random.uniform(20, 60),
            "congestion_level": np.random.uniform(0, 1),
            "estimated_delay": np.random.uniform(0, 30),
            "road_quality": np.random.uniform(0.5, 1.0)
        }
    
    @staticmethod
    def generate_weather_data() -> Dict[str, any]:
        """Tạo dữ liệu thời tiết mẫu"""
        weather_conditions = ["sunny", "cloudy", "rainy", "stormy"]
        
        return {
            "condition": np.random.choice(weather_conditions),
            "temperature": np.random.uniform(25, 35),
            "humidity": np.random.uniform(60, 90),
            "wind_speed": np.random.uniform(5, 25),
            "visibility": np.random.uniform(1, 10)
        }
