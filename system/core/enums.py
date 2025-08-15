"""
Core Enums - Các enum định nghĩa trạng thái và loại rác
"""

from enum import Enum


class WasteType(Enum):
    """Enum cho các loại rác thải"""
    ORGANIC = "organic"          # Rác hữu cơ
    PLASTIC = "plastic"          # Nhựa
    GLASS = "glass"              # Thủy tinh
    METAL = "metal"              # Kim loại
    PAPER = "paper"              # Giấy
    CARDBOARD = "cardboard"      # Bìa carton
    BATTERY = "battery"          # Pin
    CLOTHES = "clothes"          # Quần áo
    SHOES = "shoes"              # Giày dép
    GENERAL = "general"          # Rác thải chung


class BinStatus(Enum):
    """Trạng thái thùng rác"""
    OK = "OK"                    # Bình thường
    NEAR_FULL = "NEAR_FULL"      # Gần đầy
    FULL = "FULL"                # Đầy


class TrafficCondition(Enum):
    """Điều kiện giao thông"""
    CLEAR = "clear"              # Thông thoáng
    LIGHT = "light"              # Ít xe
    MODERATE = "moderate"        # Trung bình
    HEAVY = "heavy"              # Đông xe
    BLOCKED = "blocked"          # Tắc đường


class MovementDirection(Enum):
    """Hướng di chuyển"""
    NORTH = "north"
    SOUTH = "south" 
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"
