"""
Smart Waste Management System - Refactored Package
Hệ thống quản lý rác thải thông minh được tái cấu trúc

Author: Smart Waste Management Team
Date: August 2025
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Smart Waste Management Team"

# Core imports
from .core import *
from .interfaces import *
from .utils import *
from .config import *

# Main application
from .main import SmartWasteManagementSystem

__all__ = [
    'SmartWasteManagementSystem'
]
