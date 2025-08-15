"""
GUI Helper - Các helper functions cho giao diện
"""

import os
import sys
import matplotlib
from typing import Optional, Tuple


class GUIHelper:
    """Helper class cho các tác vụ GUI"""
    
    @staticmethod
    def setup_matplotlib_backend() -> str:
        """
        Tự động detect và setup matplotlib backend phù hợp
        
        Returns:
            Backend name được sử dụng
        """
        # Check if we're in a headless environment
        if os.environ.get('DISPLAY') is None and sys.platform.startswith('linux'):
            matplotlib.use('Agg')
            return 'Agg'
        
        # Try different backends in order of preference
        backends_to_try = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTKAgg']
        
        for backend in backends_to_try:
            try:
                matplotlib.use(backend)
                import matplotlib.pyplot as plt
                # Test if backend works
                fig, ax = plt.subplots()
                plt.close(fig)
                return backend
            except Exception:
                continue
        
        # Fallback to Agg
        matplotlib.use('Agg')
        return 'Agg'
    
    @staticmethod
    def check_gui_availability() -> Tuple[bool, str]:
        """
        Kiểm tra xem GUI có khả dụng không
        
        Returns:
            Tuple (is_available, message)
        """
        try:
            backend = GUIHelper.setup_matplotlib_backend()
            
            if backend == 'Agg':
                return False, "Headless environment detected - GUI not available"
            
            # Test GUI creation
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(1, 1))
            plt.close(fig)
            
            return True, f"GUI available with backend: {backend}"
            
        except Exception as e:
            return False, f"GUI not available: {str(e)}"
    
    @staticmethod
    def get_optimal_figure_size() -> Tuple[int, int]:
        """
        Lấy kích thước figure tối ưu dựa trên screen
        
        Returns:
            Tuple (width, height) in inches
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get screen size if possible
            if hasattr(plt, 'get_current_fig_manager'):
                try:
                    fig = plt.figure(figsize=(1, 1))
                    manager = fig.canvas.manager
                    if hasattr(manager, 'window'):
                        # Try to get screen dimensions
                        screen_width = 1920  # Default
                        screen_height = 1080
                        
                        # Calculate optimal size (80% of screen)
                        dpi = 100  # Default DPI
                        width = int((screen_width * 0.8) / dpi)
                        height = int((screen_height * 0.8) / dpi)
                        
                        plt.close(fig)
                        return (width, height)
                    
                    plt.close(fig)
                except Exception:
                    pass
            
        except Exception:
            pass
        
        # Default size
        return (14, 10)
    
    @staticmethod
    def create_color_scheme() -> dict:
        """
        Tạo color scheme cho ứng dụng
        
        Returns:
            Dictionary với các màu sắc
        """
        return {
            # Waste bin colors
            'bin_ok': '#4CAF50',          # Green
            'bin_near_full': '#FF9800',   # Orange  
            'bin_full': '#F44336',        # Red
            
            # Route colors
            'route_normal': '#2196F3',    # Blue
            'route_optimal': '#4CAF50',   # Green
            'route_traffic': '#FF5722',   # Deep Orange
            
            # UI colors
            'background': '#FAFAFA',      # Light Gray
            'text_primary': '#212121',    # Dark Gray
            'text_secondary': '#757575',  # Medium Gray
            'accent': '#3F51B5',          # Indigo
            
            # Status colors
            'success': '#4CAF50',         # Green
            'warning': '#FF9800',         # Orange
            'error': '#F44336',           # Red
            'info': '#2196F3'             # Blue
        }
    
    @staticmethod
    def validate_coordinates(lat: float, lng: float) -> bool:
        """
        Validate GPS coordinates
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            True if valid, False otherwise
        """
        return (-90 <= lat <= 90) and (-180 <= lng <= 180)
    
    @staticmethod
    def format_distance(distance_km: float) -> str:
        """
        Format distance for display
        
        Args:
            distance_km: Distance in kilometers
            
        Returns:
            Formatted string
        """
        if distance_km < 1:
            return f"{distance_km * 1000:.0f}m"
        else:
            return f"{distance_km:.2f}km"
    
    @staticmethod
    def format_time(time_minutes: float) -> str:
        """
        Format time for display
        
        Args:
            time_minutes: Time in minutes
            
        Returns:
            Formatted string
        """
        if time_minutes < 60:
            return f"{time_minutes:.0f} phút"
        else:
            hours = int(time_minutes // 60)
            minutes = int(time_minutes % 60)
            return f"{hours}h {minutes}p"
    
    @staticmethod
    def format_fuel(fuel_liters: float) -> str:
        """
        Format fuel consumption for display
        
        Args:
            fuel_liters: Fuel in liters
            
        Returns:
            Formatted string
        """
        return f"{fuel_liters:.2f}L"
    
    @staticmethod
    def get_status_icon(status: str) -> str:
        """
        Get icon for status
        
        Args:
            status: Status string
            
        Returns:
            Unicode icon
        """
        icons = {
            'OK': '✅',
            'NEAR_FULL': '⚠️',
            'FULL': '❌',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️'
        }
        
        return icons.get(status, '❓')
    
    @staticmethod
    def create_progress_bar(current: float, maximum: float, width: int = 20) -> str:
        """
        Create ASCII progress bar
        
        Args:
            current: Current value
            maximum: Maximum value
            width: Bar width in characters
            
        Returns:
            Progress bar string
        """
        if maximum == 0:
            percentage = 0
        else:
            percentage = min(current / maximum, 1.0)
        
        filled = int(width * percentage)
        bar = '█' * filled + '░' * (width - filled)
        
        return f"[{bar}] {percentage*100:.1f}%"
