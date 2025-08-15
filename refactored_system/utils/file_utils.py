"""
File Utils - Utilities for file operations
"""

import os
import json
import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
        """
        Save data to JSON file
        
        Args:
            data: Data to save
            file_path: Path to save file
            indent: JSON indentation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error saving JSON to {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Any]:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error loading JSON from {file_path}: {e}")
            return None
    
    @staticmethod
    def save_csv(data: List[Dict], file_path: str) -> bool:
        """
        Save data to CSV file
        
        Args:
            data: List of dictionaries to save
            file_path: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        if not data:
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            
            return True
            
        except Exception as e:
            print(f"Error saving CSV to {file_path}: {e}")
            return False
    
    @staticmethod
    def load_csv(file_path: str) -> Optional[List[Dict]]:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of dictionaries or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
                
        except Exception as e:
            print(f"Error loading CSV from {file_path}: {e}")
            return None
    
    @staticmethod
    def save_pickle(data: Any, file_path: str) -> bool:
        """
        Save data using pickle
        
        Args:
            data: Data to save
            file_path: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            print(f"Error saving pickle to {file_path}: {e}")
            return False
    
    @staticmethod
    def load_pickle(file_path: str) -> Optional[Any]:
        """
        Load data from pickle file
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Loaded data or None if failed
        """
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            print(f"Error loading pickle from {file_path}: {e}")
            return None
    
    @staticmethod
    def create_backup(file_path: str) -> Optional[str]:
        """
        Create backup of file
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path to backup file or None if failed
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{file_path}.backup_{timestamp}"
            
            import shutil
            shutil.copy2(file_path, backup_path)
            
            return backup_path
            
        except Exception as e:
            print(f"Error creating backup of {file_path}: {e}")
            return None
    
    @staticmethod
    def ensure_directory(directory: str) -> bool:
        """
        Ensure directory exists
        
        Args:
            directory: Directory path
            
        Returns:
            True if directory exists or was created
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
            
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    
    @staticmethod
    def get_file_size(file_path: str) -> Optional[int]:
        """
        Get file size in bytes
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes or None if failed
        """
        try:
            return os.path.getsize(file_path)
            
        except Exception:
            return None
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size for display
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        
        return f"{size_bytes:.1f} TB"
    
    @staticmethod
    def list_files(directory: str, extension: str = None) -> List[str]:
        """
        List files in directory
        
        Args:
            directory: Directory to search
            extension: File extension to filter (e.g., '.json')
            
        Returns:
            List of file paths
        """
        try:
            files = []
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isfile(item_path):
                    if extension is None or item.endswith(extension):
                        files.append(item_path)
            
            return sorted(files)
            
        except Exception as e:
            print(f"Error listing files in {directory}: {e}")
            return []
    
    @staticmethod
    def clean_old_files(directory: str, days_old: int = 7) -> int:
        """
        Clean files older than specified days
        
        Args:
            directory: Directory to clean
            days_old: Files older than this will be deleted
            
        Returns:
            Number of files deleted
        """
        if not os.path.exists(directory):
            return 0
        
        try:
            import time
            
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)
            deleted_count = 0
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isfile(item_path):
                    file_time = os.path.getmtime(item_path)
                    
                    if file_time < cutoff_time:
                        os.remove(item_path)
                        deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning old files in {directory}: {e}")
            return 0
