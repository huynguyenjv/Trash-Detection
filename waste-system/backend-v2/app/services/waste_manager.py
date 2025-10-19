"""
Waste Manager Module
Handles waste counting and statistics
"""

from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class WasteManager:
    def __init__(self):
        """Initialize waste manager"""
        # Total counters
        self.totals = {
            'organic': 0,
            'recyclable': 0,
            'hazardous': 0,
            'other': 0
        }
        
        # Recent detections (last N items)
        self.recent_detections: List[Dict[str, Any]] = []
        self.max_recent = 100
    
    def update(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Update counters from new detections
        
        Args:
            detections: List of detection dicts
            
        Returns:
            Current counts by category
        """
        counts = defaultdict(int)
        
        for detection in detections:
            category = detection.get('category', 'other')
            counts[category] += 1
            
            # Update totals
            if category in self.totals:
                self.totals[category] += 1
            
            # Add to recent (with timestamp)
            recent_item = {
                'timestamp': datetime.now().isoformat(),
                'label': detection['label'],
                'category': category,
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            }
            self.recent_detections.append(recent_item)
        
        # Keep only last N items
        if len(self.recent_detections) > self.max_recent:
            self.recent_detections = self.recent_detections[-self.max_recent:]
        
        return dict(counts)
    
    def get_stats(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get current statistics
        
        Args:
            limit: Number of recent items to return
            
        Returns:
            Dict with totals and recent detections
        """
        return {
            'totals': self.totals.copy(),
            'recent': self.recent_detections[-limit:]
        }
    
    def reset(self):
        """Reset all counters"""
        for key in self.totals:
            self.totals[key] = 0
        self.recent_detections.clear()
