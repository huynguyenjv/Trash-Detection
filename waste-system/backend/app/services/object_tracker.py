"""
Object Tracker for unique detection
Only saves to DB when object disappears (lifecycle complete)
"""

import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class ObjectData:
    """Data structure for a tracked object"""
    
    def __init__(self, obj_id: str, label: str, category: str, bbox: list, confidence: float, first_seen: float):
        self.id = obj_id
        self.label = label
        self.category = category
        self.bbox = bbox  # [x1, y1, x2, y2] - corner coordinates
        self.confidence = confidence
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.detection_count = 1
        self.total_confidence = confidence
    
    def update(self, detection: dict, timestamp: float):
        """Update object with new detection from same object"""
        self.last_seen = timestamp
        self.detection_count += 1
        self.total_confidence += detection['confidence']
        
        # Update bbox using exponential moving average (smoother)
        alpha = 0.3  # Weight for new detection
        new_bbox = detection['bbox']
        self.bbox = [
            alpha * new_bbox[i] + (1 - alpha) * self.bbox[i]
            for i in range(4)
        ]
        
        # Update confidence (keep max)
        self.confidence = max(self.confidence, detection['confidence'])
    
    def to_detection_record(self) -> dict:
        """Convert to format for database saving"""
        return {
            'label': self.label,
            'category': self.category,
            'bbox': [round(x) for x in self.bbox],
            'confidence': round(self.confidence, 2),
            'avg_confidence': round(self.total_confidence / self.detection_count, 2),
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'duration': round(self.last_seen - self.first_seen, 2),
            'frame_count': self.detection_count
        }
    
    def get_duration(self) -> float:
        """Get how long this object has been tracked"""
        return self.last_seen - self.first_seen


class ObjectTracker:
    """
    Track unique objects across frames using spatial overlap (IOU)
    
    Strategy:
    1. Each frame: Match detections to existing tracked objects
    2. If match found (same label + high IOU): Update existing object
    3. If no match: Create new tracked object
    4. If object not seen for N seconds: Mark as disappeared, save to DB
    
    This ensures: 1 physical object = 1 database record
    """
    
    def __init__(self, disappear_threshold: float = 3.0, iou_threshold: float = 0.4):
        """
        Args:
            disappear_threshold: Seconds without detection before object considered disappeared
            iou_threshold: Minimum IOU to consider same object (0.4 = 40% overlap)
        """
        self.active_objects: Dict[str, ObjectData] = {}
        self.next_id = 1
        self.disappear_threshold = disappear_threshold
        self.iou_threshold = iou_threshold
        
        print(f"🎯 ObjectTracker initialized: disappear={disappear_threshold}s, IOU={iou_threshold}")
    
    def update(self, detections: List[dict]) -> Tuple[Dict[str, ObjectData], List[dict]]:
        """
        Update tracker with new frame detections
        
        Args:
            detections: List of detections from current frame
                        [{'label': 'bottle', 'category': 'recyclable', 'bbox': [...], 'confidence': 0.85}, ...]
        
        Returns:
            Tuple of:
            - active_objects: Currently tracked objects (still in view)
            - completed_objects: Objects that just disappeared (ready to save to DB)
        """
        current_time = time.time()
        matched_ids = set()
        
        # Match each detection to existing objects
        for det in detections:
            object_id = self._find_matching_object(det)
            
            if object_id:
                # Update existing object
                self.active_objects[object_id].update(det, current_time)
                matched_ids.add(object_id)
            else:
                # New object detected
                new_id = self._create_new_object(det, current_time)
                matched_ids.add(new_id)
        
        # Check for disappeared objects (not matched in current frame)
        completed_objects = []
        for obj_id, obj_data in list(self.active_objects.items()):
            if obj_id not in matched_ids:
                time_since_seen = current_time - obj_data.last_seen
                
                if time_since_seen > self.disappear_threshold:
                    # Object disappeared → Complete lifecycle → Save to DB
                    record = obj_data.to_detection_record()
                    completed_objects.append(record)
                    
                    print(f"✅ Object disappeared: {obj_data.label} (tracked {obj_data.get_duration():.1f}s, {obj_data.detection_count} frames)")
                    
                    del self.active_objects[obj_id]
        
        return self.active_objects, completed_objects
    
    def _find_matching_object(self, detection: dict) -> Optional[str]:
        """
        Find existing tracked object that matches this detection
        
        Match criteria:
        1. Same label (e.g., both are 'bottle')
        2. High spatial overlap (IOU > threshold)
        """
        label = detection['label']
        bbox = detection['bbox']
        
        best_match_id = None
        best_iou = 0
        
        for obj_id, obj_data in self.active_objects.items():
            if obj_data.label == label:
                iou = self._calculate_iou(bbox, obj_data.bbox)
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_match_id = obj_id
                    best_iou = iou
        
        return best_match_id
    
    def _create_new_object(self, detection: dict, timestamp: float) -> str:
        """Create new tracked object"""
        obj_id = f"{detection['label']}_{self.next_id}"
        self.next_id += 1
        
        self.active_objects[obj_id] = ObjectData(
            obj_id=obj_id,
            label=detection['label'],
            category=detection['category'],
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            first_seen=timestamp
        )
        
        print(f"🆕 New object tracked: {obj_id} at bbox={detection['bbox']}")
        
        return obj_id
    
    def _calculate_iou(self, bbox1: list, bbox2: list) -> float:
        """
        Calculate Intersection over Union (IOU) between two bounding boxes
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2] - corner coordinates
        
        Returns:
            IOU score (0.0 to 1.0)
        """
        # Unpack corner coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection rectangle
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        return iou
    
    def force_save_all(self) -> List[dict]:
        """
        Force save all active objects to DB
        
        Used when:
        - WebSocket disconnects
        - Session ends
        - System shutdown
        
        Returns:
            List of detection records for all active objects
        """
        completed = []
        
        for obj_data in self.active_objects.values():
            record = obj_data.to_detection_record()
            completed.append(record)
            print(f"💾 Force saved: {obj_data.label} (tracked {obj_data.get_duration():.1f}s)")
        
        self.active_objects.clear()
        
        return completed
    
    def get_stats(self) -> dict:
        """Get current tracking statistics"""
        return {
            'active_objects': len(self.active_objects),
            'next_id': self.next_id,
            'objects_by_label': self._count_by_label()
        }
    
    def _count_by_label(self) -> dict:
        """Count active objects by label"""
        counts = {}
        for obj_data in self.active_objects.values():
            counts[obj_data.label] = counts.get(obj_data.label, 0) + 1
        return counts
