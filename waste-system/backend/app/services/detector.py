"""
YOLOv8 Detection Module
Handles waste detection using YOLOv8 model with multi-object support
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import torch
import os


class WasteDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLO model (default: yolov8n.pt)
        """
        # Fix PyTorch 2.6+ compatibility
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
        
        try:
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except:
            pass
        
        print(f"Loading YOLO model: {model_path}")
        
        # Try loading with patched torch.load
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"⚠️  First load attempt failed, using patched method...")
            # Monkey-patch torch.load
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            try:
                self.model = YOLO(model_path)
            finally:
                torch.load = original_load
        
        print(f"✅ Model loaded successfully!")
        
        # Waste category mapping - Updated for trash detection dataset
        self.waste_mapping = {
            # Recyclable materials
            'bottle': 'recyclable',
            'cup': 'recyclable', 
            'wine glass': 'recyclable',
            'fork': 'recyclable',
            'knife': 'recyclable',
            'spoon': 'recyclable',
            'bowl': 'recyclable',
            'book': 'recyclable',
            
            # Trash detection dataset classes
            'paper': 'recyclable',        # Paper → recyclable
            'cardboard': 'recyclable',    # Cardboard → recyclable  
            'plastic': 'recyclable',      # Plastic → recyclable
            'glass': 'recyclable',        # Glass → recyclable
            'metal': 'recyclable',        # Metal → recyclable
            
            'biological': 'organic',      # Biological waste → organic
            
            'battery': 'hazardous',       # Battery → hazardous
            
            'clothes': 'other',           # Clothes → other (textile waste)
            'shoes': 'other',             # Shoes → other  
            'trash': 'other',             # General trash → other
            
            # Original COCO mappings for organic
            'banana': 'organic',
            'apple': 'organic',
            'orange': 'organic',
            'broccoli': 'organic',
            'carrot': 'organic',
            'hot dog': 'organic',
            'pizza': 'organic',
            'donut': 'organic',
            'cake': 'organic',
            'sandwich': 'organic',
            
            # Original COCO mappings for hazardous
            'cell phone': 'hazardous',
            'laptop': 'hazardous',
            'mouse': 'hazardous',
            'keyboard': 'hazardous',
            'remote': 'hazardous',
            'scissors': 'hazardous',
            'hair drier': 'hazardous',
            
            # Ignore (not waste)
            'person': 'ignore',
            'car': 'ignore',
            'truck': 'ignore',
            'bus': 'ignore',
            'bicycle': 'ignore',
            'motorcycle': 'ignore'
        }
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25, 
               iou_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Detect objects in single frame
        
        Args:
            frame: OpenCV image (BGR)
            conf_threshold: Confidence threshold (default: 0.25 - balanced for real-time)
            iou_threshold: IoU threshold for NMS (default: 0.6 - tighter boxes, closer to training 0.7)
            
        Returns:
            List of detections with bbox, label, confidence, category
        
        Note:
            - Training used iou=0.7, using 0.6 for inference ensures tight bounding boxes
            - Higher IoU = more aggressive NMS = fewer overlapping boxes = tighter fit
        """
        results = self.model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                label = self.model.names[class_id]
                category = self.waste_mapping.get(label, 'other')
                
                # Skip ignored objects
                if category == 'ignore':
                    continue
                
                x1, y1, x2, y2 = box.astype(int)
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'label': label,
                    'confidence': float(score),
                    'category': category
                }
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray], conf_threshold: float = 0.25,
                    iou_threshold: float = 0.6) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in multiple frames (batch processing)
        
        Args:
            frames: List of OpenCV images
            conf_threshold: Confidence threshold (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.6 - tighter boxes)
            
        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []
        
        for frame in frames:
            detections = self.detect(frame, conf_threshold, iou_threshold)
            all_detections.append(detections)
        
        return all_detections
    
    @staticmethod
    def bytes_to_frame(image_bytes: bytes) -> np.ndarray:
        """
        Convert bytes to OpenCV frame
        
        Args:
            image_bytes: Image bytes (JPEG/PNG)
            
        Returns:
            OpenCV image (BGR)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
