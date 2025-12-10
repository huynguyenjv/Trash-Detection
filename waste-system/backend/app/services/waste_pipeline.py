"""
2-Stage Waste Detection Pipeline
Stage 1: YOLOv8 Detection (bbox)
Stage 2: YOLOv8 Classification (waste type)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import torch
import os


class WastePipeline:
    """
    Complete waste detection pipeline:
    1. Detect objects with bounding boxes (YOLO Detection)
    2. Classify each detected object (YOLO Classification)
    """
    
    def __init__(self, 
                 detection_model_path: str = 'yolov8n.pt',
                 classification_model_path: Optional[str] = None,
                 use_classification: bool = False):
        """
        Initialize 2-stage pipeline
        
        Args:
            detection_model_path: Path to YOLO detection model
            classification_model_path: Path to YOLO classification model (optional)
            use_classification: Enable classification stage (default: False)
        """
        # Fix PyTorch compatibility
        os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
        
        print("=" * 60)
        print("🚀 Initializing Waste Detection Pipeline")
        print("=" * 60)
        
        # Stage 1: Detection Model
        print(f"\n📍 Stage 1: Loading Detection Model")
        print(f"   Model: {detection_model_path}")
        self.detector = self._load_model(detection_model_path)
        print("   ✅ Detection model loaded!")
        
        # Stage 2: Classification Model (optional)
        self.use_classification = use_classification and classification_model_path is not None
        
        if self.use_classification:
            print(f"\n📍 Stage 2: Loading Classification Model")
            print(f"   Model: {classification_model_path}")
            self.classifier = self._load_model(classification_model_path)
            print("   ✅ Classification model loaded!")
            
            # Class names from classification model
            if hasattr(self.classifier, 'names'):
                self.class_names = self.classifier.names
                print(f"   📊 Classes: {list(self.class_names.values())}")
            else:
                self.class_names = {}
                print("   ⚠️  Warning: No class names found in model")
        else:
            self.classifier = None
            self.class_names = {}
            print("\n📍 Stage 2: Classification DISABLED")
            print("   Using rule-based category mapping")
        
        # Fallback: Rule-based waste mapping (when classification disabled)
        self.waste_mapping = {
            # Recyclable
            'bottle': 'recyclable',
            'cup': 'recyclable',
            'wine glass': 'recyclable',
            'fork': 'recyclable',
            'knife': 'recyclable',
            'spoon': 'recyclable',
            'bowl': 'recyclable',
            'book': 'recyclable',
            
            # Organic
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
            
            # Hazardous
            'cell phone': 'hazardous',
            'laptop': 'hazardous',
            'mouse': 'hazardous',
            'keyboard': 'hazardous',
            'remote': 'hazardous',
            'scissors': 'hazardous',
            'hair drier': 'hazardous',
        }
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Ready!")
        print("=" * 60 + "\n")
    
    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLO model with PyTorch 2.6+ compatibility"""
        try:
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except:
            pass
        
        try:
            return YOLO(model_path)
        except Exception as e:
            # Fallback: Patch torch.load
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            try:
                return YOLO(model_path)
            finally:
                torch.load = original_load
    
    def process_frame(self, 
                     frame: np.ndarray,
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Process single frame through full pipeline
        
        Args:
            frame: Input image (BGR format)
            conf_threshold: Confidence threshold for detection (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.6 - tighter boxes, closer to training 0.7)
        
        Returns:
            List of detections with bbox, label, confidence, category
        
        Note:
            - Training used iou=0.7, using 0.6 for inference ensures tight bounding boxes
            - Higher IoU = more aggressive NMS = fewer overlapping boxes = tighter fit
        """
        if self.use_classification:
            return self._process_with_classification(frame, conf_threshold, iou_threshold)
        else:
            return self._process_detection_only(frame, conf_threshold, iou_threshold)
    
    def _process_detection_only(self,
                                frame: np.ndarray,
                                conf_threshold: float,
                                iou_threshold: float) -> List[Dict[str, Any]]:
        """
        Stage 1 only: Detection with rule-based mapping
        """
        results = self.detector(
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
                label = self.detector.names[class_id]
                category = self.waste_mapping.get(label, 'other')
                
                # Skip ignored objects
                if category == 'ignore':
                    continue
                
                x1, y1, x2, y2 = box.astype(int)
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],  # [x1, y1, x2, y2] - same as detector.py
                    'label': label,
                    'confidence': float(score),
                    'category': category,
                    'pipeline_stage': 'detection_only'
                }
                detections.append(detection)
        
        return detections
    
    def _process_with_classification(self,
                                    frame: np.ndarray,
                                    conf_threshold: float,
                                    iou_threshold: float) -> List[Dict[str, Any]]:
        """
        Full 2-stage pipeline: Detection → Classification
        """
        # Stage 1: Detection
        detection_results = self.detector(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in detection_results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            det_scores = result.boxes.conf.cpu().numpy()
            
            for box, det_score in zip(boxes, det_scores):
                x1, y1, x2, y2 = box.astype(int)
                
                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop detected object
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Stage 2: Classification
                try:
                    cls_result = self.classifier(crop, verbose=False)
                    
                    if cls_result and hasattr(cls_result[0], 'probs'):
                        # Get top class
                        probs = cls_result[0].probs
                        class_id = int(probs.top1)
                        cls_confidence = float(probs.top1conf)
                        
                        # Get class name
                        waste_class = self.class_names.get(class_id, f'class_{class_id}')
                        
                        # Map to category
                        category = self._map_class_to_category(waste_class)
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],  # [x1, y1, x2, y2] - same as detector.py
                            'label': waste_class,
                            'confidence': cls_confidence,
                            'category': category,
                            'detection_confidence': float(det_score),
                            'pipeline_stage': 'detection_classification'
                        }
                        detections.append(detection)
                    
                except Exception as e:
                    print(f"⚠️  Classification failed for bbox [{x1},{y1},{x2},{y2}]: {e}")
                    continue
        
        return detections
    
    def _map_class_to_category(self, waste_class: str) -> str:
        """
        Map classification result to waste category
        
        TODO: Update this mapping based on your classification model classes
        
        Args:
            waste_class: Class name from classification model
        
        Returns:
            Waste category: 'organic', 'recyclable', 'hazardous', or 'other'
        """
        # Example mapping - UPDATE THIS based on your classes!
        category_mapping = {
            # Recyclable
            'plastic': 'recyclable',
            'plastic_bottle': 'recyclable',
            'glass': 'recyclable',
            'glass_bottle': 'recyclable',
            'metal': 'recyclable',
            'metal_can': 'recyclable',
            'paper': 'recyclable',
            'cardboard': 'recyclable',
            
            # Organic
            'organic': 'organic',
            'food_waste': 'organic',
            'compost': 'organic',
            
            # Hazardous
            'battery': 'hazardous',
            'electronics': 'hazardous',
            'chemical': 'hazardous',
            
            # Other
            'other': 'other',
            'mixed': 'other'
        }
        
        return category_mapping.get(waste_class.lower(), 'other')
    
    def bytes_to_frame(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to OpenCV frame"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return {
            'use_classification': self.use_classification,
            'detection_model': 'loaded',
            'classification_model': 'loaded' if self.classifier else 'disabled',
            'classification_classes': list(self.class_names.values()) if self.class_names else [],
            'num_classes': len(self.class_names)
        }
