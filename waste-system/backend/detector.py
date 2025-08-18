"""
YOLOv8 Detection Module
Handles waste detection using YOLOv8 model
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io


class WasteDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize YOLOv8 detector
        Args:
            model_path: Path to custom trained model, defaults to YOLOv8n
        """
        if model_path and model_path.endswith('.pt'):
            self.model = YOLO(model_path)
            print(f"Loaded custom model: {model_path}")
        else:
            # Use YOLOv8n as default
            self.model = YOLO('yolov8n.pt')
            print("Loaded default YOLOv8n model")
        
        # Waste classification mapping
        self.waste_categories = {
            # Common objects that can be waste
            'bottle': 'recyclable',
            'cup': 'recyclable', 
            'can': 'recyclable',
            'plastic_bag': 'other',
            'paper': 'recyclable',
            'cardboard': 'recyclable',
            'food': 'organic',
            'apple': 'organic',
            'banana': 'organic',
            'orange': 'organic',
            'sandwich': 'organic',
            'hot_dog': 'organic',
            'pizza': 'organic',
            'donut': 'organic',
            'cake': 'organic',
            'battery': 'hazardous',
            'cell_phone': 'hazardous',
            'laptop': 'hazardous',
            'mouse': 'hazardous',
            'keyboard': 'hazardous',
            'tv': 'hazardous',
            'scissors': 'hazardous',
            'teddy_bear': 'other',
            'book': 'recyclable',
            'clock': 'other',
            'vase': 'other',
            'toothbrush': 'other'
        }
    
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        try:
            # Decode base64 string
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format (BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
        except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None
    
    def detect_waste(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect waste objects in image
        Args:
            image: OpenCV image (BGR format)
            confidence_threshold: Minimum confidence for detection
        Returns:
            List of detections with bbox, label, confidence, category
        """
        try:
            # Run inference
            results = self.model(image, conf=confidence_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        # Determine waste category
                        waste_category = self.waste_categories.get(class_name, 'other')
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # [x, y, width, height]
                            'label': class_name,
                            'confidence': float(confidence),
                            'category': waste_category,
                            'class_id': class_id
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def detect_from_base64(self, base64_string: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect waste from base64 encoded image
        Args:
            base64_string: Base64 encoded image
            confidence_threshold: Minimum confidence for detection
        Returns:
            List of detections
        """
        image = self.base64_to_image(base64_string)
        if image is not None:
            return self.detect_waste(image, confidence_threshold)
        else:
            return []
    
    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from detections
        Args:
            detections: List of detection results
        Returns:
            Summary with counts by category
        """
        summary = {
            'total': len(detections),
            'organic': 0,
            'recyclable': 0, 
            'hazardous': 0,
            'other': 0,
            'by_class': {}
        }
        
        for detection in detections:
            category = detection.get('category', 'other')
            if category in summary:
                summary[category] += 1
            
            # Count by class
            class_name = detection.get('label', 'unknown')
            if class_name in summary['by_class']:
                summary['by_class'][class_name] += 1
            else:
                summary['by_class'][class_name] = 1
        
        return summary


# Global detector instance
detector = None

def get_detector(model_path: str = None) -> WasteDetector:
    """Get global detector instance"""
    global detector
    if detector is None:
        detector = WasteDetector(model_path)
    return detector
