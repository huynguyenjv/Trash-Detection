"""
YOLOv8 Detection Module
Handles waste detection using YOLOv8 model
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io
import os


class WasteDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize YOLOv8 detector
        Args:
            model_path: Path to custom trained model, defaults to YOLOv8n
        """
        # Fix for PyTorch 2.6+ compatibility
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Set environment variable for PyTorch weights loading
        os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
        
        try:
            # Import ultralytics classes for safe loading
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except Exception as e:
            print(f"Warning: Could not add safe globals: {e}")
            # Try alternative approach
            try:
                import torch.serialization
                torch.serialization._clear_safe_globals()
            except:
                pass
            
        try:
            if model_path and model_path.endswith('.pt') and os.path.exists(model_path):
                print(f"Loading custom model: {model_path}")
                self.model = YOLO(model_path)
                print(f"Successfully loaded custom model: {model_path}")
            else:
                # Use YOLOv8n as default - will download if not exists
                print("Loading default YOLOv8n model...")
                self.model = YOLO('yolov8n.pt')
                print("Successfully loaded default YOLOv8n model")
        except Exception as e:
            print(f"Error loading model with safe mode: {e}")
            # Try with monkey-patching torch.load
            import torch.serialization
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            try:
                if model_path and model_path.endswith('.pt') and os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    print(f"Loaded model with patched method: {model_path}")
                else:
                    self.model = YOLO('yolov8n.pt')
                    print("Loaded default model with patched method")
            except Exception as fallback_error:
                print(f"All loading methods failed: {fallback_error}")
                raise RuntimeError(f"Could not load YOLO model: {fallback_error}")
            finally:
                # Restore original torch.load
                torch.load = original_load
        
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
        Detect waste objects in image with precise bounding boxes
        Args:
            image: OpenCV image (BGR format)
            confidence_threshold: Minimum confidence for detection
        Returns:
            List of detections with precise bbox, label, confidence, category
        """
        try:
            # Run inference with high precision settings
            results = self.model(
                image, 
                conf=confidence_threshold, 
                iou=0.4,  # Lower IoU for better separation of close objects
                verbose=False,
                augment=True,  # Test time augmentation for better accuracy
                max_det=100  # Allow more detections
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get precise bounding box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Ensure coordinates are within image bounds
                        h, w = image.shape[:2]
                        x1 = max(0, min(w-1, float(x1)))
                        y1 = max(0, min(h-1, float(y1)))
                        x2 = max(0, min(w-1, float(x2)))
                        y2 = max(0, min(h-1, float(y2)))
                        
                        # Calculate width and height
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Skip very small detections (likely noise)
                        if width < 10 or height < 10:
                            continue
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        # Determine waste category
                        waste_category = self.waste_categories.get(class_name, 'other')
                        
                        detection = {
                            # Use xyxy format for precise bounding boxes
                            'bbox': [x1, y1, x2, y2],  # [x1, y1, x2, y2] format for precise corners
                            'label': class_name,
                            'confidence': confidence,
                            'category': waste_category,
                            'class_id': class_id,
                            'area': width * height,  # Area for size filtering
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]  # Center point
                        }
                        
                        detections.append(detection)
            
            # Sort by confidence (highest first) and area (larger first)
            detections.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
            
            print(f"✅ Detected {len(detections)} waste objects with precise bounding boxes")
            for det in detections:
                print(f"  - {det['label']} ({det['category']}): {det['confidence']:.3f} confidence")
            
            return detections
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw precise bounding boxes and labels on image
        Args:
            image: OpenCV image (BGR format)
            detections: List of detection results
        Returns:
            Image with drawn detections
        """
        if not detections:
            return image.copy()
        
        result_image = image.copy()
        h, w = image.shape[:2]
        
        # Color mapping for different waste categories
        category_colors = {
            'organic': (0, 255, 0),      # Green
            'recyclable': (0, 165, 255), # Orange  
            'hazardous': (0, 0, 255),    # Red
            'other': (255, 255, 0)       # Cyan
        }
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            label = detection['label']
            confidence = detection['confidence']
            category = detection['category']
            
            # Get color for category
            color = category_colors.get(category, (128, 128, 128))
            
            # Draw bounding box with thickness based on confidence
            thickness = max(1, int(3 * confidence))
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw corner markers for precise boundaries
            corner_length = 15
            corner_thickness = thickness + 1
            
            # Top-left corner
            cv2.line(result_image, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
            cv2.line(result_image, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
            
            # Top-right corner
            cv2.line(result_image, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
            cv2.line(result_image, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
            
            # Bottom-left corner
            cv2.line(result_image, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
            cv2.line(result_image, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
            
            # Bottom-right corner
            cv2.line(result_image, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
            cv2.line(result_image, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
            
            # Create label with category and confidence
            label_text = f"{label} ({category})"
            conf_text = f"{confidence:.2f}"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            
            (label_w, label_h), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale - 0.1, text_thickness - 1)
            
            # Position label above bounding box
            label_y = y1 - 10
            if label_y < label_h + 10:
                label_y = y2 + label_h + 10
            
            # Draw label background
            cv2.rectangle(result_image, 
                         (x1, label_y - label_h - 5), 
                         (x1 + max(label_w, conf_w) + 10, label_y + 5), 
                         color, -1)
            
            # Draw label text
            cv2.putText(result_image, label_text, 
                       (x1 + 5, label_y - 2), font, font_scale, (255, 255, 255), text_thickness)
            
            # Draw confidence below label
            cv2.putText(result_image, conf_text, 
                       (x1 + 5, label_y + conf_h + 8), font, font_scale - 0.1, (255, 255, 255), text_thickness - 1)
            
            # Draw object number
            cv2.circle(result_image, (x2 - 15, y1 + 15), 12, color, -1)
            cv2.putText(result_image, str(i + 1), 
                       (x2 - 20, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw statistics summary
        stats_text = f"Objects: {len(detections)}"
        cv2.putText(result_image, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return result_image
    
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
