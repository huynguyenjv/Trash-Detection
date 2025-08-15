"""
Detection Engine - Module xử lý YOLO detection
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GPSCoordinate, WasteBin
from core.enums import WasteType, BinStatus

logger = logging.getLogger(__name__)


class DetectionEngine:
    """Engine xử lý YOLO detection cho rác thải"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize detection engine
        
        Args:
            model_path: Đường dẫn đến YOLO model
        """
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            logger.info(f"✅ Loaded YOLO model: {model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load YOLO model: {e}")
            self.model = None
            self.model_loaded = False
    
    def detect_waste(self, image: np.ndarray) -> List[Dict]:
        """
        Detect waste objects in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detection results
        """
        if not self.model_loaded:
            return self._simulate_detection(image)
        
        try:
            results = self.model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                            'confidence': float(box.conf[0]),
                            'class_id': int(box.cls[0]),
                            'class_name': self.model.names[int(box.cls[0])],
                            'waste_type': self._map_class_to_waste_type(int(box.cls[0]))
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """Simulate detection when model is not available"""
        h, w = image.shape[:2]
        
        # Generate random detections for demo
        np.random.seed(42)
        num_detections = np.random.randint(0, 4)
        
        detections = []
        waste_types = list(WasteType)
        
        for i in range(num_detections):
            x1 = np.random.randint(0, w//2)
            y1 = np.random.randint(0, h//2)
            x2 = x1 + np.random.randint(50, 200)
            y2 = y1 + np.random.randint(50, 200)
            
            detection = {
                'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                'confidence': np.random.uniform(0.6, 0.95),
                'class_id': i,
                'class_name': f"waste_{i}",
                'waste_type': np.random.choice(waste_types)
            }
            detections.append(detection)
        
        return detections
    
    def _map_class_to_waste_type(self, class_id: int) -> WasteType:
        """Map YOLO class ID to WasteType"""
        # Mapping based on common YOLO classes
        class_mapping = {
            39: WasteType.PLASTIC,    # bottle
            40: WasteType.GLASS,      # wine glass
            41: WasteType.PLASTIC,    # cup
            42: WasteType.METAL,      # knife/fork
            43: WasteType.CARDBOARD,  # bowl
            44: WasteType.ORGANIC,    # banana
            45: WasteType.ORGANIC,    # apple
            46: WasteType.ORGANIC,    # sandwich
            47: WasteType.ORGANIC,    # orange
        }
        
        return class_mapping.get(class_id, WasteType.GENERAL)
    
    def analyze_bin_status(self, image: np.ndarray, bin_location: GPSCoordinate) -> Dict:
        """
        Analyze waste bin status from image
        
        Args:
            image: Image containing waste bin
            bin_location: GPS location of the bin
            
        Returns:
            Analysis result with status and fill level
        """
        detections = self.detect_waste(image)
        
        # Simple analysis based on number of detected items
        fill_level = min(len(detections) * 25, 100)  # Each detection = 25%
        
        if fill_level >= 90:
            status = BinStatus.FULL
        elif fill_level >= 70:
            status = BinStatus.NEAR_FULL
        else:
            status = BinStatus.OK
        
        waste_types = [d['waste_type'] for d in detections]
        
        return {
            'status': status,
            'fill_level': fill_level,
            'detected_waste_types': waste_types,
            'detection_count': len(detections),
            'confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0,
            'location': bin_location
        }
    
    def process_realtime_stream(self, video_source: int = 0) -> None:
        """
        Process real-time video stream for waste detection
        
        Args:
            video_source: Video source (0 for webcam)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error("Cannot open video source")
            return
        
        logger.info("🎥 Starting real-time waste detection...")
        logger.info("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Cannot read frame")
                    break
                
                # Detect waste every 10 frames for performance
                if frame_count % 10 == 0:
                    detections = self.detect_waste(frame)
                    frame = self._draw_detections(frame, detections)
                
                # Display frame
                cv2.imshow('Smart Waste Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'waste_detection_{frame_count}.jpg', frame)
                    logger.info(f"Saved frame: waste_detection_{frame_count}.jpg")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = self._get_waste_color(detection['waste_type'])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection['waste_type'].value}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for label
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add info text
        info_text = f"Detected items: {len(detections)}"
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result_image
    
    def _get_waste_color(self, waste_type: WasteType) -> Tuple[int, int, int]:
        """Get color for waste type (BGR format)"""
        colors = {
            WasteType.ORGANIC: (0, 255, 0),     # Green
            WasteType.PLASTIC: (255, 0, 0),     # Blue
            WasteType.GLASS: (0, 255, 255),     # Yellow
            WasteType.METAL: (128, 128, 128),   # Gray
            WasteType.PAPER: (255, 255, 255),   # White
            WasteType.CARDBOARD: (139, 69, 19), # Brown
            WasteType.BATTERY: (0, 0, 255),     # Red
            WasteType.CLOTHES: (255, 0, 255),   # Magenta
            WasteType.SHOES: (0, 128, 255),     # Orange
            WasteType.GENERAL: (128, 128, 128)  # Gray
        }
        return colors.get(waste_type, (128, 128, 128))
