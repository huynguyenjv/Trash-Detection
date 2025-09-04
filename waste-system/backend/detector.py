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
        
        # Waste classification mapping for better accuracy
        self.waste_categories = {
            # Common COCO classes that can be waste
            'bottle': 'recyclable',
            'cup': 'recyclable', 
            'fork': 'recyclable',
            'knife': 'recyclable',
            'spoon': 'recyclable',
            'bowl': 'recyclable',
            'banana': 'organic',
            'apple': 'organic',
            'orange': 'organic',
            'carrot': 'organic',
            'hot dog': 'organic',
            'pizza': 'organic',
            'donut': 'organic',
            'cake': 'organic',
            'cell phone': 'hazardous',
            'laptop': 'hazardous',
            'mouse': 'hazardous',
            'keyboard': 'hazardous',
            'book': 'recyclable',
            'scissors': 'hazardous',
            'teddy bear': 'other',
            'hair drier': 'hazardous',
            'toothbrush': 'other',
            'person': 'ignore',  # Ignore people
            'car': 'ignore',
            'truck': 'ignore',
            'bus': 'ignore'
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
    
    def detect_waste(self, image: np.ndarray, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Detect waste objects in image with precise bounding boxes
        Args:
            image: OpenCV image (BGR format)
            confidence_threshold: Minimum confidence for detection
        Returns:
            List of detections with precise bbox, label, confidence, category
        """
        try:
            print(f"🔍 Running detection with confidence_threshold={confidence_threshold}")
            print(f"📐 Image shape: {image.shape}")
            
            # Run inference with enhanced settings for better accuracy
            results = self.model(
                image, 
                conf=confidence_threshold,  # Lower default threshold
                iou=0.45,  # Improved IoU threshold for better object separation
                verbose=False,
                augment=True,  # Test time augmentation for better accuracy
                max_det=300,  # Allow more detections
                imgsz=640,  # Fixed input size for consistency
                half=False  # Use FP32 for better accuracy
            )
            
            print(f"📊 Raw detection results: {len(results)} result(s)")
            
            detections = []
            all_detections = []  # Track all detections before filtering
            
            for result in results:
                boxes = result.boxes
                print(f"📦 Boxes found: {boxes is not None}")
                
                if boxes is not None:
                    print(f"📊 Number of raw boxes: {len(boxes)}")
                    
                    for i, box in enumerate(boxes):
                        # Get precise bounding box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        print(f"   🔍 Box {i}: {class_name} (conf: {confidence:.3f}, coords: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}])")
                        
                        # Track all detections for debugging
                        all_detections.append({
                            'label': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'class_id': class_id
                        })
                        
                        # Skip ignored classes (people, vehicles) - but allow others
                        if self.waste_categories.get(class_name) == 'ignore':
                            print(f"   ⏭️  Skipping ignored class: {class_name}")
                            continue
                        
                        # Ensure coordinates are within image bounds and valid
                        h, w = image.shape[:2]
                        x1 = max(0, min(w-1, float(x1)))
                        y1 = max(0, min(h-1, float(y1)))
                        x2 = max(0, min(w-1, float(x2)))
                        y2 = max(0, min(h-1, float(y2)))
                        
                        # Ensure x1 < x2 and y1 < y2
                        if x1 >= x2 or y1 >= y2:
                            print(f"   ❌ Invalid bbox coordinates: ({x1}, {y1}, {x2}, {y2})")
                            continue
                        
                        # Calculate width and height
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Very lenient size filtering - for webcam detection
                        min_size = 10  # Very small minimum size  
                        # Don't limit max size for webcam - objects can fill entire frame
                        # max_size = min(w, h) * 0.99   # Allow almost full frame objects
                        
                        if width < min_size or height < min_size:
                            print(f"   ❌ Object too small: {width}x{height} < {min_size}")
                            continue
                            
                        # Remove max size check for webcam detection
                        # if width > max_size or height > max_size:
                        #     print(f"   ❌ Object too large: {width}x{height} > {max_size}")
                        #     continue
                        
                        # Determine waste category - be more inclusive
                        waste_category = self.waste_categories.get(class_name, 'other')
                        
                        # Calculate box area and aspect ratio for filtering
                        box_area = width * height
                        aspect_ratio = width / height if height > 0 else 1
                        
                        # More lenient aspect ratio filtering
                        if aspect_ratio > 10 or aspect_ratio < 0.1:
                            print(f"   ❌ Unrealistic aspect ratio: {aspect_ratio:.2f}")
                            continue
                        
                        detection = {
                            # Use xyxy format for precise bounding boxes
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Integer coordinates
                            'label': class_name,
                            'confidence': confidence,
                            'category': waste_category,
                            'class_id': class_id,
                            'area': box_area,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'aspect_ratio': aspect_ratio
                        }
                        
                        detections.append(detection)
                        print(f"   ✅ Added detection: {class_name} ({waste_category})")
            
            print(f"📊 Detection summary:")
            print(f"   Raw detections: {len(all_detections)}")
            print(f"   Filtered detections: {len(detections)}")
            
            # Apply Non-Maximum Suppression to remove overlapping detections
            if len(detections) > 1:
                detections = self._apply_advanced_nms(detections, iou_threshold=0.4)
                print(f"   After NMS: {len(detections)}")
            
            # Sort by confidence (highest first) and area (larger first)
            detections.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
            
            if len(detections) > 0:
                print(f"✅ Detected {len(detections)} waste objects with precise bounding boxes")
                for det in detections:
                    print(f"  - {det['label']} ({det['category']}): {det['confidence']:.3f} confidence, area: {det['area']:.0f}")
            else:
                print(f"⚠️  No objects detected. Raw detections found: {len(all_detections)}")
                if len(all_detections) > 0:
                    print("   Raw detections (before filtering):")
                    for det in all_detections[:5]:  # Show first 5
                        print(f"     - {det['label']}: {det['confidence']:.3f}")
            
            return detections
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                   confidence_threshold: float = 0.5, nms_threshold: float = 0.4) -> np.ndarray:
        """
        Draw precise bounding boxes and labels on image with improved accuracy
        Args:
            image: OpenCV image (BGR format)
            detections: List of detection results
            confidence_threshold: Minimum confidence to display detection
            nms_threshold: Non-maximum suppression threshold
        Returns:
            Image with drawn detections
        """
        if not detections:
            return image.copy()
        
        result_image = image.copy()
        h, w = image.shape[:2]
        
        # Filter detections by confidence
        filtered_detections = []
        for detection in detections:
            if detection['confidence'] >= confidence_threshold:
                filtered_detections.append(detection)
        
        if not filtered_detections:
            return result_image
        
        # Apply Non-Maximum Suppression to remove overlapping boxes
        filtered_detections = self._apply_advanced_nms(filtered_detections, nms_threshold)
        
        # Color mapping for different waste categories
        category_colors = {
            'organic': (0, 255, 0),      # Green
            'recyclable': (0, 165, 255), # Orange  
            'hazardous': (0, 0, 255),    # Red
            'other': (255, 255, 0)       # Cyan
        }
        
        for i, detection in enumerate(filtered_detections):
            # Validate and normalize bbox coordinates
            bbox = self._validate_bbox(detection['bbox'], w, h)
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = bbox
            
            label = detection['label']
            confidence = detection['confidence']
            category = detection.get('category', 'other')
            
            # Get color for category
            color = category_colors.get(category, (128, 128, 128))
            
            # Calculate adaptive thickness based on box size and confidence
            box_area = (x2 - x1) * (y2 - y1)
            img_area = w * h
            size_factor = min(3.0, max(1.0, (box_area / img_area) * 50))
            thickness = max(1, int(2 * confidence * size_factor))
            
            # Draw main bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw precise corner markers
            corner_length = max(8, min(25, int((x2-x1 + y2-y1) / 20)))
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
            
            # Add center point for precise positioning
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(result_image, (center_x, center_y), 3, color, -1)
            
            # Create adaptive label positioning
            label_text = f"{label} ({category})"
            conf_text = f"{confidence:.2f}"
            
            # Calculate adaptive font size
            font = cv2.FONT_HERSHEY_SIMPLEX
            box_width = x2 - x1
            font_scale = max(0.4, min(0.8, box_width / 200))
            text_thickness = max(1, int(font_scale * 2))
            
            (label_w, label_h), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale * 0.8, text_thickness)
            
            # Smart label positioning - avoid overlap
            label_bg_height = label_h + conf_h + 15
            
            # Try to place above first
            if y1 - label_bg_height - 5 >= 0:
                label_y = y1 - 10
                bg_y1 = label_y - label_h - 5
                bg_y2 = label_y + conf_h + 10
            else:
                # Place below if no space above
                label_y = y2 + label_h + 10
                bg_y1 = y2 + 5
                bg_y2 = bg_y1 + label_bg_height
            
            # Ensure label stays within image bounds
            label_x = max(0, min(x1, w - max(label_w, conf_w) - 10))
            bg_x2 = min(w, label_x + max(label_w, conf_w) + 10)
            
            # Draw label background with transparency effect
            overlay = result_image.copy()
            cv2.rectangle(overlay, (label_x, bg_y1), (bg_x2, bg_y2), color, -1)
            result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
            
            # Draw border around label
            cv2.rectangle(result_image, (label_x, bg_y1), (bg_x2, bg_y2), color, 2)
            
            # Draw label text with outline for better readability
            text_color = (255, 255, 255)
            outline_color = (0, 0, 0)
            
            # Text outline
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                cv2.putText(result_image, label_text, 
                        (label_x + 5 + dx, label_y - 2 + dy), 
                        font, font_scale, outline_color, text_thickness + 1)
                cv2.putText(result_image, conf_text, 
                        (label_x + 5 + dx, label_y + conf_h + 8 + dy), 
                        font, font_scale * 0.8, outline_color, text_thickness)
            
            # Main text
            cv2.putText(result_image, label_text, 
                    (label_x + 5, label_y - 2), font, font_scale, text_color, text_thickness)
            cv2.putText(result_image, conf_text, 
                    (label_x + 5, label_y + conf_h + 8), font, font_scale * 0.8, text_color, text_thickness)
            
            # Draw object number with better visibility
            number_radius = max(12, int(corner_length * 0.8))
            number_x = min(x2 - number_radius - 5, w - number_radius - 5)
            number_y = max(y1 + number_radius + 5, number_radius + 5)
            
            # Number background
            cv2.circle(result_image, (number_x, number_y), number_radius, color, -1)
            cv2.circle(result_image, (number_x, number_y), number_radius, (0, 0, 0), 2)
            
            # Number text
            number_text = str(i + 1)
            (num_w, num_h), _ = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(result_image, number_text, 
                    (number_x - num_w//2, number_y + num_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw enhanced statistics
        stats_bg_height = 40
        stats_text = f"Objects: {len(filtered_detections)} | Conf ≥ {confidence_threshold}"
        
        # Stats background
        cv2.rectangle(result_image, (5, 5), (400, stats_bg_height), (0, 0, 0), -1)
        cv2.rectangle(result_image, (5, 5), (400, stats_bg_height), (0, 255, 0), 2)
        
        cv2.putText(result_image, stats_text, (15, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_image

    def _validate_bbox(self, bbox: List[float], img_width: int, img_height: int) -> Optional[List[int]]:
        """
        Validate and normalize bounding box coordinates
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Handle normalized coordinates (0-1 range)
            if all(0 <= coord <= 1 for coord in bbox):
                x1 = int(x1 * img_width)
                y1 = int(y1 * img_height)
                x2 = int(x2 * img_width)
                y2 = int(y2 * img_height)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure x1 < x2 and y1 < y2
            if x1 >= x2 or y1 >= y2:
                return None
            
            # Clamp coordinates to image boundaries
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(x1 + 1, min(x2, img_width))
            y2 = max(y1 + 1, min(y2, img_height))
            
            # Filter out boxes that are too small
            box_area = (x2 - x1) * (y2 - y1)
            img_area = img_width * img_height
            
            # Only filter out extremely small boxes - allow large boxes for webcam detection
            if box_area < 0.0001 * img_area:
                return None
                
            return [x1, y1, x2, y2]
        except (ValueError, TypeError, IndexError) as e:
            print(f"Error validating bbox: {e}")
            return None

    def _apply_advanced_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Apply improved Non-Maximum Suppression to remove overlapping boxes
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Keep the detection with highest confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Remove detections with high overlap
            remaining = []
            for det in detections:
                # Calculate IoU
                iou = self._calculate_iou(current['bbox'], det['bbox'])
                
                # Keep if IoU is below threshold OR different classes with low overlap
                if (iou < iou_threshold or 
                    (current['label'] != det['label'] and iou < iou_threshold * 1.5)):
                    remaining.append(det)
            detections = remaining
        
        return keep

    def _calculate_iou(self, box1: List, box2: List) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
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
    
    def test_model_basic(self) -> bool:
        """Test if model can detect basic objects"""
        try:
            # Create a simple test image
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_img[200:400, 200:400] = [255, 255, 255]  # White square
            
            # Try detection with very low threshold
            results = self.model(test_img, conf=0.01, verbose=False)
            
            total_detections = 0
            if results[0].boxes is not None:
                total_detections = len(results[0].boxes)
            
            print(f"🧪 Model test: Found {total_detections} objects in test image")
            
            # Test with random image (more realistic)
            random_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results2 = self.model(random_img, conf=0.01, verbose=False)
            
            total_detections2 = 0
            if results2[0].boxes is not None:
                total_detections2 = len(results2[0].boxes)
                
            print(f"🧪 Model test: Found {total_detections2} objects in random image")
            
            # Print available classes
            print(f"📝 Model can detect {len(self.model.names)} classes:")
            for i, name in self.model.names.items():
                print(f"   {i}: {name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False

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
