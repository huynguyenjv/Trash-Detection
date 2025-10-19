"""
Realtime Waste Detection & Classification Test Script
Test detection và classification trực tiếp từ webcam không cần web interface
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent / 'waste-system' / 'backend'))

try:
    from detector import WasteDetector
except ImportError:
    print("❌ Không thể import WasteDetector. Đảm bảo bạn đã cài đặt dependencies.")
    sys.exit(1)


class RealtimeWasteDetectionTest:
    """Class để test realtime detection"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize detector
        Args:
            model_path: Path to custom model (None = use default YOLOv8n)
            confidence_threshold: Minimum confidence for detection
        """
        print("🚀 Initializing Realtime Waste Detection Test...")
        
        # Load detector
        try:
            if model_path and Path(model_path).exists():
                print(f"📦 Loading custom model: {model_path}")
                self.detector = WasteDetector(model_path)
            else:
                print("📦 Loading default YOLOv8n model...")
                self.detector = WasteDetector()
        except Exception as e:
            print(f"❌ Error loading detector: {e}")
            raise
        
        self.confidence_threshold = confidence_threshold
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'organic': 0,
            'recyclable': 0,
            'hazardous': 0,
            'other': 0,
            'fps_history': [],
            'detection_time_history': []
        }
        
        # Category colors (BGR format for OpenCV)
        self.category_colors = {
            'organic': (0, 255, 0),      # Green
            'recyclable': (255, 165, 0),  # Blue  
            'hazardous': (0, 0, 255),     # Red
            'other': (0, 255, 255)        # Yellow
        }
        
        print("✅ Detector initialized successfully!")
    
    def draw_detections_on_frame(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        Args:
            frame: OpenCV frame (BGR)
            detections: List of detection results
        Returns:
            Annotated frame
        """
        result_frame = frame.copy()
        h, w = frame.shape[:2]
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            category = detection.get('category', 'other')
            
            x1, y1, x2, y2 = bbox
            
            # Get color for category
            color = self.category_colors.get(category, (128, 128, 128))
            
            # Draw bounding box
            thickness = 2 if confidence > 0.7 else 1
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw corner markers
            corner_length = 15
            corner_thickness = 3
            
            # Top-left
            cv2.line(result_frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
            cv2.line(result_frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
            
            # Top-right
            cv2.line(result_frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
            cv2.line(result_frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
            
            # Bottom-left
            cv2.line(result_frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
            cv2.line(result_frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
            
            # Bottom-right
            cv2.line(result_frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
            cv2.line(result_frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
            
            # Draw center point
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(result_frame, (center_x, center_y), 4, color, -1)
            
            # Prepare label text
            label_text = f"{label} ({category})"
            conf_text = f"{confidence:.2%}"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale * 0.8, font_thickness)
            
            # Position label
            label_y = y1 - 10
            if label_y < text_h + conf_h + 20:
                label_y = y2 + text_h + conf_h + 20
            
            # Draw label background
            bg_x1 = x1
            bg_y1 = label_y - text_h - conf_h - 10
            bg_x2 = x1 + max(text_w, conf_w) + 10
            bg_y2 = label_y + 5
            
            # Semi-transparent background
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
            
            # Draw border
            cv2.rectangle(result_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
            
            # Draw text with outline
            text_color = (255, 255, 255)
            outline_color = (0, 0, 0)
            
            # Label text
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(result_frame, label_text, (bg_x1 + 5 + dx, label_y - conf_h - 5 + dy),
                           font, font_scale, outline_color, font_thickness + 1)
            cv2.putText(result_frame, label_text, (bg_x1 + 5, label_y - conf_h - 5),
                       font, font_scale, text_color, font_thickness)
            
            # Confidence text
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(result_frame, conf_text, (bg_x1 + 5 + dx, label_y + dy),
                           font, font_scale * 0.8, outline_color, font_thickness)
            cv2.putText(result_frame, conf_text, (bg_x1 + 5, label_y),
                       font, font_scale * 0.8, text_color, font_thickness)
            
            # Draw object number
            number_radius = 15
            number_x = min(x2 - number_radius - 5, w - number_radius - 5)
            number_y = max(y1 + number_radius + 5, number_radius + 5)
            
            cv2.circle(result_frame, (number_x, number_y), number_radius, color, -1)
            cv2.circle(result_frame, (number_x, number_y), number_radius, (0, 0, 0), 2)
            
            number_text = str(i + 1)
            (num_w, num_h), _ = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(result_frame, number_text,
                       (number_x - num_w // 2, number_y + num_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def draw_info_panel(self, frame, fps, detection_time, detections):
        """
        Draw information panel on frame
        Args:
            frame: OpenCV frame
            fps: Current FPS
            detection_time: Detection time in seconds
            detections: List of detections
        """
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        panel_height = 180
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (0, 0), (w, panel_height), (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "REALTIME WASTE DETECTION TEST", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        # Stats
        y_offset = 70
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        
        # FPS and Detection Time
        cv2.putText(frame, f"FPS: {fps:.1f} | Detection: {detection_time*1000:.1f}ms",
                   (20, y_offset), font, font_scale, (0, 255, 0), 2)
        y_offset += 30
        
        # Resolution and Objects
        cv2.putText(frame, f"Resolution: {w}x{h} | Objects: {len(detections)}",
                   (20, y_offset), font, font_scale, (255, 255, 255), 2)
        y_offset += 30
        
        # Category counts
        if detections:
            category_counts = {'organic': 0, 'recyclable': 0, 'hazardous': 0, 'other': 0}
            for det in detections:
                cat = det.get('category', 'other')
                if cat in category_counts:
                    category_counts[cat] += 1
            
            stats_text = f"🍂 Organic: {category_counts['organic']} | ♻️ Recyclable: {category_counts['recyclable']} | "
            stats_text += f"⚠️ Hazardous: {category_counts['hazardous']} | 🗑️ Other: {category_counts['other']}"
            
            cv2.putText(frame, stats_text, (20, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 30
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save screenshot | 'r' to reset stats",
                   (20, y_offset), font, 0.5, (200, 200, 200), 1)
    
    def update_stats(self, detections):
        """Update statistics"""
        self.stats['total_detections'] += len(detections)
        
        for det in detections:
            category = det.get('category', 'other')
            if category in self.stats:
                self.stats[category] += 1
    
    def print_session_summary(self):
        """Print session summary"""
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"Total Frames Processed: {self.stats['total_frames']}")
        print(f"Total Objects Detected: {self.stats['total_detections']}")
        print(f"\n📦 Detection Breakdown:")
        print(f"  🍂 Organic:     {self.stats['organic']}")
        print(f"  ♻️  Recyclable: {self.stats['recyclable']}")
        print(f"  ⚠️  Hazardous:  {self.stats['hazardous']}")
        print(f"  🗑️  Other:      {self.stats['other']}")
        
        if self.stats['fps_history']:
            avg_fps = np.mean(self.stats['fps_history'])
            print(f"\n⚡ Average FPS: {avg_fps:.2f}")
        
        if self.stats['detection_time_history']:
            avg_det_time = np.mean(self.stats['detection_time_history']) * 1000
            print(f"⏱️  Average Detection Time: {avg_det_time:.2f}ms")
        
        print("="*60)
    
    def run_webcam_test(self, camera_id=0):
        """
        Run realtime detection from webcam
        Args:
            camera_id: Camera device ID (default 0)
        """
        print(f"\n🎥 Starting webcam test (Camera ID: {camera_id})")
        print("="*60)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'r' to reset statistics")
        print("  - Press '+' to increase confidence threshold")
        print("  - Press '-' to decrease confidence threshold")
        print("="*60)
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cap = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Camera: {width}x{height} @ {fps_cap} FPS")
        print(f"🎯 Confidence Threshold: {self.confidence_threshold}")
        print("\n🚀 Starting detection... (Press 'q' to quit)\n")
        
        # FPS calculation
        prev_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                self.stats['total_frames'] += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                self.stats['fps_history'].append(fps)
                
                # Run detection
                detection_start = time.time()
                detections = self.detector.detect_waste(frame, confidence_threshold=0.15)  # LOW threshold for testing
                detection_time = time.time() - detection_start
                self.stats['detection_time_history'].append(detection_time)
                
                # Update stats
                self.update_stats(detections)
                
                # Draw detections
                annotated_frame = self.draw_detections_on_frame(frame, detections)
                
                # Draw info panel
                self.draw_info_panel(annotated_frame, fps, detection_time, detections)
                
                # Show frame
                cv2.imshow('Realtime Waste Detection Test', annotated_frame)
                
                # Print detection info periodically
                if frame_count % 30 == 0 and detections:
                    print(f"🎯 Frame {frame_count}: Detected {len(detections)} objects")
                    for det in detections[:3]:  # Show first 3
                        print(f"  - {det['label']} ({det['category']}): {det['confidence']:.2%}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n⏹️  Stopping test...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):
                    # Reset stats
                    self.stats = {
                        'total_frames': 0,
                        'total_detections': 0,
                        'organic': 0,
                        'recyclable': 0,
                        'hazardous': 0,
                        'other': 0,
                        'fps_history': [],
                        'detection_time_history': []
                    }
                    print("🔄 Statistics reset")
                elif key == ord('+') or key == ord('='):
                    # Increase confidence threshold
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"🎯 Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease confidence threshold
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"🎯 Confidence threshold: {self.confidence_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print summary
            self.print_session_summary()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Realtime Waste Detection Test')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--model', type=str, default=None, help='Path to custom model')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Model paths to try
    model_paths = [
        args.model,
        'models/final.pt',
        'waste-system/backend/models/final.pt',
        './final.pt'
    ]
    
    model_path = None
    if args.model:
        if Path(args.model).exists():
            model_path = args.model
    else:
        for path in model_paths:
            if path and Path(path).exists():
                model_path = path
                break
    
    try:
        # Initialize tester
        tester = RealtimeWasteDetectionTest(
            model_path=model_path,
            confidence_threshold=args.confidence
        )
        
        # Run webcam test
        tester.run_webcam_test(camera_id=args.camera)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
