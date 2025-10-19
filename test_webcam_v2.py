"""
Webcam Test Script for Backend V2
Realtime waste detection with YOLOv8n
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'waste-system' / 'backend-v2'))

from detector import WasteDetector
from waste_manager import WasteManager
import cv2
import time


class WebcamTester:
    def __init__(self):
        """Initialize webcam tester"""
        print("=" * 60)
        print("🎥 WEBCAM DETECTION TEST - Backend V2")
        print("=" * 60)
        
        # Initialize detector
        print("\n🤖 Initializing YOLOv8n detector...")
        self.detector = WasteDetector('yolov8n.pt')
        
        # Initialize waste manager
        self.waste_manager = WasteManager()
        
        # Colors for categories
        self.colors = {
            'organic': (0, 255, 0),      # Green
            'recyclable': (255, 165, 0),  # Orange
            'hazardous': (0, 0, 255),     # Red
            'other': (0, 255, 255)        # Yellow
        }
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            category = det['category']
            
            color = self.colors.get(category, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label_text = f"{label}: {category} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Background
            cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            
            # Text
            cv2.putText(result, label_text, (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return result
    
    def draw_info_panel(self, frame, detections):
        """Draw info panel with statistics"""
        h, w = frame.shape[:2]
        
        # Semi-transparent black panel at top
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (0, 0), (w, panel_height), (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "WASTE DETECTION - BACKEND V2", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Objects count
        cv2.putText(frame, f"Objects: {len(detections)}", (300, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Category counts
        stats = self.waste_manager.get_stats()
        totals = stats['totals']
        
        y_pos = 90
        stats_text = f"Organic: {totals['organic']} | Recyclable: {totals['recyclable']} | Hazardous: {totals['hazardous']} | Other: {totals['other']}"
        cv2.putText(frame, stats_text, (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run webcam detection
        
        Args:
            camera_id: Camera ID (default: 0)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
        """
        print(f"\n📹 Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"✅ Camera opened successfully!")
        print(f"\n🎮 Controls:")
        print(f"   SPACE - Take screenshot")
        print(f"   R     - Reset statistics")
        print(f"   Q/ESC - Quit")
        print(f"\n🚀 Starting detection...\n")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Update FPS
            current_time = time.time()
            self.fps = 1.0 / (current_time - self.last_time) if self.frame_count > 0 else 0
            self.last_time = current_time
            self.frame_count += 1
            
            # Detect
            detections = self.detector.detect(frame, conf_threshold, iou_threshold)
            
            # Update stats
            if detections:
                self.waste_manager.update(detections)
            
            # Draw detections
            result = self.draw_detections(frame, detections)
            
            # Draw info panel
            result = self.draw_info_panel(result, detections)
            
            # Show frame
            cv2.imshow('Waste Detection - Backend V2', result)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\n👋 Exiting...")
                break
            
            elif key == ord(' '):  # SPACE
                # Take screenshot
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, result)
                print(f"📸 Screenshot saved: {filename}")
            
            elif key == ord('r'):  # R
                # Reset stats
                self.waste_manager.reset()
                self.frame_count = 0
                print("🔄 Statistics reset!")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "=" * 60)
        print("📊 FINAL STATISTICS")
        print("=" * 60)
        
        stats = self.waste_manager.get_stats()
        totals = stats['totals']
        
        print(f"\n🗑️  Total Detections:")
        print(f"   🍂 Organic:     {totals['organic']}")
        print(f"   ♻️  Recyclable: {totals['recyclable']}")
        print(f"   ⚠️  Hazardous:  {totals['hazardous']}")
        print(f"   🗑️  Other:      {totals['other']}")
        print(f"\n📦 Total Objects: {sum(totals.values())}")
        print(f"📹 Total Frames: {self.frame_count}")
        print(f"\n✅ Test complete!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Webcam Detection Test - Backend V2')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold (default: 0.45)')
    
    args = parser.parse_args()
    
    try:
        tester = WebcamTester()
        tester.run(
            camera_id=args.camera,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
