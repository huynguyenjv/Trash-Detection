"""
Test detection with precise bounding boxes
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from detector import WasteDetector

def test_detection_with_precise_bbox():
    """Test detection with precise bounding boxes"""
    print("🧪 Testing waste detection with precise bounding boxes...")
    
    try:
        # Initialize detector
        model_path = "../../models/final.pt" if os.path.exists("../../models/final.pt") else None
        detector = WasteDetector(model_path)
        
        # Create a test image with some objects
        # For demo, we'll create a simple test image
        test_image = np.ones((640, 640, 3), dtype=np.uint8) * 50
        
        # Draw some test objects
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)  # White square
        cv2.rectangle(test_image, (300, 150), (450, 250), (0, 255, 0), -1)      # Green rectangle
        cv2.circle(test_image, (500, 400), 50, (0, 0, 255), -1)                 # Red circle
        
        print("🔍 Running detection...")
        detections = detector.detect_waste(test_image, confidence_threshold=0.3)
        
        if detections:
            print(f"✅ Found {len(detections)} objects:")
            for i, det in enumerate(detections):
                bbox = det['bbox']
                print(f"  {i+1}. {det['label']} ({det['category']})")
                print(f"     Confidence: {det['confidence']:.3f}")
                print(f"     BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                print(f"     Area: {det['area']:.1f}px²")
                print(f"     Center: [{det['center'][0]:.1f}, {det['center'][1]:.1f}]")
                print()
            
            # Draw detections
            result_image = detector.draw_detections(test_image, detections)
            
            # Save result
            output_path = "test_detection_result.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"💾 Saved result image: {output_path}")
            
            # Show statistics
            stats = {}
            for det in detections:
                category = det['category']
                stats[category] = stats.get(category, 0) + 1
            
            print("📊 Detection Statistics:")
            for category, count in stats.items():
                print(f"  - {category}: {count} objects")
            
        else:
            print("❌ No objects detected")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_with_precise_bbox()
