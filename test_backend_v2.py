"""
Quick Test Script for Backend V2
Test detection với YOLOv8n default model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'waste-system' / 'backend-v2'))

from detector import WasteDetector
import cv2


def test_detection(image_path):
    """Test detection on image"""
    
    print("=" * 60)
    print("🚀 WASTE DETECTION TEST - Backend V2")
    print("=" * 60)
    
    # Check image
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    print(f"\n📸 Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image")
        return
    
    h, w = image.shape[:2]
    print(f"✅ Image loaded: {w}x{h}")
    
    # Initialize detector
    print(f"\n🤖 Initializing YOLOv8n detector...")
    detector = WasteDetector('yolov8n.pt')
    
    # Detect
    print(f"\n🔍 Running detection...")
    detections = detector.detect(image, conf_threshold=0.25, iou_threshold=0.45)
    
    print(f"\n📊 RESULTS:")
    print(f"   Found {len(detections)} objects")
    
    if len(detections) == 0:
        print("\n⚠️  No objects detected!")
        return
    
    # Group by category
    by_category = {'organic': [], 'recyclable': [], 'hazardous': [], 'other': []}
    for det in detections:
        category = det['category']
        by_category[category].append(det)
    
    print(f"\n📦 Detections by Category:")
    print(f"   🍂 Organic:     {len(by_category['organic'])}")
    print(f"   ♻️  Recyclable: {len(by_category['recyclable'])}")
    print(f"   ⚠️  Hazardous:  {len(by_category['hazardous'])}")
    print(f"   🗑️  Other:      {len(by_category['other'])}")
    
    print(f"\n📋 Detailed Detections:")
    print("-" * 60)
    
    for i, det in enumerate(detections):
        label = det['label']
        conf = det['confidence']
        category = det['category']
        bbox = det['bbox']
        
        print(f"{i+1}. {label.upper()} ({category})")
        print(f"   Confidence: {conf:.2%}")
        print(f"   BBox: {bbox}")
        print(f"   Size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]} pixels")
    
    # Draw on image
    print(f"\n🎨 Drawing detections...")
    
    colors = {
        'organic': (0, 255, 0),      # Green
        'recyclable': (255, 165, 0),  # Orange
        'hazardous': (0, 0, 255),     # Red
        'other': (0, 255, 255)        # Yellow
    }
    
    result = image.copy()
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['confidence']
        category = det['category']
        
        color = colors.get(category, (128, 128, 128))
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label_text = f"{label}: {category} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background for text
        cv2.rectangle(result, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
        
        # Text
        cv2.putText(result, label_text, (x1+5, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save result
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_detected_v2.jpg"
    cv2.imwrite(str(output_path), result)
    print(f"✅ Result saved: {output_path}")
    
    # Display
    print(f"\n🖼️  Displaying result (press any key to close)...")
    cv2.imshow('Detection Result - Backend V2', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\n✅ TEST COMPLETE!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_backend_v2.py <image_path>")
        print("\nExample:")
        print("  python test_backend_v2.py image.jpg")
    else:
        test_detection(sys.argv[1])
