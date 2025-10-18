"""
Test Waste Detection on Images
Script để test detection trên ảnh tĩnh
"""

import cv2
import sys
from pathlib import Path
import argparse

# Add backend path
sys.path.append(str(Path(__file__).parent / 'waste-system' / 'backend'))

try:
    from detector import WasteDetector
except ImportError:
    print("❌ Error importing detector. Run: pip install ultralytics opencv-python")
    sys.exit(1)


def test_image_detection(image_path, model_path=None, confidence=0.15, output_path=None):
    """
    Test detection on a single image
    
    Args:
        image_path: Path to input image
        model_path: Path to custom model (None = use default)
        confidence: Confidence threshold
        output_path: Path to save output image (optional)
    """
    print("🚀 Image Detection Test")
    print("=" * 60)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    print(f"📸 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"✅ Image loaded: {w}x{h}")
    
    # Initialize detector
    print(f"\n📦 Loading detector...")
    try:
        if model_path and Path(model_path).exists():
            print(f"   Using custom model: {model_path}")
            detector = WasteDetector(model_path)
        else:
            print(f"   Using default YOLOv8n model")
            detector = WasteDetector()
        print("✅ Detector loaded!")
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return
    
    # Run detection
    print(f"\n🔍 Running detection (confidence >= {confidence})...")
    detections = detector.detect_waste(image, confidence_threshold=confidence)
    
    print(f"\n📊 Results:")
    print(f"   Found {len(detections)} objects")
    
    if len(detections) == 0:
        print("\n⚠️  No objects detected!")
        print("💡 Tips:")
        print("   - Try lowering confidence threshold: --confidence 0.1")
        print("   - Check if image has clear objects")
        print("   - Try with a different image")
        return
    
    # Print detection details
    print("\n📦 Detected Objects:")
    print("-" * 60)
    
    category_counts = {'organic': 0, 'recyclable': 0, 'hazardous': 0, 'other': 0}
    
    for i, det in enumerate(detections):
        label = det['label']
        conf = det['confidence']
        category = det.get('category', 'other')
        bbox = det['bbox']
        
        # Count categories
        if category in category_counts:
            category_counts[category] += 1
        
        # Print info
        print(f"{i+1}. {label} ({category})")
        print(f"   Confidence: {conf:.2%}")
        print(f"   BBox: {bbox}")
        print(f"   Size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]} pixels")
    
    # Print category summary
    print("\n📈 Category Summary:")
    print(f"   🍂 Organic:     {category_counts['organic']}")
    print(f"   ♻️  Recyclable: {category_counts['recyclable']}")
    print(f"   ⚠️  Hazardous:  {category_counts['hazardous']}")
    print(f"   🗑️  Other:      {category_counts['other']}")
    
    # Draw detections
    print("\n🎨 Drawing detections on image...")
    
    colors = {
        'organic': (0, 255, 0),      # Green
        'recyclable': (255, 165, 0),  # Orange  
        'hazardous': (0, 0, 255),     # Red
        'other': (0, 255, 255)        # Yellow
    }
    
    result_image = image.copy()
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['confidence']
        category = det.get('category', 'other')
        
        color = colors.get(category, (128, 128, 128))
        
        # Draw bounding box (THICKER for visibility)
        thickness = 4
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with confidence - SIMPLE VERSION like your example
        label_text = f"{label}: {category} {conf:.2f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Position label above box (or below if not enough space)
        label_y = y1 - 10
        if label_y - text_h - 10 < 0:
            label_y = y2 + text_h + 10
        
        # Draw text background
        bg_x1 = x1
        bg_y1 = label_y - text_h - 5
        bg_x2 = x1 + text_w + 10
        bg_y2 = label_y + 5
        
        cv2.rectangle(result_image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # Draw text
        cv2.putText(result_image, label_text, (bg_x1 + 5, label_y),
                   font, font_scale, (255, 255, 255), font_thickness)
    
    # Add info panel
    panel_height = 100
    panel = result_image.copy()
    cv2.rectangle(panel, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.75, result_image, 0.25, 0, result_image)
    cv2.rectangle(result_image, (0, 0), (w, panel_height), (0, 255, 0), 2)
    
    # Title
    cv2.putText(result_image, "WASTE DETECTION TEST", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    
    # Stats
    stats_text = f"Objects: {len(detections)} | Organic: {category_counts['organic']} | "
    stats_text += f"Recyclable: {category_counts['recyclable']} | Hazardous: {category_counts['hazardous']} | Other: {category_counts['other']}"
    cv2.putText(result_image, stats_text, (20, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save output
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"✅ Output saved: {output_path}")
    else:
        # Auto-generate output name
        input_file = Path(image_path)
        output_file = input_file.parent / f"{input_file.stem}_detected{input_file.suffix}"
        cv2.imwrite(str(output_file), result_image)
        print(f"✅ Output saved: {output_file}")
    
    # Display image
    print("\n🖼️  Displaying result (press any key to close)...")
    cv2.imshow('Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✅ Test complete!")


def main():
    parser = argparse.ArgumentParser(description='Test Waste Detection on Image')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default=None, help='Path to custom model')
    parser.add_argument('--confidence', type=float, default=0.15, 
                       help='Confidence threshold (default: 0.15)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Path to save output image (optional)')
    
    args = parser.parse_args()
    
    # Find model if not specified
    model_path = args.model
    if not model_path or not Path(model_path).exists():
        # Try to find detection models (not classification)
        model_paths = [
            'models/yolov8n.pt',  # Default detection model
            'yolov8n.pt',
            'waste-system/backend/models/yolov8n.pt',
        ]
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        # If no model found, use None to download default
        if model_path and not Path(model_path).exists():
            model_path = None
            print("⚠️  Using default YOLOv8n (will download if needed)")
    
    # Check if model is classification model (final.pt has 10 classes for classification)
    if model_path and 'final.pt' in str(model_path):
        print("⚠️  WARNING: final.pt appears to be a CLASSIFICATION model (10 classes)")
        print("    For DETECTION of multiple objects, use YOLOv8n detection model instead")
        print("    Switching to YOLOv8n...")
        model_path = None
    
    test_image_detection(
        image_path=args.image,
        model_path=model_path,
        confidence=args.confidence,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
