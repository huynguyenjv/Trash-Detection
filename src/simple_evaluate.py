#!/usr/bin/env python3
"""
Simple Evaluate Script - Debug version
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def test_simple_evaluation():
    """Test evaluation đơn giản"""
    print("🧪 SIMPLE EVALUATION TEST")
    print("=" * 50)
    
    # 1. Load model
    model_path = "../models/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model không tồn tại: {model_path}")
        return
    
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # 2. Get class names
    class_names = list(model.names.values())
    print(f"🏷️ Classes: {class_names}")
    
    # 3. Test với một ảnh
    test_images_dir = Path("../data/processed/images/test")
    test_labels_dir = Path("../data/processed/labels/test")
    
    if not test_images_dir.exists():
        print(f"❌ Test images không tồn tại: {test_images_dir}")
        return
    
    if not test_labels_dir.exists():
        print(f"❌ Test labels không tồn tại: {test_labels_dir}")
        return
    
    # Lấy ảnh đầu tiên
    image_files = list(test_images_dir.glob("*.jpg"))
    if not image_files:
        print("❌ Không có ảnh test")
        return
    
    test_image = image_files[0]
    print(f"🖼️ Test image: {test_image.name}")
    
    # 4. Load ảnh
    image = cv2.imread(str(test_image))
    if image is None:
        print("❌ Không thể load ảnh")
        return
    
    print(f"📏 Image shape: {image.shape}")
    
    # 5. Predict
    print("🔍 Predicting...")
    try:
        results = model(image, conf=0.25, device="cpu", verbose=False)
        print("✅ Prediction thành công!")
        
        # Kiểm tra results
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            print(f"📦 Detected {len(boxes)} objects")
            
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                
                print(f"   Object {i+1}: {class_name} ({conf:.2%})")
        else:
            print("🔍 No objects detected")
    
    except Exception as e:
        print(f"❌ Prediction lỗi: {e}")
        return
    
    # 6. Kiểm tra ground truth
    label_file = test_labels_dir / f"{test_image.stem}.txt"
    print(f"🏷️ Label file: {label_file}")
    
    if label_file.exists():
        print("✅ Label file tồn tại")
        
        try:
            with open(label_file, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    gt_class = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    print(f"🎯 Ground Truth: {gt_class} (class_id: {class_id})")
                else:
                    print("⚠️ Label file trống")
        except Exception as e:
            print(f"❌ Đọc label lỗi: {e}")
    else:
        print("❌ Label file không tồn tại")

    print("\n✅ Test hoàn thành!")

if __name__ == "__main__":
    test_simple_evaluation()
