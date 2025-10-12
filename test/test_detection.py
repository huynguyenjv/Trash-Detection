#!/usr/bin/env python3
"""
Simple test script for trash detection
"""

from ultralytics import YOLO
import cv2
import numpy as np

def test_detection():
    """Test detection with pre-trained YOLOv8 model"""
    print("🚀 Khởi tạo YOLO model...")
    
    # Load pre-trained model
    model = YOLO('D:/MasterUIT/Trash-Detection/models/final.pt')
    print("✅ Model loaded successfully!")
    
    # Test with sample image (create a simple test image)
    print("📸 Tạo ảnh test...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Run detection
    print("🔍 Chạy detection...")
    results = model(test_image)
    
    print("✅ Detection completed!")
    print(f"📊 Number of detected objects: {len(results[0].boxes) if results[0].boxes is not None else 0}")
    
    # If you want to test with webcam
    print("\n🎥 Để test với webcam, bấm 'y' (cần camera):")
    choice = input().lower()
    
    if choice == 'y':
        print("📹 Mở webcam... (Bấm 'q' để thoát)")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Không thể mở camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection on frame
            results = model(frame, verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Display
            cv2.imshow('Trash Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test hoàn thành!")

if __name__ == "__main__":
    test_detection()
