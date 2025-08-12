#!/usr/bin/env python3
"""
Trash Detection - Quick Run Guide
Hướng dẫn chạy nhanh dự án Phát hiện rác
"""

def print_usage():
    print("🚀 TRASH DETECTION PROJECT - HƯỚNG DẪN SỬ DỤNG")
    print("=" * 60)
    print()
    
    print("📁 1. CẤU TRÚC DỰ ÁN:")
    print("   src/               # Source code")
    print("   ├── train.py       # Huấn luyện model")
    print("   ├── detect.py      # Phát hiện real-time")
    print("   ├── evaluate.py    # Đánh giá model")
    print("   └── data_preprocessing.py  # Tiền xử lý data")
    print("   notebooks/         # Jupyter tutorials")
    print("   data/             # Dataset")
    print("   models/           # Trained models")
    print()
    
    print("🔧 2. SETUP ENVIRONMENT:")
    print("   cd /home/huynguyen/source/Trash-Detection")
    print("   source trash_detection_env/bin/activate")
    print()
    
    print("⚡ 3. CHẠY NHANH:")
    print()
    print("   a) Test detection với pre-trained model:")
    print("      python test_detection.py")
    print()
    print("   b) Chạy full pipeline (cần dataset):")
    print("      python run_pipeline.py")
    print()
    print("   c) Chỉ training:")
    print("      python src/train.py")
    print()
    print("   d) Chỉ detection:")
    print("      python src/detect.py --model yolov8n.pt")
    print()
    print("   e) Mở Jupyter notebook tutorial:")
    print("      jupyter notebook notebooks/trash_detection_tutorial.ipynb")
    print()
    
    print("📊 4. DATASET SETUP:")
    print("   - Tự động: Setup Kaggle API (xem setup_kaggle.py)")
    print("   - Thủ công: Tải từ https://www.kaggle.com/datasets/mostafaabla/garbage-classification-v2")
    print("   - Test: Dùng pre-trained YOLOv8n (không cần dataset)")
    print()
    
    print("🎯 5. SỬ DỤNG TÍNH NĂNG:")
    print("   ✅ Object Detection với YOLOv8")
    print("   ✅ Real-time detection qua webcam")
    print("   ✅ Batch processing ảnh/video")
    print("   ✅ Model evaluation & metrics")
    print("   ✅ Transfer learning")
    print("   ✅ Data preprocessing automation")
    print()
    
    print("🚨 6. TROUBLESHOOTING:")
    print("   - Module not found: Kiểm tra virtual environment")
    print("   - Camera không mở: Cài driver webcam")
    print("   - CUDA error: Kiểm tra GPU driver")
    print("   - Dataset error: Setup Kaggle API hoặc tải manual")
    print()
    
    print("📞 7. READY TO RUN COMMANDS:")
    print("-" * 40)
    print("# Activate environment")
    print("source trash_detection_env/bin/activate")
    print()
    print("# Quick test")
    print("python test_detection.py")
    print()
    print("# Full pipeline")
    print("python run_pipeline.py")
    print("-" * 40)

if __name__ == "__main__":
    print_usage()
