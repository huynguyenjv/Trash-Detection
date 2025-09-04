#!/usr/bin/env python3
"""
Quick Start Script cho Multi-Dataset Processing
Thiết lập nhanh môi trường và chạy gộp dataset

Author: Huy Nguyen  
Date: August 2025
"""

import os
import sys
import json
from pathlib import Path
import subprocess

def check_requirements():
    """Kiểm tra các requirements cần thiết"""
    print("🔍 Kiểm tra requirements...")
    
    # Kiểm tra các package cơ bản trước
    basic_packages = ['yaml', 'tqdm']
    missing_packages = []
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    # Kiểm tra kaggle riêng biệt (không authenticate)
    try:
        import importlib.util
        spec = importlib.util.find_spec("kaggle")
        if spec is not None:
            print("✅ kaggle (package)")
        else:
            missing_packages.append('kaggle')
            print("❌ kaggle")
    except Exception:
        missing_packages.append('kaggle')
        print("❌ kaggle")
    
    if missing_packages:
        print(f"\n⚠️  Thiếu các package: {', '.join(missing_packages)}")
        print("Cài đặt bằng: pip install " + " ".join(missing_packages))
        return False
    
    return True


def setup_kaggle_api():
    """Thiết lập Kaggle API"""
    print("\n🔑 Thiết lập Kaggle API...")
    
    # Windows: thử cả hai vị trí
    kaggle_locations = [
        Path.home() / ".kaggle" / "kaggle.json",    # Linux/Mac style
        Path.home() / ".config" / "kaggle" / "kaggle.json"  # Standard location
    ]
    
    kaggle_file = None
    for location in kaggle_locations:
        if location.exists():
            kaggle_file = location
            break
    
    if kaggle_file:
        print(f"✅ Kaggle API key đã tồn tại tại: {kaggle_file}")
        return True
    
    print("❌ Chưa có Kaggle API key")
    print("\nHướng dẫn thiết lập:")
    print("1. Đăng nhập vào https://kaggle.com")
    print("2. Vào Account → API → Create New API Token") 
    print("3. Download file kaggle.json")
    print("4. Đặt file vào một trong các vị trí:")
    for location in kaggle_locations:
        print(f"   - {location}")
    
    setup_now = input("\nBạn muốn thiết lập ngay không? (y/n): ").lower()
    
    if setup_now == 'y':
        # Chọn vị trí đặt file (dùng vị trí đầu tiên)
        kaggle_dir = kaggle_locations[0].parent
        kaggle_file = kaggle_locations[0]
        
        # Tạo thư mục
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        # Nhập thông tin
        username = input("Kaggle username: ").strip()
        api_key = input("Kaggle API key: ").strip()
        
        if username and api_key:
            # Tạo file
            kaggle_config = {"username": username, "key": api_key}
            with open(kaggle_file, 'w') as f:
                json.dump(kaggle_config, f)
            
            # Set permissions (Unix only)
            if os.name != 'nt':  # Không phải Windows
                os.chmod(kaggle_file, 0o600)
            
            print(f"✅ Đã thiết lập Kaggle API tại: {kaggle_file}")
            return True
    
    return False


def create_directory_structure():
    """Tạo cấu trúc thư mục cần thiết"""
    print("\n📁 Tạo cấu trúc thư mục...")
    
    directories = [
        "source_datasets",
        "merged_dataset", 
        "logs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ {dir_name}/")
    
    return True


def show_dataset_info():
    """Hiển thị thông tin về datasets sẽ download"""
    print("\n📊 Datasets sẽ được download:")
    
    datasets = [
        ("arkadiyhacks/drinking-waste-classification", "~50MB"),
        ("youssefelebiary/household-trash-recycling-dataset", "~200MB"),
        ("vencerlanz09/taco-dataset-yolo-format", "~500MB"), 
        ("spellsharp/garbage-data", "~100MB")
    ]
    
    total_size = "~850MB"
    
    for i, (dataset, size) in enumerate(datasets, 1):
        print(f"  {i}. {dataset} ({size})")
    
    print(f"\nTổng dung lượng ước tính: {total_size}")
    print("Thời gian download: 5-15 phút tùy vào tốc độ mạng")


def run_preprocessing():
    """Chạy quá trình tiền xử lý"""
    print("\n🚀 Bắt đầu quá trình gộp dataset...")
    
    try:
        # Import và chạy
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from data_preprocessing import MultiDatasetConfig, MultiDatasetProcessor
        
        config = MultiDatasetConfig()
        processor = MultiDatasetProcessor(config)
        
        # Hỏi có download không
        download = input("Download datasets từ Kaggle? (y/n): ").lower()
        if download == 'y':
            try:
                print("⬇️  Đang download datasets...")
                processor.download_datasets()
            except Exception as e:
                print(f"❌ Lỗi download: {e}")
                print("💡 Gợi ý: Bạn có thể bỏ qua download và dùng datasets có sẵn")
                continue_choice = input("Tiếp tục mà không download? (y/n): ").lower()
                if continue_choice != 'y':
                    return False
        
        # Gộp datasets
        print("🔧 Đang gộp datasets...")
        processor.process_all_datasets()
        
        print("\n✅ HOÀN THÀNH!")
        print(f"📁 Dataset đã gộp tại: {config.output_dataset_path}")
        print(f"📋 Báo cáo tại: {config.output_dataset_path}/dataset_summary.json")
        
        return True
        
    except ImportError:
        print("❌ Không thể import data_preprocessing module")
        print("Hãy đảm bảo file data_preprocessing.py ở trong thư mục src/")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def show_next_steps():
    """Hiển thị các bước tiếp theo"""
    print("\n🎯 Bước tiếp theo:")
    print("1. Kiểm tra dataset đã gộp trong thư mục merged_dataset/")
    print("2. Xem báo cáo trong dataset_summary.json")
    print("3. Sử dụng data.yaml để train YOLOv8:")
    print("   ```python")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('yolov8n.pt')")
    print("   model.train(data='merged_dataset/data.yaml', epochs=100)")
    print("   ```")


def main():
    """Hàm main"""
    print("🗂️  QUICK START - MULTI-DATASET PROCESSING")
    print("=" * 50)
    
    # Kiểm tra requirements
    if not check_requirements():
        print("\n❌ Vui lòng cài đặt missing packages trước")
        return
    
    # Thiết lập Kaggle API
    if not setup_kaggle_api():
        print("\n⚠️  Có thể bỏ qua nếu datasets đã có sẵn")
    
    # Tạo thư mục
    create_directory_structure()
    
    # Hiển thị thông tin datasets
    show_dataset_info()
    
    # Chạy preprocessing
    proceed = input("\n🚀 Bạn có muốn tiếp tục? (y/n): ").lower()
    if proceed == 'y':
        if run_preprocessing():
            show_next_steps()
        else:
            print("\n❌ Quá trình thất bại. Vui lòng kiểm tra logs.")
    else:
        print("\n✋ Dừng lại. Bạn có thể chạy lại script này bất cứ lúc nào.")
    
    print("\n📚 Xem thêm hướng dẫn chi tiết trong DATASET_MERGING_GUIDE.md")


if __name__ == "__main__":
    main()
