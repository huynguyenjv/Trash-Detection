"""
Script test cho data preprocessing
Kiểm tra các chức năng của MultiDatasetProcessor

Author: Huy Nguyen
Date: August 2025
"""

import sys
import tempfile
import shutil
from pathlib import Path
import yaml
import json

# Thêm src vào Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import MultiDatasetConfig, MultiDatasetProcessor


def create_mock_dataset(base_path: Path, dataset_name: str, classes: list) -> Path:
    """Tạo mock dataset để test"""
    dataset_path = base_path / dataset_name
    
    # Tạo cấu trúc thư mục
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels" 
    
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    
    # Tạo data.yaml
    data_config = {
        'nc': len(classes),
        'names': classes
    }
    
    with open(dataset_path / "data.yaml", 'w') as f:
        yaml.dump(data_config, f)
    
    # Tạo một số file ảnh và label giả
    for i in range(5):
        for j, class_name in enumerate(classes):
            # Tạo file ảnh giả (text file)
            img_file = images_dir / f"{class_name}_{i:03d}.jpg"
            with open(img_file, 'w') as f:
                f.write("fake image data")
            
            # Tạo file label
            label_file = labels_dir / f"{class_name}_{i:03d}.txt"
            # Format YOLO: class_id x_center y_center width height
            with open(label_file, 'w') as f:
                f.write(f"{j} 0.5 0.5 0.8 0.8\n")
    
    print(f"Đã tạo mock dataset: {dataset_name} với {len(classes)} classes")
    return dataset_path


def test_class_mapping():
    """Test chức năng ánh xạ class"""
    print("=== TEST CLASS MAPPING ===")
    
    config = MultiDatasetConfig()
    processor = MultiDatasetProcessor(config)
    
    # Test một số ánh xạ
    test_cases = [
        ("Aluminium Can", "can"),
        ("Glass Bottle", "bottle"),
        ("plastic_bag", "plastic_bag"),
        ("cardboard", "cardboard"),
        ("biological", "organic"),
        ("unknown_class", None),
    ]
    
    for original, expected in test_cases:
        result = processor.map_class_name(original)
        status = "✅" if result == expected else "❌"
        print(f"{status} {original} -> {result} (expected: {expected})")
    
    print()


def test_mock_processing():
    """Test xử lý với mock datasets"""
    print("=== TEST MOCK PROCESSING ===")
    
    # Tạo thư mục temp
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Cấu hình test
        config = MultiDatasetConfig()
        config.source_datasets_path = temp_path / "source"
        config.output_dataset_path = temp_path / "output"
        config.temp_path = temp_path / "temp"
        
        # Tạo mock datasets
        mock_datasets = [
            ("drinking-waste", ["Aluminium Can", "Glass Bottle", "Plastic Bottle"]),
            ("household-trash", ["cardboard", "glass", "metal", "paper"]),
            ("taco-dataset", ["Bottle", "Can", "Plastic bag"])
        ]
        
        for name, classes in mock_datasets:
            create_mock_dataset(config.source_datasets_path, name, classes)
        
        # Khởi tạo processor
        processor = MultiDatasetProcessor(config)
        
        try:
            # Test xử lý
            processor.process_all_datasets()
            
            # Kiểm tra kết quả
            output_path = config.output_dataset_path
            
            checks = [
                (output_path / "data.yaml", "data.yaml"),
                (output_path / "dataset_summary.json", "summary"),
                (output_path / "images" / "train", "train images"),
                (output_path / "images" / "val", "val images"), 
                (output_path / "labels" / "train", "train labels"),
                (output_path / "labels" / "val", "val labels"),
            ]
            
            print("Kiểm tra các file/thư mục đầu ra:")
            for path, description in checks:
                exists = path.exists()
                status = "✅" if exists else "❌"
                print(f"{status} {description}: {path}")
            
            # Kiểm tra nội dung data.yaml
            if (output_path / "data.yaml").exists():
                with open(output_path / "data.yaml") as f:
                    data_config = yaml.safe_load(f)
                print(f"✅ data.yaml có {data_config['nc']} classes")
                print(f"   Classes: {data_config['names'][:5]}...")  # Chỉ hiển thị 5 đầu
            
            # Kiểm tra summary
            if (output_path / "dataset_summary.json").exists():
                with open(output_path / "dataset_summary.json") as f:
                    summary = json.load(f)
                print(f"✅ Summary: {summary['total_classes']} master classes")
                for split, info in summary['splits'].items():
                    print(f"   {split}: {info['total_images']} images")
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình test: {e}")
            
    print()


def test_config_validation():
    """Test validation config"""
    print("=== TEST CONFIG VALIDATION ===")
    
    # Test config mặc định
    config = MultiDatasetConfig()
    
    checks = [
        (config.source_datasets_path, "source_datasets_path"),
        (config.output_dataset_path, "output_dataset_path"),
        (config.train_ratio, "train_ratio"),
        (config.val_ratio, "val_ratio"),
        (config.kaggle_datasets, "kaggle_datasets"),
    ]
    
    print("Config mặc định:")
    for value, name in checks:
        print(f"  {name}: {value}")
    
    # Test validation tỷ lệ
    total_ratio = config.train_ratio + config.val_ratio
    status = "✅" if abs(total_ratio - 1.0) < 0.01 else "❌"
    print(f"{status} Tổng tỷ lệ train + val = {total_ratio}")
    
    # Test danh sách kaggle datasets
    status = "✅" if len(config.kaggle_datasets) > 0 else "❌"
    print(f"{status} Có {len(config.kaggle_datasets)} kaggle datasets")
    
    print()


def main():
    """Chạy tất cả tests"""
    print("CHẠY TESTS CHO DATA PREPROCESSING")
    print("=" * 50)
    print()
    
    try:
        # Chạy các test
        test_config_validation()
        test_class_mapping()
        test_mock_processing()
        
        print("=" * 50)
        print("✅ TẤT CẢ TESTS HOÀN THÀNH")
        print()
        print("Nếu tất cả tests pass, script đã sẵn sàng sử dụng!")
        print("Để chạy thực tế:")
        print("1. Cấu hình Kaggle API")
        print("2. python data_preprocessing.py")
        
    except Exception as e:
        print(f"❌ TESTS THẤT BẠI: {e}")
        raise


if __name__ == "__main__":
    main()
