"""
Script demo cho việc gộp nhiều dataset trash detection
Hướng dẫn sử dụng data_preprocessing.py mới

Author: Huy Nguyen
Date: August 2025
"""

import logging
from pathlib import Path
from data_preprocessing import MultiDatasetConfig, MultiDatasetProcessor

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_manual_setup():
    """Demo thiết lập thủ công các dataset"""
    
    print("=== DEMO THIẾT LẬP THỦ CÔNG ===")
    print()
    
    # Cấu hình tùy chỉnh
    config = MultiDatasetConfig()
    config.source_datasets_path = Path("my_source_datasets") 
    config.output_dataset_path = Path("my_merged_dataset")
    config.train_ratio = 0.85  # 85% train, 15% val
    config.val_ratio = 0.15
    
    print(f"Thư mục nguồn: {config.source_datasets_path}")
    print(f"Thư mục đầu ra: {config.output_dataset_path}")
    print(f"Tỷ lệ train/val: {config.train_ratio}/{config.val_ratio}")
    print()
    
    # Khởi tạo processor
    processor = MultiDatasetProcessor(config)
    
    print("Master classes sẽ được sử dụng:")
    for i, cls in enumerate(processor.master_classes):
        print(f"  {i}: {cls}")
    print()
    
    print("Một số ví dụ ánh xạ class:")
    examples = [
        "Aluminium Can -> can",
        "Glass Bottle -> bottle", 
        "Plastic bag -> plastic_bag",
        "cardboard -> cardboard",
        "biological -> organic"
    ]
    for example in examples:
        print(f"  {example}")
    print()
    
    # Chỉ tạo cấu trúc thư mục, không chạy xử lý
    print("Đã tạo cấu trúc thư mục cần thiết.")
    print("Để chạy thực tế, hãy gọi: processor.process_all_datasets()")


def demo_full_process():
    """Demo quy trình đầy đủ"""
    
    print("=== DEMO QUY TRÌNH ĐẦY ĐỦ ===")
    print()
    
    # Sử dụng config mặc định
    config = MultiDatasetConfig()
    processor = MultiDatasetProcessor(config)
    
    print("Các bước sẽ được thực hiện:")
    print("1. Tạo cấu trúc thư mục")
    print("2. [Tùy chọn] Download datasets từ Kaggle")
    print("3. Duyệt qua từng dataset trong source_datasets/")
    print("4. Đọc file data.yaml của mỗi dataset")
    print("5. Ánh xạ class names sang master classes")
    print("6. Copy ảnh và chuyển đổi labels")
    print("7. Phân chia ngẫu nhiên thành train/val")
    print("8. Tạo data.yaml cuối cùng")
    print("9. Tạo báo cáo tóm tắt")
    print()
    
    # Hiển thị danh sách dataset sẽ download
    print("Datasets từ Kaggle sẽ được download:")
    for i, dataset in enumerate(config.kaggle_datasets, 1):
        print(f"  {i}. {dataset}")
    print()
    
    print("Để chạy đầy đủ:")
    print("python data_preprocessing.py")
    print()
    print("Hoặc trong code Python:")
    print("processor.download_datasets()  # Nếu cần")
    print("processor.process_all_datasets()")


def demo_custom_mapping():
    """Demo tùy chỉnh ánh xạ class"""
    
    print("=== DEMO TÙY CHỈNH ÁNH XẠ CLASS ===")
    print()
    
    config = MultiDatasetConfig()
    processor = MultiDatasetProcessor(config)
    
    # Thêm ánh xạ tùy chỉnh
    custom_mappings = {
        'food_waste': 'organic',
        'newspaper': 'paper',
        'aluminum_can': 'can',
        'plastic_container': 'plastic',
        'cardboard_box': 'cardboard',
    }
    
    processor.class_mapping.update(custom_mappings)
    
    print("Đã thêm các ánh xạ tùy chỉnh:")
    for old_name, new_name in custom_mappings.items():
        print(f"  {old_name} -> {new_name}")
    print()
    
    # Tùy chỉnh master classes
    additional_classes = ['electronic', 'textile']
    processor.master_classes.extend(additional_classes)
    
    # Cập nhật mapping
    processor.master_class_to_id = {
        name: idx for idx, name in enumerate(processor.master_classes)
    }
    
    print("Đã thêm master classes:")
    for cls in additional_classes:
        print(f"  {cls}")
    print()
    
    print(f"Tổng cộng {len(processor.master_classes)} master classes")


def print_structure_example():
    """In ví dụ cấu trúc thư mục"""
    
    print("=== CẤU TRÚC THƯ MỤC MẪU ===")
    print()
    
    print("Cấu trúc đầu vào (source_datasets/):")
    print("source_datasets/")
    print("├── drinking-waste-classification/")
    print("│   ├── images/")
    print("│   │   ├── train/")
    print("│   │   └── val/") 
    print("│   ├── labels/")
    print("│   │   ├── train/")
    print("│   │   └── val/")
    print("│   └── data.yaml")
    print("├── household-trash-recycling/")
    print("│   ├── images/")
    print("│   ├── labels/")
    print("│   └── data.yaml")
    print("└── ...")
    print()
    
    print("Cấu trúc đầu ra (merged_dataset/):")
    print("merged_dataset/")
    print("├── images/")
    print("│   ├── train/")
    print("│   │   ├── drinking-waste_00001.jpg")
    print("│   │   ├── household-trash_00001.jpg")
    print("│   │   └── ...")
    print("│   └── val/")
    print("│       ├── drinking-waste_00002.jpg")
    print("│       └── ...")
    print("├── labels/")
    print("│   ├── train/")
    print("│   │   ├── drinking-waste_00001.txt") 
    print("│   │   ├── household-trash_00001.txt")
    print("│   │   └── ...")
    print("│   └── val/")
    print("│       └── ...")
    print("├── data.yaml")
    print("└── dataset_summary.json")


def main():
    """Hàm main demo"""
    
    print("SCRIPT DEMO - GỘP NHIỀU DATASET TRASH DETECTION")
    print("=" * 50)
    print()
    
    while True:
        print("Chọn demo:")
        print("1. Thiết lập thủ công")
        print("2. Quy trình đầy đủ") 
        print("3. Tùy chỉnh ánh xạ class")
        print("4. Cấu trúc thư mục mẫu")
        print("5. Thoát")
        print()
        
        choice = input("Nhập lựa chọn (1-5): ").strip()
        print()
        
        if choice == '1':
            demo_manual_setup()
        elif choice == '2':
            demo_full_process()
        elif choice == '3':
            demo_custom_mapping()
        elif choice == '4':
            print_structure_example()
        elif choice == '5':
            break
        else:
            print("Lựa chọn không hợp lệ!")
        
        print()
        input("Nhấn Enter để tiếp tục...")
        print()


if __name__ == "__main__":
    main()
