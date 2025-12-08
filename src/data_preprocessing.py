"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing - Tiền xử lý dữ liệu cho Trash Detection

Mô tả:
    Module này xử lý và chuẩn bị dữ liệu cho quá trình huấn luyện:
    - Gộp nhiều dataset từ Kaggle thành dataset thống nhất
    - Chuyển đổi format annotation sang YOLO format
    - Chia dataset thành train/val/test
    - Tạo file cấu hình dataset.yaml cho YOLOv8

Author: Huy Nguyen
Email: huynguyen@example.com
Date: August 2025
Version: 1.0.0
License: MIT
"""

import os
import json
import shutil
import logging
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import yaml
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MultiDatasetConfig:
    """Cấu hình cho việc gộp nhiều dataset"""
    source_datasets_path: Path = Path("source_datasets")
    output_dataset_path: Path = Path("merged_dataset")
    temp_path: Path = Path("temp_merged")
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # Danh sách các dataset từ Kaggle
    kaggle_datasets: List[str] = None
    
    def __post_init__(self):
        if self.kaggle_datasets is None:
            self.kaggle_datasets = [
                "arkadiyhacks/drinking-waste-classification",
                "youssefelebiary/household-trash-recycling-dataset", 
                "vencerlanz09/taco-dataset-yolo-format",
                "spellsharp/garbage-data"
            ]


class MultiDatasetProcessor:
    """Class chính để gộp nhiều dataset"""
    
    def __init__(self, config: MultiDatasetConfig):
        self.config = config
        
        # Danh sách lớp thống nhất (Master Class List)
        self.master_classes = [
            'bottle',
            'can', 
            'cardboard',
            'plastic_bag',
            'glass',
            'paper',
            'metal',
            'organic',
            'plastic',
            'battery',
            'clothes',
            'shoes',
            'trash'
        ]
        
        # Bản đồ ánh xạ lớp từ các dataset gốc sang master classes
        self.class_mapping = {
            # Từ Drinking Waste Classification dataset
            'Aluminium Can': 'can',
            'Glass Bottle': 'bottle',
            'Plastic Bottle (PET)': 'bottle',
            'Plastic Bottle': 'bottle',
            'bottle': 'bottle',
            'can': 'can',
            
            # Từ Household Trash Recycling dataset
            'cardboard': 'cardboard',
            'glass': 'glass',
            'metal': 'metal',
            'paper': 'paper',
            'plastic': 'plastic',
            'trash': 'trash',
            
            # Từ TACO dataset
            'Bottle': 'bottle',
            'Can': 'can',
            'Plastic bag': 'plastic_bag',
            'Plastic_bag': 'plastic_bag',
            'Cardboard': 'cardboard',
            'Glass': 'glass',
            'Metal': 'metal',
            'Paper': 'paper',
            'Plastic': 'plastic',
            'Battery': 'battery',
            'Clothes': 'clothes',
            'Shoes': 'shoes',
            'Organic': 'organic',
            
            # Từ Waste Classification YOLOv8 dataset
            'biological': 'organic',
            'brown-glass': 'glass',
            'green-glass': 'glass',
            'white-glass': 'glass',
            'clothes': 'clothes',
            'shoes': 'shoes',
            'battery': 'battery',
            'trash': 'trash',
            
            # Thêm các ánh xạ khác có thể gặp
            'organic': 'organic',
            'bio': 'organic',
            'food': 'organic',
            'aluminum': 'can',
            'aluminium': 'can',
            'tin': 'can',
            'carton': 'cardboard',
            'box': 'cardboard',
            'newspaper': 'paper',
            'magazine': 'paper',
            'bag': 'plastic_bag',
            'sack': 'plastic_bag',
        }
        
        # Tạo dictionary để map từ tên class sang ID
        self.master_class_to_id = {name: idx for idx, name in enumerate(self.master_classes)}
        
        # Khởi tạo thư mục
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Tạo cấu trúc thư mục"""
        # Xóa và tạo lại thư mục output nếu đã tồn tại
        if self.config.output_dataset_path.exists():
            shutil.rmtree(self.config.output_dataset_path)
        
        if self.config.temp_path.exists():
            shutil.rmtree(self.config.temp_path)
        
        directories = [
            self.config.source_datasets_path,
            self.config.output_dataset_path,
            self.config.output_dataset_path / "images" / "train",
            self.config.output_dataset_path / "images" / "val", 
            self.config.output_dataset_path / "labels" / "train",
            self.config.output_dataset_path / "labels" / "val",
            self.config.temp_path / "images",
            self.config.temp_path / "labels",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Đã tạo cấu trúc thư mục cần thiết")
    
    def download_datasets(self) -> None:
        """Download các dataset từ Kaggle"""
        try:
            import kaggle
            
            logger.info("Bắt đầu download các dataset từ Kaggle...")
            
            # Kiểm tra Kaggle API key
            kaggle_locations = [
                Path.home() / ".kaggle" / "kaggle.json",
                Path.home() / ".config" / "kaggle" / "kaggle.json"
            ]
            
            kaggle_file = None
            for location in kaggle_locations:
                if location.exists():
                    kaggle_file = location
                    break
            
            if not kaggle_file:
                raise FileNotFoundError(
                    "Không tìm thấy Kaggle API key. "
                    "Vui lòng tạo file kaggle.json với API credentials tại:\n"
                    f"  - {kaggle_locations[0]}\n"
                    f"  - {kaggle_locations[1]}\n"
                    "Hoặc sử dụng environment variables."
                )
            
            for dataset_name in self.config.kaggle_datasets:
                dataset_folder = self.config.source_datasets_path / dataset_name.split('/')[-1]
                
                if dataset_folder.exists() and any(dataset_folder.iterdir()):
                    logger.info(f"Dataset {dataset_name} đã tồn tại, bỏ qua download")
                    continue
                
                logger.info(f"Đang download dataset: {dataset_name}")
                
                # Download và giải nén dataset
                kaggle.api.dataset_download_files(
                    dataset_name,
                    path=dataset_folder,
                    unzip=True
                )
                
                logger.info(f"Đã download xong: {dataset_name}")
            
            logger.info("Hoàn thành download tất cả dataset!")
            
        except ImportError:
            logger.error("Vui lòng cài đặt kaggle: pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi download dataset: {e}")
            raise
    
    
    def read_dataset_yaml(self, dataset_path: Path) -> Dict:
        """Đọc file data.yaml từ dataset"""
        yaml_files = list(dataset_path.glob("**/data.yaml")) + list(dataset_path.glob("**/dataset.yaml"))
        
        if not yaml_files:
            logger.warning(f"Không tìm thấy file yaml trong {dataset_path}")
            return {}
        
        try:
            with open(yaml_files[0], 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Lỗi khi đọc {yaml_files[0]}: {e}")
            return {}
    
    def map_class_name(self, original_class: str) -> Optional[str]:
        """Ánh xạ tên class gốc sang master class"""
        # Chuẩn hóa tên class (lowercase, loại bỏ khoảng trắng thừa)
        original_class = original_class.strip().lower().replace(' ', '_').replace('-', '_')
        
        # Tìm trong mapping
        for old_name, new_name in self.class_mapping.items():
            if old_name.lower().replace(' ', '_').replace('-', '_') == original_class:
                return new_name
        
        # Nếu không tìm thấy, kiểm tra xem có trùng trực tiếp với master class không
        if original_class in [cls.lower() for cls in self.master_classes]:
            return original_class
        
        logger.warning(f"Không tìm thấy ánh xạ cho class: {original_class}")
        return None
    
    def process_single_dataset(self, dataset_path: Path) -> int:
        """Xử lý một dataset riêng lẻ và copy vào thư mục temp"""
        dataset_name = dataset_path.name
        logger.info(f"Đang xử lý dataset: {dataset_name}")
        
        processed_count = 0
        
        # Đọc file yaml để lấy class names
        dataset_config = self.read_dataset_yaml(dataset_path)
        old_class_names = dataset_config.get('names', [])
        
        if not old_class_names:
            logger.warning(f"Không tìm thấy class names trong {dataset_name}")
            # Thử tìm từ cấu trúc thư mục
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"
            
            if not (images_dir.exists() and labels_dir.exists()):
                logger.warning(f"Không tìm thấy thư mục images/labels trong {dataset_name}")
                return 0
        
        # Tìm tất cả file ảnh
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))
        
        # Xử lý từng file ảnh
        for img_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
            # Tìm file label tương ứng
            label_path = self.find_corresponding_label(img_path, dataset_path)
            
            if not label_path or not label_path.exists():
                continue
            
            # Đọc và xử lý file label
            new_label_content = self.process_label_file(
                label_path, old_class_names, dataset_name
            )
            
            if new_label_content is None:
                continue
            
            # Tạo tên file mới (thêm prefix dataset để tránh trùng)
            new_filename = f"{dataset_name}_{processed_count:05d}{img_path.suffix}"
            
            try:
                # Copy ảnh vào temp
                new_img_path = self.config.temp_path / "images" / new_filename
                shutil.copy2(img_path, new_img_path)
                
                # Lưu label mới
                new_label_path = self.config.temp_path / "labels" / f"{new_filename.rsplit('.', 1)[0]}.txt"
                with open(new_label_path, 'w') as f:
                    f.write(new_label_content)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Lỗi khi copy {img_path}: {e}")
        
        logger.info(f"Đã xử lý {processed_count} ảnh từ dataset {dataset_name}")
        return processed_count
    
    def find_corresponding_label(self, img_path: Path, dataset_root: Path) -> Optional[Path]:
        """Tìm file label tương ứng với file ảnh"""
        # Lấy tên file không có extension
        img_stem = img_path.stem
        
        # Tìm trong các thư mục labels có thể
        possible_label_dirs = [
            dataset_root / "labels",
            dataset_root / "annotations", 
            img_path.parent.parent / "labels",
            img_path.parent / "labels",
        ]
        
        for label_dir in possible_label_dirs:
            if label_dir.exists():
                label_path = label_dir / f"{img_stem}.txt"
                if label_path.exists():
                    return label_path
        
        return None
    
    def process_label_file(self, label_path: Path, old_class_names: List[str], dataset_name: str) -> Optional[str]:
        """Xử lý file label và chuyển đổi class IDs"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                old_class_id = int(parts[0])
                coordinates = parts[1:]
                
                # Lấy tên class cũ
                if old_class_id >= len(old_class_names):
                    logger.warning(f"Class ID {old_class_id} vượt quá danh sách class trong {dataset_name}")
                    continue
                
                old_class_name = old_class_names[old_class_id]
                
                # Ánh xạ sang master class
                new_class_name = self.map_class_name(old_class_name)
                
                if new_class_name is None:
                    continue
                
                # Lấy ID mới
                new_class_id = self.master_class_to_id[new_class_name]
                
                # Tạo dòng mới
                new_line = f"{new_class_id} " + " ".join(coordinates) + "\n"
                new_lines.append(new_line)
            
            return "".join(new_lines)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý label file {label_path}: {e}")
            return None
    
    def split_merged_data(self) -> None:
        """Phân chia dữ liệu đã gộp thành train/val"""
        logger.info("Đang phân chia dữ liệu thành train/val...")
        
        # Lấy tất cả file ảnh từ temp
        temp_images_dir = self.config.temp_path / "images"
        all_image_files = list(temp_images_dir.glob("*"))
        
        # Xáo trộn ngẫu nhiên
        random.seed(42)
        random.shuffle(all_image_files)
        
        # Tính số lượng cho train/val
        total_files = len(all_image_files)
        train_count = int(total_files * self.config.train_ratio)
        
        train_files = all_image_files[:train_count]
        val_files = all_image_files[train_count:]
        
        logger.info(f"Phân chia: {len(train_files)} train, {len(val_files)} val")
        
        # Di chuyển file vào thư mục train/val
        self._move_files_to_split(train_files, "train")
        self._move_files_to_split(val_files, "val")
        
        # Xóa thư mục temp
        shutil.rmtree(self.config.temp_path)
        logger.info("Đã xóa thư mục tạm thời")
    
    def _move_files_to_split(self, files: List[Path], split_name: str) -> None:
        """Di chuyển file vào thư mục train hoặc val"""
        for img_file in tqdm(files, desc=f"Moving to {split_name}"):
            # Di chuyển ảnh
            dest_img = self.config.output_dataset_path / "images" / split_name / img_file.name
            shutil.move(str(img_file), str(dest_img))
            
            # Di chuyển label
            label_name = f"{img_file.stem}.txt"
            src_label = self.config.temp_path / "labels" / label_name
            dest_label = self.config.output_dataset_path / "labels" / split_name / label_name
            
            if src_label.exists():
                shutil.move(str(src_label), str(dest_label))
    
    def create_final_dataset_yaml(self) -> None:
        """Tạo file dataset.yaml cuối cùng"""
        dataset_config = {
            'path': str(self.config.output_dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'nc': len(self.master_classes),
            'names': self.master_classes
        }
        
        yaml_path = self.config.output_dataset_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Đã tạo data.yaml tại: {yaml_path}")
    
    def create_summary_report(self) -> None:
        """Tạo báo cáo tóm tắt dataset gộp"""
        logger.info("Tạo báo cáo tóm tắt...")
        
        summary = {
            'master_classes': self.master_classes,
            'total_classes': len(self.master_classes),
            'class_mapping': self.class_mapping,
            'splits': {}
        }
        
        # Đếm số file trong mỗi split
        for split_name in ['train', 'val']:
            split_dir = self.config.output_dataset_path / "images" / split_name
            image_files = list(split_dir.glob("*"))
            summary['splits'][split_name] = {
                'total_images': len(image_files)
            }
        
        # Lưu báo cáo
        report_path = self.config.output_dataset_path / "dataset_summary.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # In báo cáo
        logger.info("=== BÁO CÁO DATASET GỘP ===")
        logger.info(f"Tổng số master classes: {summary['total_classes']}")
        logger.info(f"Master classes: {', '.join(summary['master_classes'])}")
        
        total_images = 0
        for split_name, split_info in summary['splits'].items():
            count = split_info['total_images']
            total_images += count
            logger.info(f"{split_name}: {count} ảnh")
        
        logger.info(f"Tổng cộng: {total_images} ảnh")
    
    def process_all_datasets(self) -> None:
        """Xử lý tất cả các dataset"""
        logger.info("=== BẮT ĐẦU GỘP CÁC DATASET ===")
        
        # Kiểm tra thư mục source datasets
        if not self.config.source_datasets_path.exists():
            logger.error(f"Không tìm thấy thư mục {self.config.source_datasets_path}")
            logger.info("Vui lòng download các dataset trước hoặc chạy download_datasets()")
            return
        
        total_processed = 0
        
        # Xử lý từng dataset con
        for dataset_dir in self.config.source_datasets_path.iterdir():
            if dataset_dir.is_dir():
                count = self.process_single_dataset(dataset_dir)
                total_processed += count
        
        logger.info(f"Đã xử lý tổng cộng {total_processed} ảnh từ tất cả dataset")
        
        if total_processed == 0:
            logger.error("Không có ảnh nào được xử lý. Vui lòng kiểm tra cấu trúc dataset.")
            return
        
        # Phân chia train/val
        self.split_merged_data()
        
        # Tạo dataset.yaml
        self.create_final_dataset_yaml()
        
        # Tạo báo cáo
        self.create_summary_report()
        
        logger.info("=== HOÀN THÀNH GỘP DATASET ===")


def main():
    """Hàm main"""
    try:
        # Khởi tạo config
        config = MultiDatasetConfig()
        
        # Khởi tạo processor
        processor = MultiDatasetProcessor(config)
        
        # Tùy chọn: Download datasets từ Kaggle (nếu cần)
        download_choice = input("Bạn có muốn download dataset từ Kaggle? (y/n): ").lower()
        if download_choice == 'y':
            processor.download_datasets()
        
        # Gộp tất cả dataset
        processor.process_all_datasets()
        
        logger.info("Chương trình hoàn thành thành công!")
        
    except Exception as e:
        logger.error(f"Lỗi chương trình: {e}")
        raise


if __name__ == "__main__":
    main()
