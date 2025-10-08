#!/usr/bin/env python3
"""
Data Preprocessing cho Classification Model - TrashNet Dataset
Xử lý dataset TrashNet/Garbage Classification và tạo cấu trúc folder cho YOLOv8 classification

Author: Huy Nguyen
Date: September 2025
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import yaml
import requests
import zipfile
from tqdm import tqdm
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import json

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing_classification.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    """Cấu hình cho data preprocessing classification"""
    # Dataset paths
    raw_data_dir: Path = Path("data/classification/raw")
    processed_data_dir: Path = Path("data/classification/processed")
    
    # TrashNet dataset info
    trashnet_url: str = "https://github.com/garythung/trashnet/archive/master.zip"
    kaggle_dataset: str = "asdasdasasdas/garbage-classification"  # Backup dataset
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Image processing
    img_size: Tuple[int, int] = (224, 224)  # Standard for classification
    min_images_per_class: int = 50
    
    # Class mapping - unify different naming conventions
    class_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        """Khởi tạo class mapping sau khi tạo object"""
        if self.class_mapping is None:
            self.class_mapping = {
                # TrashNet original classes
                "cardboard": "cardboard",
                "glass": "glass", 
                "metal": "metal",
                "paper": "paper",
                "plastic": "plastic",
                "trash": "other",
                
                # Additional mappings for other datasets
                "battery": "electronics",
                "biological": "organic",
                "clothes": "textiles",
                "shoes": "textiles",
                
                # Normalize variations
                "aluminium": "metal",
                "aluminum": "metal",
                "tin": "metal",
                "steel": "metal",
                "iron": "metal",
                
                "bottles": "glass",
                "jar": "glass",
                "window": "glass",
                
                "newspaper": "paper",
                "magazine": "paper",
                "book": "paper",
                "cardboard box": "cardboard",
                "carton": "cardboard",
                
                "bag": "plastic",
                "bottle": "plastic",
                "container": "plastic",
                "cup": "plastic",
                "packaging": "plastic",
                
                # Organic waste
                "food": "organic",
                "fruit": "organic",
                "vegetable": "organic",
                "leaf": "organic",
                "wood": "organic",
            }


class GarbageDataProcessor:
    """Processor for Garbage dataset for trash classification."""
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.classification_dir = processed_data_dir / "classification"
        
        # Garbage dataset paths - using existing local dataset
        self.garbage_dir = raw_data_dir / "garbage-dataset"
        
    def _create_directories(self) -> None:
        """Tạo các thư mục cần thiết"""
        directories = [
            self.config.raw_data_dir,
            self.config.processed_data_dir,
            self.config.processed_data_dir / "train",
            self.config.processed_data_dir / "val",
            self.config.processed_data_dir / "test",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info("Đã tạo cấu trúc thư mục classification")
    
    def download_trashnet_dataset(self) -> None:
        """Tải xuống TrashNet dataset từ GitHub"""
        zip_path = self.config.raw_data_dir / "trashnet-master.zip"
        
        if zip_path.exists():
            logger.info("TrashNet dataset đã tồn tại, bỏ qua việc tải xuống")
            return
            
        logger.info("Đang tải xuống TrashNet dataset từ GitHub...")
        
        try:
            response = requests.get(self.config.trashnet_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading TrashNet",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
                    
            logger.info(f"Đã tải xuống: {zip_path}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải xuống TrashNet dataset: {e}")
            logger.info("Hãy tải dataset thủ công từ: https://github.com/garythung/trashnet")
            raise
    
    def extract_trashnet_dataset(self) -> None:
        """Giải nén TrashNet dataset"""
        zip_path = self.config.raw_data_dir / "trashnet-master.zip"
        extract_path = self.config.raw_data_dir / "trashnet-master"
        
        if extract_path.exists():
            logger.info("TrashNet dataset đã được giải nén")
        else:
            logger.info("Đang giải nén TrashNet dataset...")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.config.raw_data_dir)
                    
                logger.info(f"Đã giải nén vào: {extract_path}")
                
            except Exception as e:
                logger.error(f"Lỗi khi giải nén: {e}")
                raise
        
        # Extract dataset-resized.zip nếu có
        dataset_zip = extract_path / "data" / "dataset-resized.zip"
        dataset_extract_path = self.config.raw_data_dir / "dataset-resized"
        
        if dataset_zip.exists() and not dataset_extract_path.exists():
            logger.info("Đang giải nén dataset-resized.zip...")
            
            try:
                with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.config.raw_data_dir)
                    
                logger.info(f"Đã giải nén dataset images vào: {dataset_extract_path}")
                
            except Exception as e:
                logger.error(f"Lỗi khi giải nén dataset-resized.zip: {e}")
                raise
    
    def scan_dataset_structure(self) -> Dict[str, List[Path]]:
        """
        Quét cấu trúc dataset để tìm tất cả ảnh theo class
        
        Returns:
            Dictionary {class_name: [list_of_image_paths]}
        """
        logger.info("Đang quét cấu trúc dataset...")
        
        # Có thể có nhiều cấu trúc dataset khác nhau
        possible_data_dirs = [
            self.config.raw_data_dir / "trashnet-master" / "data",
            self.config.raw_data_dir / "trashnet-master",
            self.config.raw_data_dir / "dataset-resized",
            self.config.raw_data_dir / "garbage_classification",
            self.config.raw_data_dir,
        ]
        
        class_images = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for data_dir in possible_data_dirs:
            if not data_dir.exists():
                continue
                
            logger.info(f"Quét thư mục: {data_dir}")
            
            # Tìm các thư mục con (các class)
            for class_dir in data_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                    
                class_name = class_dir.name.lower()
                
                # Bỏ qua các thư mục không phải class
                if class_name in ['__pycache__', '.git', 'scripts', 'docs']:
                    continue
                
                # Tìm tất cả ảnh trong thư mục class
                image_files = []
                for ext in image_extensions:
                    image_files.extend(class_dir.glob(f'*{ext}'))
                    image_files.extend(class_dir.glob(f'*{ext.upper()}'))
                
                if image_files:
                    # Map class name nếu cần
                    mapped_class = self.config.class_mapping.get(class_name, class_name)
                    
                    if mapped_class not in class_images:
                        class_images[mapped_class] = []
                    
                    class_images[mapped_class].extend(image_files)
                    logger.info(f"Tìm thấy {len(image_files)} ảnh cho class '{class_name}' -> '{mapped_class}'")
        
        # Lọc classes có đủ ảnh
        filtered_class_images = {}
        for class_name, images in class_images.items():
            if len(images) >= self.config.min_images_per_class:
                filtered_class_images[class_name] = images
                logger.info(f"Class '{class_name}': {len(images)} ảnh (đạt yêu cầu)")
            else:
                logger.warning(f"Class '{class_name}': {len(images)} ảnh (ít hơn {self.config.min_images_per_class}, bỏ qua)")
        
        self.class_names = sorted(list(filtered_class_images.keys()))
        self.class_counts = {name: len(images) for name, images in filtered_class_images.items()}
        
        logger.info(f"Tổng cộng {len(self.class_names)} classes hợp lệ: {self.class_names}")
        
        return filtered_class_images
    
    def preprocess_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Tiền xử lý một ảnh (resize, normalize, etc.)
        
        Args:
            image_path: Đường dẫn ảnh gốc
            output_path: Đường dẫn ảnh đầu ra
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            # Đọc ảnh
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Không thể đọc ảnh: {image_path}")
                return False
            
            # Resize ảnh
            image_resized = cv2.resize(image, self.config.img_size, interpolation=cv2.INTER_AREA)
            
            # Tạo thư mục output nếu chưa có
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu ảnh
            success = cv2.imwrite(str(output_path), image_resized)
            
            if not success:
                logger.warning(f"Không thể lưu ảnh: {output_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Lỗi khi xử lý ảnh {image_path}: {e}")
            return False
    
    def split_and_process_data(self, class_images: Dict[str, List[Path]]) -> None:
        """
        Chia dataset thành train/val/test và xử lý ảnh
        
        Args:
            class_images: Dictionary {class_name: [image_paths]}
        """
        logger.info("Đang chia dataset và xử lý ảnh...")
        
        # Thống kê
        total_processed = 0
        split_stats = {'train': {}, 'val': {}, 'test': {}}
        
        for class_name, image_paths in class_images.items():
            logger.info(f"Xử lý class '{class_name}' với {len(image_paths)} ảnh...")
            
            # Trộn danh sách ảnh
            np.random.seed(42)  # Để reproducible
            shuffled_paths = np.random.permutation(image_paths)
            
            # Chia train/val/test
            n_total = len(shuffled_paths)
            n_train = int(n_total * self.config.train_ratio)
            n_val = int(n_total * self.config.val_ratio)
            
            train_paths = shuffled_paths[:n_train]
            val_paths = shuffled_paths[n_train:n_train + n_val]
            test_paths = shuffled_paths[n_train + n_val:]
            
            splits = {
                'train': train_paths,
                'val': val_paths,
                'test': test_paths
            }
            
            # Xử lý từng split
            for split_name, paths in splits.items():
                split_class_dir = self.config.processed_data_dir / split_name / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                processed_count = 0
                
                for i, img_path in enumerate(tqdm(paths, desc=f"{split_name}/{class_name}")):
                    # Tạo tên file mới
                    img_extension = img_path.suffix
                    new_filename = f"{class_name}_{split_name}_{i:04d}{img_extension}"
                    output_path = split_class_dir / new_filename
                    
                    # Xử lý ảnh
                    if self.preprocess_image(img_path, output_path):
                        processed_count += 1
                
                split_stats[split_name][class_name] = processed_count
                total_processed += processed_count
                
                logger.info(f"  {split_name}: {processed_count}/{len(paths)} ảnh")
        
        logger.info(f"Đã xử lý tổng cộng {total_processed} ảnh")
        
        # Lưu thống kê split
        stats_path = self.config.processed_data_dir / "split_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(split_stats, f, indent=2)
        
        logger.info(f"Đã lưu thống kê split: {stats_path}")
    
    def create_dataset_yaml(self) -> None:
        """Tạo file dataset.yaml cho YOLO classification training"""
        dataset_config = {
            'path': str(self.config.processed_data_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.config.processed_data_dir / "dataset_classification.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Đã tạo dataset config: {yaml_path}")
        
        # Lưu class mapping
        mapping_path = self.config.processed_data_dir / "class_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(self.config.class_mapping, f, indent=2)
            
        logger.info(f"Đã lưu class mapping: {mapping_path}")
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Tạo thống kê dataset chi tiết"""
        stats = {
            'total_classes': len(self.class_names),
            'class_names': self.class_names,
            'original_class_counts': self.class_counts,
            'splits': {}
        }
        
        total_images = 0
        
        for split in ['train', 'val', 'test']:
            split_dir = self.config.processed_data_dir / split
            split_stats = {
                'total_images': 0,
                'classes': {}
            }
            
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        image_count = len([f for f in class_dir.iterdir() 
                                         if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
                        
                        split_stats['classes'][class_name] = image_count
                        split_stats['total_images'] += image_count
            
            stats['splits'][split] = split_stats
            total_images += split_stats['total_images']
        
        stats['total_images'] = total_images
        
        # Lưu statistics    
        stats_path = self.config.processed_data_dir / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Đã tạo thống kê dataset: {stats_path}")
        
        # In thống kê
        logger.info("=== THỐNG KÊ DATASET CLASSIFICATION ===")
        logger.info(f"Tổng số classes: {stats['total_classes']}")
        logger.info(f"Tổng số ảnh: {stats['total_images']}")
        
        for split, data in stats['splits'].items():
            logger.info(f"{split.upper()}: {data['total_images']} ảnh")
            for class_name, count in data['classes'].items():
                logger.info(f"  {class_name}: {count}")
        
        return stats
    
    def validate_dataset(self) -> bool:
        """Kiểm tra tính hợp lệ của dataset đã xử lý"""
        logger.info("Đang kiểm tra dataset...")
        
        valid = True
        
        # Kiểm tra cấu trúc thư mục
        for split in ['train', 'val', 'test']:
            split_dir = self.config.processed_data_dir / split
            if not split_dir.exists():
                logger.error(f"Thiếu thư mục {split}")
                valid = False
                continue
            
            # Kiểm tra từng class
            for class_name in self.class_names:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    logger.warning(f"Thiếu thư mục {split}/{class_name}")
                    continue
                
                # Đếm ảnh
                image_count = len([f for f in class_dir.iterdir() 
                                 if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
                
                if image_count == 0:
                    logger.warning(f"Không có ảnh trong {split}/{class_name}")
        
        # Kiểm tra file config
        yaml_path = self.config.processed_data_dir / "dataset_classification.yaml"
        if not yaml_path.exists():
            logger.error("Thiếu file dataset_classification.yaml")
            valid = False
        
        if valid:
            logger.info("Dataset hợp lệ!")
        else:
            logger.error("Dataset có vấn đề!")
        
        return valid
    
    def run_preprocessing(self) -> Dict[str, Any]:
        """Chạy toàn bộ quá trình preprocessing"""
        try:
            logger.info("=== BẮT ĐẦU PREPROCESSING CLASSIFICATION DATASET ===")
            
            # 1. Tải và giải nén dataset (nếu cần)
            try:
                self.download_trashnet_dataset()
                self.extract_trashnet_dataset()
            except Exception as e:
                logger.warning(f"Không thể tải dataset tự động: {e}")
                logger.info("Hãy đảm bảo dataset đã được tải thủ công vào thư mục raw")
            
            # 2. Quét cấu trúc dataset
            class_images = self.scan_dataset_structure()
            
            if not class_images:
                raise ValueError("Không tìm thấy dataset hợp lệ!")
            
            # 3. Chia và xử lý data
            self.split_and_process_data(class_images)
            
            # 4. Tạo config files
            self.create_dataset_yaml()
            
            # 5. Tạo thống kê
            stats = self.generate_statistics()
            
            # 6. Kiểm tra dataset
            self.validate_dataset()
            
            logger.info("=== HOÀN THÀNH PREPROCESSING CLASSIFICATION DATASET ===")
            
            return stats
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình preprocessing: {e}")
            raise


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="TrashNet Dataset Preprocessing cho Classification")
    parser.add_argument("--data-dir", type=str, default="data/classification",
                       help="Thư mục chứa data")
    parser.add_argument("--img-size", type=int, default=224,
                       help="Kích thước ảnh sau khi resize")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Tỷ lệ train split")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Tỷ lệ validation split")
    parser.add_argument("--min-images", type=int, default=50,
                       help="Số ảnh tối thiểu mỗi class")
    
    args = parser.parse_args()
    
    # Khởi tạo config
    config = ClassificationConfig(
        raw_data_dir=Path(args.data_dir) / "raw",
        processed_data_dir=Path(args.data_dir) / "processed",
        img_size=(args.img_size, args.img_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        min_images_per_class=args.min_images
    )
    
    # Chạy preprocessing
    processor = TrashNetProcessor(config)
    processor.run_preprocessing()


if __name__ == "__main__":
    main()
