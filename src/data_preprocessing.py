"""
Script tiền xử lý dữ liệu cho dự án Trash Detection
Chuyển đổi dataset classification thành object detection format cho YOLOv8

Author: Huy Nguyen
Date: August 2025
"""

import os
import json
import shutil
import logging
import zipfile
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import kaggle

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
class DatasetConfig:
    """Cấu hình dataset"""
    dataset_name: str = "sumn2u/garbage-classification-v2"
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    bbox_coverage: float = 0.8  # Tỷ lệ bounding box so với kích thước ảnh
    min_image_size: int = 224


class DataPreprocessor:
    """Class chính để xử lý dữ liệu"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.class_names: List[str] = []
        self.class_to_id: Dict[str, int] = {}
        
        # Tạo các thư mục cần thiết
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Tạo cấu trúc thư mục"""
        directories = [
            self.config.raw_data_path,
            self.config.processed_data_path,
            self.config.processed_data_path / "images" / "train",
            self.config.processed_data_path / "images" / "val",
            self.config.processed_data_path / "images" / "test",
            self.config.processed_data_path / "labels" / "train",
            self.config.processed_data_path / "labels" / "val",
            self.config.processed_data_path / "labels" / "test",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Đã tạo thư mục: {directory}")
    
    def download_dataset(self) -> None:
        """Download dataset từ Kaggle"""
        try:
            logger.info(f"Bắt đầu download dataset: {self.config.dataset_name}")
            
            # Kiểm tra Kaggle API key
            if not os.path.exists(os.path.expanduser("~/.config/kaggle/kaggle.json")):
                raise FileNotFoundError(
                    "Không tìm thấy Kaggle API key. "
                    "Vui lòng tạo file ~/.config/kaggle/kaggle.json với API credentials"
                )
            
            # Download dataset
            kaggle.api.dataset_download_files(
                self.config.dataset_name,
                path=self.config.raw_data_path,
                unzip=True
            )
            
            logger.info("Download dataset thành công!")
            
        except Exception as e:
            logger.error(f"Lỗi khi download dataset: {e}")
            raise
    
    def extract_class_names(self) -> None:
        """Trích xuất tên các class từ thư mục dataset"""
        try:
            # Tìm thư mục chứa các class
            dataset_root = None
            for item in self.config.raw_data_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    potential_classes = [d for d in item.iterdir() if d.is_dir()]
                    if len(potential_classes) > 1:  # Có nhiều class
                        dataset_root = item
                        break
            
            if dataset_root is None:
                raise ValueError("Không tìm thấy thư mục chứa các class")
            
            # Lấy danh sách class names
            self.class_names = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
            self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
            
            logger.info(f"Tìm thấy {len(self.class_names)} classes: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Lỗi khi extract class names: {e}")
            raise
    
    def generate_bounding_box(self, image_path: Path) -> Tuple[float, float, float, float]:
        """
        Tạo bounding box cho ảnh (giả định object chiếm phần lớn ảnh)
        
        Returns:
            tuple: (x_center, y_center, width, height) - normalized coordinates
        """
        try:
            # Đọc ảnh để lấy kích thước
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path}")
            
            h, w = image.shape[:2]
            
            # Tạo bounding box ở giữa ảnh với kích thước coverage% của ảnh
            coverage = self.config.bbox_coverage
            
            # Tính toán bounding box
            box_w = w * coverage
            box_h = h * coverage
            
            # Center của bounding box (giữa ảnh)
            center_x = w / 2
            center_y = h / 2
            
            # Normalize coordinates (0-1)
            x_center = center_x / w
            y_center = center_y / h
            width = box_w / w
            height = box_h / h
            
            return x_center, y_center, width, height
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo bounding box cho {image_path}: {e}")
            # Trả về bounding box mặc định
            return 0.5, 0.5, 0.8, 0.8
    
    def create_yolo_annotation(self, image_path: Path, class_name: str) -> str:
        """
        Tạo annotation file format YOLO
        
        Args:
            image_path: Đường dẫn đến ảnh
            class_name: Tên class
            
        Returns:
            str: Nội dung annotation
        """
        class_id = self.class_to_id[class_name]
        x_center, y_center, width, height = self.generate_bounding_box(image_path)
        
        # Format YOLO: class_id x_center y_center width height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    
    def collect_all_images(self) -> List[Tuple[Path, str]]:
        """
        Thu thập tất cả ảnh và class tương ứng
        
        Returns:
            List[Tuple[Path, str]]: List các tuple (image_path, class_name)
        """
        all_images = []
        
        # Tìm thư mục chứa các class
        dataset_root = None
        for item in self.config.raw_data_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                potential_classes = [d for d in item.iterdir() if d.is_dir()]
                if len(potential_classes) > 1:
                    dataset_root = item
                    break
        
        if dataset_root is None:
            raise ValueError("Không tìm thấy thư mục chứa các class")
        
        # Thu thập ảnh từ mỗi class
        for class_dir in dataset_root.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            if class_name not in self.class_names:
                continue
            
            # Lấy tất cả ảnh trong class
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    # Kiểm tra ảnh có hợp lệ không
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None and min(img.shape[:2]) >= self.config.min_image_size:
                            all_images.append((img_path, class_name))
                    except Exception as e:
                        logger.warning(f"Bỏ qua ảnh lỗi {img_path}: {e}")
        
        logger.info(f"Tổng số ảnh hợp lệ: {len(all_images)}")
        return all_images
    
    def split_dataset(self, all_images: List[Tuple[Path, str]]) -> Dict[str, List[Tuple[Path, str]]]:
        """
        Chia dataset thành train/val/test
        
        Args:
            all_images: List tất cả ảnh và class
            
        Returns:
            Dict chứa split data
        """
        # Tách images và labels
        image_paths = [item[0] for item in all_images]
        class_labels = [item[1] for item in all_images]
        
        # Stratified split để đảm bảo phân bố class
        # Tạo train+val và test
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            image_paths, class_labels,
            test_size=self.config.test_ratio,
            stratify=class_labels,
            random_state=42
        )
        
        # Chia train+val thành train và val
        val_ratio_adjusted = self.config.val_ratio / (1 - self.config.test_ratio)
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images, train_val_labels,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=42
        )
        
        # Kết hợp lại thành tuples
        train_data = list(zip(train_images, train_labels))
        val_data = list(zip(val_images, val_labels))
        test_data = list(zip(test_images, test_labels))
        
        logger.info(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def process_split_data(self, split_name: str, data: List[Tuple[Path, str]]) -> None:
        """
        Xử lý và copy ảnh + tạo annotations cho một split
        
        Args:
            split_name: 'train', 'val', hoặc 'test'
            data: List ảnh và class cho split này
        """
        images_dir = self.config.processed_data_path / "images" / split_name
        labels_dir = self.config.processed_data_path / "labels" / split_name
        
        logger.info(f"Đang xử lý {split_name} split với {len(data)} ảnh...")
        
        for idx, (img_path, class_name) in enumerate(tqdm(data, desc=f"Processing {split_name}")):
            try:
                # Tạo tên file mới
                new_filename = f"{split_name}_{idx:05d}{img_path.suffix}"
                
                # Copy ảnh
                new_img_path = images_dir / new_filename
                shutil.copy2(img_path, new_img_path)
                
                # Tạo annotation file
                annotation_content = self.create_yolo_annotation(img_path, class_name)
                annotation_path = labels_dir / f"{new_filename.rsplit('.', 1)[0]}.txt"
                
                with open(annotation_path, 'w') as f:
                    f.write(annotation_content)
                    
            except Exception as e:
                logger.error(f"Lỗi khi xử lý ảnh {img_path}: {e}")
    
    def create_dataset_yaml(self) -> None:
        """Tạo file dataset.yaml cho YOLOv8"""
        dataset_config = {
            'path': str(self.config.processed_data_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.config.processed_data_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Đã tạo dataset.yaml tại: {yaml_path}")
    
    def create_summary_report(self, split_data: Dict[str, List[Tuple[Path, str]]]) -> None:
        """Tạo báo cáo tóm tắt dataset"""
        summary = {
            'total_classes': len(self.class_names),
            'class_names': self.class_names,
            'splits': {}
        }
        
        for split_name, data in split_data.items():
            # Đếm số ảnh mỗi class
            class_counts = {}
            for _, class_name in data:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            summary['splits'][split_name] = {
                'total_images': len(data),
                'class_distribution': class_counts
            }
        
        # Lưu báo cáo
        report_path = self.config.processed_data_path / "dataset_summary.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # In báo cáo
        logger.info("=== BÁO CÁO DATASET ===")
        logger.info(f"Tổng số classes: {summary['total_classes']}")
        logger.info(f"Classes: {', '.join(summary['class_names'])}")
        
        for split_name, split_info in summary['splits'].items():
            logger.info(f"\n{split_name.upper()}:")
            logger.info(f"  Tổng ảnh: {split_info['total_images']}")
            for class_name, count in split_info['class_distribution'].items():
                logger.info(f"  {class_name}: {count} ảnh")
    
    def run_preprocessing(self) -> None:
        """Chạy toàn bộ quá trình tiền xử lý"""
        try:
            logger.info("=== BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ===")
            
            # Bước 1: Download dataset
            if not any(self.config.raw_data_path.iterdir()):
                self.download_dataset()
            else:
                logger.info("Dataset đã tồn tại, bỏ qua download")
            
            # Bước 2: Extract class names
            self.extract_class_names()
            
            # Bước 3: Collect tất cả ảnh
            all_images = self.collect_all_images()
            
            # Bước 4: Split dataset
            split_data = self.split_dataset(all_images)
            
            # Bước 5: Process từng split
            for split_name, data in split_data.items():
                self.process_split_data(split_name, data)
            
            # Bước 6: Tạo dataset.yaml
            self.create_dataset_yaml()
            
            # Bước 7: Tạo báo cáo
            self.create_summary_report(split_data)
            
            logger.info("=== HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU ===")
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình tiền xử lý: {e}")
            raise


def main():
    """Hàm main"""
    try:
        # Khởi tạo config
        config = DatasetConfig()
        
        # Khởi tạo preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Chạy tiền xử lý
        preprocessor.run_preprocessing()
        
    except Exception as e:
        logger.error(f"Lỗi chương trình: {e}")
        raise


if __name__ == "__main__":
    main()
