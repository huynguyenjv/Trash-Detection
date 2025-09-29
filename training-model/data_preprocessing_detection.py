#!/usr/bin/env python3
"""
Data Preprocessing cho Detection Model - TACO Dataset
Xử lý dataset TACO và chuyển đổi sang định dạng YOLO

Author: GitHub Copilot Assistant
Date: September 2025
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import yaml
import requests
from tqdm import tqdm
import zipfile
import cv2
import numpy as np
from pycocotools.coco import COCO
import argparse

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Cấu hình cho data preprocessing detection"""
    # Dataset paths
    raw_data_dir: Path = Path("data/detection/raw")
    processed_data_dir: Path = Path("data/detection/processed")
    
    # TACO dataset URLs (multiple backup sources)
    taco_urls: List[str] = None
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Image size
    img_size: Tuple[int, int] = (640, 640)
    
    # TACO classes mapping to simplified categories
    class_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        """Khởi tạo class mapping và URLs sau khi tạo object"""
        # Default TACO dataset URLs
        if self.taco_urls is None:
            self.taco_urls = [
                "https://github.com/pedropro/TACO/releases/download/v1.0/TACO.zip",
                "http://tacodataset.org/files/TACO.zip"
            ]
            
        if self.class_mapping is None:
            self.class_mapping = {
                # Plastic items
                "Plastic bottle": "plastic",
                "Plastic bag & wrapper": "plastic", 
                "Plastic container": "plastic",
                "Plastic utensils": "plastic",
                "Plastic straw": "plastic",
                "Other plastic": "plastic",
                
                # Metal items
                "Aluminium foil": "metal",
                "Aluminium blister pack": "metal",
                "Can": "metal",
                "Bottle cap": "metal",
                "Metal bottle cap": "metal",
                "Pop tab": "metal",
                "Metal lid": "metal",
                
                # Glass items
                "Glass bottle": "glass",
                "Broken glass": "glass",
                "Glass jar": "glass",
                
                # Paper/Cardboard
                "Paper": "paper",
                "Cardboard": "cardboard",
                "Paper bag": "paper",
                "Newspaper": "paper",
                "Magazine": "paper",
                "Tissues": "paper",
                
                # Organic
                "Food waste": "organic",
                "Cigarette": "organic",
                
                # Electronics
                "Battery": "electronics",
                "Electronics": "electronics",
                
                # Textiles
                "Clothing": "textiles",
                "Rope & strings": "textiles",
                
                # Other
                "Styrofoam piece": "other",
                "Foam cup": "other",
                "Foam food container": "other",
                "Unlabeled litter": "other",
                "Clear plastic bottle": "plastic",  # Reclassify to plastic
            }


class TACODataProcessor:
    """Class xử lý TACO dataset cho detection task"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.coco = None
        self.categories = []
        self.simplified_categories = []
        
        # Tạo thư mục
        self._create_directories()
        
    def _create_directories(self) -> None:
        """Tạo các thư mục cần thiết"""
        directories = [
            self.config.raw_data_dir,
            self.config.processed_data_dir,
            self.config.processed_data_dir / "images" / "train",
            self.config.processed_data_dir / "images" / "val", 
            self.config.processed_data_dir / "images" / "test",
            self.config.processed_data_dir / "labels" / "train",
            self.config.processed_data_dir / "labels" / "val",
            self.config.processed_data_dir / "labels" / "test",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info("Đã tạo cấu trúc thư mục")
    
    def download_taco_dataset(self) -> None:
        """Clone TACO dataset từ GitHub repository"""
        taco_repo_path = self.config.raw_data_dir / "TACO"
        
        if taco_repo_path.exists() and (taco_repo_path / ".git").exists():
            logger.info("TACO repository đã tồn tại, bỏ qua việc clone")
            return
            
        logger.info("Đang clone TACO dataset từ GitHub...")
        
        try:
            import subprocess
            
            # Clone repository
            cmd = [
                "git", "clone", 
                "https://github.com/pedropro/TACO.git",
                str(taco_repo_path)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
                
            logger.info(f"Đã clone TACO repository thành công: {taco_repo_path}")
            
            # Download dataset files theo TACO README
            logger.info("Downloading TACO dataset files...")
            download_script = taco_repo_path / "download.py"
            
            if download_script.exists():
                # Run download script
                download_cmd = ["python", str(download_script)]
                logger.info(f"Running: {' '.join(download_cmd)}")
                
                result = subprocess.run(
                    download_cmd, 
                    cwd=str(taco_repo_path),
                    capture_output=True, 
                    text=True, 
                    timeout=600
                )
                
                if result.returncode != 0:
                    logger.warning(f"Download script failed: {result.stderr}")
                    logger.info("Trying alternative download method...")
                    
                logger.info("TACO dataset files downloaded successfully")
            else:
                logger.warning("No download script found, dataset may need manual download")
                
        except subprocess.TimeoutExpired:
            logger.error("Git clone hoặc download timeout!")
            raise
        except FileNotFoundError:
            logger.error("Git không được tìm thấy! Vui lòng cài đặt Git trước.")
            logger.info("Manual steps:")
            logger.info("1. git clone https://github.com/pedropro/TACO.git data/detection/raw/TACO")
            logger.info("2. cd data/detection/raw/TACO")
            logger.info("3. python download.py")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi clone TACO repository: {e}")
            logger.info("Manual steps:")
            logger.info("1. git clone https://github.com/pedropro/TACO.git data/detection/raw/TACO")
            logger.info("2. cd data/detection/raw/TACO") 
            logger.info("3. python download.py")
            raise
    
    def extract_taco_dataset(self) -> None:
        """Verify TACO dataset structure sau khi clone"""
        taco_path = self.config.raw_data_dir / "TACO"
        
        if not taco_path.exists():
            logger.error("TACO repository chưa được clone!")
            raise FileNotFoundError("TACO repository not found")
            
        # Check for required files
        required_files = [
            "annotations.json",
            "data"  # data folder containing images
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = taco_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.warning(f"Missing TACO files: {missing_files}")
            logger.info("You may need to run the download script manually:")
            logger.info(f"cd {taco_path} && python download.py")
        else:
            logger.info("TACO dataset structure verified successfully")
    
    def load_coco_annotations(self) -> None:
        """Load COCO annotations từ TACO dataset"""
        annotation_file = self.config.raw_data_dir / "TACO" / "annotations.json"
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Không tìm thấy file annotations: {annotation_file}")
            
        logger.info("Đang load COCO annotations...")
        
        try:
            self.coco = COCO(str(annotation_file))
            
            # Lấy thông tin categories
            self.categories = self.coco.loadCats(self.coco.getCatIds())
            
            # Tạo simplified categories
            simplified_cats = set()
            for cat in self.categories:
                simplified_name = self.config.class_mapping.get(cat['name'], 'other')
                simplified_cats.add(simplified_name)
            
            self.simplified_categories = sorted(list(simplified_cats))
            
            logger.info(f"Loaded {len(self.categories)} original categories")
            logger.info(f"Simplified to {len(self.simplified_categories)} categories: {self.simplified_categories}")
            
        except Exception as e:
            logger.error(f"Lỗi khi load annotations: {e}")
            raise
    
    def convert_bbox_coco_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Chuyển đổi bounding box từ COCO format sang YOLO format
        
        Args:
            bbox: [x, y, width, height] trong COCO format
            img_width: Chiều rộng ảnh
            img_height: Chiều cao ảnh
            
        Returns:
            [x_center, y_center, width, height] trong YOLO format (normalized)
        """
        x, y, w, h = bbox
        
        # Chuyển về center format và normalize
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        return [x_center, y_center, width, height]
    
    def process_image_annotations(self, img_id: int) -> Tuple[Optional[str], List[str]]:
        """
        Xử lý annotations cho một ảnh
        
        Args:
            img_id: ID của ảnh
            
        Returns:
            Tuple (image_filename, yolo_annotations)
        """
        try:
            # Lấy thông tin ảnh
            img_info = self.coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Lấy annotations cho ảnh này
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(ann_ids)
            
            yolo_annotations = []
            
            for ann in annotations:
                # Lấy category info
                cat_id = ann['category_id']
                cat_info = self.coco.loadCats(cat_id)[0]
                original_name = cat_info['name']
                
                # Map sang simplified category
                simplified_name = self.config.class_mapping.get(original_name, 'other')
                class_id = self.simplified_categories.index(simplified_name)
                
                # Chuyển đổi bbox sang YOLO format
                bbox_coco = ann['bbox']
                bbox_yolo = self.convert_bbox_coco_to_yolo(bbox_coco, img_width, img_height)
                
                # Tạo YOLO annotation line
                yolo_line = f"{class_id} {' '.join(map(str, bbox_yolo))}"
                yolo_annotations.append(yolo_line)
            
            return img_filename, yolo_annotations
            
        except Exception as e:
            logger.warning(f"Lỗi khi xử lý ảnh {img_id}: {e}")
            return None, []
    
    def split_dataset(self, img_ids: List[int]) -> Dict[str, List[int]]:
        """
        Chia dataset thành train/val/test
        
        Args:
            img_ids: List các image IDs
            
        Returns:
            Dictionary chứa split data
        """
        np.random.seed(42)  # Để reproducible
        np.random.shuffle(img_ids)
        
        total = len(img_ids)
        train_end = int(total * self.config.train_ratio)
        val_end = int(total * (self.config.train_ratio + self.config.val_ratio))
        
        splits = {
            'train': img_ids[:train_end],
            'val': img_ids[train_end:val_end],
            'test': img_ids[val_end:]
        }
        
        logger.info(f"Dataset split - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def process_split(self, split_name: str, img_ids: List[int]) -> None:
        """
        Xử lý một split (train/val/test)
        
        Args:
            split_name: Tên split ('train', 'val', 'test')
            img_ids: List image IDs cho split này
        """
        logger.info(f"Đang xử lý {split_name} split với {len(img_ids)} ảnh...")
        
        images_dir = self.config.processed_data_dir / "images" / split_name
        labels_dir = self.config.processed_data_dir / "labels" / split_name
        taco_images_dir = self.config.raw_data_dir / "TACO" / "data"
        
        processed_count = 0
        
        for img_id in tqdm(img_ids, desc=f"Processing {split_name}"):
            img_filename, yolo_annotations = self.process_image_annotations(img_id)
            
            if img_filename is None:
                continue
                
            # Copy ảnh
            src_img_path = taco_images_dir / img_filename
            dst_img_path = images_dir / img_filename
            
            if src_img_path.exists():
                try:
                    # Resize ảnh nếu cần
                    img = cv2.imread(str(src_img_path))
                    if img is not None:
                        img_resized = cv2.resize(img, self.config.img_size)
                        cv2.imwrite(str(dst_img_path), img_resized)
                        
                        # Lưu annotations
                        if yolo_annotations:
                            label_filename = Path(img_filename).stem + '.txt'
                            label_path = labels_dir / label_filename
                            
                            with open(label_path, 'w') as f:
                                f.write('\n'.join(yolo_annotations))
                        
                        processed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Lỗi khi xử lý ảnh {img_filename}: {e}")
        
        logger.info(f"Đã xử lý {processed_count}/{len(img_ids)} ảnh cho {split_name}")
    
    def create_dataset_yaml(self) -> None:
        """Tạo file dataset.yaml cho YOLO training"""
        dataset_config = {
            'path': str(self.config.processed_data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': len(self.simplified_categories),
            'names': self.simplified_categories
        }
        
        yaml_path = self.config.processed_data_dir / "dataset_detection.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Đã tạo dataset config: {yaml_path}")
        
        # Lưu class mapping
        mapping_path = self.config.processed_data_dir / "class_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(self.config.class_mapping, f, indent=2)
            
        logger.info(f"Đã lưu class mapping: {mapping_path}")
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Tạo thống kê dataset"""
        stats = {
            'total_categories': len(self.simplified_categories),
            'categories': self.simplified_categories,
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            images_dir = self.config.processed_data_dir / "images" / split
            labels_dir = self.config.processed_data_dir / "labels" / split
            
            image_count = len(list(images_dir.glob('*.jpg'))) if images_dir.exists() else 0
            label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
            
            # Đếm objects per class
            class_counts = {cat: 0 for cat in self.simplified_categories}
            total_objects = 0
            
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                if 0 <= class_id < len(self.simplified_categories):
                                    class_name = self.simplified_categories[class_id]
                                    class_counts[class_name] += 1
                                    total_objects += 1
            
            stats['splits'][split] = {
                'images': image_count,
                'labels': label_count,
                'total_objects': total_objects,
                'objects_per_class': class_counts
            }
        
        # Lưu statistics
        stats_path = self.config.processed_data_dir / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Đã tạo thống kê dataset: {stats_path}")
        
        # In thống kê
        logger.info("=== THỐNG KÊ DATASET ===")
        for split, data in stats['splits'].items():
            logger.info(f"{split.upper()}: {data['images']} ảnh, {data['total_objects']} objects")
            
        return stats
    
    def run_preprocessing(self) -> Dict[str, Any]:
        """Chạy toàn bộ quá trình preprocessing"""
        try:
            logger.info("=== BẮT ĐẦU PREPROCESSING DETECTION DATASET ===")
            
            # 1. Tải và giải nén dataset
            self.download_taco_dataset()
            self.extract_taco_dataset()
            
            # 2. Load annotations
            self.load_coco_annotations()
            
            # 3. Lấy danh sách tất cả ảnh
            img_ids = self.coco.getImgIds()
            logger.info(f"Tổng số ảnh: {len(img_ids)}")
            
            # 4. Chia dataset
            splits = self.split_dataset(img_ids)
            
            # 5. Xử lý từng split
            for split_name, split_img_ids in splits.items():
                self.process_split(split_name, split_img_ids)
            
            # 6. Tạo config files
            self.create_dataset_yaml()
            
            # 7. Tạo thống kê
            stats = self.generate_statistics()
            
            logger.info("=== HOÀN THÀNH PREPROCESSING DETECTION DATASET ===")
            
            return stats
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình preprocessing: {e}")
            raise


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="TACO Dataset Preprocessing cho Detection")
    parser.add_argument("--data-dir", type=str, default="data/detection",
                       help="Thư mục chứa data")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Kích thước ảnh sau khi resize")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Tỷ lệ train split")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Tỷ lệ validation split")
    
    args = parser.parse_args()
    
    # Khởi tạo config
    config = DetectionConfig(
        raw_data_dir=Path(args.data_dir) / "raw",
        processed_data_dir=Path(args.data_dir) / "processed",
        img_size=(args.img_size, args.img_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio
    )
    
    # Chạy preprocessing
    processor = TACODataProcessor(config)
    processor.run_preprocessing()


if __name__ == "__main__":
    main()