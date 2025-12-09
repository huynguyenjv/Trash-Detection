#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Roboflow Garbage Classification Dataset

Mô tả:
    Script tải dataset Garbage Classification từ Roboflow Universe
    Dataset này có bounding box annotations phù hợp cho:
    - Object Detection (YOLOv8)
    - Image Classification
    
    Dataset: GARBAGE CLASSIFICATION 3
    URL: https://universe.roboflow.com/material-identification/garbage-classification-3
    Classes: BIODEGRADABLE, CARDBOARD, CLOTH, GLASS, METAL, PAPER, PLASTIC (7 classes)
    Images: ~10,000 images với bounding box annotations

Author: Huy Nguyen
Email: huynguyen@example.com
Date: December 2025
Version: 1.0.0
License: MIT
"""

import os
import sys
import logging
from pathlib import Path
import zipfile
import shutil
import yaml

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_roboflow_installed():
    """Kiểm tra roboflow đã được cài đặt chưa"""
    try:
        from roboflow import Roboflow
        return True
    except ImportError:
        logger.error("roboflow chưa được cài đặt!")
        logger.info("Chạy: pip install roboflow")
        return False


def download_garbage_dataset(api_key: str = None, download_dir: str = "data/roboflow_garbage"):
    """
    Tải dataset Garbage Classification 3 từ Roboflow
    
    Args:
        api_key: Roboflow API key (lấy từ https://app.roboflow.com/settings/api)
        download_dir: Thư mục lưu dataset
    """
    from roboflow import Roboflow
    
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("DOWNLOADING ROBOFLOW GARBAGE CLASSIFICATION DATASET")
    logger.info("="*60)
    
    # Kết nối Roboflow
    if api_key:
        rf = Roboflow(api_key=api_key)
    else:
        # Sử dụng public API hoặc yêu cầu input
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            logger.info("\n" + "="*60)
            logger.info("HƯỚNG DẪN LẤY ROBOFLOW API KEY:")
            logger.info("1. Truy cập: https://app.roboflow.com/")
            logger.info("2. Đăng ký/Đăng nhập tài khoản")
            logger.info("3. Vào Settings > API > Copy API Key")
            logger.info("="*60 + "\n")
            api_key = input("Nhập Roboflow API Key: ").strip()
        
        rf = Roboflow(api_key=api_key)
    
    # Tải dataset
    logger.info("Đang tải dataset từ Roboflow Universe...")
    logger.info("Project: material-identification/garbage-classification-3")
    
    try:
        project = rf.workspace("material-identification").project("garbage-classification-3")
        version = project.version(2)  # Version 2 là phổ biến nhất
        
        # Download với YOLO format
        dataset = version.download("yolov8", location=str(download_path))
        
        logger.info(f"✅ Dataset đã được tải về: {download_path}")
        
        return str(download_path)
        
    except Exception as e:
        logger.error(f"Lỗi khi tải dataset: {e}")
        logger.info("\n" + "="*60)
        logger.info("CÁCH TẢI THỦ CÔNG:")
        logger.info("1. Truy cập: https://universe.roboflow.com/material-identification/garbage-classification-3")
        logger.info("2. Click 'Download Dataset' > chọn 'YOLOv8'")
        logger.info("3. Giải nén vào thư mục: data/roboflow_garbage")
        logger.info("="*60)
        return None


def prepare_dataset_structure(source_dir: str, target_dir: str = "data/detection"):
    """
    Chuẩn bị cấu trúc dataset cho training
    
    Args:
        source_dir: Thư mục chứa dataset đã tải
        target_dir: Thư mục đích để lưu dataset đã chuẩn bị
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    logger.info("="*60)
    logger.info("PREPARING DATASET STRUCTURE")
    logger.info("="*60)
    
    # Tạo cấu trúc thư mục
    for split in ['train', 'valid', 'test']:
        (target_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Copy files
    split_mapping = {
        'train': 'train',
        'valid': 'valid',
        'val': 'valid',
        'test': 'test'
    }
    
    total_images = 0
    for src_split, dst_split in split_mapping.items():
        src_images = source_path / src_split / "images"
        src_labels = source_path / src_split / "labels"
        
        if src_images.exists():
            dst_images = target_path / "images" / dst_split
            dst_labels = target_path / "labels" / dst_split
            
            # Copy images
            for img_file in src_images.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img_file, dst_images / img_file.name)
                    total_images += 1
            
            # Copy labels
            if src_labels.exists():
                for label_file in src_labels.glob("*.txt"):
                    shutil.copy2(label_file, dst_labels / label_file.name)
            
            logger.info(f"Copied {src_split} -> {dst_split}")
    
    logger.info(f"✅ Total images prepared: {total_images}")
    
    # Tạo dataset.yaml
    create_dataset_yaml(target_path)
    
    return target_path


def create_dataset_yaml(dataset_dir: Path):
    """Tạo file dataset.yaml cho YOLO training"""
    
    # Classes cho Garbage Classification 3
    classes = [
        'BIODEGRADABLE',
        'CARDBOARD', 
        'CLOTH',
        'GLASS',
        'METAL',
        'PAPER',
        'PLASTIC'
    ]
    
    yaml_content = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes)
    }
    
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"✅ Created dataset.yaml: {yaml_path}")
    logger.info(f"   Classes ({len(classes)}): {', '.join(classes)}")
    
    return yaml_path


def download_alternative_dataset():
    """
    Tải dataset thay thế nếu không có Roboflow API key
    Sử dụng TACO dataset hoặc dataset public khác
    """
    logger.info("="*60)
    logger.info("ALTERNATIVE DATASET OPTIONS")
    logger.info("="*60)
    
    datasets = [
        {
            "name": "TACO Dataset",
            "url": "https://github.com/pedropro/TACO",
            "description": "Trash Annotations in Context - 1500+ images, 60 categories",
            "format": "COCO"
        },
        {
            "name": "Garbage Classification (Kaggle)",
            "url": "https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification",
            "description": "2500+ images, 6 classes (classification only)",
            "format": "ImageFolder"
        },
        {
            "name": "Roboflow Garbage Classification 3",
            "url": "https://universe.roboflow.com/material-identification/garbage-classification-3",
            "description": "10k images, 7 classes with bounding boxes",
            "format": "YOLOv8"
        }
    ]
    
    logger.info("Các dataset phù hợp cho Detection + Classification:\n")
    for i, ds in enumerate(datasets, 1):
        logger.info(f"{i}. {ds['name']}")
        logger.info(f"   URL: {ds['url']}")
        logger.info(f"   Mô tả: {ds['description']}")
        logger.info(f"   Format: {ds['format']}")
        logger.info("")
    
    logger.info("KHUYẾN NGHỊ: Sử dụng Roboflow Garbage Classification 3")
    logger.info("Lý do: Có bounding box annotations, 10k ảnh, 7 classes chuẩn")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Roboflow Garbage Dataset")
    parser.add_argument("--api-key", type=str, help="Roboflow API Key")
    parser.add_argument("--output", type=str, default="data/roboflow_garbage",
                       help="Output directory for downloaded dataset")
    parser.add_argument("--prepare", action="store_true",
                       help="Prepare dataset structure after download")
    parser.add_argument("--alternatives", action="store_true",
                       help="Show alternative dataset options")
    
    args = parser.parse_args()
    
    if args.alternatives:
        download_alternative_dataset()
        return
    
    # Kiểm tra roboflow
    if not check_roboflow_installed():
        logger.info("Cài đặt roboflow: pip install roboflow")
        download_alternative_dataset()
        return
    
    # Tải dataset
    downloaded_path = download_garbage_dataset(
        api_key=args.api_key,
        download_dir=args.output
    )
    
    if downloaded_path and args.prepare:
        prepare_dataset_structure(downloaded_path)
    
    logger.info("\n" + "="*60)
    logger.info("HOÀN TẤT!")
    logger.info("="*60)
    logger.info("Bước tiếp theo:")
    logger.info("1. python main.py --mode detection --train")
    logger.info("2. python main.py --mode classification --train")
    logger.info("="*60)


if __name__ == "__main__":
    main()
