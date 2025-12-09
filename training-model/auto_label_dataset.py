#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Label Dataset for Detection Training

Mô tả:
    Script tự động tạo bounding box labels cho dataset classification
    sử dụng pre-trained object detection model hoặc
    tạo full-image bounding box cho các ảnh chỉ chứa 1 object.
    
    Phương pháp:
    1. Full-image bounding box: Giả sử ảnh chứa 1 object chiếm toàn bộ ảnh
    2. Auto-detect: Sử dụng YOLOv8 pre-trained để detect objects
    3. Manual annotation: Hướng dẫn sử dụng LabelImg hoặc Roboflow

Author: Huy Nguyen
Email: huynguyen@example.com
Date: December 2025
Version: 1.0.0
License: MIT
"""

import os
import sys
import cv2
import numpy as np
import shutil
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AutoLabelConfig:
    """Cấu hình cho auto-labeling"""
    # Paths
    source_dir: Path = Path("../data/raw/garbage-dataset")
    output_dir: Path = Path("data/detection_from_classification")
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Image processing
    img_size: Tuple[int, int] = (640, 640)
    
    # Labeling method
    method: str = "center_crop"  # "full_image", "center_crop", "auto_detect"
    
    # Center crop params (giả sử object ở giữa ảnh)
    crop_ratio: float = 0.8  # Object chiếm 80% ảnh
    
    # Random seed
    random_seed: int = 42


class AutoLabeler:
    """Tự động tạo labels cho detection từ classification dataset"""
    
    def __init__(self, config: AutoLabelConfig):
        self.config = config
        
        # Class mapping
        self.class_mapping = {
            "battery": "other",
            "biological": "organic",
            "cardboard": "cardboard",
            "clothes": "other",
            "glass": "glass",
            "metal": "metal",
            "paper": "paper",
            "plastic": "plastic",
            "shoes": "other",
            "trash": "other"
        }
        
        # Unified classes for detection (6 classes)
        self.unified_classes = ["cardboard", "glass", "metal", "organic", "other", "paper", "plastic"]
        
        random.seed(config.random_seed)
    
    def create_directories(self):
        """Tạo cấu trúc thư mục"""
        output = self.config.output_dir
        
        for split in ['train', 'val', 'test']:
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created directory structure at: {output}")
    
    def get_all_images(self) -> Dict[str, List[Path]]:
        """Lấy tất cả ảnh theo class"""
        images_by_class = {}
        
        for class_dir in self.config.source_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                images = []
                
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                    images.extend(list(class_dir.glob(f"*{ext}")))
                
                if images:
                    images_by_class[class_name] = images
                    logger.info(f"Class '{class_name}': {len(images)} images")
        
        return images_by_class
    
    def create_yolo_label(self, img_path: Path, class_name: str, 
                          method: str = "center_crop") -> str:
        """
        Tạo YOLO format label cho ảnh
        
        YOLO format: class_id center_x center_y width height
        Tất cả giá trị normalized (0-1)
        """
        # Map to unified class
        unified_class = self.class_mapping.get(class_name.lower(), "other")
        class_id = self.unified_classes.index(unified_class)
        
        if method == "full_image":
            # Object chiếm toàn bộ ảnh
            cx, cy, w, h = 0.5, 0.5, 1.0, 1.0
            
        elif method == "center_crop":
            # Object ở giữa, chiếm crop_ratio của ảnh
            ratio = self.config.crop_ratio
            cx, cy = 0.5, 0.5
            w, h = ratio, ratio
            
        elif method == "auto_detect":
            # Sử dụng model để detect
            bbox = self._auto_detect_bbox(img_path)
            if bbox:
                cx, cy, w, h = bbox
            else:
                # Fallback to center crop
                ratio = self.config.crop_ratio
                cx, cy = 0.5, 0.5
                w, h = ratio, ratio
        else:
            # Default center crop
            ratio = 0.8
            cx, cy = 0.5, 0.5
            w, h = ratio, ratio
        
        # Đảm bảo giá trị trong phạm vi [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0.1, min(1, w))  # Min width 10%
        h = max(0.1, min(1, h))  # Min height 10%
        
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    
    def _auto_detect_bbox(self, img_path: Path) -> Optional[Tuple[float, float, float, float]]:
        """
        Tự động detect bounding box sử dụng edge detection hoặc contour
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            # Chuyển sang grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Blur để giảm noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Tìm contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Lấy contour lớn nhất
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                img_h, img_w = img.shape[:2]
                
                # Normalize
                cx = (x + w/2) / img_w
                cy = (y + h/2) / img_h
                nw = w / img_w
                nh = h / img_h
                
                # Nếu contour quá nhỏ, fallback
                if nw < 0.1 or nh < 0.1:
                    return None
                
                return (cx, cy, nw, nh)
            
            return None
            
        except Exception as e:
            logger.debug(f"Auto-detect failed for {img_path}: {e}")
            return None
    
    def process_dataset(self, method: str = "center_crop"):
        """
        Xử lý toàn bộ dataset
        """
        logger.info("="*60)
        logger.info("AUTO-LABELING CLASSIFICATION DATASET FOR DETECTION")
        logger.info(f"Method: {method}")
        logger.info("="*60)
        
        # Tạo thư mục
        self.create_directories()
        
        # Lấy tất cả ảnh
        images_by_class = self.get_all_images()
        
        # Gộp tất cả ảnh với class labels
        all_data = []
        for class_name, images in images_by_class.items():
            for img_path in images:
                all_data.append((img_path, class_name))
        
        # Shuffle
        random.shuffle(all_data)
        
        # Split data
        total = len(all_data)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)
        
        splits = {
            'train': all_data[:train_end],
            'val': all_data[train_end:val_end],
            'test': all_data[val_end:]
        }
        
        # Process each split
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, data in splits.items():
            logger.info(f"\nProcessing {split_name} split ({len(data)} images)...")
            
            for img_path, class_name in data:
                try:
                    # Copy image
                    dst_img = self.config.output_dir / "images" / split_name / img_path.name
                    
                    # Resize và save
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img_resized = cv2.resize(img, self.config.img_size)
                    cv2.imwrite(str(dst_img), img_resized)
                    
                    # Create label
                    label = self.create_yolo_label(img_path, class_name, method)
                    
                    # Save label
                    label_name = img_path.stem + ".txt"
                    dst_label = self.config.output_dir / "labels" / split_name / label_name
                    
                    with open(dst_label, 'w') as f:
                        f.write(label + "\n")
                    
                    stats[split_name] += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
        
        # Tạo dataset.yaml
        self.create_dataset_yaml()
        
        # In thống kê
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Train: {stats['train']} images")
        logger.info(f"Val: {stats['val']} images")
        logger.info(f"Test: {stats['test']} images")
        logger.info(f"Total: {sum(stats.values())} images")
        logger.info(f"Classes: {self.unified_classes}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info("="*60)
        
        return stats
    
    def create_dataset_yaml(self):
        """Tạo file dataset.yaml"""
        yaml_content = {
            'path': str(self.config.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {i: name for i, name in enumerate(self.unified_classes)},
            'nc': len(self.unified_classes)
        }
        
        yaml_path = self.config.output_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Created dataset.yaml: {yaml_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-label classification dataset for detection")
    parser.add_argument("--source", type=str, default="../data/raw/garbage-dataset",
                       help="Source classification dataset directory")
    parser.add_argument("--output", type=str, default="data/detection_from_classification",
                       help="Output detection dataset directory")
    parser.add_argument("--method", type=str, default="center_crop",
                       choices=["full_image", "center_crop", "auto_detect"],
                       help="Labeling method")
    parser.add_argument("--crop-ratio", type=float, default=0.8,
                       help="Crop ratio for center_crop method (0.1-1.0)")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Image size for resizing")
    
    args = parser.parse_args()
    
    config = AutoLabelConfig(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        img_size=(args.img_size, args.img_size),
        crop_ratio=args.crop_ratio
    )
    
    labeler = AutoLabeler(config)
    labeler.process_dataset(method=args.method)
    
    logger.info("\n" + "="*60)
    logger.info("KHUYẾN NGHỊ CHO BOUNDING BOX CHÍNH XÁC:")
    logger.info("="*60)
    logger.info("1. Sử dụng dataset có sẵn annotations (Roboflow Garbage Classification 3)")
    logger.info("   python download_roboflow_dataset.py --api-key YOUR_KEY")
    logger.info("")
    logger.info("2. Hoặc label thủ công với công cụ:")
    logger.info("   - Roboflow: https://roboflow.com (miễn phí, dễ dùng)")
    logger.info("   - LabelImg: pip install labelImg")
    logger.info("   - CVAT: https://cvat.ai")
    logger.info("="*60)


if __name__ == "__main__":
    main()
