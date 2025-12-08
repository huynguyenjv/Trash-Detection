#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing for Object Detection - Trash Detection System

Mô tả:
    Tiền xử lý dataset cho Object Detection:
    - Xử lý dataset TACO (COCO format)
    - Chuyển đổi sang YOLO format
    - Chia train/val/test
    - Tạo file dataset.yaml

Author: Huy Nguyen
Email: huynguyen@example.com
Date: October 2025
Version: 1.0.0
License: MIT
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import random
from dataclasses import dataclass

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
    raw_data_dir: Path = Path("../data/raw")
    processed_data_dir: Path = Path("data/processed")
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Image processing
    img_size: Tuple[int, int] = (640, 640)
    
    # Class mapping (TACO -> Simplified)
    class_mapping: Dict[str, str] = None
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.class_mapping is None:
            self.class_mapping = {
                # Paper
                "Paper": "paper",
                "Cardboard": "cardboard", 
                
                # Plastic  
                "Plastic bag": "plastic",
                "Plastic bottle": "plastic",
                "Plastic container": "plastic",
                "Plastic cup": "plastic",
                "Plastic lid": "plastic",
                "Plastic straw": "plastic",
                "Plastic utensils": "plastic",
                
                # Metal
                "Aluminium foil": "metal",
                "Aluminium blister pack": "metal", 
                "Can": "metal",
                "Metal bottle cap": "metal",
                "Metal lid": "metal",
                
                # Glass
                "Glass bottle": "glass",
                "Glass cup": "glass",
                "Glass jar": "glass",
                
                # Organic
                "Food waste": "organic",
                
                # Other
                "Battery": "other",
                "Cigarette": "other",
                "Shoe": "other",
                "Sock": "other"
            }


class TACODataProcessor:
    """Processor for TACO dataset for trash detection."""
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.detection_dir = processed_data_dir / "detection"
        
        # TACO dataset paths - updated to use existing local dataset structure
        self.data_dir = Path("data/detection/raw/data")  # Relative to current directory
        self.annotations_file = self.data_dir / "annotations.json"
        self.batch_dirs = [self.data_dir / f"batch_{i}" for i in range(1, 16)]  # batch_1 to batch_15
        
        # TACO category mapping to simplified classes
        self.taco_categories = {
            0: "Aluminium foil", 1: "Battery", 2: "Aluminium blister pack", 3: "Carded blister pack",
            4: "Other plastic bottle", 5: "Clear plastic bottle", 6: "Glass bottle", 7: "Plastic bottle cap",
            8: "Metal bottle cap", 9: "Broken glass", 10: "Food Can", 11: "Aerosol", 12: "Drink can",
            13: "Toilet tube", 14: "Other carton", 15: "Egg carton", 16: "Drink carton", 17: "Corrugated carton",
            18: "Meal carton", 19: "Pizza box", 20: "Paper cup", 21: "Disposable plastic cup", 22: "Foam cup",
            23: "Glass cup", 24: "Other plastic cup", 25: "Food waste", 26: "Glass jar", 27: "Plastic lid",
            28: "Metal lid", 29: "Other plastic", 30: "Magazine paper", 31: "Tissues", 32: "Wrapping paper",
            33: "Normal paper", 34: "Paper bag", 35: "Plastified paper bag", 36: "Plastic film",
            37: "Six pack rings", 38: "Garbage bag", 39: "Other plastic wrapper", 40: "Single-use carrier bag",
            41: "Polypropylene bag", 42: "Crisp packet", 43: "Spread tub", 44: "Tupperware",
            45: "Disposable food container", 46: "Foam food container", 47: "Other plastic container",
            48: "Plastic glooves", 49: "Plastic utensils", 50: "Pop tab", 51: "Rope & strings",
            52: "Scrap metal", 53: "Shoe", 54: "Squeezable tube", 55: "Plastic straw", 56: "Paper straw",
            57: "Styrofoam piece", 58: "Unlabeled litter", 59: "Cigarette"
        }
        
        # Simplified class mapping
        self.class_mapping = {
            # Paper & Cardboard
            "Normal paper": "paper", "Magazine paper": "paper", "Tissues": "paper", "Wrapping paper": "paper",
            "Paper cup": "paper", "Paper bag": "paper", "Paper straw": "paper",
            "Toilet tube": "cardboard", "Other carton": "cardboard", "Egg carton": "cardboard",
            "Drink carton": "cardboard", "Corrugated carton": "cardboard", "Meal carton": "cardboard",
            "Pizza box": "cardboard", "Plastified paper bag": "cardboard",
            
            # Plastic
            "Other plastic bottle": "plastic", "Clear plastic bottle": "plastic", "Plastic bottle cap": "plastic",
            "Disposable plastic cup": "plastic", "Other plastic cup": "plastic", "Plastic lid": "plastic",
            "Other plastic": "plastic", "Plastic film": "plastic", "Six pack rings": "plastic",
            "Garbage bag": "plastic", "Other plastic wrapper": "plastic", "Single-use carrier bag": "plastic",
            "Polypropylene bag": "plastic", "Crisp packet": "plastic", "Spread tub": "plastic",
            "Tupperware": "plastic", "Disposable food container": "plastic", "Foam food container": "plastic",
            "Other plastic container": "plastic", "Plastic glooves": "plastic", "Plastic utensils": "plastic",
            "Plastic straw": "plastic", "Squeezable tube": "plastic", "Styrofoam piece": "plastic",
            "Foam cup": "plastic",
            
            # Metal
            "Aluminium foil": "metal", "Aluminium blister pack": "metal", "Metal bottle cap": "metal",
            "Food Can": "metal", "Aerosol": "metal", "Drink can": "metal", "Metal lid": "metal",
            "Pop tab": "metal", "Scrap metal": "metal",
            
            # Glass
            "Glass bottle": "glass", "Glass cup": "glass", "Glass jar": "glass", "Broken glass": "glass",
            
            # Organic
            "Food waste": "organic",
            
            # Other
            "Battery": "other", "Carded blister pack": "other", "Rope & strings": "other",
            "Shoe": "other", "Unlabeled litter": "other", "Cigarette": "other"
        }
        
        # Unified classes
        self.unified_classes = list(set(self.class_mapping.values()))
        self.unified_classes.sort()
    
    def _create_directories(self) -> None:
        """Tạo các thư mục cần thiết"""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.detection_dir / "images" / "train",
            self.detection_dir / "images" / "val", 
            self.detection_dir / "images" / "test",
            self.detection_dir / "labels" / "train",
            self.detection_dir / "labels" / "val",
            self.detection_dir / "labels" / "test",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def check_taco_dataset(self) -> bool:
        """Check if TACO dataset exists locally."""
        try:
            if self.data_dir.exists() and self.annotations_file.exists():
                # Check if batch directories exist
                batch_count = 0
                for batch_dir in self.batch_dirs:
                    if batch_dir.exists():
                        batch_count += 1
                        
                logger.info(f"TACO dataset found at {self.data_dir}")
                logger.info(f"Found {batch_count} batch directories")
                logger.info(f"Annotations file: {self.annotations_file}")
                return True
            else:
                logger.error(f"TACO dataset not found at {self.data_dir}")
                logger.error("Please ensure the dataset is available in the data/detection/raw/data directory")
                return False
            
        except Exception as e:
            logger.error(f"Error checking TACO dataset: {e}")
            return False
    
    def load_annotations(self) -> Tuple[Dict, Dict]:
        """Load and process COCO annotations from TACO dataset"""
        try:
            logger.info(f"Loading annotations from {self.annotations_file}")
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data.get('images', []))} images and {len(data.get('annotations', []))} annotations")
            
            # Process categories from the loaded data
            categories = {}
            simplified_categories = {}
            
            # Use categories from the data if available, otherwise use our predefined mapping
            if 'categories' in data and data['categories']:
                for cat in data['categories']:
                    cat_id = cat['id']
                    cat_name = cat['name']
                    categories[cat_id] = cat_name
                    # Map to simplified class
                    simplified_name = self.class_mapping.get(cat_name, 'other')
                    simplified_categories[cat_id] = simplified_name
                    logger.debug(f"Category {cat_id}: '{cat_name}' -> '{simplified_name}'")
            else:
                # Use predefined TACO categories
                logger.info("Using predefined TACO categories")
                for cat_id, cat_name in self.taco_categories.items():
                    categories[cat_id] = cat_name
                    simplified_name = self.class_mapping.get(cat_name, 'other')
                    simplified_categories[cat_id] = simplified_name
            
            logger.info(f"Total categories: {len(categories)}")
            logger.info(f"Unified classes: {self.unified_classes}")
            
            return data, simplified_categories
            
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return None, None
    
    def get_all_images(self) -> List[str]:
        """Get all image files from batch directories"""
        all_images = []
        try:
            for batch_dir in self.batch_dirs:
                if batch_dir.exists():
                    # Get all image files in the batch directory
                    for img_ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
                        batch_images = list(batch_dir.glob(img_ext))
                        all_images.extend([img.name for img in batch_images])
            
            logger.info(f"Found {len(all_images)} total images across all batches")
            return all_images
            
        except Exception as e:
            logger.error(f"Error getting images: {e}")
            return []

    def process_annotations(self, data: Dict, simplified_categories: Dict) -> Dict:
        """Process COCO annotations to YOLO format"""
        try:
            # Get all available images
            all_image_files = self.get_all_images()
            
            # Group annotations by image if available
            image_annotations = {}
            if 'annotations' in data and data['annotations']:
                for ann in data['annotations']:
                    image_id = ann['image_id']
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    image_annotations[image_id].append(ann)
            
            # Create mapping from image_id to filename if images info is available
            image_id_to_filename = {}
            if 'images' in data and data['images']:
                for img in data['images']:
                    image_id_to_filename[img['id']] = img['file_name']
            
            processed_data = {}
            
            # If we have proper COCO annotations
            if 'images' in data and data['images'] and 'annotations' in data and data['annotations']:
                logger.info("Processing with full COCO annotations")
                for img in data['images']:
                    img_id = img['id']
                    img_filename = img['file_name']
                    
                    # Extract just filename from full path (remove batch_X/ prefix)
                    base_filename = Path(img_filename).name
                    
                    # Skip if image file doesn't exist in our batch directories  
                    if base_filename not in all_image_files:
                        logger.debug(f"Skipping {img_filename} - file not found in batches")
                        continue
                    
                    # Get image annotations
                    annotations = image_annotations.get(img_id, [])
                    
                    # Convert to YOLO format
                    yolo_labels = []
                    for ann in annotations:
                        # Get bounding box from annotation
                        if 'bbox' in ann:
                            x, y, w, h = ann['bbox']  # COCO bbox format: [x, y, width, height]
                        elif 'segmentation' in ann and ann['segmentation']:
                            # Convert segmentation to bbox
                            segmentation = ann['segmentation'][0]  # Take first polygon
                            x_coords = [segmentation[i] for i in range(0, len(segmentation), 2)]
                            y_coords = [segmentation[i] for i in range(1, len(segmentation), 2)]
                            
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            x, y = x_min, y_min
                            w, h = x_max - x_min, y_max - y_min
                        else:
                            continue  # Skip if no bbox or segmentation
                        
                        # Convert to YOLO format (cx, cy, w, h) normalized
                        img_w, img_h = img.get('width', 640), img.get('height', 640)
                        cx = (x + w/2) / img_w
                        cy = (y + h/2) / img_h
                        nw = w / img_w
                        nh = h / img_h
                        
                        # Ensure values are within bounds
                        cx = max(0, min(1, cx))
                        cy = max(0, min(1, cy))
                        nw = max(0, min(1, nw))
                        nh = max(0, min(1, nh))
                        
                        # Get class id
                        category_id = ann['category_id']
                        if category_id in simplified_categories:
                            simplified_name = simplified_categories[category_id]
                            class_id = self.unified_classes.index(simplified_name)
                            yolo_labels.append([class_id, cx, cy, nw, nh])
                    
                    # Use base filename as key for consistency
                    processed_data[base_filename] = {
                        'image_info': img,
                        'labels': yolo_labels,
                        'original_path': img_filename  # Keep original path for reference
                    }
            else:
                # If no annotations available, create entries for all images without labels
                logger.info("No annotations found, creating empty label files for all images")
                for img_filename in all_image_files:
                    processed_data[img_filename] = {
                        'image_info': {'file_name': img_filename, 'width': 640, 'height': 640},
                        'labels': []  # Empty labels
                    }
            
            logger.info(f"Processed {len(processed_data)} images")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing annotations: {e}")
            return {}
    
    def split_dataset(self, processed_data: Dict, train_ratio: float = 0.7, val_ratio: float = 0.2) -> Dict:
        """Split dataset into train/val/test sets"""
        try:
            # Get all image filenames
            all_images = list(processed_data.keys())
            random.seed(42)
            random.shuffle(all_images)
            
            # Calculate split indices
            total = len(all_images)
            train_end = int(total * train_ratio)
            val_end = int(total * (train_ratio + val_ratio))
            
            # Split data
            splits = {
                'train': all_images[:train_end],
                'val': all_images[train_end:val_end],
                'test': all_images[val_end:]
            }
            
            logger.info(f"Dataset split - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
            return splits
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            return {}
    
    def find_image_path(self, img_filename: str) -> Optional[Path]:
        """Find the full path of an image file in batch directories"""
        for batch_dir in self.batch_dirs:
            if batch_dir.exists():
                img_path = batch_dir / img_filename
                if img_path.exists():
                    return img_path
        return None

    def copy_files_to_splits(self, processed_data: Dict, splits: Dict, img_size: Tuple[int, int] = (640, 640)) -> bool:
        """Copy and process images and labels to appropriate split directories"""
        try:
            stats = {'train': 0, 'val': 0, 'test': 0}
            
            for split_name, image_list in splits.items():
                logger.info(f"Processing {split_name} split: {len(image_list)} images")
                
                # Create directories for this split
                images_dir = self.detection_dir / "images" / split_name
                labels_dir = self.detection_dir / "labels" / split_name
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)
                
                for img_filename in image_list:
                    try:
                        # Find image in batch directories
                        src_img_path = self.find_image_path(img_filename)
                        
                        if src_img_path and src_img_path.exists():
                            dst_img_path = images_dir / img_filename
                            
                            # Read and resize image
                            img = cv2.imread(str(src_img_path))
                            if img is not None:
                                img_resized = cv2.resize(img, img_size)
                                cv2.imwrite(str(dst_img_path), img_resized)
                                
                                # Create label file
                                label_filename = img_filename.rsplit('.', 1)[0] + '.txt'  # Handle both .jpg and .JPG
                                label_path = labels_dir / label_filename
                                
                                labels = processed_data[img_filename]['labels']
                                with open(label_path, 'w') as f:
                                    for label in labels:
                                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                                
                                stats[split_name] += 1
                            else:
                                logger.warning(f"Could not read image: {src_img_path}")
                        else:
                            logger.warning(f"Image not found: {img_filename}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {img_filename}: {e}")
                        continue
                
                logger.info(f"Completed {split_name}: {stats[split_name]} images processed")
            
            logger.info(f"Total images processed: {sum(stats.values())}")
            return sum(stats.values()) > 0
            
        except Exception as e:
            logger.error(f"Error copying files: {e}")
            return False
    
    def create_dataset_yaml(self) -> bool:
        """Create dataset.yaml file for YOLOv8 training"""
        try:
            yaml_content = {
                'path': str(self.detection_dir.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'names': {i: name for i, name in enumerate(self.unified_classes)}
            }
            
            yaml_path = self.detection_dir / "dataset.yaml"
            
            # Write YAML manually to ensure proper format
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(f"# TACO Dataset for Trash Detection\n")
                f.write(f"path: {yaml_content['path']}\n")
                f.write(f"train: {yaml_content['train']}\n")
                f.write(f"val: {yaml_content['val']}\n")
                f.write(f"test: {yaml_content['test']}\n")
                f.write(f"\n# Classes\n")
                f.write(f"nc: {len(yaml_content['names'])}\n")
                f.write(f"names:\n")
                for i, name in yaml_content['names'].items():
                    f.write(f"  {i}: {name}\n")
            
            logger.info(f"Dataset YAML created: {yaml_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dataset YAML: {e}")
            return False
    
    def validate_dataset(self) -> bool:
        """Validate the processed dataset"""
        try:
            validation_results = {
                'splits': {},
                'total_images': 0,
                'total_labels': 0,
                'classes': self.unified_classes
            }
            
            for split in ['train', 'val', 'test']:
                images_dir = self.detection_dir / "images" / split
                labels_dir = self.detection_dir / "labels" / split
                
                if not images_dir.exists() or not labels_dir.exists():
                    logger.error(f"Split directories not found for: {split}")
                    return False
                
                image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
                label_count = len(list(labels_dir.glob("*.txt")))
                
                validation_results['splits'][split] = {
                    'images': image_count,
                    'labels': label_count
                }
                validation_results['total_images'] += image_count
                validation_results['total_labels'] += label_count
                
                logger.info(f"Split '{split}': {image_count} images, {label_count} labels")
            
            # Save validation results
            validation_file = self.detection_dir / "validation_results.json"
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset validation completed. Total: {validation_results['total_images']} images")
            logger.info(f"Validation results saved to {validation_file}")
            
            return validation_results['total_images'] > 0
            
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False
    
    def run_preprocessing(self) -> bool:
        """Run the complete preprocessing pipeline"""
        try:
            logger.info("=== BẮT ĐẦU PREPROCESSING DETECTION DATASET ===")
            
            # Step 1: Check dataset availability
            if not self.check_taco_dataset():
                return False
            
            # Step 2: Create directories
            self._create_directories()
            
            # Step 3: Load annotations
            logger.info("Loading COCO annotations...")
            data, simplified_categories = self.load_annotations()
            if data is None:
                return False
            
            # Step 4: Process annotations to YOLO format
            logger.info("Converting annotations to YOLO format...")
            processed_data = self.process_annotations(data, simplified_categories)
            if not processed_data:
                return False
            
            # Step 5: Split dataset
            logger.info("Splitting dataset...")
            splits = self.split_dataset(processed_data)
            if not splits:
                return False
            
            # Step 6: Copy files to split directories  
            logger.info("Copying files to split directories...")
            if not self.copy_files_to_splits(processed_data, splits):
                return False
            
            # Step 7: Create dataset YAML
            logger.info("Creating dataset.yaml...")
            if not self.create_dataset_yaml():
                return False
            
            # Step 8: Validate dataset
            logger.info("Validating processed dataset...")
            if not self.validate_dataset():
                return False
            
            logger.info("=== DETECTION DATASET PREPROCESSING HOÀN THÀNH! ===")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình preprocessing: {e}")
            return False


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess TACO dataset for trash detection")
    parser.add_argument("--raw-data-dir", type=str, default="data/detection/raw",
                        help="Path to raw data directory")
    parser.add_argument("--processed-data-dir", type=str, default="data/processed", 
                        help="Path to processed data directory")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = TACODataProcessor(Path(args.raw_data_dir), Path(args.processed_data_dir))
    
    # Run preprocessing
    success = processor.run_preprocessing()
    
    if success:
        logger.info("Detection data preprocessing completed successfully!")
        logger.info(f"Processed data saved to: {processor.detection_dir}")
        logger.info("You can now use this dataset for YOLOv8 training!")
        return 0
    else:
        logger.error("Detection data preprocessing failed!")
        return 1


if __name__ == "__main__":
    exit(main())