#!/usr/bin/env python3
"""
Data Preprocessing for Trash Detection
Xử lý dataset TACO để huấn luyện mô hình detection
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
        
        # TACO dataset paths - updated to use existing local dataset
        self.taco_dir = raw_data_dir / "taco-dataset"
        self.annotations_file = self.taco_dir / "annotations" / "instances.json"  # Correct path
        self.images_dir = self.taco_dir / "images"
        
        # Default class mapping
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
            if self.taco_dir.exists() and self.annotations_file.exists():
                logger.info(f"TACO dataset found at {self.taco_dir}")
                return True
            else:
                logger.error(f"TACO dataset not found at {self.taco_dir}")
                logger.error("Please ensure the dataset is available in the raw data directory")
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
            
            logger.info(f"Loaded {len(data['images'])} images and {len(data['annotations'])} annotations")
            
            # Process categories
            categories = {}
            simplified_categories = {}
            
            for cat in data['categories']:
                categories[cat['id']] = cat['name']
                # Map to simplified class
                simplified_name = self.class_mapping.get(cat['name'], 'other')
                simplified_categories[cat['id']] = simplified_name
                logger.info(f"Category {cat['id']}: '{cat['name']}' -> '{simplified_name}'")
            
            logger.info(f"Total categories: {len(categories)}")
            logger.info(f"Unified classes: {self.unified_classes}")
            
            return data, simplified_categories
            
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return None, None
    
    def process_annotations(self, data: Dict, simplified_categories: Dict) -> Dict:
        """Process COCO annotations to YOLO format"""
        try:
            # Group annotations by image
            image_annotations = {}
            for ann in data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)
            
            # Process each image
            processed_data = {}
            for img in data['images']:
                img_id = img['id']
                
                # Get image annotations
                annotations = image_annotations.get(img_id, [])
                if not annotations:
                    continue
                
                # Convert to YOLO format
                yolo_labels = []
                for ann in annotations:
                    # Get segmentation and convert to bbox
                    if 'segmentation' in ann and ann['segmentation']:
                        segmentation = ann['segmentation'][0]  # Take first polygon
                        
                        # Convert segmentation to bbox
                        x_coords = [segmentation[i] for i in range(0, len(segmentation), 2)]
                        y_coords = [segmentation[i] for i in range(1, len(segmentation), 2)]
                        
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        x, y = x_min, y_min
                        w, h = x_max - x_min, y_max - y_min
                        
                        # Convert to YOLO format (cx, cy, w, h) normalized
                        img_w, img_h = img['width'], img['height']
                        cx = (x + w/2) / img_w
                        cy = (y + h/2) / img_h
                        nw = w / img_w
                        nh = h / img_h
                    else:
                        # Skip annotation if no segmentation
                        continue
                    
                    # Get class id (using simplified categories)
                    category_id = ann['category_id']
                    original_name = data['categories'][category_id - 1]['name']  # COCO IDs start from 1
                    simplified_name = self.class_mapping.get(original_name, 'other')
                    class_id = self.unified_classes.index(simplified_name)
                    
                    yolo_labels.append([class_id, cx, cy, nw, nh])
                
                processed_data[img['file_name']] = {
                    'image_info': img,
                    'labels': yolo_labels
                }
            
            logger.info(f"Processed {len(processed_data)} images with annotations")
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
                        # Copy and resize image
                        src_img_path = self.images_dir / img_filename
                        dst_img_path = images_dir / img_filename
                        
                        if src_img_path.exists():
                            # Read and resize image
                            img = cv2.imread(str(src_img_path))
                            if img is not None:
                                img_resized = cv2.resize(img, img_size)
                                cv2.imwrite(str(dst_img_path), img_resized)
                                
                                # Create label file
                                label_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
                                label_path = labels_dir / label_filename
                                
                                labels = processed_data[img_filename]['labels']
                                with open(label_path, 'w') as f:
                                    for label in labels:
                                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                                
                                stats[split_name] += 1
                            else:
                                logger.warning(f"Could not read image: {src_img_path}")
                        else:
                            logger.warning(f"Image not found: {src_img_path}")
                            
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
    parser.add_argument("--raw-data-dir", type=str, default="../data/raw",
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
        return 0
    else:
        logger.error("Detection data preprocessing failed!")
        return 1


if __name__ == "__main__":
    exit(main())