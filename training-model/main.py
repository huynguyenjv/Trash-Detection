#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Training Pipeline - Trash Detection System

Mô tả:
    Pipeline huấn luyện tích hợp cho hệ thống phát hiện rác thải:
    - Data Preprocessing (sử dụng external scripts)
    - Detection Model Training (YOLOv8)
    - Classification Model Training
    - Comprehensive Evaluation
    - Real-time Detection Pipeline
    - Lazy loading cho dependencies

Author: Huy Nguyen
Email: huynguyen@example.com
Date: October 2025
Version: 1.0.0
License: MIT
"""

import os
import sys
import logging
import argparse
import yaml
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def lazy_import_ultralytics():
    """Lazy import ultralytics khi cần thiết"""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as e:
        logger.error(f"ultralytics không được cài đặt: {e}")
        logger.info("Chạy: pip install ultralytics để cài đặt")
        return None


def lazy_import_torch():
    """Lazy import torch khi cần thiết"""
    try:
        import torch
        
        # Fix PyTorch 2.6 weights_only issue  
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load
        
        return torch
    except ImportError as e:
        logger.error(f"torch không được cài đặt: {e}")
        logger.info("Chạy: pip install torch để cài đặt")
        return None


def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    required_packages = ['ultralytics', 'torch', 'yaml', 'pathlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'ultralytics':
                lazy_import_ultralytics()
            elif package == 'torch':
                lazy_import_torch()
            elif package == 'yaml':
                import yaml
            elif package == 'pathlib':
                from pathlib import Path
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Thiếu packages: {missing_packages}")
        logger.info("Chạy: pip install " + " ".join(missing_packages))
        return False
    
    return True


# ==================== CONFIGURATIONS ====================

@dataclass
class DetectionTrainingConfig:
    """Cấu hình cho training detection model"""
    # Model config
    model_name: str = "yolov8n.pt"
    pretrained: bool = True
    
    # Dataset
    data_yaml: str = "data/processed/detection/dataset.yaml"
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.01
    device: str = "auto"
    
    # Output paths
    save_dir: Path = Path("results/detection")
    experiment_name: str = "detection_v1"
    
    def __post_init__(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ClassificationTrainingConfig:
    """Cấu hình cho training classification model"""
    # Model config
    model_name: str = "yolov8n-cls.pt"
    pretrained: bool = True
    
    # Dataset
    data_yaml: str = "data/processed/classification"
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 32
    img_size: int = 224
    learning_rate: float = 0.001
    device: str = "auto"
    
    # Output paths
    save_dir: Path = Path("results/classification")
    experiment_name: str = "classification_v1"
    
    def __post_init__(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationConfig:
    """Cấu hình cho evaluation"""
    # Model paths
    detection_model_path: str = "models/detection/best.pt"
    classification_model_path: str = "models/classification/best.pt"
    
    # Dataset paths
    detection_data_yaml: str = "data/detection/processed/dataset.yaml"
    classification_data_yaml: str = "data/classification/processed/dataset.yaml"
    
    # Evaluation settings
    detection_conf_thresholds: List[float] = None
    device: str = "auto"
    
    # Output
    results_dir: Path = Path("results/evaluation")
    experiment_name: str = "evaluation_v1"
    
    def __post_init__(self):
        if self.detection_conf_thresholds is None:
            self.detection_conf_thresholds = [0.1, 0.25, 0.5, 0.75]
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """Cấu hình cho detection pipeline"""
    # Model paths
    detection_model_path: str = "models/detection/best.pt"
    classification_model_path: str = "models/classification/best.pt"
    
    # Detection settings
    detection_conf_threshold: float = 0.25
    detection_iou_threshold: float = 0.45
    device: str = "auto"


# ==================== TRAINING CLASSES ====================

class DetectionTrainer:
    """Class để training detection model"""
    
    def __init__(self, config: DetectionTrainingConfig):
        self.config = config
        self.model = None
        self.training_results = {}
        self.validation_results = {}
        
        logger.info(f"Initialized DetectionTrainer: {self.config.experiment_name}")
    
    def setup_model(self):
        """Khởi tạo YOLO model"""
        try:
            YOLO = lazy_import_ultralytics()
            if YOLO is None:
                raise ImportError("ultralytics không available")
            
            logger.info(f"Loading model: {self.config.model_name}")
            model = YOLO(self.config.model_name, verbose=True)
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def train_model(self) -> Dict[str, Any]:
        """Train detection model"""
        try:
            logger.info("=== STARTING DETECTION MODEL TRAINING ===")
            
            # Check if dataset exists
            if not Path(self.config.data_yaml).exists():
                raise FileNotFoundError(f"Dataset yaml not found: {self.config.data_yaml}")
            
            # Setup model
            self.model = self.setup_model()
            
            # Setup training arguments
            train_args = {
                'data': str(self.config.data_yaml),
                'epochs': self.config.epochs,
                'batch': self.config.batch_size,
                'imgsz': self.config.img_size,
                'lr0': self.config.learning_rate,
                'device': self.config.device,
                'project': str(self.config.save_dir),
                'name': self.config.experiment_name,
                'verbose': True,
                'plots': True,
            }
            
            logger.info(f"Training arguments: {train_args}")
            
            # Start training
            start_time = time.time()
            results = self.model.train(**train_args)
            training_time = time.time() - start_time
            
            # Save training results
            self.training_results = {
                'training_time': training_time,
                'model_path': str(results.save_dir / 'weights' / 'best.pt') if hasattr(results, 'save_dir') else None,
                'last_model_path': str(results.save_dir / 'weights' / 'last.pt') if hasattr(results, 'save_dir') else None,
                'results_dir': str(results.save_dir) if hasattr(results, 'save_dir') else None
            }
            
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Best model saved at: {self.training_results.get('model_path')}")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def validate_model(self) -> Dict[str, Any]:
        """Validate trained model"""
        try:
            logger.info("=== VALIDATING DETECTION MODEL ===")
            
            if self.model is None:
                YOLO = lazy_import_ultralytics()
                if YOLO is None:
                    raise ImportError("ultralytics không available")
                
                best_model_path = self.training_results.get('model_path')
                if best_model_path and Path(best_model_path).exists():
                    self.model = YOLO(best_model_path)
                else:
                    raise ValueError("No trained model available for validation")
            
            # Run validation
            val_results = self.model.val(
                data=str(self.config.data_yaml),
                device=self.config.device,
                verbose=True
            )
            
            # Extract key metrics
            self.validation_results = {
                'mAP50': float(val_results.box.map50) if hasattr(val_results, 'box') else 0,
                'mAP50-95': float(val_results.box.map) if hasattr(val_results, 'box') else 0,
                'precision': float(val_results.box.mp) if hasattr(val_results, 'box') else 0,
                'recall': float(val_results.box.mr) if hasattr(val_results, 'box') else 0,
            }
            
            logger.info("Validation Results:")
            for metric, value in self.validation_results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    def run_full_training(self) -> Dict[str, Any]:
        """Chạy toàn bộ training pipeline"""
        try:
            logger.info("=== RUNNING FULL DETECTION TRAINING PIPELINE ===")
            
            # Train model
            training_results = self.train_model()
            
            # Validate model
            validation_results = self.validate_model()
            
            # Combine results
            full_results = {
                'training_results': training_results,
                'validation_results': validation_results,
                'model_paths': {
                    'best': training_results.get('model_path'),
                    'last': training_results.get('last_model_path')
                },
                'summary': {
                    'mAP50': validation_results.get('mAP50', 0),
                    'precision': validation_results.get('precision', 0),
                    'recall': validation_results.get('recall', 0),
                    'training_time': training_results.get('training_time', 0)
                }
            }
            
            logger.info("=== DETECTION TRAINING PIPELINE COMPLETED ===")
            return full_results
            
        except Exception as e:
            logger.error(f"Error in full training pipeline: {e}")
            raise


class ClassificationTrainer:
    """Class để training classification model"""
    
    def __init__(self, config: ClassificationTrainingConfig):
        self.config = config
        self.model = None
        self.training_results = {}
        self.validation_results = {}
        
        logger.info(f"Initialized ClassificationTrainer: {self.config.experiment_name}")
    
    def setup_model(self):
        """Khởi tạo YOLO classification model"""
        try:
            YOLO = lazy_import_ultralytics()
            if YOLO is None:
                raise ImportError("ultralytics không available")
                
            logger.info(f"Loading classification model: {self.config.model_name}")
            model = YOLO(self.config.model_name, verbose=True)
            
            logger.info("Classification model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            raise
    
    def train_model(self) -> Dict[str, Any]:
        """Train classification model"""
        try:
            logger.info("=== STARTING CLASSIFICATION MODEL TRAINING ===")
            
            # Check if dataset exists
            if not Path(self.config.data_yaml).exists():
                raise FileNotFoundError(f"Dataset path not found: {self.config.data_yaml}")
            
            # Setup model
            self.model = self.setup_model()
            
            # Setup training arguments
            train_args = {
                'data': str(self.config.data_yaml),
                'epochs': self.config.epochs,
                'batch': self.config.batch_size,
                'imgsz': self.config.img_size,
                'lr0': self.config.learning_rate,
                'device': self.config.device,
                'project': str(self.config.save_dir),
                'name': self.config.experiment_name,
                'verbose': True,
                'plots': True,
            }
            
            logger.info(f"Classification training arguments: {train_args}")
            
            # Start training
            start_time = time.time()
            results = self.model.train(**train_args)
            training_time = time.time() - start_time
            
            # Save training results
            self.training_results = {
                'training_time': training_time,
                'model_path': str(results.save_dir / 'weights' / 'best.pt') if hasattr(results, 'save_dir') else None,
                'last_model_path': str(results.save_dir / 'weights' / 'last.pt') if hasattr(results, 'save_dir') else None,
                'results_dir': str(results.save_dir) if hasattr(results, 'save_dir') else None
            }
            
            logger.info(f"Classification training completed in {training_time:.2f}s")
            return self.training_results
            
        except Exception as e:
            logger.error(f"Error during classification training: {e}")
            raise
    
    def validate_model(self) -> Dict[str, Any]:
        """Validate trained classification model"""
        try:
            logger.info("=== VALIDATING CLASSIFICATION MODEL ===")
            
            if self.model is None:
                YOLO = lazy_import_ultralytics()
                if YOLO is None:
                    raise ImportError("ultralytics không available")
                
                best_model_path = self.training_results.get('model_path')
                if best_model_path and Path(best_model_path).exists():
                    self.model = YOLO(best_model_path)
                else:
                    raise ValueError("No trained classification model available for validation")
            
            # Run validation
            val_results = self.model.val(
                data=str(self.config.data_yaml),
                device=self.config.device,
                verbose=True
            )
            
            # Extract key metrics for classification
            self.validation_results = {
                'top1_accuracy': float(val_results.top1) if hasattr(val_results, 'top1') else 0,
                'top5_accuracy': float(val_results.top5) if hasattr(val_results, 'top5') else 0,
            }
            
            logger.info("Classification Validation Results:")
            for metric, value in self.validation_results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Error during classification validation: {e}")
            raise
    
    def run_full_training(self) -> Dict[str, Any]:
        """Chạy toàn bộ classification training pipeline"""
        try:
            logger.info("=== RUNNING FULL CLASSIFICATION TRAINING PIPELINE ===")
            
            # Train model
            training_results = self.train_model()
            
            # Validate model
            validation_results = self.validate_model()
            
            # Combine results
            full_results = {
                'training_results': training_results,
                'validation_results': validation_results,
                'model_paths': {
                    'best': training_results.get('model_path'),
                    'last': training_results.get('last_model_path')
                },
                'summary': {
                    'top1_accuracy': validation_results.get('top1_accuracy', 0),
                    'top5_accuracy': validation_results.get('top5_accuracy', 0),
                    'training_time': training_results.get('training_time', 0)
                }
            }
            
            logger.info("=== CLASSIFICATION TRAINING PIPELINE COMPLETED ===")
            return full_results
            
        except Exception as e:
            logger.error(f"Error in full classification training pipeline: {e}")
            raise


# ==================== EVALUATION CLASS ====================

class ComprehensiveEvaluator:
    """Comprehensive evaluation cho cả detection và classification models"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {}
        
        logger.info(f"Initialized ComprehensiveEvaluator: {self.config.experiment_name}")
    
    def evaluate_detection_model(self) -> Dict[str, Any]:
        """Evaluate detection model"""
        try:
            logger.info("=== EVALUATING DETECTION MODEL ===")
            
            YOLO = lazy_import_ultralytics()
            if YOLO is None:
                raise ImportError("ultralytics không available")
            
            if not Path(self.config.detection_model_path).exists():
                raise FileNotFoundError(f"Detection model not found: {self.config.detection_model_path}")
            
            # Load model
            model = YOLO(self.config.detection_model_path)
            
            # Run validation với default threshold
            val_results = model.val(
                data=str(self.config.detection_data_yaml),
                device=self.config.device,
                verbose=True
            )
            
            # Extract metrics
            detection_results = {
                'mAP50': float(val_results.box.map50) if hasattr(val_results, 'box') else 0,
                'mAP50-95': float(val_results.box.map) if hasattr(val_results, 'box') else 0,
                'precision': float(val_results.box.mp) if hasattr(val_results, 'box') else 0,
                'recall': float(val_results.box.mr) if hasattr(val_results, 'box') else 0,
                'model_path': self.config.detection_model_path
            }
            
            logger.info(f"Detection mAP@50: {detection_results['mAP50']:.4f}")
            return detection_results
            
        except Exception as e:
            logger.error(f"Error evaluating detection model: {e}")
            raise
    
    def evaluate_classification_model(self) -> Dict[str, Any]:
        """Evaluate classification model"""
        try:
            logger.info("=== EVALUATING CLASSIFICATION MODEL ===")
            
            YOLO = lazy_import_ultralytics()
            if YOLO is None:
                raise ImportError("ultralytics không available")
            
            if not Path(self.config.classification_model_path).exists():
                raise FileNotFoundError(f"Classification model not found: {self.config.classification_model_path}")
            
            # Load model
            model = YOLO(self.config.classification_model_path)
            
            # Run validation
            val_results = model.val(
                data=str(self.config.classification_data_yaml),
                device=self.config.device,
                verbose=True
            )
            
            # Extract metrics
            classification_results = {
                'top1_accuracy': float(val_results.top1) if hasattr(val_results, 'top1') else 0,
                'top5_accuracy': float(val_results.top5) if hasattr(val_results, 'top5') else 0,
                'model_path': self.config.classification_model_path
            }
            
            logger.info(f"Classification Top-1 Accuracy: {classification_results['top1_accuracy']:.4f}")
            return classification_results
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            raise
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Chạy toàn bộ evaluation pipeline"""
        try:
            logger.info("=== RUNNING COMPREHENSIVE EVALUATION ===")
            
            # Evaluate detection model
            detection_results = self.evaluate_detection_model()
            
            # Evaluate classification model  
            classification_results = self.evaluate_classification_model()
            
            # Combine results
            self.results = {
                'detection_evaluation': detection_results,
                'classification_evaluation': classification_results,
                'evaluation_config': {
                    'detection_model': self.config.detection_model_path,
                    'classification_model': self.config.classification_model_path,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Save results
            self.save_results()
            
            logger.info("=== COMPREHENSIVE EVALUATION COMPLETED ===")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            raise
    
    def save_results(self):
        """Save evaluation results"""
        try:
            results_path = self.config.results_dir / f"{self.config.experiment_name}_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")


# ==================== DETECTION PIPELINE ====================

class TrashDetectionPipeline:
    """Real-time trash detection pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detection_model = None
        self.classification_model = None
        
        # Setup models
        self._setup_models()
        
        logger.info("TrashDetectionPipeline initialized successfully")
    
    def _setup_models(self):
        """Setup detection và classification models"""
        try:
            YOLO = lazy_import_ultralytics()
            if YOLO is None:
                logger.warning("ultralytics không available, pipeline sẽ không hoạt động")
                return
                
            # Load detection model
            if Path(self.config.detection_model_path).exists():
                self.detection_model = YOLO(self.config.detection_model_path)
                logger.info(f"Detection model loaded: {self.config.detection_model_path}")
            else:
                logger.warning(f"Detection model not found: {self.config.detection_model_path}")
            
            # Load classification model
            if Path(self.config.classification_model_path).exists():
                self.classification_model = YOLO(self.config.classification_model_path)
                logger.info(f"Classification model loaded: {self.config.classification_model_path}")
            else:
                logger.warning(f"Classification model not found: {self.config.classification_model_path}")
        
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process single image"""
        try:
            if self.detection_model is None:
                raise RuntimeError("Detection model not loaded")
            
            # Run detection
            detection_results = self.detection_model(
                image_path,
                conf=self.config.detection_conf_threshold,
                iou=self.config.detection_iou_threshold,
                device=self.config.device,
                verbose=False
            )
            
            # Count detections
            total_objects = 0
            for result in detection_results:
                if result.boxes is not None:
                    total_objects += len(result.boxes)
            
            return {
                'detections': total_objects,
                'summary': {
                    'total_objects': total_objects,
                    'classified_objects': 0  # Placeholder
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise


# ==================== MAIN PIPELINE CLASS ====================

class TrashDetectionTrainingPipeline:
    """Main class để orchestrate toàn bộ training pipeline"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        
        # Load configuration
        self._load_config()
        
        # Setup directories
        self._setup_directories()
        
        logger.info("Training pipeline initialized")
    
    def _load_config(self) -> None:
        """Load configuration từ YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_directories(self) -> None:
        """Setup các directories cần thiết"""
        try:
            directories = [
                'data/detection/raw',
                'data/detection/processed',
                'data/classification/raw', 
                'data/classification/processed',
                'models/detection',
                'models/classification',
                'results/detection',
                'results/classification',
                'results/evaluation',
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            logger.info("Directories setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up directories: {e}")
            raise
    
    def run_data_preprocessing(self) -> Dict[str, Any]:
        """Step 1: Data preprocessing (sử dụng external scripts)"""
        try:
            logger.info("=== STEP 1: DATA PREPROCESSING ===")
            
            preprocessing_results = {
                'detection': {'status': 'use_external_script', 'script': 'data_preprocessing_detection.py'},
                'classification': {'status': 'use_external_script', 'script': 'data_preprocessing_classification.py'}
            }
            
            logger.info("Data preprocessing requires external scripts:")
            logger.info("  - Detection: python data_preprocessing_detection.py")
            logger.info("  - Classification: python data_preprocessing_classification.py")
            
            self.results['preprocessing'] = preprocessing_results
            return preprocessing_results
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def run_detection_training(self) -> Dict[str, Any]:
        """Step 2: Train detection model"""
        try:
            logger.info("=== STEP 2: DETECTION MODEL TRAINING ===")
            
            # Load detection config
            det_config = self.config.get('detection', {})
            
            # Create training config
            training_config = DetectionTrainingConfig(
                model_name=det_config.get('model_name', 'yolov8n.pt'),
                data_yaml=det_config.get('data_yaml', 'data/processed/detection/dataset.yaml'),
                epochs=det_config.get('epochs', 100),
                batch_size=det_config.get('batch_size', 16),
                img_size=det_config.get('img_size', 640),
                learning_rate=det_config.get('learning_rate', 0.01),
                device=det_config.get('device', 'auto'),
                save_dir=Path(det_config.get('save_dir', 'results/detection')),
                experiment_name=det_config.get('experiment_name', 'detection_v1'),
            )
            
            # Train model
            trainer = DetectionTrainer(training_config)
            detection_results = trainer.run_full_training()
            
            self.results['detection_training'] = detection_results
            
            logger.info("=== DETECTION TRAINING COMPLETED ===")
            return detection_results
            
        except Exception as e:
            logger.error(f"Error in detection training: {e}")
            raise
    
    def run_classification_training(self) -> Dict[str, Any]:
        """Step 3: Train classification model"""
        try:
            logger.info("=== STEP 3: CLASSIFICATION MODEL TRAINING ===")
            
            # Load classification config
            cls_config = self.config.get('classification', {})
            
            # Create training config
            training_config = ClassificationTrainingConfig(
                model_name=cls_config.get('model_name', 'yolov8n-cls.pt'),
                data_yaml=cls_config.get('data_yaml', 'data/processed/classification'),
                epochs=cls_config.get('epochs', 50),
                batch_size=cls_config.get('batch_size', 32),
                img_size=cls_config.get('img_size', 224),
                learning_rate=cls_config.get('learning_rate', 0.001),
                device=cls_config.get('device', 'auto'),
                save_dir=Path(cls_config.get('save_dir', 'results/classification')),
                experiment_name=cls_config.get('experiment_name', 'classification_v1'),
            )
            
            # Train model
            trainer = ClassificationTrainer(training_config)
            classification_results = trainer.run_full_training()
            
            self.results['classification_training'] = classification_results
            
            logger.info("=== CLASSIFICATION TRAINING COMPLETED ===")
            return classification_results
            
        except Exception as e:
            logger.error(f"Error in classification training: {e}")
            raise
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Step 4: Comprehensive evaluation"""
        try:
            logger.info("=== STEP 4: COMPREHENSIVE EVALUATION ===")
            
            # Load evaluation config
            eval_config = self.config.get('evaluation', {})
            
            # Update model paths từ training results
            detection_model_path = eval_config.get('detection_model_path', 'models/detection/best.pt')
            classification_model_path = eval_config.get('classification_model_path', 'models/classification/best.pt')
            
            if 'detection_training' in self.results:
                detection_model_path = self.results['detection_training']['model_paths']['best']
            
            if 'classification_training' in self.results:
                classification_model_path = self.results['classification_training']['model_paths']['best']
            
            # Create evaluation config
            evaluation_config = EvaluationConfig(
                detection_model_path=detection_model_path,
                classification_model_path=classification_model_path,
                detection_data_yaml=eval_config.get('detection_data_yaml', 'data/detection/processed/dataset.yaml'),
                classification_data_yaml=eval_config.get('classification_data_yaml', 'data/classification/processed'),
                device=eval_config.get('device', 'auto'),
                results_dir=Path(eval_config.get('results_dir', 'results/evaluation')),
                experiment_name=eval_config.get('experiment_name', 'evaluation_v1'),
            )
            
            # Run evaluation
            evaluator = ComprehensiveEvaluator(evaluation_config)
            evaluation_results = evaluator.run_full_evaluation()
            
            self.results['evaluation'] = evaluation_results
            
            logger.info("=== COMPREHENSIVE EVALUATION COMPLETED ===")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            raise
    
    def run_pipeline_integration_test(self) -> Dict[str, Any]:
        """Step 5: Test pipeline integration"""
        try:
            logger.info("=== STEP 5: PIPELINE INTEGRATION TEST ===")
            
            # Load pipeline config
            pipe_config = self.config.get('pipeline', {})
            
            # Update model paths từ training results
            detection_model_path = pipe_config.get('detection_model_path', 'models/detection/best.pt')
            classification_model_path = pipe_config.get('classification_model_path', 'models/classification/best.pt')
            
            if 'detection_training' in self.results:
                detection_model_path = self.results['detection_training']['model_paths']['best']
            
            if 'classification_training' in self.results:
                classification_model_path = self.results['classification_training']['model_paths']['best']
            
            # Create pipeline config
            pipeline_config = PipelineConfig(
                detection_model_path=detection_model_path,
                classification_model_path=classification_model_path,
                detection_conf_threshold=pipe_config.get('detection_conf_threshold', 0.25),
                detection_iou_threshold=pipe_config.get('detection_iou_threshold', 0.45),
                device=pipe_config.get('device', 'auto'),
            )
            
            # Initialize pipeline
            pipeline = TrashDetectionPipeline(pipeline_config)
            
            integration_results = {
                'pipeline_initialized': True,
                'detection_model_loaded': pipeline.detection_model is not None,
                'classification_model_loaded': pipeline.classification_model is not None,
                'success_rate': 1.0 if pipeline.detection_model is not None else 0.0
            }
            
            self.results['integration_test'] = integration_results
            
            logger.info("=== PIPELINE INTEGRATION TEST COMPLETED ===")
            return integration_results
            
        except Exception as e:
            logger.error(f"Error in pipeline integration test: {e}")
            raise
    
    def save_pipeline_results(self) -> None:
        """Lưu kết quả toàn bộ pipeline"""
        try:
            # Create final results
            final_results = {
                'pipeline_info': {
                    'config_path': self.config_path,
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0.0-lightweight'
                },
                'results': self.results,
                'summary': self._create_pipeline_summary()
            }
            
            # Save to JSON
            results_path = Path('results') / 'pipeline_results.json'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Pipeline results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
    
    def _create_pipeline_summary(self) -> Dict[str, Any]:
        """Tạo summary của toàn bộ pipeline"""
        summary = {
            'preprocessing_completed': 'preprocessing' in self.results,
            'detection_training_completed': 'detection_training' in self.results,
            'classification_training_completed': 'classification_training' in self.results,
            'evaluation_completed': 'evaluation' in self.results,
            'integration_test_completed': 'integration_test' in self.results,
        }
        
        # Add key metrics nếu có
        if 'detection_training' in self.results:
            det_validation = self.results['detection_training'].get('validation_results', {})
            summary['detection_mAP50'] = det_validation.get('mAP50', 0)
        
        if 'classification_training' in self.results:
            cls_validation = self.results['classification_training'].get('validation_results', {})
            summary['classification_accuracy'] = cls_validation.get('top1_accuracy', 0)
        
        return summary
    
    def run_full_pipeline(self, steps: Optional[str] = None) -> Dict[str, Any]:
        """Chạy toàn bộ training pipeline"""
        try:
            logger.info("=== STARTING FULL TRAINING PIPELINE ===")
            
            # Check dependencies
            if not check_dependencies():
                logger.error("Missing required dependencies. Please install them first.")
                return {'error': 'missing_dependencies'}
            
            # Xác định steps cần chạy
            if steps:
                selected_steps = [s.strip() for s in steps.split(',')]
            else:
                selected_steps = ['preprocessing', 'detection', 'classification', 'evaluation', 'integration']
            
            logger.info(f"Selected steps: {selected_steps}")
            
            # Run từng step
            if 'preprocessing' in selected_steps:
                self.run_data_preprocessing()
            
            if 'detection' in selected_steps:
                self.run_detection_training()
            
            if 'classification' in selected_steps:
                self.run_classification_training()
            
            if 'evaluation' in selected_steps:
                self.run_comprehensive_evaluation()
            
            if 'integration' in selected_steps:
                self.run_pipeline_integration_test()
            
            # Save results
            self.save_pipeline_results()
            
            # Print final summary
            self.print_final_summary()
            
            logger.info("=== FULL TRAINING PIPELINE COMPLETED ===")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {e}")
            # Save partial results
            try:
                self.save_pipeline_results()
            except:
                pass
            raise
    
    def print_final_summary(self) -> None:
        """In tóm tắt cuối cùng"""
        try:
            summary = self._create_pipeline_summary()
            
            print("\n" + "="*60)
            print("🎯 TRASH DETECTION TRAINING PIPELINE COMPLETED")
            print("="*60)
            
            print("\n📋 PIPELINE STATUS:")
            status_items = [
                ("Data Preprocessing", summary['preprocessing_completed']),
                ("Detection Training", summary['detection_training_completed']), 
                ("Classification Training", summary['classification_training_completed']),
                ("Comprehensive Evaluation", summary['evaluation_completed']),
                ("Integration Testing", summary['integration_test_completed'])
            ]
            
            for item, status in status_items:
                print(f"   {'✅' if status else '❌'} {item}")
            
            print("\n📊 KEY METRICS:")
            if 'detection_mAP50' in summary:
                print(f"   🎯 Detection mAP@50: {summary['detection_mAP50']:.4f}")
            if 'classification_accuracy' in summary:
                print(f"   🎯 Classification Accuracy: {summary['classification_accuracy']:.4f}")
            
            print("\n📁 OUTPUTS:")
            print("   📂 Models: models/detection/best.pt, models/classification/best.pt")
            print("   📂 Results: results/pipeline_results.json")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing final summary: {e}")


# ==================== STANDALONE FUNCTIONS ====================

def run_detection_only(config_path: str, source: str, output: str = None):
    """Chạy detection trên image/video"""
    try:
        if not check_dependencies():
            logger.error("Missing dependencies for detection")
            return
            
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Setup pipeline config
        pipe_config = config.get('pipeline', {})
        pipeline_config = PipelineConfig(
            detection_model_path=pipe_config.get('detection_model_path', 'models/detection/best.pt'),
            classification_model_path=pipe_config.get('classification_model_path', 'models/classification/best.pt'),
            detection_conf_threshold=pipe_config.get('detection_conf_threshold', 0.25),
        )
        
        # Initialize pipeline
        pipeline = TrashDetectionPipeline(pipeline_config)
        
        # Process source
        if Path(source).is_file():
            # Single image/video
            result = pipeline.process_image(source)
            print(f"Detection completed. Found {result['summary']['total_objects']} objects.")
        else:
            print(f"Source not found: {source}")
            
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        raise


def run_detection_training_only(config_path: str):
    """Chỉ chạy detection training"""
    try:
        if not check_dependencies():
            logger.error("Missing dependencies for detection training")
            return
            
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load detection config
        det_config = config.get('detection', {})
        
        # Create training config
        training_config = DetectionTrainingConfig(
            model_name=det_config.get('model_name', 'yolov8n.pt'),
            data_yaml=det_config.get('data_yaml', 'data/detection/processed/dataset.yaml'),
            epochs=det_config.get('epochs', 100),
            batch_size=det_config.get('batch_size', 16),
            img_size=det_config.get('img_size', 640),
            learning_rate=det_config.get('learning_rate', 0.01),
            device=det_config.get('device', 'auto'),
            save_dir=Path(det_config.get('save_dir', 'results/detection')),
            experiment_name=det_config.get('experiment_name', 'detection_v1'),
        )
        
        # Train model
        trainer = DetectionTrainer(training_config)
        results = trainer.run_full_training()
        
        print("Detection training completed successfully!")
        print(f"Model saved at: {results['model_paths']['best']}")
        print(f"mAP@50: {results['summary']['mAP50']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in detection training: {e}")
        raise


def run_classification_training_only(config_path: str):
    """Chỉ chạy classification training"""
    try:
        if not check_dependencies():
            logger.error("Missing dependencies for classification training")
            return
            
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load classification config
        cls_config = config.get('classification', {})
        
        # Create training config
        training_config = ClassificationTrainingConfig(
            model_name=cls_config.get('model_name', 'yolov8n-cls.pt'),
            data_yaml=cls_config.get('data_yaml', 'data/classification/processed'),
            epochs=cls_config.get('epochs', 50),
            batch_size=cls_config.get('batch_size', 32),
            img_size=cls_config.get('img_size', 224),
            learning_rate=cls_config.get('learning_rate', 0.001),
            device=cls_config.get('device', 'auto'),
            save_dir=Path(cls_config.get('save_dir', 'results/classification')),
            experiment_name=cls_config.get('experiment_name', 'classification_v1'),
        )
        
        # Train model
        trainer = ClassificationTrainer(training_config)
        results = trainer.run_full_training()
        
        print("Classification training completed successfully!")
        print(f"Model saved at: {results['model_paths']['best']}")
        print(f"Top-1 Accuracy: {results['summary']['top1_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in classification training: {e}")
        raise


def run_evaluation_only(config_path: str):
    """Chỉ chạy evaluation"""
    try:
        if not check_dependencies():
            logger.error("Missing dependencies for evaluation")
            return
            
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load evaluation config
        eval_config = config.get('evaluation', {})
        
        # Create evaluation config
        evaluation_config = EvaluationConfig(
            detection_model_path=eval_config.get('detection_model_path', 'models/detection/best.pt'),
            classification_model_path=eval_config.get('classification_model_path', 'models/classification/best.pt'),
            detection_data_yaml=eval_config.get('detection_data_yaml', 'data/detection/processed/dataset.yaml'),
            classification_data_yaml=eval_config.get('classification_data_yaml', 'data/classification/processed'),
            device=eval_config.get('device', 'auto'),
        )
        
        # Run evaluation
        evaluator = ComprehensiveEvaluator(evaluation_config)
        results = evaluator.run_full_evaluation()
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        raise


# ==================== MAIN FUNCTION ====================

def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="Trash Detection Training Pipeline (Lightweight Integrated)")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--steps", type=str, default=None,
                       help="Steps to run (comma-separated): preprocessing,detection,classification,evaluation,integration")
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Run full training pipeline")
    
    # Individual training operations
    parser.add_argument("--train-detection", action="store_true", help="Train detection model only")
    parser.add_argument("--train-classification", action="store_true", help="Train classification model only")
    
    # Standalone operations
    parser.add_argument("--detect", action="store_true", help="Run detection only")
    parser.add_argument("--source", type=str, help="Source for detection (image/video path)")
    parser.add_argument("--output", type=str, help="Output path for detection results")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    
    # System checks
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    
    args = parser.parse_args()
    
    try:
        if args.check_deps:
            # Check dependencies
            if check_dependencies():
                print("✅ All dependencies are available")
            else:
                print("❌ Some dependencies are missing")
                print("Run: pip install ultralytics torch pyyaml")
            return
            
        if args.detect:
            # Run detection only
            if not args.source:
                print("Error: --source is required for detection")
                return
            run_detection_only(args.config, args.source, args.output)
            
        elif args.train_detection:
            # Run detection training only
            run_detection_training_only(args.config)
            
        elif args.train_classification:
            # Run classification training only
            run_classification_training_only(args.config)
            
        elif args.evaluate:
            # Run evaluation only
            run_evaluation_only(args.config)
            
        elif args.full_pipeline or args.steps:
            # Run training pipeline
            pipeline = TrashDetectionTrainingPipeline(args.config)
            results = pipeline.run_full_pipeline(args.steps)
            
            if 'error' not in results:
                logger.info("Training pipeline completed successfully!")
            
        else:
            # Interactive mode
            print("🚀 Trash Detection Training Pipeline (Lightweight Integrated)")
            print("="*60)
            print("Available commands:")
            print("1. Check dependencies:      --check-deps")
            print("2. Full pipeline:           --full-pipeline")
            print("3. Specific steps:          --steps preprocessing,detection,classification")
            print("4. Detection training only: --train-detection")
            print("5. Classification training: --train-classification")  
            print("6. Detection inference:     --detect --source image.jpg")
            print("7. Evaluation only:         --evaluate")
            print()
            print("📋 Data preprocessing (run these first):")
            print("   python data_preprocessing_detection.py")
            print("   python data_preprocessing_classification.py")
            print()
            print("🎯 Example usage:")
            print("   # Check system dependencies")
            print("   python main.py --check-deps")
            print()
            print("   # Train individual models")
            print("   python main.py --train-detection")
            print("   python main.py --train-classification")
            print()
            print("   # Full pipeline")
            print("   python main.py --config configs/training_config.yaml --full-pipeline")
            print("   python main.py --steps detection,classification")
            print()
            print("   # Inference and evaluation")
            print("   python main.py --detect --source test_image.jpg")
            print("   python main.py --evaluate")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()