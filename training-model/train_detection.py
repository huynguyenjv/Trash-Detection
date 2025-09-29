#!/usr/bin/env python3
"""
Training Detection Model - YOLOv8 Fine-tuning cho Object Detection
Fine-tune YOLOv8 detection model cho trash detection task

Author: GitHub Copilot Assistant  
Date: September 2025
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import argparse
import json

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionTrainingConfig:
    """Cấu hình cho training detection model"""
    # Model config
    model_name: str = "yolov8n.pt"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    pretrained: bool = True
    
    # Dataset
    data_yaml: str = "data/detection/processed/dataset_detection.yaml"
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Augmentations
    hsv_h: float = 0.015  # HSV-Hue augmentation
    hsv_s: float = 0.7    # HSV-Saturation augmentation  
    hsv_v: float = 0.4    # HSV-Value augmentation
    degrees: float = 0.0  # Rotation augmentation
    translate: float = 0.1 # Translation augmentation
    scale: float = 0.5    # Scale augmentation
    shear: float = 0.0    # Shear augmentation
    perspective: float = 0.0 # Perspective augmentation
    flipud: float = 0.0   # Vertical flip augmentation
    fliplr: float = 0.5   # Horizontal flip augmentation
    mosaic: float = 1.0   # Mosaic augmentation
    mixup: float = 0.0    # Mixup augmentation
    
    # Training settings
    patience: int = 10    # Early stopping patience
    save_period: int = 5  # Save model every N epochs
    device: str = "auto"  # auto, cpu, cuda, mps
    workers: int = 8      # Number of worker threads
    
    # Validation
    val_split: float = 0.1
    save_json: bool = True
    save_hybrid: bool = False
    
    # Output paths
    project_name: str = "trash_detection"
    experiment_name: str = "detection_v1"
    save_dir: Path = Path("results/detection")
    
    # Monitoring
    plots: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Khởi tạo sau khi tạo object"""
        self.save_dir.mkdir(parents=True, exist_ok=True)


class DetectionTrainer:
    """Class chính để training detection model"""
    
    def __init__(self, config: DetectionTrainingConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        self.dataset_info: Dict[str, Any] = {}
        self.training_results: Dict[str, Any] = {}
        
        # Setup device
        self._setup_device()
        
        # Load dataset info
        self._load_dataset_info()
        
        # Initialize model
        self._initialize_model()
    
    def _setup_device(self) -> None:
        """Setup device cho training"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using Apple MPS")
            else:
                self.device = "cpu"
                logger.info("Using CPU")
        else:
            self.device = self.config.device
        
        logger.info(f"Training device: {self.device}")
    
    def _load_dataset_info(self) -> None:
        """Load thông tin dataset"""
        try:
            if not Path(self.config.data_yaml).exists():
                raise FileNotFoundError(f"Dataset YAML not found: {self.config.data_yaml}")
            
            with open(self.config.data_yaml, 'r') as f:
                self.dataset_info = yaml.safe_load(f)
            
            logger.info(f"Dataset loaded: {self.config.data_yaml}")
            logger.info(f"Classes ({self.dataset_info['nc']}): {self.dataset_info['names']}")
            logger.info(f"Dataset path: {self.dataset_info['path']}")
            
        except Exception as e:
            logger.error(f"Error loading dataset info: {e}")
            raise
    
    def _initialize_model(self) -> None:
        """Khởi tạo YOLO model"""
        try:
            logger.info(f"Initializing model: {self.config.model_name}")
            
            # Load pretrained model hoặc tạo mới
            if self.config.pretrained:
                self.model = YOLO(self.config.model_name)
                logger.info("Loaded pretrained weights")
            else:
                # Load architecture only (random weights)
                self.model = YOLO(self.config.model_name.replace('.pt', '.yaml'))
                logger.info("Initialized with random weights")
            
            # In thông tin model
            logger.info(f"Model: {self.model.model}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def setup_training_config(self) -> Dict[str, Any]:
        """Tạo cấu hình training cho YOLO"""
        train_config = {
            # Data
            'data': self.config.data_yaml,
            
            # Training params
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.img_size,
            'lr0': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'momentum': self.config.momentum,
            
            # Augmentation
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'shear': self.config.shear,
            'perspective': self.config.perspective,
            'flipud': self.config.flipud,
            'fliplr': self.config.fliplr,
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            
            # Training settings
            'patience': self.config.patience,
            'save_period': self.config.save_period,
            'device': self.device,
            'workers': self.config.workers,
            
            # Validation & saving
            'val': True,
            'save': True,
            'save_json': self.config.save_json,
            'save_hybrid': self.config.save_hybrid,
            
            # Output
            'project': str(self.config.save_dir),
            'name': self.config.experiment_name,
            'exist_ok': True,
            
            # Monitoring
            'plots': self.config.plots,
            'verbose': self.config.verbose,
        }
        
        return train_config
    
    def train_model(self) -> Dict[str, Any]:
        """Chạy training process"""
        try:
            logger.info("=== BẮT ĐẦU TRAINING DETECTION MODEL ===")
            
            # Setup training config
            train_config = self.setup_training_config()
            
            # Log training configuration
            logger.info("Training Configuration:")
            for key, value in train_config.items():
                logger.info(f"  {key}: {value}")
            
            # Start training
            start_time = datetime.now()
            logger.info(f"Training started at: {start_time}")
            
            results = self.model.train(**train_config)
            
            end_time = datetime.now()
            training_duration = end_time - start_time
            logger.info(f"Training completed at: {end_time}")
            logger.info(f"Training duration: {training_duration}")
            
            # Lưu training results
            self.training_results = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': training_duration.total_seconds(),
                'config': train_config,
                'results': results,
                'best_weights': str(results.save_dir / 'weights' / 'best.pt'),
                'last_weights': str(results.save_dir / 'weights' / 'last.pt'),
            }
            
            logger.info("=== TRAINING COMPLETED ===")
            return self.training_results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def validate_model(self, weights_path: Optional[str] = None) -> Dict[str, Any]:
        """Chạy validation trên test set"""
        try:
            logger.info("=== VALIDATING MODEL ===")
            
            # Load best weights nếu có
            if weights_path:
                model = YOLO(weights_path)
                logger.info(f"Loaded weights: {weights_path}")
            else:
                model = self.model
            
            # Chạy validation
            val_results = model.val(
                data=self.config.data_yaml,
                split='test',
                batch=self.config.batch_size,
                imgsz=self.config.img_size,
                device=self.device,
                save_json=True,
                plots=True,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': float(val_results.box.map50),
                'mAP50-95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'mAP50_per_class': val_results.box.map50_per_class.tolist() if val_results.box.map50_per_class is not None else [],
                'mAP_per_class': val_results.box.map_per_class.tolist() if val_results.box.map_per_class is not None else [],
            }
            
            logger.info("Validation Results:")
            logger.info(f"  mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    def generate_training_plots(self) -> None:
        """Tạo các biểu đồ training"""
        try:
            if not self.training_results:
                logger.warning("No training results available for plotting")
                return
            
            logger.info("Generating training plots...")
            
            # Results directory
            results_dir = Path(self.training_results['best_weights']).parent.parent
            
            # Load training results CSV nếu có
            csv_path = results_dir / 'results.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # Tạo plots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Training Results', fontsize=16)
                
                # Loss plots
                if 'train/box_loss' in df.columns:
                    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                    if 'val/box_loss' in df.columns:
                        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                    axes[0, 0].set_title('Box Loss')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
                
                # mAP plots
                if 'metrics/mAP50(B)' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
                    if 'metrics/mAP50-95(B)' in df.columns:
                        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
                    axes[0, 1].set_title('mAP Metrics')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('mAP')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
                
                # Precision/Recall
                if 'metrics/precision(B)' in df.columns:
                    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                    if 'metrics/recall(B)' in df.columns:
                        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                    axes[1, 0].set_title('Precision & Recall')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Score')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True)
                
                # Learning rate
                if 'lr/pg0' in df.columns:
                    axes[1, 1].plot(df['epoch'], df['lr/pg0'])
                    axes[1, 1].set_title('Learning Rate')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('LR')
                    axes[1, 1].grid(True)
                
                plt.tight_layout()
                
                # Lưu plot
                plot_path = self.config.save_dir / f"{self.config.experiment_name}_training_plots.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training plots saved: {plot_path}")
                
                plt.show()
            
        except Exception as e:
            logger.error(f"Error generating training plots: {e}")
    
    def save_training_summary(self) -> None:
        """Lưu tóm tắt training"""
        try:
            summary = {
                'experiment_name': self.config.experiment_name,
                'model_name': self.config.model_name,
                'dataset_info': self.dataset_info,
                'training_config': {
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size,
                    'img_size': self.config.img_size,
                    'learning_rate': self.config.learning_rate,
                    'device': self.device,
                },
                'training_results': self.training_results,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Lưu JSON summary
            summary_path = self.config.save_dir / f"{self.config.experiment_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Training summary saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving training summary: {e}")
    
    def run_full_training(self) -> Dict[str, Any]:
        """Chạy toàn bộ quá trình training"""
        try:
            logger.info("=== STARTING FULL TRAINING PIPELINE ===")
            
            # 1. Train model
            training_results = self.train_model()
            
            # 2. Validate model
            best_weights = training_results['best_weights']
            validation_results = self.validate_model(best_weights)
            
            # 3. Generate plots
            self.generate_training_plots()
            
            # 4. Save summary
            self.training_results['validation_results'] = validation_results
            self.save_training_summary()
            
            # 5. Final results
            final_results = {
                'training': training_results,
                'validation': validation_results,
                'model_paths': {
                    'best': best_weights,
                    'last': training_results['last_weights']
                }
            }
            
            logger.info("=== TRAINING PIPELINE COMPLETED ===")
            logger.info(f"Best model: {best_weights}")
            logger.info(f"Final mAP@0.5: {validation_results['mAP50']:.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 Detection Model")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="Model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--data", type=str, default="data/detection/processed/dataset_detection.yaml",
                       help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Image size")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--project", type=str, default="results/detection",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="detection_v1",
                       help="Experiment name")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo config
        config = DetectionTrainingConfig(
            model_name=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            learning_rate=args.lr,
            device=args.device,
            save_dir=Path(args.project),
            experiment_name=args.name
        )
        
        # Khởi tạo trainer
        trainer = DetectionTrainer(config)
        
        # Resume training nếu có
        if args.resume:
            logger.info(f"Resuming training from: {args.resume}")
            trainer.model = YOLO(args.resume)
        
        # Chạy training
        results = trainer.run_full_training()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()