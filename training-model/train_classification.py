#!/usr/bin/env python3
"""
Training Classification Model - YOLOv8 Fine-tuning cho Image Classification
Fine-tune YOLOv8 classification model cho trash classification task

Author: Huy Nguyen
Date: September 2025
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
from sklearn.metrics import confusion_matrix, classification_report

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_classification.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationTrainingConfig:
    """Cấu hình cho training classification model"""
    # Model config
    model_name: str = "yolov8n-cls.pt"  # yolov8n-cls, yolov8s-cls, yolov8m-cls, yolov8l-cls, yolov8x-cls
    pretrained: bool = True
    
    # Dataset
    data_yaml: str = "data/classification/processed/dataset_classification.yaml"
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 32  # Larger batch for classification
    img_size: int = 224   # Standard size for classification
    learning_rate: float = 0.001  # Lower LR for classification
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Augmentations for classification
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 15.0    # More rotation for classification
    translate: float = 0.1
    scale: float = 0.9       # Scale augmentation
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5      # Horizontal flip
    auto_augment: str = "randaugment"  # AutoAugment policy
    erasing: float = 0.4     # Random erasing
    
    # Training settings
    patience: int = 20       # More patience for classification
    save_period: int = 5
    device: str = "auto"
    workers: int = 8
    
    # Optimizer settings
    optimizer: str = "AdamW"  # AdamW often better for classification
    lr_decay: float = 0.01
    
    # Validation
    val_split: float = 0.1
    
    # Output paths
    project_name: str = "trash_classification"
    experiment_name: str = "classification_v1"
    save_dir: Path = Path("results/classification")
    
    # Monitoring
    plots: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Khởi tạo sau khi tạo object"""
        self.save_dir.mkdir(parents=True, exist_ok=True)


class ClassificationTrainer:
    """Class chính để training classification model"""
    
    def __init__(self, config: ClassificationTrainingConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        self.dataset_info: Dict[str, Any] = {}
        self.training_results: Dict[str, Any] = {}
        self.class_names: List[str] = []
        
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
            
            self.class_names = self.dataset_info['names']
            
            logger.info(f"Dataset loaded: {self.config.data_yaml}")
            logger.info(f"Classes ({self.dataset_info['nc']}): {self.class_names}")
            logger.info(f"Dataset path: {self.dataset_info['path']}")
            
        except Exception as e:
            logger.error(f"Error loading dataset info: {e}")
            raise
    
    def _initialize_model(self) -> None:
        """Khởi tạo YOLO classification model"""
        try:
            logger.info(f"Initializing classification model: {self.config.model_name}")
            
            # Load pretrained model hoặc tạo mới
            if self.config.pretrained:
                self.model = YOLO(self.config.model_name)
                logger.info("Loaded pretrained classification weights")
            else:
                # Load architecture only (random weights)
                model_yaml = self.config.model_name.replace('.pt', '.yaml')
                self.model = YOLO(model_yaml)
                logger.info("Initialized with random weights")
            
            # In thông tin model
            logger.info(f"Model: {self.model.model}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def setup_training_config(self) -> Dict[str, Any]:
        """Tạo cấu hình training cho YOLO classification"""
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
            'optimizer': self.config.optimizer,
            'lrf': self.config.lr_decay,
            
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
            'auto_augment': self.config.auto_augment,
            'erasing': self.config.erasing,
            
            # Training settings
            'patience': self.config.patience,
            'save_period': self.config.save_period,
            'device': self.device,
            'workers': self.config.workers,
            
            # Validation & saving
            'val': True,
            'save': True,
            
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
            logger.info("=== BẮT ĐẦU TRAINING CLASSIFICATION MODEL ===")
            
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
            logger.info("=== VALIDATING CLASSIFICATION MODEL ===")
            
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
                plots=True,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'top1_accuracy': float(val_results.top1),
                'top5_accuracy': float(val_results.top5) if hasattr(val_results, 'top5') else None,
            }
            
            # Per-class metrics nếu có
            if hasattr(val_results, 'confusion_matrix') and val_results.confusion_matrix is not None:
                cm = val_results.confusion_matrix.matrix
                per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
                metrics['per_class_accuracy'] = per_class_accuracy.tolist()
            
            logger.info("Validation Results:")
            logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            if metrics['top5_accuracy'] is not None:
                logger.info(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    def generate_detailed_evaluation(self, weights_path: str) -> Dict[str, Any]:
        """Tạo evaluation chi tiết với confusion matrix và classification report"""
        try:
            logger.info("Generating detailed evaluation...")
            
            model = YOLO(weights_path)
            
            # Test trên test set
            test_dir = Path(self.dataset_info['path']) / 'test'
            
            all_predictions = []
            all_true_labels = []
            
            # Predict từng class folder
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = test_dir / class_name
                if not class_dir.exists():
                    continue
                
                # Lấy tất cả ảnh trong class folder
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                image_files = [f for f in class_dir.iterdir() 
                             if f.suffix.lower() in image_extensions]
                
                logger.info(f"Evaluating {len(image_files)} images for class '{class_name}'")
                
                for img_path in image_files:
                    # Predict
                    results = model(str(img_path), verbose=False)
                    
                    if results and len(results) > 0:
                        # Lấy predicted class (top-1)
                        probs = results[0].probs
                        if probs is not None:
                            predicted_class_idx = probs.top1
                            all_predictions.append(predicted_class_idx)
                            all_true_labels.append(class_idx)
            
            if not all_predictions:
                logger.warning("No predictions generated")
                return {}
            
            # Tạo confusion matrix
            cm = confusion_matrix(all_true_labels, all_predictions)
            
            # Classification report
            report = classification_report(
                all_true_labels, all_predictions,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Accuracy per class
            per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
            
            evaluation_results = {
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'per_class_accuracy': per_class_accuracy.tolist(),
                'overall_accuracy': report['accuracy'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'], 
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall'],
                'weighted_f1': report['weighted avg']['f1-score'],
            }
            
            logger.info("Detailed Evaluation Results:")
            logger.info(f"  Overall Accuracy: {evaluation_results['overall_accuracy']:.4f}")
            logger.info(f"  Macro F1-Score: {evaluation_results['macro_f1']:.4f}")
            logger.info(f"  Weighted F1-Score: {evaluation_results['weighted_f1']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in detailed evaluation: {e}")
            return {}
    
    def plot_confusion_matrix(self, evaluation_results: Dict[str, Any]) -> None:
        """Vẽ confusion matrix"""
        try:
            if 'confusion_matrix' not in evaluation_results:
                logger.warning("No confusion matrix data available")
                return
            
            cm = np.array(evaluation_results['confusion_matrix'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Raw confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=ax1)
            ax1.set_title('Confusion Matrix (Counts)')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Normalized confusion matrix
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=ax2)
            ax2.set_title('Confusion Matrix (Normalized)')
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
            
            plt.tight_layout()
            
            # Lưu plot
            plot_path = self.config.save_dir / f"{self.config.experiment_name}_confusion_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
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
                fig.suptitle('Classification Training Results', fontsize=16)
                
                # Loss plots
                if 'train/loss' in df.columns:
                    axes[0, 0].plot(df['epoch'], df['train/loss'], label='Train Loss')
                    if 'val/loss' in df.columns:
                        axes[0, 0].plot(df['epoch'], df['val/loss'], label='Val Loss')
                    axes[0, 0].set_title('Training Loss')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
                
                # Accuracy plots
                if 'metrics/accuracy_top1' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top1'], label='Top-1 Accuracy')
                    if 'metrics/accuracy_top5' in df.columns:
                        axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top5'], label='Top-5 Accuracy')
                    axes[0, 1].set_title('Accuracy')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Accuracy')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
                
                # Learning rate
                if 'lr/pg0' in df.columns:
                    axes[1, 0].plot(df['epoch'], df['lr/pg0'])
                    axes[1, 0].set_title('Learning Rate')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('LR')
                    axes[1, 0].grid(True)
                
                # Per-class accuracy nếu có
                if hasattr(self, 'evaluation_results') and 'per_class_accuracy' in self.evaluation_results:
                    per_class_acc = self.evaluation_results['per_class_accuracy']
                    axes[1, 1].bar(range(len(self.class_names)), per_class_acc)
                    axes[1, 1].set_title('Per-Class Accuracy')
                    axes[1, 1].set_xlabel('Class')
                    axes[1, 1].set_ylabel('Accuracy')
                    axes[1, 1].set_xticks(range(len(self.class_names)))
                    axes[1, 1].set_xticklabels(self.class_names, rotation=45)
                    axes[1, 1].grid(True, axis='y')
                
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
                    'optimizer': self.config.optimizer,
                    'device': self.device,
                },
                'training_results': self.training_results,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Thêm evaluation results nếu có
            if hasattr(self, 'evaluation_results'):
                summary['evaluation_results'] = self.evaluation_results
            
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
            logger.info("=== STARTING FULL CLASSIFICATION TRAINING PIPELINE ===")
            
            # 1. Train model
            training_results = self.train_model()
            
            # 2. Validate model
            best_weights = training_results['best_weights']
            validation_results = self.validate_model(best_weights)
            
            # 3. Detailed evaluation
            detailed_evaluation = self.generate_detailed_evaluation(best_weights)
            self.evaluation_results = detailed_evaluation
            
            # 4. Plot confusion matrix
            if detailed_evaluation:
                self.plot_confusion_matrix(detailed_evaluation)
            
            # 5. Generate training plots
            self.generate_training_plots()
            
            # 6. Save summary
            self.training_results['validation_results'] = validation_results
            self.training_results['detailed_evaluation'] = detailed_evaluation
            self.save_training_summary()
            
            # 7. Final results
            final_results = {
                'training': training_results,
                'validation': validation_results,
                'detailed_evaluation': detailed_evaluation,
                'model_paths': {
                    'best': best_weights,
                    'last': training_results['last_weights']
                }
            }
            
            logger.info("=== CLASSIFICATION TRAINING PIPELINE COMPLETED ===")
            logger.info(f"Best model: {best_weights}")
            logger.info(f"Final Top-1 Accuracy: {validation_results['top1_accuracy']:.4f}")
            if detailed_evaluation:
                logger.info(f"Overall Accuracy: {detailed_evaluation['overall_accuracy']:.4f}")
                logger.info(f"Macro F1-Score: {detailed_evaluation['macro_f1']:.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 Classification Model")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt",
                       help="Model name (yolov8n-cls, yolov8s-cls, yolov8m-cls, etc.)")
    parser.add_argument("--data", type=str, default="data/classification/processed/dataset_classification.yaml",
                       help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=224,
                       help="Image size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--project", type=str, default="results/classification",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="classification_v1",
                       help="Experiment name")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo config
        config = ClassificationTrainingConfig(
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
        trainer = ClassificationTrainer(config)
        
        # Resume training nếu có
        if args.resume:
            logger.info(f"Resuming training from: {args.resume}")
            trainer.model = YOLO(args.resume)
        
        # Chạy training
        results = trainer.run_full_training()
        
        logger.info("Classification training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
