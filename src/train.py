"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train - Huấn luyện mô hình YOLOv8 cho Trash Detection

Mô tả:
    Module này thực hiện quá trình huấn luyện mô hình:
    - Load pretrained YOLOv8 model
    - Fine-tune với custom trash detection dataset
    - Lưu best/last model weights
    - Export training metrics và visualization

Author: Huy Nguyen
Email: huynguyen@example.com
Date: August 2025
Version: 1.0.0
License: MIT
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Model name constants
YOLOV8N_PT = "yolov8n.pt"
YOLOV8S_PT = "yolov8s.pt"

@dataclass
class TrainingConfig:
    """Cấu hình training"""
    # Model configuration
    model_name: str = YOLOV8N_PT  # yolov8n.pt (fast) or yolov8m.pt (balanced)
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 640
    
    # Learning rate
    lr0: float = 0.01  # Initial learning rate
    lrf: float = 0.01  # Final learning rate (fraction of lr0)
    
    # Data augmentation
    hsv_h: float = 0.015  # HSV-Hue augmentation
    hsv_s: float = 0.7    # HSV-Saturation augmentation
    hsv_v: float = 0.4    # HSV-Value augmentation
    degrees: float = 0.0  # Rotation degrees
    translate: float = 0.1 # Translation
    scale: float = 0.5    # Scaling
    shear: float = 0.0    # Shearing
    perspective: float = 0.0 # Perspective
    flipud: float = 0.0   # Vertical flip probability
    fliplr: float = 0.5   # Horizontal flip probability
    mosaic: float = 1.0   # Mosaic augmentation probability
    mixup: float = 0.0    # MixUp augmentation probability
    
    # Paths
    data_yaml: Path = Path("../data/processed/dataset.yaml")
    project_dir: Path = Path("../runs/detect")
    name: str = "trash_detection"
    
    # Device
    device: str = "auto"  # auto, cpu, or cuda
    
    # Workers
    workers: int = 8
    
    # Validation
    val: bool = True
    save_period: int = -1  # Save checkpoint every n epochs (-1 = disabled)


class TrashDetectionTrainer:
    """Class chính để training model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        
        # Kiểm tra cấu hình
        self._validate_config()
        
        # Setup device
        self._setup_device()
        
    def _validate_config(self) -> None:
        """Kiểm tra tính hợp lệ của config"""
        if not self.config.data_yaml.exists():
            raise FileNotFoundError(f"Không tìm thấy file dataset.yaml: {self.config.data_yaml}")
        
        # Kiểm tra batch size hợp lý với VRAM
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            # Tính toán batch size an toàn hơn
            if gpu_memory >= 16:
                recommended_batch = 16
            elif gpu_memory >= 12:
                recommended_batch = 8  # Giảm xuống để an toàn
            elif gpu_memory >= 8:
                recommended_batch = 6
            elif gpu_memory >= 4:
                recommended_batch = 4
            else:
                recommended_batch = 2
            
            if self.config.batch_size > recommended_batch:
                logger.warning(
                    f"Batch size {self.config.batch_size} quá lớn cho GPU "
                    f"(VRAM: {gpu_memory:.1f}GB). Đổi thành: {recommended_batch}"
                )
                self.config.batch_size = recommended_batch
    
    def _setup_device(self) -> None:
        """Setup device cho training"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")
            else:
                self.device = "cpu"
                logger.info("Sử dụng CPU")
        else:
            self.device = self.config.device
            logger.info(f"Sử dụng device: {self.device}")
    
    def load_model(self) -> None:
        """Load pre-trained YOLOv8 model"""
        try:
            logger.info(f"Đang load model: {self.config.model_name}")
            self.model = YOLO(self.config.model_name)
            
            # In thông tin model
            logger.info("Model đã load thành công!")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Lỗi khi load model: {e}")
            raise
    
    def prepare_training_args(self) -> Dict[str, Any]:
        """Chuẩn bị arguments cho training"""
        training_args = {
            # Data
            'data': str(self.config.data_yaml),
            
            # Training parameters
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.image_size,
            
            # Learning rate
            'lr0': self.config.lr0,
            'lrf': self.config.lrf,
            
            # Data augmentation
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
            
            # Paths and naming
            'project': str(self.config.project_dir),
            'name': self.config.name,
            
            # Device and workers
            'device': self.device,
            'workers': self.config.workers,
            
            # Validation
            'val': self.config.val,
            'save_period': self.config.save_period,
            
            # Additional settings
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,  # Rectangular training
            'cos_lr': False,  # Cosine learning rate scheduler
            'close_mosaic': 10,  # Disable mosaic in last n epochs
            'resume': False,  # Resume from checkpoint
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,  # Dataset fraction to train on
            'profile': False,  # Profile ONNX and TensorRT speeds
            'freeze': None,  # Freeze layers: backbone=10, all=24
            'multi_scale': False,  # Multi-scale training
            'overlap_mask': True,  # Overlap masks
            'mask_ratio': 4,  # Mask downsample ratio
            'dropout': 0.0,  # Use dropout regularization
        }
        
        return training_args
    
    def print_training_summary(self, training_args: Dict[str, Any]) -> None:
        """In tóm tắt cấu hình training"""
        logger.info("=== TÓMTẮT CẤU HÌNH TRAINING ===")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Dataset: {training_args['data']}")
        logger.info(f"Device: {training_args['device']}")
        logger.info(f"Epochs: {training_args['epochs']}")
        logger.info(f"Batch size: {training_args['batch']}")
        logger.info(f"Image size: {training_args['imgsz']}")
        logger.info(f"Initial LR: {training_args['lr0']}")
        logger.info(f"Workers: {training_args['workers']}")
        logger.info(f"Output: {training_args['project']}/{training_args['name']}")
        logger.info("===================================")
    
    def train(self) -> str:
        """
        Bắt đầu training model
        
        Returns:
            str: Đường dẫn đến model weights tốt nhất
        """
        if self.model is None:
            raise ValueError("Model chưa được load. Gọi load_model() trước.")
        
        try:
            # Chuẩn bị training arguments
            training_args = self.prepare_training_args()
            
            # In tóm tắt
            self.print_training_summary(training_args)
            
            logger.info("=== BẮT ĐẦU TRAINING ===")
            
            # Bắt đầu training
            self.model.train(**training_args)
            
            logger.info("=== HOÀN THÀNH TRAINING ===")
            
            # Đường dẫn đến model weights tốt nhất
            best_weights_path = Path(training_args['project']) / training_args['name'] / "weights" / "best.pt"
            
            if best_weights_path.exists():
                logger.info(f"Model weights tốt nhất: {best_weights_path}")
                
                # Copy model weights to models directory
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                final_model_path = models_dir / "trash_detection_best.pt"
                
                import shutil
                shutil.copy2(best_weights_path, final_model_path)
                logger.info(f"Đã copy model weights đến: {final_model_path}")
                
                return str(final_model_path)
            else:
                raise FileNotFoundError("Không tìm thấy file weights tốt nhất")
                
        except Exception as e:
            logger.error(f"Lỗi trong quá trình training: {e}")
            raise
    
    def validate_model(self, weights_path: str) -> Dict[str, float]:
        """
        Validate model trên validation set
        
        Args:
            weights_path: Đường dẫn đến model weights
            
        Returns:
            Dict chứa các metrics
        """
        try:
            logger.info("=== ĐÁNH GIÁ MODEL ===")
            
            # Load model với weights đã train
            model = YOLO(weights_path)
            
            # Validate
            results = model.val(
                data=str(self.config.data_yaml),
                imgsz=self.config.image_size,
                batch=self.config.batch_size,
                device=self.device,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
            }
            
            logger.info("Kết quả validation:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Lỗi khi validate model: {e}")
            raise
    
    def plot_training_results(self, project_path: Optional[str] = None) -> None:
        """Vẽ biểu đồ kết quả training"""
        try:
            if project_path is None:
                project_path = self.config.project_dir / self.config.name
            else:
                project_path = Path(project_path)
            
            results_csv = project_path / "results.csv"
            
            if not results_csv.exists():
                logger.warning("Không tìm thấy file results.csv")
                return
            
            import pandas as pd
            
            # Đọc results
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            # Tạo plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Results', fontsize=16)
            
            # Loss plots
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # mAP plots
            if 'metrics/mAP50(B)' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
                if 'metrics/mAP50-95(B)' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
                axes[0, 1].set_title('Mean Average Precision')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Precision and Recall
            if 'metrics/precision(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                axes[1, 0].set_title('Precision & Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning Rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df['epoch'], df['lr/pg0'])
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Lưu plot
            plot_path = project_path / "training_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ training: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Lỗi khi vẽ biểu đồ: {e}")


def main():
    """Hàm main"""
    try:
        # Khởi tạo config với setting an toàn
        config = TrainingConfig()
        
        # Cấu hình an toàn cho GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory >= 16:
                config.model_name = YOLOV8S_PT  # Small model thay vì medium
                config.batch_size = 8
            elif gpu_memory >= 12:
                config.model_name = YOLOV8N_PT  # Nano model - nhẹ nhất
                config.batch_size = 6
            elif gpu_memory >= 8:
                config.model_name = YOLOV8N_PT
                config.batch_size = 4
            else:
                config.model_name = YOLOV8N_PT
                config.batch_size = 2
            
        else:
            logger.info("Sử dụng CPU - training sẽ chậm")
            config.batch_size = 2
            config.epochs = 5
        
        logger.info(f"Final config: Model={config.model_name}, Batch={config.batch_size}, Epochs={config.epochs}")
        
        # Khởi tạo trainer
        trainer = TrashDetectionTrainer(config)
        
        # Load model
        trainer.load_model()
        
        # Bắt đầu training
        best_weights_path = trainer.train()
        
        # Validate model
        metrics = trainer.validate_model(best_weights_path)
        
        # Vẽ biểu đồ kết quả
        trainer.plot_training_results()
        
        logger.info("=== HOÀN THÀNH TẤT CẢ ===")
        logger.info(f"Model weights tốt nhất: {best_weights_path}")
        logger.info("Metrics cuối cùng:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Lỗi chương trình: {e}")
        raise


if __name__ == "__main__":
    main()
