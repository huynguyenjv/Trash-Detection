"""
Script đánh giá performance của model Trash Detection

Author: Huy Nguyen
Date: August 2025
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from ultralytics import YOLO
import torch

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Cấu hình evaluation"""
    # Model và data paths
    model_path: str = "../models/last.pt"
    data_yaml: str = "../data/processed/dataset.yaml"
    
    # Evaluation parameters
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Visualization
    max_examples: int = 10  # Số lượng ví dụ hiển thị tối đa
    figsize: Tuple[int, int] = (15, 10)
    
    # Device
    device: str = "auto"
    
    # Output directory
    output_dir: Path = Path("evaluation_results")


class ModelEvaluator:
    """Class chính để đánh giá model"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = []
        
        # Tạo thư mục output
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Setup device
        self._setup_device()
        
        # Load model
        self.load_model()
    
    def _setup_device(self) -> None:
        """Setup device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("Sử dụng CPU")
        else:
            self.device = self.config.device
    
    def load_model(self) -> None:
        """Load model đã train"""
        try:
            if not Path(self.config.model_path).exists():
                raise FileNotFoundError(f"Không tìm thấy model: {self.config.model_path}")
            
            logger.info(f"Đang load model: {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            
            # Get class names
            self.class_names = list(self.model.names.values())
            logger.info(f"Model đã load với {len(self.class_names)} classes")
            logger.info(f"Classes: {', '.join(self.class_names)}")
            
        except Exception as e:
            logger.error(f"Lỗi khi load model: {e}")
            raise
    
    def validate_on_test_set(self) -> Dict[str, float]:
        """
        Đánh giá model trên test set
        
        Returns:
            Dict chứa các metrics
        """
        try:
            logger.info("=== ĐÁNH GIÁ TRÊN TEST SET ===")
            
            # Validate model
            results = self.model.val(
                data=self.config.data_yaml,
                split='test',
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
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
            
            # Metrics per class
            if hasattr(results.box, 'map50_per_class'):
                per_class_map50 = results.box.map50_per_class.cpu().numpy()
                for i, class_name in enumerate(self.class_names):
                    if i < len(per_class_map50):
                        metrics[f'mAP50_{class_name}'] = float(per_class_map50[i])
            
            logger.info("Kết quả validation:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Lỗi khi validate: {e}")
            raise
    
    def analyze_predictions(self, test_dir: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Phân tích predictions chi tiết
        
        Args:
            test_dir: Thư mục chứa test images
            
        Returns:
            Tuple (ground_truth_labels, predicted_labels)
        """
        if test_dir is None:
            test_dir = "../data/processed/images/test"
        
        test_path = Path(test_dir)
        if not test_path.exists():
            raise FileNotFoundError(f"Không tìm thấy test directory: {test_path}")
        
        logger.info("Đang phân tích predictions...")
        
        ground_truth_labels = []
        predicted_labels = []
        
        # Lấy tất cả ảnh test
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        test_images = [f for f in test_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        for img_path in test_images:
            try:
                # Đọc ảnh
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Predict
                results = self.model(
                    image,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # Lấy ground truth từ annotation file
                annotation_path = Path("data/processed/labels/test") / f"{img_path.stem}.txt"
                gt_class = self._get_ground_truth_class(annotation_path)
                
                # Lấy predicted class (class có confidence cao nhất)
                pred_class = self._get_predicted_class(results)
                
                if gt_class is not None and pred_class is not None:
                    ground_truth_labels.append(gt_class)
                    predicted_labels.append(pred_class)
                    
            except Exception as e:
                logger.warning(f"Lỗi khi process ảnh {img_path}: {e}")
                continue
        
        logger.info(f"Đã phân tích {len(ground_truth_labels)} predictions")
        return ground_truth_labels, predicted_labels
    
    def _get_ground_truth_class(self, annotation_path: Path) -> Optional[str]:
        """Lấy ground truth class từ annotation file"""
        try:
            if not annotation_path.exists():
                return None
            
            with open(annotation_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    if 0 <= class_id < len(self.class_names):
                        return self.class_names[class_id]
            return None
            
        except Exception:
            return None
    
    def _get_predicted_class(self, results) -> Optional[str]:
        """Lấy predicted class từ YOLO results"""
        try:
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Lấy detection có confidence cao nhất
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                max_conf_idx = np.argmax(confidences)
                class_id = class_ids[max_conf_idx]
                
                if 0 <= class_id < len(self.class_names):
                    return self.class_names[class_id]
            
            return None
            
        except Exception:
            return None
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str]) -> None:
        """Vẽ confusion matrix"""
        try:
            # Tạo confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Plot
            plt.figure(figsize=self.config.figsize)
            
            # Raw confusion matrix
            plt.subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title('Confusion Matrix (Counts)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Normalized confusion matrix
            plt.subplot(1, 2, 2)
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title('Confusion Matrix (Normalized)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            plt.tight_layout()
            
            # Lưu plot
            save_path = self.config.output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu confusion matrix: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Lỗi khi plot confusion matrix: {e}")
    
    def generate_classification_report(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Tạo báo cáo classification chi tiết"""
        try:
            # Tạo classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Convert to DataFrame để dễ đọc
            df_report = pd.DataFrame(report).transpose()
            
            # Lưu báo cáo
            report_path = self.config.output_dir / "classification_report.csv"
            df_report.to_csv(report_path)
            
            logger.info("=== CLASSIFICATION REPORT ===")
            logger.info(f"\n{df_report.to_string()}")
            
            # In thông tin tóm tắt
            logger.info("\n=== TÓMTẮT ===")
            logger.info(f"Accuracy: {report['accuracy']:.4f}")
            logger.info(f"Macro avg precision: {report['macro avg']['precision']:.4f}")
            logger.info(f"Macro avg recall: {report['macro avg']['recall']:.4f}")
            logger.info(f"Macro avg f1-score: {report['macro avg']['f1-score']:.4f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo classification report: {e}")
            raise
    
    def visualize_predictions(self, num_examples: int = None) -> None:
        """Visualize một số ví dụ predictions"""
        if num_examples is None:
            num_examples = self.config.max_examples
        
        test_images_dir = Path("../data/processed/images/test")
        if not test_images_dir.exists():
            logger.warning("Không tìm thấy test images directory")
            return
        
        # Lấy random samples
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        test_images = [f for f in test_images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if len(test_images) == 0:
            logger.warning("Không tìm thấy test images")
            return
        
        # Random sample
        np.random.seed(42)
        sample_images = np.random.choice(test_images, 
                                       size=min(num_examples, len(test_images)), 
                                       replace=False)
        
        # Setup plot
        cols = min(5, len(sample_images))
        rows = (len(sample_images) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, img_path in enumerate(sample_images):
            try:
                # Đọc ảnh
                image = cv2.imread(str(img_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Predict
                results = self.model(
                    image,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # Vẽ predictions lên ảnh
                result_image = self._draw_predictions_on_image(image_rgb, results)
                
                # Lấy ground truth
                annotation_path = Path("data/processed/labels/test") / f"{img_path.stem}.txt"
                gt_class = self._get_ground_truth_class(annotation_path)
                pred_class = self._get_predicted_class(results)
                
                # Plot
                axes[i].imshow(result_image)
                axes[i].set_title(f"GT: {gt_class}\nPred: {pred_class}", fontsize=10)
                axes[i].axis('off')
                
            except Exception as e:
                logger.warning(f"Lỗi khi visualize {img_path}: {e}")
                axes[i].text(0.5, 0.5, "Error", ha='center', va='center')
                axes[i].axis('off')
        
        # Ẩn axes không sử dụng
        for j in range(len(sample_images), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        # Lưu plot
        save_path = self.config.output_dir / "prediction_examples.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Đã lưu ví dụ predictions: {save_path}")
        
        plt.show()
    
    def _draw_predictions_on_image(self, image: np.ndarray, results) -> np.ndarray:
        """Vẽ predictions lên ảnh"""
        result_image = image.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score >= self.config.conf_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Vẽ bounding box
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Vẽ label
                    label = f"{class_name}: {score:.2f}"
                    cv2.putText(result_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_image
    
    def plot_per_class_performance(self, metrics: Dict[str, float]) -> None:
        """Vẽ performance theo từng class"""
        try:
            # Lọc metrics per class
            per_class_metrics = {}
            for metric, value in metrics.items():
                if metric.startswith('mAP50_'):
                    class_name = metric.replace('mAP50_', '')
                    per_class_metrics[class_name] = value
            
            if not per_class_metrics:
                logger.warning("Không có per-class metrics để plot")
                return
            
            # Tạo plot
            plt.figure(figsize=(12, 6))
            
            classes = list(per_class_metrics.keys())
            values = list(per_class_metrics.values())
            
            bars = plt.bar(classes, values, color='skyblue', alpha=0.8)
            
            # Thêm giá trị lên các bar
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.title('mAP50 per Class')
            plt.xlabel('Class')
            plt.ylabel('mAP50')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Lưu plot
            save_path = self.config.output_dir / "per_class_performance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu per-class performance: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Lỗi khi plot per-class performance: {e}")
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Chạy đánh giá đầy đủ"""
        try:
            logger.info("=== BẮT ĐẦU ĐÁNH GIÁ MODEL ===")
            
            results = {}
            
            # 1. Validate trên test set
            logger.info("1. Đánh giá trên test set...")
            validation_metrics = self.validate_on_test_set()
            results['validation_metrics'] = validation_metrics
            
            # 2. Phân tích predictions chi tiết
            logger.info("2. Phân tích predictions...")
            y_true, y_pred = self.analyze_predictions()
            
            if len(y_true) > 0:
                # 3. Confusion matrix
                logger.info("3. Tạo confusion matrix...")
                self.plot_confusion_matrix(y_true, y_pred)
                
                # 4. Classification report
                logger.info("4. Tạo classification report...")
                classification_metrics = self.generate_classification_report(y_true, y_pred)
                results['classification_metrics'] = classification_metrics
            else:
                logger.warning("Không có dữ liệu để tạo confusion matrix và classification report")
            
            # 5. Visualize predictions
            logger.info("5. Visualize predictions...")
            self.visualize_predictions()
            
            # 6. Plot per-class performance
            logger.info("6. Vẽ per-class performance...")
            self.plot_per_class_performance(validation_metrics)
            
            logger.info("=== HOÀN THÀNH ĐÁNH GIÁ ===")
            
            return results
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình đánh giá: {e}")
            raise


def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Đánh giá Model Trash Detection")
    parser.add_argument("--model", type=str, default="../models/final.pt",
                       help="Đường dẫn model weights")
    parser.add_argument("--data", type=str, default="../data/processed/dataset.yaml",
                       help="Đường dẫn dataset.yaml")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--output", type=str, default="evaluation_results",
                       help="Thư mục lưu kết quả")
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo config
        config = EvaluationConfig(
            model_path=args.model,
            data_yaml=args.data,
            conf_threshold=args.conf,
            device=args.device,
            output_dir=Path(args.output)
        )
        
        # Khởi tạo evaluator
        evaluator = ModelEvaluator(config)
        
        # Chạy đánh giá đầy đủ
        results = evaluator.run_full_evaluation()
        
        logger.info("Đánh giá hoàn thành!")
        
    except Exception as e:
        logger.error(f"Lỗi chương trình: {e}")
        raise


if __name__ == "__main__":
    main()
