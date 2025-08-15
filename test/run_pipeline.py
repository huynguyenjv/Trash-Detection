"""
Script chạy toàn bộ pipeline Trash Detection từ A-Z

Author: Huy Nguyen
Date: August 2025
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrashDetectionPipeline:
    """Class chính để chạy toàn bộ pipeline"""
    
    def __init__(self, skip_preprocessing: bool = False, skip_training: bool = False):
        self.skip_preprocessing = skip_preprocessing
        self.skip_training = skip_training
        
    def run_preprocessing(self) -> bool:
        """Chạy data preprocessing"""
        try:
            logger.info("=== BƯỚC 1: DATA PREPROCESSING ===")
            
            if self.skip_preprocessing:
                logger.info("Bỏ qua preprocessing (sử dụng --skip-preprocessing)")
                return True
            
            # Check if processed data already exists
            processed_data_path = Path("data/processed")
            if processed_data_path.exists() and any(processed_data_path.iterdir()):
                logger.info("Dữ liệu đã được xử lý. Sử dụng --skip-preprocessing để bỏ qua.")
                response = input("Bạn có muốn xử lý lại dữ liệu? (y/n): ")
                if response.lower() != 'y':
                    return True
            
            # Import and run preprocessing
            from data_preprocessing import DataPreprocessor, DatasetConfig
            
            config = DatasetConfig()
            preprocessor = DataPreprocessor(config)
            preprocessor.run_preprocessing()
            
            logger.info("✅ Data preprocessing hoàn thành!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong preprocessing: {e}")
            return False
    
    def run_training(self) -> bool:
        """Chạy model training"""
        try:
            logger.info("=== BƯỚC 2: MODEL TRAINING ===")
            
            if self.skip_training:
                logger.info("Bỏ qua training (sử dụng --skip-training)")
                return True
            
            # Check if model already exists
            model_path = Path("models/trash_detection_best.pt")
            if model_path.exists():
                logger.info("Model đã tồn tại. Sử dụng --skip-training để bỏ qua.")
                response = input("Bạn có muốn train lại model? (y/n): ")
                if response.lower() != 'y':
                    return True
            
            # Import and run training
            from train import TrashDetectionTrainer, TrainingConfig
            import torch
            
            # Auto config based on hardware
            config = TrainingConfig()
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 8:
                    config.model_name = "yolov8m.pt"
                    config.batch_size = 32
                elif gpu_memory >= 4:
                    config.batch_size = 16
                else:
                    config.batch_size = 8
                    
                logger.info(f"GPU Memory: {gpu_memory:.1f}GB, Batch size: {config.batch_size}")
            
            trainer = TrashDetectionTrainer(config)
            trainer.load_model()
            
            # Train
            best_weights_path = trainer.train()
            
            # Validate
            metrics = trainer.validate_model(best_weights_path)
            
            # Plot results
            trainer.plot_training_results()
            
            logger.info("✅ Model training hoàn thành!")
            logger.info(f"Best weights: {best_weights_path}")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong training: {e}")
            return False
    
    def run_evaluation(self) -> bool:
        """Chạy model evaluation"""
        try:
            logger.info("=== BƯỚC 3: MODEL EVALUATION ===")
            
            # Check if model exists
            model_path = Path("models/trash_detection_best.pt")
            if not model_path.exists():
                logger.error("Không tìm thấy model. Vui lòng chạy training trước.")
                return False
            
            # Import and run evaluation
            from evaluate import ModelEvaluator, EvaluationConfig
            
            config = EvaluationConfig()
            evaluator = ModelEvaluator(config)
            
            # Run evaluation
            results = evaluator.run_full_evaluation()
            
            logger.info("✅ Model evaluation hoàn thành!")
            if 'validation_metrics' in results:
                logger.info("Validation metrics:")
                for metric, value in results['validation_metrics'].items():
                    logger.info(f"  {metric}: {value:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong evaluation: {e}")
            return False
    
    def run_demo_detection(self) -> bool:
        """Chạy demo detection"""
        try:
            logger.info("=== BƯỚC 4: DEMO DETECTION ===")
            
            # Check if model exists
            model_path = Path("models/trash_detection_best.pt")
            if not model_path.exists():
                logger.error("Không tìm thấy model. Vui lòng chạy training trước.")
                return False
            
            # Ask user for demo type
            print("\nChọn loại demo:")
            print("1. Webcam real-time detection")
            print("2. Test trên ảnh mẫu")
            print("3. Bỏ qua demo")
            
            choice = input("Lựa chọn của bạn (1/2/3): ")
            
            if choice == "1":
                logger.info("Khởi động webcam demo...")
                from detect import TrashDetector, DetectionConfig
                
                config = DetectionConfig()
                detector = TrashDetector(config)
                
                print("Webcam sẽ mở. Nhấn 'q' để thoát.")
                detector.detect_video_stream(source=0)
                
            elif choice == "2":
                logger.info("Test trên ảnh mẫu...")
                # Tạo test image nếu có
                test_images_dir = Path("data/processed/images/test")
                if test_images_dir.exists():
                    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
                    if test_images:
                        from detect import TrashDetector, DetectionConfig
                        
                        config = DetectionConfig()
                        detector = TrashDetector(config)
                        
                        # Test trên ảnh đầu tiên
                        sample_image = test_images[0]
                        logger.info(f"Testing trên: {sample_image}")
                        
                        detections = detector.detect_image(str(sample_image), show=True)
                        logger.info(f"Phát hiện {len(detections)} objects")
                    else:
                        logger.warning("Không tìm thấy test images")
                else:
                    logger.warning("Không tìm thấy test images directory")
            
            else:
                logger.info("Bỏ qua demo")
            
            logger.info("✅ Demo hoàn thành!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong demo: {e}")
            return False
    
    def run_full_pipeline(self) -> None:
        """Chạy toàn bộ pipeline"""
        logger.info("🚀 BẮT ĐẦU TRASH DETECTION PIPELINE")
        logger.info("=" * 50)
        
        success_steps = 0
        total_steps = 4
        
        # Step 1: Preprocessing
        if self.run_preprocessing():
            success_steps += 1
        else:
            logger.error("Pipeline dừng do lỗi preprocessing")
            return
        
        # Step 2: Training
        if self.run_training():
            success_steps += 1
        else:
            logger.error("Pipeline dừng do lỗi training")
            return
        
        # Step 3: Evaluation
        if self.run_evaluation():
            success_steps += 1
        else:
            logger.warning("Evaluation thất bại, nhưng pipeline tiếp tục")
        
        # Step 4: Demo
        if self.run_demo_detection():
            success_steps += 1
        else:
            logger.warning("Demo thất bại, nhưng pipeline đã hoàn thành")
        
        # Summary
        logger.info("=" * 50)
        logger.info(f"🎉 PIPELINE HOÀN THÀNH: {success_steps}/{total_steps} bước thành công")
        
        if success_steps >= 3:
            logger.info("✅ Model đã sẵn sàng sử dụng!")
            logger.info("Để chạy detection:")
            logger.info("  cd src")
            logger.info("  python detect.py --mode webcam --source 0")
        else:
            logger.warning("⚠️  Pipeline không hoàn thành. Vui lòng kiểm tra logs.")


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="Trash Detection Pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Bỏ qua data preprocessing")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Bỏ qua model training")
    parser.add_argument("--step", choices=["preprocessing", "training", "evaluation", "demo"],
                       help="Chỉ chạy một bước cụ thể")
    
    args = parser.parse_args()
    
    # Khởi tạo pipeline
    pipeline = TrashDetectionPipeline(
        skip_preprocessing=args.skip_preprocessing,
        skip_training=args.skip_training
    )
    
    # Chạy theo step cụ thể hoặc full pipeline
    if args.step:
        if args.step == "preprocessing":
            pipeline.run_preprocessing()
        elif args.step == "training":
            pipeline.run_training()
        elif args.step == "evaluation":
            pipeline.run_evaluation()
        elif args.step == "demo":
            pipeline.run_demo_detection()
    else:
        # Chạy full pipeline
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
