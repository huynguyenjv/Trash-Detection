#!/usr/bin/env python3
"""
Main Training Pipeline - Trash Detection System
Orchestrate toàn bộ training pipeline theo instruction.md

Pipeline Flow:
1. Data Preprocessing (Detection + Classification)
2. Detection Model Training
3. Classification Model Training  
4. Comprehensive Evaluation
5. Pipeline Integration Testing

Author: Huy Nguyen
Date: September 2025
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Import các modules
from data_preprocessing_detection import TACODataProcessor
from data_preprocessing_classification import GarbageDataProcessor
from train_detection import DetectionTrainer, DetectionTrainingConfig
from train_classification import ClassificationTrainer, ClassificationTrainingConfig
from evaluate import ComprehensiveEvaluator, EvaluationConfig
from detect import TrashDetectionPipeline, PipelineConfig

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
            paths = self.config.get('paths', {})
            
            directories = [
                paths.get('data_dir', 'data'),
                paths.get('models_dir', 'models'),
                paths.get('results_dir', 'results'),
                paths.get('logs_dir', 'logs'),
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
        """Step 1: Data preprocessing cho cả detection và classification"""
        try:
            logger.info("=== STEP 1: DATA PREPROCESSING ===")
            
            preprocessing_results = {}
            
            # 1.1 TACO Dataset Processing (Detection)
            logger.info("1.1 Processing TACO dataset for detection...")
            taco_config = self.config.get('datasets', {}).get('taco', {})
            
            # Import DetectionConfig
            from data_preprocessing_detection import DetectionConfig
            
            # Create detection config
            detection_config = DetectionConfig(
                raw_data_dir=Path(taco_config.get('base_dir', 'data/detection/raw')),
                processed_data_dir=Path(taco_config.get('processed_dir', 'data/detection/processed')),
                train_ratio=taco_config.get('train_split', 0.7),
                val_ratio=taco_config.get('val_split', 0.2),
                test_ratio=taco_config.get('test_split', 0.1)
            )
            
            taco_processor = TACODataProcessor(
                raw_data_dir=detection_config.raw_data_dir,
                processed_data_dir=detection_config.processed_data_dir
            )
            
            detection_results = taco_processor.run_preprocessing()
            preprocessing_results['detection'] = detection_results
            
            # 1.2 TrashNet Dataset Processing (Classification)
            logger.info("1.2 Processing TrashNet dataset for classification...")
            trashnet_config = self.config.get('datasets', {}).get('trashnet', {})
            
            # Import ClassificationConfig
            from data_preprocessing_classification import ClassificationConfig
            
            # Create classification config
            classification_config = ClassificationConfig(
                raw_data_dir=Path(trashnet_config.get('base_dir', 'data/classification/raw')),
                processed_data_dir=Path(trashnet_config.get('processed_dir', 'data/classification/processed')),
                train_ratio=trashnet_config.get('train_split', 0.7),
                val_ratio=trashnet_config.get('val_split', 0.2),
                test_ratio=trashnet_config.get('test_split', 0.1)
            )
            
            garbage_processor = GarbageDataProcessor(
                raw_data_dir=classification_config.raw_data_dir,
                processed_data_dir=classification_config.processed_data_dir
            )
            
            classification_results = garbage_processor.run_preprocessing(classification_config)
            preprocessing_results['classification'] = classification_results
            
            self.results['preprocessing'] = preprocessing_results
            
            logger.info("=== DATA PREPROCESSING COMPLETED ===")
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
                data_yaml=det_config.get('data_yaml', 'D:/MasterUIT/Trash-Detection/training-model/data/processed/detection/dataset.yaml'),
                epochs=det_config.get('epochs', 100),
                batch_size=det_config.get('batch_size', 16),
                img_size=det_config.get('img_size', 640),
                learning_rate=det_config.get('learning_rate', 0.01),
                device=det_config.get('device', 'auto'),
                save_dir=Path(det_config.get('save_dir', 'results/detection')),
                experiment_name=det_config.get('experiment_name', 'detection_v1'),
                # Thêm các parameters khác từ config
                weight_decay=det_config.get('weight_decay', 0.0005),
                momentum=det_config.get('momentum', 0.937),
                patience=det_config.get('patience', 50),
                save_period=det_config.get('save_period', 10),
                workers=det_config.get('workers', 8),
                optimizer=det_config.get('optimizer', 'SGD'),
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
                data_yaml=cls_config.get('data_yaml', 'data/classification/processed/dataset_classification.yaml'),
                epochs=cls_config.get('epochs', 50),
                batch_size=cls_config.get('batch_size', 32),
                img_size=cls_config.get('img_size', 224),
                learning_rate=cls_config.get('learning_rate', 0.001),
                device=cls_config.get('device', 'auto'),
                save_dir=Path(cls_config.get('save_dir', 'results/classification')),
                experiment_name=cls_config.get('experiment_name', 'classification_v1'),
                # Thêm các parameters khác từ config
                weight_decay=cls_config.get('weight_decay', 0.0005),
                momentum=cls_config.get('momentum', 0.937),
                patience=cls_config.get('patience', 20),
                save_period=cls_config.get('save_period', 5),
                workers=cls_config.get('workers', 8),
                optimizer=cls_config.get('optimizer', 'AdamW'),
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
            
            # Nếu có training results, sử dụng best weights
            if 'detection_training' in self.results:
                detection_model_path = self.results['detection_training']['model_paths']['best']
            
            if 'classification_training' in self.results:
                classification_model_path = self.results['classification_training']['model_paths']['best']
            
            # Create evaluation config
            evaluation_config = EvaluationConfig(
                detection_model_path=detection_model_path,
                classification_model_path=classification_model_path,
                detection_data_yaml=eval_config.get('detection_data_yaml', 'data/detection/processed/dataset_detection.yaml'),
                classification_data_yaml=eval_config.get('classification_data_yaml', 'data/classification/processed/dataset_classification.yaml'),
                device=eval_config.get('device', 'auto'),
                results_dir=Path(eval_config.get('results_dir', 'results/evaluation')),
                experiment_name=eval_config.get('experiment_name', 'evaluation_v1'),
                save_plots=eval_config.get('save_plots', True),
                show_plots=eval_config.get('show_plots', False),  # Disable interactive plots in pipeline
                detection_conf_thresholds=eval_config.get('detection_conf_thresholds', [0.1, 0.25, 0.5, 0.75])
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
                classification_conf_threshold=pipe_config.get('classification_conf_threshold', 0.5),
                device=pipe_config.get('device', 'auto'),
                max_workers=pipe_config.get('max_workers', 4),
                batch_classification=pipe_config.get('batch_classification', True)
            )
            
            # Initialize pipeline
            pipeline = TrashDetectionPipeline(pipeline_config)
            
            # Test với sample images nếu có
            test_images = []
            sample_dir = Path('data/classification/processed/test')
            if sample_dir.exists():
                # Lấy một vài sample images từ mỗi class
                for class_dir in sample_dir.iterdir():
                    if class_dir.is_dir():
                        images = list(class_dir.glob('*.jpg'))[:2]  # 2 images per class
                        test_images.extend(images)
            
            integration_results = {
                'pipeline_initialized': True,
                'detection_model_loaded': pipeline.detection_model is not None,
                'classification_model_loaded': pipeline.classification_model is not None,
                'test_results': []
            }
            
            # Test trên sample images
            if test_images:
                logger.info(f"Testing pipeline on {len(test_images)} sample images...")
                
                for img_path in test_images[:5]:  # Test chỉ 5 images
                    try:
                        result = pipeline.process_image(str(img_path))
                        integration_results['test_results'].append({
                            'image': str(img_path),
                            'success': True,
                            'detections': result['summary']['total_objects'],
                            'classified': result['summary']['classified_objects']
                        })
                    except Exception as e:
                        integration_results['test_results'].append({
                            'image': str(img_path),
                            'success': False,
                            'error': str(e)
                        })
            
            successful_tests = sum(1 for r in integration_results['test_results'] if r['success'])
            integration_results['success_rate'] = successful_tests / len(integration_results['test_results']) if integration_results['test_results'] else 0
            
            self.results['integration_test'] = integration_results
            
            logger.info("=== PIPELINE INTEGRATION TEST COMPLETED ===")
            logger.info(f"Integration test success rate: {integration_results['success_rate']:.2%}")
            
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
                    'version': '1.0.0'
                },
                'results': self.results,
                'summary': self._create_pipeline_summary()
            }
            
            # Save to JSON
            results_path = Path('results') / 'pipeline_results.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Pipeline results saved to: {results_path}")
            
            # Save summary
            summary_path = Path('results') / 'pipeline_summary.md'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(self._create_summary_markdown())
            
            logger.info(f"Pipeline summary saved to: {summary_path}")
            
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
        
        if 'evaluation' in self.results:
            eval_data = self.results['evaluation']
            if 'detection_evaluation' in eval_data:
                det_eval = eval_data['detection_evaluation']
                if 'best_threshold' in det_eval:
                    best_results = det_eval['threshold_results'][det_eval['best_threshold']]
                    summary['final_detection_mAP50'] = best_results.get('mAP50', 0)
            
            if 'classification_evaluation' in eval_data:
                cls_eval = eval_data['classification_evaluation']
                if 'detailed_metrics' in cls_eval:
                    summary['final_classification_accuracy'] = cls_eval['detailed_metrics'].get('overall_accuracy', 0)
        
        if 'integration_test' in self.results:
            summary['integration_success_rate'] = self.results['integration_test'].get('success_rate', 0)
        
        return summary
    
    def _create_summary_markdown(self) -> str:
        """Tạo summary dạng Markdown"""
        summary = self._create_pipeline_summary()
        
        md_content = f"""# Trash Detection Training Pipeline Results

## Pipeline Information
- **Config File**: {self.config_path}
- **Timestamp**: {datetime.now().isoformat()}
- **Version**: 1.0.0

## Pipeline Status
- ✅ Data Preprocessing: {'✅' if summary['preprocessing_completed'] else '❌'}
- ✅ Detection Training: {'✅' if summary['detection_training_completed'] else '❌'}
- ✅ Classification Training: {'✅' if summary['classification_training_completed'] else '❌'}
- ✅ Comprehensive Evaluation: {'✅' if summary['evaluation_completed'] else '❌'}
- ✅ Integration Testing: {'✅' if summary['integration_test_completed'] else '❌'}

## Key Performance Metrics

### Detection Model
- **Training mAP@50**: {summary.get('detection_mAP50', 'N/A')}
- **Final mAP@50**: {summary.get('final_detection_mAP50', 'N/A')}

### Classification Model  
- **Training Accuracy**: {summary.get('classification_accuracy', 'N/A')}
- **Final Accuracy**: {summary.get('final_classification_accuracy', 'N/A')}

### Pipeline Integration
- **Integration Success Rate**: {f"{summary.get('integration_success_rate', 0):.2%}" if summary.get('integration_success_rate') is not None else 'N/A'}

## Model Files
- **Detection Model**: `models/detection/best.pt`
- **Classification Model**: `models/classification/best.pt`

## Usage Examples

### Training Individual Models
```bash
# Train detection model
python train_detection.py --config configs/training_config.yaml

# Train classification model  
python train_classification.py --config configs/training_config.yaml
```

### Run Complete Pipeline
```bash
# Run full training pipeline
python main.py --config configs/training_config.yaml --full-pipeline

# Run specific steps
python main.py --config configs/training_config.yaml --steps preprocessing,detection
```

### Real-time Detection
```bash
# Webcam detection
python detect.py --source 0

# Video file detection
python detect.py --source video.mp4 --output output.mp4

# Image detection
python detect.py --source image.jpg --output result.jpg
```

### Evaluation
```bash
# Comprehensive evaluation
python evaluate.py --detection-model models/detection/best.pt --classification-model models/classification/best.pt
```

## Directory Structure
```
training-model/
├── data/
│   ├── detection/
│   │   ├── raw/
│   │   └── processed/
│   └── classification/
│       ├── raw/
│       └── processed/
├── models/
│   ├── detection/
│   └── classification/
├── results/
│   ├── detection/
│   ├── classification/
│   └── evaluation/
├── configs/
├── logs/
└── *.py files
```

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return md_content
    
    def run_full_pipeline(self, steps: Optional[str] = None) -> Dict[str, Any]:
        """Chạy toàn bộ training pipeline"""
        try:
            logger.info("=== STARTING FULL TRAINING PIPELINE ===")
            
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
            if 'final_detection_mAP50' in summary:
                print(f"   🎯 Detection mAP@50: {summary['final_detection_mAP50']:.4f}")
            if 'final_classification_accuracy' in summary:
                print(f"   🎯 Classification Accuracy: {summary['final_classification_accuracy']:.4f}")
            if 'integration_success_rate' in summary:
                print(f"   🎯 Integration Success Rate: {summary['integration_success_rate']:.2%}")
            
            print("\n📁 OUTPUTS:")
            print("   📂 Models: models/detection/best.pt, models/classification/best.pt")
            print("   📂 Results: results/pipeline_results.json")
            print("   📂 Summary: results/pipeline_summary.md")
            
            print("\n🚀 NEXT STEPS:")
            print("   1. Test real-time detection: python detect.py --source 0")
            print("   2. Evaluate custom images: python detect.py --source image.jpg")
            print("   3. Run comprehensive evaluation: python evaluate.py")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing final summary: {e}")


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="Trash Detection Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--steps", type=str, default=None,
                       help="Steps to run (comma-separated): preprocessing,detection,classification,evaluation,integration")
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Run full training pipeline")
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo pipeline
        pipeline = TrashDetectionTrainingPipeline(args.config)
        
        if args.full_pipeline or args.steps:
            # Chạy pipeline
            results = pipeline.run_full_pipeline(args.steps)
            logger.info("Training pipeline completed successfully!")
        else:
            # Interactive mode
            print("Available commands:")
            print("1. Run full pipeline: --full-pipeline")
            print("2. Run specific steps: --steps preprocessing,detection,classification")
            print("3. Example: python main.py --config configs/training_config.yaml --full-pipeline")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
