"""
Demo chạy hệ thống theo dõi rác thải real-time với camera
Kết hợp YOLOv8 detection + Smart Routing System

Usage:
    python demo_realtime.py --model models/trash_safe_best.pt --camera 0
    python demo_realtime.py --model models/trash_safe_best.pt --video video.mp4
    python demo_realtime.py --model models/trash_safe_best.pt --image image.jpg

Author: Smart Waste Management System
Date: August 2025
"""

import cv2
import argparse
import time
import numpy as np
from pathlib import Path
import logging

from smart_routing_system import (
    RealTimeWasteDetector, SmartRoutingSystem, create_sample_data,
    WasteType, MapVisualizer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WasteDetectionDemo:
    """Demo class cho waste detection real-time"""
    
    def __init__(self, model_path: str, threshold: int = 10):
        """
        Args:
            model_path: Đường dẫn đến YOLOv8 model
            threshold: Threshold để trigger routing
        """
        # Tạo routing system với dữ liệu mẫu
        self.routing_system = create_sample_data()
        
        # Tạo detector
        self.detector = RealTimeWasteDetector(model_path, self.routing_system)
        self.detector.waste_counter.threshold = threshold
        
        # Tracking variables
        self.handled_types = set()
        self.last_route_time = 0
        self.route_cooldown = 30  # seconds giữa các lần tìm đường
        
        logger.info(f"Initialized demo with model: {model_path}")
        logger.info(f"Threshold for routing: {threshold}")
    
    def process_camera(self, camera_id: int = 0):
        """Xử lý camera real-time"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return
        
        logger.info(f"Starting camera {camera_id}. Press 'q' to quit, 'r' to reset counters")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, triggered_types = self.detector.process_frame(frame)
            
            # Handle triggered types
            self._handle_triggered_types(triggered_types)
            
            # Add instructions
            self._draw_instructions(annotated_frame)
            
            # Display
            cv2.imshow('Smart Waste Detection', annotated_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset_counters()
            elif key == ord('s'):
                self._save_current_state()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video(self, video_path: str):
        """Xử lý video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video {video_path}")
            return
        
        logger.info(f"Processing video: {video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                annotated_frame, triggered_types = self.detector.process_frame(frame)
                self._handle_triggered_types(triggered_types)
            else:
                annotated_frame = frame
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                       (10, annotated_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self._draw_instructions(annotated_frame)
            
            # Display
            cv2.imshow('Smart Waste Detection - Video', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset_counters()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path: str):
        """Xử lý single image"""
        frame = cv2.imread(image_path)
        
        if frame is None:
            logger.error(f"Cannot load image {image_path}")
            return
        
        logger.info(f"Processing image: {image_path}")
        
        # Process image
        annotated_frame, triggered_types = self.detector.process_frame(frame)
        self._handle_triggered_types(triggered_types)
        
        self._draw_instructions(annotated_frame)
        
        # Display
        cv2.imshow('Smart Waste Detection - Image', annotated_frame)
        
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _handle_triggered_types(self, triggered_types: list):
        """Xử lý các loại rác đạt threshold"""
        current_time = time.time()
        
        for waste_type in triggered_types:
            # Kiểm tra cooldown
            if (waste_type not in self.handled_types and 
                current_time - self.last_route_time > self.route_cooldown):
                
                logger.info(f"🚨 THRESHOLD REACHED: {waste_type.value}")
                
                # Tìm đường
                result = self.detector.handle_threshold_reached(waste_type)
                
                if result:
                    self.last_route_time = current_time
                    self.handled_types.add(waste_type)
                    
                    # Log kết quả
                    logger.info(f"📍 Route found to {result.target_bin.id}")
                    logger.info(f"📏 Distance: {result.total_distance:.2f}km")
                    logger.info(f"⏱️ ETA: {result.estimated_time:.1f}min")
                    
                    # Hiển thị map (non-blocking)
                    self._show_route_async(result, waste_type)
    
    def _show_route_async(self, result, waste_type):
        """Hiển thị route map không chặn luồng chính"""
        try:
            fig = MapVisualizer.plot_route(self.routing_system, result, waste_type)
            # Save thay vì show để không block
            filename = f"route_{waste_type.value}_{int(time.time())}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"📁 Route map saved: {filename}")
            import matplotlib.pyplot as plt
            plt.close(fig)  # Close để giải phóng memory
        except Exception as e:
            logger.error(f"Error saving route map: {e}")
    
    def _draw_instructions(self, frame):
        """Vẽ hướng dẫn lên frame"""
        instructions = [
            "Controls:",
            "Q - Quit",
            "R - Reset counters", 
            "S - Save state"
        ]
        
        y_start = frame.shape[0] - 120
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (frame.shape[1] - 200, y_start + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _reset_counters(self):
        """Reset tất cả counters"""
        for waste_type in WasteType:
            self.detector.waste_counter.reset_type(waste_type)
        self.handled_types.clear()
        logger.info("🔄 All counters reset")
    
    def _save_current_state(self):
        """Lưu trạng thái hiện tại"""
        timestamp = int(time.time())
        state_info = {
            'timestamp': timestamp,
            'counts': dict(self.detector.waste_counter.counts),
            'robot_position': {
                'lat': self.routing_system.current_position.lat,
                'lng': self.routing_system.current_position.lng
            } if self.routing_system.current_position else None,
            'handled_types': [wt.value for wt in self.handled_types]
        }
        
        import json
        filename = f"waste_state_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(state_info, f, indent=2)
        
        logger.info(f"💾 State saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Smart Waste Detection Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLOv8 model file (.pt)')
    parser.add_argument('--camera', type=int, 
                       help='Camera ID (e.g., 0 for default camera)')
    parser.add_argument('--video', type=str,
                       help='Path to video file')
    parser.add_argument('--image', type=str,
                       help='Path to image file')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Threshold for triggering routing (default: 10)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    sources = [args.camera is not None, args.video is not None, args.image is not None]
    if sum(sources) != 1:
        print("❌ Please specify exactly one input source: --camera, --video, or --image")
        return
    
    if args.video and not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return
    
    if args.image and not Path(args.image).exists():
        print(f"❌ Image file not found: {args.image}")
        return
    
    # Create demo
    try:
        demo = WasteDetectionDemo(args.model, args.threshold)
        
        # Run appropriate processing
        if args.camera is not None:
            demo.process_camera(args.camera)
        elif args.video:
            demo.process_video(args.video)
        elif args.image:
            demo.process_image(args.image)
            
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    main()
