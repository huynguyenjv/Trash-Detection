"""
FastAPI Backend for Waste Detection System
Main application with REST API and WebSocket support
"""

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import time
import asyncio
from datetime import datetime

from detector import WasteDetector
from waste_manager import WasteManager
from pathfinding import AStarPathfinder


# Initialize app
app = FastAPI(title="Waste Detection Backend", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
detector = WasteDetector('yolov8n.pt')  # Use YOLOv8n default
waste_manager = WasteManager()
pathfinder = AStarPathfinder()


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "version": "2.0",
        "model": "YOLOv8n",
        "message": "Waste Detection Backend"
    }


@app.post("/detect")
async def detect_images(files: List[UploadFile] = File(...)):
    """
    Detect waste objects in one or more images
    
    Args:
        files: List of uploaded image files
        
    Returns:
        Detection results with summary statistics
    """
    results = []
    summaries = []
    
    for file in files:
        # Read image
        image_bytes = await file.read()
        frame = detector.bytes_to_frame(image_bytes)
        
        # Detect
        timestamp = time.time()
        detections = detector.detect(frame, conf_threshold=0.25, iou_threshold=0.45)
        
        # Update stats
        counts = waste_manager.update(detections)
        
        # Add result
        results.append({
            'timestamp': timestamp,
            'detections': detections
        })
        
        summaries.append({
            'timestamp': timestamp,
            'counts': counts
        })
    
    return {
        'count': len(results),
        'results': results,
        'summaries': summaries
    }


@app.get("/stats")
async def get_stats(limit: int = 20):
    """
    Get current waste statistics
    
    Args:
        limit: Number of recent detections to return
        
    Returns:
        Total counts and recent detections
    """
    stats = waste_manager.get_stats(limit=limit)
    return stats


@app.get("/path")
async def get_paths(starts: str):
    """
    Get optimal paths from waste locations to nearest bins
    
    Args:
        starts: Comma-separated list of coordinates (e.g., "5,5;10,17")
        
    Returns:
        Paths dictionary with bin, path, and distance for each start
    """
    # Parse start positions
    try:
        start_positions = []
        for pos_str in starts.split(';'):
            x, y = map(int, pos_str.split(','))
            start_positions.append((x, y))
    except:
        return {"error": "Invalid start positions format"}
    
    # Find paths
    paths = pathfinder.find_nearest_bin_for_each(start_positions)
    
    return {"paths": paths}


@app.websocket("/ws/detect")
async def websocket_detection(websocket: WebSocket):
    """
    WebSocket endpoint for realtime detection
    
    Client sends: JSON with base64 image
    Server sends: JSON detection results
    """
    await websocket.accept()
    print("🔌 WebSocket connected")
    
    try:
        while True:
            # Receive data (JSON)
            data = await websocket.receive_json()
            
            # Extract base64 image
            image_base64 = data.get('image', '')
            if not image_base64:
                continue
            
            # Convert base64 to bytes
            import base64
            image_bytes = base64.b64decode(image_base64)
            
            # Convert to frame
            frame = detector.bytes_to_frame(image_bytes)
            
            if frame is None:
                print("⚠️ Failed to decode frame")
                continue
            
            # Detect
            timestamp = time.time()
            detections = detector.detect(frame, conf_threshold=0.25, iou_threshold=0.45)
            
            # Update stats (optional - for tracking)
            waste_manager.update(detections)
            
            # Send result
            result = {
                'timestamp': timestamp,
                'detections': detections
            }
            
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        import traceback
        print(f"❌ WebSocket error: {e}")
        print(traceback.format_exc())


@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """
    WebSocket endpoint for realtime statistics
    
    Server sends: JSON stats every second
    """
    await websocket.accept()
    print("📊 Stats WebSocket connected")
    
    try:
        while True:
            # Get current stats
            stats = waste_manager.get_stats()
            
            # Send stats
            await websocket.send_json(stats)
            
            # Wait 1 second before next update
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print("📊 Stats WebSocket disconnected")
    except Exception as e:
        print(f"❌ Stats WebSocket error: {e}")


@app.post("/reset")
async def reset_stats():
    """Reset all statistics"""
    waste_manager.reset()
    return {"message": "Statistics reset successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
