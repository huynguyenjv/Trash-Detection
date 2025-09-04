"""
FastAPI Backend for Smart Waste Detection System
Handles YOLOv8 detection, WebSocket streaming, and pathfinding
"""
import asyncio
import json
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from detector import get_detector
from waste_manager import get_waste_manager  
from pathfinding import get_pathfinder


# Pydantic models
class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image
    confidence_threshold: float = 0.25  # Lower default threshold


class FrameData(BaseModel):
    type: str
    image: str  # Base64 encoded image


# Initialize FastAPI app
app = FastAPI(
    title="Smart Waste Detection API",
    description="YOLOv8-powered waste detection with A* pathfinding",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
detector = None
waste_manager = None
pathfinder = None


class ConnectionManager:
    """Manages WebSocket connections"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# WebSocket connection manager
manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global detector, waste_manager, pathfinder
    
    print("🚀 Starting Smart Waste Detection System...")
    
    # Initialize detector (try to use custom model if available)
    try:
        # Try to load custom model first
        model_paths = [
            "../../models/final.pt",
            "../models/final.pt", 
            "./models/final.pt",
            "final.pt"
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                detector = get_detector(model_path)
                model_loaded = True
                print(f"✅ Loaded custom model: {model_path}")
                break
            except:
                continue
        
        if not model_loaded:
            detector = get_detector()  # Use default YOLOv8n
            print("✅ Loaded default YOLOv8n model")
        
        # Test model functionality
        print("🧪 Testing model functionality...")
        detector.test_model_basic()
            
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        detector = get_detector()  # Fallback to default
    
    # Initialize waste manager
    waste_manager = get_waste_manager()
    print("✅ Waste manager initialized")
    
    # Initialize pathfinder
    pathfinder = get_pathfinder()
    print("✅ A* pathfinder initialized")
    
    print("🎯 Backend ready for connections!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Smart Waste Detection API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "detection": "/detect",
            "stats": "/stats", 
            "path": "/path",
            "websocket": "/ws/detect"
        }
    }


@app.post("/detect")
async def detect_waste(request: DetectionRequest):
    """
    Detect waste in uploaded image
    """
    try:
        if not detector:
            raise HTTPException(status_code=500, detail="Detector not initialized")
        
        # Run detection
        detections = detector.detect_from_base64(
            request.image, 
            request.confidence_threshold
        )
        
        # Update waste manager statistics
        if waste_manager:
            waste_manager.update_stats(detections)
        
        # Get detection summary
        summary = detector.get_detection_summary(detections)
        
        return {
            "success": True,
            "detections": detections,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get current waste statistics"""
    try:
        if not waste_manager:
            raise HTTPException(status_code=500, detail="Waste manager not initialized")
        
        stats = waste_manager.get_current_stats()
        trends = waste_manager.get_trend_data()
        
        return {
            "success": True,
            "current": stats,
            "trends": trends,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bins")
async def get_bins():
    """Get all waste bin locations"""
    try:
        if not waste_manager:
            raise HTTPException(status_code=500, detail="Waste manager not initialized")
        
        bins = waste_manager.get_all_bins()
        
        return {
            "success": True,
            "bins": bins,
            "count": len(bins),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Bins error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/path")
async def get_path(
    lat: float = Query(..., description="Starting latitude"),
    lon: float = Query(..., description="Starting longitude"),
    dest_lat: float = Query(None, description="Destination latitude"),
    dest_lon: float = Query(None, description="Destination longitude"),
    waste_type: str = Query("other", description="Type of waste for optimal bin selection")
):
    """
    Calculate optimal path to nearest appropriate waste bin
    """
    try:
        if not waste_manager or not pathfinder:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        # If destination not specified, find nearest appropriate bin
        if dest_lat is None or dest_lon is None:
            nearest_bin = waste_manager.find_nearest_bin(lat, lon, waste_type)
            if not nearest_bin:
                raise HTTPException(status_code=404, detail="No suitable waste bin found")
            
            dest_lat = nearest_bin['lat']
            dest_lon = nearest_bin['lon']
            bin_info = nearest_bin
        else:
            bin_info = {"lat": dest_lat, "lon": dest_lon, "name": "Custom destination"}
        
        # Calculate route using A*
        route = pathfinder.calculate_route(lat, lon, dest_lat, dest_lon)
        
        # Return in the format expected by frontend
        return {
            "success": True,
            "path": route.get("path", [[lat, lon], [dest_lat, dest_lon]]),
            "distance": route.get("distance", 1000),
            "duration": route.get("duration", 300),
            "found_path": route.get("found_path", False),
            "destination": bin_info,
            "start": {"lat": lat, "lon": lon},
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Path error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection
    """
    await manager.connect(websocket)
    print("🔌 Client connected to WebSocket")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                frame_data = json.loads(data)
                
                if frame_data.get("type") == "frame":
                    image_data = frame_data.get("image")
                    
                    if image_data and detector:
                        # Run detection
                        detections = detector.detect_from_base64(image_data)
                        
                        # Update statistics and broadcast
                        if waste_manager:
                            waste_manager.update_stats(detections)
                            
                            # Broadcast updated stats to all clients
                            stats = waste_manager.get_current_stats()
                            # Convert datetime to string for JSON serialization
                            if 'last_updated' in stats and stats['last_updated']:
                                stats['last_updated'] = stats['last_updated'].isoformat()
                            
                            await manager.broadcast({
                                "type": "stats_update",
                                "stats": stats,
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        # Debug: Print detection data being sent
                        print(f"🔍 Sending {len(detections)} detections to frontend")
                        for i, det in enumerate(detections[:3]):  # Print first 3 detections
                            print(f"   Detection {i}: {det}")
                        
                        # Send results back to client
                        response = {
                            "type": "detection_result",
                            "detections": detections,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        await websocket.send_text(json.dumps(response))
                        
                elif frame_data.get("type") == "ping":
                    # Handle ping
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON data"
                }))
            except Exception as e:
                print(f"WebSocket processing error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": str(e)
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("🔌 Client disconnected from WebSocket")
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """
    WebSocket endpoint for real-time statistics
    """
    await manager.connect(websocket)
    print("📊 Stats client connected to WebSocket")
    
    try:
        # Send initial stats
        if waste_manager:
            stats = waste_manager.get_current_stats()
            # Convert datetime to string
            if 'last_updated' in stats and stats['last_updated']:
                stats['last_updated'] = stats['last_updated'].isoformat()
            
            initial_message = {
                "type": "stats_update",
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(initial_message))
        
        # Keep connection alive and send periodic updates
        while True:
            try:
                # Wait for any message or timeout after 30 seconds
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle ping messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON
                    
            except asyncio.TimeoutError:
                # Send periodic stats update
                if waste_manager:
                    stats = waste_manager.get_current_stats()
                    # Convert datetime to string
                    if 'last_updated' in stats and stats['last_updated']:
                        stats['last_updated'] = stats['last_updated'].isoformat()
                    
                    heartbeat_message = {
                        "type": "heartbeat",
                        "stats": stats,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(heartbeat_message))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("📊 Stats client disconnected from WebSocket")
    except Exception as e:
        print(f"Stats WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    print("🗑️ Smart Waste Detection Backend Starting...")
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
