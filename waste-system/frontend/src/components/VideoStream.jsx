import { useEffect, useRef, useState } from 'react';
import DetectionSettings from './DetectionSettings';

const VideoStream = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detections, setDetections] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [ws, setWs] = useState(null);
  const [detectionSettings, setDetectionSettings] = useState({
    confidence: 0.5,
    autoDetect: true,
    minObjectSize: 10
  });
  const [stats, setStats] = useState({
    total: 0,
    organic: 0,
    recyclable: 0,
    hazardous: 0,
    other: 0
  });

  useEffect(() => {
    // Initialize WebSocket connection with retry logic
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    let reconnectTimeout;

    const connectWebSocket = () => {
      try {
        const websocket = new WebSocket('ws://localhost:8000/ws/detect');
        
        websocket.onopen = () => {
          console.log('✅ WebSocket connected successfully');
          setWs(websocket);
          reconnectAttempts = 0; // Reset counter on successful connection
        };

        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('📥 Received data:', data);
            if (data.detections) {
              setDetections(data.detections);
              drawDetections(data.detections);
            }
          } catch (parseError) {
            console.error('❌ Error parsing WebSocket data:', parseError);
          }
        };

        websocket.onclose = (event) => {
          console.log('🔌 WebSocket disconnected:', event.code, event.reason);
          setWs(null);
          
          // Attempt to reconnect if not intentional close
          if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`🔄 Attempting to reconnect... (${reconnectAttempts}/${maxReconnectAttempts})`);
            reconnectTimeout = setTimeout(connectWebSocket, 2000 * reconnectAttempts);
          }
        };

        websocket.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          console.log('🔍 Make sure backend is running on http://localhost:8000');
          console.log('🔍 Check if WebSocket endpoint /ws/detect is available');
        };

        return websocket;
      } catch (error) {
        console.error('❌ Failed to create WebSocket:', error);
        return null;
      }
    };

    const websocket = connectWebSocket();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.close(1000, 'Component unmounting');
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Cannot access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  const drawDetections = (detections) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;
    
    const ctx = canvas.getContext('2d');
    const { videoWidth, videoHeight } = video;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw detection boxes
    detections.forEach(detection => {
      const { bbox, label, confidence } = detection;
      const [x, y, width, height] = bbox;
      
      // Scale bbox to canvas size
      const scaleX = canvas.width / videoWidth;
      const scaleY = canvas.height / videoHeight;
      
      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
      const scaledWidth = width * scaleX;
      const scaledHeight = height * scaleY;
      
      // Draw bounding box
      ctx.strokeStyle = confidence > 0.7 ? '#10b981' : '#f59e0b';
      ctx.lineWidth = 2;
      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
      
      // Draw label background
      const labelText = `${label} ${Math.round(confidence * 100)}%`;
      ctx.font = '14px Arial';
      const textMetrics = ctx.measureText(labelText);
      const textWidth = textMetrics.width;
      
      ctx.fillStyle = confidence > 0.7 ? '#10b981' : '#f59e0b';
      ctx.fillRect(scaledX, scaledY - 25, textWidth + 10, 25);
      
      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(labelText, scaledX + 5, scaledY - 8);
    });
  };

  const sendFrameToBackend = () => {
    if (!videoRef.current || !ws || ws.readyState !== WebSocket.OPEN) return;
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64 and send
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    ws.send(JSON.stringify({ 
      type: 'frame',
      image: imageData.split(',')[1] // Remove data:image/jpeg;base64, prefix
    }));
  };

  // Send frames periodically when streaming
  useEffect(() => {
    let interval;
    if (isStreaming && ws) {
      interval = setInterval(sendFrameToBackend, 100); // 10 FPS
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isStreaming, ws]);

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800">
          Live Detection Stream
        </h2>
        <div className="space-x-2">
          {!isStreaming ? (
            <button
              onClick={startCamera}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
              Start Camera
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              Stop Camera
            </button>
          )}
        </div>
      </div>
      
      <div className="relative bg-gray-100 rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-auto"
          style={{ maxHeight: '400px' }}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          width="640"
          height="480"
        />
        
        {!isStreaming && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-200">
            <div className="text-center">
              <div className="text-gray-400 text-4xl mb-2">📹</div>
              <p className="text-gray-600">Click "Start Camera" to begin detection</p>
            </div>
          </div>
        )}
      </div>
      
      <div className="mt-4 flex items-center justify-between text-sm text-gray-600">
        <div className="flex items-center space-x-4">
          <span className="flex items-center">
            <div className={`w-2 h-2 rounded-full mr-2 ${
              isStreaming ? 'bg-green-500' : 'bg-gray-400'
            }`}></div>
            {isStreaming ? 'Streaming' : 'Not streaming'}
          </span>
          <span className="flex items-center">
            <div className={`w-2 h-2 rounded-full mr-2 ${
              ws && ws.readyState === WebSocket.OPEN ? 'bg-blue-500' : 'bg-red-500'
            }`}></div>
            {ws && ws.readyState === WebSocket.OPEN ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <span>Detections: {detections.length}</span>
      </div>
    </div>
  );
};

export default VideoStream;
