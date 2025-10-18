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
  const [sessionStats, setSessionStats] = useState({
    total: 0,
    organic: 0,
    recyclable: 0,
    hazardous: 0,
    other: 0,
    startTime: null,
    detectionHistory: []
  });
  const [showSessionSummary, setShowSessionSummary] = useState(false);

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
            
            if (data.detections) {
              console.log('🎯 Frontend received detections:', data.detections.length);
              if (data.detections.length > 0) {
                console.log('📦 Detection details:', data.detections.map(d => ({
                  label: d.label,
                  category: d.category,
                  confidence: d.confidence,
                  bbox: d.bbox,
                  size: `${d.bbox[2] - d.bbox[0]}x${d.bbox[3] - d.bbox[1]}`
                })));
              }
              setDetections(data.detections);
              drawDetections(data.detections);
              
              // Update session statistics
              updateSessionStats(data.detections);
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

  const updateSessionStats = (detections) => {
    const categoryCounts = {
      total: detections.length,
      organic: 0,
      recyclable: 0,
      hazardous: 0,
      other: 0
    };
    
    detections.forEach(detection => {
      const category = detection.category || 'other';
      if (category in categoryCounts) {
        categoryCounts[category]++;
      }
    });
    
    setSessionStats(prev => ({
      ...prev,
      ...categoryCounts,
      detectionHistory: [...prev.detectionHistory.slice(-19), {
        timestamp: new Date(),
        detections: detections.length,
        categories: categoryCounts
      }]
    }));
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait for video metadata to load before proceeding
        await new Promise((resolve) => {
          videoRef.current.onloadedmetadata = () => {
            console.log(`📹 Video metadata loaded: ${videoRef.current.videoWidth}x${videoRef.current.videoHeight}`);
            resolve();
          };
        });
        
        setIsStreaming(true);
        setSessionStats(prev => ({
          ...prev,
          startTime: new Date(),
          detectionHistory: []
        }));
        setShowSessionSummary(false);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Cannot access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
      
      // Show session summary
      setShowSessionSummary(true);
    }
  };

  const drawDetections = (detections) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;
    
    const ctx = canvas.getContext('2d');
    
    // Get actual rendered video dimensions
    const displayWidth = video.offsetWidth;
    const displayHeight = video.offsetHeight;
    
    // Set canvas size to match displayed video size exactly
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // TESTING: Backend now returns RAW coordinates, we need to scale them
    const streamWidth = video.videoWidth || 640;
    const streamHeight = video.videoHeight || 480;
    const scaleX = displayWidth / streamWidth;
    const scaleY = displayHeight / streamHeight;
    
    console.log(`🎯 Canvas: ${displayWidth}x${displayHeight}, Stream: ${streamWidth}x${streamHeight}, Scale: ${scaleX.toFixed(3)}x${scaleY.toFixed(3)}`);
    
    // Draw detection boxes
    detections.forEach((detection, index) => {
      const { bbox, label, confidence, category } = detection;
      
      // Backend sends RAW coordinates, need to scale them
      const [x1_raw, y1_raw, x2_raw, y2_raw] = bbox;
      
      // Scale to display size
      const x1 = Math.round(x1_raw * scaleX);
      const y1 = Math.round(y1_raw * scaleY);
      const x2 = Math.round(x2_raw * scaleX);
      const y2 = Math.round(y2_raw * scaleY);
      
      const width = x2 - x1;
      const height = y2 - y1;
      
      // Debug log for first few detections
      if (index < 3) {
        console.log(`🎯 Detection ${index}: Raw [${x1_raw}, ${y1_raw}, ${x2_raw}, ${y2_raw}] -> Scaled [${x1}, ${y1}, ${x2}, ${y2}] (${width}x${height})`);
      }
      
      // Category-based colors
      const categoryColors = {
        'organic': '#10b981',      // Green
        'recyclable': '#3b82f6',   // Blue
        'hazardous': '#ef4444',    // Red
        'other': '#f59e0b'         // Orange
      };
      
      const color = categoryColors[category] || '#6b7280';
      
      // Draw bounding box with better visibility
      ctx.strokeStyle = color;
      ctx.lineWidth = confidence > 0.7 ? 3 : 2;
      ctx.strokeRect(x1, y1, width, height);
      
      // Draw corner markers for precise positioning
      const cornerSize = 8;
      ctx.fillStyle = color;
      
      // Top-left corner
      ctx.fillRect(x1 - 2, y1 - 2, cornerSize, 2);
      ctx.fillRect(x1 - 2, y1 - 2, 2, cornerSize);
      
      // Top-right corner  
      ctx.fillRect(x2 - cornerSize + 2, y1 - 2, cornerSize, 2);
      ctx.fillRect(x2 - 2, y1 - 2, 2, cornerSize);
      
      // Bottom-left corner
      ctx.fillRect(x1 - 2, y2 - cornerSize + 2, 2, cornerSize);
      ctx.fillRect(x1 - 2, y2 - 2, cornerSize, 2);
      
      // Bottom-right corner
      ctx.fillRect(x2 - 2, y2 - cornerSize + 2, 2, cornerSize);
      ctx.fillRect(x2 - cornerSize + 2, y2 - 2, cornerSize, 2);
      
      // Draw center point
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 3, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      
      // Draw label with better positioning
      const labelText = `${label} (${category})`;
      const confText = `${Math.round(confidence * 100)}%`;
      
      ctx.font = '12px Arial';
      const labelMetrics = ctx.measureText(labelText);
      const confMetrics = ctx.measureText(confText);
      const maxWidth = Math.max(labelMetrics.width, confMetrics.width);
      
      // Position label above box if possible, otherwise below
      let labelY = y1 - 30;
      if (labelY < 0) {
        labelY = y2 + 30;
      }
      
      // Draw label background
      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY - 20, maxWidth + 10, 25);
      
      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(labelText, x1 + 5, labelY - 8);
      ctx.font = '10px Arial';
      ctx.fillText(confText, x1 + 5, labelY - 2);
      
      // Draw object number
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x2 - 15, y1 + 15, 12, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.font = 'bold 10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText((index + 1).toString(), x2 - 15, y1 + 19);
      ctx.textAlign = 'left';
    });
  };

  // Handle video resize and ensure canvas stays in sync
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current && videoRef.current && isStreaming) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        
        // Update canvas size to match video display size
        canvas.width = video.offsetWidth;
        canvas.height = video.offsetHeight;
        
        console.log(`🔄 Canvas resized to: ${canvas.width}x${canvas.height}`);
      }
    };

    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [isStreaming]);

  // Send frames periodically when streaming
  useEffect(() => {
    let interval;
    if (isStreaming && ws) {
      interval = setInterval(() => {
        if (!videoRef.current || !ws || ws.readyState !== WebSocket.OPEN) return;
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const video = videoRef.current;
        
        // Use actual video stream dimensions for backend processing
        const streamWidth = video.videoWidth || 640;
        const streamHeight = video.videoHeight || 480;
        const displayWidth = video.offsetWidth;
        const displayHeight = video.offsetHeight;
        
        canvas.width = streamWidth;
        canvas.height = streamHeight;
        ctx.drawImage(video, 0, 0);
        
        // Convert to base64 and send with all dimension info
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Debug log dimensions every 30 frames (3 seconds at 10 FPS)
        if (Math.random() < 0.1) {
          console.log(`📸 Sending dimensions: Stream(${streamWidth}x${streamHeight}) Display(${displayWidth}x${displayHeight})`);
        }
        
        ws.send(JSON.stringify({ 
          type: 'frame',
          image: imageData.split(',')[1], // Remove data:image/jpeg;base64, prefix
          dimensions: {
            streamWidth: streamWidth,
            streamHeight: streamHeight,
            displayWidth: displayWidth,
            displayHeight: displayHeight
          }
        }));
      }, 100); // 10 FPS
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isStreaming, ws]);

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800">
          Live Detection Stream
        </h2>
        <div className="space-x-2">
          {!isStreaming ? (
            <button
              onClick={startCamera}
              className="px-4 py-2 text-white transition-colors bg-green-600 rounded-md hover:bg-green-700"
            >
              Start Camera
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="px-4 py-2 text-white transition-colors bg-red-600 rounded-md hover:bg-red-700"
            >
              Stop Camera
            </button>
          )}
        </div>
      </div>
      
      <div className="relative overflow-hidden bg-gray-100 rounded-lg">
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
          style={{ 
            width: '100%', 
            height: '100%',
            objectFit: 'contain' 
          }}
        />
        
        {!isStreaming && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-200">
            <div className="text-center">
              <div className="mb-2 text-4xl text-gray-400">📹</div>
              <p className="text-gray-600">Click "Start Camera" to begin detection</p>
            </div>
          </div>
        )}
      </div>
      
      <div className="flex items-center justify-between mt-4 text-sm text-gray-600">
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
      
      {/* Session Summary Modal */}
      {showSessionSummary && sessionStats.startTime && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="w-full max-w-md p-6 mx-4 bg-white rounded-lg">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">📊 Session Summary</h3>
              <button
                onClick={() => setShowSessionSummary(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              {/* Session Duration */}
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {Math.round((new Date() - sessionStats.startTime) / 1000 / 60)} min
                </div>
                <div className="text-sm text-gray-600">Session Duration</div>
              </div>
              
              {/* Total Detections */}
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">
                  {sessionStats.total}
                </div>
                <div className="text-sm text-gray-600">Total Objects Detected</div>
              </div>
              
              {/* Category Breakdown */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 text-center rounded-lg bg-green-50">
                  <div className="text-lg font-bold text-green-600">{sessionStats.organic}</div>
                  <div className="text-xs text-green-600">🍂 Organic</div>
                </div>
                <div className="p-3 text-center rounded-lg bg-blue-50">
                  <div className="text-lg font-bold text-blue-600">{sessionStats.recyclable}</div>
                  <div className="text-xs text-blue-600">♻️ Recyclable</div>
                </div>
                <div className="p-3 text-center rounded-lg bg-red-50">
                  <div className="text-lg font-bold text-red-600">{sessionStats.hazardous}</div>
                  <div className="text-xs text-red-600">⚠️ Hazardous</div>
                </div>
                <div className="p-3 text-center rounded-lg bg-gray-50">
                  <div className="text-lg font-bold text-gray-600">{sessionStats.other}</div>
                  <div className="text-xs text-gray-600">🗑️ Other</div>
                </div>
              </div>
              
              {/* Detection Timeline */}
              {sessionStats.detectionHistory.length > 0 && (
                <div>
                  <h4 className="mb-2 text-sm font-medium text-gray-700">📈 Detection Timeline</h4>
                  <div className="space-y-1 overflow-y-auto max-h-32">
                    {sessionStats.detectionHistory.slice(-5).reverse().map((entry) => (
                      <div key={entry.timestamp.getTime()} className="flex justify-between py-1 text-xs text-gray-600 border-b">
                        <span>{entry.timestamp.toLocaleTimeString()}</span>
                        <span className="font-medium">{entry.detections} objects</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <button
                onClick={() => setShowSessionSummary(false)}
                className="w-full px-4 py-2 text-white transition-colors bg-blue-600 rounded-md hover:bg-blue-700"
              >
                Close Summary
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoStream;
