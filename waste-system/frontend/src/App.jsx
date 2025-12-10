import VideoStream from './components/VideoStream';
import RealTimeStats from './components/RealTimeStats';
import ControlPanel from './components/ControlPanel';
import MapView from './components/MapView';
import { useState } from 'react';

function App() {
  const [showMap, setShowMap] = useState(false);
  const [detectedWaste, setDetectedWaste] = useState(null);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">🗑️</span>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Smart Waste Detection System
                </h1>
                <p className="text-sm text-gray-600">
                  AI-powered waste detection and routing
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowMap(!showMap)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  showMap 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {showMap ? '📹 Video' : '🗺️ Bản đồ'}
              </button>
              <div className="text-right">
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">
                  YOLOv8 Detection
                </p>
                <p className="text-xs text-gray-500">
                  Real-time Analysis
                </p>
              </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
          {/* Left Column - Video/Map and Stats */}
          <div className="lg:col-span-2 space-y-6">
            {/* Video Stream or Map View */}
            <div className="h-auto">
              {showMap ? (
                <MapView 
                  autoFindRoute={false}
                  detectedWaste={detectedWaste}
                />
              ) : (
                <VideoStream onWasteDetected={setDetectedWaste} />
              )}
            </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
          {/* Left Column - Video and Stats */}
          <div className="lg:col-span-2 space-y-6">
            {/* Video Stream */}
            <div className="h-auto">
              <VideoStream />
            </div>
            
            {/* Real-time Statistics */}
            <div className="h-auto">
              <RealTimeStats />
            </div>
          </div>

          {/* Right Column - Controls */}
          <div className="lg:col-span-1">
            <div className="sticky top-6">
              <ControlPanel />
            </div>
          <div className="text-center text-sm text-gray-600">
            <p>© 2024 Smart Waste Detection System</p>
            <p className="mt-1">
              Powered by YOLOv8, FastAPI, React & Goong Maps (Weighted Score + Dijkstra + A*)
            </p>
          </div>assName="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="text-center text-sm text-gray-600">
            <p>© 2024 Smart Waste Detection System</p>
            <p className="mt-1">
              Powered by YOLOv8, FastAPI, React & A* Pathfinding
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
