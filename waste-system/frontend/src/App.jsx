import { useState } from 'react';
import VideoStream from './components/VideoStream';
import RealTimeStats from './components/RealTimeStats';
import ControlPanel from './components/ControlPanel';
import MapView from './components/MapView';

function App() {
  // State for auto-routing when waste is detected
  const [detectedWaste, setDetectedWaste] = useState(null);
  const [showMap, setShowMap] = useState(false);

  // Callback when VideoStream detects waste above threshold
  const handleWasteDetected = (wasteInfo) => {
    console.log('🎯 App received waste detection:', wasteInfo);
    setDetectedWaste(wasteInfo);
    setShowMap(true); // Auto-show map when waste detected
  };

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
                </p>
              </div>
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
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
          {/* Left Column - Video and Stats */}
          <div className="lg:col-span-2 space-y-6">
            {/* Video Stream - with waste detection callback */}
            <div className="h-auto">
              <VideoStream 
                onWasteDetected={handleWasteDetected}
                routeThreshold={0.7}  // Trigger route when confidence >= 70%
              />
            </div>
            
            {/* Map View - Auto-show when waste detected */}
            {showMap && (
              <div className="h-96">
                <MapView 
                  autoFindRoute={detectedWaste !== null}
                  detectedWaste={detectedWaste}
                />
              </div>
            )}

            {/* Toggle Map Button */}
            <div className="flex justify-center">
              <button
                onClick={() => setShowMap(!showMap)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  showMap 
                    ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {showMap ? '🗺️ Hide Map' : '🗺️ Show Map'}
              </button>
              {detectedWaste && (
                <button
                  onClick={() => setDetectedWaste(null)}
                  className="ml-2 px-4 py-2 bg-orange-500 text-white rounded-md text-sm font-medium hover:bg-orange-600 transition-colors"
                >
                  🔄 Reset Route
                </button>
              )}
            </div>
            
            {/* Real-time Statistics */}
            <div className="h-auto">
              <RealTimeStats />
            </div>
          </div>

          {/* Right Column - Controls */}
          <div className="lg:col-span-1">
            <div className="sticky top-6 space-y-4">
              <ControlPanel />
              
              {/* Detection Status Card */}
              {detectedWaste && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h3 className="font-semibold text-green-800 mb-2">🎯 Waste Detected!</h3>
                  <div className="text-sm text-green-700 space-y-1">
                    <p><strong>Type:</strong> {detectedWaste.label}</p>
                    <p><strong>Category:</strong> {detectedWaste.type}</p>
                    <p><strong>Confidence:</strong> {(detectedWaste.confidence * 100).toFixed(1)}%</p>
                    <p className="text-xs text-green-600 mt-2">
                      📍 Auto-routing to nearest bin...
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
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
