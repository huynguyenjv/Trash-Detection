import { useState } from 'react';
import VideoStream from './components/VideoStream';
import RealTimeStats from './components/RealTimeStats';
import ControlPanel from './components/ControlPanel';
import MapView from './components/MapView';

function App() {
  // State for auto-routing when waste is detected
  const [detectedWaste, setDetectedWaste] = useState(null);
  const [activeTab, setActiveTab] = useState('detection'); // 'detection' | 'map' | 'stats'
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Callback when VideoStream detects waste above threshold
  const handleWasteDetected = (wasteInfo) => {
    console.log('🎯 App received waste detection:', wasteInfo);
    setDetectedWaste(wasteInfo);
    // Auto-switch to map tab when waste detected
    setActiveTab('map');
  };

  const handleResetRoute = () => {
    setDetectedWaste(null);
    setActiveTab('detection');
  };

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white overflow-hidden">
      {/* Compact Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-green-400 to-blue-500 rounded-lg flex items-center justify-center">
              <span className="text-xl">🗑️</span>
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">
                Smart Waste Detection
              </h1>
              <p className="text-xs text-gray-400">
                YOLOv8 + Goong Maps Routing
              </p>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="flex items-center space-x-4">
            {/* Detection Alert */}
            {detectedWaste && (
              <div className="flex items-center space-x-2 px-3 py-1.5 bg-green-600/20 border border-green-500/50 rounded-full animate-pulse">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-green-400 font-medium">
                  {detectedWaste.label} detected ({(detectedWaste.confidence * 100).toFixed(0)}%)
                </span>
                <button 
                  onClick={handleResetRoute}
                  className="text-green-400 hover:text-white ml-1"
                >
                  ✕
                </button>
              </div>
            )}

            {/* Connection Status */}
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-gray-400">Live</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Video/Map View */}
        <div className="flex-1 flex flex-col">
          {/* Tab Navigation */}
          <div className="bg-gray-800 px-4 py-2 flex items-center space-x-1 border-b border-gray-700">
            <button
              onClick={() => setActiveTab('detection')}
              className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors ${
                activeTab === 'detection'
                  ? 'bg-gray-900 text-white border-t-2 border-blue-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              📹 Detection
            </button>
            <button
              onClick={() => setActiveTab('map')}
              className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors flex items-center space-x-2 ${
                activeTab === 'map'
                  ? 'bg-gray-900 text-white border-t-2 border-green-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <span>🗺️ Map & Routing</span>
              {detectedWaste && (
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors ${
                activeTab === 'stats'
                  ? 'bg-gray-900 text-white border-t-2 border-purple-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              📊 Statistics
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 p-4 overflow-auto">
            {activeTab === 'detection' && (
              <div className="h-full">
                <VideoStream 
                  onWasteDetected={handleWasteDetected}
                  routeThreshold={0.7}
                />
              </div>
            )}

            {activeTab === 'map' && (
              <div className="h-full bg-gray-800 rounded-lg overflow-hidden">
                <MapView 
                  autoFindRoute={detectedWaste !== null}
                  detectedWaste={detectedWaste}
                />
              </div>
            )}

            {activeTab === 'stats' && (
              <div className="h-full">
                <RealTimeStats />
              </div>
            )}
          </div>
        </div>

        {/* Right Sidebar - Controls */}
        <div className={`bg-gray-800 border-l border-gray-700 flex-shrink-0 transition-all duration-300 ${
          sidebarCollapsed ? 'w-12' : 'w-80'
        }`}>
          {/* Sidebar Toggle */}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full py-2 px-3 text-gray-400 hover:text-white hover:bg-gray-700 flex items-center justify-center border-b border-gray-700"
          >
            {sidebarCollapsed ? '◀' : '▶'}
          </button>

          {!sidebarCollapsed && (
            <div className="p-4 space-y-4 overflow-auto h-full">
              {/* Detection Status */}
              {detectedWaste && (
                <div className="bg-gradient-to-r from-green-600/20 to-blue-600/20 border border-green-500/30 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-green-400 flex items-center">
                      <span className="mr-2">🎯</span> Waste Detected
                    </h3>
                    <button 
                      onClick={handleResetRoute}
                      className="text-gray-400 hover:text-white text-sm"
                    >
                      Reset
                    </button>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Type:</span>
                      <span className="text-white font-medium">{detectedWaste.label}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Category:</span>
                      <span className={`font-medium ${
                        detectedWaste.type === 'organic' ? 'text-green-400' :
                        detectedWaste.type === 'recyclable' ? 'text-blue-400' :
                        detectedWaste.type === 'hazardous' ? 'text-red-400' :
                        'text-gray-400'
                      }`}>
                        {detectedWaste.type}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Confidence:</span>
                      <span className="text-white font-medium">
                        {(detectedWaste.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    {/* Progress bar for confidence */}
                    <div className="mt-2">
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full transition-all"
                          style={{ width: `${detectedWaste.confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => setActiveTab('map')}
                    className="w-full mt-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    📍 View Route to Bin
                  </button>
                </div>
              )}

              {/* Quick Actions */}
              <div className="bg-gray-700/50 rounded-lg p-4">
                <h3 className="font-semibold text-white mb-3 flex items-center">
                  <span className="mr-2">⚡</span> Quick Actions
                </h3>
                <div className="space-y-2">
                  <button 
                    onClick={() => setActiveTab('detection')}
                    className="w-full py-2 px-3 bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 rounded-lg text-sm font-medium transition-colors text-left flex items-center"
                  >
                    <span className="mr-2">📹</span> Start Detection
                  </button>
                  <button 
                    onClick={() => setActiveTab('map')}
                    className="w-full py-2 px-3 bg-green-600/20 hover:bg-green-600/40 text-green-400 rounded-lg text-sm font-medium transition-colors text-left flex items-center"
                  >
                    <span className="mr-2">🗺️</span> View Map
                  </button>
                  <button 
                    onClick={() => setActiveTab('stats')}
                    className="w-full py-2 px-3 bg-purple-600/20 hover:bg-purple-600/40 text-purple-400 rounded-lg text-sm font-medium transition-colors text-left flex items-center"
                  >
                    <span className="mr-2">📊</span> View Statistics
                  </button>
                </div>
              </div>

              {/* Control Panel */}
              <ControlPanel />

              {/* System Info */}
              <div className="bg-gray-700/50 rounded-lg p-4 text-xs text-gray-400">
                <h4 className="font-medium text-gray-300 mb-2">System Info</h4>
                <div className="space-y-1">
                  <p>• Model: YOLOv8 Custom</p>
                  <p>• Routing: Goong Maps API</p>
                  <p>• Threshold: 70% confidence</p>
                  <p>• Algorithm: 5 custom options</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Status Bar */}
      <footer className="bg-gray-800 border-t border-gray-700 px-4 py-1.5 flex-shrink-0">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center space-x-4">
            <span>© 2024 Smart Waste Detection System</span>
            <span>•</span>
            <span>YOLOv8 + FastAPI + React</span>
          </div>
          <div className="flex items-center space-x-4">
            <span className="flex items-center">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full mr-1.5"></span>
              Backend Connected
            </span>
            <span className="flex items-center">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mr-1.5"></span>
              Goong API Ready
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
