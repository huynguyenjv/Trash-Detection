import VideoStream from './components/VideoStream';
import WasteStats from './components/WasteStats';
import MapView from './components/MapView';

function App() {
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
            {/* Video Stream */}
            <div className="h-auto">
              <VideoStream />
            </div>
            
            {/* Statistics */}
            <div className="h-auto">
              <WasteStats />
            </div>
          </div>

          {/* Right Column - Map */}
          <div className="lg:col-span-1">
            <div className="sticky top-6">
              <MapView />
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
