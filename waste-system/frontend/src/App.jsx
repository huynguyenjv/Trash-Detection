import { useState } from 'react';
import VideoStream from './components/VideoStream';
import RealTimeStats from './components/RealTimeStats';
import ControlPanel from './components/ControlPanel';
import MapView from './components/MapView';

function App() {
  // State cho session summary khi tắt camera
  const [sessionSummary, setSessionSummary] = useState(null);
  const [showMap, setShowMap] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [findRouteRequest, setFindRouteRequest] = useState(null);

  // Callback khi tắt camera - nhận session summary
  const handleSessionEnd = (summary) => {
    console.log('📊 Session ended with summary:', summary);
    setSessionSummary(summary);
    setIsStreaming(false);
  };

  // Callback khi bắt đầu camera
  const handleSessionStart = () => {
    setSessionSummary(null);
    setIsStreaming(true);
    setFindRouteRequest(null);
  };

  // Xử lý khi người dùng nhấn "Tìm đường đi"
  const handleFindRoute = (category) => {
    console.log('🗺️ Finding route for category:', category);
    setFindRouteRequest({
      category: category,
      timestamp: Date.now()
    });
    setShowMap(true);
  };

  // Reset tất cả
  const handleReset = () => {
    setSessionSummary(null);
    setFindRouteRequest(null);
    setShowMap(false);
  };

  // Xác định category chính dựa trên session summary
  const getMainCategory = () => {
    if (!sessionSummary) return null;
    
    const categories = [
      { name: 'organic', count: sessionSummary.organic || 0 },
      { name: 'recyclable', count: sessionSummary.recyclable || 0 },
      { name: 'hazardous', count: sessionSummary.hazardous || 0 },
      { name: 'other', count: sessionSummary.other || 0 }
    ];
    
    const sorted = categories.sort((a, b) => b.count - a.count);
    return sorted[0].count > 0 ? sorted[0].name : null;
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
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                isStreaming ? 'bg-green-100' : 'bg-gray-100'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  isStreaming ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                }`}></div>
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
              <VideoStream 
                onSessionEnd={handleSessionEnd}
                onSessionStart={handleSessionStart}
              />
            </div>
            
            {/* Session Summary Card - Hiển thị sau khi tắt camera */}
            {sessionSummary && !isStreaming && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    📊 Kết quả phát hiện
                  </h3>
                  <button
                    onClick={handleReset}
                    className="text-sm text-gray-500 hover:text-gray-700"
                  >
                    Đặt lại
                  </button>
                </div>
                
                {/* Thống kê */}
                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {sessionSummary.organic || 0}
                    </div>
                    <div className="text-xs text-green-600">🍂 Hữu cơ</div>
                  </div>
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {sessionSummary.recyclable || 0}
                    </div>
                    <div className="text-xs text-blue-600">♻️ Tái chế</div>
                  </div>
                  <div className="text-center p-3 bg-red-50 rounded-lg">
                    <div className="text-2xl font-bold text-red-600">
                      {sessionSummary.hazardous || 0}
                    </div>
                    <div className="text-xs text-red-600">⚠️ Nguy hại</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-600">
                      {sessionSummary.other || 0}
                    </div>
                    <div className="text-xs text-gray-600">🗑️ Khác</div>
                  </div>
                </div>
                
                {/* Tổng số */}
                <div className="text-center mb-6 p-4 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg">
                  <div className="text-3xl font-bold text-gray-800">
                    {sessionSummary.total || 0}
                  </div>
                  <div className="text-sm text-gray-600">Tổng số rác phát hiện được</div>
                </div>
                
                {/* Nút tìm đường */}
                {sessionSummary.total > 0 && (
                  <div className="space-y-3">
                    <p className="text-sm text-gray-600 text-center mb-3">
                      Chọn loại rác để tìm thùng rác gần nhất:
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                      {sessionSummary.organic > 0 && (
                        <button
                          onClick={() => handleFindRoute('organic')}
                          className="flex items-center justify-center space-x-2 px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                        >
                          <span>🍂</span>
                          <span>Thùng rác hữu cơ</span>
                        </button>
                      )}
                      {sessionSummary.recyclable > 0 && (
                        <button
                          onClick={() => handleFindRoute('recyclable')}
                          className="flex items-center justify-center space-x-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                        >
                          <span>♻️</span>
                          <span>Thùng tái chế</span>
                        </button>
                      )}
                      {sessionSummary.hazardous > 0 && (
                        <button
                          onClick={() => handleFindRoute('hazardous')}
                          className="flex items-center justify-center space-x-2 px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                        >
                          <span>⚠️</span>
                          <span>Thùng rác nguy hại</span>
                        </button>
                      )}
                      {sessionSummary.other > 0 && (
                        <button
                          onClick={() => handleFindRoute('general')}
                          className="flex items-center justify-center space-x-2 px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                        >
                          <span>🗑️</span>
                          <span>Thùng rác chung</span>
                        </button>
                      )}
                    </div>
                    
                    {/* Nút tìm đường tự động */}
                    <button
                      onClick={() => handleFindRoute(getMainCategory())}
                      className="w-full mt-4 px-4 py-3 bg-gradient-to-r from-green-600 to-blue-600 text-white rounded-lg hover:from-green-700 hover:to-blue-700 transition-colors font-medium"
                    >
                      🗺️ Tìm đường đến thùng rác gần nhất
                    </button>
                  </div>
                )}
              </div>
            )}
            
            {/* Map View - Chỉ hiển thị khi có yêu cầu tìm đường */}
            {showMap && (
              <div className="h-96">
                <MapView 
                  findRouteRequest={findRouteRequest}
                  onRouteFound={(route) => console.log('Route found:', route)}
                />
              </div>
            )}

            {/* Toggle Map Button */}
            <div className="flex justify-center space-x-2">
              <button
                onClick={() => setShowMap(!showMap)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  showMap 
                    ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {showMap ? '🗺️ Ẩn bản đồ' : '🗺️ Hiện bản đồ'}
              </button>
              {findRouteRequest && (
                <button
                  onClick={() => setFindRouteRequest(null)}
                  className="px-4 py-2 bg-orange-500 text-white rounded-md text-sm font-medium hover:bg-orange-600 transition-colors"
                >
                  🔄 Xóa đường đi
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
              
              {/* Hướng dẫn sử dụng */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="font-semibold text-blue-800 mb-2">📖 Hướng dẫn</h3>
                <ol className="text-sm text-blue-700 space-y-2 list-decimal list-inside">
                  <li>Nhấn <strong>"Start Camera"</strong> để bắt đầu</li>
                  <li>Đưa rác vào camera để phát hiện</li>
                  <li>Nhấn <strong>"Stop Camera"</strong> khi xong</li>
                  <li>Xem thống kê và nhấn <strong>"Tìm đường"</strong></li>
                </ol>
              </div>
              
              {/* Status Card */}
              {isStreaming && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h3 className="font-semibold text-green-800 mb-2">🎥 Đang phát hiện...</h3>
                  <p className="text-sm text-green-700">
                    Camera đang hoạt động. Đưa rác vào khung hình để phát hiện.
                  </p>
                </div>
              )}
              
              {findRouteRequest && (
                <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                  <h3 className="font-semibold text-purple-800 mb-2">🗺️ Đang tìm đường...</h3>
                  <p className="text-sm text-purple-700">
                    Tìm thùng rác <strong>{findRouteRequest.category}</strong> gần nhất
                  </p>
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
