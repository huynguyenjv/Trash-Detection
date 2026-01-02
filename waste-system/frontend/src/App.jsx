import { useState } from 'react';
import VideoStream from './components/VideoStream';
import RealTimeStats from './components/RealTimeStats';
import ControlPanel from './components/ControlPanel';
import MapView from './components/MapView';
import Dashboard from './components/Dashboard';

function App() {
  // State cho session summary khi tắt camera
  const [sessionSummary, setSessionSummary] = useState(null);
  const [showMap, setShowMap] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [findRouteRequest, setFindRouteRequest] = useState(null);
  
  // State cho confidence threshold - controlled from ControlPanel
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  
  // State cho Dashboard
  const [showDashboard, setShowDashboard] = useState(false);

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
    <div className="flex flex-col h-screen overflow-hidden text-white bg-gray-900">
      {/* Dashboard Modal */}
      {showDashboard && (
        <Dashboard onClose={() => setShowDashboard(false)} />
      )}
      
      {/* Compact Header */}
      <header className="flex-shrink-0 px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-green-400 to-blue-500">
              <span className="text-xl">🗑️</span>
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">Phát hiện rác thông minh</h1>
              <p className="text-xs text-gray-400">Phát hiện và định tuyến bằng AI</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setShowDashboard(true)}
              className="px-4 py-2 text-sm font-medium text-white bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2"
            >
              📊 Dashboard
            </button>
            <div className="text-right">
              <p className="text-sm font-medium text-white">YOLOv8 Detection</p>
              <p className="text-xs text-gray-400">Phân tích thời gian thực</p>
            </div>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
              isStreaming ? 'bg-green-900' : 'bg-gray-700'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isStreaming ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
              }`}></div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="px-4 py-6 mx-auto max-w-7xl">
          <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-3">
            {/* Left Column - Video and Stats */}
            <div className="space-y-6 lg:col-span-2">
              {/* Video Stream */}
              <div className="h-auto">
                <VideoStream 
                  onSessionEnd={handleSessionEnd}
                  onSessionStart={handleSessionStart}
                  confidenceThreshold={confidenceThreshold}
                />
              </div>
              
              {/* Session Summary Card - Hiển thị sau khi tắt camera */}
              {sessionSummary && !isStreaming && (
                <div className="p-6 bg-gray-800 rounded-lg shadow-md">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">
                      📊 Kết quả phát hiện
                    </h3>
                    <button
                      onClick={handleReset}
                      className="text-sm text-gray-400 hover:text-white"
                    >
                      Đặt lại
                    </button>
                  </div>
                  
                  {/* Thống kê */}
                  <div className="grid grid-cols-4 gap-4 mb-6">
                    <div className="p-3 text-center rounded-lg bg-green-900/50">
                      <div className="text-2xl font-bold text-green-400">
                        {sessionSummary.organic || 0}
                      </div>
                      <div className="text-xs text-green-400">🍂 Hữu cơ</div>
                    </div>
                    <div className="p-3 text-center rounded-lg bg-blue-900/50">
                      <div className="text-2xl font-bold text-blue-400">
                        {sessionSummary.recyclable || 0}
                      </div>
                      <div className="text-xs text-blue-400">♻️ Tái chế</div>
                    </div>
                    <div className="p-3 text-center rounded-lg bg-red-900/50">
                      <div className="text-2xl font-bold text-red-400">
                        {sessionSummary.hazardous || 0}
                      </div>
                      <div className="text-xs text-red-400">⚠️ Nguy hại</div>
                    </div>
                    <div className="p-3 text-center rounded-lg bg-gray-700/50">
                      <div className="text-2xl font-bold text-gray-300">
                        {sessionSummary.other || 0}
                      </div>
                      <div className="text-xs text-gray-400">🗑️ Khác</div>
                    </div>
                  </div>
                  
                  {/* Tổng số */}
                  <div className="p-4 mb-6 text-center rounded-lg bg-gradient-to-r from-blue-900/50 to-green-900/50">
                    <div className="text-3xl font-bold text-white">
                      {sessionSummary.total || 0}
                    </div>
                    <div className="text-sm text-gray-400">Tổng số rác phát hiện được</div>
                  </div>
                  
                  {/* Nút tìm đường */}
                  {sessionSummary.total > 0 && (
                    <div className="space-y-3">
                      <p className="mb-3 text-sm text-center text-gray-400">
                        Chọn loại rác để tìm thùng rác gần nhất:
                      </p>
                      <div className="grid grid-cols-2 gap-3">
                        {sessionSummary.organic > 0 && (
                          <button
                            onClick={() => handleFindRoute('organic')}
                            className="flex items-center justify-center px-4 py-3 space-x-2 text-white transition-colors bg-green-600 rounded-lg hover:bg-green-700"
                          >
                            <span>🍂</span>
                            <span>Thùng rác hữu cơ</span>
                          </button>
                        )}
                        {sessionSummary.recyclable > 0 && (
                          <button
                            onClick={() => handleFindRoute('recyclable')}
                            className="flex items-center justify-center px-4 py-3 space-x-2 text-white transition-colors bg-blue-600 rounded-lg hover:bg-blue-700"
                          >
                            <span>♻️</span>
                            <span>Thùng tái chế</span>
                          </button>
                        )}
                        {sessionSummary.hazardous > 0 && (
                          <button
                            onClick={() => handleFindRoute('hazardous')}
                            className="flex items-center justify-center px-4 py-3 space-x-2 text-white transition-colors bg-red-600 rounded-lg hover:bg-red-700"
                          >
                            <span>⚠️</span>
                            <span>Thùng rác nguy hại</span>
                          </button>
                        )}
                        {sessionSummary.other > 0 && (
                          <button
                            onClick={() => handleFindRoute('general')}
                            className="flex items-center justify-center px-4 py-3 space-x-2 text-white transition-colors bg-gray-600 rounded-lg hover:bg-gray-700"
                          >
                            <span>🗑️</span>
                            <span>Thùng rác chung</span>
                          </button>
                        )}
                      </div>
                      
                      {/* Nút tìm đường tự động */}
                      <button
                        onClick={() => handleFindRoute(getMainCategory())}
                        className="w-full px-4 py-3 mt-4 font-medium text-white transition-colors rounded-lg bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700"
                      >
                        🗺️ Tìm đường đến thùng rác gần nhất
                      </button>
                    </div>
                  )}
                </div>
              )}
              
              {/* Map View - Chỉ hiển thị khi có yêu cầu tìm đường */}
              {showMap && (
                <div style={{ height: '550px' }}>
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
                      ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {showMap ? '🗺️ Ẩn bản đồ' : '🗺️ Hiện bản đồ'}
                </button>
                {findRouteRequest && (
                  <button
                    onClick={() => setFindRouteRequest(null)}
                    className="px-4 py-2 text-sm font-medium text-white transition-colors bg-orange-500 rounded-md hover:bg-orange-600"
                  >
                    🔄 Xóa đường đi
                  </button>
                )}
              </div>
              
              {/* Thống kê theo dõi - Dữ liệu từ database */}
              <div className="h-auto">
                <RealTimeStats />
              </div>
            </div>

            {/* Right Column - Controls */}
            <div className="lg:col-span-1">
              <div className="sticky space-y-4 top-6">
                <ControlPanel 
                  confidenceThreshold={confidenceThreshold}
                  onConfidenceChange={setConfidenceThreshold}
                />
                
                {/* Hướng dẫn sử dụng */}
                <div className="p-4 border border-blue-700 rounded-lg bg-blue-900/30">
                  <h3 className="mb-2 font-semibold text-blue-300">📖 Hướng dẫn</h3>
                  <ol className="space-y-2 text-sm text-blue-200 list-decimal list-inside">
                    <li>Nhấn <strong>"Bắt đầu phát hiện"</strong> để bắt đầu</li>
                    <li>Đưa rác vào camera để phát hiện</li>
                    <li>Nhấn <strong>"Dừng"</strong> khi xong</li>
                    <li>Xem thống kê và nhấn <strong>"Tìm đường"</strong></li>
                  </ol>
                </div>
                
                {/* Dashboard Quick Access */}
                <button
                  onClick={() => setShowDashboard(true)}
                  className="w-full p-4 border border-purple-700 rounded-lg bg-purple-900/30 hover:bg-purple-800/50 transition-colors text-left"
                >
                  <h3 className="mb-2 font-semibold text-purple-300 flex items-center gap-2">
                    📊 Xem Dashboard Chi Tiết
                  </h3>
                  <p className="text-sm text-purple-200">
                    Thống kê đầy đủ, biểu đồ theo ngày, địa điểm thu gom
                  </p>
                </button>
                
                {/* Status Card */}
                {isStreaming && (
                  <div className="p-4 border border-green-700 rounded-lg bg-green-900/30">
                    <h3 className="mb-2 font-semibold text-green-300">🎥 Đang phát hiện...</h3>
                    <p className="text-sm text-green-200">
                      Camera đang hoạt động. Đưa rác vào khung hình để phát hiện.
                    </p>
                  </div>
                )}
                
                {findRouteRequest && (
                  <div className="p-4 border border-purple-700 rounded-lg bg-purple-900/30">
                    <h3 className="mb-2 font-semibold text-purple-300">🗺️ Đang tìm đường...</h3>
                    <p className="text-sm text-purple-200">
                      Tìm thùng rác <strong>{findRouteRequest.category}</strong> gần nhất
                    </p>
                  </div>
                )}

                {/* System Info */}
                <div className="p-4 text-xs text-gray-400 rounded-lg bg-gray-700/50">
                  <h4 className="mb-2 font-medium text-gray-300">System Info</h4>
                  <div className="space-y-1">
                    <p>• Model: YOLOv8 Custom</p>
                    <p>• Routing: Goong Maps API</p>
                    <p>• Threshold: 70% confidence</p>
                    <p>• Algorithm: 5 custom options</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

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
