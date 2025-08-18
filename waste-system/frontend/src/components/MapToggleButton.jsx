import { useState } from 'react';
import MapView from './MapView';

const MapModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto" style={{ scrollBehavior: 'smooth' }}>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      ></div>
      
      {/* Modal */}
      <div className="relative min-h-screen flex items-start sm:items-center justify-center p-2 sm:p-4">
        <div className="relative bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[95vh] sm:max-h-[90vh] flex flex-col overflow-hidden mt-2 sm:mt-0">
          {/* Header */}
          <div className="flex items-center justify-between p-3 sm:p-4 border-b flex-shrink-0">
            <h2 className="text-lg sm:text-xl font-semibold text-gray-900 flex items-center">
              🗺️ <span className="ml-2">Waste Detection & A* Pathfinding Map</span>
            </h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <svg className="w-5 h-5 sm:w-6 sm:h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          {/* Map Content - Scrollable */}
          <div className="flex-1 overflow-y-auto overflow-x-hidden scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
            <div className="h-full min-h-[60vh]">
              <MapView />
            </div>
          </div>
          
          {/* Footer */}
          <div className="p-3 sm:p-4 border-t bg-gray-50 flex-shrink-0">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs sm:text-sm text-gray-600 space-y-2 sm:space-y-0">
              <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-1 sm:space-y-0 sm:space-x-4">
                <span>🎯 Click waste location to find route</span>
                <span>🗑️ Different colors = different waste types</span>
              </div>
              <div className="flex items-center space-x-2 text-xs">
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>Organic</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span>Recyclable</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>Hazardous</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const MapToggleButton = () => {
  const [isMapOpen, setIsMapOpen] = useState(false);

  return (
    <>
      <div className="space-y-3">
        {/* Map Toggle Button */}
        <button
          onClick={() => setIsMapOpen(true)}
          className="w-full py-3 px-4 bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-lg font-medium hover:from-green-600 hover:to-blue-600 transition-all duration-200 shadow-md hover:shadow-lg flex items-center justify-center space-x-2"
        >
          <span>🗺️</span>
          <span>Hiển thị bản đồ A*</span>
        </button>

        {/* Quick Actions */}
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setIsMapOpen(true)}
            className="py-2 px-3 bg-blue-100 text-blue-700 text-sm rounded hover:bg-blue-200 transition-colors"
          >
            📍 Tìm thùng rác
          </button>
          <button
            onClick={() => {
              setIsMapOpen(true);
              // Auto-trigger pathfinding when map opens
              setTimeout(() => {
                const findRouteButtons = document.querySelectorAll('button:contains("Find Route")');
                if (findRouteButtons.length > 0) {
                  findRouteButtons[0].click();
                }
              }, 1000);
            }}
            className="py-2 px-3 bg-purple-100 text-purple-700 text-sm rounded hover:bg-purple-200 transition-colors"
          >
            🎯 Tìm đường ngắn nhất
          </button>
        </div>

        {/* Info */}
        <div className="text-xs text-gray-600 space-y-1 p-3 bg-gray-50 rounded">
          <div className="flex items-center space-x-2">
            <span>🤖</span>
            <span><strong>A* Algorithm:</strong> Tìm đường đi ngắn nhất</span>
          </div>
          <div className="flex items-center space-x-2">
            <span>🎯</span>
            <span><strong>Smart Routing:</strong> Chọn thùng phù hợp với loại rác</span>
          </div>
          <div className="flex items-center space-x-2">
            <span>📊</span>
            <span><strong>Real-time:</strong> Cập nhật vị trí liên tục</span>
          </div>
        </div>
      </div>

      {/* Map Modal */}
      <MapModal isOpen={isMapOpen} onClose={() => setIsMapOpen(false)} />
    </>
  );
};

export default MapToggleButton;
