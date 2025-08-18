import { useState } from 'react';
import MapToggleButton from './MapToggleButton';

const ControlPanel = () => {
  const [confidence, setConfidence] = useState(0.5);

  const handleConfidenceChange = (newConfidence) => {
    setConfidence(newConfidence);
    console.log(`🎯 Confidence threshold changed to: ${newConfidence}`);
    
    // Send to backend via WebSocket or API
    // TODO: Implement actual backend communication
  };

  const confidencePresets = [
    { label: 'Thấp (0.3)', value: 0.3, color: 'bg-orange-500' },
    { label: 'Trung bình (0.5)', value: 0.5, color: 'bg-blue-500' },
    { label: 'Cao (0.7)', value: 0.7, color: 'bg-green-500' },
    { label: 'Rất cao (0.9)', value: 0.9, color: 'bg-purple-500' }
  ];

  return (
    <div className="bg-white p-4 rounded-lg shadow-md space-y-4">
      {/* Header */}
      <div className="border-b pb-2">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center">
          🎛️ Bảng điều khiển
        </h3>
        <p className="text-sm text-gray-600">
          Điều chỉnh confidence và tìm đường đi tối ưu
        </p>
      </div>

      {/* Confidence Settings */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-700">
            🎯 Confidence Threshold
          </label>
          <span className="px-2 py-1 bg-gray-100 rounded text-sm font-mono">
            {confidence.toFixed(2)}
          </span>
        </div>

        {/* Slider */}
        <div className="space-y-2">
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={confidence}
            onChange={(e) => handleConfidenceChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
          />
          
          {/* Preset buttons */}
          <div className="grid grid-cols-2 gap-2">
            {confidencePresets.map((preset) => (
              <button
                key={preset.value}
                onClick={() => handleConfidenceChange(preset.value)}
                className={`px-3 py-2 text-xs text-white rounded transition-all duration-200 ${
                  confidence === preset.value 
                    ? `${preset.color} shadow-md scale-105` 
                    : 'bg-gray-400 hover:bg-gray-500'
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Confidence explanation */}
        <div className="p-3 bg-gray-50 rounded text-xs">
          {confidence < 0.4 ? (
            <div className="text-orange-700">
              <span className="font-semibold">⚠️ Confidence thấp:</span>
              <br />• Phát hiện nhiều object hơn
              <br />• Có thể có false positive
              <br />• Phù hợp khi cần detect tất cả
            </div>
          ) : confidence < 0.7 ? (
            <div className="text-blue-700">
              <span className="font-semibold">✅ Confidence cân bằng:</span>
              <br />• Tỷ lệ chính xác tốt
              <br />• Phù hợp cho hầu hết trường hợp
              <br />• Khuyên dùng mặc định
            </div>
          ) : (
            <div className="text-green-700">
              <span className="font-semibold">🎯 Confidence cao:</span>
              <br />• Chỉ phát hiện object chắc chắn
              <br />• Ít false positive
              <br />• Có thể bỏ lỡ object mờ/xa
            </div>
          )}
        </div>
      </div>

      {/* Pathfinding */}
      <div className="space-y-3 border-t pt-3">
        <h4 className="text-sm font-medium text-gray-700 flex items-center">
          🗺️ Tìm đường đi A* Algorithm
        </h4>
        
        {/* Map Toggle Component */}
        <MapToggleButton />
      </div>

      {/* Statistics */}
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-3 rounded-lg text-sm">
        <div className="text-gray-700 font-medium mb-2">📊 Thống kê hiện tại</div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between">
            <span>Confidence:</span>
            <span className="font-mono text-blue-600">{Math.round(confidence * 100)}%</span>
          </div>
          <div className="flex justify-between">
            <span>Trạng thái:</span>
            <span className="text-green-600">🟢 Sẵn sàng</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
