import { useState, useEffect } from 'react';

const DetectionSettings = ({ onSettingsChange }) => {
  const [confidence, setConfidence] = useState(0.5);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [autoDetect, setAutoDetect] = useState(true);
  const [minObjectSize, setMinObjectSize] = useState(10);

  useEffect(() => {
    // Gửi cài đặt về parent component
    onSettingsChange({
      confidence,
      autoDetect,
      minObjectSize
    });
  }, [confidence, autoDetect, minObjectSize, onSettingsChange]);

  const presetConfidences = [
    { label: 'Thấp (Nhiều object)', value: 0.3 },
    { label: 'Trung bình', value: 0.5 },
    { label: 'Cao (Chính xác)', value: 0.7 },
    { label: 'Rất cao', value: 0.9 }
  ];

  return (
    <div className="bg-white p-4 rounded-lg shadow-md mb-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-800">⚙️ Cài đặt Detection</h3>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-blue-600 hover:text-blue-800"
        >
          {showAdvanced ? 'Ẩn' : 'Hiện'} cài đặt nâng cao
        </button>
      </div>

      {/* Confidence Threshold */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          🎯 Confidence Threshold: {confidence.toFixed(2)}
        </label>
        <div className="flex items-center space-x-2">
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <span className="text-sm text-gray-600 min-w-[40px]">
            {Math.round(confidence * 100)}%
          </span>
        </div>
        
        {/* Quick preset buttons */}
        <div className="flex flex-wrap gap-2 mt-2">
          {presetConfidences.map((preset) => (
            <button
              key={preset.value}
              onClick={() => setConfidence(preset.value)}
              className={`px-2 py-1 text-xs rounded ${
                confidence === preset.value
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Explanation */}
        <div className="mt-2 text-xs text-gray-600">
          {confidence < 0.4 && (
            <span className="text-orange-600">
              ⚠️ Confidence thấp: Sẽ detect nhiều object nhưng có thể có false positive
            </span>
          )}
          {confidence >= 0.4 && confidence < 0.7 && (
            <span className="text-blue-600">
              ✅ Confidence cân bằng: Tốt cho hầu hết trường hợp
            </span>
          )}
          {confidence >= 0.7 && (
            <span className="text-green-600">
              🎯 Confidence cao: Chỉ detect object rất chắc chắn
            </span>
          )}
        </div>
      </div>

      {/* Auto Detection Toggle */}
      <div className="mb-4">
        <label className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={autoDetect}
            onChange={(e) => setAutoDetect(e.target.checked)}
            className="rounded"
          />
          <span className="text-sm font-medium text-gray-700">
            🔄 Tự động detect liên tục
          </span>
        </label>
      </div>

      {/* Advanced Settings */}
      {showAdvanced && (
        <div className="border-t pt-3">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Cài đặt nâng cao</h4>
          
          {/* Minimum Object Size */}
          <div className="mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              📏 Kích thước object tối thiểu: {minObjectSize}px
            </label>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={minObjectSize}
              onChange={(e) => setMinObjectSize(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="text-xs text-gray-600 mt-1">
              Object nhỏ hơn {minObjectSize}px sẽ bị bỏ qua
            </div>
          </div>
        </div>
      )}

      {/* Real-time stats */}
      <div className="mt-4 p-3 bg-gray-50 rounded text-sm">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="font-medium text-gray-700">Confidence</div>
            <div className="text-blue-600 font-bold">{Math.round(confidence * 100)}%</div>
          </div>
          <div>
            <div className="font-medium text-gray-700">Min Size</div>
            <div className="text-green-600 font-bold">{minObjectSize}px</div>
          </div>
          <div>
            <div className="font-medium text-gray-700">Auto</div>
            <div className={`font-bold ${autoDetect ? 'text-green-600' : 'text-gray-400'}`}>
              {autoDetect ? 'ON' : 'OFF'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectionSettings;
