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
    <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <span>⚙️</span> Detection Settings
        </h3>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
        >
          {showAdvanced ? 'Hide' : 'Show'} Advanced
        </button>
      </div>

      {/* Confidence Threshold */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          🎯 Confidence: {confidence.toFixed(2)}
        </label>
        <div className="flex items-center space-x-2">
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
          <span className="text-sm text-cyan-400 font-medium min-w-[40px]">
            {Math.round(confidence * 100)}%
          </span>
        </div>
        
        {/* Quick preset buttons */}
        <div className="flex flex-wrap gap-2 mt-2">
          {presetConfidences.map((preset) => (
            <button
              key={preset.value}
              onClick={() => setConfidence(preset.value)}
              className={`px-2 py-1 text-xs rounded-lg transition-all ${
                confidence === preset.value
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Explanation */}
        <div className="mt-2 text-xs">
          {confidence < 0.4 && (
            <span className="text-yellow-400">
              ⚠️ Low: More objects, possible false positives
            </span>
          )}
          {confidence >= 0.4 && confidence < 0.7 && (
            <span className="text-cyan-400">
              ✅ Balanced: Good for most cases
            </span>
          )}
          {confidence >= 0.7 && (
            <span className="text-green-400">
              🎯 High: Only confident detections
            </span>
          )}
        </div>
      </div>

      {/* Auto Detection Toggle */}
      <div className="mb-4">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={autoDetect}
            onChange={(e) => setAutoDetect(e.target.checked)}
            className="rounded bg-gray-700 border-gray-600 text-cyan-500 focus:ring-cyan-500"
          />
          <span className="text-sm font-medium text-gray-300">
            🔄 Continuous detection
          </span>
        </label>
      </div>

      {/* Advanced Settings */}
      {showAdvanced && (
        <div className="border-t border-gray-700 pt-3">
          <h4 className="text-xs font-semibold text-gray-400 mb-2 uppercase">Advanced</h4>
          
          {/* Minimum Object Size */}
          <div className="mb-3">
            <label className="block text-sm font-medium text-gray-300 mb-1">
              📏 Min object size: {minObjectSize}px
            </label>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={minObjectSize}
              onChange={(e) => setMinObjectSize(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <div className="text-xs text-gray-500 mt-1">
              Objects smaller than {minObjectSize}px will be ignored
            </div>
          </div>
        </div>
      )}

      {/* Real-time stats */}
      <div className="mt-4 p-3 bg-gray-900/50 rounded-lg">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-xs text-gray-400">Confidence</div>
            <div className="text-cyan-400 font-bold">{Math.round(confidence * 100)}%</div>
          </div>
          <div>
            <div className="text-xs text-gray-400">Min Size</div>
            <div className="text-green-400 font-bold">{minObjectSize}px</div>
          </div>
          <div>
            <div className="text-xs text-gray-400">Auto</div>
            <div className={`font-bold ${autoDetect ? 'text-green-400' : 'text-gray-500'}`}>
              {autoDetect ? 'ON' : 'OFF'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectionSettings;
