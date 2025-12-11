import { useState } from 'react';

const ControlPanel = () => {
  const [confidence, setConfidence] = useState(0.7);

  const handleConfidenceChange = (newConfidence) => {
    setConfidence(newConfidence);
    console.log(`🎯 Confidence threshold changed to: ${newConfidence}`);
  };

  const confidencePresets = [
    { label: 'Low', value: 0.3, color: 'from-orange-500 to-orange-600' },
    { label: 'Medium', value: 0.5, color: 'from-blue-500 to-blue-600' },
    { label: 'High', value: 0.7, color: 'from-green-500 to-green-600' },
    { label: 'Very High', value: 0.9, color: 'from-purple-500 to-purple-600' }
  ];

  return (
    <div className="bg-gray-700/50 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="border-b border-gray-600 pb-2">
        <h3 className="text-sm font-semibold text-white flex items-center">
          <span className="mr-2">🎛️</span> Detection Settings
        </h3>
      </div>

      {/* Confidence Settings */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-gray-300">
            Confidence Threshold
          </label>
          <span className="px-2 py-0.5 bg-gray-600 rounded text-xs font-mono text-white">
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>

        {/* Slider */}
        <div className="space-y-3">
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={confidence}
            onChange={(e) => handleConfidenceChange(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          
          {/* Preset buttons */}
          <div className="grid grid-cols-4 gap-1">
            {confidencePresets.map((preset) => (
              <button
                key={preset.value}
                onClick={() => handleConfidenceChange(preset.value)}
                className={`px-2 py-1.5 text-xs rounded transition-all duration-200 ${
                  confidence === preset.value 
                    ? `bg-gradient-to-r ${preset.color} text-white shadow-md` 
                    : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Confidence explanation */}
        <div className="p-2 bg-gray-800/50 rounded text-xs">
          {confidence < 0.4 ? (
            <div className="text-orange-400">
              <span className="font-semibold">⚠️ Low:</span> More detections, may have false positives
            </div>
          ) : confidence < 0.7 ? (
            <div className="text-blue-400">
              <span className="font-semibold">✅ Balanced:</span> Good accuracy, recommended
            </div>
          ) : (
            <div className="text-green-400">
              <span className="font-semibold">🎯 High:</span> Only confident detections
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
