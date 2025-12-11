import { useState, useEffect } from 'react';

const RealTimeStats = () => {
  const [stats, setStats] = useState({
    total: 0,
    organic: 0,
    recyclable: 0,
    hazardous: 0,
    other: 0,
    lastUpdated: null
  });
  const [isConnected, setIsConnected] = useState(false);
  const [detectionHistory, setDetectionHistory] = useState([]);

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const connectToStats = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/stats');
        
        ws.onopen = () => {
          console.log('📊 Connected to stats WebSocket');
          setIsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('📊 Received real-time stats:', data);
            
            // Backend sends: {totals: {organic, recyclable, hazardous, other}, recent: [...]}
            if (data.totals) {
              const totals = data.totals;
              const total = Object.values(totals).reduce((sum, val) => sum + val, 0);
              
              setStats({
                total: total,
                organic: totals.organic || 0,
                recyclable: totals.recyclable || 0,
                hazardous: totals.hazardous || 0,
                other: totals.other || 0,
                lastUpdated: new Date()
              });
              
              // Add to history
              setDetectionHistory(prev => [
                ...prev.slice(-9), // Keep last 9 entries
                {
                  timestamp: new Date(),
                  total: total,
                  breakdown: totals
                }
              ]);
            }
          } catch (error) {
            console.error('❌ Error parsing stats data:', error);
          }
        };

        ws.onclose = () => {
          console.log('📊 Stats WebSocket disconnected');
          setIsConnected(false);
          // Retry connection after 3 seconds
          setTimeout(connectToStats, 3000);
        };

        ws.onerror = (error) => {
          console.error('❌ Stats WebSocket error:', error);
          setIsConnected(false);
        };

        return ws;
      } catch (error) {
        console.error('❌ Failed to connect to stats WebSocket:', error);
        return null;
      }
    };

    const ws = connectToStats();
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const wasteCategories = [
    { key: 'organic', label: 'Organic', color: 'text-green-400', bgColor: 'bg-green-500/20', borderColor: 'border-green-500/30', icon: '🍂' },
    { key: 'recyclable', label: 'Recyclable', color: 'text-blue-400', bgColor: 'bg-blue-500/20', borderColor: 'border-blue-500/30', icon: '♻️' },
    { key: 'hazardous', label: 'Hazardous', color: 'text-red-400', bgColor: 'bg-red-500/20', borderColor: 'border-red-500/30', icon: '⚠️' },
    { key: 'other', label: 'Other', color: 'text-gray-400', bgColor: 'bg-gray-500/20', borderColor: 'border-gray-500/30', icon: '🗑️' }
  ];

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg space-y-6 h-full">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-700 pb-3">
        <h3 className="text-xl font-semibold text-white flex items-center">
          <span className="mr-2">📊</span> Real-time Statistics
        </h3>
        <div className={`flex items-center space-x-2 text-sm px-3 py-1 rounded-full ${
          isConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
        }`}>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
          <span>{isConnected ? 'Live' : 'Offline'}</span>
        </div>
      </div>

      {/* Total Count */}
      <div className="text-center py-4">
        <div className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          {stats.total}
        </div>
        <div className="text-sm text-gray-400 mt-1">
          Total Objects Detected
        </div>
        {stats.lastUpdated && (
          <div className="text-xs text-gray-500 mt-2">
            Updated: {stats.lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Category Breakdown */}
      <div className="grid grid-cols-2 gap-4">
        {wasteCategories.map(category => (
          <div 
            key={category.key} 
            className={`p-4 rounded-lg border ${category.bgColor} ${category.borderColor}`}
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">{category.icon}</span>
              <span className={`text-2xl font-bold ${category.color}`}>
                {stats[category.key] || 0}
              </span>
            </div>
            <div className={`text-sm font-medium ${category.color} mt-2`}>
              {category.label}
            </div>
            {/* Progress bar */}
            <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
              <div 
                className={`h-1 rounded-full transition-all ${
                  category.key === 'organic' ? 'bg-green-400' :
                  category.key === 'recyclable' ? 'bg-blue-400' :
                  category.key === 'hazardous' ? 'bg-red-400' : 'bg-gray-400'
                }`}
                style={{ width: `${stats.total > 0 ? (stats[category.key] / stats.total * 100) : 0}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Detection History */}
      {detectionHistory.length > 0 && (
        <div className="border-t border-gray-700 pt-4">
          <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center">
            <span className="mr-2">📈</span> Recent Activity
          </h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {detectionHistory.slice(-5).reverse().map((entry, index) => (
              <div 
                key={index} 
                className="flex items-center justify-between text-sm bg-gray-700/50 rounded px-3 py-2"
              >
                <span className="text-gray-400">{entry.timestamp.toLocaleTimeString()}</span>
                <span className="font-medium text-blue-400">
                  {entry.total} detected
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Status Information */}
      <div className="text-xs text-gray-500 space-y-1 p-3 bg-gray-900/50 rounded-lg">
        <div className="flex items-center"><span className="mr-2">🔄</span> Auto-updates from detection</div>
        <div className="flex items-center"><span className="mr-2">📡</span> WebSocket real-time data</div>
      </div>
    </div>
  );
};

export default RealTimeStats;
