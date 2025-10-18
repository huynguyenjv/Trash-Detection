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
    { key: 'organic', label: 'Organic', color: 'text-green-600', bgColor: 'bg-green-100', icon: '🍂' },
    { key: 'recyclable', label: 'Recyclable', color: 'text-blue-600', bgColor: 'bg-blue-100', icon: '♻️' },
    { key: 'hazardous', label: 'Hazardous', color: 'text-red-600', bgColor: 'bg-red-100', icon: '⚠️' },
    { key: 'other', label: 'Other', color: 'text-gray-600', bgColor: 'bg-gray-100', icon: '🗑️' }
  ];

  return (
    <div className="bg-white p-4 rounded-lg shadow-md space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b pb-2">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center">
          📊 Real-time Statistics
        </h3>
        <div className={`flex items-center space-x-2 text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span>{isConnected ? 'Live' : 'Disconnected'}</span>
        </div>
      </div>

      {/* Total Count */}
      <div className="text-center">
        <div className="text-3xl font-bold text-blue-600">
          {stats.total}
        </div>
        <div className="text-sm text-gray-600">
          Total Objects Detected
        </div>
        {stats.lastUpdated && (
          <div className="text-xs text-gray-500 mt-1">
            Last updated: {stats.lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Category Breakdown */}
      <div className="grid grid-cols-2 gap-3">
        {wasteCategories.map(category => (
          <div key={category.key} className={`p-3 rounded-lg ${category.bgColor}`}>
            <div className="flex items-center justify-between">
              <span className="text-lg">{category.icon}</span>
              <span className={`text-xl font-bold ${category.color}`}>
                {stats[category.key] || 0}
              </span>
            </div>
            <div className={`text-sm font-medium ${category.color} mt-1`}>
              {category.label}
            </div>
          </div>
        ))}
      </div>

      {/* Detection History */}
      {detectionHistory.length > 0 && (
        <div className="border-t pt-3">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            📈 Recent Activity
          </h4>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {detectionHistory.slice(-5).reverse().map((entry, index) => (
              <div key={index} className="flex items-center justify-between text-xs text-gray-600 py-1">
                <span>{entry.timestamp.toLocaleTimeString()}</span>
                <span className="font-medium text-blue-600">
                  {entry.total} objects
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Status Information */}
      <div className="text-xs text-gray-500 space-y-1 p-2 bg-gray-50 rounded">
        <div>🔄 Auto-updates from detection system</div>
        <div>📡 WebSocket connection for real-time data</div>
        <div>📊 Statistics calculated per detection session</div>
      </div>
    </div>
  );
};

export default RealTimeStats;
