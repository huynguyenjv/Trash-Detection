import { useEffect, useState } from 'react';

const WasteStats = () => {
  const [stats, setStats] = useState({
    total: 0,
    organic: 0,
    recyclable: 0,
    hazardous: 0,
    other: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/stats');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('📊 Received stats data:', data);
      
      // Backend returns { success: true, current: {...}, trends: {...} }
      if (data.success && data.current) {
        setStats(data.current);
        console.log('✅ Updated stats:', data.current);
      } else {
        console.warn('⚠️ Stats data format incorrect:', data);
        // Fallback to direct data if structure is different
        setStats(data);
      }
      setError(null);
    } catch (err) {
      console.error('❌ Error fetching stats:', err);
      setError('Failed to fetch statistics');
      // Set mock data for development
      setStats({
        total: 0,
        organic: 0,
        recyclable: 0,
        hazardous: 0,
        other: 0
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchStats();
    
    // Update stats every 5 seconds
    const interval = setInterval(fetchStats, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const wasteTypes = [
    {
      key: 'organic',
      label: 'Organic Waste',
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      icon: '🍂'
    },
    {
      key: 'recyclable',
      label: 'Recyclable',
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
      icon: '♻️'
    },
    {
      key: 'hazardous',
      label: 'Hazardous',
      color: 'text-red-600',
      bgColor: 'bg-red-100',
      icon: '☢️'
    },
    {
      key: 'other',
      label: 'Other Waste',
      color: 'text-gray-600',
      bgColor: 'bg-gray-100',
      icon: '🗑️'
    }
  ];

  if (loading && stats.total === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Waste Statistics</h2>
        <div className="animate-pulse space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="flex items-center justify-between p-3 bg-gray-100 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gray-300 rounded"></div>
                <div className="w-24 h-4 bg-gray-300 rounded"></div>
              </div>
              <div className="w-8 h-4 bg-gray-300 rounded"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800">Waste Statistics</h2>
        <button
          onClick={fetchStats}
          className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
          disabled={loading}
        >
          {loading ? '⟳' : '↻'} Refresh
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      <div className="space-y-3">
        {/* Total Count */}
        <div className="p-4 bg-gradient-to-r from-purple-100 to-purple-200 rounded-lg border border-purple-300">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">📊</span>
              <div>
                <p className="text-sm text-purple-700 font-medium">Total Waste Detected</p>
                <p className="text-2xl font-bold text-purple-800">{stats.total}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Individual Categories */}
        {wasteTypes.map(({ key, label, color, bgColor, icon }) => {
          const count = stats[key] || 0;
          const percentage = stats.total > 0 ? Math.round((count / stats.total) * 100) : 0;
          
          return (
            <div key={key} className={`p-3 ${bgColor} rounded-lg border`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <span className="text-xl">{icon}</span>
                  <div>
                    <p className={`text-sm font-medium ${color}`}>{label}</p>
                    <p className="text-xs text-gray-600">{percentage}% of total</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-lg font-bold ${color}`}>{count}</p>
                  {stats.total > 0 && (
                    <div className="w-16 bg-gray-200 rounded-full h-1.5 mt-1">
                      <div 
                        className={`h-1.5 rounded-full ${color.replace('text-', 'bg-')}`}
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Additional Info */}
      <div className="mt-4 pt-3 border-t border-gray-200">
        <div className="flex justify-between items-center text-xs text-gray-500">
          <span>Last updated: {new Date().toLocaleTimeString()}</span>
          <span className="flex items-center">
            <div className={`w-2 h-2 rounded-full mr-1 ${
              loading ? 'bg-yellow-400' : error ? 'bg-red-400' : 'bg-green-400'
            }`}></div>
            {loading ? 'Updating...' : error ? 'Error' : 'Live'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default WasteStats;
