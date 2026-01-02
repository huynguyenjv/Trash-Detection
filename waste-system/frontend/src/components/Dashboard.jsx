import { useState, useEffect, useCallback } from 'react';

// Simple chart components since we don't have a chart library
const BarChart = ({ data, dataKey, color, height = 200 }) => {
  if (!data || data.length === 0) return <div className="text-gray-500 text-center py-8">Không có dữ liệu</div>;
  
  const maxValue = Math.max(...data.map(d => d[dataKey] || 0), 1);
  
  return (
    <div className="flex items-end justify-between gap-1" style={{ height: `${height}px` }}>
      {data.map((item, index) => {
        const value = item[dataKey] || 0;
        const barHeight = (value / maxValue) * 100;
        return (
          <div key={index} className="flex flex-col items-center flex-1 group">
            <div className="relative w-full flex flex-col items-center">
              <span className="text-xs text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity absolute -top-5">
                {value}
              </span>
              <div
                className="w-full rounded-t transition-all duration-300 hover:opacity-80"
                style={{
                  height: `${Math.max(barHeight, 2)}%`,
                  backgroundColor: color,
                  minHeight: '4px'
                }}
              />
            </div>
            <span className="text-xs text-gray-500 mt-1 truncate w-full text-center">
              {item.display || item.date?.slice(-5) || item.hour || ''}
            </span>
          </div>
        );
      })}
    </div>
  );
};

const MultiBarChart = ({ data, height = 200 }) => {
  if (!data || data.length === 0) return <div className="text-gray-500 text-center py-8">Không có dữ liệu</div>;
  
  const categories = ['organic', 'recyclable', 'hazardous', 'other'];
  const colors = {
    organic: '#22c55e',
    recyclable: '#3b82f6',
    hazardous: '#ef4444',
    other: '#6b7280'
  };
  
  const maxValue = Math.max(...data.flatMap(d => categories.map(c => d[c] || 0)), 1);
  
  return (
    <div className="flex items-end justify-between gap-2" style={{ height: `${height}px` }}>
      {data.map((item, index) => (
        <div key={index} className="flex items-end gap-0.5 flex-1 group">
          {categories.map(cat => {
            const value = item[cat] || 0;
            const barHeight = (value / maxValue) * 100;
            return (
              <div
                key={cat}
                className="flex-1 rounded-t transition-all duration-300 hover:opacity-80"
                style={{
                  height: `${Math.max(barHeight, 1)}%`,
                  backgroundColor: colors[cat],
                  minHeight: value > 0 ? '4px' : '1px'
                }}
                title={`${cat}: ${value}`}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
};

const PieChart = ({ data, size = 200 }) => {
  if (!data || data.length === 0 || data.every(d => d.count === 0)) {
    return <div className="text-gray-500 text-center py-8">Không có dữ liệu</div>;
  }
  
  const total = data.reduce((sum, d) => sum + d.count, 0);
  let currentAngle = 0;
  
  const segments = data.map(item => {
    const angle = (item.count / total) * 360;
    const startAngle = currentAngle;
    currentAngle += angle;
    return { ...item, startAngle, angle };
  });
  
  const getCoordinates = (angle, radius) => {
    const rad = (angle - 90) * (Math.PI / 180);
    return {
      x: size / 2 + radius * Math.cos(rad),
      y: size / 2 + radius * Math.sin(rad)
    };
  };
  
  return (
    <div className="flex items-center justify-center gap-4">
      <svg width={size} height={size} className="transform -rotate-90">
        {segments.map((segment, index) => {
          if (segment.count === 0) return null;
          
          const radius = size / 2 - 10;
          const start = getCoordinates(segment.startAngle, radius);
          const end = getCoordinates(segment.startAngle + segment.angle, radius);
          const largeArc = segment.angle > 180 ? 1 : 0;
          
          const pathData = segment.angle >= 360
            ? `M ${size/2} 10 A ${radius} ${radius} 0 1 1 ${size/2 - 0.01} 10 A ${radius} ${radius} 0 1 1 ${size/2} 10`
            : `M ${size/2} ${size/2} L ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y} Z`;
          
          return (
            <path
              key={index}
              d={pathData}
              fill={segment.color}
              className="hover:opacity-80 transition-opacity cursor-pointer"
            />
          );
        })}
        <circle cx={size/2} cy={size/2} r={size/4} fill="#1f2937" />
        <text
          x={size/2}
          y={size/2}
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-white text-2xl font-bold transform rotate-90"
          style={{ transformOrigin: 'center' }}
        >
          {total}
        </text>
      </svg>
      <div className="flex flex-col gap-2">
        {data.map((item, index) => (
          <div key={index} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
            <span className="text-sm text-gray-300">{item.label}: {item.count} ({item.percentage?.toFixed(1)}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const StatCard = ({ title, value, subtitle, icon, color, trend }) => (
  <div className={`p-4 rounded-lg bg-gray-800 border border-gray-700 hover:border-${color}-500 transition-colors`}>
    <div className="flex items-center justify-between mb-2">
      <span className="text-2xl">{icon}</span>
      {trend !== undefined && (
        <span className={`text-xs px-2 py-1 rounded ${trend >= 0 ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'}`}>
          {trend >= 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(1)}%
        </span>
      )}
    </div>
    <div className={`text-3xl font-bold text-${color}-400 mb-1`}>{value?.toLocaleString() || 0}</div>
    <div className="text-sm text-gray-400">{title}</div>
    {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
  </div>
);

const Dashboard = ({ onClose }) => {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedPeriod, setSelectedPeriod] = useState('week');
  const [selectedCategory, setSelectedCategory] = useState(null);
  
  // Data states
  const [overview, setOverview] = useState(null);
  const [dailyChart, setDailyChart] = useState(null);
  const [hourlyChart, setHourlyChart] = useState(null);
  const [categoryDistribution, setCategoryDistribution] = useState(null);
  const [locationStats, setLocationStats] = useState(null);
  const [binStats, setBinStats] = useState(null);
  const [topDetections, setTopDetections] = useState(null);
  const [sessions, setSessions] = useState(null);
  const [categoryDetails, setCategoryDetails] = useState(null);
  
  // Detections table state
  const [detectionsData, setDetectionsData] = useState(null);
  const [detectionsLoading, setDetectionsLoading] = useState(false);
  const [detectionsFilters, setDetectionsFilters] = useState({
    category: '',
    label: '',
    dateFrom: '',
    dateTo: '',
    minConfidence: '',
    sessionId: ''
  });
  const [detectionsPage, setDetectionsPage] = useState(0);
  const [detectionsSortBy, setDetectionsSortBy] = useState('detected_at');
  const [detectionsSortOrder, setDetectionsSortOrder] = useState('desc');
  
  const API_BASE = 'http://localhost:8000';
  
  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [
        overviewRes,
        dailyRes,
        hourlyRes,
        distributionRes,
        locationRes,
        binRes,
        topRes,
        sessionsRes
      ] = await Promise.all([
        fetch(`${API_BASE}/dashboard/overview`),
        fetch(`${API_BASE}/dashboard/daily-chart?days=${selectedPeriod === 'week' ? 7 : selectedPeriod === 'month' ? 30 : 365}`),
        fetch(`${API_BASE}/dashboard/hourly-chart`),
        fetch(`${API_BASE}/dashboard/category-distribution?period=${selectedPeriod === 'week' ? 'week' : selectedPeriod === 'month' ? 'month' : 'all'}`),
        fetch(`${API_BASE}/dashboard/by-location?days=30`),
        fetch(`${API_BASE}/dashboard/bin-statistics`),
        fetch(`${API_BASE}/dashboard/top-detections?limit=10&days=30`),
        fetch(`${API_BASE}/dashboard/sessions?limit=10`)
      ]);
      
      setOverview(await overviewRes.json());
      setDailyChart(await dailyRes.json());
      setHourlyChart(await hourlyRes.json());
      setCategoryDistribution(await distributionRes.json());
      setLocationStats(await locationRes.json());
      setBinStats(await binRes.json());
      setTopDetections(await topRes.json());
      setSessions(await sessionsRes.json());
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  }, [selectedPeriod]);
  
  const fetchCategoryDetails = useCallback(async (category) => {
    try {
      const res = await fetch(`${API_BASE}/dashboard/by-category/${category}?days=30`);
      setCategoryDetails(await res.json());
    } catch (error) {
      console.error('Error fetching category details:', error);
    }
  }, []);
  
  // Fetch detections with filters
  const fetchDetections = useCallback(async () => {
    setDetectionsLoading(true);
    try {
      const params = new URLSearchParams();
      params.append('skip', detectionsPage * 50);
      params.append('limit', 50);
      params.append('sort_by', detectionsSortBy);
      params.append('sort_order', detectionsSortOrder);
      
      if (detectionsFilters.category) params.append('category', detectionsFilters.category);
      if (detectionsFilters.label) params.append('label', detectionsFilters.label);
      if (detectionsFilters.dateFrom) params.append('date_from', detectionsFilters.dateFrom);
      if (detectionsFilters.dateTo) params.append('date_to', detectionsFilters.dateTo);
      if (detectionsFilters.minConfidence) params.append('min_confidence', detectionsFilters.minConfidence);
      if (detectionsFilters.sessionId) params.append('session_id', detectionsFilters.sessionId);
      
      const res = await fetch(`${API_BASE}/dashboard/detections?${params.toString()}`);
      setDetectionsData(await res.json());
    } catch (error) {
      console.error('Error fetching detections:', error);
    } finally {
      setDetectionsLoading(false);
    }
  }, [detectionsPage, detectionsSortBy, detectionsSortOrder, detectionsFilters]);
  
  useEffect(() => {
    fetchData();
  }, [fetchData]);
  
  useEffect(() => {
    if (selectedCategory) {
      fetchCategoryDetails(selectedCategory);
    }
  }, [selectedCategory, fetchCategoryDetails]);
  
  useEffect(() => {
    if (activeTab === 'detections') {
      fetchDetections();
    }
  }, [activeTab, fetchDetections]);
  
  const handleFilterChange = (key, value) => {
    setDetectionsFilters(prev => ({ ...prev, [key]: value }));
    setDetectionsPage(0); // Reset to first page when filter changes
  };
  
  const handleSort = (column) => {
    if (detectionsSortBy === column) {
      setDetectionsSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setDetectionsSortBy(column);
      setDetectionsSortOrder('desc');
    }
  };
  
  const clearFilters = () => {
    setDetectionsFilters({
      category: '',
      label: '',
      dateFrom: '',
      dateTo: '',
      minConfidence: '',
      sessionId: ''
    });
    setDetectionsPage(0);
  };
  
  const categoryColors = {
    organic: 'green',
    recyclable: 'blue',
    hazardous: 'red',
    other: 'gray'
  };
  
  const categoryLabels = {
    organic: 'Hữu cơ',
    recyclable: 'Tái chế',
    hazardous: 'Nguy hại',
    other: 'Khác'
  };
  
  const categoryIcons = {
    organic: '🍂',
    recyclable: '♻️',
    hazardous: '⚠️',
    other: '🗑️'
  };
  
  if (loading && !overview) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-900/95">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Đang tải dữ liệu thống kê...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="fixed inset-0 z-50 overflow-auto bg-gray-900">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              📊 Dashboard Thống Kê
            </h1>
            <div className="flex gap-2 ml-8">
              {['overview', 'detections', 'categories', 'locations', 'sessions'].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === tab
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {tab === 'overview' && 'Tổng quan'}
                  {tab === 'detections' && '📋 Danh sách rác'}
                  {tab === 'categories' && 'Theo loại rác'}
                  {tab === 'locations' && 'Theo địa điểm'}
                  {tab === 'sessions' && 'Lịch sử phiên'}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-4">
            <select
              value={selectedPeriod}
              onChange={(e) => setSelectedPeriod(e.target.value)}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
            >
              <option value="week">7 ngày qua</option>
              <option value="month">30 ngày qua</option>
              <option value="year">Cả năm</option>
            </select>
            <button
              onClick={fetchData}
              className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
              title="Làm mới"
            >
              🔄
            </button>
            <button
              onClick={onClose}
              className="p-2 bg-red-600 rounded-lg hover:bg-red-700 transition-colors text-white"
            >
              ✕
            </button>
          </div>
        </div>
      </div>
      
      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              <StatCard
                title="Tổng phát hiện"
                value={overview?.all_time?.total}
                subtitle="Tất cả thời gian"
                icon="📦"
                color="white"
              />
              <StatCard
                title="Hữu cơ"
                value={overview?.all_time?.organic}
                subtitle={`${((overview?.all_time?.organic / overview?.all_time?.total) * 100 || 0).toFixed(1)}% tổng số`}
                icon="🍂"
                color="green"
              />
              <StatCard
                title="Tái chế"
                value={overview?.all_time?.recyclable}
                subtitle={`${((overview?.all_time?.recyclable / overview?.all_time?.total) * 100 || 0).toFixed(1)}% tổng số`}
                icon="♻️"
                color="blue"
              />
              <StatCard
                title="Nguy hại"
                value={overview?.all_time?.hazardous}
                subtitle={`${((overview?.all_time?.hazardous / overview?.all_time?.total) * 100 || 0).toFixed(1)}% tổng số`}
                icon="⚠️"
                color="red"
              />
              <StatCard
                title="Phiên làm việc"
                value={overview?.all_time?.sessions}
                subtitle="Tổng số phiên"
                icon="📱"
                color="purple"
              />
            </div>
            
            {/* Time-based Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <span>📅</span> Hôm nay
                </h3>
                <div className="text-3xl font-bold text-blue-400">{overview?.today?.total || 0}</div>
                <div className="text-sm text-gray-400 mt-2">
                  {overview?.today?.sessions || 0} phiên • {overview?.today?.organic || 0} hữu cơ • {overview?.today?.recyclable || 0} tái chế
                </div>
              </div>
              <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <span>📆</span> Tuần này
                </h3>
                <div className="text-3xl font-bold text-green-400">{overview?.weekly?.total || 0}</div>
                <div className="text-sm text-gray-400 mt-2">
                  {overview?.weekly?.sessions || 0} phiên • Trung bình {((overview?.weekly?.total || 0) / 7).toFixed(1)}/ngày
                </div>
              </div>
              <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <span>🗓️</span> Tháng này
                </h3>
                <div className="text-3xl font-bold text-yellow-400">{overview?.monthly?.total || 0}</div>
                <div className="text-sm text-gray-400 mt-2">
                  {overview?.monthly?.sessions || 0} phiên • Trung bình {((overview?.monthly?.total || 0) / 30).toFixed(1)}/ngày
                </div>
              </div>
            </div>
            
            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Daily Chart */}
              <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>📈</span> Biểu đồ theo ngày
                </h3>
                <div className="mb-4 flex gap-2 flex-wrap">
                  <span className="text-xs px-2 py-1 bg-green-900 text-green-400 rounded">● Hữu cơ</span>
                  <span className="text-xs px-2 py-1 bg-blue-900 text-blue-400 rounded">● Tái chế</span>
                  <span className="text-xs px-2 py-1 bg-red-900 text-red-400 rounded">● Nguy hại</span>
                  <span className="text-xs px-2 py-1 bg-gray-700 text-gray-400 rounded">● Khác</span>
                </div>
                <MultiBarChart data={dailyChart?.chart_data?.slice(-14)} height={180} />
              </div>
              
              {/* Pie Chart */}
              <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>🥧</span> Phân bố theo loại
                </h3>
                <PieChart data={categoryDistribution?.distribution} size={200} />
              </div>
            </div>
            
            {/* Hourly Chart */}
            <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span>⏰</span> Thống kê theo giờ (Hôm nay)
              </h3>
              <BarChart data={hourlyChart?.chart_data} dataKey="total" color="#3b82f6" height={150} />
            </div>
            
            {/* Top Detections & Bins */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Top Detected Items */}
              <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>🏆</span> Rác phát hiện nhiều nhất
                </h3>
                <div className="space-y-3">
                  {topDetections?.top_items?.slice(0, 5).map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <span className="text-lg font-bold text-gray-500">#{index + 1}</span>
                        <div>
                          <div className="font-medium text-white">{item.label}</div>
                          <div className="text-xs text-gray-400">{categoryLabels[item.category]}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-blue-400">{item.count}</div>
                        <div className="text-xs text-gray-500">{item.percentage?.toFixed(1)}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Bin Statistics */}
              <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <span>🗑️</span> Thống kê thùng rác ({binStats?.total_bins || 0} thùng)
                </h3>
                <div className="space-y-3">
                  {Object.entries(binStats?.by_category || {}).map(([cat, stats]) => (
                    <div key={cat} className="p-3 bg-gray-700/50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span>{categoryIcons[cat]}</span>
                          <span className="font-medium text-white">{categoryLabels[cat]}</span>
                        </div>
                        <span className="text-sm text-gray-400">{stats.count} thùng</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Độ đầy TB: {stats.avg_fill?.toFixed(1)}%</span>
                        <span className="text-gray-400">Phát hiện: {stats.total_detections}</span>
                      </div>
                      <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden">
                        <div
                          className={`h-full bg-${categoryColors[cat]}-500 rounded-full`}
                          style={{ width: `${stats.avg_fill || 0}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Detections Tab - All Detections Table with Filters */}
        {activeTab === 'detections' && (
          <div className="space-y-6">
            {/* Filters Section */}
            <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <span>🔍</span> Bộ lọc
                </h3>
                <button
                  onClick={clearFilters}
                  className="px-3 py-1 text-sm bg-gray-700 rounded hover:bg-gray-600 text-gray-300"
                >
                  Xóa bộ lọc
                </button>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {/* Category Filter */}
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Loại rác</label>
                  <select
                    value={detectionsFilters.category}
                    onChange={(e) => handleFilterChange('category', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                  >
                    <option value="">Tất cả</option>
                    <option value="organic">🍂 Hữu cơ</option>
                    <option value="recyclable">♻️ Tái chế</option>
                    <option value="hazardous">⚠️ Nguy hại</option>
                    <option value="other">🗑️ Khác</option>
                  </select>
                </div>
                
                {/* Label Filter */}
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Tên rác</label>
                  <input
                    type="text"
                    value={detectionsFilters.label}
                    onChange={(e) => handleFilterChange('label', e.target.value)}
                    placeholder="Tìm theo tên..."
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                  />
                </div>
                
                {/* Date From */}
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Từ ngày</label>
                  <input
                    type="date"
                    value={detectionsFilters.dateFrom}
                    onChange={(e) => handleFilterChange('dateFrom', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                  />
                </div>
                
                {/* Date To */}
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Đến ngày</label>
                  <input
                    type="date"
                    value={detectionsFilters.dateTo}
                    onChange={(e) => handleFilterChange('dateTo', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                  />
                </div>
                
                {/* Min Confidence */}
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Độ tin cậy tối thiểu</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={detectionsFilters.minConfidence}
                    onChange={(e) => handleFilterChange('minConfidence', e.target.value)}
                    placeholder="0.0 - 1.0"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                  />
                </div>
                
                {/* Session ID */}
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Phiên ID</label>
                  <input
                    type="number"
                    value={detectionsFilters.sessionId}
                    onChange={(e) => handleFilterChange('sessionId', e.target.value)}
                    placeholder="Session ID"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                  />
                </div>
              </div>
              
              {/* Apply filter button */}
              <div className="mt-4 flex justify-end">
                <button
                  onClick={fetchDetections}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
                >
                  <span>🔍</span> Áp dụng bộ lọc
                </button>
              </div>
            </div>
            
            {/* Results Summary */}
            <div className="flex items-center justify-between">
              <div className="text-gray-400">
                Tìm thấy <span className="text-white font-bold">{detectionsData?.total || 0}</span> kết quả
                {detectionsData?.filters && Object.values(detectionsData.filters).some(v => v) && (
                  <span className="text-blue-400 ml-2">(đã lọc)</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <span>Sắp xếp:</span>
                <select
                  value={`${detectionsSortBy}-${detectionsSortOrder}`}
                  onChange={(e) => {
                    const [field, order] = e.target.value.split('-');
                    setDetectionsSortBy(field);
                    setDetectionsSortOrder(order);
                  }}
                  className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white"
                >
                  <option value="detected_at-desc">Mới nhất</option>
                  <option value="detected_at-asc">Cũ nhất</option>
                  <option value="confidence-desc">Độ tin cậy cao</option>
                  <option value="confidence-asc">Độ tin cậy thấp</option>
                  <option value="label-asc">Tên A-Z</option>
                  <option value="label-desc">Tên Z-A</option>
                </select>
              </div>
            </div>
            
            {/* Detections Table */}
            <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
              {detectionsLoading ? (
                <div className="p-8 text-center">
                  <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <p className="text-gray-400">Đang tải...</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-700/50">
                      <tr className="text-left text-sm text-gray-400">
                        <th className="p-4 font-medium">ID</th>
                        <th className="p-4 font-medium cursor-pointer hover:text-white" onClick={() => handleSort('label')}>
                          Tên rác {detectionsSortBy === 'label' && (detectionsSortOrder === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="p-4 font-medium cursor-pointer hover:text-white" onClick={() => handleSort('category')}>
                          Loại {detectionsSortBy === 'category' && (detectionsSortOrder === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="p-4 font-medium cursor-pointer hover:text-white" onClick={() => handleSort('confidence')}>
                          Độ tin cậy {detectionsSortBy === 'confidence' && (detectionsSortOrder === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="p-4 font-medium">Phiên</th>
                        <th className="p-4 font-medium cursor-pointer hover:text-white" onClick={() => handleSort('detected_at')}>
                          Thời gian {detectionsSortBy === 'detected_at' && (detectionsSortOrder === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="p-4 font-medium">Vị trí</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detectionsData?.detections?.map((det, index) => (
                        <tr 
                          key={det.id} 
                          className={`border-t border-gray-700/50 hover:bg-gray-700/30 ${
                            index % 2 === 0 ? 'bg-gray-800/30' : ''
                          }`}
                        >
                          <td className="p-4 text-gray-500">#{det.id}</td>
                          <td className="p-4">
                            <span className="font-medium text-white">{det.label}</span>
                          </td>
                          <td className="p-4">
                            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium
                              ${det.category === 'organic' ? 'bg-green-900/50 text-green-400' : ''}
                              ${det.category === 'recyclable' ? 'bg-blue-900/50 text-blue-400' : ''}
                              ${det.category === 'hazardous' ? 'bg-red-900/50 text-red-400' : ''}
                              ${det.category === 'other' ? 'bg-gray-700 text-gray-400' : ''}
                            `}>
                              {categoryIcons[det.category]} {categoryLabels[det.category]}
                            </span>
                          </td>
                          <td className="p-4">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-2 bg-gray-600 rounded-full overflow-hidden">
                                <div 
                                  className={`h-full rounded-full ${
                                    det.confidence >= 0.8 ? 'bg-green-500' : 
                                    det.confidence >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                  style={{ width: `${det.confidence * 100}%` }}
                                />
                              </div>
                              <span className="text-sm text-gray-300">{(det.confidence * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                          <td className="p-4 text-gray-400">#{det.session_id}</td>
                          <td className="p-4 text-gray-300 text-sm">
                            {new Date(det.detected_at).toLocaleString('vi-VN')}
                          </td>
                          <td className="p-4 text-gray-400 text-sm">
                            {det.latitude && det.longitude ? (
                              <span title={`${det.latitude}, ${det.longitude}`}>📍 Có vị trí</span>
                            ) : (
                              <span className="text-gray-600">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  
                  {(!detectionsData?.detections || detectionsData.detections.length === 0) && (
                    <div className="p-8 text-center text-gray-400">
                      Không tìm thấy kết quả nào
                    </div>
                  )}
                </div>
              )}
              
              {/* Pagination */}
              {detectionsData?.total > 0 && (
                <div className="p-4 border-t border-gray-700 flex items-center justify-between">
                  <div className="text-sm text-gray-400">
                    Hiển thị {detectionsPage * 50 + 1} - {Math.min((detectionsPage + 1) * 50, detectionsData.total)} / {detectionsData.total}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setDetectionsPage(prev => Math.max(0, prev - 1))}
                      disabled={detectionsPage === 0}
                      className="px-3 py-1 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-white"
                    >
                      ← Trước
                    </button>
                    <span className="px-3 py-1 text-gray-400">
                      Trang {detectionsPage + 1} / {Math.ceil(detectionsData.total / 50)}
                    </span>
                    <button
                      onClick={() => setDetectionsPage(prev => prev + 1)}
                      disabled={!detectionsData.has_more}
                      className="px-3 py-1 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-white"
                    >
                      Sau →
                    </button>
                  </div>
                </div>
              )}
            </div>
            
            {/* Available Labels Quick Filter */}
            {detectionsData?.available_labels?.length > 0 && (
              <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                <h4 className="text-sm font-medium text-gray-400 mb-3">Lọc nhanh theo tên:</h4>
                <div className="flex flex-wrap gap-2">
                  {detectionsData.available_labels.slice(0, 20).map(label => (
                    <button
                      key={label}
                      onClick={() => handleFilterChange('label', label)}
                      className={`px-3 py-1 rounded-full text-sm transition-colors ${
                        detectionsFilters.label === label
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                  {detectionsData.available_labels.length > 20 && (
                    <span className="px-3 py-1 text-gray-500 text-sm">
                      +{detectionsData.available_labels.length - 20} khác
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Categories Tab */}
        {activeTab === 'categories' && (
          <div className="space-y-6">
            {/* Category Selection */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {['organic', 'recyclable', 'hazardous', 'other'].map(cat => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`p-6 rounded-lg border-2 transition-all ${
                    selectedCategory === cat
                      ? `bg-${categoryColors[cat]}-900/50 border-${categoryColors[cat]}-500`
                      : 'bg-gray-800 border-gray-700 hover:border-gray-500'
                  }`}
                >
                  <div className="text-3xl mb-2">{categoryIcons[cat]}</div>
                  <div className="text-lg font-semibold text-white">{categoryLabels[cat]}</div>
                  <div className={`text-2xl font-bold text-${categoryColors[cat]}-400 mt-2`}>
                    {overview?.all_time?.[cat] || 0}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {((overview?.all_time?.[cat] / overview?.all_time?.total) * 100 || 0).toFixed(1)}% tổng số
                  </div>
                </button>
              ))}
            </div>
            
            {/* Category Details */}
            {selectedCategory && categoryDetails && (
              <div className="space-y-6">
                <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <span>{categoryIcons[selectedCategory]}</span>
                    Chi tiết {categoryLabels[selectedCategory]} - 30 ngày qua
                  </h3>
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                      <div className="text-2xl font-bold text-white">{categoryDetails.total_count}</div>
                      <div className="text-sm text-gray-400">Tổng phát hiện</div>
                    </div>
                    <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                      <div className="text-2xl font-bold text-white">{categoryDetails.avg_per_day?.toFixed(1)}</div>
                      <div className="text-sm text-gray-400">Trung bình/ngày</div>
                    </div>
                    <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                      <div className="text-2xl font-bold text-white">{categoryDetails.related_bins?.length || 0}</div>
                      <div className="text-sm text-gray-400">Thùng rác liên quan</div>
                    </div>
                  </div>
                  <h4 className="text-md font-medium text-white mb-3">Biểu đồ theo ngày</h4>
                  <BarChart
                    data={categoryDetails.daily_chart}
                    dataKey="count"
                    color={categoryColors[selectedCategory] === 'green' ? '#22c55e' : 
                           categoryColors[selectedCategory] === 'blue' ? '#3b82f6' :
                           categoryColors[selectedCategory] === 'red' ? '#ef4444' : '#6b7280'}
                    height={200}
                  />
                </div>
                
                {/* Related Bins */}
                {categoryDetails.related_bins?.length > 0 && (
                  <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
                    <h3 className="text-lg font-semibold text-white mb-4">
                      Thùng rác {categoryLabels[selectedCategory]}
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {categoryDetails.related_bins.map(bin => (
                        <div key={bin.id} className="p-4 bg-gray-700/50 rounded-lg">
                          <div className="font-medium text-white mb-2">{bin.name}</div>
                          <div className="text-sm text-gray-400 mb-2">{bin.address || 'Không có địa chỉ'}</div>
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-gray-400">Độ đầy:</span>
                            <span className={`font-medium ${bin.current_fill > 80 ? 'text-red-400' : 'text-green-400'}`}>
                              {bin.current_fill?.toFixed(0)}%
                            </span>
                          </div>
                          <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${bin.current_fill > 80 ? 'bg-red-500' : 'bg-green-500'}`}
                              style={{ width: `${bin.current_fill || 0}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {!selectedCategory && (
              <div className="text-center py-12 text-gray-400">
                <span className="text-4xl block mb-4">👆</span>
                Chọn một loại rác để xem chi tiết
              </div>
            )}
          </div>
        )}
        
        {/* Locations Tab */}
        {activeTab === 'locations' && (
          <div className="space-y-6">
            <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span>📍</span> Thống kê theo địa điểm thu gom ({locationStats?.total_bins || 0} điểm)
              </h3>
              
              {/* Summary */}
              <div className="grid grid-cols-4 gap-4 mb-6">
                {Object.entries(overview?.bins?.by_category || {}).map(([cat, count]) => (
                  <div key={cat} className="p-4 bg-gray-700/50 rounded-lg text-center">
                    <div className="text-2xl mb-1">{categoryIcons[cat]}</div>
                    <div className="text-xl font-bold text-white">{count}</div>
                    <div className="text-xs text-gray-400">{categoryLabels[cat]}</div>
                  </div>
                ))}
              </div>
              
              {/* Location List */}
              <div className="space-y-4">
                {locationStats?.locations?.map((loc, index) => (
                  <div key={loc.bin_id} className="p-4 bg-gray-700/50 rounded-lg">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-lg font-bold text-gray-500">#{index + 1}</span>
                          <span className="font-medium text-white">{loc.bin_name}</span>
                          <span className={`text-xs px-2 py-0.5 rounded bg-${categoryColors[loc.category]}-900 text-${categoryColors[loc.category]}-400`}>
                            {categoryLabels[loc.category]}
                          </span>
                        </div>
                        <div className="text-sm text-gray-400">{loc.address || 'Không có địa chỉ'}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-xl font-bold text-blue-400">{loc.total_nearby_detections}</div>
                        <div className="text-xs text-gray-500">phát hiện gần đây</div>
                      </div>
                    </div>
                    
                    {/* Category breakdown */}
                    <div className="grid grid-cols-4 gap-2 mt-3">
                      {Object.entries(loc.detections_by_category).map(([cat, count]) => (
                        <div key={cat} className="text-center p-2 bg-gray-800 rounded">
                          <div className="text-sm font-medium text-white">{count}</div>
                          <div className="text-xs text-gray-500">{categoryLabels[cat]}</div>
                        </div>
                      ))}
                    </div>
                    
                    {/* Fill level */}
                    <div className="mt-3 flex items-center gap-2">
                      <span className="text-xs text-gray-400">Độ đầy:</span>
                      <div className="flex-1 h-2 bg-gray-600 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${loc.current_fill > 80 ? 'bg-red-500' : loc.current_fill > 50 ? 'bg-yellow-500' : 'bg-green-500'}`}
                          style={{ width: `${loc.current_fill || 0}%` }}
                        />
                      </div>
                      <span className="text-xs text-white font-medium">{loc.current_fill?.toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {/* Sessions Tab */}
        {activeTab === 'sessions' && (
          <div className="space-y-6">
            <div className="p-6 bg-gray-800 rounded-lg border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span>📱</span> Lịch sử phiên làm việc ({sessions?.total || 0} phiên)
              </h3>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-sm text-gray-400 border-b border-gray-700">
                      <th className="pb-3 px-2">ID</th>
                      <th className="pb-3 px-2">Thời gian bắt đầu</th>
                      <th className="pb-3 px-2">Thời lượng</th>
                      <th className="pb-3 px-2">Tổng</th>
                      <th className="pb-3 px-2">🍂</th>
                      <th className="pb-3 px-2">♻️</th>
                      <th className="pb-3 px-2">⚠️</th>
                      <th className="pb-3 px-2">🗑️</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessions?.sessions?.map(session => (
                      <tr key={session.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                        <td className="py-3 px-2 text-gray-400">#{session.id}</td>
                        <td className="py-3 px-2 text-white">
                          {new Date(session.started_at).toLocaleString('vi-VN')}
                        </td>
                        <td className="py-3 px-2 text-gray-400">
                          {session.duration_seconds
                            ? `${Math.floor(session.duration_seconds / 60)}m ${Math.floor(session.duration_seconds % 60)}s`
                            : 'Đang chạy'}
                        </td>
                        <td className="py-3 px-2 text-white font-bold">{session.total_detections}</td>
                        <td className="py-3 px-2 text-green-400">{session.organic_count}</td>
                        <td className="py-3 px-2 text-blue-400">{session.recyclable_count}</td>
                        <td className="py-3 px-2 text-red-400">{session.hazardous_count}</td>
                        <td className="py-3 px-2 text-gray-400">{session.other_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {(!sessions?.sessions || sessions.sessions.length === 0) && (
                <div className="text-center py-8 text-gray-400">
                  Chưa có phiên làm việc nào
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
