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
  
  // Waste bins management state
  const [wasteBins, setWasteBins] = useState([]);
  const [wasteBinsLoading, setWasteBinsLoading] = useState(false);
  const [wasteBinsPage, setWasteBinsPage] = useState(0);
  const BINS_PER_PAGE = 10;
  const [showBinModal, setShowBinModal] = useState(false);
  const [editingBin, setEditingBin] = useState(null);
  const [binFormData, setBinFormData] = useState({
    name: '',
    category: 'other',
    latitude: '',
    longitude: '',
    address: '',
    capacity: 100,
    current_fill: 0
  });
  const [binFormError, setBinFormError] = useState('');
  
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
  
  // Fetch waste bins
  const fetchWasteBins = useCallback(async () => {
    setWasteBinsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/bins?active_only=false`);
      const data = await res.json();
      setWasteBins(data);
    } catch (error) {
      console.error('Error fetching waste bins:', error);
    } finally {
      setWasteBinsLoading(false);
    }
  }, []);
  
  // Create or update waste bin
  const handleSaveBin = async () => {
    setBinFormError('');
    
    // Validate
    if (!binFormData.name.trim()) {
      setBinFormError('Vui lòng nhập tên bãi rác');
      return;
    }
    if (!binFormData.latitude || !binFormData.longitude) {
      setBinFormError('Vui lòng nhập tọa độ');
      return;
    }
    
    try {
      const payload = {
        name: binFormData.name,
        category: binFormData.category,
        latitude: parseFloat(binFormData.latitude),
        longitude: parseFloat(binFormData.longitude),
        address: binFormData.address || '',
        capacity: parseFloat(binFormData.capacity) || 100,
        current_fill: parseFloat(binFormData.current_fill) || 0
      };
      
      let res;
      if (editingBin) {
        // Update existing bin
        res = await fetch(`${API_BASE}/bins/${editingBin.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      } else {
        // Create new bin
        res = await fetch(`${API_BASE}/bins`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      }
      
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Lỗi khi lưu');
      }
      
      // Refresh list and close modal
      await fetchWasteBins();
      setShowBinModal(false);
      setEditingBin(null);
      resetBinForm();
    } catch (error) {
      console.error('Error saving bin:', error);
      setBinFormError(error.message || 'Lỗi khi lưu bãi rác');
    }
  };
  
  // Delete waste bin
  const handleDeleteBin = async (binId) => {
    if (!confirm('Bạn có chắc muốn xóa bãi rác này?')) return;
    
    try {
      const res = await fetch(`${API_BASE}/bins/${binId}`, {
        method: 'DELETE'
      });
      
      if (!res.ok) {
        throw new Error('Lỗi khi xóa');
      }
      
      await fetchWasteBins();
    } catch (error) {
      console.error('Error deleting bin:', error);
      alert('Lỗi khi xóa bãi rác');
    }
  };
  
  // Toggle bin active status
  const handleToggleBinActive = async (bin) => {
    try {
      const res = await fetch(`${API_BASE}/bins/${bin.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_active: !bin.is_active })
      });
      
      if (!res.ok) {
        throw new Error('Lỗi khi cập nhật');
      }
      
      await fetchWasteBins();
    } catch (error) {
      console.error('Error toggling bin status:', error);
    }
  };
  
  // Open edit modal
  const handleEditBin = (bin) => {
    setEditingBin(bin);
    setBinFormData({
      name: bin.name,
      category: bin.category,
      latitude: bin.latitude.toString(),
      longitude: bin.longitude.toString(),
      address: bin.address || '',
      capacity: bin.capacity,
      current_fill: bin.current_fill
    });
    setBinFormError('');
    setShowBinModal(true);
  };
  
  // Reset form
  const resetBinForm = () => {
    setBinFormData({
      name: '',
      category: 'other',
      latitude: '',
      longitude: '',
      address: '',
      capacity: 100,
      current_fill: 0
    });
    setBinFormError('');
  };
  
  // Open add modal
  const handleAddBin = () => {
    setEditingBin(null);
    resetBinForm();
    setShowBinModal(true);
  };
  
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
  
  useEffect(() => {
    if (activeTab === 'wastebins') {
      fetchWasteBins();
    }
  }, [activeTab, fetchWasteBins]);
  
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
  
  const tabs = [
    { id: 'overview', label: 'Tổng quan', icon: '📊' },
    { id: 'detections', label: 'Danh sách rác', icon: '📋' },
    { id: 'categories', label: 'Theo loại rác', icon: '🏷️' },
    { id: 'locations', label: 'Theo địa điểm', icon: '📍' },
    { id: 'wastebins', label: 'Quản lý bãi rác', icon: '🗺️' },
    { id: 'sessions', label: 'Lịch sử phiên', icon: '📅' },
  ];
  
  return (
    <div className="fixed inset-0 z-50 overflow-auto bg-gray-900">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-800 border-b border-gray-700">
        {/* Top bar with title and controls */}
        <div className="px-4 md:px-6 py-3 md:py-4">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <h1 className="text-lg md:text-2xl font-bold text-white flex items-center gap-2">
              📊 <span className="hidden sm:inline">Dashboard</span> Thống Kê
            </h1>
            <div className="flex items-center gap-2 md:gap-4">
              <select
                value={selectedPeriod}
                onChange={(e) => setSelectedPeriod(e.target.value)}
                className="px-2 md:px-3 py-1.5 md:py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-xs md:text-sm"
              >
                <option value="week">7 ngày</option>
                <option value="month">30 ngày</option>
                <option value="year">Cả năm</option>
              </select>
              <button
                onClick={fetchData}
                className="p-1.5 md:p-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
                title="Làm mới"
              >
                🔄
              </button>
              <button
                onClick={onClose}
                className="p-1.5 md:p-2 bg-red-600 rounded-lg hover:bg-red-700 transition-colors text-white"
              >
                ✕
              </button>
            </div>
          </div>
        </div>
        
        {/* Tab Navigation - Horizontal scroll on mobile */}
        <div className="px-4 md:px-6 pb-2">
          <div className="max-w-7xl mx-auto">
            <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide -mx-4 px-4 md:mx-0 md:px-0">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 md:px-4 py-2 rounded-lg text-xs md:text-sm font-medium transition-colors whitespace-nowrap flex-shrink-0 ${
                    activeTab === tab.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span className="hidden sm:inline">{tab.label}</span>
                  <span className="sm:hidden">{tab.label.split(' ')[0]}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 md:py-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3 md:gap-4">
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
        
        {/* Waste Bins Management Tab */}
        {activeTab === 'wastebins' && (
          <div className="space-y-6">
            {/* Header with Add Button */}
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <span>🗺️</span> Quản lý bãi rác ({wasteBins.length} địa điểm)
              </h3>
              <button
                onClick={handleAddBin}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <span>➕</span> Thêm bãi rác
              </button>
            </div>
            
            {/* Stats Summary */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 md:gap-4">
              <div className="p-3 md:p-4 bg-gray-800 rounded-lg border border-gray-700 text-center">
                <div className="text-xl md:text-2xl font-bold text-white">{wasteBins.length}</div>
                <div className="text-xs text-gray-400">Tổng số</div>
              </div>
              <div className="p-3 md:p-4 bg-gray-800 rounded-lg border border-green-700 text-center">
                <div className="text-xl md:text-2xl font-bold text-green-400">{wasteBins.filter(b => b.category === 'organic').length}</div>
                <div className="text-xs text-gray-400">🍂 Hữu cơ</div>
              </div>
              <div className="p-3 md:p-4 bg-gray-800 rounded-lg border border-blue-700 text-center">
                <div className="text-xl md:text-2xl font-bold text-blue-400">{wasteBins.filter(b => b.category === 'recyclable').length}</div>
                <div className="text-xs text-gray-400">♻️ Tái chế</div>
              </div>
              <div className="p-3 md:p-4 bg-gray-800 rounded-lg border border-red-700 text-center">
                <div className="text-xl md:text-2xl font-bold text-red-400">{wasteBins.filter(b => b.category === 'hazardous').length}</div>
                <div className="text-xs text-gray-400">⚠️ Nguy hại</div>
              </div>
              <div className="p-3 md:p-4 bg-gray-800 rounded-lg border border-gray-600 text-center col-span-2 sm:col-span-1">
                <div className="text-xl md:text-2xl font-bold text-gray-400">{wasteBins.filter(b => b.category === 'other').length}</div>
                <div className="text-xs text-gray-400">🗑️ Khác</div>
              </div>
            </div>
            
            {/* Bins Table - Desktop */}
            <div className="hidden md:block p-4 md:p-6 bg-gray-800 rounded-lg border border-gray-700">
              {wasteBinsLoading ? (
                <div className="text-center py-8">
                  <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <p className="text-gray-400">Đang tải...</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full table-auto">
                    <thead>
                      <tr className="text-left text-xs text-gray-400 border-b border-gray-700">
                        <th className="pb-3 px-2">#</th>
                        <th className="pb-3 px-2">Tên</th>
                        <th className="pb-3 px-2">Loại</th>
                        <th className="pb-3 px-2">Địa chỉ</th>
                        <th className="pb-3 px-2">Sức chứa</th>
                        <th className="pb-3 px-2">Độ đầy</th>
                        <th className="pb-3 px-2 text-center">Hành động</th>
                      </tr>
                    </thead>
                    <tbody>
                      {wasteBins
                        .slice(wasteBinsPage * BINS_PER_PAGE, (wasteBinsPage + 1) * BINS_PER_PAGE)
                        .map(bin => {
                        const catStyle = {
                          organic: 'bg-green-900/50 text-green-400 border-green-700',
                          recyclable: 'bg-blue-900/50 text-blue-400 border-blue-700',
                          hazardous: 'bg-red-900/50 text-red-400 border-red-700',
                          other: 'bg-gray-700/50 text-gray-400 border-gray-600'
                        }[bin.category] || 'bg-gray-700/50 text-gray-400 border-gray-600';
                        
                        return (
                        <tr key={bin.id} className={`border-b border-gray-700/50 hover:bg-gray-700/30 ${!bin.is_active ? 'opacity-50' : ''}`}>
                          <td className="py-3 px-2 text-gray-400 text-sm">{bin.id}</td>
                          <td className="py-3 px-2">
                            <div className="text-white font-medium text-sm">{bin.name}</div>
                            <div className="text-gray-500 text-xs">{bin.latitude?.toFixed(4)}, {bin.longitude?.toFixed(4)}</div>
                          </td>
                          <td className="py-3 px-2">
                            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs border whitespace-nowrap ${catStyle}`}>
                              {categoryIcons[bin.category]} {categoryLabels[bin.category]}
                            </span>
                          </td>
                          <td className="py-3 px-2 text-gray-400 text-sm max-w-[200px] truncate">{bin.address || '-'}</td>
                          <td className="py-3 px-2 text-white text-sm">{bin.capacity}</td>
                          <td className="py-3 px-2">
                            <div className="flex items-center gap-2">
                              <div className="w-12 h-2 bg-gray-600 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${bin.current_fill > 80 ? 'bg-red-500' : bin.current_fill > 50 ? 'bg-yellow-500' : 'bg-green-500'}`}
                                  style={{ width: `${bin.current_fill || 0}%` }}
                                />
                              </div>
                              <span className="text-xs text-gray-400">{bin.current_fill?.toFixed(0)}%</span>
                            </div>
                          </td>
                          <td className="py-3 px-2">
                            <div className="flex items-center justify-center gap-1">
                              <button
                                onClick={() => handleToggleBinActive(bin)}
                                className={`p-1.5 rounded text-xs ${bin.is_active ? 'bg-green-900 text-green-400' : 'bg-gray-700 text-gray-400'}`}
                                title={bin.is_active ? 'Đang hoạt động' : 'Đã tắt'}
                              >
                                {bin.is_active ? '✓' : '✗'}
                              </button>
                              <button
                                onClick={() => handleEditBin(bin)}
                                className="p-1.5 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs"
                                title="Sửa"
                              >
                                ✏️
                              </button>
                              <button
                                onClick={() => handleDeleteBin(bin.id)}
                                className="p-1.5 bg-red-600 hover:bg-red-700 rounded text-white text-xs"
                                title="Xóa"
                              >
                                🗑️
                              </button>
                            </div>
                          </td>
                        </tr>
                      )})}
                    </tbody>
                  </table>
                  
                  {wasteBins.length === 0 && (
                    <div className="text-center py-8 text-gray-400">
                      Chưa có bãi rác nào. Nhấn "Thêm bãi rác" để tạo mới.
                    </div>
                  )}
                  
                  {/* Pagination */}
                  {wasteBins.length > BINS_PER_PAGE && (
                    <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-700">
                      <div className="text-sm text-gray-400">
                        Hiển thị {wasteBinsPage * BINS_PER_PAGE + 1} - {Math.min((wasteBinsPage + 1) * BINS_PER_PAGE, wasteBins.length)} / {wasteBins.length}
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setWasteBinsPage(0)}
                          disabled={wasteBinsPage === 0}
                          className="px-2 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-white"
                        >
                          ⏮️
                        </button>
                        <button
                          onClick={() => setWasteBinsPage(p => Math.max(0, p - 1))}
                          disabled={wasteBinsPage === 0}
                          className="px-3 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-white"
                        >
                          ← Trước
                        </button>
                        <span className="px-3 py-1 text-sm text-white">
                          {wasteBinsPage + 1} / {Math.ceil(wasteBins.length / BINS_PER_PAGE)}
                        </span>
                        <button
                          onClick={() => setWasteBinsPage(p => Math.min(Math.ceil(wasteBins.length / BINS_PER_PAGE) - 1, p + 1))}
                          disabled={wasteBinsPage >= Math.ceil(wasteBins.length / BINS_PER_PAGE) - 1}
                          className="px-3 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-white"
                        >
                          Sau →
                        </button>
                        <button
                          onClick={() => setWasteBinsPage(Math.ceil(wasteBins.length / BINS_PER_PAGE) - 1)}
                          disabled={wasteBinsPage >= Math.ceil(wasteBins.length / BINS_PER_PAGE) - 1}
                          className="px-2 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-white"
                        >
                          ⏭️
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {/* Bins Cards - Mobile */}
            <div className="md:hidden space-y-3">
              {wasteBinsLoading ? (
                <div className="text-center py-8">
                  <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <p className="text-gray-400">Đang tải...</p>
                </div>
              ) : wasteBins.length === 0 ? (
                <div className="text-center py-8 text-gray-400 bg-gray-800 rounded-lg">
                  Chưa có bãi rác nào. Nhấn "Thêm bãi rác" để tạo mới.
                </div>
              ) : (
                <>
                {wasteBins
                  .slice(wasteBinsPage * BINS_PER_PAGE, (wasteBinsPage + 1) * BINS_PER_PAGE)
                  .map(bin => {
                  const catStyle = {
                    organic: { bg: 'bg-green-900/30', text: 'text-green-400', border: 'border-green-700' },
                    recyclable: { bg: 'bg-blue-900/30', text: 'text-blue-400', border: 'border-blue-700' },
                    hazardous: { bg: 'bg-red-900/30', text: 'text-red-400', border: 'border-red-700' },
                    other: { bg: 'bg-gray-700/30', text: 'text-gray-400', border: 'border-gray-600' }
                  }[bin.category] || { bg: 'bg-gray-700/30', text: 'text-gray-400', border: 'border-gray-600' };
                  
                  return (
                    <div 
                      key={bin.id} 
                      className={`p-4 bg-gray-800 rounded-lg border ${catStyle.border} ${!bin.is_active ? 'opacity-50' : ''}`}
                    >
                      {/* Header */}
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-gray-500 text-xs">#{bin.id}</span>
                            <span className={`px-2 py-0.5 rounded text-xs ${catStyle.bg} ${catStyle.text}`}>
                              {categoryIcons[bin.category]} {categoryLabels[bin.category]}
                            </span>
                          </div>
                          <h4 className="text-white font-medium text-sm">{bin.name}</h4>
                        </div>
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => handleEditBin(bin)}
                            className="p-1.5 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs"
                          >
                            ✏️
                          </button>
                          <button
                            onClick={() => handleDeleteBin(bin.id)}
                            className="p-1.5 bg-red-600 hover:bg-red-700 rounded text-white text-xs"
                          >
                            🗑️
                          </button>
                        </div>
                      </div>
                      
                      {/* Info Grid */}
                      <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                        <div>
                          <span className="text-gray-500">📍 Địa chỉ:</span>
                          <p className="text-gray-300 truncate">{bin.address || '-'}</p>
                        </div>
                        <div>
                          <span className="text-gray-500">🌐 Tọa độ:</span>
                          <p className="text-gray-300 font-mono text-[10px]">
                            {bin.latitude?.toFixed(4)}, {bin.longitude?.toFixed(4)}
                          </p>
                        </div>
                        <div>
                          <span className="text-gray-500">📦 Sức chứa:</span>
                          <p className="text-white">{bin.capacity} tấn/ngày</p>
                        </div>
                        <div>
                          <span className="text-gray-500">Độ đầy:</span>
                          <div className="flex items-center gap-2 mt-0.5">
                            <div className="flex-1 h-2 bg-gray-600 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${bin.current_fill > 80 ? 'bg-red-500' : bin.current_fill > 50 ? 'bg-yellow-500' : 'bg-green-500'}`}
                                style={{ width: `${bin.current_fill || 0}%` }}
                              />
                            </div>
                            <span className="text-gray-400 w-8">{bin.current_fill?.toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Footer */}
                      <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                        <button
                          onClick={() => handleToggleBinActive(bin)}
                          className={`px-3 py-1 rounded text-xs ${bin.is_active ? 'bg-green-900 text-green-400' : 'bg-gray-700 text-gray-400'}`}
                        >
                          {bin.is_active ? '✓ Đang hoạt động' : '✗ Đã tắt'}
                        </button>
                      </div>
                    </div>
                  );
                })}
                
                {/* Mobile Pagination */}
                {wasteBins.length > BINS_PER_PAGE && (
                  <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg border border-gray-700">
                    <button
                      onClick={() => setWasteBinsPage(p => Math.max(0, p - 1))}
                      disabled={wasteBinsPage === 0}
                      className="px-3 py-2 text-sm bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 text-white"
                    >
                      ← Trước
                    </button>
                    <span className="text-sm text-gray-400">
                      {wasteBinsPage + 1} / {Math.ceil(wasteBins.length / BINS_PER_PAGE)}
                    </span>
                    <button
                      onClick={() => setWasteBinsPage(p => Math.min(Math.ceil(wasteBins.length / BINS_PER_PAGE) - 1, p + 1))}
                      disabled={wasteBinsPage >= Math.ceil(wasteBins.length / BINS_PER_PAGE) - 1}
                      className="px-3 py-2 text-sm bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 text-white"
                    >
                      Sau →
                    </button>
                  </div>
                )}
                </>
              )}
            </div>
          </div>
        )}
        
        {/* Bin Modal */}
        {showBinModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 w-full max-w-lg">
              <h3 className="text-lg font-semibold text-white mb-4">
                {editingBin ? '✏️ Sửa bãi rác' : '➕ Thêm bãi rác mới'}
              </h3>
              
              {binFormError && (
                <div className="mb-4 p-3 bg-red-900/50 border border-red-700 rounded text-red-400 text-sm">
                  {binFormError}
                </div>
              )}
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Tên bãi rác *</label>
                  <input
                    type="text"
                    value={binFormData.name}
                    onChange={(e) => setBinFormData(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                    placeholder="VD: Điểm thu gom rác Quận 1"
                  />
                </div>
                
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Loại rác</label>
                  <select
                    value={binFormData.category}
                    onChange={(e) => setBinFormData(prev => ({ ...prev, category: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                  >
                    <option value="organic">🍂 Hữu cơ</option>
                    <option value="recyclable">♻️ Tái chế</option>
                    <option value="hazardous">⚠️ Nguy hại</option>
                    <option value="other">🗑️ Khác</option>
                  </select>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Vĩ độ (Latitude) *</label>
                    <input
                      type="number"
                      step="any"
                      value={binFormData.latitude}
                      onChange={(e) => setBinFormData(prev => ({ ...prev, latitude: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                      placeholder="VD: 10.7769"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Kinh độ (Longitude) *</label>
                    <input
                      type="number"
                      step="any"
                      value={binFormData.longitude}
                      onChange={(e) => setBinFormData(prev => ({ ...prev, longitude: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                      placeholder="VD: 106.7009"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Địa chỉ</label>
                  <input
                    type="text"
                    value={binFormData.address}
                    onChange={(e) => setBinFormData(prev => ({ ...prev, address: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                    placeholder="VD: 123 Nguyễn Huệ, Quận 1, TP.HCM"
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Sức chứa (tấn/ngày)</label>
                    <input
                      type="number"
                      value={binFormData.capacity}
                      onChange={(e) => setBinFormData(prev => ({ ...prev, capacity: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Độ đầy hiện tại (%)</label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={binFormData.current_fill}
                      onChange={(e) => setBinFormData(prev => ({ ...prev, current_fill: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                    />
                  </div>
                </div>
              </div>
              
              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowBinModal(false);
                    setEditingBin(null);
                    resetBinForm();
                  }}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
                >
                  Hủy
                </button>
                <button
                  onClick={handleSaveBin}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                >
                  {editingBin ? 'Cập nhật' : 'Thêm mới'}
                </button>
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
