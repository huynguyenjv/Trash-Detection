import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import { findNearestBin as findNearestBinRoute, checkRoutingHealth, getAllBins, ALGORITHMS } from '../services/routingService';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom SVG icons for waste bins by category
const createBinIcon = (category) => {
  const colors = {
    organic: { bg: '#16a34a', border: '#15803d', icon: '🍂' },      // Green
    recyclable: { bg: '#2563eb', border: '#1d4ed8', icon: '♻️' },   // Blue
    hazardous: { bg: '#dc2626', border: '#b91c1c', icon: '☢️' },    // Red
    other: { bg: '#6b7280', border: '#4b5563', icon: '🗑️' },        // Gray
    general: { bg: '#6b7280', border: '#4b5563', icon: '🗑️' }       // Gray
  };
  
  const style = colors[category] || colors.other;
  
  return L.divIcon({
    className: 'custom-bin-icon',
    html: `
      <div style="
        position: relative;
        width: 36px;
        height: 42px;
        display: flex;
        flex-direction: column;
        align-items: center;
      ">
        <div style="
          width: 32px;
          height: 32px;
          background: ${style.bg};
          border: 2px solid ${style.border};
          border-radius: 50% 50% 50% 0;
          transform: rotate(-45deg);
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        ">
          <span style="
            transform: rotate(45deg);
            font-size: 14px;
            line-height: 1;
          ">${style.icon}</span>
        </div>
        <div style="
          width: 8px;
          height: 8px;
          background: ${style.bg};
          border-radius: 50%;
          margin-top: -4px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        "></div>
      </div>
    `,
    iconSize: [36, 42],
    iconAnchor: [18, 42],
    popupAnchor: [0, -42],
  });
};

// Icon for detected waste
const wasteIcon = L.divIcon({
  className: 'custom-waste-icon',
  html: `
    <div style="
      width: 28px;
      height: 28px;
      background: linear-gradient(135deg, #f97316, #ea580c);
      border: 2px solid #c2410c;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 8px rgba(249,115,22,0.5);
      animation: pulse 2s infinite;
    ">
      <span style="font-size: 14px;">⚠️</span>
    </div>
    <style>
      @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
      }
    </style>
  `,
  iconSize: [28, 28],
  iconAnchor: [14, 14],
  popupAnchor: [0, -14],
});

// Icon for current location
const currentLocationIcon = L.divIcon({
  className: 'custom-location-icon',
  html: `
    <div style="
      position: relative;
      width: 24px;
      height: 24px;
    ">
      <div style="
        position: absolute;
        width: 24px;
        height: 24px;
        background: rgba(59, 130, 246, 0.3);
        border-radius: 50%;
        animation: locationPulse 2s infinite;
      "></div>
      <div style="
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 14px;
        height: 14px;
        background: #3b82f6;
        border: 3px solid white;
        border-radius: 50%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
      "></div>
    </div>
    <style>
      @keyframes locationPulse {
        0%, 100% { transform: scale(1); opacity: 0.6; }
        50% { transform: scale(1.8); opacity: 0; }
      }
    </style>
  `,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
  popupAnchor: [0, -12],
});

// Component to handle map updates
const MapUpdater = ({ center, path }) => {
  const map = useMap();
  
  useEffect(() => {
    if (center) {
      map.setView(center, map.getZoom());
    }
  }, [center, map]);

  return null;
};

const MapView = ({ findRouteRequest = null, onRouteFound = null }) => {
  const [currentLocation, setCurrentLocation] = useState(null);
  const [wasteBins, setWasteBins] = useState([]);
  const [wasteLocations, setWasteLocations] = useState([]);
  const [selectedPath, setSelectedPath] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('weighted');
  const [routingServiceStatus, setRoutingServiceStatus] = useState(null);

  // Default center (Ho Chi Minh City) - use useMemo to avoid recreating array
  const defaultCenter = [10.8231, 106.6297];

  useEffect(() => {
    // Get user's current location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setCurrentLocation([latitude, longitude]);
        },
        (error) => {
          console.error('Error getting location:', error);
          setCurrentLocation([10.8231, 106.6297]); // Use literal instead of defaultCenter
        }
      );
    } else {
      setCurrentLocation([10.8231, 106.6297]); // Use literal instead of defaultCenter
    }

    // Check routing service health
    checkRoutingHealth().then(status => {
      setRoutingServiceStatus(status);
      console.log('Routing service status:', status);
    });

    // Load waste bins from backend API
    getAllBins()
      .then(bins => {
        if (bins && bins.length > 0) {
          // Convert backend format to frontend format
          const formattedBins = bins.map(bin => ({
            id: bin.id,
            position: [bin.latitude, bin.longitude],
            name: bin.name,
            type: bin.category?.toLowerCase() || 'general',
            address: bin.address,
            capacity: bin.capacity,
            current_fill: bin.current_fill
          }));
          setWasteBins(formattedBins);
          console.log('Loaded bins from API:', formattedBins.length);
        } else {
          // Fallback to mock data if no bins in database
          console.warn('No bins in database, using mock data');
          setWasteBins([
            { id: 1, position: [10.8231, 106.6297], name: 'Central Waste Bin', type: 'general' },
            { id: 2, position: [10.8331, 106.6397], name: 'Recycling Center', type: 'recyclable' },
            { id: 3, position: [10.8131, 106.6197], name: 'Organic Waste Bin', type: 'organic' },
            { id: 4, position: [10.8431, 106.6497], name: 'Hazardous Waste Facility', type: 'hazardous' },
          ]);
        }
      })
      .catch(err => {
        console.error('Failed to load bins from API:', err);
        // Fallback to mock data
        setWasteBins([
          { id: 1, position: [10.8231, 106.6297], name: 'Central Waste Bin', type: 'general' },
          { id: 2, position: [10.8331, 106.6397], name: 'Recycling Center', type: 'recyclable' },
          { id: 3, position: [10.8131, 106.6197], name: 'Organic Waste Bin', type: 'organic' },
          { id: 4, position: [10.8431, 106.6497], name: 'Hazardous Waste Facility', type: 'hazardous' },
        ]);
      });

    // Mock waste detection locations (these will be updated by real detections)
    setWasteLocations([
      { id: 1, position: [10.8200, 106.6250], type: 'plastic', confidence: 0.85 },
      { id: 2, position: [10.8280, 106.6350], type: 'organic', confidence: 0.92 },
      { id: 3, position: [10.8150, 106.6150], type: 'paper', confidence: 0.78 },
    ]);
  }, []); // Empty dependency - run only once on mount

  // Xử lý findRouteRequest từ App.jsx - chỉ tìm đường khi người dùng click nút
  useEffect(() => {
    if (findRouteRequest && findRouteRequest.category && currentLocation) {
      console.log('🗺️ MapView: Route request received!');
      console.log('📍 Current location:', currentLocation);
      console.log('🗑️ Requested category:', findRouteRequest.category);
      
      // Tìm đường đến thùng rác theo category
      findNearestBin(currentLocation, findRouteRequest.category);
    }
  }, [findRouteRequest, currentLocation]);

  const findNearestBin = async (wasteLocation, wasteType) => {
    setLoading(true);
    setError(null);
    
    try {
      // Use routing service to find nearest bin with route
      const result = await findNearestBinRoute({
        latitude: wasteLocation[0],
        longitude: wasteLocation[1],
        category: wasteType === 'organic' ? 'organic' : 
                  (wasteType === 'plastic' || wasteType === 'paper') ? 'recyclable' :
                  wasteType === 'hazardous' ? 'hazardous' : null,
        vehicle: 'truck',
        algorithm: selectedAlgorithm
      });

      if (result && result.nearest_bin) {
        // Build path from coordinates (already decoded by routingService)
        // routingService.findNearestBin automatically decodes polyline to coordinates
        let path;
        if (result.route?.coordinates && result.route.coordinates.length > 0) {
          // Use decoded coordinates from routing service
          path = result.route.coordinates;
        } else {
          // Fallback to straight line
          path = [
            wasteLocation,
            [result.nearest_bin.latitude, result.nearest_bin.longitude]
          ];
        }

        // Convert distance and duration to correct units
        // Backend returns distance_km and duration_minutes, frontend expects meters and seconds
        const distanceMeters = result.route?.distance_km 
          ? result.route.distance_km * 1000 
          : (result.distance_km ? result.distance_km * 1000 : 0);
        const durationSeconds = result.route?.duration_minutes 
          ? result.route.duration_minutes * 60 
          : 300;

        const pathInfo = {
          path: path,
          distance: distanceMeters,
          duration: durationSeconds,
          from: wasteLocation,
          to: [result.nearest_bin.latitude, result.nearest_bin.longitude],
          binInfo: {
            name: result.nearest_bin.name || 'Waste Bin',
            type: result.nearest_bin.category || 'general',
            position: [result.nearest_bin.latitude, result.nearest_bin.longitude],
            ...result.nearest_bin
          },
          algorithm: result.route?.algorithm_used || selectedAlgorithm,
          routeScore: result.route?.route_score,
          method: result.method
        };
        
        setSelectedPath(pathInfo);
        
        // Gọi callback khi tìm được đường
        if (onRouteFound) {
          onRouteFound(pathInfo);
        }
      } else {
        throw new Error('No route found');
      }
    } catch (err) {
      console.error('Error finding path:', err);
      setError('Failed to calculate route using routing service. Using fallback.');
      
      // Fallback: Use local calculation
      let compatibleBins = wasteBins;
      if (wasteType === 'organic') {
        compatibleBins = wasteBins.filter(bin => bin.type === 'organic' || bin.type === 'general');
      } else if (wasteType === 'plastic' || wasteType === 'paper') {
        compatibleBins = wasteBins.filter(bin => bin.type === 'recyclable' || bin.type === 'general');
      } else if (wasteType === 'hazardous') {
        compatibleBins = wasteBins.filter(bin => bin.type === 'hazardous');
      }

      if (compatibleBins.length === 0) {
        compatibleBins = wasteBins.filter(bin => bin.type === 'general');
      }

      if (compatibleBins.length > 0) {
        // Find nearest bin using Euclidean distance
        let nearestBin = null;
        let minDistance = Infinity;

        compatibleBins.forEach(bin => {
          const distance = Math.sqrt(
            Math.pow(wasteLocation[0] - bin.position[0], 2) +
            Math.pow(wasteLocation[1] - bin.position[1], 2)
          );
          if (distance < minDistance) {
            minDistance = distance;
            nearestBin = bin;
          }
        });

        if (nearestBin) {
          setSelectedPath({
            path: [wasteLocation, nearestBin.position],
            distance: minDistance * 111000, // Rough conversion to meters
            duration: 300,
            from: wasteLocation,
            to: nearestBin.position,
            binInfo: nearestBin,
            algorithm: 'fallback',
            method: 'straight_line'
          });
        }
      }
    } finally {
      setLoading(false);
    }
  };

  const clearPath = () => {
    setSelectedPath(null);
    setError(null);
  };

  const findShortestPath = () => {
    if (wasteLocations.length > 0 && currentLocation) {
      // Find path to the first detected waste location
      const firstWaste = wasteLocations[0];
      findNearestBin(firstWaste.position, firstWaste.type);
    } else if (currentLocation && wasteBins.length > 0) {
      // Find path from current location to nearest general bin
      const generalBins = wasteBins.filter(bin => bin.type === 'general');
      const targetBin = generalBins.length > 0 ? generalBins[0] : wasteBins[0];
      
      // Simulate finding path from current location
      findNearestBin(currentLocation, 'general');
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden flex flex-col h-full">
      <div className="flex justify-between items-center p-2 px-3 bg-gray-800 border-b border-gray-700 flex-shrink-0">
        <div className="flex items-center gap-2">
          <span>🗺️</span>
          <span className="text-white font-medium text-sm">Bản đồ</span>
        </div>
        <div className="flex gap-2 items-center">
          <select
            value={selectedAlgorithm}
            onChange={(e) => setSelectedAlgorithm(e.target.value)}
            className="px-2 py-1 text-xs rounded bg-gray-700 text-white border border-gray-600"
          >
            {Object.entries(ALGORITHMS).map(([key, algo]) => (
              <option key={key} value={key}>{algo.name}</option>
            ))}
          </select>
          
          <button
            onClick={findShortestPath}
            disabled={loading || (wasteLocations.length === 0 && !currentLocation)}
            className="px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed"
          >
            {loading ? '...' : '🎯 Tìm'}
          </button>
          {selectedPath && (
            <button
              onClick={clearPath}
              className="px-2 py-1 text-xs bg-gray-600 text-white rounded hover:bg-gray-500"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Status bar */}
      <div className="flex items-center gap-3 px-3 py-1 text-xs bg-gray-900/50 flex-shrink-0">
        {routingServiceStatus && (
          <div className="flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${routingServiceStatus.goong_enabled ? 'bg-green-400' : 'bg-gray-400'}`}></span>
            <span className={routingServiceStatus.goong_enabled ? 'text-green-400' : 'text-gray-400'}>
              {routingServiceStatus.goong_enabled ? 'Goong API' : 'Local'}
            </span>
          </div>
        )}
        {error && (
          <div className="flex items-center gap-1 text-amber-400">
            <span>⚠</span>
            <span className="truncate max-w-[200px]">{error}</span>
          </div>
        )}
      </div>

      <div className="flex-1 flex flex-col min-h-0 p-2">
        <div className="relative flex-1 rounded-lg overflow-hidden border border-gray-700/50">
          <MapContainer
            center={currentLocation || defaultCenter}
            zoom={15}
            className="h-full w-full"
            style={{ height: '100%', width: '100%' }}
            zoomControl={true}
          >
          {/* Google Maps Tile Layer */}
          <TileLayer
            url="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
            attribution='&copy; <a href="https://www.google.com/maps">Google Maps</a>'
          />
          
          <MapUpdater center={currentLocation} path={selectedPath} />

          {/* Current Location */}
          {currentLocation && (
            <Marker position={currentLocation} icon={currentLocationIcon}>
              <Popup>
                <div className="text-center">
                  <strong>Vị trí của bạn</strong>
                  <br />
                  <small>{currentLocation[0].toFixed(4)}, {currentLocation[1].toFixed(4)}</small>
                </div>
              </Popup>
            </Marker>
          )}

          {/* Waste Bins */}
          {wasteBins.map(bin => (
            <Marker key={bin.id} position={bin.position} icon={createBinIcon(bin.type)}>
              <Popup>
                <div className="min-w-[180px]">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">
                      {bin.type === 'organic' ? '🍂' : bin.type === 'recyclable' ? '♻️' : bin.type === 'hazardous' ? '☢️' : '🗑️'}
                    </span>
                    <strong className="text-gray-800">{bin.name}</strong>
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-0.5 rounded text-xs text-white ${
                        bin.type === 'organic' ? 'bg-green-600' :
                        bin.type === 'recyclable' ? 'bg-blue-600' :
                        bin.type === 'hazardous' ? 'bg-red-600' : 'bg-gray-600'
                      }`}>
                        {bin.type === 'organic' ? 'Hữu cơ' :
                         bin.type === 'recyclable' ? 'Tái chế' :
                         bin.type === 'hazardous' ? 'Nguy hại' : 'Khác'}
                      </span>
                    </div>
                    {bin.address && (
                      <div className="text-gray-600 text-xs">📍 {bin.address}</div>
                    )}
                    {bin.capacity && (
                      <div className="text-gray-600 text-xs">📦 Sức chứa: {bin.capacity} tấn/ngày</div>
                    )}
                    {bin.current_fill !== undefined && (
                      <div className="mt-2">
                        <div className="flex justify-between text-xs text-gray-500 mb-1">
                          <span>Độ đầy</span>
                          <span>{bin.current_fill?.toFixed(0)}%</span>
                        </div>
                        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${
                              bin.current_fill > 80 ? 'bg-red-500' : 
                              bin.current_fill > 50 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${bin.current_fill}%` }}
                          />
                        </div>
                      </div>
                    )}
                    <div className="text-gray-400 text-xs mt-1">
                      🌐 {bin.position[0].toFixed(5)}, {bin.position[1].toFixed(5)}
                    </div>
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Waste Locations */}
          {wasteLocations.map(waste => (
            <Marker key={waste.id} position={waste.position} icon={wasteIcon}>
              <Popup>
                <div>
                  <strong>Rác phát hiện</strong>
                  <br />
                  <span className="text-sm">Loại: {waste.type}</span>
                  <br />
                  <span className="text-sm">Độ tin cậy: {Math.round(waste.confidence * 100)}%</span>
                  <br />
                  <button
                    onClick={() => findNearestBin(waste.position, waste.type)}
                    className="mt-2 px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                    disabled={loading}
                  >
                    {loading ? 'Đang tìm...' : 'Tìm đường'}
                  </button>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Route Path */}
          {selectedPath && (
            <>
              {/* Route shadow for better visibility */}
              <Polyline
                positions={selectedPath.path}
                color="#1e3a5f"
                weight={8}
                opacity={0.5}
              />
              {/* Main route line */}
              <Polyline
                positions={selectedPath.path}
                color="#22d3ee"
                weight={5}
                opacity={0.9}
                dashArray="12, 8"
              />
            </>
          )}
          </MapContainer>
        </div>
      </div>

      {/* Route Information - Compact */}
      {selectedPath && (
        <div className="mx-3 mb-2 p-2 bg-gray-800/80 border border-cyan-700/30 rounded-lg flex-shrink-0">
          <div className="flex items-center justify-between gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-cyan-400">📍</span>
              <span className="text-white font-medium truncate max-w-[120px]">{selectedPath.binInfo.name}</span>
              <span className="text-gray-500">|</span>
              <span className="text-gray-400 capitalize text-xs">{selectedPath.binInfo.type}</span>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <span className="text-cyan-400 font-bold">
                  {selectedPath.distance >= 1000 
                    ? `${(selectedPath.distance / 1000).toFixed(1)}km`
                    : `${Math.round(selectedPath.distance)}m`
                  }
                </span>
              </div>
              <div className="text-center">
                <span className="text-green-400 font-bold">{Math.round(selectedPath.duration / 60)}min</span>
              </div>
              <div className="text-xs text-gray-500">
                {selectedPath.method === 'goong_maps' ? '🛣️' : '📐'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Map Legend - minimal */}
      <div className="pb-2 flex justify-center gap-4 text-xs text-gray-500 flex-shrink-0">
        <span>📍 Bạn</span>
        <span>🗑️ Thùng rác</span>
        <span>⚠️ Rác</span>
        {selectedPath && <span className="text-cyan-400">--- Tuyến đường</span>}
      </div>
    </div>
  );
};

export default MapView;
