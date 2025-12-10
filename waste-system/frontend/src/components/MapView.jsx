import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import { findNearestBin, getAllBins, ALGORITHMS, checkRoutingHealth } from '../services/routingService';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom icons for different marker types
const wasteIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/32/3221/3221897.png',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

const binIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/32/484/484662.png',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

const currentLocationIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/32/684/684908.png',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
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

const MapView = ({ autoFindRoute = false, detectedWaste = null }) => {
  const [currentLocation, setCurrentLocation] = useState(null);
  const [wasteBins, setWasteBins] = useState([]);
  const [wasteLocations, setWasteLocations] = useState([]);
  const [selectedPath, setSelectedPath] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('weighted');
  const [routingServiceStatus, setRoutingServiceStatus] = useState(null);

  // Default center (Hanoi, Vietnam)
  useEffect(() => {
    // Check routing service status
    checkRoutingHealth().then(status => {
      setRoutingServiceStatus(status);
      console.log('Routing service status:', status);
    });

    // Get user's current location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setCurrentLocation([latitude, longitude]);
        },
        (error) => {
          console.error('Error getting location:', error);
          setCurrentLocation(defaultCenter);
        }
      );
    } else {
      setCurrentLocation(defaultCenter);
    }

    // Load waste bins from backend
    loadWasteBins();

    // Mock waste detection locations (for demo)
    setWasteLocations([
      { id: 1, position: [21.0200, 105.8400], type: 'plastic', confidence: 0.85 },
      { id: 2, position: [21.0350, 105.8500], type: 'organic', confidence: 0.92 },
      { id: 3, position: [21.0150, 105.8300], type: 'paper', confidence: 0.78 },
    ]);
  }, []);

  const loadWasteBins = async () => {
    try {
      const bins = await getAllBins();
      setWasteBins(bins.map(bin => ({
        id: bin.id,
        position: [bin.latitude, bin.longitude],
        name: bin.name,
        type: bin.category,
        address: bin.address,
        capacity: bin.capacity,
        fillLevel: bin.fill_level
      })));
    } catch (error) {
      console.error('Error loading bins:', error);
      // Auto-trigger pathfinding after a short delay
      setTimeout(() => {
        findNearestBinRoute(newWaste.position, newWaste.type);
      }, 500);2, position: [21.0378, 105.8345], name: 'Trung tâm tái chế Ba Đình', type: 'recyclable' },
        { id: 3, position: [21.0150, 105.8400], name: 'Thùng rác hữu cơ', type: 'organic' },
      ]);
    }
  const findNearestBinRoute = async (wasteLocation, wasteType) => {
    setLoading(true);
    setError(null);
    
    try {
      // Map waste type to category
      let category = null;
      if (wasteType === 'organic') {
        category = 'organic';
      } else if (wasteType === 'plastic' || wasteType === 'paper') {
        category = 'recyclable';
      }

      // Call routing API with custom algorithm
      const result = await findNearestBin({
        latitude: wasteLocation[0],
        longitude: wasteLocation[1],
        category: category,
        vehicle: 'foot',
        algorithm: selectedAlgorithm
      });
      if (compatibleBins.length === 0) {
        compatibleBins = wasteBins.filter(bin => bin.type === 'general');
      }

      // Find nearest bin
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

      if (result && result.route) {
        const bin = result.nearest_bin;
        const route = result.route;
        
        // Use coordinates from API (100+ points following real streets)
        const pathCoordinates = route.coordinates && route.coordinates.length > 0
          ? route.coordinates
          : [wasteLocation, [bin.latitude, bin.longitude]]; // Fallback

        setSelectedPath({
          path: pathCoordinates,
          distance: route.distance_km * 1000, // Convert to meters
          duration: route.duration_minutes * 60, // Convert to seconds
          from: wasteLocation,
          to: [bin.latitude, bin.longitude],
          binInfo: {
            id: bin.id,
            name: bin.name,
            type: bin.category,
            position: [bin.latitude, bin.longitude],
            address: bin.address
          },
          method: result.method,
          algorithm: route.algorithm_used || selectedAlgorithm,
          routeScore: route.route_score,
          totalAlternatives: route.total_alternatives,
          evaluatedBins: result.evaluated_bins
        });
      } else {
        throw new Error('No route found');
      }
    } catch (err) {
      console.error('Error finding path:', err);
      setError(`Không thể tính đường đi: ${err.message}`);
      
      // Fallback: show straight line
      if (wasteBins.length > 0) {
        const nearestBin = wasteBins[0];
        setSelectedPath({
          path: [wasteLocation, nearestBin.position],
          distance: calculateDistance(wasteLocation, nearestBin.position),
          duration: 300,
          from: wasteLocation,
          to: nearestBin.position,
          binInfo: nearestBin,
          method: 'fallback'
        });
      }   ) * 111000, // Rough conversion to meters
          duration: 300, // Mock duration
          from: wasteLocation,
          to: nearestBin.position,
          binInfo: nearestBin
        });
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
  const findShortestPath = () => {
        <div className="flex flex-wrap gap-2">
          {/* Algorithm Selector */}
          <select
            value={selectedAlgorithm}
            onChange={(e) => setSelectedAlgorithm(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            title={ALGORITHMS[selectedAlgorithm]?.description}
          >
            {Object.entries(ALGORITHMS).map(([key, algo]) => (
              <option key={key} value={key}>
                {algo.icon} {algo.name}
              </option>
            ))}
          </select>
          
          <button
            onClick={findShortestPath}
            disabled={loading || (wasteLocations.length === 0 && !currentLocation)}
            className="px-4 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            🎯 Tìm đường đi
          </button>
  };

  const calculateDistance = (pos1, pos2) => {
    // Haversine formula for straight-line distance
    const R = 6371000; // Earth radius in meters
    const dLat = (pos2[0] - pos1[0]) * Math.PI / 180;
  return (
    <div className="bg-white rounded-lg shadow-md p-4 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4 flex-shrink-0">
        <h2 className="text-xl font-semibold text-gray-800">
          Bản đồ phát hiện rác
          {routingServiceStatus && (
            <span className={`ml-2 text-xs px-2 py-1 rounded ${
              routingServiceStatus.goong_enabled 
                ? 'bg-green-100 text-green-700' 
                : 'bg-yellow-100 text-yellow-700'
            }`}>
              {routingServiceStatus.goong_enabled ? '✅ Goong Maps' : '⚠️ Fallback'}
            </span>
          )}
        </h2>
        <div className="flex flex-wrap gap-2">h.sqrt(1 - a));
    return R * c;
  };    <div className="flex flex-wrap gap-2">
          <button
            onClick={findShortestPath}
            disabled={loading || (wasteLocations.length === 0 && !currentLocation)}
            className="px-4 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            🎯 Tìm đường ngắn nhất
          </button>
          {selectedPath && (
            <button
              onClick={clearPath}
              className="px-3 py-1 text-sm bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
            >
              Clear Route
            </button>
          )}
          {loading && (
            <span className="text-sm text-blue-600 flex items-center">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Calculating route...
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-3 p-2 bg-yellow-100 border border-yellow-300 text-yellow-700 text-sm rounded flex-shrink-0">
          <div className="flex items-center">
            <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {error}
          </div>
        </div>
      )}

      <div className="flex-1 flex flex-col min-h-0">
        <div className="relative flex-1 rounded-lg overflow-hidden border">
          <MapContainer
            center={currentLocation || defaultCenter}
            zoom={13}
            className="h-full w-full"
          >
          <TileLayer
                  <button
                    onClick={() => findNearestBinRoute(waste.position, waste.type)}
                    className="mt-2 px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                    disabled={loading}
                  >
                    {loading ? 'Đang tìm...' : 'Tìm đường đi'}
                  </button>ion */}
          {currentLocation && (
            <Marker position={currentLocation} icon={currentLocationIcon}>
              <Popup>
                <div className="text-center">
                  <strong>Your Location</strong>
                  <br />
                  <small>{currentLocation[0].toFixed(4)}, {currentLocation[1].toFixed(4)}</small>
                </div>
              </Popup>
            </Marker>
          )}

          {/* Waste Bins */}
          {wasteBins.map(bin => (
            <Marker key={bin.id} position={bin.position} icon={binIcon}>
              <Popup>
                <div>
      {/* Route Information */}
      {selectedPath && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg flex-shrink-0">
          <div className="flex justify-between items-start mb-2">
            <h3 className="font-semibold text-blue-800">Thông tin đường đi</h3>
            {selectedPath.algorithm && (
              <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                {ALGORITHMS[selectedPath.algorithm]?.icon} {ALGORITHMS[selectedPath.algorithm]?.name}
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-gray-600">Điểm đến:</span>
              <p className="font-medium">{selectedPath.binInfo.name}</p>
            </div>
            <div>
              <span className="text-gray-600">Khoảng cách:</span>
              <p className="font-medium text-green-700">
                {selectedPath.distance < 1000 
                  ? `${Math.round(selectedPath.distance)}m`
                  : `${(selectedPath.distance / 1000).toFixed(2)}km`
                }
              </p>
            </div>
            <div>
              <span className="text-gray-600">Thời gian:</span>
              <p className="font-medium text-blue-700">{Math.round(selectedPath.duration / 60)} phút</p>
            </div>
            <div>
              <span className="text-gray-600">Loại:</span>
              <p className="font-medium capitalize">{selectedPath.binInfo.type}</p>
            </div>
            {selectedPath.routeScore && (
              <div>
                <span className="text-gray-600">Điểm số:</span>
                <p className="font-medium">{selectedPath.routeScore.toFixed(2)}</p>
              </div>
            )}
            {selectedPath.evaluatedBins && (
              <div>
                <span className="text-gray-600">Đã so sánh:</span>
                <p className="font-medium">{selectedPath.evaluatedBins} thùng rác</p>
              </div>
            )}
          </div>
          {selectedPath.method && (
            <div className="mt-2 pt-2 border-t border-blue-200 text-xs text-gray-600">
              Phương thức: <span className="font-medium">
                {selectedPath.method === 'goong_maps' ? '🗺️ Goong Maps API' : '📏 Đường thẳng'}
              </span>
              {selectedPath.totalAlternatives > 1 && (
                <span className="ml-2">• {selectedPath.totalAlternatives} đường đi đã xét</span>
              )}
            </div>
          )}
        </div>
      )}            className="mt-2 px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                    disabled={loading}
                  >
                    {loading ? 'Finding...' : 'Find Route'}
                  </button>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Route Path */}
          {selectedPath && (
            <Polyline
              positions={selectedPath.path}
              color="blue"
              weight={3}
              opacity={0.7}
            />
          )}
        </MapContainer>
      </div>

      {/* Route Information */}
      {selectedPath && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg flex-shrink-0">
          <h3 className="font-semibold text-blue-800 mb-2">Route Information</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Destination:</span>
              <p className="font-medium">{selectedPath.binInfo.name}</p>
            </div>
            <div>
              <span className="text-gray-600">Distance:</span>
              <p className="font-medium">{Math.round(selectedPath.distance)}m</p>
            </div>
            <div>
              <span className="text-gray-600">Est. Time:</span>
              <p className="font-medium">{Math.round(selectedPath.duration / 60)} min</p>
            </div>
            <div>
              <span className="text-gray-600">Type:</span>
              <p className="font-medium capitalize">{selectedPath.binInfo.type}</p>
            </div>
          </div>
        </div>
      )}

      {/* Map Legend */}
      <div className="mt-4 flex justify-center space-x-6 text-xs text-gray-600 flex-shrink-0">
        <div className="flex items-center">
          <span className="w-3 h-3 bg-red-500 rounded-full mr-1"></span>
          Current Location
        </div>
        <div className="flex items-center">
          <span className="w-3 h-3 bg-green-500 rounded-full mr-1"></span>
          Waste Detected
        </div>
        <div className="flex items-center">
          <span className="w-3 h-3 bg-blue-500 rounded-full mr-1"></span>
          Waste Bins
        </div>
        <div className="flex items-center">
          <span className="w-3 h-0.5 bg-blue-600 mr-1"></span>
          Route
        </div>
      </div>
      </div>
    </div>
  );
};

export default MapView;
