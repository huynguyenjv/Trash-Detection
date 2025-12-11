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
        vehicle: 'foot',
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
    <div className="bg-gray-800 rounded-xl overflow-hidden h-full flex flex-col">
      <div className="flex justify-between items-center p-3 bg-gradient-to-r from-gray-800 to-gray-700 border-b border-gray-700 flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-lg">🗺️</span>
          <span className="text-white font-medium text-sm">Waste Detection Map</span>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          {/* Algorithm Selector */}
          <select
            value={selectedAlgorithm}
            onChange={(e) => setSelectedAlgorithm(e.target.value)}
            className="px-3 py-1.5 text-sm rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
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
            className="px-4 py-1.5 bg-gradient-to-r from-green-500 to-green-600 text-white text-sm font-medium rounded-lg hover:from-green-600 hover:to-green-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all shadow-lg"
          >
            🎯 Find Route
          </button>
          {selectedPath && (
            <button
              onClick={clearPath}
              className="px-3 py-1.5 text-sm bg-gray-600 text-white rounded-lg hover:bg-gray-500 transition-colors"
            >
              ✕ Clear
            </button>
          )}
          {loading && (
            <span className="text-sm text-cyan-400 flex items-center">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-cyan-400" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Calculating...
            </span>
          )}
        </div>
      </div>

      {/* Routing Service Status */}
      {routingServiceStatus && (
        <div className={`mx-3 mt-2 px-3 py-1.5 text-xs rounded-lg flex items-center gap-2 flex-shrink-0 ${
          routingServiceStatus.goong_enabled ? 'bg-green-900/30 text-green-400 border border-green-800/50' : 'bg-yellow-900/30 text-yellow-400 border border-yellow-800/50'
        }`}>
          <span className={`w-2 h-2 rounded-full ${routingServiceStatus.goong_enabled ? 'bg-green-400' : 'bg-yellow-400'}`}></span>
          {routingServiceStatus.goong_enabled ? '✓ Goong Maps API Active' : '⚠ Fallback Mode (Straight Line)'}
        </div>
      )}

      {error && (
        <div className="mx-3 mt-2 p-2 bg-yellow-900/30 border border-yellow-800/50 text-yellow-400 text-sm rounded-lg flex-shrink-0">
          <div className="flex items-center">
            <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {error}
          </div>
        </div>
      )}

      <div className="flex-1 flex flex-col min-h-0 p-3 pt-2">
        <div className="relative flex-1 rounded-lg overflow-hidden border border-gray-700">
          <MapContainer
            center={currentLocation || defaultCenter}
            zoom={13}
            className="h-full w-full"
          >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          <MapUpdater center={currentLocation} path={selectedPath} />

          {/* Current Location */}
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
                  <strong>{bin.name}</strong>
                  <br />
                  <span className="text-sm text-gray-600">Type: {bin.type}</span>
                  <br />
                  <small>{bin.position[0].toFixed(4)}, {bin.position[1].toFixed(4)}</small>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Waste Locations */}
          {wasteLocations.map(waste => (
            <Marker key={waste.id} position={waste.position} icon={wasteIcon}>
              <Popup>
                <div>
                  <strong>Waste Detected</strong>
                  <br />
                  <span className="text-sm">Type: {waste.type}</span>
                  <br />
                  <span className="text-sm">Confidence: {Math.round(waste.confidence * 100)}%</span>
                  <br />
                  <button
                    onClick={() => findNearestBin(waste.position, waste.type)}
                    className="mt-2 px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
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
        <div className="mx-3 mb-3 p-3 bg-gradient-to-r from-cyan-900/30 to-blue-900/30 border border-cyan-800/50 rounded-lg flex-shrink-0">
          <h3 className="font-semibold text-cyan-400 mb-2 flex items-center gap-2">
            <span>🛤️</span> Route Information
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Destination:</span>
              <p className="font-medium text-white">{selectedPath.binInfo.name}</p>
            </div>
            <div>
              <span className="text-gray-400">Distance:</span>
              <p className="font-medium text-cyan-400">
                {selectedPath.distance >= 1000 
                  ? `${(selectedPath.distance / 1000).toFixed(2)} km`
                  : `${Math.round(selectedPath.distance)} m`
                }
              </p>
            </div>
            <div>
              <span className="text-gray-400">Est. Time:</span>
              <p className="font-medium text-green-400">{Math.round(selectedPath.duration / 60)} min</p>
            </div>
            <div>
              <span className="text-gray-400">Type:</span>
              <p className="font-medium text-white capitalize">{selectedPath.binInfo.type}</p>
            </div>
            <div>
              <span className="text-gray-400">Algorithm:</span>
              <p className="font-medium text-purple-400">{ALGORITHMS[selectedPath.algorithm]?.name || selectedPath.algorithm}</p>
            </div>
            <div>
              <span className="text-gray-400">Method:</span>
              <p className="font-medium text-white capitalize">{selectedPath.method?.replace('_', ' ') || 'N/A'}</p>
            </div>
          </div>
          {selectedPath.routeScore && (
            <div className="mt-2 pt-2 border-t border-cyan-800/50">
              <span className="text-gray-400 text-xs">Route Score: </span>
              <span className="font-medium text-cyan-400">{selectedPath.routeScore}</span>
            </div>
          )}
        </div>
      )}

      {/* Map Legend */}
      <div className="pb-3 flex justify-center space-x-6 text-xs text-gray-400 flex-shrink-0">
        <div className="flex items-center">
          <span className="w-2.5 h-2.5 bg-red-500 rounded-full mr-1.5"></span>
          Location
        </div>
        <div className="flex items-center">
          <span className="w-2.5 h-2.5 bg-green-500 rounded-full mr-1.5"></span>
          Waste
        </div>
        <div className="flex items-center">
          <span className="w-2.5 h-2.5 bg-blue-500 rounded-full mr-1.5"></span>
          Bins
        </div>
        <div className="flex items-center">
          <span className="w-4 h-0.5 bg-blue-500 mr-1.5"></span>
          Route
        </div>
      </div>
      </div>
    </div>
  );
};

export default MapView;
