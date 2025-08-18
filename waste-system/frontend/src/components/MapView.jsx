import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';

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

  // Default center (Ho Chi Minh City)
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
          setCurrentLocation(defaultCenter);
        }
      );
    } else {
      setCurrentLocation(defaultCenter);
    }

    // Load waste bins data (mock data for now)
    setWasteBins([
      { id: 1, position: [10.8231, 106.6297], name: 'Central Waste Bin', type: 'general' },
      { id: 2, position: [10.8331, 106.6397], name: 'Recycling Center', type: 'recyclable' },
      { id: 3, position: [10.8131, 106.6197], name: 'Organic Waste Bin', type: 'organic' },
      { id: 4, position: [10.8431, 106.6497], name: 'Hazardous Waste Facility', type: 'hazardous' },
    ]);

    // Mock waste detection locations
    setWasteLocations([
      { id: 1, position: [10.8200, 106.6250], type: 'plastic', confidence: 0.85 },
      { id: 2, position: [10.8280, 106.6350], type: 'organic', confidence: 0.92 },
      { id: 3, position: [10.8150, 106.6150], type: 'paper', confidence: 0.78 },
    ]);
  }, [defaultCenter]);

  // Auto-find route for detected waste
  useEffect(() => {
    if (autoFindRoute && detectedWaste && currentLocation) {
      // Add detected waste to locations and automatically find route
      const newWaste = {
        id: Date.now(),
        position: currentLocation, // Use current location as waste location
        type: detectedWaste.type || 'other',
        confidence: detectedWaste.confidence || 0.8
      };
      
      setWasteLocations(prev => [...prev, newWaste]);
      
      // Auto-trigger pathfinding after a short delay
      setTimeout(() => {
        findNearestBin(newWaste.position, newWaste.type);
      }, 500);
    }
  }, [autoFindRoute, detectedWaste, currentLocation]);

  const findNearestBin = async (wasteLocation, wasteType) => {
    setLoading(true);
    setError(null);
    
    try {
      // Filter bins by waste type compatibility
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

      if (nearestBin) {
        // Get path from backend API
        const response = await fetch(
          `http://localhost:8000/path?lat=${wasteLocation[0]}&lon=${wasteLocation[1]}&dest_lat=${nearestBin.position[0]}&dest_lon=${nearestBin.position[1]}`
        );

        if (!response.ok) {
          throw new Error('Failed to get path from server');
        }

        const pathData = await response.json();
        setSelectedPath({
          path: pathData.path || [wasteLocation, nearestBin.position],
          distance: pathData.distance || minDistance,
          duration: pathData.duration || Math.round(minDistance * 1000), // Mock duration
          from: wasteLocation,
          to: nearestBin.position,
          binInfo: nearestBin
        });
      }
    } catch (err) {
      console.error('Error finding path:', err);
      setError('Failed to calculate route. Using direct line.');
      
      // Fallback: show direct line to nearest bin using general bins
      const generalBins = wasteBins.filter(bin => bin.type === 'general');
      const fallbackBins = generalBins.length > 0 ? generalBins : wasteBins;
      
      if (fallbackBins.length > 0) {
        const nearestBin = fallbackBins[0];
        setSelectedPath({
          path: [wasteLocation, nearestBin.position],
          distance: Math.sqrt(
            Math.pow(wasteLocation[0] - nearestBin.position[0], 2) +
            Math.pow(wasteLocation[1] - nearestBin.position[1], 2)
          ) * 111000, // Rough conversion to meters
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
      // Find path from current location to nearest general bin
      const generalBins = wasteBins.filter(bin => bin.type === 'general');
      const targetBin = generalBins.length > 0 ? generalBins[0] : wasteBins[0];
      
      // Simulate finding path from current location
      findNearestBin(currentLocation, 'general');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-4 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4 flex-shrink-0">
        <h2 className="text-xl font-semibold text-gray-800">Waste Detection Map</h2>
        <div className="flex flex-wrap gap-2">
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
