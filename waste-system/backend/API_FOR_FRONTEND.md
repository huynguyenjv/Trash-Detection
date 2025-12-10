# 🔌 API Documentation for Frontend Integration

## Base URL

```
Development: http://localhost:8000
Production: https://your-domain.com
```

---

## 🗺️ Routing APIs

### 1. Find Nearest Waste Bin

**Endpoint:** `POST /api/routing/nearest-bin`

**Description:** Tìm thùng rác gần nhất với route thực tế

**Request:**
```typescript
interface NearestBinRequest {
  latitude: number;      // Vị trí hiện tại
  longitude: number;
  category?: 'general' | 'organic' | 'recyclable';  // Optional filter
  vehicle?: 'car' | 'bike' | 'foot';  // Phương tiện di chuyển
}
```

**Response:**
```typescript
interface NearestBinResponse {
  bin: {
    id: number;
    name: string;1
    category: string;
    location: {
      latitude: number;
      longitude: number;
    };
    capacity: number;
    fill_level: number;
  };
  route: {
    distance_meters: number;
    distance_km: number;
    duration_seconds: number;
    duration_minutes: number;
    polyline: string;  // Encoded polyline
    coordinates: [number, number][];  // [[lat, lon], ...]
    steps: Array<{
      instruction: string;
      distance: number;
    }>;
  };
  method: 'goong_maps' | 'straight_line';
}
```

**Example:**
```javascript
const response = await fetch('http://localhost:8000/api/routing/nearest-bin', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    latitude: 21.0285,
    longitude: 105.8542,
    category: 'recyclable',
    vehicle: 'foot'
  })
});

const data = await response.json();
console.log(`Nearest bin: ${data.bin.name}`);
console.log(`Distance: ${data.route.distance_km} km`);
console.log(`Duration: ${data.route.duration_minutes} minutes`);
```

---

### 2. Calculate Route

**Endpoint:** `POST /api/routing/route`

**Description:** Tính đường đi giữa 2 điểm bất kỳ

**Request:**
```typescript
interface RouteRequest {
  origin_lat: number;
  origin_lng: number;
  dest_lat: number;
  dest_lng: number;
  vehicle?: 'car' | 'bike' | 'foot';
}
```

**Response:**
```typescript
interface RouteResponse {
  method: 'goong_maps' | 'straight_line';
  route: {
    distance_meters: number;
    distance_km: number;
    duration_seconds: number;
    duration_minutes: number;
    polyline: string;
    coordinates: [number, number][];
    steps: Array<{
      instruction: string;
      distance: number;
      duration: number;
    }>;
  };
}
```

---

### 3. Optimize Collection Route

**Endpoint:** `POST /api/routing/optimize-route`

**Description:** Tối ưu lộ trình thu gom nhiều thùng rác (TSP)

**Request:**
```typescript
interface OptimizeRouteRequest {
  origin_lat: number;
  origin_lng: number;
  dest_lat: number;      // Usually same as origin (return to depot)
  dest_lng: number;
  bin_ids: number[];     // Danh sách IDs thùng rác cần thu gom
  vehicle?: 'car' | 'bike';
}
```

**Response:**
```typescript
interface OptimizeRouteResponse {
  optimized_order: number[];  // Thứ tự thăm tối ưu
  bins: Array<{
    id: number;
    name: string;
    location: { latitude: number; longitude: number };
  }>;
  total_distance_km: number;
  total_duration_minutes: number;
  routes: Array<{
    from: string;
    to: string;
    distance_km: number;
    duration_minutes: number;
    polyline: string;
    coordinates: [number, number][];
  }>;
}
```

---

### 4. Distance Matrix

**Endpoint:** `GET /api/routing/distance-matrix`

**Description:** Ma trận khoảng cách giữa nhiều điểm

**Query Parameters:**
```
origins=lat1,lng1;lat2,lng2
destinations=lat3,lng3;lat4,lng4
vehicle=car
```

**Response:**
```typescript
interface DistanceMatrixResponse {
  matrix: number[][];  // matrix[i][j] = distance from origin[i] to dest[j]
  origins: Array<{ latitude: number; longitude: number }>;
  destinations: Array<{ latitude: number; longitude: number }>;
  unit: 'km';
}
```

---

### 5. Routing Health Check

**Endpoint:** `GET /api/routing/health`

**Description:** Kiểm tra trạng thái routing service

**Response:**
```typescript
interface HealthResponse {
  goong_enabled: boolean;
  api_key_configured: boolean;
  fallback_available: boolean;
  status: 'ready' | 'fallback_mode';
}
```

---

## 🗑️ Waste Bin APIs

### 1. Get All Bins

**Endpoint:** `GET /api/bins`

**Response:**
```typescript
interface Bin {
  id: number;
  name: string;
  category: 'general' | 'organic' | 'recyclable';
  latitude: number;
  longitude: number;
  capacity: number;
  fill_level: number;
  status: 'active' | 'full' | 'maintenance';
}

type BinsResponse = Bin[];
```

---

### 2. Get Bins by Category

**Endpoint:** `GET /api/bins/category/{category}`

**Parameters:**
- `category`: 'general' | 'organic' | 'recyclable'

---

### 3. Create Bin

**Endpoint:** `POST /api/bins`

**Request:**
```typescript
interface CreateBinRequest {
  name: string;
  category: 'general' | 'organic' | 'recyclable';
  latitude: number;
  longitude: number;
  capacity: number;
}
```

---

## 📊 Statistics APIs

### 1. Get Detection Stats

**Endpoint:** `GET /api/stats/detections`

**Query Parameters:**
```
period=today|week|month
category=recyclable
```

**Response:**
```typescript
interface DetectionStats {
  total_detections: number;
  by_category: {
    recyclable: number;
    organic: number;
    general: number;
  };
  by_date: Array<{
    date: string;
    count: number;
  }>;
}
```

---

## 🎨 Frontend Integration Examples

### React Example

```typescript
import { useState, useEffect } from 'react';

function NearestBinFinder() {
  const [userLocation, setUserLocation] = useState<{lat: number, lng: number}>();
  const [nearestBin, setNearestBin] = useState<any>();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Get user location
    navigator.geolocation.getCurrentPosition((position) => {
      setUserLocation({
        lat: position.coords.latitude,
        lng: position.coords.longitude
      });
    });
  }, []);

  const findNearestBin = async () => {
    if (!userLocation) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/routing/nearest-bin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: userLocation.lat,
          longitude: userLocation.lng,
          category: 'recyclable',
          vehicle: 'foot'
        })
      });
      
      const data = await response.json();
      setNearestBin(data);
    } catch (error) {
      console.error('Error finding bin:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button onClick={findNearestBin} disabled={loading}>
        {loading ? 'Searching...' : 'Find Nearest Bin'}
      </button>
      
      {nearestBin && (
        <div>
          <h3>{nearestBin.bin.name}</h3>
          <p>Distance: {nearestBin.route.distance_km.toFixed(2)} km</p>
          <p>Duration: {nearestBin.route.duration_minutes.toFixed(1)} min</p>
          
          {/* Render map with route */}
          <MapView
            origin={userLocation}
            destination={nearestBin.bin.location}
            route={nearestBin.route.coordinates}
          />
        </div>
      )}
    </div>
  );
}
```

### Vue Example

```vue
<template>
  <div>
    <button @click="findNearestBin" :disabled="loading">
      {{ loading ? 'Searching...' : 'Find Nearest Bin' }}
    </button>
    
    <div v-if="nearestBin">
      <h3>{{ nearestBin.bin.name }}</h3>
      <p>Distance: {{ nearestBin.route.distance_km.toFixed(2) }} km</p>
      <p>Duration: {{ nearestBin.route.duration_minutes.toFixed(1) }} min</p>
      
      <MapView
        :origin="userLocation"
        :destination="nearestBin.bin.location"
        :route="nearestBin.route.coordinates"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const userLocation = ref(null);
const nearestBin = ref(null);
const loading = ref(false);

onMounted(() => {
  navigator.geolocation.getCurrentPosition((position) => {
    userLocation.value = {
      lat: position.coords.latitude,
      lng: position.coords.longitude
    };
  });
});

const findNearestBin = async () => {
  if (!userLocation.value) return;
  
  loading.value = true;
  try {
    const response = await fetch('http://localhost:8000/api/routing/nearest-bin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: userLocation.value.lat,
        longitude: userLocation.value.lng,
        category: 'recyclable',
        vehicle: 'foot'
      })
    });
    
    nearestBin.value = await response.json();
  } catch (error) {
    console.error('Error:', error);
  } finally {
    loading.value = false;
  }
};
</script>
```

---

## 🗺️ Map Integration

### Display Route on Map (Leaflet)

```javascript
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

function displayRoute(mapContainer, routeData) {
  // Initialize map
  const map = L.map(mapContainer).setView([21.0285, 105.8542], 14);
  
  // Add OpenStreetMap tiles
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
  
  // Draw route polyline
  const coordinates = routeData.route.coordinates.map(coord => [coord[0], coord[1]]);
  L.polyline(coordinates, { color: 'blue', weight: 5 }).addTo(map);
  
  // Add markers
  L.marker(coordinates[0]).bindPopup('START').addTo(map);
  L.marker(coordinates[coordinates.length - 1]).bindPopup('DESTINATION').addTo(map);
  
  // Fit bounds
  map.fitBounds(coordinates);
}
```

### Display Route on Map (Google Maps)

```javascript
function displayRoute(map, routeData) {
  const path = routeData.route.coordinates.map(coord => ({
    lat: coord[0],
    lng: coord[1]
  }));
  
  const routeLine = new google.maps.Polyline({
    path: path,
    geodesic: true,
    strokeColor: '#2196F3',
    strokeOpacity: 1.0,
    strokeWeight: 5
  });
  
  routeLine.setMap(map);
  
  // Add markers
  new google.maps.Marker({
    position: path[0],
    map: map,
    title: 'START'
  });
  
  new google.maps.Marker({
    position: path[path.length - 1],
    map: map,
    title: 'DESTINATION'
  });
  
  // Fit bounds
  const bounds = new google.maps.LatLngBounds();
  path.forEach(point => bounds.extend(point));
  map.fitBounds(bounds);
}
```

---

## ⚡ Performance Tips

1. **Cache nearest bin results** - User location không thay đổi nhiều
2. **Debounce API calls** - Tránh gọi quá nhiều khi user di chuyển
3. **Use distance matrix** - Nếu cần tính nhiều routes cùng lúc
4. **Handle offline** - Fallback to straight-line distance khi không có API

---

## 🔒 CORS Configuration

Backend đã enable CORS. Nếu gặp vấn đề, check:

```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🐛 Error Handling

```typescript
async function callAPI(endpoint: string, options: RequestInit) {
  try {
    const response = await fetch(endpoint, options);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'API Error');
    }
    
    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);
    // Show user-friendly error message
    alert('Không thể kết nối đến server. Vui lòng thử lại.');
    return null;
  }
}
```

---

## 📱 Mobile Integration (React Native)

```typescript
import { useState, useEffect } from 'react';
import * as Location from 'expo-location';
import MapView, { Polyline, Marker } from 'react-native-maps';

function NearestBinScreen() {
  const [location, setLocation] = useState(null);
  const [route, setRoute] = useState(null);

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status === 'granted') {
        const loc = await Location.getCurrentPositionAsync({});
        setLocation(loc.coords);
      }
    })();
  }, []);

  const findBin = async () => {
    const response = await fetch('YOUR_API_URL/api/routing/nearest-bin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: location.latitude,
        longitude: location.longitude,
        vehicle: 'foot'
      })
    });
    
    const data = await response.json();
    setRoute(data);
  };

  return (
    <MapView
      region={{
        latitude: location?.latitude || 21.0285,
        longitude: location?.longitude || 105.8542,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      }}
    >
      {route && (
        <>
          <Polyline
            coordinates={route.route.coordinates.map(c => ({
              latitude: c[0],
              longitude: c[1]
            }))}
            strokeColor="#2196F3"
            strokeWidth={5}
          />
          <Marker
            coordinate={{
              latitude: route.bin.location.latitude,
              longitude: route.bin.location.longitude
            }}
            title={route.bin.name}
          />
        </>
      )}
    </MapView>
  );
}
```

---

## ✅ Testing

```bash
# Test nearest bin endpoint
curl -X POST http://localhost:8000/api/routing/nearest-bin \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 21.0285,
    "longitude": 105.8542,
    "vehicle": "foot"
  }'

# Test health
curl http://localhost:8000/api/routing/health
```
