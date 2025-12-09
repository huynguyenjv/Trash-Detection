# 🌐 HƯỚNG DẪN TÍCH HỢP FRONTEND

API đã **SẴN SÀNG** để tích hợp vào web! Tài liệu này hướng dẫn frontend developer cách sử dụng.

---

## 🚀 QUICK START

### 1. Kiểm tra Backend
```bash
# Khởi động backend (nếu chưa chạy)
cd waste-system/backend
python main.py

# Backend sẽ chạy tại: http://localhost:8000
```

### 2. Test API
```bash
# Mở browser: http://localhost:8000/docs
# Hoặc test bằng curl:
curl http://localhost:8000/routing/health
```

### 3. API Base URL
```javascript
const API_BASE_URL = "http://localhost:8000";
```

---

## 📡 API ENDPOINTS CHO FRONTEND

### 1️⃣ **Kiểm tra Goong Maps có hoạt động không**

```javascript
// GET /routing/health
const checkRouting = async () => {
  const response = await fetch(`${API_BASE_URL}/routing/health`);
  const data = await response.json();
  
  console.log(data);
  // {
  //   "goong_enabled": true,
  //   "api_key_configured": true,
  //   "status": "ready"
  // }
  
  return data.status === "ready";
};
```

---

### 2️⃣ **Tìm thùng rác gần nhất** ⭐ QUAN TRỌNG NHẤT

```javascript
// POST /routing/nearest-bin
const findNearestBin = async (userLat, userLng, wasteCategory, vehicle = 'foot') => {
  const response = await fetch(`${API_BASE_URL}/routing/nearest-bin`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      latitude: userLat,
      longitude: userLng,
      category: wasteCategory, // "recyclable", "organic", "hazardous", "other"
      vehicle: vehicle // "foot", "bike", "car"
    })
  });
  
  const data = await response.json();
  
  return data;
  // {
  //   "method": "goong_maps",
  //   "nearest_bin": {
  //     "id": 5,
  //     "name": "Thùng rác tái chế A",
  //     "category": "recyclable",
  //     "latitude": 21.03,
  //     "longitude": 105.85,
  //     "address": "123 Đường ABC",
  //     "capacity": 75.5
  //   },
  //   "route": {
  //     "distance_km": 0.8,
  //     "distance_text": "0.8 km",
  //     "duration_minutes": 10.5,
  //     "duration_text": "11 phút",
  //     "polyline": "encoded_polyline_string",
  //     "steps": [
  //       {
  //         "instruction": "Đi về hướng đông trên Đường ABC",
  //         "distance_meters": 200,
  //         "duration_seconds": 120,
  //         "maneuver": "turn-left"
  //       }
  //     ]
  //   }
  // }
};

// USAGE:
const result = await findNearestBin(21.0285, 105.8542, "recyclable", "foot");
console.log(`Thùng gần nhất: ${result.nearest_bin.name}`);
console.log(`Khoảng cách: ${result.route.distance_km}km`);
console.log(`Thời gian: ${result.route.duration_minutes} phút`);
```

---

### 3️⃣ **Lấy route giữa 2 điểm**

```javascript
// POST /routing/route
const getRoute = async (fromLat, fromLng, toLat, toLng, vehicle = 'foot') => {
  const response = await fetch(`${API_BASE_URL}/routing/route`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      origin_lat: fromLat,
      origin_lng: fromLng,
      dest_lat: toLat,
      dest_lng: toLng,
      vehicle: vehicle
    })
  });
  
  const data = await response.json();
  return data.route;
};
```

---

### 4️⃣ **Decode polyline để vẽ trên map**

```javascript
// GET /routing/decode-polyline?encoded={polyline}
const decodePolyline = async (encodedPolyline) => {
  const response = await fetch(
    `${API_BASE_URL}/routing/decode-polyline?encoded=${encodeURIComponent(encodedPolyline)}`
  );
  
  const data = await response.json();
  
  return data.coordinates;
  // [
  //   { lat: 21.0285, lng: 105.8542 },
  //   { lat: 21.0286, lng: 105.8543 },
  //   ...
  // ]
};
```

---

### 5️⃣ **Lấy danh sách thùng rác**

```javascript
// GET /bins
const getAllBins = async () => {
  const response = await fetch(`${API_BASE_URL}/bins`);
  const bins = await response.json();
  return bins;
};

// GET /bins/category/{category}
const getBinsByCategory = async (category) => {
  const response = await fetch(`${API_BASE_URL}/bins/category/${category}`);
  const bins = await response.json();
  return bins;
};
```

---

## 🗺️ TÍCH HỢP VỚI GOONG MAP

### Setup Goong Map trong React:

```javascript
import React, { useEffect, useRef, useState } from 'react';

const MapComponent = () => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [userLocation, setUserLocation] = useState(null);
  const [nearestBin, setNearestBin] = useState(null);
  const [route, setRoute] = useState(null);

  // 1. Khởi tạo map
  useEffect(() => {
    if (!map.current) {
      map.current = new goongjs.Map({
        container: mapContainer.current,
        style: 'https://tiles.goong.io/assets/goong_map_web.json',
        center: [105.8542, 21.0285], // Hanoi
        zoom: 14,
        accessToken: 'YOUR_GOONG_MAP_TOKEN' // Get from https://account.goong.io/
      });
    }
  }, []);

  // 2. Lấy vị trí người dùng
  const getUserLocation = () => {
    navigator.geolocation.getCurrentPosition((position) => {
      const lat = position.coords.latitude;
      const lng = position.coords.longitude;
      
      setUserLocation({ lat, lng });
      
      // Center map tại vị trí user
      map.current.flyTo({ center: [lng, lat], zoom: 15 });
      
      // Add marker cho user
      new goongjs.Marker({ color: 'red' })
        .setLngLat([lng, lat])
        .setPopup(new goongjs.Popup().setHTML('<h3>Vị trí của bạn</h3>'))
        .addTo(map.current);
    });
  };

  // 3. Tìm thùng gần nhất
  const findNearestBin = async (wasteCategory) => {
    if (!userLocation) {
      alert('Vui lòng bật GPS!');
      return;
    }

    // Gọi API
    const response = await fetch('http://localhost:8000/routing/nearest-bin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: userLocation.lat,
        longitude: userLocation.lng,
        category: wasteCategory,
        vehicle: 'foot'
      })
    });

    const data = await response.json();
    
    setNearestBin(data.nearest_bin);
    setRoute(data.route);

    // Add marker cho thùng rác
    new goongjs.Marker({ color: 'green' })
      .setLngLat([data.nearest_bin.longitude, data.nearest_bin.latitude])
      .setPopup(
        new goongjs.Popup().setHTML(`
          <h3>${data.nearest_bin.name}</h3>
          <p>${data.nearest_bin.address}</p>
          <p>Khoảng cách: ${data.route.distance_km}km</p>
          <p>Thời gian: ${data.route.duration_minutes} phút</p>
        `)
      )
      .addTo(map.current);

    // Vẽ route trên map
    await drawRoute(data.route.polyline);
  };

  // 4. Vẽ route lên map
  const drawRoute = async (encodedPolyline) => {
    // Decode polyline
    const response = await fetch(
      `http://localhost:8000/routing/decode-polyline?encoded=${encodeURIComponent(encodedPolyline)}`
    );
    const { coordinates } = await response.json();

    // Remove old route
    if (map.current.getSource('route')) {
      map.current.removeLayer('route');
      map.current.removeSource('route');
    }

    // Add route source
    map.current.addSource('route', {
      type: 'geojson',
      data: {
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: coordinates.map(c => [c.lng, c.lat])
        }
      }
    });

    // Add route layer
    map.current.addLayer({
      id: 'route',
      type: 'line',
      source: 'route',
      paint: {
        'line-color': '#3b82f6',
        'line-width': 4,
        'line-opacity': 0.8
      }
    });
  };

  return (
    <div>
      <div ref={mapContainer} style={{ width: '100%', height: '500px' }} />
      
      <div style={{ marginTop: '20px' }}>
        <button onClick={getUserLocation}>
          📍 Lấy vị trí của tôi
        </button>
        
        <button onClick={() => findNearestBin('recyclable')}>
          🗑️ Tìm thùng tái chế
        </button>
        
        <button onClick={() => findNearestBin('organic')}>
          🍎 Tìm thùng hữu cơ
        </button>
      </div>

      {nearestBin && route && (
        <div style={{ marginTop: '20px', padding: '20px', background: '#f0f0f0' }}>
          <h3>Thùng rác gần nhất</h3>
          <p><strong>Tên:</strong> {nearestBin.name}</p>
          <p><strong>Địa chỉ:</strong> {nearestBin.address}</p>
          <p><strong>Khoảng cách:</strong> {route.distance_km}km</p>
          <p><strong>Thời gian:</strong> {route.duration_minutes} phút</p>
          
          <h4>Hướng dẫn đi:</h4>
          <ol>
            {route.steps.map((step, index) => (
              <li key={index}>
                {step.instruction} ({step.distance_meters}m)
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
};

export default MapComponent;
```

---

## 📦 COMPONENT EXAMPLE - React Hooks

### Custom Hook: `useNearestBin`

```javascript
import { useState } from 'react';

const useNearestBin = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const findNearestBin = async (latitude, longitude, category, vehicle = 'foot') => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/routing/nearest-bin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latitude, longitude, category, vehicle })
      });

      if (!response.ok) {
        throw new Error('Failed to find nearest bin');
      }

      const data = await response.json();
      setResult(data);
      return data;

    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { findNearestBin, loading, error, result };
};

// USAGE:
const MyComponent = () => {
  const { findNearestBin, loading, result } = useNearestBin();

  const handleSearch = async () => {
    await findNearestBin(21.0285, 105.8542, 'recyclable');
  };

  return (
    <div>
      <button onClick={handleSearch} disabled={loading}>
        {loading ? 'Đang tìm...' : 'Tìm thùng gần nhất'}
      </button>

      {result && (
        <div>
          <h3>{result.nearest_bin.name}</h3>
          <p>{result.route.distance_km}km - {result.route.duration_minutes} phút</p>
        </div>
      )}
    </div>
  );
};
```

---

## 🎨 UI/UX EXAMPLES

### 1. Bottom Sheet với thông tin thùng rác

```javascript
const BinInfoSheet = ({ bin, route }) => {
  return (
    <div className="bottom-sheet">
      <div className="bin-icon">🗑️</div>
      
      <h2>{bin.name}</h2>
      <p className="address">{bin.address}</p>
      
      <div className="route-info">
        <div className="info-item">
          <span className="icon">📏</span>
          <span className="label">Khoảng cách</span>
          <span className="value">{route.distance_km} km</span>
        </div>
        
        <div className="info-item">
          <span className="icon">⏱️</span>
          <span className="label">Thời gian</span>
          <span className="value">{route.duration_minutes} phút</span>
        </div>
        
        <div className="info-item">
          <span className="icon">🗑️</span>
          <span className="label">Dung lượng</span>
          <span className="value">{bin.capacity}%</span>
        </div>
      </div>
      
      <button className="navigate-btn">
        🧭 Bắt đầu dẫn đường
      </button>
      
      <div className="directions">
        <h3>Hướng dẫn chi tiết:</h3>
        {route.steps.map((step, i) => (
          <div key={i} className="step">
            <span className="step-number">{i + 1}</span>
            <span className="step-instruction">{step.instruction}</span>
            <span className="step-distance">{step.distance_meters}m</span>
          </div>
        ))}
      </div>
    </div>
  );
};
```

### 2. Category Selector

```javascript
const WasteCategorySelector = ({ onSelect }) => {
  const categories = [
    { id: 'recyclable', name: 'Tái chế', icon: '♻️', color: '#10b981' },
    { id: 'organic', name: 'Hữu cơ', icon: '🍎', color: '#f59e0b' },
    { id: 'hazardous', name: 'Nguy hại', icon: '☢️', color: '#ef4444' },
    { id: 'other', name: 'Khác', icon: '🗑️', color: '#6b7280' }
  ];

  return (
    <div className="category-selector">
      <h3>Bạn muốn vứt loại rác gì?</h3>
      <div className="category-grid">
        {categories.map(cat => (
          <button
            key={cat.id}
            className="category-btn"
            style={{ borderColor: cat.color }}
            onClick={() => onSelect(cat.id)}
          >
            <span className="icon">{cat.icon}</span>
            <span className="name">{cat.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
```

---

## ⚡ PERFORMANCE TIPS

### 1. Cache API responses
```javascript
const cache = new Map();

const findNearestBinCached = async (lat, lng, category) => {
  const key = `${lat},${lng},${category}`;
  
  if (cache.has(key)) {
    return cache.get(key);
  }
  
  const result = await findNearestBin(lat, lng, category);
  cache.set(key, result);
  
  return result;
};
```

### 2. Debounce user location updates
```javascript
import { debounce } from 'lodash';

const debouncedSearch = debounce((lat, lng, category) => {
  findNearestBin(lat, lng, category);
}, 500);
```

---

## 🐛 ERROR HANDLING

```javascript
const findNearestBinSafe = async (lat, lng, category) => {
  try {
    const response = await fetch('http://localhost:8000/routing/nearest-bin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latitude: lat, longitude: lng, category, vehicle: 'foot' })
    });

    if (!response.ok) {
      if (response.status === 404) {
        alert('Không tìm thấy thùng rác nào!');
      } else if (response.status === 503) {
        alert('Dịch vụ routing chưa được cấu hình');
      } else {
        alert('Có lỗi xảy ra, vui lòng thử lại');
      }
      return null;
    }

    return await response.json();

  } catch (error) {
    console.error('Error:', error);
    alert('Không thể kết nối đến server');
    return null;
  }
};
```

---

## 📱 RESPONSIVE DESIGN

```css
/* Map container */
.map-container {
  width: 100%;
  height: 60vh;
  position: relative;
}

@media (max-width: 768px) {
  .map-container {
    height: 50vh;
  }
}

/* Bottom sheet */
.bottom-sheet {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: white;
  border-radius: 20px 20px 0 0;
  padding: 20px;
  box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
  transform: translateY(100%);
  transition: transform 0.3s;
}

.bottom-sheet.open {
  transform: translateY(0);
}
```

---

## ✅ CHECKLIST TÍCH HỢP

- [ ] Cài đặt Goong Maps SDK
- [ ] Setup API base URL
- [ ] Implement getUserLocation()
- [ ] Implement findNearestBin()
- [ ] Vẽ route trên map
- [ ] Hiển thị thông tin thùng rác
- [ ] Hiển thị turn-by-turn directions
- [ ] Error handling
- [ ] Loading states
- [ ] Responsive design
- [ ] Test trên mobile

---

## 🎯 DEMO FLOW

```
1. User mở app
   ↓
2. Bấm "Lấy vị trí của tôi"
   ↓
3. Map center tại vị trí user
   ↓
4. Chọn loại rác (recyclable/organic/hazardous/other)
   ↓
5. Gọi API findNearestBin
   ↓
6. Hiển thị:
   - Marker thùng rác gần nhất
   - Route (polyline) từ user → bin
   - Bottom sheet với info
   ↓
7. User xem hướng dẫn chi tiết
   ↓
8. Bấm "Bắt đầu dẫn đường"
```

---

## 📞 SUPPORT

Nếu gặp vấn đề:
1. Kiểm tra backend có đang chạy: `http://localhost:8000/docs`
2. Kiểm tra Goong Maps status: `http://localhost:8000/routing/health`
3. Xem console log để debug
4. Kiểm tra network tab trong DevTools

---

## 🚀 READY TO INTEGRATE!

API đã sẵn sàng, chỉ cần:
1. Copy các function examples trên
2. Thêm vào React/Vue/Angular app
3. Tích hợp với Goong Map
4. Test và deploy!

**Happy coding! 🎉**
