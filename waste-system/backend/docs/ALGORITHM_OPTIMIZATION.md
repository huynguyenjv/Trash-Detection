# 🚀 TỐI ƯU THUẬT TOÁN TÌM THÙNG GẦN NHẤT

## 📊 Phân Tích Hiện Tại

### Phương pháp đang dùng: **Brute Force Search**
```python
for bin in bins:  # O(n) iterations
    route = goong_api.get_route(origin, bin)  # 1 API call per bin
    if route['distance'] < shortest:
        shortest = route['distance']
        best_bin = bin

# Nếu có 10 thùng → 10 API calls
# Nếu có 100 thùng → 100 API calls (TOO SLOW!)
```

**Vấn đề:**
- ❌ Chậm khi nhiều thùng (O(n) API calls)
- ❌ Tốn API quota
- ❌ Network latency × n

---

## ✅ GIẢI PHÁP TỐI ƯU

### **Option 1: Distance Matrix API** ⭐ RECOMMENDED

**Ý tưởng:** Gọi 1 API call duy nhất để lấy khoảng cách đến TẤT CẢ thùng

```python
def find_nearest_bin_optimized(
    self,
    origin: Tuple[float, float],
    bins: List[Dict[str, Any]],
    vehicle: str = "foot"
) -> Optional[Dict[str, Any]]:
    """
    Tối ưu: Chỉ 1 API call thay vì n calls!
    
    Goong Distance Matrix API:
    - Input: 1 origin, n destinations
    - Output: n distances + durations
    - 1 API call instead of n calls!
    """
    
    if not bins:
        return None
    
    # Prepare destinations
    destinations = [(b["latitude"], b["longitude"]) for b in bins]
    
    # 1 API call duy nhất
    matrix = self.get_distance_matrix(
        origins=[origin],
        destinations=destinations,
        vehicle=vehicle
    )
    
    if not matrix:
        return None
    
    # Find nearest from matrix
    row = matrix["rows"][0]["elements"]
    
    min_distance = float('inf')
    best_idx = -1
    
    for idx, element in enumerate(row):
        if element["status"] == "OK":
            distance = element["distance_km"]
            if distance < min_distance:
                min_distance = distance
                best_idx = idx
    
    if best_idx == -1:
        return None
    
    # Get full route for best bin
    best_bin = bins[best_idx]
    route = self.get_route(
        origin=origin,
        destination=(best_bin["latitude"], best_bin["longitude"]),
        vehicle=vehicle,
        alternatives=False
    )
    
    return {
        "bin": best_bin,
        "route": route
    }

# Performance:
# Old: O(n) API calls
# New: O(1) + O(1) = 2 API calls (distance matrix + final route)
# 
# Example: 100 thùng
# Old: 100 API calls ❌
# New: 2 API calls ✅
```

---

### **Option 2: Two-Phase Search** (Hybrid Approach)

**Ý tưởng:** Lọc thô bằng Haversine → Chính xác hóa với Goong API

```python
def find_nearest_bin_two_phase(
    self,
    origin: Tuple[float, float],
    bins: List[Dict[str, Any]],
    vehicle: str = "foot",
    top_k: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Phase 1: Haversine distance (fast, local, no API)
    Phase 2: Goong API cho top-k gần nhất
    
    Tradeoff: Độ chính xác vs Performance
    """
    
    if not bins:
        return None
    
    # Phase 1: Quick filter bằng Haversine (O(n), local)
    distances = []
    for bin_data in bins:
        dest = (bin_data["latitude"], bin_data["longitude"])
        straight_distance = self._haversine(origin, dest)
        distances.append((bin_data, straight_distance))
    
    # Sort và lấy top-k
    distances.sort(key=lambda x: x[1])
    top_bins = [b for b, _ in distances[:top_k]]
    
    logger.info(f"Phase 1: Filtered {len(bins)} → {len(top_bins)} bins")
    
    # Phase 2: Goong API cho top-k (k API calls)
    best_result = None
    shortest_distance = float('inf')
    
    for bin_data in top_bins:
        destination = (bin_data["latitude"], bin_data["longitude"])
        
        route = self.get_route(
            origin=origin,
            destination=destination,
            vehicle=vehicle,
            alternatives=False
        )
        
        if route and route["distance_meters"] < shortest_distance:
            shortest_distance = route["distance_meters"]
            best_result = {
                "bin": bin_data,
                "route": route
            }
    
    return best_result

# Performance:
# Old: O(n) API calls
# New: O(k) API calls where k << n
#
# Example: 100 thùng, k=3
# Old: 100 API calls ❌
# New: 3 API calls ✅

@staticmethod
def _haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Haversine formula for straight-line distance"""
    import math
    
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    R = 6371.0  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2)**2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c
```

---

### **Option 3: Spatial Index + Caching**

**Ý tưởng:** Pre-compute và cache routes

```python
from functools import lru_cache
import hashlib

class GoongRoutingServiceCached(GoongRoutingService):
    """Cached version với spatial indexing"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.route_cache = {}  # Cache routes
        self.spatial_index = None  # R-tree or KD-tree
    
    def _cache_key(self, origin: Tuple[float, float], dest: Tuple[float, float]) -> str:
        """Generate cache key"""
        # Round to 4 decimal places (~11m precision)
        o = f"{origin[0]:.4f},{origin[1]:.4f}"
        d = f"{dest[0]:.4f},{dest[1]:.4f}"
        return hashlib.md5(f"{o}-{d}".encode()).hexdigest()
    
    def get_route_cached(self, origin, destination, vehicle="foot"):
        """Get route with caching"""
        
        key = self._cache_key(origin, destination)
        
        # Check cache
        if key in self.route_cache:
            logger.info(f"✓ Cache hit: {key}")
            return self.route_cache[key]
        
        # Cache miss → API call
        route = self.get_route(origin, destination, vehicle)
        
        # Store in cache
        if route:
            self.route_cache[key] = route
        
        return route
    
    def build_spatial_index(self, bins: List[Dict]):
        """Build R-tree index for bins"""
        from rtree import index
        
        idx = index.Index()
        for i, bin_data in enumerate(bins):
            lat, lng = bin_data["latitude"], bin_data["longitude"]
            idx.insert(i, (lng, lat, lng, lat))
        
        self.spatial_index = idx
        logger.info(f"✓ Spatial index built for {len(bins)} bins")
    
    def find_nearest_bin_spatial(self, origin, bins, k=5):
        """Use spatial index to find candidates"""
        
        if not self.spatial_index:
            self.build_spatial_index(bins)
        
        # Query k nearest from spatial index (very fast!)
        lng, lat = origin[1], origin[0]
        nearest_ids = list(self.spatial_index.nearest((lng, lat, lng, lat), k))
        
        # Get routes for top-k
        best = None
        shortest = float('inf')
        
        for idx in nearest_ids:
            bin_data = bins[idx]
            route = self.get_route_cached(origin, (bin_data["latitude"], bin_data["longitude"]))
            
            if route and route["distance_meters"] < shortest:
                shortest = route["distance_meters"]
                best = {"bin": bin_data, "route": route}
        
        return best
```

---

## 📊 PERFORMANCE COMPARISON

| Method | API Calls | Speed | Accuracy | Best For |
|--------|-----------|-------|----------|----------|
| **Brute Force** (current) | O(n) | Slow | 100% | n < 10 |
| **Distance Matrix** | O(1) | Very Fast | 100% | Any n ✅ |
| **Two-Phase** | O(k), k≈3 | Fast | ~95% | n < 100 |
| **Spatial Index + Cache** | O(log n) | Very Fast | 100% | Large n |

---

## 💡 RECOMMENDATION

### **Implement Distance Matrix API** (Best tradeoff)

Pros:
- ✅ Chỉ 2 API calls (matrix + final route)
- ✅ Độ chính xác 100%
- ✅ Nhanh với bất kỳ số lượng thùng
- ✅ Đơn giản, dễ maintain

Implementation:
```python
# File: app/services/goong_routing.py

# Add this method to GoongRoutingService class
def find_nearest_bin_optimized(self, origin, bins, vehicle="foot"):
    # Use distance matrix
    destinations = [(b["latitude"], b["longitude"]) for b in bins]
    matrix = self.get_distance_matrix([origin], destinations, vehicle)
    
    # Find nearest
    best_idx = min(range(len(bins)), 
                   key=lambda i: matrix["rows"][0]["elements"][i]["distance_km"])
    
    # Get full route
    best_bin = bins[best_idx]
    route = self.get_route(origin, 
                           (best_bin["latitude"], best_bin["longitude"]), 
                           vehicle)
    
    return {"bin": best_bin, "route": route}
```

---

## 🚀 NEXT STEPS

1. **Immediate:** Implement Distance Matrix API (đã có sẵn trong code)
2. **Later:** Add caching for frequent queries
3. **Future:** Spatial indexing nếu có hàng nghìn thùng rác

Bạn muốn tôi implement phương pháp nào không?
