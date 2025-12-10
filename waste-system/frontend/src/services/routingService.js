/**
 * Routing API Service
 * Tích hợp với backend routing API (Goong Maps + Custom Algorithms)
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Get route between two points
 * @param {Object} params - Route parameters
 * @param {number} params.origin_lat - Origin latitude
 * @param {number} params.origin_lng - Origin longitude
 * @param {number} params.dest_lat - Destination latitude
 * @param {number} params.dest_lng - Destination longitude
 * @param {string} params.vehicle - Vehicle type: 'car', 'bike', 'foot'
 * @param {string} params.algorithm - Algorithm: 'weighted', 'dijkstra', 'astar', 'multi_criteria', 'greedy'
 * @returns {Promise<Object>} Route data with coordinates, distance, duration
 */
export async function getRoute({
  origin_lat,
  origin_lng,
  dest_lat,
  dest_lng,
  vehicle = 'foot',
  algorithm = 'weighted'
}) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/routing/route`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        origin_lat,
        origin_lng,
        dest_lat,
        dest_lng,
        vehicle,
        algorithm
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get route');
    }

    const data = await response.json();
    
    // Decode polyline to coordinates if available
    if (data.route?.polyline && data.method === 'goong_maps') {
      const coords = await decodePolyline(data.route.polyline);
      data.route.coordinates = coords;
    }
    
    return data;
  } catch (error) {
    console.error('Error getting route:', error);
    throw error;
  }
}

/**
 * Find nearest waste bin with route
 * @param {Object} params - Search parameters
 * @param {number} params.latitude - Current latitude
 * @param {number} params.longitude - Current longitude
 * @param {string} params.category - Waste category: 'general', 'organic', 'recyclable'
 * @param {string} params.vehicle - Vehicle type: 'car', 'bike', 'foot'
 * @param {string} params.algorithm - Algorithm: 'weighted', 'dijkstra', 'astar', 'multi_criteria', 'greedy'
 * @returns {Promise<Object>} Nearest bin with route information
 */
export async function findNearestBin({
  latitude,
  longitude,
  category = null,
  vehicle = 'foot',
  algorithm = 'weighted'
}) {
  try {
    const body = {
      latitude,
      longitude,
      vehicle,
      algorithm
    };
    
    if (category) {
      body.category = category;
    }

    const response = await fetch(`${API_BASE_URL}/api/routing/nearest-bin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to find nearest bin');
    }

    const data = await response.json();
    
    // Decode polyline if available
    if (data.route?.polyline && data.method === 'goong_maps') {
      const coords = await decodePolyline(data.route.polyline);
      data.route.coordinates = coords;
    }
    
    return data;
  } catch (error) {
    console.error('Error finding nearest bin:', error);
    throw error;
  }
}

/**
 * Optimize collection route for multiple bins
 * @param {Object} params - Optimization parameters
 * @param {number} params.origin_lat - Start latitude
 * @param {number} params.origin_lng - Start longitude
 * @param {number} params.dest_lat - End latitude
 * @param {number} params.dest_lng - End longitude
 * @param {number[]} params.bin_ids - Array of bin IDs to visit
 * @param {string} params.vehicle - Vehicle type
 * @returns {Promise<Object>} Optimized route
 */
export async function optimizeRoute({
  origin_lat,
  origin_lng,
  dest_lat,
  dest_lng,
  bin_ids,
  vehicle = 'car'
}) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/routing/optimize-route`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        origin_lat,
        origin_lng,
        dest_lat,
        dest_lng,
        bin_ids,
        vehicle
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to optimize route');
    }

    const data = await response.json();
    
    // Decode polyline
    if (data.optimization?.polyline) {
      const coords = await decodePolyline(data.optimization.polyline);
      data.optimization.coordinates = coords;
    }
    
    return data;
  } catch (error) {
    console.error('Error optimizing route:', error);
    throw error;
  }
}

/**
 * Decode polyline string to coordinates
 * @param {string} encoded - Encoded polyline string
 * @returns {Promise<Array>} Array of [lat, lng] coordinates
 */
export async function decodePolyline(encoded) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/routing/decode-polyline?encoded=${encodeURIComponent(encoded)}`
    );

    if (!response.ok) {
      throw new Error('Failed to decode polyline');
    }

    const data = await response.json();
    return data.coordinates.map(coord => [coord.lat, coord.lng]);
  } catch (error) {
    console.error('Error decoding polyline:', error);
    // Fallback: return empty array
    return [];
  }
}

/**
 * Get all waste bins
 * @returns {Promise<Array>} Array of bin data
 */
export async function getAllBins() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/bins`);

    if (!response.ok) {
      throw new Error('Failed to get bins');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting bins:', error);
    throw error;
  }
}

/**
 * Check routing service health
 * @returns {Promise<Object>} Service status
 */
export async function checkRoutingHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/routing/health`);

    if (!response.ok) {
      throw new Error('Routing service unavailable');
    }

    return await response.json();
  } catch (error) {
    console.error('Error checking routing health:', error);
    return {
      goong_enabled: false,
      status: 'offline'
    };
  }
}

/**
 * Algorithm descriptions for UI
 */
export const ALGORITHMS = {
  weighted: {
    name: 'Weighted Score',
    description: 'Cân bằng khoảng cách (70%) và thời gian (30%)',
    icon: '⚖️',
    useCase: 'Tốt nhất cho xe thu gom rác'
  },
  dijkstra: {
    name: 'Dijkstra',
    description: 'Ưu tiên đường ngắn nhất',
    icon: '📏',
    useCase: 'Tiết kiệm nhiên liệu tối đa'
  },
  astar: {
    name: 'A* Search',
    description: 'Kết hợp khoảng cách và heuristic',
    icon: '🎯',
    useCase: 'Cân bằng tốc độ và hiệu quả'
  },
  multi_criteria: {
    name: 'Multi-Criteria',
    description: 'Xét nhiều yếu tố: khoảng cách, thời gian, giao thông, nhiên liệu',
    icon: '🔬',
    useCase: 'Tối ưu toàn diện'
  },
  greedy: {
    name: 'Greedy (Nhanh)',
    description: 'Chọn nhanh đường gần nhất',
    icon: '⚡',
    useCase: 'Quyết định nhanh, real-time'
  }
};

export default {
  getRoute,
  findNearestBin,
  optimizeRoute,
  decodePolyline,
  getAllBins,
  checkRoutingHealth,
  ALGORITHMS
};
