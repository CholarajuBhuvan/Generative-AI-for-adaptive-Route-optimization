# API Reference

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
Currently, the API does not require authentication. In production, implement proper authentication mechanisms.

## Endpoints

### Route Optimization

#### POST /optimize-route
Optimize a route using AI models.

**Request Body:**
```json
{
  "start_lat": 40.7128,
  "start_lng": -74.0060,
  "end_lat": 40.7589,
  "end_lng": -73.9851,
  "constraints": {
    "max_time": 60,
    "max_distance": 50,
    "avoid_tolls": false,
    "avoid_highways": false
  },
  "user_preferences": {
    "time_weight": 0.4,
    "distance_weight": 0.3,
    "cost_weight": 0.2,
    "traffic_weight": 0.1
  },
  "travel_mode": "driving",
  "departure_time": "2024-01-15T10:00:00Z",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "route_id": "route_20240115_100000_1234",
  "coordinates": [
    {
      "lat": 40.7128,
      "lng": -74.0060,
      "time": 0,
      "confidence": 0.95
    }
  ],
  "total_distance_km": 15.2,
  "total_time_minutes": 25,
  "total_cost": 7.60,
  "confidence_score": 0.87,
  "ai_model_used": "transformer",
  "traffic_analysis": {
    "overall_metrics": {
      "average_traffic_level": 0.45,
      "congested_segments": 2
    }
  },
  "alternatives": [],
  "metadata": {
    "generation_time": "2024-01-15T10:00:00Z",
    "travel_mode": "driving"
  }
}
```

### Traffic Data

#### GET /traffic-data
Get real-time traffic data for a location.

**Query Parameters:**
- `lat` (float): Latitude
- `lng` (float): Longitude  
- `radius_km` (float, optional): Search radius in kilometers (default: 5.0)

**Response:**
```json
{
  "timestamp": "2024-01-15T10:00:00Z",
  "location": [40.7128, -74.0060],
  "traffic_metrics": {
    "level": 0.45,
    "average_speed_kmh": 35.2,
    "average_density": 28.5,
    "average_occupancy": 0.42,
    "total_volume": 1200,
    "confidence": 0.8
  },
  "incidents": {
    "count": 2,
    "impact_score": 0.3,
    "details": []
  },
  "predictions": {
    "available": true,
    "horizons": {
      "15_min": {
        "predicted_speed_kmh": 32.1,
        "traffic_level": 0.52
      }
    }
  }
}
```

### User Feedback

#### POST /feedback
Submit user feedback for route optimization learning.

**Request Body:**
```json
{
  "route_id": "route_20240115_100000_1234",
  "user_id": "user123",
  "rating": 4,
  "feedback_type": "route_quality",
  "comments": "Great route, avoided heavy traffic"
}
```

#### POST /route-performance
Submit route performance data for model improvement.

**Request Body:**
```json
{
  "route_id": "route_20240115_100000_1234",
  "predicted_time": 25.0,
  "actual_time": 28.5,
  "predicted_distance": 15.2,
  "actual_distance": 15.8,
  "traffic_accuracy": 0.85,
  "user_satisfaction": 0.8
}
```

### Analytics

#### GET /analytics
Get comprehensive system analytics.

**Response:**
```json
{
  "ai_engine": {
    "total_requests": 1250,
    "model_performance": {
      "transformer": {
        "count": 500,
        "avg_confidence": 0.85
      }
    }
  },
  "learning_engine": {
    "learning_status": {
      "enabled": true,
      "users_tracked": 45
    }
  },
  "traffic_service": {
    "cache_size": 150,
    "providers_count": 3
  }
}
```

#### GET /user-preferences/{user_id}
Get learned preferences for a specific user.

#### POST /user-preferences/{user_id}/reset
Reset learning data for a user.

### Traffic Predictions

#### GET /traffic-prediction
Get traffic prediction for a location.

**Query Parameters:**
- `lat` (float): Latitude
- `lng` (float): Longitude
- `time_horizon_minutes` (int, optional): Prediction horizon in minutes (default: 60)

### System Status

#### GET /models/status
Get status of all AI models.

#### GET /health
Detailed health check for all system components.

## WebSocket Endpoint

### WS /ws/live-updates
Real-time updates for route optimization.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live-updates');
```

**Subscribe to updates:**
```json
{
  "type": "subscribe",
  "topic": "route_updates"
}
```

**Receive updates:**
```json
{
  "type": "route_update",
  "route_id": "route_20240115_100000_1234",
  "model_used": "transformer",
  "confidence": 0.87,
  "timestamp": "2024-01-15T10:00:00Z"
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error description",
  "timestamp": "2024-01-15T10:00:00Z",
  "path": "/api/v1/optimize-route"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Rate Limiting

Currently, no rate limiting is implemented. In production, implement appropriate rate limiting based on your requirements.

## Examples

### Python Example
```python
import requests

# Optimize a route
response = requests.post('http://localhost:8000/api/v1/optimize-route', json={
    'start_lat': 40.7128,
    'start_lng': -74.0060,
    'end_lat': 40.7589,
    'end_lng': -73.9851,
    'travel_mode': 'driving'
})

route = response.json()
print(f"Route distance: {route['total_distance_km']} km")
print(f"Route time: {route['total_time_minutes']} minutes")
```

### JavaScript Example
```javascript
// Optimize a route
const response = await fetch('/api/v1/optimize-route', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    start_lat: 40.7128,
    start_lng: -74.0060,
    end_lat: 40.7589,
    end_lng: -73.9851,
    travel_mode: 'driving'
  })
});

const route = await response.json();
console.log(`Route distance: ${route.total_distance_km} km`);
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/optimize-route" \
  -H "Content-Type: application/json" \
  -d '{
    "start_lat": 40.7128,
    "start_lng": -74.0060,
    "end_lat": 40.7589,
    "end_lng": -73.9851,
    "travel_mode": "driving"
  }'
```
