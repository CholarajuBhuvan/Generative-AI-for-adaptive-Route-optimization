"""
Ultra-simple and fast AI engine without external API dependencies
For immediate testing and fallback scenarios
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import math

from app.models.route_models import RouteRequest, RouteResult


class SimpleFastEngine:
    """
    Ultra-simplified AI Engine with no external dependencies
    Uses only basic math for immediate response
    """
    
    def __init__(self):
        self.route_cache = {}
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the simple engine"""
        try:
            logging.info("Initializing Simple Fast AI Engine...")
            self.is_initialized = True
            logging.info("Simple Fast AI Engine initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Simple Fast AI Engine: {e}")
            raise
    
    async def optimize_route(self, request: RouteRequest) -> RouteResult:
        """
        Ultra-fast route optimization using only basic calculations
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.route_cache:
                logging.info("Using cached route result")
                return self.route_cache[cache_key]
            
            # Generate simple but realistic route
            route_coordinates = self._generate_simple_route(
                request.start_point, request.end_point
            )
            
            # Calculate basic metrics
            total_distance_km = self._calculate_total_distance(route_coordinates)
            total_time_minutes = self._estimate_travel_time(
                total_distance_km, request.travel_mode
            )
            
            # Simple cost calculation
            total_cost = self._calculate_basic_cost(total_distance_km, request.travel_mode)
            
            # Basic traffic analysis
            traffic_analysis = self._get_simple_traffic_analysis()
            
            # Adjust time based on traffic
            traffic_multiplier = 1.0 + (traffic_analysis.get('traffic_level', 0.3) * 0.2)
            total_time_minutes *= traffic_multiplier
            
            # Generate one simple alternative route
            alternatives = [self._generate_simple_alternative(request)]
            
            # Create result
            result = RouteResult(
                route_id=f"simple_route_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(100, 999)}",
                coordinates=route_coordinates,
                total_distance_km=round(total_distance_km, 2),
                total_time_minutes=round(total_time_minutes, 1),
                total_cost=round(total_cost, 2),
                confidence_score=0.80,  # Good confidence for simple routing
                ai_model_used="simple_fast_routing",
                traffic_analysis=traffic_analysis,
                alternatives=alternatives,
                metadata={
                    'generation_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'algorithm': 'simple_interpolation',
                    'cache_used': False,
                    'external_apis': 'none'
                }
            )
            
            # Cache the result
            self.route_cache[cache_key] = result
            
            # Limit cache size
            if len(self.route_cache) > 50:
                # Remove oldest entry
                oldest_key = next(iter(self.route_cache))
                del self.route_cache[oldest_key]
            
            return result
            
        except Exception as e:
            logging.error(f"Simple route optimization failed: {e}")
            # Return a very basic fallback route
            return self._generate_fallback_route(request)
    
    def _generate_cache_key(self, request: RouteRequest) -> str:
        """Generate a cache key for the request"""
        return f"{request.start_point[0]:.3f}_{request.start_point[1]:.3f}_" \
               f"{request.end_point[0]:.3f}_{request.end_point[1]:.3f}_{request.travel_mode}"
    
    def _generate_simple_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Dict[str, float]]:
        """Generate a simple but realistic-looking route"""
        
        # Calculate distance to determine number of points
        distance = self._calculate_distance(start, end)
        
        # More points for longer distances, fewer for short ones
        if distance < 1:  # Less than 1km
            num_points = 3
        elif distance < 5:  # Less than 5km
            num_points = 5
        elif distance < 20:  # Less than 20km
            num_points = 8
        else:  # Long distance
            num_points = 12
        
        coordinates = []
        
        for i in range(num_points + 1):
            t = i / num_points
            
            # Basic linear interpolation
            lat = start[0] + (end[0] - start[0]) * t
            lng = start[1] + (end[1] - start[1]) * t
            
            # Add some realistic variation to simulate road curves
            if 0 < i < num_points:
                # Create gentle S-curves that simulate real road paths
                curve_factor = np.sin(t * np.pi * 2) * 0.0005 * (1 - abs(0.5 - t) * 2)
                lat += curve_factor
                lng += curve_factor * 0.5  # Less variation in longitude
            
            # Calculate progress percentage
            progress = t * 100
            
            coordinates.append({
                'lat': round(lat, 6),
                'lng': round(lng, 6),
                'time': round(progress, 1),
                'confidence': 0.85
            })
        
        return coordinates
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points using Haversine formula"""
        lat1, lng1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lng2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        return 6371 * c
    
    def _calculate_total_distance(self, coordinates: List[Dict[str, float]]) -> float:
        """Calculate total distance of the route"""
        total_distance = 0.0
        
        for i in range(len(coordinates) - 1):
            point1 = (coordinates[i]['lat'], coordinates[i]['lng'])
            point2 = (coordinates[i + 1]['lat'], coordinates[i + 1]['lng'])
            total_distance += self._calculate_distance(point1, point2)
        
        return total_distance
    
    def _estimate_travel_time(self, distance_km: float, travel_mode: str) -> float:
        """Estimate travel time based on distance and mode"""
        # Realistic speeds based on distance for different travel modes (km/h)
        if travel_mode == 'driving':
            if distance_km < 20:
                speed = 30  # Urban city driving
            elif distance_km < 100:
                speed = 50  # Mixed roads
            else:
                speed = 60  # Highway driving
        elif travel_mode == 'walking':
            speed = 5
        elif travel_mode == 'cycling':
            speed = 15
        elif travel_mode == 'transit':
            speed = 25
        else:
            speed = 45
        
        time_hours = distance_km / speed
        
        # Add overhead for stops and delays
        if distance_km > 100:
            overhead = 1.15  # Long distance delays
        elif distance_km > 50:
            overhead = 1.10
        else:
            overhead = 1.08
        
        return time_hours * 60 * overhead  # Convert to minutes with overhead
    
    def _calculate_basic_cost(self, distance_km: float, travel_mode: str) -> float:
        """Calculate basic cost estimate"""
        cost_per_km = {
            'driving': 8.0,     # Fuel + wear (conservative)
            'walking': 0.0,     # Free
            'cycling': 0.3,     # Minimal maintenance
            'transit': 1.8      # Public transport
        }
        
        return distance_km * cost_per_km.get(travel_mode, 8.0)
    
    def _get_simple_traffic_analysis(self) -> Dict[str, Any]:
        """Get basic traffic analysis based on time of day"""
        current_hour = datetime.now().hour
        
        # Simple traffic level based on time of day (India timezone)
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            traffic_level = 0.6  # High traffic during rush hours
            congestion = 'moderate'
            speed = 30
        elif 10 <= current_hour <= 16:
            traffic_level = 0.3  # Medium traffic during day
            congestion = 'light'
            speed = 45
        elif 20 <= current_hour <= 23:
            traffic_level = 0.4  # Evening traffic
            congestion = 'light'
            speed = 40
        else:
            traffic_level = 0.1  # Low traffic at night/early morning
            congestion = 'free'
            speed = 50
        
        return {
            'traffic_level': traffic_level,
            'status': 'estimated',
            'incidents_count': 0,
            'average_speed_kmh': speed,
            'congestion_level': congestion,
            'analysis_type': 'time_based_estimation'
        }
    
    def _generate_simple_alternative(self, request: RouteRequest) -> Dict[str, Any]:
        """Generate a simple alternative route"""
        start = request.start_point
        end = request.end_point
        
        # Create alternative by going through a slightly offset midpoint
        offset = 0.004  # Small offset for alternative
        mid_lat = (start[0] + end[0]) / 2 + offset
        mid_lng = (start[1] + end[1]) / 2 - offset  # Opposite direction
        
        # Generate alternative coordinates
        alt_coordinates = [
            {'lat': start[0], 'lng': start[1], 'time': 0, 'confidence': 0.75},
            {'lat': mid_lat, 'lng': mid_lng, 'time': 50, 'confidence': 0.75},
            {'lat': end[0], 'lng': end[1], 'time': 100, 'confidence': 0.75}
        ]
        
        alt_distance = self._calculate_total_distance(alt_coordinates)
        alt_time = self._estimate_travel_time(alt_distance, request.travel_mode)
        
        return {
            'route_id': f"alt_simple_{np.random.randint(100, 999)}",
            'coordinates': alt_coordinates,
            'total_distance_km': round(alt_distance, 2),
            'total_time_minutes': round(alt_time, 1),
            'confidence_score': 0.70,
            'description': 'Simple alternative route'
        }
    
    def _generate_fallback_route(self, request: RouteRequest) -> RouteResult:
        """Generate a very basic fallback route when all else fails"""
        start = request.start_point
        end = request.end_point
        
        # Create a minimal 2-point route
        coordinates = [
            {'lat': start[0], 'lng': start[1], 'time': 0, 'confidence': 0.5},
            {'lat': end[0], 'lng': end[1], 'time': 100, 'confidence': 0.5}
        ]
        
        distance = self._calculate_distance(start, end)
        time = self._estimate_travel_time(distance, request.travel_mode)
        cost = self._calculate_basic_cost(distance, request.travel_mode)
        
        return RouteResult(
            route_id=f"fallback_simple_{np.random.randint(100, 999)}",
            coordinates=coordinates,
            total_distance_km=round(distance, 2),
            total_time_minutes=round(time, 1),
            total_cost=round(cost, 2),
            confidence_score=0.5,
            ai_model_used="fallback_simple",
            traffic_analysis={'status': 'unavailable'},
            alternatives=[],
            metadata={'algorithm': 'direct_line_fallback', 'external_apis': 'none'}
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            'initialized': self.is_initialized,
            'type': 'simple_fast_engine',
            'cache_size': len(self.route_cache),
            'external_dependencies': False,
            'models_available': {
                'simple_routing': True,
                'caching': True,
                'fallback': True
            }
        }
    
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics"""
        return {
            'engine_type': 'simple_fast_engine',
            'cache_hit_ratio': 'N/A',
            'average_generation_time_ms': '< 50ms',
            'total_routes_generated': len(self.route_cache),
            'external_api_calls': 0
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.route_cache.clear()
        logging.info("Simple Fast AI Engine cleaned up")
