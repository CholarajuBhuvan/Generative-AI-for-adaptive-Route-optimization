"""
Fast AI Engine optimized for speed
Uses simplified algorithms for quick route generation
"""

import asyncio
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass
import json
import math

from app.models.route_models import RouteRequest, RouteResult
from app.services.traffic_service import traffic_service
from app.core.config import settings


class FastAIEngine:
    """
    Simplified AI Engine optimized for speed over complexity
    Uses basic algorithms and heuristics for fast route generation
    """
    
    def __init__(self):
        self.route_cache = {}
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the fast engine"""
        try:
            logging.info("Initializing Fast AI Engine...")
            self.is_initialized = True
            logging.info("Fast AI Engine initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Fast AI Engine: {e}")
            raise
    
    async def optimize_route(self, request: RouteRequest) -> RouteResult:
        """
        Fast route optimization using simplified algorithms
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.route_cache:
                logging.info("Using cached route result")
                return self.route_cache[cache_key]
            
            # Generate route using proper road-based routing
            route_coordinates = await self._generate_road_based_route(
                request.start_point, request.end_point, request.travel_mode
            )
            
            # Calculate basic metrics
            total_distance_km = self._calculate_total_distance(route_coordinates)
            total_time_minutes = self._estimate_travel_time(
                total_distance_km, request.travel_mode
            )
            
            # Simple cost calculation
            total_cost = self._calculate_basic_cost(total_distance_km, request.travel_mode)
            
            # Get minimal traffic data (cached for speed)
            traffic_analysis = await self._get_fast_traffic_analysis(
                request.start_point, request.end_point
            )
            
            # Adjust time based on traffic
            traffic_multiplier = 1.0 + (traffic_analysis.get('traffic_level', 0.5) * 0.3)
            total_time_minutes *= traffic_multiplier
            
            # Generate one simple alternative route
            alternatives = [self._generate_simple_alternative(request)]
            
            # Create result
            result = RouteResult(
                route_id=f"fast_route_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(100, 999)}",
                coordinates=route_coordinates,
                total_distance_km=round(total_distance_km, 2),
                total_time_minutes=round(total_time_minutes, 1),
                total_cost=round(total_cost, 2),
                confidence_score=0.85,  # Good confidence for basic routing
                ai_model_used="fast_basic_routing",
                traffic_analysis=traffic_analysis,
                alternatives=alternatives,
                metadata={
                    'generation_time': (datetime.now() - start_time).total_seconds(),
                    'algorithm': 'fast_interpolation',
                    'cache_used': False
                }
            )
            
            # Cache the result
            self.route_cache[cache_key] = result
            
            # Limit cache size
            if len(self.route_cache) > 100:
                # Remove oldest entry
                oldest_key = next(iter(self.route_cache))
                del self.route_cache[oldest_key]
            
            return result
            
        except Exception as e:
            logging.error(f"Fast route optimization failed: {e}")
            # Return a very basic fallback route
            return self._generate_fallback_route(request)
    
    def _generate_cache_key(self, request: RouteRequest) -> str:
        """Generate a cache key for the request"""
        return f"{request.start_point[0]:.4f}_{request.start_point[1]:.4f}_" \
               f"{request.end_point[0]:.4f}_{request.end_point[1]:.4f}_{request.travel_mode}"
    
    async def _generate_road_based_route(self, start: Tuple[float, float], end: Tuple[float, float], travel_mode: str) -> List[Dict[str, float]]:
        """Generate a route following actual roads using OSRM routing API"""
        try:
            # Use OSRM routing service for road-based routing
            profile = self._get_osrm_profile(travel_mode)
            
            # Format coordinates for OSRM (longitude,latitude)
            start_coord = f"{start[1]},{start[0]}"
            end_coord = f"{end[1]},{end[0]}"
            
            # OSRM API call
            url = f"https://router.project-osrm.org/route/v1/{profile}/{start_coord};{end_coord}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'false'
            }
            
            response = requests.get(url, params=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('routes') and len(data['routes']) > 0:
                    route = data['routes'][0]
                    geometry = route['geometry']
                    
                    # Convert GeoJSON coordinates to our format
                    coordinates = []
                    total_points = len(geometry['coordinates'])
                    
                    for i, coord in enumerate(geometry['coordinates']):
                        lon, lat = coord[0], coord[1]
                        progress = (i / (total_points - 1)) * 100 if total_points > 1 else 0
                        
                        coordinates.append({
                            'lat': round(lat, 6),
                            'lng': round(lon, 6),
                            'time': round(progress, 1),
                            'confidence': 0.95  # High confidence for road-based routes
                        })
                    
                    logging.info(f"Generated road-based route with {len(coordinates)} points using OSRM")
                    return coordinates
            
            # If OSRM fails, log warning and fallback
            logging.warning(f"OSRM routing failed with status {response.status_code}, falling back to basic route")
            
        except Exception as e:
            logging.warning(f"Road-based routing failed: {e}, falling back to basic route")
        
        # Fallback to basic interpolated route if OSRM fails
        return self._generate_basic_route(start, end)
    
    def _get_osrm_profile(self, travel_mode: str) -> str:
        """Map travel mode to OSRM routing profile"""
        profile_map = {
            'driving': 'driving',
            'walking': 'foot',
            'cycling': 'bicycle',
            'transit': 'driving'  # Use driving for transit as approximation
        }
        return profile_map.get(travel_mode, 'driving')
    
    def _generate_basic_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Dict[str, float]]:
        """Generate a basic route using interpolation with some road-like curves"""
        
        # Number of waypoints based on distance
        distance = self._calculate_distance(start, end)
        num_points = max(5, min(15, int(distance * 20)))  # 5-15 points based on distance
        
        coordinates = []
        
        for i in range(num_points + 1):
            t = i / num_points
            
            # Linear interpolation
            lat = start[0] + (end[0] - start[0]) * t
            lng = start[1] + (end[1] - start[1]) * t
            
            # Add some realistic road-like variation (small curves)
            if i > 0 and i < num_points:
                # Add small random variation to simulate road curves
                variation = 0.001  # Small variation
                lat += np.random.uniform(-variation, variation) * (1 - abs(0.5 - t) * 2)
                lng += np.random.uniform(-variation, variation) * (1 - abs(0.5 - t) * 2)
            
            coordinates.append({
                'lat': round(lat, 6),
                'lng': round(lng, 6),
                'time': round(t * 100, 1),  # Progress percentage
                'confidence': 0.9
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
        r = 6371
        
        return r * c
    
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
        # Realistic average speeds for different travel modes (km/h)
        if travel_mode == 'driving':
            # Adjust driving speed based on distance
            if distance_km < 20:
                speed = 30  # Urban city driving
            elif distance_km < 100:
                speed = 50  # Mixed urban-highway
            else:
                speed = 60  # Long distance highway
        elif travel_mode == 'walking':
            speed = 5
        elif travel_mode == 'cycling':
            speed = 15
        elif travel_mode == 'transit':
            speed = 25
        else:
            speed = 45
        
        time_hours = distance_km / speed
        
        # Add realistic overhead for stops, traffic, etc.
        if distance_km > 100:
            overhead = 1.15  # 15% extra for long trips
        elif distance_km > 50:
            overhead = 1.12  # 12% extra
        else:
            overhead = 1.08  # 8% extra for city traffic
        
        return time_hours * 60 * overhead  # Convert to minutes with overhead
    
    def _calculate_basic_cost(self, distance_km: float, travel_mode: str) -> float:
        """Calculate basic cost estimate"""
        cost_per_km = {
            'driving': 8.5,     # Fuel + wear
            'walking': 0.0,     # Free
            'cycling': 0.5,     # Minimal maintenance
            'transit': 2.0      # Public transport
        }
        
        return distance_km * cost_per_km.get(travel_mode, 8.5)
    
    async def _get_fast_traffic_analysis(self, start: Tuple[float, float], end: Tuple[float, float]) -> Dict[str, Any]:
        """Get basic traffic analysis without heavy API calls"""
        try:
            # Use cached or simplified traffic data
            current_hour = datetime.now().hour
            
            # Simple traffic level based on time of day
            if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
                traffic_level = 0.7  # High traffic during rush hours
            elif 10 <= current_hour <= 16:
                traffic_level = 0.4  # Medium traffic during day
            else:
                traffic_level = 0.2  # Low traffic at night/early morning
            
            return {
                'traffic_level': traffic_level,
                'status': 'estimated',
                'incidents_count': 0,
                'average_speed_kmh': 40 * (1 - traffic_level * 0.5),
                'congestion_level': 'light' if traffic_level < 0.3 else 'moderate' if traffic_level < 0.6 else 'heavy'
            }
            
        except Exception as e:
            logging.warning(f"Fast traffic analysis failed: {e}")
            return {
                'traffic_level': 0.3,
                'status': 'unavailable',
                'incidents_count': 0,
                'average_speed_kmh': 40,
                'congestion_level': 'unknown'
            }
    
    def _generate_simple_alternative(self, request: RouteRequest) -> Dict[str, Any]:
        """Generate a simple alternative route using nearby waypoint"""
        try:
            start = request.start_point
            end = request.end_point
            
            # Calculate a waypoint slightly offset from the direct route midpoint
            mid_lat = (start[0] + end[0]) / 2 + 0.008  # Larger offset for more realistic detour
            mid_lng = (start[1] + end[1]) / 2 + 0.008
            
            # Try to get a route through the waypoint using OSRM
            profile = self._get_osrm_profile(request.travel_mode)
            
            # Format coordinates: start -> waypoint -> end
            waypoint_route = f"{start[1]},{start[0]};{mid_lng},{mid_lat};{end[1]},{end[0]}"
            url = f"https://router.project-osrm.org/route/v1/{profile}/{waypoint_route}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'false'
            }
            
            response = requests.get(url, params=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('routes') and len(data['routes']) > 0:
                    route = data['routes'][0]
                    geometry = route['geometry']
                    
                    # Convert to our format
                    alt_coordinates = []
                    total_points = len(geometry['coordinates'])
                    
                    for i, coord in enumerate(geometry['coordinates']):
                        lon, lat = coord[0], coord[1]
                        progress = (i / (total_points - 1)) * 100 if total_points > 1 else 0
                        
                        alt_coordinates.append({
                            'lat': round(lat, 6),
                            'lng': round(lon, 6),
                            'time': round(progress, 1),
                            'confidence': 0.8
                        })
                else:
                    # Fallback to basic alternative
                    alt_coordinates = self._generate_basic_alternative(start, end)
            else:
                # Fallback to basic alternative
                alt_coordinates = self._generate_basic_alternative(start, end)
                
        except Exception:
            # Fallback to basic alternative
            alt_coordinates = self._generate_basic_alternative(request.start_point, request.end_point)
        
        alt_distance = self._calculate_total_distance(alt_coordinates)
        alt_time = self._estimate_travel_time(alt_distance, request.travel_mode)
        
        return {
            'route_id': f"alt_{np.random.randint(100, 999)}",
            'coordinates': alt_coordinates,
            'total_distance_km': round(alt_distance, 2),
            'total_time_minutes': round(alt_time, 1),
            'confidence_score': 0.75,
            'description': 'Alternative route with slight detour'
        }
    
    def _generate_basic_alternative(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Dict[str, float]]:
        """Generate a basic alternative route using interpolation"""
        # Create a simple detour by going through an offset midpoint
        mid_lat = (start[0] + end[0]) / 2 + 0.005  # Small offset
        mid_lng = (start[1] + end[1]) / 2 + 0.005
        
        coordinates = []
        
        # Start to midpoint (3 points)
        for i in range(3):
            t = i / 2
            lat = start[0] + (mid_lat - start[0]) * t
            lng = start[1] + (mid_lng - start[1]) * t
            coordinates.append({
                'lat': round(lat, 6), 
                'lng': round(lng, 6), 
                'time': t * 50, 
                'confidence': 0.7
            })
        
        # Midpoint to end (2 more points)
        for i in range(1, 3):
            t = i / 2
            lat = mid_lat + (end[0] - mid_lat) * t
            lng = mid_lng + (end[1] - mid_lng) * t
            coordinates.append({
                'lat': round(lat, 6), 
                'lng': round(lng, 6), 
                'time': 50 + t * 50, 
                'confidence': 0.7
            })
        
        return coordinates
    
    def _generate_fallback_route(self, request: RouteRequest) -> RouteResult:
        """Generate a very basic fallback route when all else fails"""
        start = request.start_point
        end = request.end_point
        
        # Just create a direct line with 3 points
        coordinates = [
            {'lat': start[0], 'lng': start[1], 'time': 0, 'confidence': 0.6},
            {'lat': (start[0] + end[0])/2, 'lng': (start[1] + end[1])/2, 'time': 50, 'confidence': 0.6},
            {'lat': end[0], 'lng': end[1], 'time': 100, 'confidence': 0.6}
        ]
        
        distance = self._calculate_distance(start, end)
        time = self._estimate_travel_time(distance, request.travel_mode)
        cost = self._calculate_basic_cost(distance, request.travel_mode)
        
        return RouteResult(
            route_id=f"fallback_{np.random.randint(100, 999)}",
            coordinates=coordinates,
            total_distance_km=round(distance, 2),
            total_time_minutes=round(time, 1),
            total_cost=round(cost, 2),
            confidence_score=0.6,
            ai_model_used="fallback_direct",
            traffic_analysis={'status': 'unavailable'},
            alternatives=[],
            metadata={'algorithm': 'direct_line_fallback'}
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            'initialized': self.is_initialized,
            'type': 'fast_ai_engine',
            'cache_size': len(self.route_cache),
            'models_available': {
                'fast_routing': True,
                'caching': True,
                'fallback': True
            }
        }
    
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics"""
        return {
            'engine_type': 'fast_ai_engine',
            'cache_hit_ratio': 'N/A',
            'average_generation_time_ms': '< 100ms',
            'total_routes_generated': len(self.route_cache)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.route_cache.clear()
        logging.info("Fast AI Engine cleaned up")
