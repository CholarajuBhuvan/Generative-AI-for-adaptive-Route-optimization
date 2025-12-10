"""
Simplified AI Models for Route Optimization (Demo Version)
These models work without heavy dependencies like PyTorch
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json


class SimpleRouteGenerator:
    """Simple route generator for demonstration purposes"""
    
    def __init__(self):
        self.name = "Simple Route Generator"
    
    def generate_route(self, 
                      start_point: Tuple[float, float],
                      end_point: Tuple[float, float],
                      constraints: Dict,
                      num_waypoints: int = 10) -> List[Dict]:
        """Generate a simple route with waypoints"""
        
        # Calculate distance
        distance = self._calculate_distance(start_point, end_point)
        
        # Generate waypoints
        coordinates = []
        for i in range(num_waypoints + 2):  # Include start and end
            t = i / (num_waypoints + 1)
            
            # Add some randomness for realistic routing
            lat_offset = random.uniform(-0.002, 0.002)
            lon_offset = random.uniform(-0.002, 0.002)
            
            lat = start_point[0] + t * (end_point[0] - start_point[0]) + lat_offset
            lng = start_point[1] + t * (end_point[1] - start_point[1]) + lon_offset
            
            coordinates.append({
                'lat': lat,
                'lng': lng,
                'time': i * (distance / num_waypoints * 0.5),  # Estimate time
                'confidence': 0.8 + random.uniform(-0.1, 0.1)
            })
        
        return coordinates
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Simple distance calculation
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # Rough km conversion


class SimpleTrafficPredictor:
    """Simple traffic predictor"""
    
    def __init__(self):
        self.name = "Simple Traffic Predictor"
    
    def predict_traffic(self, location: Tuple[float, float], time: datetime) -> Dict:
        """Predict traffic conditions"""
        
        # Simple traffic simulation based on time and location
        hour = time.hour
        
        # Rush hour simulation
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_traffic = 0.7
        elif 12 <= hour <= 14:
            base_traffic = 0.5
        else:
            base_traffic = 0.3
        
        # Add location-based variation
        lat, lng = location
        location_factor = 0.2 * np.sin(lat * 10) * np.cos(lng * 10)
        
        traffic_level = max(0.1, min(0.9, base_traffic + location_factor))
        
        return {
            'level': traffic_level,
            'speed_kmh': 50 - (traffic_level * 30),
            'confidence': 0.8
        }


class SimpleRouteOptimizer:
    """Simple route optimizer that combines multiple strategies"""
    
    def __init__(self):
        self.route_generator = SimpleRouteGenerator()
        self.traffic_predictor = SimpleTrafficPredictor()
        self.name = "Simple Route Optimizer"
    
    def optimize_route(self, 
                      start_point: Tuple[float, float],
                      end_point: Tuple[float, float],
                      constraints: Dict,
                      user_preferences: Dict) -> Dict:
        """Optimize route using simple algorithms"""
        
        # Generate base route
        base_route = self.route_generator.generate_route(
            start_point, end_point, constraints
        )
        
        # Calculate metrics
        total_distance = self._calculate_route_distance(base_route)
        total_time = self._estimate_route_time(base_route, start_point, end_point)
        
        # Apply user preferences
        optimized_route = self._apply_preferences(base_route, user_preferences)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(start_point, end_point, constraints)
        
        return {
            'route': {
                'coordinates': optimized_route,
                'total_distance_km': total_distance,
                'total_time_minutes': total_time,
                'total_cost': total_distance * 0.5,  # Simple cost calculation
                'confidence_score': 0.8,
                'model_used': self.name
            },
            'alternatives': alternatives,
            'traffic_analysis': {
                'overall_traffic_level': 0.4,
                'congested_segments': 0,
                'recommendations': ['Route appears clear']
            }
        }
    
    def _calculate_route_distance(self, route: List[Dict]) -> float:
        """Calculate total route distance"""
        total_distance = 0.0
        for i in range(len(route) - 1):
            p1 = (route[i]['lat'], route[i]['lng'])
            p2 = (route[i + 1]['lat'], route[i + 1]['lng'])
            total_distance += self.route_generator._calculate_distance(p1, p2)
        return total_distance
    
    def _estimate_route_time(self, route: List[Dict], start: Tuple, end: Tuple) -> float:
        """Estimate route time based on traffic"""
        distance = self.route_generator._calculate_distance(start, end)
        
        # Get traffic prediction for midpoint
        mid_lat = (start[0] + end[0]) / 2
        mid_lng = (start[1] + end[1]) / 2
        traffic = self.traffic_predictor.predict_traffic((mid_lat, mid_lng), datetime.now())
        
        # Estimate time based on distance and traffic
        base_speed = 50  # km/h
        adjusted_speed = base_speed * (1 - traffic['level'] * 0.5)
        time_hours = distance / adjusted_speed
        
        return time_hours * 60  # Convert to minutes
    
    def _apply_preferences(self, route: List[Dict], preferences: Dict) -> List[Dict]:
        """Apply user preferences to route"""
        # For simplicity, just return the route as-is
        # In a real implementation, this would modify the route based on preferences
        return route
    
    def _generate_alternatives(self, start: Tuple, end: Tuple, constraints: Dict) -> List[Dict]:
        """Generate alternative routes"""
        alternatives = []
        
        # Generate 2-3 alternative routes with slight variations
        for i in range(3):
            alt_route = self.route_generator.generate_route(
                start, end, constraints, num_waypoints=8 + i
            )
            
            alt_distance = self._calculate_route_distance(alt_route)
            alt_time = self._estimate_route_time(alt_route, start, end)
            
            alternatives.append({
                'coordinates': alt_route[:5],  # Limit for response size
                'total_distance_km': alt_distance,
                'total_time_minutes': alt_time,
                'confidence_score': 0.7 + i * 0.05,
                'model_used': f"Alternative {i + 1}"
            })
        
        return alternatives


# Global instances
simple_route_optimizer = SimpleRouteOptimizer()
simple_traffic_predictor = SimpleTrafficPredictor()
