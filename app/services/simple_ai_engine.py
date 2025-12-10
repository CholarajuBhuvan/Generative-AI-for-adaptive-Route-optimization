"""
Simplified AI Engine for Route Optimization (Demo Version)
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from app.models.simple_models import simple_route_optimizer, simple_traffic_predictor


@dataclass
class RouteRequest:
    """Request for route optimization"""
    start_point: Tuple[float, float]  # lat, lng
    end_point: Tuple[float, float]    # lat, lng
    constraints: Dict[str, Any]       # max_time, max_distance, preferences, etc.
    user_preferences: Dict[str, float] # weight preferences for different objectives
    travel_mode: str = "driving"      # driving, walking, cycling, transit
    departure_time: Optional[datetime] = None


@dataclass
class RouteResult:
    """Result of route optimization"""
    route_id: str
    coordinates: List[Dict[str, float]]  # List of {lat, lng, time, confidence}
    total_distance_km: float
    total_time_minutes: float
    total_cost: float
    confidence_score: float
    ai_model_used: str
    traffic_analysis: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SimpleAIEngine:
    """
    Simplified AI Engine for demonstration purposes
    """
    
    def __init__(self):
        self.is_initialized = False
        self.request_history = []
    
    async def initialize(self):
        """Initialize the AI engine"""
        try:
            logging.info("Initializing Simple AI Engine...")
            self.is_initialized = True
            logging.info("Simple AI Engine initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Simple AI Engine: {e}")
            raise
    
    async def optimize_route(self, request: RouteRequest) -> RouteResult:
        """
        Optimize route using simplified AI models
        
        Args:
            request: Route optimization request
            
        Returns:
            Optimized route result
        """
        if not self.is_initialized:
            raise RuntimeError("AI Engine not initialized")
        
        # Get traffic data
        traffic_data = await self._get_simple_traffic_data(request)
        
        # Generate route using simple optimizer
        route_data = await self._generate_route_with_simple_model(request, traffic_data)
        
        # Analyze traffic conditions
        traffic_analysis = await self._analyze_route_traffic(route_data['coordinates'])
        
        # Generate alternative routes
        alternatives = await self._generate_alternative_routes(request, traffic_data)
        
        # Create result
        result = RouteResult(
            route_id=f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
            coordinates=route_data['coordinates'],
            total_distance_km=route_data['total_distance_km'],
            total_time_minutes=route_data['total_time_minutes'],
            total_cost=route_data.get('total_cost', 0.0),
            confidence_score=route_data['confidence_score'],
            ai_model_used=route_data['model_used'],
            traffic_analysis=traffic_analysis,
            alternatives=alternatives,
            metadata={
                'generation_time': datetime.now().isoformat(),
                'request_constraints': request.constraints,
                'user_preferences': request.user_preferences,
                'travel_mode': request.travel_mode
            }
        )
        
        # Store in history for learning
        self._store_route_result(result)
        
        return result
    
    async def _get_simple_traffic_data(self, request: RouteRequest) -> Dict[str, Any]:
        """Get simplified traffic data"""
        # Get traffic prediction for start, end, and midpoint
        start_traffic = simple_traffic_predictor.predict_traffic(
            request.start_point, datetime.now()
        )
        end_traffic = simple_traffic_predictor.predict_traffic(
            request.end_point, datetime.now()
        )
        
        mid_lat = (request.start_point[0] + request.end_point[0]) / 2
        mid_lng = (request.start_point[1] + request.end_point[1]) / 2
        mid_traffic = simple_traffic_predictor.predict_traffic(
            (mid_lat, mid_lng), datetime.now()
        )
        
        return {
            'start_traffic': start_traffic,
            'end_traffic': end_traffic,
            'mid_traffic': mid_traffic,
            'overall_traffic_level': np.mean([
                start_traffic['level'],
                end_traffic['level'],
                mid_traffic['level']
            ])
        }
    
    async def _generate_route_with_simple_model(self, 
                                             request: RouteRequest,
                                             traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate route using simple model"""
        try:
            # Use simple route optimizer
            result = simple_route_optimizer.optimize_route(
                request.start_point,
                request.end_point,
                request.constraints,
                request.user_preferences
            )
            
            return {
                'coordinates': result['route']['coordinates'],
                'total_distance_km': result['route']['total_distance_km'],
                'total_time_minutes': result['route']['total_time_minutes'],
                'total_cost': result['route']['total_cost'],
                'confidence_score': result['route']['confidence_score'],
                'model_used': result['route']['model_used']
            }
                
        except Exception as e:
            logging.error(f"Error generating route with simple model: {e}")
            # Return fallback route
            return self._generate_fallback_route(request)
    
    def _generate_fallback_route(self, request: RouteRequest) -> Dict[str, Any]:
        """Generate a simple fallback route when models fail"""
        # Create a simple straight-line route with interpolated waypoints
        num_waypoints = 8
        coordinates = []
        
        for i in range(num_waypoints + 2):
            t = i / (num_waypoints + 1)
            lat = request.start_point[0] + t * (request.end_point[0] - request.start_point[0])
            lng = request.start_point[1] + t * (request.end_point[1] - request.start_point[1])
            
            coordinates.append({
                'lat': lat,
                'lng': lng,
                'time': i * 3,  # 3 minutes per waypoint
                'confidence': 0.6
            })
        
        total_distance = self._calculate_distance(request.start_point, request.end_point)
        
        return {
            'coordinates': coordinates,
            'total_distance_km': total_distance,
            'total_time_minutes': num_waypoints * 3,
            'total_cost': total_distance * 0.5,
            'confidence_score': 0.6,
            'model_used': 'fallback'
        }
    
    async def _analyze_route_traffic(self, coordinates: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze traffic conditions along the route"""
        try:
            # Simple traffic analysis
            traffic_levels = []
            for coord in coordinates[::2]:  # Sample every other coordinate
                traffic = simple_traffic_predictor.predict_traffic(
                    (coord['lat'], coord['lng']), datetime.now()
                )
                traffic_levels.append(traffic['level'])
            
            avg_traffic = np.mean(traffic_levels) if traffic_levels else 0.5
            
            return {
                'segments': [],
                'overall_metrics': {
                    'average_traffic_level': avg_traffic,
                    'max_traffic_level': max(traffic_levels) if traffic_levels else 0.5,
                    'min_traffic_level': min(traffic_levels) if traffic_levels else 0.5,
                    'traffic_variability': np.std(traffic_levels) if traffic_levels else 0.1,
                    'congested_segments': sum(1 for level in traffic_levels if level > 0.7)
                },
                'bottlenecks': [],
                'recommendations': [
                    'Route traffic conditions appear normal' if avg_traffic < 0.6 
                    else 'Expect moderate traffic delays'
                ]
            }
        except Exception as e:
            logging.error(f"Error analyzing route traffic: {e}")
            return {
                'error': str(e),
                'segments': [],
                'overall_metrics': {},
                'bottlenecks': [],
                'recommendations': ['Unable to analyze traffic conditions']
            }
    
    async def _generate_alternative_routes(self, 
                                         request: RouteRequest,
                                         traffic_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative routes"""
        alternatives = []
        
        try:
            # Generate alternatives with different parameters
            for i in range(2):
                alt_constraints = request.constraints.copy()
                alt_constraints['variation'] = i
                
                route_data = simple_route_optimizer.optimize_route(
                    request.start_point,
                    request.end_point,
                    alt_constraints,
                    request.user_preferences
                )
                
                alternative = {
                    'model_used': f"Alternative {i + 1}",
                    'coordinates': route_data['route']['coordinates'][:5],  # Limit for response size
                    'total_distance_km': route_data['route']['total_distance_km'],
                    'total_time_minutes': route_data['route']['total_time_minutes'],
                    'confidence_score': route_data['route']['confidence_score']
                }
                
                alternatives.append(alternative)
                
        except Exception as e:
            logging.error(f"Error generating alternatives: {e}")
        
        return alternatives
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points in kilometers"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        return c * r
    
    def _store_route_result(self, result: RouteResult):
        """Store route result for analytics"""
        self.request_history.append({
            'timestamp': datetime.now(),
            'route_id': result.route_id,
            'model_used': result.ai_model_used,
            'performance': {
                'confidence': result.confidence_score,
                'distance': result.total_distance_km,
                'time': result.total_time_minutes
            }
        })
        
        # Keep only recent history
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics for AI models"""
        if not self.request_history:
            return {'message': 'No data available'}
        
        # Calculate model performance metrics
        model_stats = {}
        
        for entry in self.request_history:
            model = entry['model_used']
            if model not in model_stats:
                model_stats[model] = {
                    'count': 0,
                    'total_confidence': 0.0,
                    'total_distance': 0.0,
                    'total_time': 0.0
                }
            
            stats = model_stats[model]
            stats['count'] += 1
            stats['total_confidence'] += entry['performance']['confidence']
            stats['total_distance'] += entry['performance']['distance']
            stats['total_time'] += entry['performance']['time']
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats['count'] > 0:
                stats['avg_confidence'] = stats['total_confidence'] / stats['count']
                stats['avg_distance'] = stats['total_distance'] / stats['count']
                stats['avg_time'] = stats['total_time'] / stats['count']
        
        return {
            'total_requests': len(self.request_history),
            'model_performance': model_stats,
            'recent_requests': len([r for r in self.request_history if 
                                  (datetime.now() - r['timestamp']).total_seconds() < 3600])
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get AI Engine status"""
        return {
            'initialized': self.is_initialized,
            'models_available': {
                'simple_optimizer': True,
                'traffic_predictor': True
            },
            'request_history_size': len(self.request_history),
            'engine_type': 'Simple AI Engine (Demo)'
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logging.info("Cleaning up Simple AI Engine...")
        self.request_history.clear()
        logging.info("Simple AI Engine cleanup completed")


# Global AI Engine instance
ai_engine = SimpleAIEngine()
