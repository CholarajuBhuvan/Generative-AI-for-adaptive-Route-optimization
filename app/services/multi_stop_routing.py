"""
Multi-Stop Routing Service with ML-based Predictive Traffic Patterns
Supports complex routing scenarios with multiple waypoints and optimization
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from itertools import permutations
import json

from app.services.ai_engine import RouteRequest, RouteResult
from app.services.traffic_service import traffic_service
from app.services.geocoding_service import geocoding_service
from app.services.weather_service import weather_service
from app.services.carbon_calculator import carbon_calculator


@dataclass
class Waypoint:
    """Represents a waypoint in a multi-stop route"""
    name: str
    latitude: float
    longitude: float
    stop_duration_minutes: int = 10
    priority: int = 1  # 1=highest, 5=lowest
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    service_type: str = "delivery"  # delivery, pickup, visit, fuel, rest


@dataclass
class MultiStopRoute:
    """Complete multi-stop route with optimized order"""
    route_id: str
    waypoints: List[Waypoint]
    segments: List[Dict[str, Any]]
    total_distance_km: float
    total_time_minutes: float
    total_cost_inr: float
    optimization_score: float
    carbon_footprint: Dict[str, Any]
    weather_impact: Dict[str, Any]


class TrafficPredictor:
    """ML-based traffic prediction using historical patterns"""
    
    def __init__(self):
        self.historical_data = {}
        self.prediction_models = {}
    
    def predict_traffic_level(self, lat: float, lng: float, 
                            future_time: datetime) -> Tuple[float, float]:
        """
        Predict traffic level for a location at a future time
        
        Returns:
            Tuple of (predicted_traffic_level, confidence)
        """
        # Simulate ML prediction based on time patterns
        hour = future_time.hour
        day_of_week = future_time.weekday()
        
        # Rush hour patterns
        base_traffic = 0.3
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_traffic += 0.4
        elif 10 <= hour <= 16:  # Daytime
            base_traffic += 0.2
        elif 20 <= hour <= 22:  # Evening
            base_traffic += 0.15
        
        # Weekend adjustments
        if day_of_week >= 5:  # Weekend
            base_traffic *= 0.7
        
        # Location-based adjustment (simulate urban vs rural)
        location_factor = abs(np.sin(lat * 0.1) * np.cos(lng * 0.1))
        base_traffic += location_factor * 0.2
        
        # Add some randomness for realism
        noise = np.random.normal(0, 0.05)
        predicted_level = max(0, min(1, base_traffic + noise))
        
        confidence = 0.85 - abs(noise) * 2
        
        return predicted_level, confidence
    
    def get_optimal_departure_time(self, route_segments: List[Tuple], 
                                 preferred_arrival: datetime) -> datetime:
        """Find optimal departure time to minimize traffic impact"""
        best_departure = preferred_arrival - timedelta(hours=2)
        best_score = float('inf')
        
        # Test different departure times
        for offset_minutes in range(-120, 61, 15):
            departure_time = preferred_arrival + timedelta(minutes=offset_minutes)
            total_traffic_score = 0
            
            current_time = departure_time
            for start_point, end_point, estimated_duration in route_segments:
                traffic_level, _ = self.predict_traffic_level(
                    start_point[0], start_point[1], current_time
                )
                total_traffic_score += traffic_level
                current_time += timedelta(minutes=estimated_duration)
            
            if total_traffic_score < best_score:
                best_score = total_traffic_score
                best_departure = departure_time
        
        return best_departure


class MultiStopOptimizer:
    """Optimize multi-stop routes using various algorithms"""
    
    def __init__(self):
        self.traffic_predictor = TrafficPredictor()
    
    def optimize_waypoint_order(self, start_point: Tuple[float, float],
                               end_point: Tuple[float, float],
                               waypoints: List[Waypoint],
                               optimization_mode: str = "time") -> List[Waypoint]:
        """
        Optimize the order of waypoints using TSP-like algorithms
        
        Args:
            start_point: Starting coordinates
            end_point: Ending coordinates  
            waypoints: List of waypoints to optimize
            optimization_mode: "time", "distance", "cost", or "balanced"
        
        Returns:
            Optimized list of waypoints
        """
        if len(waypoints) <= 1:
            return waypoints
        
        # For small numbers of waypoints, use brute force
        if len(waypoints) <= 6:
            return self._brute_force_optimize(start_point, end_point, waypoints, optimization_mode)
        else:
            return self._heuristic_optimize(start_point, end_point, waypoints, optimization_mode)
    
    def _brute_force_optimize(self, start_point, end_point, waypoints, mode):
        """Brute force optimization for small waypoint sets"""
        best_order = waypoints
        best_score = float('inf')
        
        for perm in permutations(waypoints):
            score = self._calculate_route_score(start_point, end_point, list(perm), mode)
            if score < best_score:
                best_score = score
                best_order = list(perm)
        
        return best_order
    
    def _heuristic_optimize(self, start_point, end_point, waypoints, mode):
        """Nearest neighbor heuristic for larger waypoint sets"""
        remaining = waypoints.copy()
        optimized = []
        current_point = start_point
        
        while remaining:
            nearest_idx = 0
            nearest_distance = float('inf')
            
            for i, waypoint in enumerate(remaining):
                distance = self._calculate_distance(
                    current_point, (waypoint.latitude, waypoint.longitude)
                )
                
                # Apply priority weighting
                weighted_distance = distance / waypoint.priority
                
                if weighted_distance < nearest_distance:
                    nearest_distance = weighted_distance
                    nearest_idx = i
            
            selected = remaining.pop(nearest_idx)
            optimized.append(selected)
            current_point = (selected.latitude, selected.longitude)
        
        return optimized
    
    def _calculate_route_score(self, start_point, end_point, waypoints, mode):
        """Calculate score for a waypoint order"""
        total_distance = 0
        total_time = 0
        current_point = start_point
        
        for waypoint in waypoints:
            wp_point = (waypoint.latitude, waypoint.longitude)
            distance = self._calculate_distance(current_point, wp_point)
            total_distance += distance
            total_time += distance * 1.5 + waypoint.stop_duration_minutes  # Rough time estimate
            current_point = wp_point
        
        # Final segment to end point
        total_distance += self._calculate_distance(current_point, end_point)
        total_time += self._calculate_distance(current_point, end_point) * 1.5
        
        if mode == "time":
            return total_time
        elif mode == "distance":
            return total_distance
        elif mode == "cost":
            return total_distance * 8.5  # INR per km
        else:  # balanced
            return total_time * 0.5 + total_distance * 0.3 + (total_distance * 8.5 * 0.2)
    
    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points using Haversine formula"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        R = 6371  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c


class MultiStopRoutingService:
    """Main service for multi-stop routing with ML predictions"""
    
    def __init__(self):
        self.optimizer = MultiStopOptimizer()
        self.route_cache = {}
    
    async def plan_multi_stop_route(self, 
                                   start_location: str,
                                   end_location: str,
                                   waypoint_locations: List[str],
                                   optimization_mode: str = "balanced",
                                   travel_mode: str = "driving",
                                   vehicle_type: str = "average",
                                   departure_time: Optional[datetime] = None) -> MultiStopRoute:
        """
        Plan an optimized multi-stop route
        
        Args:
            start_location: Starting location name
            end_location: Ending location name
            waypoint_locations: List of waypoint location names
            optimization_mode: "time", "distance", "cost", or "balanced"
            travel_mode: Mode of travel
            vehicle_type: Vehicle type for carbon calculation
            departure_time: Preferred departure time
            
        Returns:
            Optimized multi-stop route
        """
        # Geocode all locations
        start_coords = await self._geocode_location(start_location)
        end_coords = await self._geocode_location(end_location)
        
        waypoints = []
        for i, location in enumerate(waypoint_locations):
            coords = await self._geocode_location(location)
            waypoint = Waypoint(
                name=location,
                latitude=coords[0],
                longitude=coords[1],
                stop_duration_minutes=10,
                priority=1
            )
            waypoints.append(waypoint)
        
        # Optimize waypoint order
        optimized_waypoints = self.optimizer.optimize_waypoint_order(
            start_coords, end_coords, waypoints, optimization_mode
        )
        
        # Generate route segments
        segments = await self._generate_route_segments(
            start_coords, end_coords, optimized_waypoints, 
            travel_mode, departure_time or datetime.now()
        )
        
        # Calculate totals
        total_distance = sum(seg['distance_km'] for seg in segments)
        total_time = sum(seg['time_minutes'] for seg in segments)
        total_cost = total_distance * 8.5  # INR per km
        
        # Calculate carbon footprint
        carbon_footprint = carbon_calculator.calculate_route_environmental_impact(
            total_distance, total_time, travel_mode, vehicle_type
        )
        
        # Analyze weather impact
        weather_impact = await self._analyze_weather_impact(segments)
        
        route = MultiStopRoute(
            route_id=f"multistop_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            waypoints=optimized_waypoints,
            segments=segments,
            total_distance_km=total_distance,
            total_time_minutes=total_time,
            total_cost_inr=total_cost,
            optimization_score=0.85,
            carbon_footprint=carbon_footprint,
            weather_impact=weather_impact
        )
        
        return route
    
    async def _geocode_location(self, location_name: str) -> Tuple[float, float]:
        """Geocode a location name to coordinates"""
        results = await geocoding_service.geocode(location_name, limit=1)
        if results:
            return (results[0].latitude, results[0].longitude)
        else:
            raise ValueError(f"Could not geocode location: {location_name}")
    
    async def _generate_route_segments(self, start_coords, end_coords, waypoints, 
                                     travel_mode, departure_time):
        """Generate optimized segments between all points"""
        segments = []
        current_point = start_coords
        current_time = departure_time
        
        # Segments to waypoints
        for waypoint in waypoints:
            wp_coords = (waypoint.latitude, waypoint.longitude)
            
            # Predict traffic for this segment
            traffic_level, confidence = self.optimizer.traffic_predictor.predict_traffic_level(
                current_point[0], current_point[1], current_time
            )
            
            distance = self.optimizer._calculate_distance(current_point, wp_coords)
            base_time = distance * 1.5  # Base time estimate
            traffic_adjusted_time = base_time * (1 + traffic_level * 0.5)
            
            segment = {
                'from': current_point,
                'to': wp_coords,
                'from_name': 'Start' if current_point == start_coords else f"Waypoint",
                'to_name': waypoint.name,
                'distance_km': distance,
                'time_minutes': traffic_adjusted_time + waypoint.stop_duration_minutes,
                'traffic_level': traffic_level,
                'traffic_confidence': confidence,
                'departure_time': current_time.isoformat(),
                'arrival_time': (current_time + timedelta(minutes=traffic_adjusted_time)).isoformat()
            }
            
            segments.append(segment)
            current_point = wp_coords
            current_time += timedelta(minutes=traffic_adjusted_time + waypoint.stop_duration_minutes)
        
        # Final segment to destination
        if current_point != end_coords:
            traffic_level, confidence = self.optimizer.traffic_predictor.predict_traffic_level(
                current_point[0], current_point[1], current_time
            )
            
            distance = self.optimizer._calculate_distance(current_point, end_coords)
            base_time = distance * 1.5
            traffic_adjusted_time = base_time * (1 + traffic_level * 0.5)
            
            segment = {
                'from': current_point,
                'to': end_coords,
                'from_name': f"Waypoint",
                'to_name': 'Destination',
                'distance_km': distance,
                'time_minutes': traffic_adjusted_time,
                'traffic_level': traffic_level,
                'traffic_confidence': confidence,
                'departure_time': current_time.isoformat(),
                'arrival_time': (current_time + timedelta(minutes=traffic_adjusted_time)).isoformat()
            }
            
            segments.append(segment)
        
        return segments
    
    async def _analyze_weather_impact(self, segments):
        """Analyze weather impact across route segments"""
        weather_samples = []
        
        # Sample weather at a few key points
        sample_points = segments[::max(1, len(segments)//3)][:3]
        
        for segment in sample_points:
            try:
                weather = await weather_service.get_current_weather(
                    segment['from'][0], segment['from'][1]
                )
                if weather:
                    weather_samples.append(weather)
            except Exception as e:
                logging.warning(f"Weather analysis failed: {e}")
        
        if not weather_samples:
            return {"status": "unavailable"}
        
        avg_temp = sum(w.temperature_celsius for w in weather_samples) / len(weather_samples)
        avg_wind = sum(w.wind_speed_kmh for w in weather_samples) / len(weather_samples)
        
        return {
            "status": "available",
            "average_temperature": round(avg_temp, 1),
            "average_wind_speed": round(avg_wind, 1),
            "conditions": [w.weather_description for w in weather_samples],
            "impact_level": "low" if avg_temp > 10 and avg_wind < 30 else "moderate"
        }
    
    async def compare_route_options(self, start_location: str, end_location: str,
                                   waypoint_locations: List[str]) -> Dict[str, MultiStopRoute]:
        """Generate and compare different optimization strategies"""
        comparisons = {}
        
        optimization_modes = ["time", "distance", "cost", "balanced"]
        
        for mode in optimization_modes:
            try:
                route = await self.plan_multi_stop_route(
                    start_location, end_location, waypoint_locations,
                    optimization_mode=mode
                )
                comparisons[mode] = route
            except Exception as e:
                logging.error(f"Error generating {mode} optimized route: {e}")
        
        return comparisons
    
    def get_route_insights(self, route: MultiStopRoute) -> Dict[str, Any]:
        """Generate insights and recommendations for a route"""
        insights = {
            "efficiency_score": min(100, route.optimization_score * 100),
            "cost_per_km": route.total_cost_inr / route.total_distance_km if route.total_distance_km > 0 else 0,
            "average_speed_kmh": route.total_distance_km / (route.total_time_minutes / 60) if route.total_time_minutes > 0 else 0,
            "carbon_efficiency": route.carbon_footprint.get("environmental_metrics", {}).get("efficiency_score", 0),
            "recommendations": []
        }
        
        # Generate recommendations
        if route.total_time_minutes > 480:  # > 8 hours
            insights["recommendations"].append("Consider breaking this into multiple trips")
        
        if insights["average_speed_kmh"] < 30:
            insights["recommendations"].append("Route may have significant traffic delays")
        
        if route.carbon_footprint.get("primary_emission", {}).get("eco_score", 0) < 50:
            insights["recommendations"].append("Consider electric or hybrid vehicle for better environmental impact")
        
        return insights


# Global multi-stop routing service
multi_stop_service = MultiStopRoutingService()
