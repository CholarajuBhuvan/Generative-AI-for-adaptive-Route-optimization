"""
Real-time Traffic Data Integration Service
Integrates with multiple traffic data sources for comprehensive traffic monitoring
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import logging

from app.core.config import settings


@dataclass
class TrafficIncident:
    """Represents a traffic incident"""
    id: str
    location: Tuple[float, float]  # lat, lng
    type: str  # accident, construction, congestion, etc.
    severity: int  # 1-5 scale
    description: str
    start_time: datetime
    estimated_duration: timedelta
    affected_lanes: List[int]
    impact_radius: float  # km


@dataclass
class TrafficFlow:
    """Represents traffic flow data for a road segment"""
    segment_id: str
    location: Tuple[float, float]
    speed_kmh: float
    density: float  # vehicles per km
    occupancy: float  # percentage
    volume: int  # vehicles per hour
    timestamp: datetime
    confidence: float  # data quality confidence


@dataclass
class TrafficPrediction:
    """Traffic prediction for future time periods"""
    location: Tuple[float, float]
    time_horizon: timedelta
    predicted_speed: float
    predicted_density: float
    confidence: float
    factors: Dict[str, float]  # contributing factors


class TrafficDataProvider:
    """Base class for traffic data providers"""
    
    async def get_traffic_data(self, 
                              location: Tuple[float, float],
                              radius: float = 5.0) -> List[TrafficFlow]:
        """Get traffic data for a location"""
        raise NotImplementedError
    
    async def get_traffic_incidents(self, 
                                   location: Tuple[float, float],
                                   radius: float = 10.0) -> List[TrafficIncident]:
        """Get traffic incidents for a location"""
        raise NotImplementedError
    
    async def get_traffic_predictions(self, 
                                     location: Tuple[float, float],
                                     time_horizon: timedelta) -> List[TrafficPrediction]:
        """Get traffic predictions for a location"""
        raise NotImplementedError


class GoogleTrafficProvider(TrafficDataProvider):
    """Google Maps Traffic API provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_traffic_data(self, 
                              location: Tuple[float, float],
                              radius: float = 5.0) -> List[TrafficFlow]:
        """Get traffic data from Google Maps API"""
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")
        
        lat, lng = location
        url = f"{self.base_url}/roads/nearest"
        
        params = {
            'key': self.api_key,
            'points': f"{lat},{lng}",
            'radius': radius * 1000  # Convert to meters
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_google_traffic_data(data, location)
                else:
                    logging.error(f"Google Traffic API error: {response.status}")
                    return []
        except Exception as e:
            logging.error(f"Error fetching Google traffic data: {e}")
            return []
    
    async def get_traffic_incidents(self, 
                                   location: Tuple[float, float],
                                   radius: float = 10.0) -> List[TrafficIncident]:
        """Get traffic incidents from Google Maps API"""
        # Google Maps doesn't provide direct incident API
        # This would need to be implemented with other services
        return []
    
    async def get_traffic_predictions(self, 
                                     location: Tuple[float, float],
                                     time_horizon: timedelta) -> List[TrafficPrediction]:
        """Get traffic predictions (not directly available from Google)"""
        # This would require integration with prediction services
        return []
    
    def _parse_google_traffic_data(self, data: Dict, location: Tuple[float, float]) -> List[TrafficFlow]:
        """Parse Google Maps traffic data"""
        flows = []
        
        if 'snappedPoints' in data:
            for point in data['snappedPoints']:
                if 'location' in point:
                    lat = point['location']['latitude']
                    lng = point['location']['longitude']
                    
                    # Generate synthetic traffic data based on location
                    # In production, this would come from actual Google traffic data
                    flow = TrafficFlow(
                        segment_id=point.get('originalIndex', 'unknown'),
                        location=(lat, lng),
                        speed_kmh=self._generate_synthetic_speed(lat, lng),
                        density=self._generate_synthetic_density(lat, lng),
                        occupancy=self._generate_synthetic_occupancy(lat, lng),
                        volume=self._generate_synthetic_volume(lat, lng),
                        timestamp=datetime.now(),
                        confidence=0.8
                    )
                    flows.append(flow)
        
        return flows
    
    def _generate_synthetic_speed(self, lat: float, lng: float) -> float:
        """Generate synthetic speed data based on location"""
        # Simulate urban vs highway speed differences
        base_speed = 50.0
        variation = 20.0 * np.sin(lat * 10) * np.cos(lng * 10)
        return max(10, min(120, base_speed + variation))
    
    def _generate_synthetic_density(self, lat: float, lng: float) -> float:
        """Generate synthetic density data"""
        base_density = 30.0
        variation = 20.0 * np.sin(lat * 5) * np.cos(lng * 5)
        return max(5, min(100, base_density + variation))
    
    def _generate_synthetic_occupancy(self, lat: float, lng: float) -> float:
        """Generate synthetic occupancy data"""
        return min(1.0, max(0.0, 0.3 + 0.4 * np.sin(lat * 8) * np.cos(lng * 8)))
    
    def _generate_synthetic_volume(self, lat: float, lng: float) -> int:
        """Generate synthetic volume data"""
        base_volume = 1000
        variation = 500 * np.sin(lat * 6) * np.cos(lng * 6)
        return int(max(100, min(3000, base_volume + variation)))


class OpenStreetMapTrafficProvider(TrafficDataProvider):
    """OpenStreetMap-based traffic data provider"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_traffic_data(self, 
                              location: Tuple[float, float],
                              radius: float = 5.0) -> List[TrafficFlow]:
        """Get traffic data from OpenStreetMap"""
        # OSM doesn't provide real-time traffic data
        # This would need integration with OSM-based traffic services
        return []
    
    async def get_traffic_incidents(self, 
                                   location: Tuple[float, float],
                                   radius: float = 10.0) -> List[TrafficIncident]:
        """Get traffic incidents from OSM-based sources"""
        return []
    
    async def get_traffic_predictions(self, 
                                     location: Tuple[float, float],
                                     time_horizon: timedelta) -> List[TrafficPrediction]:
        """Get traffic predictions"""
        return []


class SyntheticTrafficProvider(TrafficDataProvider):
    """Synthetic traffic data provider for testing and demonstration"""
    
    async def __aenter__(self):
        """Enter async context manager"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager"""
        pass
    
    async def get_traffic_data(self, 
                              location: Tuple[float, float],
                              radius: float = 5.0) -> List[TrafficFlow]:
        """Generate synthetic traffic data"""
        flows = []
        
        # Generate multiple traffic points around the location
        num_points = 10
        for i in range(num_points):
            # Add random offset
            offset_lat = np.random.normal(0, radius/111)  # Rough km to degrees
            offset_lng = np.random.normal(0, radius/111)
            
            new_lat = location[0] + offset_lat
            new_lng = location[1] + offset_lng
            
            flow = TrafficFlow(
                segment_id=f"synthetic_{i}",
                location=(new_lat, new_lng),
                speed_kmh=self._generate_speed(new_lat, new_lng),
                density=self._generate_density(new_lat, new_lng),
                occupancy=self._generate_occupancy(new_lat, new_lng),
                volume=self._generate_volume(new_lat, new_lng),
                timestamp=datetime.now(),
                confidence=0.9
            )
            flows.append(flow)
        
        return flows
    
    async def get_traffic_incidents(self, 
                                   location: Tuple[float, float],
                                   radius: float = 10.0) -> List[TrafficIncident]:
        """Generate synthetic traffic incidents"""
        incidents = []
        
        # Random chance of incidents
        if np.random.random() < 0.3:  # 30% chance
            num_incidents = np.random.randint(1, 4)
            
            for i in range(num_incidents):
                offset_lat = np.random.normal(0, radius/111)
                offset_lng = np.random.normal(0, radius/111)
                
                incident = TrafficIncident(
                    id=f"incident_{i}",
                    location=(location[0] + offset_lat, location[1] + offset_lng),
                    type=np.random.choice(['accident', 'construction', 'congestion']),
                    severity=np.random.randint(1, 6),
                    description=f"Synthetic {np.random.choice(['accident', 'construction', 'congestion'])}",
                    start_time=datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
                    estimated_duration=timedelta(minutes=np.random.randint(30, 180)),
                    affected_lanes=list(range(np.random.randint(1, 4))),
                    impact_radius=np.random.uniform(0.5, 2.0)
                )
                incidents.append(incident)
        
        return incidents
    
    async def get_traffic_predictions(self, 
                                     location: Tuple[float, float],
                                     time_horizon: timedelta) -> List[TrafficPrediction]:
        """Generate synthetic traffic predictions"""
        predictions = []
        
        # Generate predictions for different time horizons
        time_steps = [timedelta(minutes=15), timedelta(minutes=30), timedelta(hours=1), timedelta(hours=2)]
        
        for time_step in time_steps:
            if time_step <= time_horizon:
                prediction = TrafficPrediction(
                    location=location,
                    time_horizon=time_step,
                    predicted_speed=self._predict_speed(location, time_step),
                    predicted_density=self._predict_density(location, time_step),
                    confidence=0.7,
                    factors={
                        'historical_pattern': 0.4,
                        'time_of_day': 0.3,
                        'weather_impact': 0.2,
                        'special_events': 0.1
                    }
                )
                predictions.append(prediction)
        
        return predictions
    
    def _generate_speed(self, lat: float, lng: float) -> float:
        """Generate realistic speed based on location"""
        # Simulate different road types
        if abs(lat) > 0.3 or abs(lng) > 0.3:  # Highway
            base_speed = 80.0
            variation = 20.0
        else:  # Urban
            base_speed = 35.0
            variation = 15.0
        
        speed = base_speed + variation * np.sin(lat * 10) * np.cos(lng * 10)
        return max(5, min(130, speed))
    
    def _generate_density(self, lat: float, lng: float) -> float:
        """Generate realistic density"""
        base_density = 25.0
        variation = 20.0 * np.sin(lat * 8) * np.cos(lng * 8)
        return max(5, min(80, base_density + variation))
    
    def _generate_occupancy(self, lat: float, lng: float) -> float:
        """Generate realistic occupancy"""
        return min(1.0, max(0.0, 0.2 + 0.6 * np.sin(lat * 6) * np.cos(lng * 6)))
    
    def _generate_volume(self, lat: float, lng: float) -> int:
        """Generate realistic volume"""
        base_volume = 800
        variation = 400 * np.sin(lat * 7) * np.cos(lng * 7)
        return int(max(100, min(2500, base_volume + variation)))
    
    def _predict_speed(self, location: Tuple[float, float], time_horizon: timedelta) -> float:
        """Predict future speed"""
        current_speed = self._generate_speed(location[0], location[1])
        
        # Apply time-based adjustments
        hour_factor = 1.0
        if time_horizon.total_seconds() < 3600:  # Within 1 hour
            hour_factor = 0.9  # Slight decrease
        elif time_horizon.total_seconds() < 7200:  # Within 2 hours
            hour_factor = 0.8  # More decrease
        
        return current_speed * hour_factor
    
    def _predict_density(self, location: Tuple[float, float], time_horizon: timedelta) -> float:
        """Predict future density"""
        current_density = self._generate_density(location[0], location[1])
        
        # Apply time-based adjustments
        hour_factor = 1.0
        if time_horizon.total_seconds() < 3600:
            hour_factor = 1.1  # Slight increase
        elif time_horizon.total_seconds() < 7200:
            hour_factor = 1.2  # More increase
        
        return current_density * hour_factor


class TrafficService:
    """
    Main traffic service that aggregates data from multiple providers
    """
    
    def __init__(self):
        self.providers = []
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize providers
        if settings.google_maps_api_key:
            self.providers.append(GoogleTrafficProvider(settings.google_maps_api_key))
        
        self.providers.append(OpenStreetMapTrafficProvider())
        self.providers.append(SyntheticTrafficProvider())  # Always include synthetic for demo
    
    async def get_comprehensive_traffic_data(self, 
                                           location: Tuple[float, float],
                                           radius: float = 5.0) -> Dict[str, Any]:
        """
        Get comprehensive traffic data from all available providers
        
        Args:
            location: Center location (lat, lng)
            radius: Search radius in kilometers
            
        Returns:
            Comprehensive traffic data dictionary
        """
        # Check cache first
        cache_key = f"{location[0]:.4f},{location[1]:.4f}_{radius}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_data
        
        # Collect data from all providers
        all_traffic_flows = []
        all_incidents = []
        all_predictions = []
        
        for provider in self.providers:
            try:
                async with provider:
                    # Get traffic flows
                    flows = await provider.get_traffic_data(location, radius)
                    all_traffic_flows.extend(flows)
                    
                    # Get incidents
                    incidents = await provider.get_traffic_incidents(location, radius)
                    all_incidents.extend(incidents)
                    
                    # Get predictions
                    predictions = await provider.get_traffic_predictions(location, timedelta(hours=2))
                    all_predictions.extend(predictions)
                    
            except Exception as e:
                logging.error(f"Error with traffic provider {type(provider).__name__}: {e}")
                continue
        
        # Process and aggregate data
        processed_data = self._process_traffic_data(
            all_traffic_flows, all_incidents, all_predictions, location
        )
        
        # Cache the result
        self.cache[cache_key] = (processed_data, datetime.now())
        
        return processed_data
    
    def _process_traffic_data(self, 
                             flows: List[TrafficFlow],
                             incidents: List[TrafficIncident],
                             predictions: List[TrafficPrediction],
                             center_location: Tuple[float, float]) -> Dict[str, Any]:
        """Process and aggregate traffic data from multiple sources"""
        
        # Calculate traffic metrics
        if flows:
            avg_speed = np.mean([flow.speed_kmh for flow in flows])
            avg_density = np.mean([flow.density for flow in flows])
            avg_occupancy = np.mean([flow.occupancy for flow in flows])
            total_volume = sum([flow.volume for flow in flows])
        else:
            avg_speed = 50.0  # Default values
            avg_density = 25.0
            avg_occupancy = 0.3
            total_volume = 1000
        
        # Calculate traffic level (0-1 scale)
        traffic_level = self._calculate_traffic_level(avg_speed, avg_density, avg_occupancy)
        
        # Process incidents
        active_incidents = [inc for inc in incidents if 
                           datetime.now() - inc.start_time < inc.estimated_duration]
        
        # Calculate impact scores
        incident_impact = self._calculate_incident_impact(active_incidents, center_location)
        
        # Process predictions
        prediction_summary = self._summarize_predictions(predictions)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'location': center_location,
            'traffic_metrics': {
                'level': traffic_level,  # 0-1 scale (0 = free flow, 1 = heavy congestion)
                'average_speed_kmh': avg_speed,
                'average_density': avg_density,
                'average_occupancy': avg_occupancy,
                'total_volume': total_volume,
                'confidence': np.mean([flow.confidence for flow in flows]) if flows else 0.5
            },
            'incidents': {
                'count': len(active_incidents),
                'impact_score': incident_impact,
                'details': [
                    {
                        'id': inc.id,
                        'type': inc.type,
                        'severity': inc.severity,
                        'location': inc.location,
                        'description': inc.description,
                        'estimated_duration_minutes': inc.estimated_duration.total_seconds() / 60
                    }
                    for inc in active_incidents
                ]
            },
            'predictions': prediction_summary,
            'data_sources': len(self.providers),
            'raw_data_points': len(flows)
        }
    
    def _calculate_traffic_level(self, speed: float, density: float, occupancy: float) -> float:
        """Calculate overall traffic level (0-1 scale)"""
        # Normalize metrics to 0-1 scale
        speed_score = max(0, min(1, (speed - 10) / 110))  # 10-120 km/h -> 0-1
        density_score = min(1, density / 80)  # 0-80 vehicles/km -> 0-1
        occupancy_score = occupancy  # Already 0-1
        
        # Weighted combination (lower speed and higher density/occupancy = worse traffic)
        traffic_level = (1 - speed_score) * 0.4 + density_score * 0.3 + occupancy_score * 0.3
        
        return max(0, min(1, traffic_level))
    
    def _calculate_incident_impact(self, 
                                  incidents: List[TrafficIncident],
                                  center_location: Tuple[float, float]) -> float:
        """Calculate overall impact score of incidents"""
        if not incidents:
            return 0.0
        
        total_impact = 0.0
        for incident in incidents:
            # Calculate distance to center location
            distance = self._calculate_distance(center_location, incident.location)
            
            # Impact decreases with distance
            distance_factor = max(0, 1 - distance / incident.impact_radius)
            
            # Severity factor (1-5 scale -> 0.2-1.0)
            severity_factor = incident.severity / 5.0
            
            # Combined impact
            impact = distance_factor * severity_factor
            total_impact += impact
        
        return min(1.0, total_impact)
    
    def _summarize_predictions(self, predictions: List[TrafficPrediction]) -> Dict[str, Any]:
        """Summarize traffic predictions"""
        if not predictions:
            return {'available': False}
        
        # Group by time horizon
        time_horizons = {}
        for pred in predictions:
            horizon_key = f"{int(pred.time_horizon.total_seconds() / 60)}_min"
            if horizon_key not in time_horizons:
                time_horizons[horizon_key] = []
            time_horizons[horizon_key].append(pred)
        
        summary = {'available': True, 'horizons': {}}
        
        for horizon, preds in time_horizons.items():
            avg_speed = np.mean([p.predicted_speed for p in preds])
            avg_density = np.mean([p.predicted_density for p in preds])
            avg_confidence = np.mean([p.confidence for p in preds])
            
            summary['horizons'][horizon] = {
                'predicted_speed_kmh': avg_speed,
                'predicted_density': avg_density,
                'confidence': avg_confidence,
                'traffic_level': self._calculate_traffic_level(avg_speed, avg_density, 0.3)
            }
        
        return summary
    
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
    
    async def get_route_traffic_analysis(self, 
                                       route_coordinates: List[Tuple[float, float]],
                                       radius: float = 1.0) -> Dict[str, Any]:
        """
        Analyze traffic conditions along a route
        
        Args:
            route_coordinates: List of (lat, lng) coordinates along the route
            radius: Analysis radius around each point
            
        Returns:
            Route traffic analysis
        """
        route_analysis = {
            'segments': [],
            'overall_metrics': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        segment_metrics = []
        
        for i in range(len(route_coordinates) - 1):
            # Analyze traffic for route segment
            start_point = route_coordinates[i]
            end_point = route_coordinates[i + 1]
            mid_point = ((start_point[0] + end_point[0]) / 2, 
                        (start_point[1] + end_point[1]) / 2)
            
            # Get traffic data for segment
            traffic_data = await self.get_comprehensive_traffic_data(mid_point, radius)
            
            segment_info = {
                'segment_index': i,
                'start_point': start_point,
                'end_point': end_point,
                'mid_point': mid_point,
                'traffic_level': traffic_data['traffic_metrics']['level'],
                'average_speed': traffic_data['traffic_metrics']['average_speed_kmh'],
                'incidents': traffic_data['incidents']['count'],
                'impact_score': traffic_data['incidents']['impact_score']
            }
            
            route_analysis['segments'].append(segment_info)
            segment_metrics.append(traffic_data['traffic_metrics']['level'])
        
        # Calculate overall metrics
        if segment_metrics:
            route_analysis['overall_metrics'] = {
                'average_traffic_level': np.mean(segment_metrics),
                'max_traffic_level': max(segment_metrics),
                'min_traffic_level': min(segment_metrics),
                'traffic_variability': np.std(segment_metrics),
                'congested_segments': sum(1 for level in segment_metrics if level > 0.7)
            }
            
            # Identify bottlenecks
            for i, segment in enumerate(route_analysis['segments']):
                if segment['traffic_level'] > 0.7:
                    route_analysis['bottlenecks'].append({
                        'segment_index': i,
                        'location': segment['mid_point'],
                        'traffic_level': segment['traffic_level'],
                        'incidents': segment['incidents']
                    })
            
            # Generate recommendations
            route_analysis['recommendations'] = self._generate_route_recommendations(
                route_analysis['overall_metrics'], route_analysis['bottlenecks']
            )
        
        return route_analysis
    
    def _generate_route_recommendations(self, 
                                       overall_metrics: Dict,
                                       bottlenecks: List[Dict]) -> List[str]:
        """Generate route recommendations based on traffic analysis"""
        recommendations = []
        
        if overall_metrics['average_traffic_level'] > 0.7:
            recommendations.append("Consider alternative routes due to heavy traffic")
        
        if overall_metrics['congested_segments'] > len(bottlenecks) * 0.5:
            recommendations.append("Multiple bottlenecks detected - expect significant delays")
        
        if bottlenecks:
            recommendations.append(f"Bottlenecks detected at {len(bottlenecks)} locations")
        
        if overall_metrics['traffic_variability'] > 0.3:
            recommendations.append("Traffic conditions vary significantly along route")
        
        if not recommendations:
            recommendations.append("Traffic conditions appear favorable for this route")
        
        return recommendations
    
    def clear_cache(self):
        """Clear the traffic data cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_ttl_seconds': self.cache_ttl,
            'providers_count': len(self.providers)
        }


# Global traffic service instance
traffic_service = TrafficService()
