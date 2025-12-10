"""
Weather Service for real-time weather data integration
Provides weather conditions that affect route optimization
"""

import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json

from app.core.config import settings


@dataclass
class WeatherCondition:
    """Represents weather conditions at a location"""
    location: Tuple[float, float]
    temperature_celsius: float
    feels_like_celsius: float
    humidity: int
    pressure: int
    wind_speed_kmh: float
    wind_direction: int
    visibility_km: float
    precipitation_mm: float
    weather_main: str
    weather_description: str
    clouds: int
    timestamp: datetime
    sunrise: Optional[datetime] = None
    sunset: Optional[datetime] = None


@dataclass
class WeatherForecast:
    """Weather forecast for future time periods"""
    location: Tuple[float, float]
    forecast_time: datetime
    temperature_celsius: float
    precipitation_probability: float
    precipitation_mm: float
    wind_speed_kmh: float
    weather_main: str
    weather_description: str


class WeatherService:
    """
    Weather service for route optimization
    Uses OpenWeatherMap API (free tier available)
    """
    
    def __init__(self):
        self.api_key = settings.weather_api_key or "demo_key"
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.session = None
        self.cache = {}
        self.cache_ttl = timedelta(minutes=30)  # Weather data valid for 30 minutes
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_current_weather(self, lat: float, lng: float) -> Optional[WeatherCondition]:
        """
        Get current weather conditions for a location
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            Current weather conditions or None
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Check cache
        cache_key = f"weather_{lat:.4f}_{lng:.4f}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_data
        
        # For demo purposes, generate synthetic weather data if no API key
        if self.api_key == "demo_key":
            weather = self._generate_synthetic_weather(lat, lng)
            self.cache[cache_key] = (weather, datetime.now())
            return weather
        
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lng,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    weather = self._parse_weather_data(data, (lat, lng))
                    self.cache[cache_key] = (weather, datetime.now())
                    return weather
                else:
                    logging.error(f"Weather API error: {response.status}")
                    # Return synthetic data as fallback
                    return self._generate_synthetic_weather(lat, lng)
        except Exception as e:
            logging.error(f"Error fetching weather data: {e}")
            return self._generate_synthetic_weather(lat, lng)
    
    async def get_weather_forecast(self, lat: float, lng: float, hours: int = 24) -> List[WeatherForecast]:
        """
        Get weather forecast for a location
        
        Args:
            lat: Latitude
            lng: Longitude
            hours: Number of hours to forecast
            
        Returns:
            List of weather forecasts
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # For demo purposes, generate synthetic forecast if no API key
        if self.api_key == "demo_key":
            return self._generate_synthetic_forecast(lat, lng, hours)
        
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lng,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': min(40, hours // 3)  # API provides 3-hour intervals
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_forecast_data(data, (lat, lng))
                else:
                    logging.error(f"Weather forecast API error: {response.status}")
                    return self._generate_synthetic_forecast(lat, lng, hours)
        except Exception as e:
            logging.error(f"Error fetching weather forecast: {e}")
            return self._generate_synthetic_forecast(lat, lng, hours)
    
    async def get_route_weather_analysis(self, route_points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze weather conditions along a route
        
        Args:
            route_points: List of (lat, lng) coordinates along the route
            
        Returns:
            Weather analysis for the route
        """
        # Sample weather at key points along the route
        sample_points = self._sample_route_points(route_points, max_samples=5)
        
        weather_conditions = []
        for point in sample_points:
            weather = await self.get_current_weather(point[0], point[1])
            if weather:
                weather_conditions.append(weather)
        
        if not weather_conditions:
            return {
                'error': 'No weather data available',
                'conditions': []
            }
        
        # Analyze weather impact on route
        avg_temp = sum(w.temperature_celsius for w in weather_conditions) / len(weather_conditions)
        avg_wind = sum(w.wind_speed_kmh for w in weather_conditions) / len(weather_conditions)
        avg_visibility = sum(w.visibility_km for w in weather_conditions) / len(weather_conditions)
        total_precipitation = sum(w.precipitation_mm for w in weather_conditions)
        
        # Determine weather impact on travel
        impact_score = self._calculate_weather_impact(
            avg_temp, avg_wind, avg_visibility, total_precipitation
        )
        
        return {
            'conditions': [
                {
                    'location': {'lat': w.location[0], 'lng': w.location[1]},
                    'temperature': w.temperature_celsius,
                    'weather': w.weather_description,
                    'wind_speed': w.wind_speed_kmh,
                    'visibility': w.visibility_km,
                    'precipitation': w.precipitation_mm
                }
                for w in weather_conditions
            ],
            'average_metrics': {
                'temperature': round(avg_temp, 1),
                'wind_speed': round(avg_wind, 1),
                'visibility': round(avg_visibility, 1),
                'total_precipitation': round(total_precipitation, 1)
            },
            'impact_assessment': {
                'score': impact_score,
                'level': self._get_impact_level(impact_score),
                'recommendations': self._get_weather_recommendations(impact_score, weather_conditions)
            }
        }
    
    def _parse_weather_data(self, data: Dict, location: Tuple[float, float]) -> WeatherCondition:
        """Parse OpenWeatherMap current weather data"""
        main = data.get('main', {})
        wind = data.get('wind', {})
        weather = data.get('weather', [{}])[0]
        sys = data.get('sys', {})
        
        return WeatherCondition(
            location=location,
            temperature_celsius=main.get('temp', 20),
            feels_like_celsius=main.get('feels_like', 20),
            humidity=main.get('humidity', 50),
            pressure=main.get('pressure', 1013),
            wind_speed_kmh=wind.get('speed', 0) * 3.6,  # Convert m/s to km/h
            wind_direction=wind.get('deg', 0),
            visibility_km=data.get('visibility', 10000) / 1000,
            precipitation_mm=data.get('rain', {}).get('1h', 0),
            weather_main=weather.get('main', 'Clear'),
            weather_description=weather.get('description', 'clear sky'),
            clouds=data.get('clouds', {}).get('all', 0),
            timestamp=datetime.fromtimestamp(data.get('dt', datetime.now().timestamp())),
            sunrise=datetime.fromtimestamp(sys.get('sunrise')) if sys.get('sunrise') else None,
            sunset=datetime.fromtimestamp(sys.get('sunset')) if sys.get('sunset') else None
        )
    
    def _parse_forecast_data(self, data: Dict, location: Tuple[float, float]) -> List[WeatherForecast]:
        """Parse OpenWeatherMap forecast data"""
        forecasts = []
        
        for item in data.get('list', []):
            main = item.get('main', {})
            weather = item.get('weather', [{}])[0]
            wind = item.get('wind', {})
            
            forecast = WeatherForecast(
                location=location,
                forecast_time=datetime.fromtimestamp(item.get('dt', datetime.now().timestamp())),
                temperature_celsius=main.get('temp', 20),
                precipitation_probability=item.get('pop', 0),
                precipitation_mm=item.get('rain', {}).get('3h', 0),
                wind_speed_kmh=wind.get('speed', 0) * 3.6,
                weather_main=weather.get('main', 'Clear'),
                weather_description=weather.get('description', 'clear sky')
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def _generate_synthetic_weather(self, lat: float, lng: float) -> WeatherCondition:
        """Generate synthetic weather data for demo purposes"""
        import numpy as np
        
        # Generate realistic weather based on latitude
        base_temp = 25 - abs(lat) * 0.5  # Cooler at higher latitudes
        temp_variation = np.sin(lng * 0.1) * 5
        
        return WeatherCondition(
            location=(lat, lng),
            temperature_celsius=round(base_temp + temp_variation, 1),
            feels_like_celsius=round(base_temp + temp_variation - 2, 1),
            humidity=int(50 + np.sin(lat * 0.5) * 30),
            pressure=1013,
            wind_speed_kmh=round(10 + abs(np.cos(lng * 0.2)) * 15, 1),
            wind_direction=int((lat + lng) * 10) % 360,
            visibility_km=10.0,
            precipitation_mm=0.0,
            weather_main='Clear',
            weather_description='clear sky',
            clouds=int(abs(np.sin(lat * lng)) * 50),
            timestamp=datetime.now()
        )
    
    def _generate_synthetic_forecast(self, lat: float, lng: float, hours: int) -> List[WeatherForecast]:
        """Generate synthetic weather forecast for demo purposes"""
        import numpy as np
        
        forecasts = []
        base_temp = 25 - abs(lat) * 0.5
        
        for i in range(0, hours, 3):
            forecast_time = datetime.now() + timedelta(hours=i)
            temp_variation = np.sin(i * 0.1) * 5
            
            forecast = WeatherForecast(
                location=(lat, lng),
                forecast_time=forecast_time,
                temperature_celsius=round(base_temp + temp_variation, 1),
                precipitation_probability=max(0, min(1, abs(np.sin(i * 0.2)))),
                precipitation_mm=0.0,
                wind_speed_kmh=round(10 + abs(np.cos(i * 0.15)) * 10, 1),
                weather_main='Clear' if i % 6 != 0 else 'Clouds',
                weather_description='clear sky' if i % 6 != 0 else 'few clouds'
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def _sample_route_points(self, route_points: List[Tuple[float, float]], max_samples: int = 5) -> List[Tuple[float, float]]:
        """Sample representative points from a route"""
        if len(route_points) <= max_samples:
            return route_points
        
        # Sample evenly distributed points
        indices = [int(i * len(route_points) / max_samples) for i in range(max_samples)]
        return [route_points[i] for i in indices]
    
    def _calculate_weather_impact(self, temp: float, wind: float, visibility: float, precipitation: float) -> float:
        """
        Calculate weather impact score on travel (0-1, higher is worse)
        """
        impact = 0.0
        
        # Temperature impact (extreme temperatures)
        if temp < 0 or temp > 40:
            impact += 0.3
        elif temp < 5 or temp > 35:
            impact += 0.15
        
        # Wind impact
        if wind > 50:
            impact += 0.3
        elif wind > 30:
            impact += 0.15
        
        # Visibility impact
        if visibility < 1:
            impact += 0.3
        elif visibility < 5:
            impact += 0.15
        
        # Precipitation impact
        if precipitation > 10:
            impact += 0.3
        elif precipitation > 2:
            impact += 0.15
        
        return min(1.0, impact)
    
    def _get_impact_level(self, score: float) -> str:
        """Get weather impact level description"""
        if score < 0.2:
            return 'minimal'
        elif score < 0.4:
            return 'low'
        elif score < 0.6:
            return 'moderate'
        elif score < 0.8:
            return 'high'
        else:
            return 'severe'
    
    def _get_weather_recommendations(self, impact_score: float, conditions: List[WeatherCondition]) -> List[str]:
        """Get weather-based recommendations"""
        recommendations = []
        
        if impact_score < 0.2:
            recommendations.append("Weather conditions are favorable for travel")
        
        for condition in conditions:
            if condition.precipitation_mm > 2:
                recommendations.append("Heavy rain expected - allow extra travel time")
            if condition.wind_speed_kmh > 40:
                recommendations.append("Strong winds - drive carefully")
            if condition.visibility_km < 5:
                recommendations.append("Low visibility - use headlights and reduce speed")
            if condition.temperature_celsius < 0:
                recommendations.append("Freezing temperatures - watch for ice on roads")
            if condition.temperature_celsius > 35:
                recommendations.append("High temperature - ensure vehicle cooling system is working")
        
        if not recommendations:
            recommendations.append("No weather warnings for this route")
        
        return list(set(recommendations))[:5]  # Return unique recommendations, max 5


# Global weather service instance
weather_service = WeatherService()
