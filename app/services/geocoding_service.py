"""
Geocoding Service for converting location names to coordinates and vice versa
Supports multiple providers: Nominatim (OpenStreetMap), Google Maps, etc.
"""

import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import json

from app.core.config import settings


@dataclass
class GeocodedLocation:
    """Represents a geocoded location"""
    name: str
    display_name: str
    latitude: float
    longitude: float
    address: Dict[str, str]
    place_type: str
    importance: float
    bounding_box: Optional[List[float]] = None


class GeocodingProvider:
    """Base class for geocoding providers"""
    
    async def geocode(self, location_name: str, **kwargs) -> List[GeocodedLocation]:
        """Convert location name to coordinates"""
        raise NotImplementedError
    
    async def reverse_geocode(self, lat: float, lng: float) -> Optional[GeocodedLocation]:
        """Convert coordinates to location name"""
        raise NotImplementedError


class NominatimGeocoder(GeocodingProvider):
    """Nominatim (OpenStreetMap) geocoding provider - Free and no API key required"""
    
    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org"
        self.session = None
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)
        self.last_request_time = None
        self.min_request_interval = 1.0  # Nominatim requires 1 second between requests
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Ensure we respect Nominatim's rate limiting"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = datetime.now()
    
    async def geocode(self, location_name: str, limit: int = 5, **kwargs) -> List[GeocodedLocation]:
        """
        Convert location name to coordinates
        
        Args:
            location_name: Name of the location to geocode
            limit: Maximum number of results to return
            
        Returns:
            List of geocoded locations
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Check cache
        cache_key = f"geocode_{location_name}_{limit}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_data
        
        await self._rate_limit()
        
        url = f"{self.base_url}/search"
        params = {
            'q': location_name,
            'format': 'json',
            'addressdetails': 1,
            'limit': limit,
            'accept-language': 'en'
        }
        
        headers = {
            'User-Agent': 'AI-Route-Optimizer/1.0 (Educational Project)'
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    results = self._parse_geocode_results(data)
                    
                    # Cache results
                    self.cache[cache_key] = (results, datetime.now())
                    
                    return results
                else:
                    logging.error(f"Nominatim geocoding error: {response.status}")
                    return []
        except Exception as e:
            logging.error(f"Error geocoding location '{location_name}': {e}")
            return []
    
    async def reverse_geocode(self, lat: float, lng: float) -> Optional[GeocodedLocation]:
        """
        Convert coordinates to location name
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            Geocoded location or None
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Check cache
        cache_key = f"reverse_{lat:.6f}_{lng:.6f}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_data
        
        await self._rate_limit()
        
        url = f"{self.base_url}/reverse"
        params = {
            'lat': lat,
            'lon': lng,
            'format': 'json',
            'addressdetails': 1,
            'accept-language': 'en'
        }
        
        headers = {
            'User-Agent': 'AI-Route-Optimizer/1.0 (Educational Project)'
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    result = self._parse_reverse_geocode_result(data)
                    
                    # Cache result
                    self.cache[cache_key] = (result, datetime.now())
                    
                    return result
                else:
                    logging.error(f"Nominatim reverse geocoding error: {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Error reverse geocoding ({lat}, {lng}): {e}")
            return None
    
    def _parse_geocode_results(self, data: List[Dict]) -> List[GeocodedLocation]:
        """Parse Nominatim geocoding results"""
        results = []
        
        for item in data:
            try:
                location = GeocodedLocation(
                    name=item.get('name', ''),
                    display_name=item.get('display_name', ''),
                    latitude=float(item.get('lat', 0)),
                    longitude=float(item.get('lon', 0)),
                    address=item.get('address', {}),
                    place_type=item.get('type', 'unknown'),
                    importance=float(item.get('importance', 0)),
                    bounding_box=item.get('boundingbox', None)
                )
                results.append(location)
            except Exception as e:
                logging.error(f"Error parsing geocode result: {e}")
                continue
        
        return results
    
    def _parse_reverse_geocode_result(self, data: Dict) -> Optional[GeocodedLocation]:
        """Parse Nominatim reverse geocoding result"""
        try:
            return GeocodedLocation(
                name=data.get('name', ''),
                display_name=data.get('display_name', ''),
                latitude=float(data.get('lat', 0)),
                longitude=float(data.get('lon', 0)),
                address=data.get('address', {}),
                place_type=data.get('type', 'unknown'),
                importance=float(data.get('importance', 0)),
                bounding_box=data.get('boundingbox', None)
            )
        except Exception as e:
            logging.error(f"Error parsing reverse geocode result: {e}")
            return None


class GoogleMapsGeocoder(GeocodingProvider):
    """Google Maps geocoding provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/geocode"
        self.session = None
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def geocode(self, location_name: str, **kwargs) -> List[GeocodedLocation]:
        """Convert location name to coordinates using Google Maps API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Check cache
        cache_key = f"google_geocode_{location_name}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_data
        
        url = f"{self.base_url}/json"
        params = {
            'address': location_name,
            'key': self.api_key
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'OK':
                        results = self._parse_google_results(data.get('results', []))
                        self.cache[cache_key] = (results, datetime.now())
                        return results
                    else:
                        logging.error(f"Google geocoding error: {data.get('status')}")
                        return []
                else:
                    logging.error(f"Google API error: {response.status}")
                    return []
        except Exception as e:
            logging.error(f"Error with Google geocoding: {e}")
            return []
    
    async def reverse_geocode(self, lat: float, lng: float) -> Optional[GeocodedLocation]:
        """Convert coordinates to location name using Google Maps API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/json"
        params = {
            'latlng': f"{lat},{lng}",
            'key': self.api_key
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'OK' and data.get('results'):
                        results = self._parse_google_results(data['results'])
                        return results[0] if results else None
                    else:
                        logging.error(f"Google reverse geocoding error: {data.get('status')}")
                        return None
                else:
                    logging.error(f"Google API error: {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Error with Google reverse geocoding: {e}")
            return None
    
    def _parse_google_results(self, results: List[Dict]) -> List[GeocodedLocation]:
        """Parse Google Maps geocoding results"""
        parsed = []
        
        for result in results:
            try:
                geometry = result.get('geometry', {})
                location = geometry.get('location', {})
                
                # Extract address components
                address = {}
                for component in result.get('address_components', []):
                    types = component.get('types', [])
                    if 'locality' in types or 'city' in types:
                        address['city'] = component.get('long_name', '')
                    elif 'administrative_area_level_1' in types:
                        address['state'] = component.get('long_name', '')
                    elif 'country' in types:
                        address['country'] = component.get('long_name', '')
                    elif 'postal_code' in types:
                        address['postcode'] = component.get('long_name', '')
                
                geocoded = GeocodedLocation(
                    name=result.get('formatted_address', '').split(',')[0],
                    display_name=result.get('formatted_address', ''),
                    latitude=location.get('lat', 0),
                    longitude=location.get('lng', 0),
                    address=address,
                    place_type=result.get('types', ['unknown'])[0],
                    importance=1.0,  # Google doesn't provide importance
                    bounding_box=None
                )
                parsed.append(geocoded)
            except Exception as e:
                logging.error(f"Error parsing Google result: {e}")
                continue
        
        return parsed


class GeocodingService:
    """
    Main geocoding service that manages multiple providers
    """
    
    def __init__(self):
        self.nominatim = NominatimGeocoder()
        self.google_geocoder = None
        
        # Initialize Google geocoder if API key is available
        if settings.google_maps_api_key:
            self.google_geocoder = GoogleMapsGeocoder(settings.google_maps_api_key)
        
        self.default_provider = 'nominatim'
    
    async def geocode(self, location_name: str, provider: Optional[str] = None, **kwargs) -> List[GeocodedLocation]:
        """
        Geocode a location name to coordinates
        
        Args:
            location_name: Name of the location
            provider: Geocoding provider to use ('nominatim' or 'google')
            
        Returns:
            List of geocoded locations
        """
        provider = provider or self.default_provider
        
        try:
            if provider == 'google' and self.google_geocoder:
                async with self.google_geocoder as geocoder:
                    return await geocoder.geocode(location_name, **kwargs)
            else:
                # Use Nominatim as default/fallback
                async with self.nominatim as geocoder:
                    return await geocoder.geocode(location_name, **kwargs)
        except Exception as e:
            logging.error(f"Geocoding error: {e}")
            return []
    
    async def reverse_geocode(self, lat: float, lng: float, provider: Optional[str] = None) -> Optional[GeocodedLocation]:
        """
        Reverse geocode coordinates to location name
        
        Args:
            lat: Latitude
            lng: Longitude
            provider: Geocoding provider to use
            
        Returns:
            Geocoded location or None
        """
        provider = provider or self.default_provider
        
        try:
            if provider == 'google' and self.google_geocoder:
                async with self.google_geocoder as geocoder:
                    return await geocoder.reverse_geocode(lat, lng)
            else:
                # Use Nominatim as default/fallback
                async with self.nominatim as geocoder:
                    return await geocoder.reverse_geocode(lat, lng)
        except Exception as e:
            logging.error(f"Reverse geocoding error: {e}")
            return None
    
    async def get_location_suggestions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get enhanced location suggestions for autocomplete with intelligent search
        
        Args:
            query: Search query
            limit: Maximum number of suggestions
            
        Returns:
            List of enhanced location suggestions
        """
        if len(query) < 2:
            return []
        
        suggestions = []
        
        try:
            # Strategy 1: Get initial results from primary geocoding
            results = await self.geocode(query, limit=limit * 2)  # Get more for filtering
            
            for result in results:
                suggestions.append({
                    'name': result.name or result.display_name.split(',')[0],
                    'display_name': result.display_name,
                    'latitude': result.latitude,
                    'longitude': result.longitude,
                    'type': result.place_type,
                    'importance': result.importance,
                    'country': result.address.get('country', ''),
                    'state': result.address.get('state', ''),
                    'city': result.address.get('city', '')
                })
            
            # Strategy 2: If we have few results, try with broader search
            if len(suggestions) < limit and len(query) >= 3:
                # Try searching with wildcards for partial matches
                broader_query = f"{query}*"
                broader_results = await self.geocode(broader_query, limit=5)
                
                for result in broader_results:
                    # Avoid duplicates
                    is_duplicate = any(
                        abs(s['latitude'] - result.latitude) < 0.001 and 
                        abs(s['longitude'] - result.longitude) < 0.001 
                        for s in suggestions
                    )
                    
                    if not is_duplicate:
                        suggestions.append({
                            'name': result.name or result.display_name.split(',')[0],
                            'display_name': result.display_name,
                            'latitude': result.latitude,
                            'longitude': result.longitude,
                            'type': result.place_type,
                            'importance': result.importance - 0.1,  # Lower priority for broader search
                            'country': result.address.get('country', ''),
                            'state': result.address.get('state', ''),
                            'city': result.address.get('city', '')
                        })
            
            # Strategy 3: Enhance scoring for better results
            def calculate_relevance_score(suggestion):
                score = suggestion['importance']
                
                # Boost exact name matches
                if query.lower() in suggestion['name'].lower():
                    score += 0.3
                
                # Boost major cities and important places
                if suggestion['type'] in ['city', 'town', 'administrative']:
                    score += 0.2
                
                # Boost Indian locations (for Indian users)
                if suggestion['country'].lower() in ['india', 'भारत']:
                    score += 0.15
                
                # Boost well-known places
                if suggestion['importance'] > 0.7:
                    score += 0.1
                
                return score
            
            # Sort by relevance score
            suggestions.sort(key=calculate_relevance_score, reverse=True)
            
            # Return top results
            return suggestions[:limit]
            
        except Exception as e:
            logging.error(f"Error getting location suggestions for '{query}': {e}")
            return []
    
    async def get_popular_destinations(self, country: str = "IN") -> List[Dict[str, Any]]:
        """
        Get popular destinations for quick access
        
        Args:
            country: Country code for popular destinations
            
        Returns:
            List of popular destination suggestions
        """
        popular_places = {
            "IN": [
                "New Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", 
                "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Agra", 
                "Goa", "Kochi", "Thiruvananthapuram", "Bhopal", "Lucknow"
            ],
            "US": [
                "New York", "Los Angeles", "Chicago", "San Francisco", 
                "Las Vegas", "Miami", "Boston", "Seattle", "Denver"
            ],
            "GB": [
                "London", "Manchester", "Birmingham", "Liverpool", 
                "Edinburgh", "Glasgow", "Bristol", "Leeds"
            ]
        }
        
        destinations = []
        places = popular_places.get(country, popular_places["IN"])
        
        for place in places[:10]:  # Limit to 10
            try:
                results = await self.geocode(place, limit=1)
                if results:
                    result = results[0]
                    destinations.append({
                        'name': result.name,
                        'display_name': result.display_name,
                        'latitude': result.latitude,
                        'longitude': result.longitude,
                        'type': result.place_type,
                        'country': result.address.get('country', ''),
                        'popularity': 'high'
                    })
            except Exception as e:
                logging.warning(f"Error geocoding popular place '{place}': {e}")
                continue
        
        return destinations


# Global geocoding service instance
geocoding_service = GeocodingService()
