"""
Route optimization data models
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


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
