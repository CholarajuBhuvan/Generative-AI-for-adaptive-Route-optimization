"""
User Profile Management and Route Favorites System
Personalized routing with saved preferences and favorite routes
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import sqlite3
from pathlib import Path

from app.core.config import settings


@dataclass
class UserPreferences:
    """User routing preferences"""
    user_id: str
    preferred_travel_mode: str = "driving"
    preferred_vehicle_type: str = "average"
    time_weight: float = 0.4
    distance_weight: float = 0.3
    cost_weight: float = 0.2
    traffic_weight: float = 0.1
    eco_friendly: bool = False
    avoid_tolls: bool = False
    avoid_highways: bool = False
    max_walking_distance: int = 1000  # meters
    preferred_departure_buffer: int = 15  # minutes
    notification_preferences: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.notification_preferences is None:
            self.notification_preferences = {
                "traffic_alerts": True,
                "weather_warnings": True,
                "route_suggestions": True,
                "carbon_tips": False
            }


@dataclass
class SavedRoute:
    """Saved/favorite route"""
    route_id: str
    user_id: str
    name: str
    start_location: Dict[str, Any]
    end_location: Dict[str, Any]
    waypoints: List[Dict[str, Any]] = None
    route_data: Dict[str, Any] = None
    tags: List[str] = None
    created_at: datetime = None
    last_used: Optional[datetime] = None
    use_count: int = 0
    is_favorite: bool = False
    
    def __post_init__(self):
        if self.waypoints is None:
            self.waypoints = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class RouteHistory:
    """Route usage history entry"""
    history_id: str
    user_id: str
    route_id: str
    timestamp: datetime
    actual_duration_minutes: Optional[float] = None
    actual_distance_km: Optional[float] = None
    user_rating: Optional[int] = None
    feedback: Optional[str] = None
    conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


class UserProfileDatabase:
    """SQLite database for user profiles and routes"""
    
    def __init__(self, db_path: str = "user_profiles.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS saved_routes (
                    route_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    route_data TEXT NOT NULL,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    is_favorite BOOLEAN DEFAULT FALSE
                );
                
                CREATE TABLE IF NOT EXISTS route_history (
                    history_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    route_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    route_data TEXT,
                    performance_data TEXT,
                    user_rating INTEGER,
                    feedback TEXT
                );
                
                CREATE TABLE IF NOT EXISTS route_shares (
                    share_id TEXT PRIMARY KEY,
                    route_id TEXT NOT NULL,
                    shared_by TEXT NOT NULL,
                    share_code TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    is_public BOOLEAN DEFAULT FALSE
                );
                
                CREATE INDEX IF NOT EXISTS idx_user_routes ON saved_routes(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_history ON route_history(user_id);
                CREATE INDEX IF NOT EXISTS idx_share_code ON route_shares(share_code);
            ''')
    
    def save_user_preferences(self, preferences: UserPreferences):
        """Save user preferences to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO user_preferences (user_id, preferences, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (preferences.user_id, json.dumps(asdict(preferences))))
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT preferences FROM user_preferences WHERE user_id = ?',
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return UserPreferences(**data)
            return None
    
    def save_route(self, route: SavedRoute):
        """Save a route to database"""
        with sqlite3.connect(self.db_path) as conn:
            route_dict = asdict(route)
            route_dict['created_at'] = route.created_at.isoformat()
            route_dict['last_used'] = route.last_used.isoformat() if route.last_used else None
            
            conn.execute('''
                INSERT OR REPLACE INTO saved_routes 
                (route_id, user_id, name, route_data, tags, created_at, last_used, use_count, is_favorite)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                route.route_id, route.user_id, route.name,
                json.dumps(route_dict), json.dumps(route.tags),
                route.created_at.isoformat(),
                route.last_used.isoformat() if route.last_used else None,
                route.use_count, route.is_favorite
            ))
    
    def get_user_routes(self, user_id: str, limit: int = 50) -> List[SavedRoute]:
        """Get user's saved routes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT route_data FROM saved_routes 
                WHERE user_id = ? 
                ORDER BY last_used DESC, created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            routes = []
            for row in cursor.fetchall():
                data = json.loads(row[0])
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                if data.get('last_used'):
                    data['last_used'] = datetime.fromisoformat(data['last_used'])
                routes.append(SavedRoute(**data))
            
            return routes
    
    def add_to_history(self, history: RouteHistory):
        """Add route to usage history"""
        with sqlite3.connect(self.db_path) as conn:
            history_dict = asdict(history)
            history_dict['timestamp'] = history.timestamp.isoformat()
            
            conn.execute('''
                INSERT INTO route_history 
                (history_id, user_id, route_id, timestamp, route_data, performance_data, user_rating, feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                history.history_id, history.user_id, history.route_id,
                history.timestamp.isoformat(), 
                json.dumps(history_dict),
                json.dumps(history.conditions),
                history.user_rating, history.feedback
            ))


class UserProfileService:
    """Main service for user profile management"""
    
    def __init__(self):
        self.db = UserProfileDatabase()
        self.active_sessions = {}
    
    async def get_or_create_user_preferences(self, user_id: str) -> UserPreferences:
        """Get existing preferences or create default ones"""
        preferences = self.db.get_user_preferences(user_id)
        if not preferences:
            preferences = UserPreferences(user_id=user_id)
            self.db.save_user_preferences(preferences)
        return preferences
    
    async def update_user_preferences(self, user_id: str, updates: Dict[str, Any]) -> UserPreferences:
        """Update user preferences"""
        preferences = await self.get_or_create_user_preferences(user_id)
        
        # Update fields
        for key, value in updates.items():
            if hasattr(preferences, key):
                setattr(preferences, key, value)
        
        self.db.save_user_preferences(preferences)
        return preferences
    
    async def save_favorite_route(self, user_id: str, route_name: str, 
                                 route_data: Dict[str, Any], tags: List[str] = None) -> SavedRoute:
        """Save a route as favorite"""
        route = SavedRoute(
            route_id=f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=user_id,
            name=route_name,
            start_location=route_data.get('start_location', {}),
            end_location=route_data.get('end_location', {}),
            waypoints=route_data.get('waypoints', []),
            route_data=route_data,
            tags=tags or [],
            is_favorite=True
        )
        
        self.db.save_route(route)
        return route
    
    async def get_user_favorites(self, user_id: str) -> List[SavedRoute]:
        """Get user's favorite routes"""
        all_routes = self.db.get_user_routes(user_id)
        return [route for route in all_routes if route.is_favorite]
    
    async def get_recent_routes(self, user_id: str, limit: int = 10) -> List[SavedRoute]:
        """Get user's recent routes"""
        return self.db.get_user_routes(user_id, limit)
    
    async def search_user_routes(self, user_id: str, query: str) -> List[SavedRoute]:
        """Search user's saved routes"""
        routes = self.db.get_user_routes(user_id)
        query_lower = query.lower()
        
        matching_routes = []
        for route in routes:
            if (query_lower in route.name.lower() or 
                any(query_lower in tag.lower() for tag in route.tags) or
                query_lower in route.start_location.get('name', '').lower() or
                query_lower in route.end_location.get('name', '').lower()):
                matching_routes.append(route)
        
        return matching_routes
    
    async def record_route_usage(self, user_id: str, route_id: str, 
                                performance_data: Dict[str, Any] = None):
        """Record that a route was used"""
        history = RouteHistory(
            history_id=f"{user_id}_{route_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=user_id,
            route_id=route_id,
            timestamp=datetime.now(),
            conditions=performance_data or {}
        )
        
        self.db.add_to_history(history)
    
    async def get_personalized_suggestions(self, user_id: str) -> Dict[str, Any]:
        """Get personalized route suggestions based on user history"""
        preferences = await self.get_or_create_user_preferences(user_id)
        recent_routes = await self.get_recent_routes(user_id, 20)
        
        # Analyze patterns
        frequent_locations = self._analyze_frequent_locations(recent_routes)
        preferred_times = self._analyze_preferred_times(recent_routes)
        
        suggestions = {
            "frequent_destinations": frequent_locations[:5],
            "preferred_travel_times": preferred_times,
            "recommended_settings": {
                "travel_mode": preferences.preferred_travel_mode,
                "vehicle_type": preferences.preferred_vehicle_type,
                "optimization_weights": {
                    "time": preferences.time_weight,
                    "distance": preferences.distance_weight,
                    "cost": preferences.cost_weight,
                    "traffic": preferences.traffic_weight
                }
            },
            "eco_recommendations": self._generate_eco_suggestions(preferences, recent_routes)
        }
        
        return suggestions
    
    def _analyze_frequent_locations(self, routes: List[SavedRoute]) -> List[Dict[str, Any]]:
        """Analyze frequently visited locations"""
        location_counts = {}
        
        for route in routes:
            # Count start locations
            start_name = route.start_location.get('name', 'Unknown')
            location_counts[start_name] = location_counts.get(start_name, 0) + 1
            
            # Count end locations  
            end_name = route.end_location.get('name', 'Unknown')
            location_counts[end_name] = location_counts.get(end_name, 0) + 1
        
        # Sort by frequency
        sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"name": name, "visit_count": count}
            for name, count in sorted_locations if name != 'Unknown'
        ]
    
    def _analyze_preferred_times(self, routes: List[SavedRoute]) -> Dict[str, Any]:
        """Analyze preferred travel times"""
        if not routes:
            return {"most_common_hour": 9, "most_common_day": "Monday"}
        
        hours = []
        days = []
        
        for route in routes:
            if route.last_used:
                hours.append(route.last_used.hour)
                days.append(route.last_used.strftime("%A"))
        
        most_common_hour = max(set(hours), key=hours.count) if hours else 9
        most_common_day = max(set(days), key=days.count) if days else "Monday"
        
        return {
            "most_common_hour": most_common_hour,
            "most_common_day": most_common_day,
            "peak_usage_hours": [h for h in hours if hours.count(h) >= 2]
        }
    
    def _generate_eco_suggestions(self, preferences: UserPreferences, 
                                 routes: List[SavedRoute]) -> List[str]:
        """Generate eco-friendly suggestions"""
        suggestions = []
        
        if not preferences.eco_friendly:
            suggestions.append("Enable eco-friendly routing to reduce carbon footprint")
        
        if preferences.preferred_vehicle_type in ['petrol_car_large', 'diesel_car_large']:
            suggestions.append("Consider switching to a smaller, more fuel-efficient vehicle")
        
        if preferences.preferred_travel_mode == "driving":
            suggestions.append("Try public transit or cycling for shorter routes")
        
        return suggestions


class RouteShareService:
    """Service for sharing routes between users"""
    
    def __init__(self):
        self.db = UserProfileDatabase()
    
    async def create_shareable_route(self, route_id: str, shared_by: str,
                                   expires_hours: int = 168,  # 1 week
                                   is_public: bool = False) -> Dict[str, str]:
        """Create a shareable route link"""
        import secrets
        
        share_code = secrets.token_urlsafe(16)
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute('''
                INSERT INTO route_shares 
                (share_id, route_id, shared_by, share_code, expires_at, is_public)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"share_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                route_id, shared_by, share_code,
                expires_at.isoformat(), is_public
            ))
        
        return {
            "share_code": share_code,
            "share_url": f"/shared-route/{share_code}",
            "expires_at": expires_at.isoformat()
        }
    
    async def get_shared_route(self, share_code: str) -> Optional[Dict[str, Any]]:
        """Get a route by share code"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute('''
                SELECT rs.route_id, rs.shared_by, rs.expires_at, sr.route_data
                FROM route_shares rs
                JOIN saved_routes sr ON rs.route_id = sr.route_id
                WHERE rs.share_code = ? AND rs.expires_at > CURRENT_TIMESTAMP
            ''', (share_code,))
            
            row = cursor.fetchone()
            if row:
                # Update access count
                conn.execute(
                    'UPDATE route_shares SET access_count = access_count + 1 WHERE share_code = ?',
                    (share_code,)
                )
                
                return {
                    "route_id": row[0],
                    "shared_by": row[1],
                    "expires_at": row[2],
                    "route_data": json.loads(row[3])
                }
        
        return None


# Global service instances
user_profile_service = UserProfileService()
route_share_service = RouteShareService()
