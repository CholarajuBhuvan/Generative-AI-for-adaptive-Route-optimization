"""
Configuration settings for the AI Route Optimization System
"""

import os
from typing import Optional
try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    google_maps_api_key: Optional[str] = Field(None, env="GOOGLE_MAPS_API_KEY")
    mapbox_api_key: Optional[str] = Field(None, env="MAPBOX_API_KEY")
    weather_api_key: Optional[str] = Field(None, env="WEATHER_API_KEY")
    
    # Database
    database_url: str = Field("sqlite:///./route_optimizer.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Server
    debug: bool = Field(True, env="DEBUG")
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    secret_key: str = Field("your-secret-key-here", env="SECRET_KEY")
    
    # External Services
    traffic_api_url: str = Field("https://api.traffic-service.com", env="TRAFFIC_API_URL")
    
    # AI Model Configuration
    model_cache_size: int = Field(1000, env="MODEL_CACHE_SIZE")
    max_route_alternatives: int = Field(3, env="MAX_ROUTE_ALTERNATIVES")  # All 3 AI models (Transformer, RL, Genetic)
    optimization_timeout: int = Field(30, env="OPTIMIZATION_TIMEOUT")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("logs/app.log", env="LOG_FILE")
    
    # AI Model Settings (optimized for speed)
    transformer_model_path: str = "models/route_transformer"
    rl_model_path: str = "models/rl_agent"
    genetic_population_size: int = 10  # Reduced for faster generation
    genetic_generations: int = 5  # Reduced from 10 to 5 for speed
    genetic_mutation_rate: float = 0.15
    
    # Route Optimization Parameters
    max_route_distance_km: float = 1000.0
    max_route_time_minutes: int = 180
    traffic_weight: float = 0.4
    distance_weight: float = 0.3
    user_preference_weight: float = 0.3
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
