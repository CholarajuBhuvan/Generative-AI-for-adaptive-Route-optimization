"""
API Routes for Generative AI Route Optimization System
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio

from fastapi import Request
from app.models.route_models import RouteRequest, RouteResult
from app.services.traffic_service import traffic_service
from app.services.learning_engine import learning_engine, UserFeedback, RoutePerformance
from app.services.websocket_manager import websocket_manager
from app.services.geocoding_service import geocoding_service
from app.services.weather_service import weather_service
from app.services.carbon_calculator import carbon_calculator
from app.services.cost_calculator import cost_calculator_service
from app.services.multi_stop_routing import multi_stop_service
from app.services.user_profiles import user_profile_service, route_share_service
from app.core.config import settings


router = APIRouter()


# Pydantic models for request/response validation
class RouteOptimizationRequest(BaseModel):
    """Request model for route optimization"""
    start_lat: Optional[float] = Field(None, ge=-90, le=90, description="Start latitude")
    start_lng: Optional[float] = Field(None, ge=-180, le=180, description="Start longitude")
    end_lat: Optional[float] = Field(None, ge=-90, le=90, description="End latitude")
    end_lng: Optional[float] = Field(None, ge=-180, le=180, description="End longitude")
    start_location_name: Optional[str] = Field(None, description="Start location name (alternative to coordinates)")
    end_location_name: Optional[str] = Field(None, description="End location name (alternative to coordinates)")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Route constraints")
    user_preferences: Optional[Dict[str, float]] = Field(default_factory=dict, description="User preference weights")
    travel_mode: str = Field(default="driving", description="Travel mode (driving, walking, cycling, transit)")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type for carbon calculation")
    departure_time: Optional[datetime] = Field(default=None, description="Departure time")
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")


class RouteOptimizationResponse(BaseModel):
    """Response model for route optimization"""
    route_id: str
    coordinates: List[Dict[str, Any]]
    total_distance_km: float
    total_time_minutes: float
    total_cost_inr: float
    confidence_score: float
    ai_model_used: str
    traffic_analysis: Dict[str, Any]
    weather_analysis: Optional[Dict[str, Any]] = None
    carbon_footprint: Optional[Dict[str, Any]] = None
    alternatives: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    start_location_info: Optional[Dict[str, Any]] = None
    end_location_info: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    route_id: str
    user_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_type: str = Field(..., description="Type of feedback")
    comments: Optional[str] = Field(default=None, description="Additional comments")


class TrafficDataRequest(BaseModel):
    """Request model for traffic data"""
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(default=5.0, ge=0.1, le=50.0)


class RoutePerformanceRequest(BaseModel):
    """Request model for route performance data"""
    route_id: str
    predicted_time: float
    actual_time: float
    predicted_distance: float
    actual_distance: float
    traffic_accuracy: float
    user_satisfaction: float


# Dependency to get current user (simplified for demo)
def get_current_user(user_id: Optional[str] = None) -> str:
    """Get current user ID (simplified for demo)"""
    return user_id or "anonymous_user"


# Route optimization endpoints
@router.post("/optimize-route", response_model=RouteOptimizationResponse)
async def optimize_route(
    request: RouteOptimizationRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    request_obj: Request = None
):
    """
    Optimize route using AI models with geocoding, weather, and carbon footprint analysis
    
    This endpoint uses multiple AI models (Transformer, RL Agent, Genetic Algorithm)
    to generate optimal routes based on real-time traffic data and user preferences.
    Supports both coordinate-based and location name-based input.
    """
    try:
        # Handle geocoding if location names are provided
        start_lat, start_lng = request.start_lat, request.start_lng
        end_lat, end_lng = request.end_lat, request.end_lng
        start_location_info = None
        end_location_info = None
        
        if request.start_location_name:
            try:
                geocoded_start = await asyncio.wait_for(
                    geocoding_service.geocode(request.start_location_name, limit=1),
                    timeout=5.0
                )
                if geocoded_start:
                    start_lat = geocoded_start[0].latitude
                    start_lng = geocoded_start[0].longitude
                    start_location_info = {
                        'name': geocoded_start[0].name,
                        'display_name': geocoded_start[0].display_name,
                        'address': geocoded_start[0].address
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Could not geocode start location: {request.start_location_name}")
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail=f"Geocoding timeout for start location: {request.start_location_name}")
        
        if request.end_location_name:
            try:
                geocoded_end = await asyncio.wait_for(
                    geocoding_service.geocode(request.end_location_name, limit=1),
                    timeout=5.0
                )
                if geocoded_end:
                    end_lat = geocoded_end[0].latitude
                    end_lng = geocoded_end[0].longitude
                    end_location_info = {
                        'name': geocoded_end[0].name,
                        'display_name': geocoded_end[0].display_name,
                        'address': geocoded_end[0].address
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Could not geocode end location: {request.end_location_name}")
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail=f"Geocoding timeout for end location: {request.end_location_name}")
        
        # Validate that we have coordinates
        if start_lat is None or start_lng is None or end_lat is None or end_lng is None:
            raise HTTPException(status_code=400, detail="Must provide either coordinates or location names for start and end points")
        
        # Reverse geocode if we only have coordinates (with timeout)
        if not start_location_info:
            try:
                reverse_start = await asyncio.wait_for(
                    geocoding_service.reverse_geocode(start_lat, start_lng),
                    timeout=3.0
                )
                if reverse_start:
                    start_location_info = {
                        'name': reverse_start.name,
                        'display_name': reverse_start.display_name,
                        'address': reverse_start.address
                    }
            except (asyncio.TimeoutError, Exception) as e:
                logging.warning(f"Reverse geocoding start failed: {e}")
                start_location_info = {'name': f"Location ({start_lat:.4f}, {start_lng:.4f})"}
        
        if not end_location_info:
            try:
                reverse_end = await asyncio.wait_for(
                    geocoding_service.reverse_geocode(end_lat, end_lng),
                    timeout=3.0
                )
                if reverse_end:
                    end_location_info = {
                        'name': reverse_end.name,
                        'display_name': reverse_end.display_name,
                        'address': reverse_end.address
                    }
            except (asyncio.TimeoutError, Exception) as e:
                logging.warning(f"Reverse geocoding end failed: {e}")
                end_location_info = {'name': f"Location ({end_lat:.4f}, {end_lng:.4f})"}
        
        # Create route request
        route_request = RouteRequest(
            start_point=(start_lat, start_lng),
            end_point=(end_lat, end_lng),
            constraints=request.constraints,
            user_preferences=request.user_preferences,
            travel_mode=request.travel_mode,
            departure_time=request.departure_time
        )
        
        # Get user preferences from learning engine
        learned_preferences = await learning_engine.get_user_preferences(user_id)
        if not route_request.user_preferences:
            route_request.user_preferences = learned_preferences
        
        # Optimize route using AI engine
        engine = request_obj.app.state.ai_engine
        result = await engine.optimize_route(route_request)
        
        # Get weather analysis for the route (with timeout)
        weather_analysis = None
        try:
            route_points = [(coord['lat'], coord['lng']) for coord in result.coordinates]
            weather_analysis = await asyncio.wait_for(
                weather_service.get_route_weather_analysis(route_points),
                timeout=2.0
            )
        except (asyncio.TimeoutError, Exception) as e:
            logging.warning(f"Weather analysis failed: {e}")
            weather_analysis = None
        
        # Calculate carbon footprint
        carbon_footprint = None
        try:
            carbon_footprint = carbon_calculator.calculate_route_environmental_impact(
                distance_km=result.total_distance_km,
                time_minutes=result.total_time_minutes,
                travel_mode=request.travel_mode,
                vehicle_type=request.vehicle_type
            )
        except Exception as e:
            logging.warning(f"Carbon footprint calculation failed: {e}")
        
        # Enhanced cost calculation using the dedicated cost calculator service
        try:
            detailed_cost = cost_calculator_service.calculate_route_cost(
                distance_km=result.total_distance_km,
                vehicle_type=request.vehicle_type or "car_petrol",
                include_tolls=True,
                route_type="mixed"  # Assume mixed route type
            )
            total_cost_inr = detailed_cost["total_cost_inr"]
        except Exception as e:
            logging.warning(f"Detailed cost calculation failed: {e}")
            # Fallback: Use AI model cost if available, otherwise calculate basic cost
            total_cost_inr = result.total_cost if result.total_cost > 0 else result.total_distance_km * 8.5
        
        # Schedule background tasks
        background_tasks.add_task(
            process_route_analytics, 
            result, 
            user_id
        )
        
        # Broadcast route update via WebSocket
        background_tasks.add_task(
            broadcast_route_update,
            result
        )
        
        # Convert to response model
        response = RouteOptimizationResponse(
            route_id=result.route_id,
            coordinates=result.coordinates,
            total_distance_km=result.total_distance_km,
            total_time_minutes=result.total_time_minutes,
            total_cost_inr=round(total_cost_inr, 2),
            confidence_score=result.confidence_score,
            ai_model_used=result.ai_model_used,
            traffic_analysis=result.traffic_analysis,
            weather_analysis=weather_analysis,
            carbon_footprint=carbon_footprint,
            alternatives=result.alternatives,
            metadata=result.metadata,
            start_location_info=start_location_info,
            end_location_info=end_location_info
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error optimizing route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traffic-data")
async def get_traffic_data(
    lat: float,
    lng: float,
    radius_km: float = 5.0
):
    """
    Get real-time traffic data for a location
    
    Returns comprehensive traffic information including flow data,
    incidents, and predictions.
    """
    try:
        traffic_data = await traffic_service.get_comprehensive_traffic_data(
            (lat, lng), radius_km
        )
        
        return JSONResponse(content=traffic_data)
        
    except Exception as e:
        logging.error(f"Error fetching traffic data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit user feedback for route optimization learning
    
    This feedback is used to improve AI models and personalize
    route recommendations.
    """
    try:
        # Create feedback object
        user_feedback = UserFeedback(
            route_id=feedback.route_id,
            user_id=feedback.user_id,
            rating=feedback.rating,
            feedback_type=feedback.feedback_type,
            comments=feedback.comments
        )
        
        # Process feedback in background
        background_tasks.add_task(
            learning_engine.process_user_feedback,
            user_feedback
        )
        
        return {"message": "Feedback submitted successfully", "feedback_id": feedback.route_id}
        
    except Exception as e:
        logging.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route-performance")
async def submit_route_performance(
    performance: RoutePerformanceRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit route performance data for model improvement
    
    This data is used to improve prediction accuracy and
    optimize AI models.
    """
    try:
        # Create performance object
        route_performance = RoutePerformance(
            route_id=performance.route_id,
            predicted_time=performance.predicted_time,
            actual_time=performance.actual_time,
            predicted_distance=performance.predicted_distance,
            actual_distance=performance.actual_distance,
            traffic_accuracy=performance.traffic_accuracy,
            user_satisfaction=performance.user_satisfaction
        )
        
        # Process performance data in background
        background_tasks.add_task(
            learning_engine.process_route_performance,
            route_performance
        )
        
        return {"message": "Performance data submitted successfully"}
        
    except Exception as e:
        logging.error(f"Error processing performance data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_analytics(request: Request):
    """
    Get comprehensive system analytics
    
    Returns performance metrics, learning insights, and system statistics.
    """
    try:
        # Get AI engine analytics
        ai_analytics = await request.app.state.ai_engine.get_performance_analytics()
        
        # Get learning engine analytics
        learning_analytics = await learning_engine.get_learning_analytics()
        
        # Get traffic service stats
        traffic_stats = traffic_service.get_cache_stats()
        
        # Get WebSocket stats
        websocket_stats = websocket_manager.get_connection_stats()
        
        return {
            "ai_engine": ai_analytics,
            "learning_engine": learning_analytics,
            "traffic_service": traffic_stats,
            "websocket_connections": websocket_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """
    Get learned preferences for a specific user
    
    Returns the AI-learned preferences based on user feedback and behavior.
    """
    try:
        preferences = await learning_engine.get_user_preferences(user_id)
        
        return {
            "user_id": user_id,
            "preferences": preferences,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error fetching user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user-preferences/{user_id}/reset")
async def reset_user_preferences(user_id: str):
    """
    Reset learning data for a specific user
    
    Clears all learned preferences and feedback history for the user.
    """
    try:
        await learning_engine.reset_learning_data(user_id)
        
        return {"message": f"Learning data reset for user {user_id}"}
        
    except Exception as e:
        logging.error(f"Error resetting user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traffic-prediction")
async def get_traffic_prediction(
    lat: float,
    lng: float,
    time_horizon_minutes: int = 60
):
    """
    Get traffic prediction for a location
    
    Uses learned patterns to predict future traffic conditions.
    """
    try:
        prediction_time = datetime.now() + timedelta(minutes=time_horizon_minutes)
        predicted_level, confidence = await learning_engine.predict_traffic(
            (lat, lng), prediction_time
        )
        
        return {
            "location": {"lat": lat, "lng": lng},
            "prediction_time": prediction_time.isoformat(),
            "predicted_traffic_level": predicted_level,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error generating traffic prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/enable")
async def enable_learning():
    """Enable adaptive learning"""
    try:
        await learning_engine.enable_learning()
        return {"message": "Adaptive learning enabled"}
    except Exception as e:
        logging.error(f"Error enabling learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/disable")
async def disable_learning():
    """Disable adaptive learning"""
    try:
        await learning_engine.disable_learning()
        return {"message": "Adaptive learning disabled"}
    except Exception as e:
        logging.error(f"Error disabling learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def get_models_status(request: Request):
    """Get status of all AI models"""
    try:
        status = await request.app.state.ai_engine.get_status()
        return status
    except Exception as e:
        logging.error(f"Error fetching models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(request: Request):
    """Detailed health check for all system components"""
    try:
        # Check AI engine
        ai_status = await request.app.state.ai_engine.get_status()
        
        # Check traffic service
        traffic_stats = traffic_service.get_cache_stats()
        
        # Check WebSocket connections
        websocket_stats = websocket_manager.get_connection_stats()
        
        # Overall health
        overall_health = (
            ai_status['initialized'] and
            len(traffic_stats['providers_count']) > 0
        )
        
        return {
            "status": "healthy" if overall_health else "degraded",
            "ai_engine": ai_status,
            "traffic_service": traffic_stats,
            "websocket_manager": websocket_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Background task functions
async def process_route_analytics(result: RouteResult, user_id: str):
    """Process route analytics in background"""
    try:
        # Log route generation
        logging.info(f"Route {result.route_id} generated for user {user_id} using {result.ai_model_used}")
        
        # Update performance metrics
        # This would typically involve more sophisticated analytics
        await asyncio.sleep(0.1)  # Simulate processing
        
    except Exception as e:
        logging.error(f"Error processing route analytics: {e}")


async def broadcast_route_update(result: RouteResult):
    """Broadcast route update via WebSocket"""
    try:
        message = {
            "type": "route_update",
            "route_id": result.route_id,
            "model_used": result.ai_model_used,
            "confidence": result.confidence_score,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast_to_topic(message, "route_updates")
        
    except Exception as e:
        logging.error(f"Error broadcasting route update: {e}")


# New Advanced Endpoints

@router.get("/geocode")
async def geocode_location(query: str, limit: int = 5):
    """
    Geocode a location name to coordinates
    
    Args:
        query: Location name to search for
        limit: Maximum number of results
        
    Returns:
        List of geocoded locations with coordinates
    """
    try:
        results = await geocoding_service.geocode(query, limit=limit)
        
        return {
            "query": query,
            "results": [
                {
                    "name": r.name,
                    "display_name": r.display_name,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "address": r.address,
                    "type": r.place_type,
                    "importance": r.importance
                }
                for r in results
            ],
            "count": len(results)
        }
    except Exception as e:
        logging.error(f"Geocoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reverse-geocode")
async def reverse_geocode_location(lat: float, lng: float):
    """
    Reverse geocode coordinates to location name
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Location information
    """
    try:
        result = await geocoding_service.reverse_geocode(lat, lng)
        
        if result:
            return {
                "name": result.name,
                "display_name": result.display_name,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "address": result.address,
                "type": result.place_type
            }
        else:
            raise HTTPException(status_code=404, detail="Location not found")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Reverse geocoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weather")
async def get_weather(lat: float, lng: float):
    """
    Get current weather conditions for a location
    
    Args:
        lat: Latitude
        lng: Longitude
        
    Returns:
        Current weather data
    """
    try:
        weather = await weather_service.get_current_weather(lat, lng)
        
        if weather:
            return {
                "location": {"lat": weather.location[0], "lng": weather.location[1]},
                "temperature_celsius": weather.temperature_celsius,
                "feels_like": weather.feels_like_celsius,
                "humidity": weather.humidity,
                "wind_speed_kmh": weather.wind_speed_kmh,
                "visibility_km": weather.visibility_km,
                "precipitation_mm": weather.precipitation_mm,
                "weather": weather.weather_main,
                "description": weather.weather_description,
                "timestamp": weather.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Weather data not available")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Weather fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weather-forecast")
async def get_weather_forecast_endpoint(lat: float, lng: float, hours: int = 24):
    """
    Get weather forecast for a location
    
    Args:
        lat: Latitude
        lng: Longitude
        hours: Number of hours to forecast
        
    Returns:
        Weather forecast data
    """
    try:
        forecast = await weather_service.get_weather_forecast(lat, lng, hours)
        
        return {
            "location": {"lat": lat, "lng": lng},
            "forecast": [
                {
                    "time": f.forecast_time.isoformat(),
                    "temperature": f.temperature_celsius,
                    "precipitation_probability": f.precipitation_probability,
                    "precipitation_mm": f.precipitation_mm,
                    "wind_speed_kmh": f.wind_speed_kmh,
                    "weather": f.weather_main,
                    "description": f.weather_description
                }
                for f in forecast
            ],
            "count": len(forecast)
        }
    except Exception as e:
        logging.error(f"Weather forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/carbon-footprint")
async def calculate_carbon_footprint(
    distance_km: float,
    travel_mode: str = "driving",
    vehicle_type: Optional[str] = None
):
    """
    Calculate carbon footprint for a route
    
    Args:
        distance_km: Distance in kilometers
        travel_mode: Mode of travel
        vehicle_type: Specific vehicle type
        
    Returns:
        Carbon emission data
    """
    try:
        emission = carbon_calculator.calculate_emissions(distance_km, travel_mode, vehicle_type)
        
        return {
            "distance_km": distance_km,
            "travel_mode": travel_mode,
            "vehicle_type": vehicle_type,
            "total_co2_kg": emission.total_co2_kg,
            "co2_per_km": emission.co2_per_km,
            "trees_to_offset": emission.trees_to_offset,
            "eco_score": emission.eco_score,
            "comparison_to_average": emission.comparison_to_average,
            "fuel_consumption_liters": emission.fuel_consumption_liters
        }
    except Exception as e:
        logging.error(f"Carbon calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/carbon-comparison")
async def compare_carbon_footprint(distance_km: float):
    """
    Compare carbon footprint across different travel modes
    
    Args:
        distance_km: Distance in kilometers
        
    Returns:
        Comparison of emissions for different modes
    """
    try:
        comparisons = carbon_calculator.compare_travel_modes(distance_km)
        
        return {
            "distance_km": distance_km,
            "comparisons": {
                mode: {
                    "total_co2_kg": emission.total_co2_kg,
                    "eco_score": emission.eco_score,
                    "trees_to_offset": emission.trees_to_offset
                }
                for mode, emission in comparisons.items()
            }
        }
    except Exception as e:
        logging.error(f"Carbon comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/location-suggestions")
async def get_location_suggestions(query: str, limit: int = 5):
    """
    Get enhanced location suggestions for autocomplete
    
    Args:
        query: Search query
        limit: Maximum number of suggestions
        
    Returns:
        List of location suggestions with enhanced data
    """
    try:
        suggestions = await geocoding_service.get_location_suggestions(query, limit)
        return {"suggestions": suggestions}
    except Exception as e:
        logging.error(f"Location suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/popular-destinations")
async def get_popular_destinations(country: str = "IN"):
    """
    Get popular destinations for quick selection
    
    Args:
        country: Country code (default: IN for India)
        
    Returns:
        List of popular destinations
    """
    try:
        destinations = await geocoding_service.get_popular_destinations(country)
        return {"destinations": destinations, "country": country}
    except Exception as e:
        logging.error(f"Popular destinations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Multi-Stop Routing Endpoints

@router.post("/multi-stop-route")
async def plan_multi_stop_route(
    start_location: str,
    end_location: str,
    waypoints: List[str],
    optimization_mode: str = "balanced",
    travel_mode: str = "driving",
    vehicle_type: str = "average"
):
    """
    Plan an optimized multi-stop route with waypoints
    
    Args:
        start_location: Starting location name
        end_location: Ending location name
        waypoints: List of waypoint location names
        optimization_mode: "time", "distance", "cost", or "balanced"
        travel_mode: Mode of travel
        vehicle_type: Vehicle type for carbon calculation
        
    Returns:
        Optimized multi-stop route
    """
    try:
        route = await multi_stop_service.plan_multi_stop_route(
            start_location=start_location,
            end_location=end_location,
            waypoint_locations=waypoints,
            optimization_mode=optimization_mode,
            travel_mode=travel_mode,
            vehicle_type=vehicle_type
        )
        
        return {
            "route_id": route.route_id,
            "waypoints": [
                {
                    "name": wp.name,
                    "latitude": wp.latitude,
                    "longitude": wp.longitude,
                    "stop_duration_minutes": wp.stop_duration_minutes,
                    "priority": wp.priority
                }
                for wp in route.waypoints
            ],
            "segments": route.segments,
            "total_distance_km": route.total_distance_km,
            "total_time_minutes": route.total_time_minutes,
            "total_cost_inr": route.total_cost_inr,
            "optimization_score": route.optimization_score,
            "carbon_footprint": route.carbon_footprint,
            "weather_impact": route.weather_impact,
            "insights": multi_stop_service.get_route_insights(route)
        }
    except Exception as e:
        logging.error(f"Multi-stop routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-stop-compare")
async def compare_multi_stop_options(
    start_location: str,
    end_location: str,
    waypoints: List[str] = []
):
    """
    Compare different optimization strategies for multi-stop routing
    
    Returns:
        Comparison of different optimization modes
    """
    try:
        comparisons = await multi_stop_service.compare_route_options(
            start_location, end_location, waypoints
        )
        
        return {
            "comparisons": {
                mode: {
                    "total_distance_km": route.total_distance_km,
                    "total_time_minutes": route.total_time_minutes,
                    "total_cost_inr": route.total_cost_inr,
                    "optimization_score": route.optimization_score,
                    "carbon_footprint_kg": route.carbon_footprint.get("primary_emission", {}).get("total_co2_kg", 0)
                }
                for mode, route in comparisons.items()
            }
        }
    except Exception as e:
        logging.error(f"Multi-stop comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# User Profile and Favorites Endpoints

@router.get("/user/{user_id}/preferences")
async def get_user_preferences(user_id: str):
    """Get user routing preferences"""
    try:
        preferences = await user_profile_service.get_or_create_user_preferences(user_id)
        return {
            "user_id": preferences.user_id,
            "preferred_travel_mode": preferences.preferred_travel_mode,
            "preferred_vehicle_type": preferences.preferred_vehicle_type,
            "optimization_weights": {
                "time_weight": preferences.time_weight,
                "distance_weight": preferences.distance_weight,
                "cost_weight": preferences.cost_weight,
                "traffic_weight": preferences.traffic_weight
            },
            "eco_friendly": preferences.eco_friendly,
            "avoid_tolls": preferences.avoid_tolls,
            "avoid_highways": preferences.avoid_highways,
            "notification_preferences": preferences.notification_preferences
        }
    except Exception as e:
        logging.error(f"User preferences error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/user/{user_id}/preferences")
async def update_user_preferences(user_id: str, updates: Dict[str, Any]):
    """Update user routing preferences"""
    try:
        preferences = await user_profile_service.update_user_preferences(user_id, updates)
        return {"message": "Preferences updated successfully", "user_id": user_id}
    except Exception as e:
        logging.error(f"Update preferences error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user/{user_id}/favorites")
async def save_favorite_route(
    user_id: str,
    route_name: str,
    route_data: Dict[str, Any],
    tags: List[str] = []
):
    """Save a route as favorite"""
    try:
        saved_route = await user_profile_service.save_favorite_route(
            user_id, route_name, route_data, tags
        )
        return {
            "message": "Route saved as favorite",
            "route_id": saved_route.route_id,
            "name": saved_route.name
        }
    except Exception as e:
        logging.error(f"Save favorite error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/favorites")
async def get_user_favorites(user_id: str):
    """Get user's favorite routes"""
    try:
        favorites = await user_profile_service.get_user_favorites(user_id)
        return {
            "favorites": [
                {
                    "route_id": route.route_id,
                    "name": route.name,
                    "start_location": route.start_location,
                    "end_location": route.end_location,
                    "tags": route.tags,
                    "created_at": route.created_at.isoformat(),
                    "last_used": route.last_used.isoformat() if route.last_used else None,
                    "use_count": route.use_count
                }
                for route in favorites
            ]
        }
    except Exception as e:
        logging.error(f"Get favorites error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/recent-routes")
async def get_recent_routes(user_id: str, limit: int = 10):
    """Get user's recent routes"""
    try:
        recent = await user_profile_service.get_recent_routes(user_id, limit)
        return {
            "recent_routes": [
                {
                    "route_id": route.route_id,
                    "name": route.name,
                    "start_location": route.start_location,
                    "end_location": route.end_location,
                    "last_used": route.last_used.isoformat() if route.last_used else None,
                    "use_count": route.use_count,
                    "is_favorite": route.is_favorite
                }
                for route in recent
            ]
        }
    except Exception as e:
        logging.error(f"Get recent routes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/suggestions")
async def get_personalized_suggestions(user_id: str):
    """Get personalized route suggestions based on user history"""
    try:
        suggestions = await user_profile_service.get_personalized_suggestions(user_id)
        return suggestions
    except Exception as e:
        logging.error(f"Suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/search-routes")
async def search_user_routes(user_id: str, query: str):
    """Search user's saved routes"""
    try:
        results = await user_profile_service.search_user_routes(user_id, query)
        return {
            "query": query,
            "results": [
                {
                    "route_id": route.route_id,
                    "name": route.name,
                    "start_location": route.start_location,
                    "end_location": route.end_location,
                    "tags": route.tags,
                    "is_favorite": route.is_favorite
                }
                for route in results
            ]
        }
    except Exception as e:
        logging.error(f"Search routes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Route Sharing Endpoints

@router.post("/routes/{route_id}/share")
async def share_route(
    route_id: str,
    shared_by: str,
    expires_hours: int = 168,
    is_public: bool = False
):
    """Create a shareable route link"""
    try:
        share_info = await route_share_service.create_shareable_route(
            route_id, shared_by, expires_hours, is_public
        )
        return share_info
    except Exception as e:
        logging.error(f"Share route error: {e}")
@router.get("/shared-route/{share_code}")
async def get_shared_route(share_code: str):
    """
    Get a shared route by share code
    
    Args:
        share_code: Unique share code for the route
        
    Returns:
        Shared route data
    """
    try:
        route_data = await route_share_service.get_shared_route(share_code)
        if not route_data:
            raise HTTPException(status_code=404, detail="Shared route not found")
        
        return {
            "success": True,
            "route": route_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get shared route error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cost Analysis Endpoints

@router.get("/cost-analysis")
async def get_detailed_cost_analysis(
    distance_km: float,
    vehicle_type: str = "car_petrol",
    include_tolls: bool = True,
    route_type: str = "mixed"
):
    """
    Get detailed cost analysis for a route
    
    Args:
        distance_km: Distance in kilometers
        vehicle_type: Type of vehicle
        include_tolls: Whether to include toll costs
        route_type: Type of route (highway, city, mixed)
        
    Returns:
        Detailed cost breakdown in Indian Rupees
    """
    try:
        cost_analysis = cost_calculator_service.calculate_route_cost(
            distance_km=distance_km,
            vehicle_type=vehicle_type,
            include_tolls=include_tolls,
            route_type=route_type
        )
        
        # Get cost comparison across vehicles
        cost_comparison = cost_calculator_service.get_cost_comparison(distance_km)
        
        # Get cost savings suggestions
        savings_suggestions = cost_calculator_service.get_cost_savings_suggestions(
            distance_km=distance_km,
            current_vehicle=vehicle_type
        )
        
        return {
            "success": True,
            "cost_analysis": cost_analysis,
            "cost_comparison": cost_comparison,
            "savings_suggestions": savings_suggestions,
            "currency": "INR"
        }
    except Exception as e:
        logging.error(f"Cost analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost-comparison")
async def get_cost_comparison_endpoint(distance_km: float):
    """
    Get cost comparison across different vehicle types
    
    Args:
        distance_km: Distance in kilometers
        
    Returns:
        Cost comparison for all vehicle types
    """
    try:
        comparison = cost_calculator_service.get_cost_comparison(distance_km)
        
        return {
            "success": True,
            "distance_km": distance_km,
            "cost_comparison": comparison,
            "currency": "INR",
            "note": "Costs are realistic estimates based on current Indian fuel prices and conditions"
        }
    except Exception as e:
        logging.error(f"Cost comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
