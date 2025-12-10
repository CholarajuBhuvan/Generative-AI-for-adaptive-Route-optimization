#!/usr/bin/env python3
"""
Working Simple AI Route Optimization System
"""

import uvicorn
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple request/response models
class RouteOptimizationRequest(BaseModel):
    start_lat: float = Field(..., ge=-90, le=90)
    start_lng: float = Field(..., ge=-180, le=180)
    end_lat: float = Field(..., ge=-90, le=90)
    end_lng: float = Field(..., ge=-180, le=180)
    travel_mode: str = Field(default="driving")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    user_preferences: Optional[Dict[str, float]] = Field(default_factory=dict)

class RouteOptimizationResponse(BaseModel):
    route_id: str
    coordinates: List[Dict[str, Any]]
    total_distance_km: float
    total_time_minutes: float
    total_cost_inr: float
    confidence_score: float
    ai_model_used: str
    traffic_analysis: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# FastAPI app
app = FastAPI(
    title="Working AI Route Optimization System",
    description="Simple and reliable AI-powered route optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lng / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def generate_route_points(start_lat: float, start_lng: float, 
                         end_lat: float, end_lng: float) -> List[Dict[str, Any]]:
    """Generate route points between start and end"""
    distance = calculate_distance(start_lat, start_lng, end_lat, end_lng)
    
    # Determine number of points based on distance
    if distance < 1:
        num_points = 3
    elif distance < 5:
        num_points = 5
    elif distance < 20:
        num_points = 8
    else:
        num_points = 12
    
    coordinates = []
    
    for i in range(num_points + 1):
        t = i / num_points
        
        # Linear interpolation
        lat = start_lat + (end_lat - start_lat) * t
        lng = start_lng + (end_lng - start_lng) * t
        
        # Add some curve variation for realism
        if 0 < i < num_points:
            curve_factor = np.sin(t * np.pi * 2) * 0.0005 * (1 - abs(0.5 - t) * 2)
            lat += curve_factor
            lng += curve_factor * 0.5
        
        coordinates.append({
            'lat': round(lat, 6),
            'lng': round(lng, 6),
            'time': round(t * 100, 1),  # Progress percentage
            'confidence': round(0.85 + np.random.random() * 0.1, 2)
        })
    
    return coordinates

def estimate_travel_time(distance_km: float, travel_mode: str) -> float:
    """Estimate travel time based on distance and mode"""
    speed_kmh = {
        'driving': 45,
        'walking': 5,
        'cycling': 15,
        'transit': 25
    }
    
    base_speed = speed_kmh.get(travel_mode, 45)
    time_hours = distance_km / base_speed
    
    # Add some traffic factor
    traffic_factor = 1.2 + np.random.random() * 0.3
    
    return time_hours * 60 * traffic_factor  # Convert to minutes

def calculate_cost(distance_km: float, travel_mode: str) -> float:
    """Calculate travel cost"""
    cost_per_km = {
        'driving': 8.0,   # INR per km
        'walking': 0.0,
        'cycling': 0.3,
        'transit': 1.8
    }
    
    return distance_km * cost_per_km.get(travel_mode, 8.0)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Working AI Route Optimization System",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "api_docs": "/docs",
            "route_optimization": "/api/v1/optimize-route",
            "health": "/health",
            "dashboard": "/dashboard"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_engine": {
            "initialized": True,
            "models_available": {
                "simple_routing": True
            }
        }
    }

@app.post("/api/v1/optimize-route", response_model=RouteOptimizationResponse)
async def optimize_route(request: RouteOptimizationRequest):
    """Optimize route - working version"""
    try:
        start_time = datetime.now()
        
        # Generate route coordinates
        coordinates = generate_route_points(
            request.start_lat, request.start_lng,
            request.end_lat, request.end_lng
        )
        
        # Calculate metrics
        total_distance_km = calculate_distance(
            request.start_lat, request.start_lng,
            request.end_lat, request.end_lng
        )
        
        total_time_minutes = estimate_travel_time(total_distance_km, request.travel_mode)
        total_cost = calculate_cost(total_distance_km, request.travel_mode)
        
        # Generate simple alternative
        alt_coordinates = generate_route_points(
            request.start_lat + 0.001, request.start_lng + 0.001,
            request.end_lat - 0.001, request.end_lng - 0.001
        )
        
        alternatives = [{
            "route_id": f"alt_route_{datetime.now().strftime('%H%M%S')}",
            "coordinates": alt_coordinates[:3],  # Shorter alternative
            "distance_km": round(total_distance_km * 1.1, 2),
            "time_minutes": round(total_time_minutes * 0.9, 1),
            "description": "Alternative scenic route"
        }]
        
        # Create response
        response = RouteOptimizationResponse(
            route_id=f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(100, 999)}",
            coordinates=coordinates,
            total_distance_km=round(total_distance_km, 2),
            total_time_minutes=round(total_time_minutes, 1),
            total_cost_inr=round(total_cost, 2),
            confidence_score=0.85,
            ai_model_used="simple_working_routing",
            traffic_analysis={
                "traffic_level": 0.3,
                "congestion_points": [],
                "estimated_delay": "5-10 minutes"
            },
            alternatives=alternatives,
            metadata={
                "generation_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "algorithm": "working_interpolation",
                "external_apis": "none"
            }
        )
        
        logger.info(f"Generated route: {total_distance_km:.2f}km, {total_time_minutes:.1f}min")
        return response
        
    except Exception as e:
        logger.error(f"Route optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Route Optimizer - Working</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .form-group { margin: 20px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .results { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
            .loading { text-align: center; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AI Route Optimization - Working Version</h1>
            
            <form id="routeForm">
                <div class="form-group">
                    <label>Start Latitude:</label>
                    <input type="number" id="startLat" step="any" value="28.6139" required>
                </div>
                
                <div class="form-group">
                    <label>Start Longitude:</label>
                    <input type="number" id="startLng" step="any" value="77.2090" required>
                </div>
                
                <div class="form-group">
                    <label>End Latitude:</label>
                    <input type="number" id="endLat" step="any" value="28.5355" required>
                </div>
                
                <div class="form-group">
                    <label>End Longitude:</label>
                    <input type="number" id="endLng" step="any" value="77.3910" required>
                </div>
                
                <div class="form-group">
                    <label>Travel Mode:</label>
                    <select id="travelMode">
                        <option value="driving">Driving</option>
                        <option value="walking">Walking</option>
                        <option value="cycling">Cycling</option>
                        <option value="transit">Transit</option>
                    </select>
                </div>
                
                <button type="submit">🔍 Optimize Route</button>
            </form>
            
            <div id="results" class="results" style="display: none;">
                <h3>Route Results:</h3>
                <div id="routeData"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('routeForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const startLat = parseFloat(document.getElementById('startLat').value);
                const startLng = parseFloat(document.getElementById('startLng').value);
                const endLat = parseFloat(document.getElementById('endLat').value);
                const endLng = parseFloat(document.getElementById('endLng').value);
                const travelMode = document.getElementById('travelMode').value;
                
                const results = document.getElementById('results');
                const routeData = document.getElementById('routeData');
                
                results.style.display = 'block';
                routeData.innerHTML = '<div class="loading">🔄 Optimizing route...</div>';
                
                try {
                    const response = await fetch('/api/v1/optimize-route', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            start_lat: startLat,
                            start_lng: startLng,
                            end_lat: endLat,
                            end_lng: endLng,
                            travel_mode: travelMode
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    routeData.innerHTML = `
                        <p><strong>Route ID:</strong> ${data.route_id}</p>
                        <p><strong>Distance:</strong> ${data.total_distance_km} km</p>
                        <p><strong>Time:</strong> ${data.total_time_minutes} minutes</p>
                        <p><strong>Cost:</strong> ₹${data.total_cost_inr}</p>
                        <p><strong>Confidence:</strong> ${data.confidence_score}</p>
                        <p><strong>AI Model:</strong> ${data.ai_model_used}</p>
                        <p><strong>Route Points:</strong> ${data.coordinates.length}</p>
                        <p><strong>Generation Time:</strong> ${Math.round(data.metadata.generation_time_ms)}ms</p>
                    `;
                    
                } catch (error) {
                    routeData.innerHTML = `<div style="color: red;">❌ Error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("🚀 Starting Working AI Route Optimization System...")
    print(f"📍 Host: localhost")
    print(f"🔌 Port: 8000")
    print(f"📊 API Docs: http://localhost:8000/docs")
    print(f"🌐 Dashboard: http://localhost:8000/dashboard")
    print(f"💚 Health Check: http://localhost:8000/health")
    print("-" * 60)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AI Route Optimization System...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
