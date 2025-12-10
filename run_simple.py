#!/usr/bin/env python3
"""
Simplified entry point for the AI Route Optimization System
Uses minimal dependencies for reliable startup
"""

import uvicorn
import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.models.route_models import RouteRequest, RouteResult
from app.services.simple_fast_engine import SimpleFastEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the simple AI engine
ai_engine = SimpleFastEngine()

# FastAPI app
app = FastAPI(
    title="AI Route Optimization System (Simplified)",
    description="Fast and reliable AI-powered route optimization",
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API
class RouteOptimizationRequest(BaseModel):
    """Request model for route optimization"""
    start_lat: Optional[float] = Field(None, ge=-90, le=90, description="Start latitude")
    start_lng: Optional[float] = Field(None, ge=-180, le=180, description="Start longitude")
    end_lat: Optional[float] = Field(None, ge=-90, le=90, description="End latitude")
    end_lng: Optional[float] = Field(None, ge=-180, le=180, description="End longitude")
    start_location_name: Optional[str] = Field(None, description="Start location name")
    end_location_name: Optional[str] = Field(None, description="End location name")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    user_preferences: Optional[Dict[str, float]] = Field(default_factory=dict)
    travel_mode: str = Field(default="driving", description="Travel mode")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    departure_time: Optional[datetime] = Field(default=None)
    user_id: Optional[str] = Field(default=None)

@app.on_event("startup")
async def startup_event():
    """Initialize AI engine on startup"""
    try:
        await ai_engine.initialize()
        logger.info("✅ Simple AI Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Engine: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Route Optimization System (Simplified)",
        "version": "1.0.0",
        "status": "active",
        "engine": "simple_fast_engine",
        "endpoints": {
            "api_docs": "/docs",
            "dashboard": "/dashboard",
            "health": "/health",
            "optimize_route": "/api/v1/optimize-route"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = await ai_engine.get_status()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ai_engine": status
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Route optimization dashboard"""
    try:
        with open("static/dashboard_enhanced.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
            <h1>Dashboard Not Found</h1>
            <p>Please check if the dashboard file exists in the static directory.</p>
            <p><a href="/docs">Go to API Documentation</a></p>
        </body></html>
        """)

@app.post("/api/v1/optimize-route")
async def optimize_route(request: RouteOptimizationRequest):
    """Optimize route using the simple AI engine"""
    try:
        # Validate coordinates
        if not all([request.start_lat, request.start_lng, request.end_lat, request.end_lng]):
            raise HTTPException(status_code=400, detail="Start and end coordinates are required")
        
        # Create route request
        route_request = RouteRequest(
            start_point=(request.start_lat, request.start_lng),
            end_point=(request.end_lat, request.end_lng),
            constraints=request.constraints,
            user_preferences=request.user_preferences,
            travel_mode=request.travel_mode,
            departure_time=request.departure_time
        )
        
        # Optimize route
        result = await ai_engine.optimize_route(route_request)
        
        # Calculate basic cost (if not included)
        if result.total_cost == 0:
            cost_per_km = {'driving': 8.0, 'walking': 0.0, 'cycling': 0.3, 'transit': 1.8}
            result.total_cost = result.total_distance_km * cost_per_km.get(request.travel_mode, 8.0)
        
        # Return formatted response
        return {
            "route_id": result.route_id,
            "coordinates": result.coordinates,
            "total_distance_km": result.total_distance_km,
            "total_time_minutes": result.total_time_minutes,
            "total_cost_inr": round(result.total_cost, 2),
            "confidence_score": result.confidence_score,
            "ai_model_used": result.ai_model_used,
            "traffic_analysis": result.traffic_analysis,
            "alternatives": result.alternatives,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Route optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")

@app.get("/api/v1/health")
async def api_health():
    """API health check"""
    return await health_check()

@app.get("/test-route")
async def test_route():
    """Test route for debugging"""
    test_request = RouteRequest(
        start_point=(28.6139, 77.2090),  # Delhi
        end_point=(28.5355, 77.3910),   # Noida
        constraints={},
        user_preferences={},
        travel_mode="driving"
    )
    
    try:
        result = await ai_engine.optimize_route(test_request)
        return {
            "status": "success",
            "route": {
                "route_id": result.route_id,
                "distance_km": result.total_distance_km,
                "time_minutes": result.total_time_minutes,
                "points": len(result.coordinates),
                "model_used": result.ai_model_used
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Simplified AI Route Optimization System...")
    print(f"📍 Host: 0.0.0.0")
    print(f"🔌 Port: 8002")
    print(f"📊 API Docs: http://localhost:8002/docs")
    print(f"🌐 Dashboard: http://localhost:8002/dashboard")
    print(f"💚 Health Check: http://localhost:8002/health")
    print(f"🧪 Test Route: http://localhost:8002/test-route")
    print("-" * 60)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AI Route Optimization System...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
