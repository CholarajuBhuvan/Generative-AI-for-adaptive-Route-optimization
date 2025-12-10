"""
Main FastAPI application for Generative AI Route Optimization System
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.database import init_db
from app.services.websocket_manager import WebSocketManager
from app.services.ai_engine import AIEngine


# WebSocket manager instance
websocket_manager = WebSocketManager()
ai_engine = AIEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    await init_db()
    # Attach engine to app state for global access without circular imports
    app.state.ai_engine = ai_engine
    await app.state.ai_engine.initialize()
    print("🚀 AI Route Optimization System started!")
    
    yield
    
    # Shutdown
    await app.state.ai_engine.cleanup()
    print("🛑 AI Route Optimization System stopped!")


app = FastAPI(
    title="Generative AI Route Optimization",
    description="Advanced AI-powered route optimization system with adaptive learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Basic exception handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)  
async def general_exception_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Generative AI Route Optimization System",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "api_docs": "/docs",
            "route_optimization": "/api/v1/optimize-route",
            "traffic_data": "/api/v1/traffic-data",
            "analytics": "/api/v1/analytics",
            "websocket": "/ws/live-updates"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_engine": await ai_engine.get_status(),
        "websocket_connections": len(websocket_manager.active_connections)
    }


@app.websocket("/ws/live-updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            await websocket_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Advanced AI Route optimization dashboard"""
    try:
        with open("static/dashboard_enhanced.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Fallback to original dashboard
        with open("static/dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
