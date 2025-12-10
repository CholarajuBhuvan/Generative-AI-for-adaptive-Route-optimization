"""
Minimal test server to verify basic functionality
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test AI Route Optimizer")

@app.get("/")
async def root():
    return {"message": "Test server is working!", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "test": True}

@app.get("/test-route")
async def test_route():
    return {
        "route_id": "test_123",
        "coordinates": [
            {"lat": 28.6139, "lng": 77.2090, "time": 0, "confidence": 0.9},
            {"lat": 28.5355, "lng": 77.3910, "time": 100, "confidence": 0.9}
        ],
        "total_distance_km": 25.5,
        "total_time_minutes": 45.0,
        "ai_model_used": "test_model"
    }

if __name__ == "__main__":
    print("🔧 Starting Test Server...")
    print("📍 Host: localhost")
    print("🔌 Port: 8001")
    print("💚 Health Check: http://localhost:8001/health")
    print("-" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
