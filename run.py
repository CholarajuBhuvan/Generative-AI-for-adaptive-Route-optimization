#!/usr/bin/env python3
"""
Main entry point for the Generative AI Route Optimization System
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings

if __name__ == "__main__":
    print("🚀 Starting Generative AI Route Optimization System...")
    print(f"📍 Host: {settings.host}")
    print(f"🔌 Port: {settings.port}")
    print(f"🐛 Debug Mode: {settings.debug}")
    print(f"📊 API Docs: http://{settings.host}:{settings.port}/docs")
    print(f"🌐 Dashboard: http://{settings.host}:{settings.port}/dashboard")
    print(f"💚 Health Check: http://{settings.host}:{settings.port}/health")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info" if not settings.debug else "debug",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AI Route Optimization System...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
