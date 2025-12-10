#!/usr/bin/env python3
"""
Comprehensive startup script for the Generative AI Route Optimization System
Includes system checks, dependency verification, and graceful startup
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
import logging

def setup_logging():
    """Setup logging configuration"""
    # Ensure stdout uses UTF-8 to avoid UnicodeEncodeError on Windows consoles
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('startup.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    logger = logging.getLogger(__name__)
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"✓ Python version: {sys.version}")
    return True


def check_dependencies():
    """Check if required dependencies are installed"""
    logger = logging.getLogger(__name__)
    
    # Core required packages
    core_packages = [
        'fastapi', 'uvicorn', 'numpy', 'requests', 'pydantic', 'sqlalchemy'
    ]
    
    # Optional ML packages
    optional_packages = [
        'pandas', 'torch', 'transformers', 'sklearn'
    ]
    
    missing_core = []
    missing_optional = []
    
    # Check core packages
    for package in core_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_core.append(package)
            logger.error(f"✗ {package} is missing (REQUIRED)")
    
    # Check optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} is installed (optional)")
        except ImportError:
            missing_optional.append(package)
            logger.warning(f"⚠ {package} is missing (optional - advanced ML features disabled)")
    
    if missing_core:
        logger.error(f"Missing REQUIRED packages: {', '.join(missing_core)}")
        logger.info("Install with: pip install fastapi uvicorn pydantic sqlalchemy requests numpy")
        return False
    
    if missing_optional:
        logger.info("Optional ML packages not installed. Basic functionality will work.")
        logger.info("For full AI features, install: pip install pandas torch transformers scikit-learn")
    
    return True


def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    directories = [
        'models', 'logs', 'static', 'learning_data', 'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Directory '{directory}' ready")
    
    return True


def check_environment():
    """Check environment configuration"""
    logger = logging.getLogger(__name__)
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        logger.warning("⚠ .env file not found, using defaults")
        logger.info("Copy env.example to .env and configure your settings")
    
    # Check API keys (optional)
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI API (optional)',
        'GOOGLE_MAPS_API_KEY': 'Google Maps API (optional)',
        'MAPBOX_API_KEY': 'Mapbox API (optional)'
    }
    
    for key, description in api_keys.items():
        if os.getenv(key):
            logger.info(f"✓ {description} configured")
        else:
            logger.warning(f"⚠ {description} not configured (optional)")


def start_server():
    """Start the FastAPI server"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Generative AI Route Optimization System...")
        
        # Import and start the application
        from app.main import app
        import uvicorn
        from app.core.config import settings
        
        logger.info(f"📍 Server will be available at: http://{settings.host}:{settings.port}")
        logger.info(f"📊 API Documentation: http://{settings.host}:{settings.port}/docs")
        logger.info(f"🌐 Web Dashboard: http://{settings.host}:{settings.port}/dashboard")
        logger.info(f"💚 Health Check: http://{settings.host}:{settings.port}/health")
        logger.info("-" * 60)
        
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info" if not settings.debug else "debug"
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        return False
    
    return True


def main():
    """Main startup function"""
    logger = setup_logging()
    
    logger.info("🔍 Generative AI Route Optimization System - Startup Check")
    logger.info("=" * 60)
    
    # Run startup checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", create_directories),
        ("Environment", check_environment)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        logger.info(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    if not all_passed:
        logger.error("\n❌ Startup checks failed. Please fix the issues above.")
        sys.exit(1)
    
    logger.info("\n✅ All startup checks passed!")
    logger.info("🚀 Ready to start the server...\n")
    
    # Start the server
    if not start_server():
        sys.exit(1)

if __name__ == "__main__":
    main()
