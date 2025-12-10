# Deployment Guide

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for API services

### Required Services
- **PostgreSQL** (optional, SQLite default)
- **Redis** (optional, for caching)
- **Web Server** (Nginx, Apache - for production)

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd GENERAT
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp env.example .env
# Edit .env with your configuration
```

### 5. Initialize Database
```bash
# Database tables will be created automatically on first run
```

### 6. Start the Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Access the Application
- **Web Dashboard**: http://localhost:8000/dashboard
- **API Documentation**: http://locdocsalhost:8000/
- **Health Check**: http://localhost:8000/health

## Production Deployment

### Docker Deployment

#### 1. Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Create docker-compose.yml
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/route_optimizer
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=route_optimizer
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app

volumes:
  postgres_data:
  redis_data:
```

#### 3. Deploy with Docker Compose
```bash
docker-compose up -d
```

### Kubernetes Deployment

#### 1. Create namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: route-optimizer
```

#### 2. Create ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: route-optimizer-config
  namespace: route-optimizer
data:
  DATABASE_URL: "postgresql://user:password@postgres:5432/route_optimizer"
  REDIS_URL: "redis://redis:6379"
  DEBUG: "False"
```

#### 3. Create Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: route-optimizer
  namespace: route-optimizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: route-optimizer
  template:
    metadata:
      labels:
        app: route-optimizer
    spec:
      containers:
      - name: app
        image: route-optimizer:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: route-optimizer-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### 4. Create Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: route-optimizer-service
  namespace: route-optimizer
spec:
  selector:
    app: route-optimizer
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Nginx Configuration

#### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws/ {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## Environment Configuration

### Production Environment Variables
```bash
# API Keys (Required)
OPENAI_API_KEY=your_openai_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
MAPBOX_API_KEY=your_mapbox_api_key

# Database
DATABASE_URL=postgresql://user:password@localhost/route_optimizer
REDIS_URL=redis://localhost:6379

# Server Configuration
DEBUG=False
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your_very_secure_secret_key_here

# External Services
TRAFFIC_API_URL=https://api.traffic-service.com
WEATHER_API_KEY=your_weather_api_key

# AI Model Configuration
MODEL_CACHE_SIZE=10000
MAX_ROUTE_ALTERNATIVES=10
OPTIMIZATION_TIMEOUT=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/route-optimizer/app.log
```

### API Keys Setup

#### Google Maps API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Maps JavaScript API and Roads API
4. Create API key and restrict it to your domain
5. Add to environment variables

#### OpenAI API
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create API key
3. Add to environment variables

#### Mapbox API
1. Go to [Mapbox](https://www.mapbox.com/)
2. Create account and get access token
3. Add to environment variables

## Monitoring and Logging

### Application Logs
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/route-optimizer/app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks
```bash
# Check application health
curl http://localhost:8000/health

# Check API endpoints
curl http://localhost:8000/api/v1/models/status

# Check database connection
curl http://localhost:8000/api/v1/analytics
```

### Performance Monitoring
```python
# Add to requirements.txt for monitoring
prometheus-client==0.16.0
grafana-api==1.0.3
```

## Security Considerations

### Production Security Checklist
- [ ] Use HTTPS in production
- [ ] Set strong SECRET_KEY
- [ ] Restrict API keys to specific domains/IPs
- [ ] Implement rate limiting
- [ ] Use environment variables for secrets
- [ ] Enable CORS properly
- [ ] Validate all inputs
- [ ] Use database connection pooling
- [ ] Implement proper error handling
- [ ] Set up firewall rules

### SSL/TLS Configuration
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # Rest of configuration...
}
```

## Backup and Recovery

### Database Backup
```bash
# PostgreSQL backup
pg_dump -h localhost -U user route_optimizer > backup.sql

# Restore
psql -h localhost -U user route_optimizer < backup.sql
```

### Model Backup
```bash
# Backup trained models
tar -czf models_backup.tar.gz models/

# Restore models
tar -xzf models_backup.tar.gz
```

## Scaling Considerations

### Horizontal Scaling
- Use load balancer (Nginx, HAProxy)
- Multiple application instances
- Database read replicas
- Redis cluster for caching

### Vertical Scaling
- Increase CPU/memory for AI model processing
- Use GPU acceleration for deep learning models
- Optimize database queries
- Implement connection pooling

### Performance Optimization
```python
# Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30
)

# Caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_traffic_data(location):
    # Cached traffic data retrieval
    pass
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Check model files exist
ls -la models/

# Verify model permissions
chmod 644 models/*.pt
```

#### 2. Database Connection Issues
```bash
# Test database connection
psql -h localhost -U user -d route_optimizer -c "SELECT 1;"

# Check connection string format
echo $DATABASE_URL
```

#### 3. API Key Issues
```bash
# Test API keys
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

#### 4. Memory Issues
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Log Analysis
```bash
# Check application logs
tail -f /var/log/route-optimizer/app.log

# Check error logs
grep ERROR /var/log/route-optimizer/app.log

# Monitor performance
grep "optimize-route" /var/log/route-optimizer/app.log | tail -100
```

## Maintenance

### Regular Tasks
1. **Daily**: Monitor logs and performance metrics
2. **Weekly**: Update dependencies and security patches
3. **Monthly**: Backup database and models
4. **Quarterly**: Review and optimize AI model performance

### Update Procedure
```bash
# 1. Backup current deployment
docker-compose down
cp -r /app /app.backup

# 2. Pull latest changes
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt

# 4. Run migrations (if any)
alembic upgrade head

# 5. Restart services
docker-compose up -d

# 6. Verify deployment
curl http://localhost:8000/health
```
