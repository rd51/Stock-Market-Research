# Deployment Guide

This guide covers various deployment options for the Stock Market Analysis application, from cloud platforms to local production setups.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Heroku Deployment](#heroku-deployment)
- [AWS EC2/ECS Deployment](#aws-ec2ecs-deployment)
- [GCP Cloud Run Deployment](#gcp-cloud-run-deployment)
- [Local Production Setup](#local-production-setup)
- [Monitoring and Logging](#monitoring-and-logging)
- [Backup Procedures](#backup-procedures)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying, ensure you have:

### Application Requirements
- Python 3.10+
- PostgreSQL (optional, falls back to SQLite)
- All dependencies from `requirements.txt`

### System Requirements
- Docker and Docker Compose
- Git
- Cloud CLI tools (AWS CLI, gcloud, heroku CLI)

### Configuration Files
- `.env` file with environment variables
- `config/data_sources.yaml`
- `config/market_holidays.yaml`

## Heroku Deployment

### 1. Prerequisites
```bash
# Install Heroku CLI
npm install -g heroku
# or
curl https://cli-assets.heroku.com/install.sh | sh

# Login to Heroku
heroku login
```

### 2. Prepare Application
```bash
# Create Heroku app
heroku create your-stock-market-app

# Add PostgreSQL addon
heroku addons:create heroku-postgresql:hobby-dev

# Set environment variables
heroku config:set LOG_LEVEL=INFO
heroku config:set DATA_SOURCES_CONFIG=config/data_sources.yaml
heroku config:set MARKET_HOLIDAYS_CONFIG=config/market_holidays.yaml
```

### 3. Create Heroku Files

**Procfile:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**runtime.txt:**
```
python-3.10.12
```

### 4. Deploy
```bash
# Add files to git
git add .
git commit -m "Prepare for Heroku deployment"

# Deploy to Heroku
git push heroku main

# Open application
heroku open
```

### 5. Database Setup
```bash
# Run database migrations/initialization
heroku run python -c "from app import init_db; init_db()"
```

## AWS EC2/ECS Deployment

### Option 1: EC2 Instance

#### 1. Launch EC2 Instance
```bash
# Create EC2 instance (t3.medium recommended)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --count 1 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-groups sg-12345678 \
  --user-data file://ec2-userdata.sh
```

**ec2-userdata.sh:**
```bash
#!/bin/bash
yum update -y
yum install -y docker git
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install docker-compose
curl -L "https://github.com/docker/compose/releases/download/v2.17.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

#### 2. Configure Security Groups
- SSH (22) - Your IP
- HTTP (80) - 0.0.0.0/0
- HTTPS (443) - 0.0.0.0/0
- Custom TCP (8501) - 0.0.0.0/0 (Streamlit)

#### 3. Deploy Application
```bash
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Clone repository
git clone https://github.com/your-repo/stock-market-analysis.git
cd stock-market-analysis

# Copy environment file
cp .env.example .env
nano .env  # Edit with your settings

# Run with docker-compose
docker-compose up -d

# Setup nginx (optional)
sudo yum install -y nginx
sudo cp nginx.conf /etc/nginx/nginx.conf
sudo systemctl start nginx
sudo systemctl enable nginx
```

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Option 2: ECS Fargate

#### 1. Create ECR Repository
```bash
# Create repository
aws ecr create-repository --repository-name stock-market-app

# Get login token
aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account.dkr.ecr.your-region.amazonaws.com

# Build and push image
docker build -t stock-market-app .
docker tag stock-market-app:latest your-account.dkr.ecr.your-region.amazonaws.com/stock-market-app:latest
docker push your-account.dkr.ecr.your-region.amazonaws.com/stock-market-app:latest
```

#### 2. Create ECS Cluster and Services
```bash
# Create cluster
aws ecs create-cluster --cluster-name stock-market-cluster

# Create task definition (task-definition.json)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster stock-market-cluster \
  --service-name stock-market-service \
  --task-definition stock-market-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

**task-definition.json:**
```json
{
  "family": "stock-market-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "stock-market-app",
      "image": "your-account.dkr.ecr.your-region.amazonaws.com/stock-market-app:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8501,
          "hostPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "DB_HOST", "value": "your-rds-endpoint"},
        {"name": "DB_NAME", "value": "stockmarket"},
        {"name": "DB_USER", "value": "postgres"},
        {"name": "DB_PASSWORD", "value": "your-password"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/stock-market-app",
          "awslogs-region": "your-region",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 3. Setup RDS PostgreSQL
```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier stock-market-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username postgres \
  --master-user-password your-password \
  --allocated-storage 20
```

## GCP Cloud Run Deployment

### 1. Prerequisites
```bash
# Install Google Cloud SDK
# Download from: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project your-project-id
```

### 2. Build and Push to GCR
```bash
# Build image
docker build -t gcr.io/your-project/stock-market-app .

# Push to Google Container Registry
docker push gcr.io/your-project/stock-market-app
```

### 3. Deploy to Cloud Run
```bash
# Deploy
gcloud run deploy stock-market-app \
  --image gcr.io/your-project/stock-market-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars "LOG_LEVEL=INFO,DB_HOST=your-cloud-sql-ip" \
  --add-cloudsql-instances your-project:us-central1:stock-market-db
```

### 4. Setup Cloud SQL PostgreSQL
```bash
# Create Cloud SQL instance
gcloud sql instances create stock-market-db \
  --database-version POSTGRES_15 \
  --tier db-f1-micro \
  --region us-central1

# Create database
gcloud sql databases create stockmarket --instance stock-market-db

# Create user
gcloud sql users create postgres \
  --instance stock-market-db \
  --password your-password
```

### 5. Setup VPC Connector (for Cloud SQL)
```bash
# Create VPC connector
gcloud compute networks vpc-access connectors create stock-market-connector \
  --region us-central1 \
  --range 10.8.0.0/28
```

## Local Production Setup

### 1. Server Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3.10 python3.10-venv postgresql postgresql-contrib nginx certbot python3-certbot-nginx

# Install Docker (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2. Database Setup
```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
CREATE DATABASE stockmarket;
CREATE USER stockuser WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE stockmarket TO stockuser;
\q
```

### 3. Application Deployment
```bash
# Clone repository
git clone https://github.com/your-repo/stock-market-analysis.git
cd stock-market-analysis

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
nano .env  # Edit configuration

# Create data directories
mkdir -p data/cache data/logs

# Run database migrations
python -c "from app import init_db; init_db()"

# Start application with gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 app:server
```

### 4. Nginx Configuration
**nginx.conf:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files
    location /static {
        alias /path/to/your/app/static;
    }
}
```

### 5. SSL Certificate
```bash
# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Test renewal
sudo certbot renew --dry-run
```

### 6. Systemd Service
**systemd service file (/etc/systemd/system/stock-market.service):**
```ini
[Unit]
Description=Stock Market Analysis App
After=network.target postgresql.service

[Service]
User=your-user
Group=your-group
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/your/app/venv/bin"
ExecStart=/path/to/your/app/venv/bin/gunicorn --bind 127.0.0.1:8000 --workers 4 app:server
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable stock-market
sudo systemctl start stock-market
```

## Monitoring and Logging

### 1. Application Logging
```python
# In your app.py or logging configuration
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/app.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
```

### 2. System Monitoring

#### Prometheus + Grafana Setup
```bash
# Docker Compose for monitoring stack
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  node-exporter:
    image: prom/node-exporter
    ports:
      - "9100:9100"

volumes:
  grafana_data:
```

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'stock-market-app'
    static_configs:
      - targets: ['localhost:8501']
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 3. Health Checks
```python
# health.py
from flask import Flask, jsonify
import psycopg2
import os

app = Flask(__name__)

@app.route('/health')
def health_check():
    health_status = {
        'status': 'healthy',
        'checks': {}
    }

    # Database check
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        conn.close()
        health_status['checks']['database'] = 'healthy'
    except Exception as e:
        health_status['checks']['database'] = f'unhealthy: {str(e)}'
        health_status['status'] = 'unhealthy'

    # Application check
    try:
        # Add your application health checks here
        health_status['checks']['application'] = 'healthy'
    except Exception as e:
        health_status['checks']['application'] = f'unhealthy: {str(e)}'
        health_status['status'] = 'unhealthy'

    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code
```

### 4. Log Aggregation

#### ELK Stack Setup
```bash
# Docker Compose for ELK stack
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.6.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.6.0
    volumes:
      - ./monitoring/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"

  kibana:
    image: docker.elastic.co/kibana/kibana:8.6.0
    ports:
      - "5601:5601"
```

**logstash.conf:**
```conf
input {
  file {
    path => "/var/log/stock-market/*.log"
    start_position => "beginning"
  }
}

filter {
  grok {
    match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{WORD:logger} - %{WORD:level} - %{GREEDYDATA:message}" }
  }
  date {
    match => ["timestamp", "ISO8601"]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "stock-market-logs-%{+YYYY.MM.dd}"
  }
}
```

## Backup Procedures

### 1. Database Backups

#### PostgreSQL Backup Script
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="stockmarket"
DB_USER="postgres"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -U $DB_USER -h localhost $DB_NAME > $BACKUP_DIR/${DB_NAME}_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/${DB_NAME}_$DATE.sql

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/${DB_NAME}_$DATE.sql.gz"
```

#### Automated Backup with Cron
```bash
# Add to crontab (crontab -e)
# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh

# Weekly full backup on Sunday
0 3 * * 0 /path/to/backup.sh
```

### 2. Application Data Backup
```bash
#!/bin/bash
# app_backup.sh

BACKUP_DIR="/path/to/backups"
APP_DIR="/path/to/stock-market-analysis"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup application data
tar -czf $BACKUP_DIR/app_data_$DATE.tar.gz -C $APP_DIR data/

# Backup configuration
tar -czf $BACKUP_DIR/app_config_$DATE.tar.gz -C $APP_DIR config/

# Backup logs
tar -czf $BACKUP_DIR/app_logs_$DATE.tar.gz -C $APP_DIR logs/

# Clean old backups
find $BACKUP_DIR -name "app_*.tar.gz" -mtime +30 -delete

echo "Application backup completed"
```

### 3. Cloud Storage Backup

#### AWS S3 Backup
```bash
#!/bin/bash
# s3_backup.sh

BUCKET_NAME="your-backup-bucket"
BACKUP_DIR="/path/to/backups"

# Upload to S3
aws s3 sync $BACKUP_DIR s3://$BUCKET_NAME/ --delete

# Set lifecycle policy for automatic deletion
aws s3api put-bucket-lifecycle-configuration \
  --bucket $BUCKET_NAME \
  --lifecycle-configuration file://lifecycle.json
```

**lifecycle.json:**
```json
{
  "Rules": [
    {
      "ID": "Delete old backups",
      "Status": "Enabled",
      "Filter": {
        "Prefix": ""
      },
      "Expiration": {
        "Days": 30
      }
    }
  ]
}
```

#### Google Cloud Storage Backup
```bash
#!/bin/bash
# gcs_backup.sh

BUCKET_NAME="your-backup-bucket"
BACKUP_DIR="/path/to/backups"

# Upload to GCS
gsutil -m rsync -r $BACKUP_DIR gs://$BUCKET_NAME/

# Set lifecycle
gsutil lifecycle set lifecycle.json gs://$BUCKET_NAME/
```

### 4. Docker Volume Backup
```bash
#!/bin/bash
# docker_backup.sh

BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Stop containers
docker-compose down

# Backup volumes
docker run --rm -v stock-market-analysis_postgres_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/postgres_data_$DATE.tar.gz -C /data .

docker run --rm -v stock-market-analysis_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/app_data_$DATE.tar.gz -C /data .

# Start containers
docker-compose up -d

echo "Docker volumes backed up"
```

### 5. Restore Procedures

#### Database Restore
```bash
# Stop application
docker-compose down

# Drop and recreate database
psql -U postgres -c "DROP DATABASE IF EXISTS stockmarket;"
psql -U postgres -c "CREATE DATABASE stockmarket;"

# Restore from backup
gunzip -c backup_file.sql.gz | psql -U postgres -d stockmarket

# Start application
docker-compose up -d
```

#### Application Data Restore
```bash
# Stop application
docker-compose down

# Extract backup
tar -xzf app_data_backup.tar.gz -C /path/to/app/

# Start application
docker-compose up -d
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

#### 2. Database Connection Issues
```bash
# Test database connection
psql -h localhost -U postgres -d stockmarket

# Check PostgreSQL logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  streamlit_app:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

#### 4. SSL Certificate Issues
```bash
# Check certificate
openssl s_client -connect your-domain.com:443

# Renew certificate
sudo certbot renew

# Check nginx configuration
sudo nginx -t
sudo systemctl reload nginx
```

#### 5. Performance Issues
```bash
# Check application logs
docker-compose logs streamlit_app

# Profile application
python -m cProfile -s time app.py

# Database query optimization
EXPLAIN ANALYZE SELECT * FROM your_table;
```

### Health Check Endpoints
- Application: `http://your-domain.com/_stcore/health`
- Database: `http://your-domain.com/health/db`
- System: `http://your-domain.com/health/system`

### Log Locations
- Application logs: `logs/app.log`
- Docker logs: `docker-compose logs`
- System logs: `/var/log/syslog`
- Nginx logs: `/var/log/nginx/`

### 6. Model Retraining and Monitoring

The application includes an automated model retraining system:

#### Starting the Retraining Scheduler
```bash
# Linux/Mac
./manage_retraining.sh start

# Windows
manage_retraining.bat start

# Or directly with Python
python retraining_scheduler.py --start
```

#### Manual Retraining
```bash
# Retrain specific model
./manage_retraining.sh retrain random_forest

# Retrain all models
./manage_retraining.sh retrain-all
```

#### Checking Status
```bash
# Check scheduler status
./manage_retraining.sh status

# Check performance degradation
./manage_retraining.sh check-performance
```

#### Configuration
The retraining system is configured via `config/retraining_config.json`:
- **Weekly retraining**: Sunday at 2:00 AM IST
- **Performance monitoring**: Every 24 hours
- **Automatic retraining**: Triggered on 15%+ performance degradation
- **Model versioning**: Keeps last 10 versions per model

#### Model Versions
Retrained models are automatically versioned and stored in `models/versions/`.
Use the scheduler to rollback to previous versions if needed.
- [Nginx Documentation](https://nginx.org/en/docs/)</content>
<parameter name="filePath">c:\Users\RAKSHANDA\Downloads\reserach\Bipllab Sir\Stock Market Analysis\DEPLOYMENT.md