# Docker Setup Instructions

## Prerequisites
- Docker installed and running
- Docker Compose installed

## Build and Run Commands

### Option 1: Using Docker directly
```bash
# Build the image
docker build -t stock-market-ai .

# Run the container
docker run -p 8501:8501 stock-market-ai
```

### Option 2: Using Docker Compose (Recommended)
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your custom settings (optional)
# nano .env

# Build and run with docker-compose
docker-compose up

# Or build and run in background
docker-compose up -d

### Development Setup
For development with live reloading:
```bash
# Copy development overrides
cp docker-compose.override.yml.example docker-compose.override.yml

# Run in development mode
docker-compose up
```

## Configuration
The application expects the following configuration files:
- `config/data_sources.yaml` - Data source configurations (created)
- `config/market_holidays.yaml` - Market holiday configurations (created)
- `.env` - Environment variables (copy from .env.example)

These files have been created with default configurations. You can modify them as needed.

## Database Configuration
The docker-compose setup includes an optional PostgreSQL database for caching and data persistence:

### PostgreSQL Service
- **Database**: stockmarket
- **User**: postgres
- **Password**: postgres
- **Port**: 5432 (exposed on host)
- **Volume**: postgres_data (persistent storage)

### Database Environment Variables
- `DB_HOST`: Database host (postgres)
- `DB_PORT`: Database port (5432)
- `DB_NAME`: Database name (stockmarket)
- `DB_USER`: Database user (postgres)
- `DB_PASSWORD`: Database password (postgres)

### Using PostgreSQL
The application will automatically use PostgreSQL if the database service is running. If PostgreSQL is not available, it will fall back to SQLite for local development.

## Health Checks
Both services include health checks:
- **PostgreSQL**: Checks database connectivity
- **Streamlit App**: Checks if the web interface is responding

The Streamlit app waits for PostgreSQL to be healthy before starting.

## Environment Variables
- `DATA_SOURCES_CONFIG`: Path to data sources configuration
- `MARKET_HOLIDAYS_CONFIG`: Path to market holidays configuration
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `DB_HOST`: Database host (postgres)
- `DB_PORT`: Database port (5432)
- `DB_NAME`: Database name (stockmarket)
- `DB_USER`: Database user (postgres)
- `DB_PASSWORD`: Database password (postgres)