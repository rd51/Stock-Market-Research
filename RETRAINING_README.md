# Model Retraining Scheduler

Automated system for maintaining and improving machine learning model performance through scheduled retraining, performance monitoring, and automatic updates.

## Features

- **Scheduled Retraining**: Weekly automatic retraining of all models
- **Performance Monitoring**: Continuous tracking of model performance metrics
- **Automatic Retraining**: Triggers retraining when performance degrades
- **Model Versioning**: Maintains history of model versions with rollback capability
- **Comprehensive Logging**: Detailed logs of all retraining activities

## Quick Start

### 1. Make Scripts Executable (Linux/Mac)
```bash
chmod +x manage_retraining.sh
```

### 2. Start the Scheduler
```bash
# Linux/Mac
./manage_retraining.sh start

# Windows
manage_retraining.bat start

# Or directly
python retraining_scheduler.py --start
```

### 3. Check Status
```bash
./manage_retraining.sh status
```

## Manual Operations

### Retrain Specific Model
```bash
./manage_retraining.sh retrain random_forest
```

### Retrain All Models
```bash
./manage_retraining.sh retrain-all
```

### Check Performance
```bash
./manage_retraining.sh check-performance
```

## Configuration

Edit `config/retraining_config.json` to customize:

```json
{
  "schedule": {
    "weekly_retraining_day": "sunday",
    "weekly_retraining_time": "02:00"
  },
  "performance_thresholds": {
    "min_r2_score": 0.6,
    "max_mae_increase_pct": 15.0
  },
  "retraining_triggers": {
    "auto_retrain_enabled": true,
    "performance_check_interval_hours": 24
  }
}
```

## Available Models

- `random_forest` - Random Forest Regressor
- `xgboost` - XGBoost Regressor
- `lightgbm` - LightGBM Regressor
- `ensemble` - Ensemble of all models
- `ols_static` - Ordinary Least Squares

## Files and Directories

- `retraining_scheduler.py` - Main scheduler module
- `manage_retraining.sh` - Linux/Mac management script
- `manage_retraining.bat` - Windows management script
- `config/retraining_config.json` - Configuration file
- `models/versions/` - Model version history
- `data/logs/retraining_history.json` - Retraining activity log
- `data/logs/model_performance.json` - Performance metrics history

## Monitoring

The scheduler automatically:
- Monitors model performance every 24 hours
- Triggers retraining when R² drops below 0.6 or MAE increases by 15%+
- Logs all activities to `data/logs/retraining_scheduler.log`
- Maintains performance history for trend analysis

## Troubleshooting

### Scheduler Not Starting
```bash
# Check Python dependencies
pip install APScheduler

# Check configuration file
python -c "import json; json.load(open('config/retraining_config.json'))"
```

### Performance Check Failing
```bash
# Check if models exist
ls -la models/*.joblib

# Check data availability
ls -la data/
```

### Logs Location
- Main logs: `data/logs/retraining_scheduler.log`
- Performance data: `data/logs/model_performance.json`
- Retraining history: `data/logs/retraining_history.json`

## API Usage

```python
from retraining_scheduler import get_scheduler

# Get scheduler instance
scheduler = get_scheduler()

# Manual retraining
result = scheduler.manual_retrain('random_forest', 'Testing new features')

# Check performance
degradation = scheduler.check_performance_degradation()

# Get model history
history = scheduler.get_model_history('random_forest')

# Rollback to previous version
scheduler.rollback_model('random_forest', 'version_id')
```</content>
<parameter name="filePath">c:\Users\RAKSHANDA\Downloads\reserach\Bipllab Sir\Stock Market Analysis\RETRAINING_README.md