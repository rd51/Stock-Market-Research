import runpy
from datetime import datetime, timedelta
import json
import os

# Run app.py as a module (so __name__ != '__main__') to import functions without running main()
mod = runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'app.py'), run_name='app_module')

# Helper to safely call clear on cached functions
def try_clear(fn):
    try:
        if hasattr(fn, 'clear'):
            fn.clear()
            print(f"Cleared cache for {fn.__name__}")
    except Exception as e:
        print(f"Failed to clear cache for {getattr(fn, '__name__', str(fn))}: {e}")

# Load config
load_config = mod.get('load_config')
config = {}
if load_config:
    try:
        config = load_config()
    except Exception as e:
        print(f"load_config failed: {e}")

# Prepare dates
default_days = int(config.get('date_range_days', 365*7))
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=default_days)).strftime('%Y-%m-%d')

# Clear caches where available
for name in ('load_market_data', 'load_trained_models', 'load_prediction_history'):
    fn = mod.get(name)
    if fn:
        try_clear(fn)

status = {
    'start_date': start_date,
    'end_date': end_date,
    'market_data_rows': None,
    'models_loaded': 0,
    'prediction_history_rows': None,
    'errors': []
}

# Load market data
load_market_data = mod.get('load_market_data')
if load_market_data:
    try:
        df = load_market_data(start_date, end_date)
        if df is not None:
            try:
                status['market_data_rows'] = int(len(df))
            except Exception:
                status['market_data_rows'] = None
        else:
            status['market_data_rows'] = None
        print(f"Market data rows: {status['market_data_rows']}")
    except Exception as e:
        status['errors'].append(f"load_market_data error: {e}")
        print(status['errors'][-1])

# Load models
load_trained_models = mod.get('load_trained_models')
if load_trained_models:
    try:
        models, scalers = load_trained_models()
        if isinstance(models, dict):
            status['models_loaded'] = len(models)
        else:
            status['models_loaded'] = 0
        print(f"Models loaded: {status['models_loaded']}")
    except Exception as e:
        status['errors'].append(f"load_trained_models error: {e}")
        print(status['errors'][-1])

# Load prediction history
load_prediction_history = mod.get('load_prediction_history')
if load_prediction_history:
    try:
        hist = load_prediction_history()
        if hist is not None:
            try:
                status['prediction_history_rows'] = int(len(hist))
            except Exception:
                status['prediction_history_rows'] = None
        else:
            status['prediction_history_rows'] = None
        print(f"Prediction history rows: {status['prediction_history_rows']}")
    except Exception as e:
        status['errors'].append(f"load_prediction_history error: {e}")
        print(status['errors'][-1])

# Write status to file
out_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'refresh_status.json'), 'w') as f:
    json.dump(status, f, indent=2)

print('Refresh complete. Status written to logs/refresh_status.json')
