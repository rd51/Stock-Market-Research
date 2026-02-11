import sys
import os
sys.path.insert(0, os.getcwd())
from real_time_monitor import RealtimeDataFeed
from prediction_updater import RealtimePredictorUpdater

if __name__ == '__main__':
    r = RealtimeDataFeed()
    print('RealtimeDataFeed initialized')
    data = r.fetch_all_latest_data(use_live=True)
    print('Fetched data keys:', list(data.keys()))

    p = RealtimePredictorUpdater(live_data_feed=r)
    print('Predictor initialized with models:', list(p.models_dict.keys()))

    res = p.update_predictions_on_new_data(use_live_data=True)
    print('Predictions:', res.get('predictions'))
    print('Consensus:', res.get('consensus'))
    print('Alerts:', res.get('alerts'))
