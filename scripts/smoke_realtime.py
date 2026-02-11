import sys
import os
sys.path.insert(0, os.getcwd())  # Ensure project root is on sys.path

from real_time_monitor import RealtimeDataFeed

if __name__ == '__main__':
    r = RealtimeDataFeed()
    print('Initialized RealtimeDataFeed')
    print('is_market_open ->', r.is_market_open())

    data = r.fetch_all_latest_data(use_live=False)
    print('\nfetch_all_latest_data(use_live=False) returned keys:', list(data.keys()))
    for k, v in data.items():
        if k in ['unemployment', 'vix', 'market_index']:
            print(k, '->', 'has_data' if v else 'None')

    # Try live mode (may return cached/fallback)
    data_live = r.fetch_all_latest_data(use_live=True)
    print('\nfetch_all_latest_data(use_live=True) returned keys:', list(data_live.keys()))
    for k, v in data_live.items():
        if k in ['unemployment', 'vix', 'market_index']:
            print('live', k, '->', 'has_data' if v else 'None')
