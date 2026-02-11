"""Smoke test: generate a PNG plot using repository market data.
Saves output to tmp/smoke_test_plot.png and prints file size.
"""
import os
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data():
    path = 'stationary_data.csv'
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.reset_index().rename(columns={'index': 'Date'})
            return df
        except Exception as e:
            print(f"Failed loading {path}: {e}")
    # fallback sample
    dates = pd.date_range(start='2020-01-01', periods=180, freq='D')
    np.random.seed(42)
    data = {'Date': dates, 'NIFTY_Close': 18000 + np.cumsum(np.random.normal(0, 50, len(dates)))}
    return pd.DataFrame(data)


if __name__ == '__main__':
    df = load_data()

    if 'Date' in df.columns:
        x = pd.to_datetime(df['Date'])
    else:
        x = df.index

    if 'NIFTY_Close' in df.columns:
        y = df['NIFTY_Close']
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise SystemExit('No numeric data to plot')
        y = df[numeric_cols[0]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color='#2b6cb0')
    ax.set_xlabel('Date')
    ax.set_ylabel('NIFTY_Close')
    ax.set_title('Smoke Test: NIFTY_Close')
    fig.autofmt_xdate()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = buf.getvalue()
    out_dir = 'tmp'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'smoke_test_plot.png')
    with open(out_path, 'wb') as f:
        f.write(data)

    size = len(data)
    print(f"Saved plot to {out_path}, size={size} bytes")
    if size == 0:
        raise SystemExit('Generated PNG has zero length')
    print('Smoke test plot generation succeeded')