"""
Data Generator for Stock Market Analysis Dashboard
==================================================

Generates realistic financial time series data for testing and demonstration.
Creates CSV files with market data, VIX, unemployment rates, and returns.

Author: GitHub Copilot
Date: February 11, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse

def generate_financial_data(start_date='2020-01-01', end_date='2024-12-31', output_file='stationary_data.csv'):
    """
    Generate realistic financial time series data.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_file (str): Output CSV file path
    """
    print(f"Generating financial data from {start_date} to {end_date}")

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate base market returns (daily returns)
    # Using realistic volatility patterns
    base_returns = np.random.normal(0.0005, 0.015, len(dates))  # ~0.05% mean, 1.5% std

    # Add some autocorrelation (momentum effect)
    for i in range(1, len(base_returns)):
        base_returns[i] += 0.2 * base_returns[i-1]  # 20% autocorrelation

    # Generate VIX (volatility index)
    # VIX tends to be higher when markets are volatile
    market_volatility = np.abs(base_returns) * 100
    vix_base = 15 + market_volatility * 50  # Base VIX around 15, spikes with volatility
    vix_noise = np.random.normal(0, 2, len(dates))
    vix = vix_base + vix_noise

    # Ensure VIX stays within reasonable bounds
    vix = np.clip(vix, 8, 80)

    # Generate unemployment rate
    # Unemployment has slower-moving trends with some seasonality
    unemployment_trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 0.5  # Seasonal component
    unemployment_random = np.random.normal(0, 0.1, len(dates))  # Random variation
    unemployment_base = 6.0 + unemployment_trend + unemployment_random.cumsum() * 0.01  # Trending component
    unemployment = np.clip(unemployment_base, 3.0, 15.0)  # Reasonable bounds

    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'VIX': vix,
        'Returns': base_returns,
        'Unemployment': unemployment
    })

    # Add some market regime detection
    data['MarketRegime'] = data['Returns'].apply(
        lambda x: 'bull' if x > 0.005 else ('bear' if x < -0.005 else 'sideways')
    )

    # Save to CSV
    data.to_csv(output_file, index=False)
    print(f"Generated {len(data)} records and saved to {output_file}")

    # Print summary statistics
    print("\nData Summary:")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Average VIX: {data['VIX'].mean():.2f}")
    print(f"Average Returns: {data['Returns'].mean():.6f}")
    print(f"Average Unemployment: {data['Unemployment'].mean():.2f}")
    print(f"Market Regimes: {data['MarketRegime'].value_counts().to_dict()}")

    return data

def generate_prediction_history(output_file='data/cache/prediction_history.csv'):
    """
    Generate sample prediction history data.

    Args:
        output_file (str): Output CSV file path
    """
    print("Generating prediction history data...")

    # Create timestamps for the last 30 days
    base_time = datetime.now()
    timestamps = [base_time - timedelta(hours=i) for i in range(0, 720, 6)]  # Every 6 hours for 30 days

    np.random.seed(123)

    data = []
    for ts in timestamps:
        # Generate realistic predictions
        consensus_mean = np.random.normal(0, 2)  # Mean prediction around 0
        consensus_std = np.random.uniform(0.5, 2.0)  # Standard deviation
        agreement = np.random.uniform(50, 95)  # Agreement percentage

        # Individual model predictions
        models = ['ols_static', 'random_forest', 'xgboost', 'lightgbm', 'ensemble']
        model_preds = {}
        model_uncertainty = {}

        for model in models:
            pred = consensus_mean + np.random.normal(0, consensus_std)
            uncertainty = np.random.uniform(0.1, 0.5)
            model_preds[model] = pred
            model_uncertainty[f'{model}_uncertainty'] = uncertainty

        # Create row
        row = {
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S%z'),
            'data_freshness': 'fresh' if (datetime.now() - ts).total_seconds() < 3600 else 'stale',
            'consensus_mean': consensus_mean,
            'consensus_std': consensus_std,
            'agreement_strength': agreement,
            'vix': 15 + np.random.normal(0, 3),
            'unemployment': 7.0 + np.random.normal(0, 0.5),
            'market_return': np.random.normal(0, 0.02),
            **model_preds,
            **model_uncertainty
        }
        data.append(row)

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} prediction records and saved to {output_file}")

    return df

def generate_model_metrics(output_file='model_evaluation_metrics.csv'):
    """
    Generate sample model evaluation metrics.

    Args:
        output_file (str): Output CSV file path
    """
    print("Generating model evaluation metrics...")

    models = ['RandomForest', 'LinearRegression', 'XGBoost', 'LightGBM', 'Ensemble']
    metrics = ['mae', 'mse', 'rmse', 'mape', 'r2_score', 'direction_accuracy',
               'mean_absolute_error', 'median_absolute_error', 'explained_variance',
               'mean_error', 'std_error']

    data = []
    for model in models:
        row = {'Model': model}
        for metric in metrics:
            if metric in ['mae', 'mse', 'rmse', 'mape', 'mean_absolute_error', 'median_absolute_error', 'mean_error', 'std_error']:
                # Error metrics (lower is better)
                value = np.random.uniform(0.1, 1.0)
            elif metric == 'r2_score':
                # R² score (higher is better)
                value = np.random.uniform(0.7, 0.95)
            elif metric == 'explained_variance':
                # Explained variance (higher is better)
                value = np.random.uniform(0.75, 0.95)
            elif metric == 'direction_accuracy':
                # Direction accuracy (higher is better)
                value = np.random.uniform(0.5, 0.9)
            row[metric] = 'value'
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated model metrics for {len(df)} models and saved to {output_file}")

    return df

def main():
    """Main function to generate all datasets."""
    parser = argparse.ArgumentParser(description='Generate financial datasets for Stock Market Analysis Dashboard')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='.', help='Output directory')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'data', 'cache'), exist_ok=True)

    # Generate datasets
    print("🚀 Starting data generation...")

    # 1. Generate main financial data
    generate_financial_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=os.path.join(args.output_dir, 'stationary_data.csv')
    )

    # 2. Generate prediction history
    generate_prediction_history(
        output_file=os.path.join(args.output_dir, 'data', 'cache', 'prediction_history.csv')
    )

    # 3. Generate model evaluation metrics
    generate_model_metrics(
        output_file=os.path.join(args.output_dir, 'model_evaluation_metrics.csv')
    )

    print("\n✅ All datasets generated successfully!")
    print("You can now run the dashboard with: streamlit run app.py")

if __name__ == "__main__":
    main()