Stock Market Analysis: AI-Driven Complex System Dynamics

Project Overview
This project implements an advanced AI-driven analytical framework to study the asymmetric and non-linear interactions between stock market dynamics, volatility (VIX), and labor market indicators as a complex adaptive system. Rather than relying on traditional linear econometric models, this research employs sophisticated machine learning techniques—including LSTM networks and tree-ensemble learning—integrated into a real-time Streamlit dashboard for interactive exploration and analysis.
The study is grounded in complexity science theory, which posits that financial and labor markets exhibit feedback mechanisms, path dependence, and regime-dependent behavior that static models cannot capture. By combining econometric analysis with AI-based non-linear modeling and real-time data visualization, this project provides a methodological innovation for understanding emerging market-labor divergences.
Research Objectives
Objective 1: Asymmetric and Non-Linear Interaction Analysis
Study how stock market dynamics, volatility shocks, and labor market indicators interact as a complex adaptive system through:

Identification of asymmetric response patterns (markets react faster than labor markets)
Detection of non-linear dependencies and feedback loops
Regime-based segmentation for volatility and market states
Lag effects and volatility clustering diagnostics

Hypotheses:

H₀₁: No meaningful asymmetric or non-linear correlation exists between unemployment, volatility, and stock market returns
H₁₁: Significant asymmetric and non-linear interactions exist, consistent with complex adaptive system behavior

Objective 2: AI Model Performance Comparison
Assess the effectiveness of AI-based non-linear models against conventional linear regression methods:

Compare predictive accuracy across different market regimes
Evaluate capacity to identify feedback loops and delayed responses
Test cross-regime stability and generalization
Measure detection of temporal dependencies and volatility clustering

Hypotheses:

H₀₂: AI-based non-linear models do not substantially outperform linear regression in detecting feedback effects
H₁₂: AI-based models significantly outperform linear models in capturing market-labor dynamics

Objective 3: Real-Time AI-Driven Analytical System
Create and deploy an interactive dashboard that:

Integrates real-time data streams for continuous market monitoring
Updates model predictions periodically
Visualizes regime shifts, volatility spikes, and labor market inertia
Captures emerging divergences and market-labor interactions

Hypotheses:

H₀₃: Real-time AI systems provide no new insights beyond static historical analysis
H₁₃: Real-time AI systems effectively capture emerging divergences and regime shifts

Dataset Description
Data Sources
IndicatorSourceFrequencyTypeUnemployment Ratedata.gov.in, labour.gov.inMonthlyTime-seriesVolatility Index (VIX)NSE IndiaDailyReal-timeStock Market Prices (Index-level)NSE India, Investing.comDailyReal-timeMacroeconomic IndicatorsVarious APIsVariesSupporting
Key Variables
Independent Variables:

VIX (Volatility Index) - Market uncertainty measure
Unemployment Rate - Labor market state
Lagged Market Returns - Historical momentum

Dependent Variables:

Stock Market Returns (index-level) - Primary outcome

Data Characteristics

Temporal Scope: Historical data for backtesting + real-time streams for live analysis
Frequency: Daily for financial markets, monthly for labor data
Stationarity: Non-stationary; regime shifts expected
Volatility Clustering: Expected; requires specialized handling
Lag Structures: Labor market effects typically appear with delay

Methodology Steps
Phase 1: Data Preparation and Exploration

Data Ingestion

Import historical data from multiple sources
Implement real-time data pipelines (daily/hourly updates)
Validate data quality and handle missing values
Align temporal frequency across datasets


Exploratory Data Analysis (EDA)

Compute descriptive statistics (mean, std, skewness, kurtosis)
Visualize time-series behavior and distribution changes
Identify structural breaks and anomalies
Analyze data quality issues and gaps


Stationarity Testing

Augmented Dickey-Fuller (ADF) tests
KPSS tests
Visual inspection of trends and seasonality
Document unit root presence


Data Preprocessing

Handle missing values (forward fill, interpolation)
Remove outliers using statistical methods
Normalize/standardize features (MinMaxScaler, StandardScaler)
Create lag features for temporal dependencies



Phase 2: Regime-Based Segmentation Analysis

Volatility Regime Classification

Calculate VIX percentiles and quartiles
Classify periods as High/Low/Medium volatility
Define regime thresholds (e.g., VIX > 75th percentile = High Regime)
Create binary/categorical regime indicators


Rolling-Window Analysis

Compute rolling correlations (30-day, 60-day, 90-day windows)
Track correlation dynamics across regimes
Identify correlation breakdown periods
Detect regime transition points


Volatility Clustering Detection

Compute ARCH/GARCH effects
Analyze volatility persistence
Detect volatility spike clustering patterns
Visualize volatility regime persistence


Asymmetric Response Testing

Compare response magnitudes for positive vs. negative shocks
Analyze lag structures: Market vs. Labor Market
Quantify asymmetry indices
Document institutional inertia effects



Phase 3: Linear Baseline Models

Ordinary Least Squares (OLS) Regression

Build static regression: Stock Returns ~ VIX + Unemployment
Include lagged returns and lagged unemployment
Compute R², Adjusted R², and F-statistics
Analyze residuals (normality, autocorrelation, heteroskedasticity)


Model Diagnostics

Durbin-Watson test (autocorrelation)
White's test (heteroskedasticity)
VIF analysis (multicollinearity)
Residual normality tests


Regime-Specific OLS Models

Fit separate OLS models for High and Low volatility regimes
Compare coefficient stability across regimes
Test for structural breaks
Evaluate predictive power in each regime



Phase 4: AI-Powered Non-Linear Models

LSTM (Long Short-Term Memory) Networks

Architecture Design: Input layer → LSTM layers (64-128 units) → Dense layers → Output
Sequence Preparation: Create sequences of 30/60-day windows
Feature Engineering: Include VIX, Unemployment, lagged returns, rolling volatility
Training:

Split data: 70% train, 15% validation, 15% test
Use Adam optimizer with learning rate scheduling
Implement early stopping and model checkpointing
Track loss on validation set


Hyperparameter Tuning: LSTM units, layers, dropout rates, batch size
Output: Predict next-day/next-week returns
Advantages: Captures temporal dependencies, handles long-range sequences, learns feedback loops


Tree-Ensemble Models (Random Forest, XGBoost, LightGBM)

Feature Engineering: Create lag features, rolling statistics, regime indicators
Model Development:

Random Forest: 100-500 trees, max_depth optimization
XGBoost: Learning rate tuning, max_depth, subsample parameters
LightGBM: Leaf-wise growth, feature fraction optimization


Training: Cross-validation (5-fold stratified)
Output: Return predictions with feature importance rankings
Advantages: Handles non-linearity, automatic feature interactions, regime awareness


Hybrid Ensemble Approach

Combine LSTM and tree-based predictions via weighted averaging
Meta-learner stacking (train second-level model on individual predictions)
Voting mechanism with regime-aware weights



Phase 5: Model Performance Evaluation

Predictive Accuracy Metrics

Mean Absolute Error (MAE)
Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
Mean Absolute Percentage Error (MAPE)
Direction Accuracy (% of correct up/down predictions)


Cross-Regime Stability

Evaluate all models on High, Low, and Medium volatility regimes
Compute per-regime RMSE and MAE
Analyze performance degradation across regimes
Test generalization to unseen regime transitions


Feedback Loop Identification

Analyze lagged effects (VIX → Returns, Unemployment → Returns)
Quantify feedback strength (correlation between predictions and actual lags)
Visualize Granger causality results
Document bidirectional causality evidence


Comparative Analysis

Create performance comparison tables (Linear vs. LSTM vs. Ensemble)
Statistical significance testing (Diebold-Mariano test)
Visualize prediction paths vs. actuals for key periods
Highlight where AI models outperform baselines



Phase 6: Real-Time Dashboard Development

Streamlit Dashboard Architecture

Multi-page layout (Home, Analysis, Models, Real-Time Monitor, Documentation)
Session state management for persistence
Caching strategies for performance optimization


Core Dashboard Pages

Home: Project overview, key hypotheses, methodology summary
Data Explorer: Interactive time-series visualization, filter by date/regime
Regime Analysis: Volatility regime charts, rolling correlations, regime transitions
Model Performance: Comparison tables, accuracy metrics, prediction vs. actual plots
Real-Time Monitor: Live data feeds, latest predictions, emerging divergences
Complex System View: Theoretical model visualization, feedback loop diagram
Download/Export: CSV/Excel export of analyses and predictions


Interactive Features

Date range selection
Regime filtering (High/Low/All)
Model selection and comparison
Metric drilling (hover details, cross-filtering)
Real-time update indicators


Real-Time Data Integration

API connections to NSE India, investing.com, labor data sources
Scheduled data refresh (daily for markets, weekly for labor data)
Error handling and data validation
Caching with refresh indicators


Visualization Components

Time-series plots with multiple y-axes
Regime bands as background shading
Prediction vs. actual overlay
Feature importance bar charts
Correlation heatmaps (overall and rolling)
Volatility clustering scatter plots
Performance metric comparison tables



Analysis and Visualizations
EDA & Data Quality

Time-series decomposition (trend, seasonality, residuals)
Distribution plots with KDE overlays
Missing data heatmap
Autocorrelation and partial autocorrelation functions (ACF/PACF)

Regime-Based Analysis

VIX regime classification stacked area chart
Rolling 30/60/90-day correlation plots with regime shading
Volatility clustering scatter (consecutive volatility vs. current)
Asymmetric response bar charts (positive vs. negative shock effects)
Regime transition timeline with annotations

Model Performance Visualization

RMSE/MAE comparison across models (bar chart)
Per-regime performance breakdown (grouped bar or faceted plots)
Actual vs. Predicted time-series overlay (line plots)
Prediction error distribution (histograms, Q-Q plots)
Feature importance rankings (SHAP, tree-based importances)
Learning curves (training/validation loss over epochs)

Complexity System Insights

Theoretical model diagram (interactive flow)
Feedback loop strength indicator (gauge charts)
Lag effect heatmap (VIX/Unemployment vs. Returns at various lags)
Granger causality matrix visualization
Regime persistence charts

Real-Time Dashboard Visualizations

Live data stream indicator (timestamp, update frequency)
Latest predictions with confidence intervals
Emerging divergence alerts (when correlations deviate from historical)
Market regime current status (visual indicator)
KPI cards (current volatility, unemployment, latest return, model agreement)
Real-time prediction tracking (30-day rolling window)

Technical Stack
Core Technologies

Python 3.8+: Data processing and modeling
Streamlit: Interactive dashboard framework
TensorFlow/Keras: LSTM network implementation
scikit-learn: Linear models, preprocessing, evaluation
XGBoost, LightGBM: Ensemble tree models
Pandas: Data manipulation and analysis
NumPy: Numerical computing
Matplotlib, Plotly, Seaborn: Visualization

Data & API Integration

requests, yfinance: API interactions
SQLAlchemy (optional): Database integration for historical cache
APScheduler (optional): Scheduled data refresh tasks

Development Environment

VS Code: IDE with Copilot support
Git: Version control
Virtual Environment (venv/conda): Dependency isolation

Installation & Setup
Prerequisites

Python 3.8 or higher
pip or conda package manager
Git

Step 1: Clone Repository
bashgit clone <repository-url>
cd stock-market-analysis
Step 2: Create Virtual Environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Install Dependencies
bashpip install -r requirements.txt
Step 4: Configure Environment Variables
bash# Create .env file with API keys and data source configurations
cp .env.example .env
# Edit .env with your settings (API keys, data paths, etc.)
Step 5: Verify Installation
bashpython -c "import streamlit, tensorflow, xgboost; print('All imports successful!')"
Running the Streamlit Dashboard
Basic Launch
bashstreamlit run app.py
The dashboard will open at http://localhost:8501
Advanced Options
bash# Run with custom configuration
streamlit run app.py --logger.level=debug

# Specify port
streamlit run app.py --server.port=8502

# Remote access
streamlit run app.py --server.address=0.0.0.0
File Structure
stock-market-analysis/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md                        # This file
│
├── data/
│   ├── raw/                        # Original data files
│   ├── processed/                  # Cleaned and preprocessed data
│   └── cache/                      # Real-time data cache
│
├── src/
│   ├── data_ingestion.py          # Data loading and API integration
│   ├── preprocessing.py            # Data cleaning and feature engineering
│   ├── regime_analysis.py          # Volatility regime and rolling correlations
│   ├── baseline_models.py          # OLS and linear models
│   ├── lstm_models.py              # LSTM network implementation
│   ├── ensemble_models.py          # XGBoost, LightGBM, Random Forest
│   ├── model_evaluation.py         # Performance metrics and comparison
│   ├── real_time_monitor.py        # Live data and prediction updates
│   └── utils.py                    # Helper functions
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_regime_segmentation.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_results_summary.ipynb
│
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
└── dashboard/
    ├── pages/
    │   ├── 01_home.py
    │   ├── 02_data_explorer.py
    │   ├── 03_regime_analysis.py
    │   ├── 04_model_comparison.py
    │   ├── 05_real_time_monitor.py
    │   ├── 06_complex_systems.py
    │   └── 07_documentation.py
    └── utils/
        ├── layout.py               # Streamlit styling
        ├── cache.py                # Data caching
        └── config.py               # Dashboard configuration
Key Features
✅ AI-Driven Non-Linear Modeling: LSTM networks and tree ensembles for complex patterns
✅ Regime-Based Analysis: Separate insights for high/low volatility periods
✅ Real-Time Integration: Live data feeds from NSE, data.gov.in, and market APIs
✅ Comprehensive Comparison: Linear baselines vs. AI models with statistical tests
✅ Interactive Dashboard: Multi-page Streamlit interface for exploration
✅ Complexity Science Framework: Grounded in complexity theory principles
✅ Asymmetric Response Modeling: Captures market-labor divergences
✅ Production-Ready Code: Modular, testable, documented codebase
Usage Examples
Running the Complete Pipeline
bash# 1. Data ingestion and preprocessing
python src/data_ingestion.py

# 2. Regime analysis
python src/regime_analysis.py

# 3. Model training and evaluation
python src/baseline_models.py
python src/lstm_models.py
python src/ensemble_models.py

# 4. Launch dashboard
streamlit run app.py
Running Individual Components
bash# Test data quality
python -m pytest tests/test_data_ingestion.py

# Verify preprocessing
python tests/test_preprocessing.py

# Validate models
python tests/test_models.py
Expected Outputs
The analysis produces:

Statistical Summaries: Mean, std, correlation matrices, regime statistics
Regime Visualizations: Time-series with volatility bands, rolling correlations
Model Outputs:

Linear: OLS coefficients, R², residual diagnostics
LSTM: Predictions with training history, loss curves
Ensemble: Feature importance, cross-validation scores


Performance Comparison: Metrics table, statistical significance tests, prediction accuracy
Dashboard Analytics: Real-time feeds, KPIs, interactive filters
Reports: EDA summary, model documentation, findings

Interpretation Guide
Understanding Results
Regime Segmentation Results:

High VIX regimes show stronger correlations (more interaction)
Low VIX regimes show weaker relationships (market stability)
Labor market changes lag market movements by 1-3 months (inertia)

Model Comparison:

AI models should capture volatility clustering better
LSTM excels at identifying lagged effects
Ensemble models provide robust regime-aware predictions
Linear models may miss feedback interactions

Real-Time Monitor:

Green zone: Model agreement and stability
Yellow zone: Divergence emerging (asymmetric response)
Red zone: Potential regime shift or model disagreement

Troubleshooting
Common Issues
IssueSolutionData not loadingCheck API connectivity, verify .env credentialsLSTM training slowReduce sequence length, use GPU accelerationDashboard lagEnable caching with @st.cache_data decoratorMissing dependenciesRun pip install -r requirements.txt --upgradeMemory errorsReduce batch size, process data in chunks
Contributing & Feedback

Report issues via GitHub Issues
Suggest improvements through Pull Requests
Share analysis findings and insights
Contribute additional models or visualizations