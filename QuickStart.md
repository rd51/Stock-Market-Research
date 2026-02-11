QUICK START GUIDE

Fastest Path to Working Dashboard (30 minutes)
Step 1: Environment Setup (5 min)
bash# Create project directory
mkdir stock-market-analysis
cd stock-market-analysis

# Clone repository (if applicable) or initialize git
git init

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Step 2: Create Directory Structure (2 min)
bash# Create necessary directories
mkdir -p data/{raw,processed,cache}
mkdir -p src
mkdir -p dashboard/pages
mkdir -p dashboard/utils
mkdir -p tests
mkdir -p notebooks
mkdir -p analyses/regime
mkdir -p analyses/models
Step 3: Configure Environment (3 min)
bash# Create .env file
cat > .env << 'EOF'
# Data Sources
DATA_START_DATE=2022-01-01
DATA_END_DATE=2024-12-31
UPDATE_FREQUENCY=daily

# API Keys (add if required)
# INVESTING_API_KEY=your_key_here
# DATA_GOV_API_KEY=your_key_here

# Model Configuration
TRAIN_TEST_SPLIT=0.7
VALIDATION_SPLIT=0.15
RANDOM_SEED=42

# Dashboard
STREAMLIT_SERVER_PORT=8501
CACHE_TTL_HOURS=6

# Logging
LOG_LEVEL=INFO
EOF
Step 4: Download Data (5 min)
bash# Create simple data download script
python -c "
import yfinance as yf
import pandas as pd

# Download VIX
vix = yf.download('^VIX', start='2022-01-01', end='2024-12-31')
vix.to_csv('data/raw/vix.csv')
print('VIX data saved')

# Download NIFTY (India stock index)
nifty = yf.download('^NSEI', start='2022-01-01', end='2024-12-31')
nifty.to_csv('data/raw/market_index.csv')
print('Market index data saved')
"
Step 5: Create Basic Streamlit App (10 min)
bash# Create app.py
cat > app.py << 'EOF'
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Stock Market AI Analytics", layout="wide")

st.title("📈 Stock Market Analysis: AI-Driven Complex Systems")
st.markdown("---")

# Load data
try:
    vix = pd.read_csv('data/raw/vix.csv', index_col=0, parse_dates=True)
    market = pd.read_csv('data/raw/market_index.csv', index_col=0, parse_dates=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latest VIX", f"{vix['Adj Close'].iloc[-1]:.2f}")
    with col2:
        returns = market['Adj Close'].pct_change().iloc[-1] * 100
        st.metric("Today's Return (%)", f"{returns:.2f}%")
    with col3:
        st.metric("Data Points", len(market))
    
    st.markdown("---")
    
    # Plot VIX
    fig = px.line(vix, y='Adj Close', title='VIX Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot Market Index
    fig2 = px.line(market, y='Adj Close', title='Market Index Over Time')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.success("✅ Dashboard loaded successfully!")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you have downloaded data to data/raw/ directory")

EOF
Step 6: Run Streamlit Dashboard (2 min)
bashstreamlit run app.py
Your dashboard should open at http://localhost:8501

Progressive Enhancement Steps
Once basic dashboard works, enhance with:
Phase 1: Add Data Processing (Prompts 1.1-1.4)
python# Follow Prompt 1.1: Data Ingestion
# Create src/data_ingestion.py
# Test with: python src/data_ingestion.py
Phase 2: Add Regime Analysis (Prompts 2.1-2.2)
python# Follow Prompt 2.1: Regime Analysis
# Create src/regime_analysis.py
# Visualize regime classification
Phase 3: Build Models (Prompts 3.1, 4.1, 5.1)
python# Follow Prompt 3.1: Create baseline OLS model
# Follow Prompt 4.1: Add LSTM model
# Follow Prompt 5.1: Add ensemble models
# Compare on same dataset
Phase 4: Implement Real-Time Updates (Prompts 7.1-7.2)
python# Add real-time data fetching
# Update predictions periodically
# Set up alerts
Phase 5: Multi-Page Dashboard (Prompts 8.1-8.7)
python# Create dashboard/pages/ directory
# Add pages for:
#   - Home
#   - Data Explorer
#   - Regime Analysis
#   - Model Comparison
#   - Real-Time Monitor
#   - Documentation

Minimal Working Example (Copy & Paste)
Minimal Data Processing
python# src/minimal_preprocessing.py
import pandas as pd
import numpy as np

def preprocess_data():
    """Minimal preprocessing pipeline"""
    
    # Load
    vix = pd.read_csv('data/raw/vix.csv', index_col=0, parse_dates=True)
    market = pd.read_csv('data/raw/market_index.csv', index_col=0, parse_dates=True)
    
    # Calculate returns
    market['returns'] = market['Adj Close'].pct_change()
    
    # Align indices
    data = pd.DataFrame({
        'vix': vix['Adj Close'],
        'market_return': market['returns']
    }).dropna()
    
    # Create regimes
    data['vix_regime'] = pd.cut(data['vix'], 
                                 bins=[0, data['vix'].quantile(0.33), 
                                       data['vix'].quantile(0.67), np.inf],
                                 labels=['Low', 'Medium', 'High'])
    
    return data

if __name__ == '__main__':
    df = preprocess_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nRegime distribution:\n{df['vix_regime'].value_counts()}")
Minimal Linear Model
python# src/minimal_ols.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def fit_baseline_model():
    """Minimal OLS baseline"""
    
    # Load preprocessed data
    df = pd.read_csv('data/processed/data.csv', index_col=0)
    
    # Prepare features and target
    X = df[['vix']].values
    y = df['market_return'].values
    
    # Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    
    return model

if __name__ == '__main__':
    fit_baseline_model()
Minimal Dashboard Page
python# dashboard/pages/01_home.py
import streamlit as st
import pandas as pd

st.title("Stock Market Analysis Dashboard")
st.markdown("### AI-Driven Complex System Dynamics")

st.markdown("""
**Research Objectives:**
1. Study asymmetric and non-linear interactions between stock markets, volatility, and labor markets
2. Compare AI models vs linear baselines
3. Create real-time analytical system

**Status:** 🟡 In Development
- Data loaded: ✅
- Basic models: 🟡 In Progress
- Dashboard: 🟡 In Development
""")

# Load and display summary stats
try:
    data = pd.read_csv('data/processed/data.csv', index_col=0)
    st.markdown("---")
    st.markdown("### Data Summary")
    st.dataframe(data.describe())
except:
    st.warning("No processed data found. Run data pipeline first.")

Common Commands Reference
bash# Data preparation
python src/data_ingestion.py
python src/preprocessing_pipeline.py

# Model training
python src/baseline_models.py
python src/lstm_models.py
python src/ensemble_models.py

# Evaluation
python src/model_evaluation.py
python -m pytest tests/

# Dashboard
streamlit run app.py
streamlit run app.py --logger.level=debug

# Clean up
rm -rf data/processed/*
rm -rf models/*

Debugging Checklist
If data doesn't load:

✅ Check data/raw/ directory exists
✅ Verify CSV files are present
✅ Check date columns are parsed correctly
✅ Ensure no missing dates cause misalignment

If models won't train:

✅ Verify data is normalized
✅ Check for NaN values
✅ Ensure train/test split is correct
✅ Check GPU availability (for LSTM)

If dashboard won't start:

✅ Verify app.py exists
✅ Check all imports are installed
✅ Try: streamlit cache clear
✅ Restart terminal and activate venv again

If models have poor performance:

✅ Verify data preprocessing is correct
✅ Check feature engineering captures relationships
✅ Validate train/test temporal split (no leakage)
✅ Ensure baseline model is reasonable
✅ Check for regime shifts in test period


Next Steps After Getting Dashboard Running

Read COPILOT_PROMPTS.md - Choose next component to build
Start with Prompt 1.1 - Implement complete data pipeline
Then Prompt 2.1 - Add regime analysis
Then Prompt 3.1 - Build linear baseline (can compare!)
Then Prompt 4.1 - Add LSTM (watch performance!)
Then Prompt 5.1 - Add ensemble (should beat linear!)
Then create pages - Multi-page Streamlit app


Expected Timeline

Week 1: Data pipeline + basic dashboard
Week 2: Regime analysis + linear models
Week 3: LSTM + ensemble models
Week 4: Real-time integration + multi-page dashboard
Week 5: Testing + deployment


Resources
For Streamlit:

https://docs.streamlit.io
https://streamlit.io/gallery

For ML Models:

TensorFlow/Keras: https://tensorflow.org
XGBoost: https://xgboost.readthedocs.io
scikit-learn: https://scikit-learn.org

For VS Code Copilot:

Inline autocomplete as you code
Use #region comments to organize
Ask for tests alongside code
Request refactoring suggestions