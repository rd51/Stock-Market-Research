Research Methodology → Implementation Mapping
This document maps each step in the research methodology (from the Stock_Market_Analysis_Final_Draft.docx) to specific Copilot prompts and implementation files.

RESEARCH METHODOLOGY MAPPING
VIII. RESEARCH METHODOLOGY
A. Research Design
Description from Paper:

"The proposed research design is based on the area of complexity science and AI-based analytics, as it is quantitative, explanatory, and system-oriented. The relationship of the stock market to the labour market is considered as a complex adaptive system..."

Implementation Path:

Conceptual Framework Setup → README.md "Methodology Steps" section

Create visualization of complex system
Use Prompt 8.6 (Complex Systems View page)


Data Foundation → Prompt 1.1 (Data Ingestion)

Load three data sources: VIX, Market Index, Unemployment
File: src/data_ingestion.py


System Architecture → Prompt 8.1 (Main Dashboard App)

Create multi-page Streamlit framework
File: app.py and dashboard/pages/




B. Data Sources and Variables
From Paper:

VIX: Volatility Index from NSE India
Stock Market Prices: Index-level data from NSE India and Investing.com
Unemployment Rate: From data.gov.in and labour.gov.in

Implementation:
Data SourcePaper RefCopilot PromptFileVariablesVIX (NSE)Table 1.2Prompt 1.1src/data_ingestion.pyVIX_Close, VIX_High, VIX_LowMarket IndexTable 1.2Prompt 1.1src/data_ingestion.pyClose, High, Low, Open, VolumeUnemploymentTable 1.2Prompt 1.1src/data_ingestion.pyUnemploymentRate
Quality Check:

Use Prompt 1.2 (Data Quality Assessment)
File: src/data_quality.py
Validates: Missing values, outliers, stationarity, alignment

Data Preparation:

Use Prompt 1.3 (Feature Engineering)
Use Prompt 1.4 (Preprocessing Pipeline)
Files: src/feature_engineering.py, src/preprocessing_pipeline.py


C. Objective 1: Asymmetric and Non-Linear Interaction Analysis
From Paper:

"The study addresses latent temporal dependencies and regime shifts in stock-labor systems... Through analyzing asymmetric response patterns... non-linear dependencies and feedback loops"

Research Question:
How can non-linear, AI-driven models capture the asymmetric and feedback-rich interactions among volatility, stock market behavior, and labor market indicators?
Null Hypothesis (H₀₁):
There isn't any meaningful asymmetric or non-linear correlation between unemployment indicators, volatility, and stock market returns.
Alternative Hypothesis (H₁₁):
There are notable asymmetric and non-linear interactions between unemployment, volatility, and stock market returns that are consistent with complex adaptive system behavior.
Analytical Methods from Paper:

Regime-based segmentation (periods of high and low volatility)
Rolling-window correlation analysis
Diagnostics for non-linear dependencies (lag effects and volatility clustering)

Implementation:
MethodImplementationCopilot PromptFileOutputRegime SegmentationClassify VIX into High/Medium/Low based on percentilesPrompt 2.1src/regime_analysis.pyregime_labels.csvRolling Correlation30/60/90-day windows: VIX vs Returns, Unemployment vs ReturnsPrompt 2.1src/regime_analysis.pyrolling_corr.csvVolatility ClusteringIdentify consecutive high-volatility periodsPrompt 2.1src/regime_analysis.pyclustering_report.csvAsymmetric ResponseAnalyze positive vs negative shock effectsPrompt 2.1src/regime_analysis.pyasymmetry_stats.jsonLag AnalysisCorrelation at lags 0-20 daysPrompt 2.1src/regime_analysis.pylag_effects.csv
Visualization (Prompt 8.3):

Dashboard page: dashboard/pages/03_regime_analysis.py
Elements:

Regime timeline with colored backgrounds
Rolling correlation plots with regime shading
Volatility clustering scatter plots
Asymmetric response bar charts
Lag structure heatmaps



Expected Findings:

✅ Regime shifts will be visible (High VIX = stronger correlations)
✅ Labor market lags market by 1-3 months
✅ Non-linear relationships confirmed by clustering
✅ Asymmetric shocks have different magnitudes


D. Objective 2: AI Model Comparison
From Paper:

"To assess how useful AI-based non-linear models are at identifying feedback loops and regime-dependent behaviour as opposed to conventional methods of linear regression."

Null Hypothesis (H₀₂):
When it comes to detecting feedback effects and regime-dependent market behavior, AI-based non-linear models do not substantially outperform conventional linear regression models.
Alternative Hypothesis (H₁₂):
When it comes to capturing feedback loops and regime-dependent behavior in market–labor dynamics, AI-based non-linear models perform noticeably better than linear regression models.
Models from Paper:
Model TypeArchitectureCopilot PromptFileWhy UsedLinear BaselineOLS RegressionPrompt 3.1src/baseline_models.pyCaptures simple proportional relationshipsLinear LaggedOLS with lagsPrompt 3.1src/baseline_models.pyTests lag effectsLSTM NetworkStacked LSTM + DensePrompt 4.1src/lstm_models.pyCaptures temporal dependencies and feedbackRandom ForestTree ensemblePrompt 5.1src/ensemble_models.pyDetects non-linearityXGBoostGradient boosted treesPrompt 5.1src/ensemble_models.pyRegime-aware predictionsLightGBMOptimized gradient boostingPrompt 5.1src/ensemble_models.pyFast, scalable alternativeEnsemble HybridWeighted average of allPrompt 5.2src/ensemble_models.pyRobust consensus predictions
Training Approach:

Use Prompt 1.4 (Data Split):

70% training, 15% validation, 15% test
IMPORTANT: Maintain temporal order (time-series split, not random)



Model Features (from methodology):
Independent Variables:
- VIX (current)
- Unemployment Rate (current)
- Market Returns (lagged versions)

Dependent Variable:
- Stock Market Returns (next day/week)
Feature Engineering (Prompt 1.3):

Lag features: VIX_lag1, VIX_lag5, Returns_lag1-20, Unemployment_lag1-3
Rolling statistics: VIX_rolling_mean_30, Returns_rolling_std_60
Regime indicators: Volatility_Regime (0/1/2)
Shock indicators: VIX_shock, Return_shock

Evaluation Metrics (Prompt 6.1):

Predictive accuracy: RMSE, MAE, MAPE, R²
Direction accuracy: % correct direction predictions
Cross-regime stability: Performance breakdown by regime
Feedback detection: Granger causality results
Lag sensitivity: Which lags matter most

Statistical Comparison (Prompt 6.2):

Diebold-Mariano test: Statistical significance of model differences
Cross-validation: 5-fold time-series cross-validation
Per-regime evaluation: Are AI models better in all regimes?

Key Comparison Table Structure:
Model           | RMSE  | MAE  | MAPE | R²    | Dir.Acc | High Reg | Medium | Low Reg | Lags Captured
OLS Static      | 0.018 | 0.013| 2.1% | 0.32  | 52%     | 0.022    | 0.018  | 0.015   | No
OLS Lagged      | 0.017 | 0.012| 1.8% | 0.38  | 54%     | 0.020    | 0.016  | 0.015   | Yes (5 days)
LSTM            | 0.014 | 0.010| 1.2% | 0.48  | 58%     | 0.018    | 0.014  | 0.012   | Yes (20 days)
Random Forest   | 0.015 | 0.011| 1.5% | 0.44  | 56%     | 0.017    | 0.015  | 0.013   | Yes (all)
XGBoost         | 0.013 | 0.009| 1.0% | 0.52  | 60%     | 0.016    | 0.013  | 0.011   | Yes (regime)
LightGBM        | 0.013 | 0.009| 1.0% | 0.51  | 59%     | 0.016    | 0.013  | 0.012   | Yes (regime)
Ensemble        | 0.012 | 0.008| 0.9% | 0.55  | 61%     | 0.015    | 0.012  | 0.010   | Yes (all)
Visualization (Prompt 8.4):

Dashboard page: dashboard/pages/04_model_comparison.py
Elements:

Summary table (all metrics)
RMSE/MAE bar charts
Direction accuracy comparison
Prediction overlay plots
Per-regime performance breakdown
Statistical significance matrix
Feature importance from ensemble models



Expected Findings:

✅ AI models (LSTM, XGBoost) outperform OLS
✅ Models capture feedback loops better with lags
✅ Performance varies by regime (some models better in high volatility)
✅ Ensemble combines best of all approaches
✅ Direction accuracy > 55% suggests predictive power


E. Objective 3: Real-Time AI-Driven Analytical System
From Paper:

"To create and execute an AI-based, real time analytic system to track the dynamics of the market-labour and new economic divergences."

Null Hypothesis (H₀₃):
Beyond static historical analysis, a real-time AI-based analytical system does not offer any new insights into emerging market–labor divergences.
Alternative Hypothesis (H₁₃):
Emerging divergences and regime shifts in market–labor dynamics are effectively captured by an analytical system based on real-time AI.
Analytical Method from Paper:

"An AI dashboard built with Streamlit is intended to:

Use periodic scraping and APIs to absorb real-time data
Update model predictions on a regular basis
Imagine changes in regime, spikes in volatility, and the inertia of the labor market"


Implementation:
ComponentPurposeCopilot PromptFileUpdate FrequencyReal-Time Data FetchingDownload latest VIX, market, unemploymentPrompt 7.1src/real_time_monitor.pyDaily (post-market close)Data PipelinePreprocess latest data with trained scalersPrompt 7.1src/real_time_monitor.pyOn data arrivalLive PredictionsRun all models on latest dataPrompt 7.2src/prediction_updater.pyOn data arrivalModel ConsensusAggregate predictions, measure disagreementPrompt 7.2src/prediction_updater.pyOn prediction runAlert DetectionIdentify divergences, regime shifts, extremesPrompt 7.2src/prediction_updater.pyContinuousStreamlit DashboardInteractive visualization of all abovePrompt 8.5dashboard/pages/05_real_time_monitor.pyEvery 1 hour
Real-Time Dashboard Components (Prompt 8.5):
Page: Real-Time Monitor (dashboard/pages/05_real_time_monitor.py)

Sections:
1. Live Status Cards
   - Current VIX (red if high)
   - Today's return (arrow + color)
   - Current unemployment
   - Latest predictions from all models
   - Model agreement/disagreement indicator

2. Model Consensus
   - All models' next-day predictions
   - Consensus prediction (average)
   - Prediction uncertainty (std)
   - Agreement strength (0-100)
   - Alert if models disagree (yellow/red)

3. Emerging Divergence Detection
   - VIX-Return correlation breakdown
   - Unemployment divergence alert
   - Regime transition probability
   - List of active alerts with severity

4. Rolling Accuracy Tracking
   - 30-day prediction accuracy per model
   - Degradation alerts
   - Per-regime accuracy

5. Market Regime Status
   - Current regime (large indicator)
   - Transition probabilities
   - Days in current regime
   - Historical regime duration

6. Recent Predictions vs Actuals
   - Last 10 trading days table
   - Actual returns vs predictions
   - Model agreement
   - Recent win rate

7. Stress Indicator
   - 0-100 score based on:
     - VIX level
     - Return volatility
     - Model disagreement
     - Regime transition risk
   - Color: Green → Yellow → Red
Alert Types:

prediction_disagreement: Models strongly disagree (std > threshold)
regime_shift_detected: Probability of entering new regime > 60%
unusual_vix_move: VIX change > 2 std in one day
correlation_breakdown: VIX-Return correlation dropping
unemployment_divergence: Labor market not following market
model_accuracy_degradation: Rolling accuracy dropping

Data Update Schedule:
Daily (4:30 PM IST):
- Fetch VIX close
- Fetch market index close
- Fetch any new unemployment data
- Run preprocessing pipeline
- Generate predictions from all models
- Update prediction history
- Generate alerts

Hourly:
- Update dashboard
- Refresh visualizations
- Check for real-time alerts

Weekly:
- Model performance analysis
- Accuracy review
- Potential retraining trigger
Expected Insights from Real-Time System:

✅ Emerging divergences detected before traditional analysis
✅ Regime shifts identified as probability increases
✅ Model consensus provides robust signal
✅ Alerts trigger on meaningful market events
✅ Real-time accuracy tracked and reported


COMPLEXITY SCIENCE MAPPING
Table 1.1 from Paper: Mapping theoretical model to complexity science principles
Complexity PrincipleHow Manifested in This StudyImplementation EvidenceNon-linearityEffects of volatility and unemployment vary across regimesRegime-specific OLS vs ensemble models (Prompt 2.1, 5.1)Feedback loopsMarket prices influence sentiment and future volatilityGranger causality + VAR analysis (Prompt 2.2)EmergenceMarket trends arise from interacting subsystemsModel consensus prediction (Prompt 7.2)AdaptationAgents adjust expectations over timeLSTM networks learning temporal patterns (Prompt 4.1)Path dependencePast crises affect future responsesRolling window analysis, volatility clustering (Prompt 2.1)AsymmetryMarkets react faster than labourLag analysis, different response magnitudes (Prompt 2.1)
Visualization Location:

Dashboard page: dashboard/pages/06_complex_systems.py (Prompt 8.6)
Shows theoretical model, feedback loops, principles evidence


ANALYSIS OUTPUTS MAPPING
Phase 1: Exploratory Analysis
Outputs from Prompts 1.2, 1.3:

analysis/eda/descriptive_statistics.csv
analysis/eda/data_quality_report.txt
analysis/eda/missing_values_visualization.png
analysis/eda/stationarity_tests.csv

Phase 2: Regime Analysis
Outputs from Prompt 2.1:

analysis/regime/regime_timeline.png
analysis/regime/regime_statistics.csv
analysis/regime/rolling_correlations.csv
analysis/regime/volatility_clustering_report.csv
analysis/regime/asymmetric_response.json
analysis/regime/lag_analysis.csv

Phase 3: Model Training & Evaluation
Outputs from Prompts 3.1, 4.1, 5.1:

models/ols_static.pkl
models/ols_lagged.pkl
models/lstm_model.h5
models/random_forest.pkl
models/xgboost.pkl
models/lightgbm.pkl
analysis/models/training_history/
analysis/models/feature_importance.csv

Phase 4: Model Evaluation
Outputs from Prompt 6.1:

analysis/evaluation/performance_metrics.csv
analysis/evaluation/regime_breakdown.csv
analysis/evaluation/statistical_tests.csv
analysis/evaluation/prediction_comparison.png
analysis/evaluation/residual_diagnostics.png

Phase 5: Real-Time Tracking
Outputs from Prompts 7.1, 7.2:

data/cache/latest_data.json
data/cache/prediction_history.csv
data/cache/alerts_log.json
analysis/realtime/daily_predictions.csv
analysis/realtime/accuracy_tracking.csv

Phase 6: Dashboard
Generated by Prompts 8.1-8.7:

Interactive visualizations
Real-time updates
Multi-page navigation
Export functionality


HYPOTHESIS VALIDATION CHECKLIST
H₀₁ / H₁₁: Asymmetric Non-Linear Interactions

 Evidence for H₁₁:

Rolling correlations show variation across regimes
Lag analysis shows non-zero correlations at specific lags
Volatility clustering detected
Asymmetry index > 0 (positive vs negative shocks differ)
Regime-specific analysis shows different patterns per regime


 Test from Prompt 2.1:

Run rolling_correlation()
Run detect_volatility_clustering()
Run analyze_asymmetric_response()
Run lag_analysis()


 Dashboard Evidence: Regime Analysis page shows visual proof

H₀₂ / H₁₂: AI Model Performance

 Evidence for H₁₂:

LSTM RMSE < OLS RMSE by > 5%
Ensemble RMSE < all individual models
Direction accuracy > 55%
Per-regime performance shows AI models stable
Diebold-Mariano test p-value < 0.05
Feedback loops detected (lag effects matter)


 Test from Prompts 3.1, 4.1, 5.1, 6.1:

Compare metrics: compare_performance()
Statistical test: diebold_mariano_test()
Cross-regime: regime_specific_metrics()
Plot: plot_model_comparison()


 Dashboard Evidence: Model Comparison page shows table & charts

H₀₃ / H₁₃: Real-Time System Value

 Evidence for H₁₃:

Real-time predictions agree with actual outcomes > 55% (direction)
Alerts fire before major market moves (predictive value)
Divergences detected show emerging patterns
Regime shifts predicted with > 60% probability
Recent model accuracy > historical average


 Test from Prompts 7.1, 7.2:

Track: get_prediction_history()
Analyze: analyze_prediction_accuracy_realtime()
Monitor: generate_alerts()


 Dashboard Evidence: Real-Time Monitor page shows live KPIs


IMPLEMENTATION SEQUENCE SUMMARY
PHASE 1: Data Foundation (Week 1)
├── Prompt 1.1: Data Ingestion → src/data_ingestion.py
├── Prompt 1.2: Data Quality → src/data_quality.py
├── Prompt 1.3: Feature Engineering → src/feature_engineering.py
└── Prompt 1.4: Pipeline → src/preprocessing_pipeline.py

PHASE 2: Analysis (Week 2)
├── Prompt 2.1: Regime Analysis → src/regime_analysis.py
├── Prompt 2.2: Causality → src/causality_analysis.py
└── Prompt 8.3: Regime Dashboard → dashboard/pages/03_regime_analysis.py

PHASE 3: Baseline & AI Models (Week 3)
├── Prompt 3.1: OLS → src/baseline_models.py
├── Prompt 4.1: LSTM → src/lstm_models.py
└── Prompt 5.1: Ensemble → src/ensemble_models.py

PHASE 4: Evaluation (Week 4)
├── Prompt 6.1: Evaluation → src/model_evaluation.py
├── Prompt 6.2: Comparison → src/comparison_report.py
└── Prompt 8.4: Comparison Dashboard → dashboard/pages/04_model_comparison.py

PHASE 5: Real-Time (Week 5)
├── Prompt 7.1: Real-Time Data → src/real_time_monitor.py
├── Prompt 7.2: Predictions → src/prediction_updater.py
└── Prompt 8.5: Real-Time Dashboard → dashboard/pages/05_real_time_monitor.py

PHASE 6: Complete System (Week 6)
├── Prompt 8.1: Main App → app.py
├── Prompt 8.2: Home/Explorer → dashboard/pages/01_home.py, 02_data_explorer.py
├── Prompt 8.6: Complex Systems → dashboard/pages/06_complex_systems.py
├── Prompt 8.7: Utils → dashboard/utils/
├── Prompt 9.1-9.3: Testing → tests/
└── Prompt 10.1-11.2: Production → Docker, deployment, monitoring

KEY METRICS TO TRACK
From Paper Methodology:
For Objective 1 (Asymmetry & Non-Linearity):

Correlation coefficient (overall and regime-specific)
Lag-wise correlations (which lags significant)
Asymmetry index (positive vs negative shock magnitude ratio)
Volatility clustering frequency
Regime transition probability

For Objective 2 (Model Comparison):

RMSE (lower is better, target < 0.015)
MAE (lower is better, target < 0.010)
MAPE (lower is better, target < 1%)
Direction accuracy (higher is better, target > 55%)
R² (higher is better, target > 0.45)
Per-regime RMSE (should be stable across regimes)
Statistical significance (Diebold-Mariano p-value < 0.05)

For Objective 3 (Real-Time System):

Live prediction accuracy (rolling 30-day)
Alert precision (% of alerts that precede actual moves)
Regime shift detection rate
Model consensus strength (0-100)
System uptime (% data available)
Update lag (hours from market close to latest prediction)