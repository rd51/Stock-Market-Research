# Financial Forecasting Model Comparison Report

**Generated:** 2026-02-08 20:12:31

---

## Model Performance Summary

Key metrics comparison across all models:

| Model | RMSE | MAE | MAPE | R² | Direction_Acc |
|-------|------|-----|------|----|---------------|
| OLS | 0.1030 | 0.0847 | 240.56 | 0.9901 | 97.7% |
| RF | 0.1102 | 0.0896 | 284.58 | 0.9887 | 97.3% |

---

## Key Findings

- **Best RMSE:** OLS (0.1030)
- **Best Direction Accuracy:** OLS (97.7%)
- **Best R² Score:** OLS (0.9901)

---

## Model Rankings

### By RMSE (Lower is Better)

1. **OLS** - RMSE: 0.1030
2. **RF** - RMSE: 0.1102

### By Direction Accuracy (Higher is Better)

1. **OLS** - Direction Acc: 97.7%
2. **RF** - Direction Acc: 97.3%

---

## Recommendations

- **Recommended Model:** OLS
- **Rationale:** Best balance of accuracy and directional prediction
- **Ensemble Weights:**
  - OLS: 0.517
  - RF: 0.483

---

## Important Caveats

- Results are based on the specific test dataset and time period
- Model performance may vary with different market conditions
- Direction accuracy is critical for trading applications
- Consider transaction costs and slippage in real trading
- Regular model retraining is recommended

---

*Report generated automatically by ComparisonReportGenerator*