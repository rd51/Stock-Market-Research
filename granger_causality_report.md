# Granger Causality Analysis Report
**Generated:** 2026-02-08 19:44:34

## Executive Summary
- **Variables Analyzed:** 3
- **Significant Relationships:** 2
- **Significance Threshold:** 0.05
- **System Complexity Score:** 16.7/100
- **Network Density:** 33.3%

## Granger Causality Results
- **VIX_to_Returns:** Significant (p = 0.0048)
- **VIX_to_Unemployment:** Not significant (p = 0.1034)
- **Returns_to_VIX:** Not significant (p = 0.2096)
- **Returns_to_Unemployment:** Not significant (p = 0.2252)
- **Unemployment_to_VIX:** Not significant (p = 0.0690)
- **Unemployment_to_Returns:** Significant (p = 0.0010)

## VAR Model Results
- **Total Relationships:** 9
- **Significant Relationships:** 1
- **Strongest Relationships:**
  - VIX -> Returns (coef: 0.002, p: 0.0064)

## Feedback Loops
- **No feedback loops detected**

## Interpretation
### Key Findings:
- Granger causality indicates predictive relationships between variables
- VAR model shows simultaneous relationships and dynamics
- IRF reveals how shocks propagate through the system
- FEVD shows which variables drive forecast uncertainty

### Practical Implications:
- Use significant causal relationships for forecasting
- Monitor variables with high FEVD contributions
- Consider feedback loops for policy analysis
- Regime-specific causality may indicate changing relationships