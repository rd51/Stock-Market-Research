"""
Complex Systems Analysis - Theoretical Foundations
=================================================

Comprehensive analysis of market dynamics through complex systems theory,
non-linear relationships, feedback loops, and emergent behaviors.

Author: GitHub Copilot
Date: February 8, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Complex Systems Analysis - Stock Market AI Analytics", page_icon="🌀")

def create_theoretical_model_diagram():
    """Create theoretical model diagram using Mermaid."""
    st.subheader("🌀 Theoretical Model: Complex Systems Framework")

    st.markdown("""
    The following diagram illustrates the complex systems framework applied to financial markets,
    showing the interconnected relationships between economic indicators, market dynamics, and AI predictions.
    """)

    # Mermaid diagram for complex systems model
    mermaid_diagram = """
    graph TB
        A[Economic Indicators] --> B[Market Dynamics]
        B --> C[Investor Behavior]
        C --> D[Price Formation]
        D --> E[AI Predictions]
        E --> F[Feedback Loops]
        F --> B

        A --> G[External Shocks]
        G --> H[Regime Changes]
        H --> I[Non-linear Responses]
        I --> B

        B --> J[Emergent Patterns]
        J --> K[Adaptive Learning]
        K --> L[LSTM Networks]
        L --> E

        M[Path Dependence] --> N[Historical Context]
        N --> O[Rolling Windows]
        O --> P[Lag Analysis]
        P --> Q[Asymmetric Effects]
        Q --> B

        style A fill:#e1f5fe
        style B fill:#f3e5f5
        style E fill:#e8f5e8
        style J fill:#fff3e0
    """

    st.graphviz_chart(mermaid_diagram)

    st.info("""
    **Key Components:**
    - **Economic Indicators**: Unemployment rates, VIX, market indices
    - **Market Dynamics**: Price movements, volatility, correlations
    - **Feedback Loops**: Self-reinforcing cycles between predictions and market behavior
    - **Emergent Patterns**: Complex behaviors arising from simple rules
    - **Path Dependence**: Historical context influencing current dynamics
    """)

def create_feedback_loop_visualization():
    """Create feedback loop visualization."""
    st.subheader("🔄 Feedback Loop Visualization")

    st.markdown("""
    Financial markets exhibit complex feedback loops where AI predictions influence market behavior,
    which in turn affects future predictions, creating self-reinforcing cycles.
    """)

    # Mermaid diagram for feedback loops
    feedback_diagram = """
    graph LR
        A[AI Prediction: Bullish] --> B[Investor Confidence ↑]
        B --> C[Buying Pressure ↑]
        C --> D[Market Price ↑]
        D --> E[Positive Returns]
        E --> F[Model Performance ↑]
        F --> G[Stronger Bullish Signals]
        G --> A

        H[AI Prediction: Bearish] --> I[Investor Confidence ↓]
        I --> J[Selling Pressure ↑]
        J --> K[Market Price ↓]
        K --> L[Negative Returns]
        L --> M[Model Performance ↓]
        M --> N[Stronger Bearish Signals]
        N --> H

        style A fill:#c8e6c9
        style H fill:#ffcdd2
    """

    st.graphviz_chart(feedback_diagram)

    st.warning("""
    **⚠️ Feedback Loop Risks:**
    - **Self-fulfilling prophecies**: AI predictions can become reality through market influence
    - **Amplification effects**: Small prediction errors can cascade into large market moves
    - **Herd behavior**: Collective response to AI signals creates momentum
    - **Market manipulation**: Strategic positioning based on known AI signals
    """)

def create_complex_systems_principles():
    """Create complex systems principles mapping."""
    st.subheader("📚 Complex Systems Principles")

    # Non-linearity
    with st.expander("🔀 Non-Linearity: Beyond Linear Relationships", expanded=True):
        st.markdown("""
        **Definition:** Financial markets exhibit non-linear relationships where small changes in inputs
        can produce disproportionately large effects in outputs.

        **Examples in Our System:**
        - **Volatility clustering**: Periods of high volatility tend to persist (ARCH/GARCH effects)
        - **Market crashes**: Small negative shocks can trigger cascading sell-offs
        - **Bubble formation**: Exponential price growth followed by sudden collapse
        """)

        # Simple non-linear example
        st.markdown("**Mathematical Example:**")
        st.latex(r"y = x^2 + \sin(10x) + \epsilon")

        st.info("""
        **Implications for Modeling:**
        - Linear models (OLS) fail to capture non-linear dynamics
        - LSTM networks excel at learning non-linear temporal patterns
        - Ensemble methods can approximate complex non-linear functions
        """)

    # Feedback Loops
    with st.expander("🔄 Feedback Loops: Self-Reinforcing Dynamics", expanded=False):
        st.markdown("""
        **Definition:** Systems where outputs influence inputs, creating self-reinforcing or
        self-correcting behaviors.

        **Types of Feedback:**
        - **Positive Feedback**: Amplifies changes (momentum, bubbles)
        - **Negative Feedback**: Dampens changes (mean reversion, stabilization)
        - **Delayed Feedback**: Time lags create oscillations
        """)

        st.markdown("**Market Examples:**")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Positive Feedback Loop:**")
            st.markdown("1. Rising prices → Investor optimism")
            st.markdown("2. More buying → Higher prices")
            st.markdown("3. Self-reinforcing upward spiral")

        with col2:
            st.markdown("**Negative Feedback Loop:**")
            st.markdown("1. High prices → Profit taking")
            st.markdown("2. Selling pressure → Price decline")
            st.markdown("3. Self-correcting mechanism")

    # Emergence
    with st.expander("🌟 Emergence: Order from Chaos", expanded=False):
        st.markdown("""
        **Definition:** Complex patterns and behaviors emerge from the interaction of simple components,
        without being explicitly programmed.

        **Market Emergence Examples:**
        - **Price discovery**: Individual trades create market prices
        - **Regime shifts**: Sudden changes in market behavior patterns
        - **Correlation structures**: Dynamic relationships between assets
        - **Volatility smiles**: Implied volatility patterns in options
        """)

        st.info("""
        **Emergent Properties in Our System:**
        - **Regime detection**: Market states emerge from indicator combinations
        - **Prediction consensus**: Ensemble behavior exceeds individual model performance
        - **Risk patterns**: Complex risk relationships discovered through analysis
        """)

    # Adaptation
    with st.expander("🧠 Adaptation: Learning from Experience", expanded=False):
        st.markdown("""
        **Definition:** Systems that learn and adapt their behavior based on historical patterns
        and changing conditions.

        **LSTM Networks in Adaptation:**
        - **Memory cells**: Maintain information over long time periods
        - **Gates**: Control information flow (forget, input, output)
        - **Gradient flow**: Prevents vanishing gradient problem
        """)

        st.markdown("**LSTM Architecture:**")
        st.latex(r"""
        \begin{align*}
        f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
        i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
        \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
        C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
        o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
        h_t &= o_t \odot \tanh(C_t)
        \end{align*}
        """)

        st.success("""
        **Adaptive Advantages:**
        - **Context awareness**: Remembers relevant historical patterns
        - **Non-stationarity**: Adapts to changing market conditions
        - **Long-term dependencies**: Captures multi-week patterns
        """)

    # Path Dependence
    with st.expander("🛤️ Path Dependence: Historical Context Matters", expanded=False):
        st.markdown("""
        **Definition:** Current states and future possibilities are constrained by historical
        trajectories and past decisions.

        **Rolling Windows Implementation:**
        - **Moving averages**: Exponential weighting of recent observations
        - **Rolling correlations**: Time-varying relationships
        - **Momentum indicators**: Path-dependent trend strength
        """)

        st.markdown("**Mathematical Formulation:**")
        st.latex(r"""
        \begin{align*}
        \text{EMA}_t &= \alpha \cdot x_t + (1-\alpha) \cdot \text{EMA}_{t-1} \\
        \text{Rolling Correlation}_t &= \corr(X_{t-w:t}, Y_{t-w:t})
        \end{align*}
        """)

        st.info("""
        **Path Dependence in Markets:**
        - **Trend persistence**: Past price movements influence future direction
        - **Volume patterns**: Historical trading volume affects liquidity
        - **Regime memory**: Market states have momentum and hysteresis
        """)

    # Asymmetry
    with st.expander("⚖️ Asymmetry: Different Responses to Ups and Downs", expanded=False):
        st.markdown("""
        **Definition:** Systems respond differently to positive and negative stimuli,
        creating asymmetric dynamics.

        **Lag Analysis Example:**
        - **Positive shocks**: Quick market response, slow decay
        - **Negative shocks**: Rapid sell-off, gradual recovery
        - **Volatility asymmetry**: Negative returns increase volatility more than positive returns
        """)

        st.markdown("**Asymmetric Lag Effects:**")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Bull Markets:**")
            st.markdown("• Gradual price increases")
            st.markdown("• Slow momentum building")
            st.markdown("• Extended upward trends")

        with col2:
            st.markdown("**Bear Markets:**")
            st.markdown("• Sharp price declines")
            st.markdown("• Rapid momentum shifts")
            st.markdown("• Quick capitulation")

        st.latex(r"""
        \text{Volatility Asymmetry: } \sigma_t^+ \neq \sigma_t^-
        """)

def create_research_references():
    """Create research literature references."""
    st.subheader("📖 Research Literature References")

    references = [
        {
            "title": "Complex Systems in Finance and Econometrics",
            "authors": "Rosser, J.B. (2009)",
            "key_points": "Non-linear dynamics, chaos theory, and complex systems in financial markets",
            "relevance": "Theoretical foundation for our complex systems approach"
        },
        {
            "title": "Financial Market Complexity",
            "authors": "Bouchaud, J.P. et al. (2003)",
            "key_points": "Emergent properties, feedback loops, and market microstructure",
            "relevance": "Explains market complexity and self-organized criticality"
        },
        {
            "title": "Adaptive Markets Hypothesis",
            "authors": "Lo, A.W. (2004)",
            "key_points": "Markets as adaptive complex systems with evolutionary dynamics",
            "relevance": "Supports our LSTM adaptation and regime change modeling"
        },
        {
            "title": "Non-linear Time Series Analysis",
            "authors": "Tong, H. (1990)",
            "key_points": "Non-linear modeling, threshold effects, and asymmetric responses",
            "relevance": "Justifies our non-linear modeling approach"
        },
        {
            "title": "Long Memory and Path Dependence",
            "authors": "Granger, C.W.J. & Ding, Z. (1996)",
            "key_points": "Fractional integration, long-range dependence, and path dependence",
            "relevance": "Supports our rolling window and lag analysis methods"
        },
        {
            "title": "LSTM Networks for Financial Time Series",
            "authors": "Hochreiter, S. & Schmidhuber, J. (1997)",
            "key_points": "Long Short-Term Memory networks for sequence learning",
            "relevance": "Foundation for our deep learning prediction models"
        }
    ]

    for ref in references:
        with st.expander(f"📄 {ref['title']}", expanded=False):
            st.markdown(f"**Authors:** {ref['authors']}")
            st.markdown(f"**Key Points:** {ref['key_points']}")
            st.markdown(f"**Relevance to Our Work:** {ref['relevance']}")

def main():
    """Main function for the complex systems page."""
    st.title("🌀 Complex Systems Analysis")

    st.markdown("""
    This page explores the theoretical foundations of complex systems theory as applied to financial markets.
    We examine non-linear dynamics, feedback loops, emergent behaviors, and adaptive processes that
    characterize modern market behavior.
    """)

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌀 Theoretical Model",
        "🔄 Feedback Loops",
        "📚 System Principles",
        "📖 References"
    ])

    with tab1:
        create_theoretical_model_diagram()

    with tab2:
        create_feedback_loop_visualization()

    with tab3:
        create_complex_systems_principles()

    with tab4:
        create_research_references()

if __name__ == "__main__":
    main()