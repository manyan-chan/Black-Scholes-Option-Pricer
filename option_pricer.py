from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import norm

# --- Configuration ---
st.set_page_config(page_title="Black-Scholes Option Pricer", layout="wide")

# --- Black-Scholes Core Functions ---


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """
    Calculates the Black-Scholes option price for European call or put.

    Args:
        S: Current underlying asset price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized decimal, e.g., 0.05 for 5%)
        sigma: Volatility of the underlying asset (annualized decimal, e.g., 0.2 for 20%)
        option_type: 'call' or 'put'
        q: Annual dividend yield (decimal, e.g., 0.02 for 2%)

    Returns:
        The theoretical option price. Returns 0 if T or sigma is non-positive.
    """
    if T <= 0 or sigma <= 0:
        # Option price is intrinsic value if expired or no volatility
        if option_type == "call":
            return max(0.0, S - K)
        elif option_type == "put":
            return max(0.0, K - S)
        else:
            return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if option_type == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
                d2
            )
        elif option_type == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(
                -d1
            )
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        return price
    except OverflowError:
        st.warning("Calculation resulted in an overflow. Check input parameters.")
        return np.nan  # Indicate calculation failure


def calculate_greeks(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> Dict[str, float]:
    """
    Calculates the Black-Scholes Greeks for a European option.

    Args:
        S: Current underlying asset price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized decimal)
        sigma: Volatility of the underlying asset (annualized decimal)
        q: Annual dividend yield (decimal)

    Returns:
        A dictionary containing the Greeks: delta_call, delta_put, gamma,
        vega, theta_call, theta_put, rho_call, rho_put.
        Returns dict with NaNs if T or sigma non-positive or calculation fails.
    """
    greeks = {
        "delta_call": np.nan,
        "delta_put": np.nan,
        "gamma": np.nan,
        "vega": np.nan,
        "theta_call": np.nan,
        "theta_put": np.nan,
        "rho_call": np.nan,
        "rho_put": np.nan,
    }
    if T <= 0 or sigma <= 0:
        # Greeks are generally undefined or zero at expiration/zero vol
        return greeks

    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        nd1_pdf = norm.pdf(d1)  # Probability Density Function

        # Calculate Greeks
        greeks["delta_call"] = np.exp(-q * T) * norm.cdf(d1)
        greeks["delta_put"] = np.exp(-q * T) * (norm.cdf(d1) - 1)
        greeks["gamma"] = np.exp(-q * T) * nd1_pdf / (S * sigma * np.sqrt(T))
        # Vega: Price change for 1% change in vol (hence / 100)
        greeks["vega"] = S * np.exp(-q * T) * nd1_pdf * np.sqrt(T) / 100

        # Theta: Price change for 1 day decrease in T (hence / 365)
        theta_call_part1 = -(S * np.exp(-q * T) * nd1_pdf * sigma / (2 * np.sqrt(T)))
        theta_call_part2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta_call_part3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        greeks["theta_call"] = (
            theta_call_part1 + theta_call_part2 + theta_call_part3
        ) / 365

        theta_put_part1 = -(S * np.exp(-q * T) * nd1_pdf * sigma / (2 * np.sqrt(T)))
        theta_put_part2 = +r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta_put_part3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        greeks["theta_put"] = (
            theta_put_part1 + theta_put_part2 + theta_put_part3
        ) / 365

        # Rho: Price change for 1% change in r (hence / 100)
        greeks["rho_call"] = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        greeks["rho_put"] = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return greeks

    except OverflowError:
        st.warning("Overflow error during Greek calculation.")
        return greeks  # Return dict with NaNs


# --- Streamlit App UI ---
st.title("📈 Black-Scholes Option Pricing Calculator")
st.markdown("Calculate theoretical prices and Greeks for European options.")

# --- Inputs ---
st.sidebar.header("Input Parameters")

# Using sliders for interactive input
S = st.sidebar.slider(
    "💰 Underlying Asset Price (S)",
    min_value=1.0,
    max_value=500.0,
    value=100.0,
    step=0.5,
    format="%.2f",
)
K = st.sidebar.slider(
    "🎯 Strike Price (K)",
    min_value=1.0,
    max_value=500.0,
    value=100.0,
    step=0.5,
    format="%.2f",
)
T_days = st.sidebar.slider(
    "⏳ Time to Expiration (Days)", min_value=1, max_value=730, value=90, step=1
)  # Max 2 years
r_pct = st.sidebar.slider(
    "🏦 Risk-Free Rate (r %)",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.1,
    format="%.1f%%",
)
sigma_pct = st.sidebar.slider(
    "📊 Volatility (σ %)",
    min_value=1.0,
    max_value=100.0,
    value=20.0,
    step=0.5,
    format="%.1f%%",
)
q_pct = st.sidebar.slider(
    "💸 Dividend Yield (q %)",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1,
    format="%.1f%%",
)

# Convert inputs to decimals / years for calculations
T = T_days / 365.0
r = r_pct / 100.0
sigma = sigma_pct / 100.0
q = q_pct / 100.0

# --- Calculations ---
call_price = black_scholes(S, K, T, r, sigma, "call", q)
put_price = black_scholes(S, K, T, r, sigma, "put", q)
greeks = calculate_greeks(S, K, T, r, sigma, q)

# --- Display Results ---
st.subheader("Calculated Option Prices")
col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Call Option Price",
        f"${call_price:.4f}" if not np.isnan(call_price) else "Error",
    )
with col2:
    st.metric(
        "Put Option Price", f"${put_price:.4f}" if not np.isnan(put_price) else "Error"
    )

st.subheader("Option Greeks")
with st.expander("Show Greeks"):
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        st.metric(
            "Delta (Call)",
            f"{greeks['delta_call']:.4f}"
            if not np.isnan(greeks["delta_call"])
            else "N/A",
        )
        st.metric(
            "Delta (Put)",
            f"{greeks['delta_put']:.4f}"
            if not np.isnan(greeks["delta_put"])
            else "N/A",
        )
    with col_g2:
        st.metric(
            "Gamma",
            f"{greeks['gamma']:.4f}" if not np.isnan(greeks["gamma"]) else "N/A",
        )
        st.metric(
            "Vega (per 1% vol)",
            f"{greeks['vega']:.4f}" if not np.isnan(greeks["vega"]) else "N/A",
        )
    with col_g3:
        st.metric(
            "Theta (Call, per day)",
            f"{greeks['theta_call']:.4f}"
            if not np.isnan(greeks["theta_call"])
            else "N/A",
        )
        st.metric(
            "Theta (Put, per day)",
            f"{greeks['theta_put']:.4f}"
            if not np.isnan(greeks["theta_put"])
            else "N/A",
        )
    with col_g4:
        st.metric(
            "Rho (Call, per 1% rate)",
            f"{greeks['rho_call']:.4f}" if not np.isnan(greeks["rho_call"]) else "N/A",
        )
        st.metric(
            "Rho (Put, per 1% rate)",
            f"{greeks['rho_put']:.4f}" if not np.isnan(greeks["rho_put"]) else "N/A",
        )

st.markdown("---")  # Visual separator

# --- Visualizations ---
st.subheader("Sensitivity Analysis")


# Function to generate data for plots
def generate_plot_data(param_name: str, param_range: np.ndarray) -> pd.DataFrame:
    """Generates call/put prices varying one parameter."""
    data = []
    for val in param_range:
        temp_S, temp_K, temp_T, temp_r, temp_sigma, temp_q = S, K, T, r, sigma, q
        if param_name == "S":
            temp_S = val
        elif param_name == "K":
            temp_K = val  # Note: Plotting vs K might be less common
        elif param_name == "T":
            temp_T = val / 365.0  # Range is in days
        elif param_name == "r":
            temp_r = val / 100.0  # Range is in %
        elif param_name == "sigma":
            temp_sigma = val / 100.0  # Range is in %
        # Ensure non-negative T and sigma for calculation
        if temp_T <= 0 or temp_sigma <= 0:
            continue

        call_p = black_scholes(
            temp_S, temp_K, temp_T, temp_r, temp_sigma, "call", temp_q
        )
        put_p = black_scholes(temp_S, temp_K, temp_T, temp_r, temp_sigma, "put", temp_q)
        if not np.isnan(call_p) and not np.isnan(put_p):
            data.append({param_name: val, "Call Price": call_p, "Put Price": put_p})
    return pd.DataFrame(data)


# 1. Price vs. Underlying Asset Price
st.markdown("#### Option Price vs. Underlying Asset Price (S)")
s_range = np.linspace(max(1.0, S * 0.7), S * 1.3, 50)  # Range around current S
df_s = generate_plot_data("S", s_range)
if not df_s.empty:
    fig_s = px.line(
        df_s,
        x="S",
        y=["Call Price", "Put Price"],
        labels={"S": "Underlying Price ($)", "value": "Option Price ($)"},
        title="Option Price Sensitivity to Underlying Price",
    )
    fig_s.add_vline(
        x=S,
        line_width=1,
        line_dash="dash",
        line_color="grey",
        annotation_text="Current S",
    )
    fig_s.add_vline(
        x=K, line_width=1, line_dash="dot", line_color="red", annotation_text="Strike K"
    )
    st.plotly_chart(fig_s, use_container_width=True)
else:
    st.warning("Could not generate plot data for varying Underlying Price.")


# 2. Price vs. Time to Expiration
st.markdown("#### Option Price vs. Time to Expiration (Days)")
t_range = np.linspace(1, max(T_days * 1.5, T_days + 90), 50)  # Range from 1 day up
df_t = generate_plot_data("T", t_range)
if not df_t.empty:
    fig_t = px.line(
        df_t,
        x="T",
        y=["Call Price", "Put Price"],
        labels={"T": "Time to Expiration (Days)", "value": "Option Price ($)"},
        title="Option Price Sensitivity to Time Decay",
    )
    fig_t.add_vline(
        x=T_days,
        line_width=1,
        line_dash="dash",
        line_color="grey",
        annotation_text="Current T",
    )
    st.plotly_chart(fig_t, use_container_width=True)
else:
    st.warning("Could not generate plot data for varying Time.")

# 3. Price vs. Volatility
st.markdown("#### Option Price vs. Volatility (σ %)")
sigma_range = np.linspace(
    max(1.0, sigma_pct * 0.5), sigma_pct * 1.5, 50
)  # Range around current sigma %
df_sigma = generate_plot_data("sigma", sigma_range)
if not df_sigma.empty:
    fig_sigma = px.line(
        df_sigma,
        x="sigma",
        y=["Call Price", "Put Price"],
        labels={"sigma": "Volatility (%)", "value": "Option Price ($)"},
        title="Option Price Sensitivity to Volatility",
    )
    fig_sigma.add_vline(
        x=sigma_pct,
        line_width=1,
        line_dash="dash",
        line_color="grey",
        annotation_text="Current σ",
    )
    st.plotly_chart(fig_sigma, use_container_width=True)
else:
    st.warning("Could not generate plot data for varying Volatility.")


# --- Disclaimer ---
st.markdown("---")
st.caption(
    "Disclaimer: This calculator provides theoretical option prices based on the Black-Scholes model for European options. Actual market prices may differ due to factors like liquidity, bid-ask spreads, market sentiment, early exercise premium (for American options), and model limitations. This tool is for educational purposes only."
)
