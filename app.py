from datetime import datetime, timedelta
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Three Soldiers at Key Levels", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("Key Level & Pattern Settings")

ticker_options = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X",
    "USDCHF=X", "USDJPY=X", "USDSGD=X", "GC=F", "BZ=F", "ZB=F",
    "BTC-USD", "EURGBP=X", "EURAUD=X", "EURNZD=X", "EURCAD=X",
    "EURCHF=X", "EURJPY=X", "EURSGD=X", "XAUEUR=X", "GBPAUD=X",
    "GBPNZD=X", "GBPCAD=X", "GBPCHF=X", "GBPJPY=X", "GBPSGD=X",
    "AUDNZD=X", "AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "AUDSGD=X",
    "AAPL", "MSFT", "SPY", "QQQ"
]
symbol = st.sidebar.selectbox("Ticker Symbol", options=ticker_options, index=0)

st.sidebar.subheader("Timeframe & History")
timeframe = st.sidebar.selectbox("Timeframe", ["60m", "1d", "1wk", "1mo", "3mo"], index=1)
history_period = st.sidebar.selectbox("History Range", ["1y", "2y", "5y", "10y", "max"], index=2)

st.sidebar.subheader("Key Level Parameters")
swing_window = st.sidebar.slider("Swing Lookback Window (Bars)", min_value=10, max_value=50, value=20, help="Lookback window used to identify local support and resistance swing levels.")
proximity_pct = st.sidebar.slider("Proximity Tolerance (%)", min_value=0.1, max_value=3.0, value=1.0, step=0.1, help="Maximum percentage distance from a key level to be considered 'near'.")

if st.sidebar.button("🔄 Run Analysis"):
    st.rerun()

# ---------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        return None


# ---------------------------------------------------------
# PATTERN & KEY LEVEL DETECTION LOGIC
# ---------------------------------------------------------
def analyze_soldiers_at_key_levels(df, timeframe, swing_window, prox_tolerance):
    if df is None or len(df) < swing_window + 5:
        return pd.DataFrame(), pd.DataFrame()
    
    work_df = df.copy()
    if timeframe in ["60m", "30m", "15m", "5m", "1m"]:
        work_df = work_df.iloc[:-1] # Drop active live candle
        
    white_soldiers_instances = []
    black_soldiers_instances = []
    
    # Calculate rolling swing highs (Resistance) and swing lows (Support)
    work_df['Resistance'] = work_df['High'].rolling(window=swing_window).max()
    work_df['Support'] = work_df['Low'].rolling(window=swing_window).min()
    
    for i in range(swing_window, len(work_df) - 1):
        c1_open = work_df["Open"].iloc[i-3]
        c1_close = work_df["Close"].iloc[i-3]
        
        c2_open = work_df["Open"].iloc[i-2]
        c2_close = work_df["Close"].iloc[i-2]
        
        c3_open = work_df["Open"].iloc[i-1]
        c3_close = work_df["Close"].iloc[i-1]
        
        next_open = work_df["Open"].iloc[i]
        next_close = work_df["Close"].iloc[i]
        next_date = work_df.index[i]
        
        current_support = work_df['Support'].iloc[i-1]
        current_resistance = work_df['Resistance'].iloc[i-1]
        pattern_price = c3_close
        
        # Check proximity to support or resistance (within tolerance %)
        dist_to_support = abs(pattern_price - current_support) / current_support * 100
        dist_to_resistance = abs(pattern_price - current_resistance) / current_resistance * 100
        
        near_support = dist_to_support <= prox_tolerance
        near_resistance = dist_to_resistance <= prox_tolerance
        near_key_level = near_support or near_resistance
        
        # --- THREE WHITE SOLDIERS ---
        is_white_soldiers = (
            (c1_close > c1_open) and 
            (c2_close > c2_open) and 
            (c3_close > c3_open) and
            (c2_close > c1_close) and 
            (c3_close > c2_close)
        )
        
        if is_white_soldiers:
            continued = next_close > c3_close
            white_soldiers_instances.append({
                "Date": next_date,
                "3rd Close": c3_close,
                "Next Close": next_close,
                "Near Key Level": near_key_level,
                "Zone Type": "Support" if near_support else ("Resistance" if near_resistance else "None"),
                "Continued": continued
            })
            
        # --- THREE BLACK SOLDIERS ---
        is_black_soldiers = (
            (c1_close < c1_open) and 
            (c2_close < c2_open) and 
            (c3_close < c3_open) and
            (c2_close < c1_close) and 
            (c3_close < c2_close)
        )
        
        if is_black_soldiers:
            continued = next_close < c3_close
            black_soldiers_instances.append({
                "Date": next_date,
                "3rd Close": c3_close,
                "Next Close": next_close,
                "Near Key Level": near_key_level,
                "Zone Type": "Support" if near_support else ("Resistance" if near_resistance else "None"),
                "Continued": continued
            })
            
    return pd.DataFrame(white_soldiers_instances), pd.DataFrame(black_soldiers_instances)


# ---------------------------------------------------------
# MAIN DASHBOARD UI
# ---------------------------------------------------------
st.title("🛡️ Three Soldiers at Key Levels Analyzer")
st.markdown(f"Tracking **Three White / Black Soldiers** formations near dynamic **Support & Resistance key levels** for **{symbol}** on **{timeframe}**.")

df = fetch_data(symbol, history_period, timeframe)

if df is None or df.empty:
    st.error(f"Could not retrieve data for ticker '{symbol}'.")
else:
    white_df, black_df = analyze_soldiers_at_key_levels(df, timeframe, swing_window, proximity_pct)
    
    col1, col2 = st.columns(2)
    
    # --- THREE WHITE SOLDIERS ---
    with col1:
        st.markdown("### 📈 Three White Soldiers (Bullish)")
        total_white = len(white_df)
        
        if total_white > 0:
            white_near_kl = white_df[white_df["Near Key Level"] == True]
            total_white_kl = len(white_near_kl)
            
            overall_prob = (white_df["Continued"].sum() / total_white) * 100
            kl_prob = (white_near_kl["Continued"].sum() / total_white_kl * 100) if total_white_kl > 0 else 0
            
            st.metric("Total Patterns Found", total_white)
            st.metric("Continuation Probability (Overall)", f"{overall_prob:.1f}%")
            st.metric(f"Continuation Probability (Near Key Level ≤ {proximity_pct}%)", f"{kl_prob:.1f}%", f"{total_white_kl} occurrences")
            
            with st.expander("🔍 View White Soldiers History"):
                st.dataframe(white_df.sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No Three White Soldiers patterns found.")

    # --- THREE BLACK SOLDIERS ---
    with col2:
        st.markdown("### 📉 Three Black Soldiers (Bearish)")
        total_black = len(black_df)
        
        if total_black > 0:
            black_near_kl = black_df[black_df["Near Key Level"] == True]
            total_black_kl = len(black_near_kl)
            
            overall_prob_b = (black_df["Continued"].sum() / total_black) * 100
            kl_prob_b = (black_near_kl["Continued"].sum() / total_black_kl * 100) if total_black_kl > 0 else 0
            
            st.metric("Total Patterns Found", total_black)
            st.metric("Continuation Probability (Overall)", f"{overall_prob_b:.1f}%")
            st.metric(f"Continuation Probability (Near Key Level ≤ {proximity_pct}%)", f"{kl_prob_b:.1f}%", f"{total_black_kl} occurrences")
            
            with st.expander("🔍 View Black Soldiers History"):
                st.dataframe(black_df.sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No Three Black Soldiers patterns found.")