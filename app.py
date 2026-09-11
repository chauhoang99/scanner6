from datetime import datetime, timedelta
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Three Soldiers Pattern Predictor", layout="wide")

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
st.sidebar.header("Pattern Settings")

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
# PATTERN DETECTION LOGIC
# ---------------------------------------------------------
def analyze_three_soldiers(df, timeframe):
    if df is None or len(df) < 5:
        return pd.DataFrame(), pd.DataFrame()
    
    work_df = df.copy()
    if timeframe in ["60m", "30m", "15m", "5m", "1m"]:
        work_df = work_df.iloc[:-1] # Drop active live candle
        
    white_soldiers_instances = []
    black_soldiers_instances = []
    
    # We need at least 3 candles for the pattern + 1 future candle to check continuation
    for i in range(3, len(work_df) - 1):
        c1_open = work_df["Open"].iloc[i-3]
        c1_close = work_df["Close"].iloc[i-3]
        
        c2_open = work_df["Open"].iloc[i-2]
        c2_close = work_df["Close"].iloc[i-2]
        
        c3_open = work_df["Open"].iloc[i-1]
        c3_close = work_df["Close"].iloc[i-1]
        
        next_open = work_df["Open"].iloc[i]
        next_close = work_df["Close"].iloc[i]
        next_date = work_df.index[i]
        
        # --- THREE WHITE SOLDIERS ---
        # 1. All three are green candles (Close > Open)
        # 2. Each close is higher than the previous close
        # 3. Each opens within or near the previous body (simplified as c2 opens > c1 open, c3 opens > c2 open or standard consecutive highs)
        is_white_soldiers = (
            (c1_close > c1_open) and 
            (c2_close > c2_open) and 
            (c3_close > c3_open) and
            (c2_close > c1_close) and 
            (c3_close > c2_close)
        )
        
        if is_white_soldiers:
            # Check continuation: Did the next candle close higher than the 3rd soldier's close?
            continued = next_close > c3_close
            white_soldiers_instances.append({
                "Date": next_date,
                "Pattern": "Three White Soldiers",
                3: c3_close,
                "Next Close": next_close,
                "Continued": continued
            })
            
        # --- THREE BLACK SOLDIERS ---
        # 1. All three are red candles (Close < Open)
        # 2. Each close is lower than the previous close
        is_black_soldiers = (
            (c1_close < c1_open) and 
            (c2_close < c2_open) and 
            (c3_close < c3_open) and
            (c2_close < c1_close) and 
            (c3_close < c2_close)
        )
        
        if is_black_soldiers:
            # Check continuation: Did the next candle close lower than the 3rd soldier's close?
            continued = next_close < c3_close
            black_soldiers_instances.append({
                "Date": next_date,
                "Pattern": "Three Black Soldiers",
                "3rd Close": c3_close,
                "Next Close": next_close,
                "Continued": continued
            })
            
    return pd.DataFrame(white_soldiers_instances), pd.DataFrame(black_soldiers_instances)


# ---------------------------------------------------------
# MAIN DASHBOARD UI
# ---------------------------------------------------------
st.title("🛡️ Three Soldiers Candlestick Pattern Analyzer")
st.markdown(f"Tracking historical occurrences and continuation probabilities of **Three White Soldiers** and **Three Black Soldiers** for **{symbol}** on **{timeframe}**.")

df = fetch_data(symbol, history_period, timeframe)

if df is None or df.empty:
    st.error(f"Could not retrieve data for ticker '{symbol}'.")
else:
    white_df, black_df = analyze_three_soldiers(df, timeframe)
    
    col1, col2 = st.columns(2)
    
    # --- THREE WHITE SOLDIERS METRICS ---
    with col1:
        st.markdown("### 📈 Three White Soldiers (Bullish)")
        total_white = len(white_df)
        
        if total_white > 0:
            continued_white = white_df["Continued"].sum()
            white_prob = (continued_white / total_white) * 100
            
            st.metric("Total Patterns Found", total_white)
            st.metric("Continuation Probability (Next Candle Closes Higher)", f"{white_prob:.1f}%", f"{continued_white} / {total_white} times")
            
            with st.expander("🔍 View White Soldiers History"):
                st.dataframe(white_df.sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No Three White Soldiers patterns found in the selected history range.")

    # --- THREE BLACK SOLDIERS METRICS ---
    with col2:
        st.markdown("### 📉 Three Black Soldiers (Bearish)")
        total_black = len(black_df)
        
        if total_black > 0:
            continued_black = black_df["Continued"].sum()
            black_prob = (continued_black / total_black) * 100
            
            st.metric("Total Patterns Found", total_black)
            st.metric("Continuation Probability (Next Candle Closes Lower)", f"{black_prob:.1f}%", f"{continued_black} / {total_black} times")
            
            with st.expander("🔍 View Black Soldiers History"):
                st.dataframe(black_df.sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No Three Black Soldiers patterns found in the selected history range.")