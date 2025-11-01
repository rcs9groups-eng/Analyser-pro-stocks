import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TA Library import with fallback
try:
    import talib
    TA_AVAILABLE = True
except ImportError:
    import ta
    TA_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="ULTIMATE CONFLUENCE TRADER PRO",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .signal-buy {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #00ff88;
    }
    .signal-sell {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #ff4444;
    }
    .signal-hold {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #ffd200;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitCloudTrader:
    def __init__(self):
        self.stock_list = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK', 
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFOSYS': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS'
        }
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbol):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            return data if not data.empty else None
        except:
            return None

    def calculate_indicators(self, data):
        if data is None or len(data) < 50:
            return data
            
        df = data.copy()
        
        # Basic Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df.fillna(method='bfill')

    def generate_signals(self, df):
        if df is None or len(df) < 50:
            return "HOLD", 50, "Insufficient data", []
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        score = 50
        
        # Moving Average Crossover
        if current['SMA_20'] > current['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            signals.append("🎯 BUY: SMA 20 crossed above SMA 50")
            score += 20
        elif current['SMA_20'] < current['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            signals.append("🎯 SELL: SMA 20 crossed below SMA 50")
            score -= 20
        
        # MACD Crossover
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append("📈 BUY: MACD crossed above Signal")
            score += 20
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append("📉 SELL: MACD crossed below Signal")
            score -= 20
        
        # RSI Signals
        if current['RSI'] < 30:
            signals.append("💰 BUY: RSI Oversold (<30)")
            score += 20
        elif current['RSI'] > 70:
            signals.append("💀 SELL: RSI Overbought (>70)")
            score -= 20
        
        # Volume Confirmation
        if current['Volume_Ratio'] > 1.5 and score > 50:
            signals.append("🔥 CONFIRMED: High volume support")
            score += 10
        elif current['Volume_Ratio'] > 1.5 and score < 50:
            signals.append("💥 CONFIRMED: High volume selling")
            score -= 10
        
        # Final Decision
        if score >= 70:
            action = "🚀 STRONG BUY"
            reason = "Multiple bullish signals"
        elif score >= 60:
            action = "📈 BUY"
            reason = "Bullish bias"
        elif score >= 40:
            action = "🔄 HOLD"
            reason = "Neutral market"
        elif score >= 30:
            action = "📉 SELL"
            reason = "Bearish bias"
        else:
            action = "💀 STRONG SELL"
            reason = "Multiple bearish signals"
        
        return action, min(max(score, 0), 100), reason, signals

    def create_chart(self, df, symbol):
        if df is None:
            return None
            
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price Chart', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # Price Chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        return fig

def main():
    st.markdown('<div class="main-header">🎯 STREAMLIT CLOUD TRADER</div>', unsafe_allow_html=True)
    
    trader = StreamlitCloudTrader()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 STOCK SELECTION")
        selected_stock = st.selectbox("Select Stock:", list(trader.stock_list.keys()))
        symbol = trader.stock_list[selected_stock]
    
    # Main Analysis
    if st.button("🎯 GENERATE TRADING SIGNALS", type="primary"):
        with st.spinner("Analyzing stock data..."):
            data = trader.get_stock_data(symbol)
            
            if data is not None:
                df = trader.calculate_indicators(data)
                action, score, reason, signals = trader.generate_signals(df)
                
                current_price = df['Close'].iloc[-1] if df is not None else 0
                
                # Display Signal
                if "BUY" in action:
                    st.markdown(f'''
                    <div class="signal-buy">
                        <h1>{action}</h1>
                        <h3>Confidence: {score}%</h3>
                        <p>{reason}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                elif "SELL" in action:
                    st.markdown(f'''
                    <div class="signal-sell">
                        <h1>{action}</h1>
                        <h3>Confidence: {score}%</h3>
                        <p>{reason}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="signal-hold">
                        <h1>{action}</h1>
                        <h3>Confidence: {score}%</h3>
                        <p>{reason}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Current Price
                st.metric("Current Price", f"₹{current_price:.2f}")
                
                # Chart
                st.subheader("📈 PRICE CHART")
                chart = trader.create_chart(df, selected_stock)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Signals
                if signals:
                    st.subheader("🔍 SIGNAL BREAKDOWN")
                    for signal in signals:
                        st.write(f"• {signal}")
            else:
                st.error("❌ Could not fetch stock data")

if __name__ == "__main__":
    main()
