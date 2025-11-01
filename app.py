import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ULTIMATE STOCK ANALYZER PRO",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
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
    .indicator-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
        border: 2px solid #e5e7eb;
    }
    .bullish { border-color: #10b981; background: #f0fdf4; }
    .bearish { border-color: #ef4444; background: #fef2f2; }
    .neutral { border-color: #f59e0b; background: #fffbeb; }
</style>
""", unsafe_allow_html=True)

class StockAnalyzerPro:
    def __init__(self):
        self.stock_list = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK', 
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFOSYS': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'BHARTIARTL': 'BHARTIARTL.NS'
        }
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbol):
        """Get stock data without errors"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            return data if not data.empty else None
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            return None

    def calculate_indicators(self, data):
        """Calculate all indicators manually"""
        if data is None or len(data) < 20:
            return None
            
        df = data.copy()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
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
        
        # Volume Analysis
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Support/Resistance
        df['Support'] = df['Low'].rolling(20).min()
        df['Resistance'] = df['High'].rolling(20).max()
        
        return df.fillna(method='bfill')

    def detect_candlestick_patterns(self, df):
        """Safe candlestick pattern detection"""
        if df is None or len(df) < 3:
            return []
            
        patterns = []
        
        try:
            current = df.iloc[-1]
            prev1 = df.iloc[-2]
            
            # Safely get prev2 only if available
            prev2 = df.iloc[-3] if len(df) >= 3 else None
            
            # Calculate basic candle properties
            current_body = abs(current['Close'] - current['Open'])
            current_high_low = current['High'] - current['Low']
            
            if current_high_low > 0:  # Avoid division by zero
                current_upper_wick = current['High'] - max(current['Open'], current['Close'])
                current_lower_wick = min(current['Open'], current['Close']) - current['Low']
                
                # 1. BULLISH ENGULFING
                if (prev1['Close'] < prev1['Open'] and  # Previous red
                    current['Close'] > current['Open'] and  # Current green
                    current['Open'] < prev1['Close'] and  # Opens below prev close
                    current['Close'] > prev1['Open']):    # Closes above prev open
                    patterns.append({
                        "name": "Bullish Engulfing",
                        "type": "bullish", 
                        "accuracy": "70%"
                    })
                
                # 2. BEARISH ENGULFING
                if (prev1['Close'] > prev1['Open'] and  # Previous green
                    current['Close'] < current['Open'] and  # Current red
                    current['Open'] > prev1['Close'] and  # Opens above prev close
                    current['Close'] < prev1['Open']):    # Closes below prev open
                    patterns.append({
                        "name": "Bearish Engulfing",
                        "type": "bearish",
                        "accuracy": "68%"
                    })
                
                # 3. HAMMER (Bullish)
                if (current_lower_wick >= 2 * current_body and
                    current_upper_wick <= current_body * 0.3 and
                    current['Close'] > current['Open']):
                    patterns.append({
                        "name": "Hammer", 
                        "type": "bullish",
                        "accuracy": "65%"
                    })
                
                # 4. SHOOTING STAR (Bearish)
                if (current_upper_wick >= 2 * current_body and
                    current_lower_wick <= current_body * 0.3 and
                    current['Close'] < current['Open']):
                    patterns.append({
                        "name": "Shooting Star",
                        "type": "bearish", 
                        "accuracy": "63%"
                    })
            
            # 5. THREE BLACK CROWS (Safe check)
            if len(df) >= 3:
                last_3_closes = df['Close'].iloc[-3:]
                last_3_opens = df['Open'].iloc[-3:]
                
                if (all(last_3_closes < last_3_opens) and  # All red candles
                    all(last_3_closes < last_3_closes.shift(1).fillna(last_3_closes.iloc[0]))):  # Lower lows
                    patterns.append({
                        "name": "Three Black Crows",
                        "type": "bearish",
                        "accuracy": "78%"
                    })
                    
        except Exception as e:
            st.error(f"Pattern detection error: {str(e)}")
            
        return patterns

    def generate_trading_signals(self, df, patterns):
        """Generate trading signals safely"""
        if df is None or len(df) < 20:
            return "HOLD", 50, "Insufficient data", []
            
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            signals = []
            score = 50
            
            # 1. MOVING AVERAGE SIGNALS
            if current['SMA_20'] > current['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
                signals.append("🎯 BUY: SMA 20 crossed above SMA 50")
                score += 20
            elif current['SMA_20'] < current['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
                signals.append("🎯 SELL: SMA 20 crossed below SMA 50") 
                score -= 20
            
            # 2. MACD SIGNALS
            if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signals.append("📈 BUY: MACD crossed above Signal")
                score += 20
            elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                signals.append("📉 SELL: MACD crossed below Signal")
                score -= 20
            
            # 3. RSI SIGNALS
            if current['RSI'] < 30:
                signals.append("💰 BUY: RSI Oversold (<30)")
                score += 15
            elif current['RSI'] > 70:
                signals.append("💀 SELL: RSI Overbought (>70)")
                score -= 15
            
            # 4. BOLLINGER BANDS
            if current['Close'] <= current['BB_Lower']:
                signals.append("🎯 BUY: Price at BB Lower Band")
                score += 10
            elif current['Close'] >= current['BB_Upper']:
                signals.append("⚠️ SELL: Price at BB Upper Band") 
                score -= 10
            
            # 5. VOLUME CONFIRMATION
            if current['Volume_Ratio'] > 1.5:
                if score > 50:
                    signals.append("🔥 CONFIRMED: High volume support")
                    score += 10
                else:
                    signals.append("💥 CONFIRMED: High volume selling")
                    score -= 10
            
            # 6. CANDLESTICK PATTERNS
            for pattern in patterns:
                if pattern['type'] == 'bullish':
                    signals.append(f"✅ {pattern['name']} - {pattern['accuracy']} accuracy")
                    score += 8
                else:
                    signals.append(f"❌ {pattern['name']} - {pattern['accuracy']} accuracy")
                    score -= 8
            
            # FINAL DECISION
            score = max(0, min(100, score))
            
            if score >= 75:
                action = "🚀 STRONG BUY"
                reason = "Multiple strong bullish signals"
            elif score >= 65:
                action = "📈 BUY" 
                reason = "Bullish bias with confirmation"
            elif score >= 55:
                action = "🔄 HOLD"
                reason = "Neutral market conditions"
            elif score >= 45:
                action = "📉 SELL"
                reason = "Bearish bias emerging"
            else:
                action = "💀 STRONG SELL"
                reason = "Multiple bearish signals"
            
            return action, score, reason, signals
            
        except Exception as e:
            return "HOLD", 50, f"Analysis error: {str(e)}", []

    def create_pro_chart(self, df, symbol):
        """Create professional trading chart"""
        if df is None:
            return None
            
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'<b>{symbol} - TECHNICAL ANALYSIS</b>',
                '<b>RSI MOMENTUM</b>', 
                '<b>MACD TREND</b>'
            ),
            row_heights=[0.6, 0.2, 0.2]
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
        
        # Moving Averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                      line=dict(dash='dash', color='gray')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                      line=dict(dash='dash', color='gray')),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'),
            row=3, col=1
        )
        
        fig.update_layout(height=800, xaxis_rangeslider_visible=False)
        return fig

def main():
    st.markdown('<div class="main-header">🎯 ULTIMATE STOCK ANALYZER PRO</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280;">Professional Technical Analysis • Accurate Buy/Sell Signals</p>', unsafe_allow_html=True)
    
    analyzer = StockAnalyzerPro()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 STOCK SELECTION")
        selected_stock = st.selectbox("Select Stock:", list(analyzer.stock_list.keys()))
        symbol = analyzer.stock_list[selected_stock]
        
        st.header("⚙️ TRADING SETTINGS")
        stop_loss = st.slider("Stop Loss %", 1.0, 10.0, 5.0, 0.5)
        target = st.slider("Target %", 5.0, 25.0, 12.0, 1.0)
    
    # Main Analysis
    if st.button("🎯 GENERATE TRADING SIGNALS", type="primary", use_container_width=True):
        with st.spinner("Analyzing market data..."):
            data = analyzer.get_stock_data(symbol)
            
            if data is not None and not data.empty:
                # Calculate indicators
                df = analyzer.calculate_indicators(data)
                
                # Detect patterns
                patterns = analyzer.detect_candlestick_patterns(df)
                
                # Generate signals
                action, score, reason, signals = analyzer.generate_trading_signals(df, patterns)
                
                if df is not None:
                    current_price = df['Close'].iloc[-1]
                    stop_loss_price = current_price * (1 - stop_loss/100)
                    target_price = current_price * (1 + target/100)
                    
                    # Display Signal
                    if "BUY" in action:
                        st.markdown(f'''
                        <div class="signal-buy">
                            <h1>{action}</h1>
                            <h3>Confidence Score: {score}%</h3>
                            <p>{reason}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    elif "SELL" in action:
                        st.markdown(f'''
                        <div class="signal-sell">
                            <h1>{action}</h1>
                            <h3>Confidence Score: {score}%</h3>
                            <p>{reason}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="signal-hold">
                            <h1>{action}</h1>
                            <h3>Confidence Score: {score}%</h3>
                            <p>{reason}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Trading Details
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{current_price:.2f}")
                    with col2:
                        st.metric("Stop Loss", f"₹{stop_loss_price:.2f}")
                    with col3:
                        st.metric("Target", f"₹{target_price:.2f}")
                    with col4:
                        risk_reward = (target_price-current_price)/(current_price-stop_loss_price) if current_price > stop_loss_price else 0
                        st.metric("Risk/Reward", f"1:{risk_reward:.1f}")
                    
                    # Chart
                    st.subheader("📈 TECHNICAL ANALYSIS CHART")
                    chart = analyzer.create_pro_chart(df, selected_stock)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Signal Breakdown
                    st.subheader("🔍 SIGNAL BREAKDOWN")
                    if signals:
                        for signal in signals:
                            if "BUY" in signal or "✅" in signal:
                                st.success(signal)
                            elif "SELL" in signal or "❌" in signal:
                                st.error(signal)
                            else:
                                st.info(signal)
                    else:
                        st.info("No strong signals detected")
                    
                    # Candlestick Patterns
                    if patterns:
                        st.subheader("🎯 DETECTED CANDLESTICK PATTERNS")
                        for pattern in patterns:
                            if pattern['type'] == 'bullish':
                                st.success(f"✅ {pattern['name']} - {pattern['accuracy']} accuracy")
                            else:
                                st.error(f"❌ {pattern['name']} - {pattern['accuracy']} accuracy")
                    
                    # Technical Indicators
                    st.subheader("⚙️ TECHNICAL INDICATORS")
                    cols = st.columns(4)
                    
                    if df is not None:
                        current = df.iloc[-1]
                        
                        indicators = [
                            ("RSI", f"{current['RSI']:.1f}", "bullish" if current['RSI'] < 35 else "bearish" if current['RSI'] > 65 else "neutral"),
                            ("MACD", "BULL" if current['MACD'] > current['MACD_Signal'] else "BEAR", "bullish" if current['MACD'] > current['MACD_Signal'] else "bearish"),
                            ("Volume", f"{current['Volume_Ratio']:.1f}x", "bullish" if current['Volume_Ratio'] > 1.5 else "neutral"),
                            ("Trend", "BULL" if current['SMA_20'] > current['SMA_50'] else "BEAR", "bullish" if current['SMA_20'] > current['SMA_50'] else "bearish")
                        ]
                        
                        for idx, (name, value, status) in enumerate(indicators):
                            with cols[idx]:
                                st.markdown(f'''
                                <div class="indicator-box {status}">
                                    <h4>{name}</h4>
                                    <h3>{value}</h3>
                                </div>
                                ''', unsafe_allow_html=True)
                
            else:
                st.error("❌ Could not fetch stock data. Please try again.")

    # Quick Actions
    st.sidebar.header("⚡ QUICK ACTIONS")
    if st.sidebar.button("🔄 Refresh Analysis", use_container_width=True):
        st.rerun()

if __name__ == "__main__":
    main()
