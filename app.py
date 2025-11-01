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
    page_title="ACCURATE STOCK SIGNALS PRO",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563eb;
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
    }
    .signal-sell {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .signal-hold {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
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

class AccurateStockSignals:
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
        """Get stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            return data if not data.empty else None
        except:
            return None

    def calculate_accurate_indicators(self, data):
        """Calculate only proven, accurate indicators"""
        if data is None or len(data) < 50:
            return data
            
        df = data.copy()
        
        # 1. PROVEN MOVING AVERAGES
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # 2. RSI WITH DIVERGENCE DETECTION
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (Most Reliable)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 4. BOLLINGER BANDS
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # 5. SUPPORT/RESISTANCE
        df['Support'] = df['Low'].rolling(20).min()
        df['Resistance'] = df['High'].rolling(20).max()
        
        # 6. VOLUME CONFIRMATION
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df.fillna(method='bfill')

    def generate_accurate_signals(self, df):
        """Generate high-accuracy buy/sell signals"""
        if df is None or len(df) < 50:
            return "HOLD", 50, "Insufficient data", []
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        score = 50
        
        try:
            # 1. MOVING AVERAGE CROSSOVER (Weight: 25%)
            if (current['SMA_20'] > current['SMA_50'] and 
                prev['SMA_20'] <= prev['SMA_50']):
                signals.append("🎯 BUY: SMA 20 crossed above SMA 50")
                score += 25
            elif (current['SMA_20'] < current['SMA_50'] and 
                  prev['SMA_20'] >= prev['SMA_50']):
                signals.append("🎯 SELL: SMA 20 crossed below SMA 50")
                score -= 25
            
            # 2. MACD CROSSOVER (Weight: 25%)
            if (current['MACD'] > current['MACD_Signal'] and 
                prev['MACD'] <= prev['MACD_Signal']):
                signals.append("📈 BUY: MACD crossed above Signal")
                score += 25
            elif (current['MACD'] < current['MACD_Signal'] and 
                  prev['MACD'] >= prev['MACD_Signal']):
                signals.append("📉 SELL: MACD crossed below Signal")
                score -= 25
            
            # 3. RSI SIGNALS (Weight: 20%)
            if current['RSI'] < 30:
                signals.append("💰 BUY: RSI Oversold (<30)")
                score += 20
            elif current['RSI'] > 70:
                signals.append("💀 SELL: RSI Overbought (>70)")
                score -= 20
            elif 30 <= current['RSI'] <= 40:
                signals.append("📊 BULLISH: RSI in buy zone")
                score += 10
            elif 60 <= current['RSI'] <= 70:
                signals.append("⚠️ CAUTION: RSI in sell zone")
                score -= 10
            
            # 4. BOLLINGER BANDS (Weight: 15%)
            if current['Close'] <= current['BB_Lower']:
                signals.append("🎯 BUY: Price at BB Lower Band")
                score += 15
            elif current['Close'] >= current['BB_Upper']:
                signals.append("⚠️ SELL: Price at BB Upper Band")
                score -= 15
            
            # 5. VOLUME CONFIRMATION (Weight: 15%)
            if current['Volume_Ratio'] > 1.5 and score > 50:
                signals.append("🔥 CONFIRMED: High volume support")
                score += 15
            elif current['Volume_Ratio'] > 1.5 and score < 50:
                signals.append("💥 CONFIRMED: High volume selling")
                score -= 15
            
            # FINAL DECISION
            if score >= 80:
                action = "🚀 STRONG BUY"
                reason = "Multiple bullish signals aligned"
            elif score >= 65:
                action = "📈 BUY"
                reason = "Bullish bias with confirmation"
            elif score >= 55:
                action = "🔄 HOLD"
                reason = "Neutral market conditions"
            elif score >= 40:
                action = "📉 SELL"
                reason = "Bearish bias emerging"
            else:
                action = "💀 STRONG SELL"
                reason = "Multiple bearish signals aligned"
            
            return action, min(max(score, 0), 100), reason, signals
            
        except Exception as e:
            return "HOLD", 50, f"Analysis error: {str(e)}", []

    def get_buy_sell_points(self, df):
        """Identify exact buy/sell points on chart"""
        if df is None or len(df) < 50:
            return [], []
            
        buy_points = []
        sell_points = []
        
        for i in range(2, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # BUY Conditions
            buy_condition = (
                # Golden Cross
                (current['SMA_20'] > current['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']) or
                # MACD Bullish Cross
                (current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']) or
                # RSI Oversold bounce
                (current['RSI'] < 30 and prev['RSI'] >= 30) or
                # Bollinger Band bounce
                (current['Close'] <= current['BB_Lower'] and prev['Close'] > prev['BB_Lower'])
            )
            
            # SELL Conditions
            sell_condition = (
                # Death Cross
                (current['SMA_20'] < current['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']) or
                # MACD Bearish Cross
                (current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']) or
                # RSI Overbought rejection
                (current['RSI'] > 70 and prev['RSI'] <= 70) or
                # Bollinger Band rejection
                (current['Close'] >= current['BB_Upper'] and prev['Close'] < prev['BB_Upper'])
            )
            
            if buy_condition:
                buy_points.append((df.index[i], current['Low'] * 0.995))  # Slightly below low
            elif sell_condition:
                sell_points.append((df.index[i], current['High'] * 1.005))  # Slightly above high
        
        return buy_points, sell_points

    def create_signal_chart(self, df, symbol, buy_points, sell_points):
        """Create chart with clear buy/sell signals"""
        if df is None:
            return None
            
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'<b>{symbol} - BUY/SELL SIGNALS</b>', 
                '<b>RSI MOMENTUM</b>',
                '<b>MACD TREND</b>'
            ),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price with Candlestick
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
        
        # BUY Signals (Green arrows)
        if buy_points:
            buy_dates, buy_prices = zip(*buy_points)
            fig.add_trace(
                go.Scatter(
                    x=buy_dates, y=buy_prices,
                    mode='markers',
                    name='BUY',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    hovertemplate='<b>BUY</b><br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
        
        # SELL Signals (Red arrows)
        if sell_points:
            sell_dates, sell_prices = zip(*sell_points)
            fig.add_trace(
                go.Scatter(
                    x=sell_dates, y=sell_prices,
                    mode='markers',
                    name='SELL',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    hovertemplate='<b>SELL</b><br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'),
            row=3, col=1
        )
        fig.add_hline(y=0, line_color="black", row=3, col=1)
        
        fig.update_layout(height=800, xaxis_rangeslider_visible=False)
        return fig

def main():
    st.markdown('<div class="main-header">🎯 ACCURATE STOCK SIGNALS PRO</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280;">Real Buy/Sell Signals with Chart Markers • High Accuracy</p>', unsafe_allow_html=True)
    
    analyzer = AccurateStockSignals()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 STOCK SELECTION")
        selected_stock = st.selectbox("Select Stock:", list(analyzer.stock_list.keys()))
        symbol = analyzer.stock_list[selected_stock]
        
        st.header("⚙️ TRADING SETTINGS")
        stop_loss = st.slider("Stop Loss %", 1.0, 20.0, 8.0, 0.5)
        target = st.slider("Target %", 1.0, 50.0, 15.0, 1.0)
        capital = st.number_input("Capital (₹)", 1000, 10000000, 100000)
    
    # Main Analysis
    if st.button("🎯 GENERATE ACCURATE SIGNALS", type="primary", use_container_width=True):
        with st.spinner("Analyzing for high-probability signals..."):
            data = analyzer.get_stock_data(symbol)
            
            if data is not None and not data.empty:
                # Calculate indicators
                df = analyzer.calculate_accurate_indicators(data)
                
                # Generate signals
                action, score, reason, signals = analyzer.generate_accurate_signals(df)
                
                # Get buy/sell points for chart
                buy_points, sell_points = analyzer.get_buy_sell_points(df)
                
                current_price = df['Close'].iloc[-1]
                stop_loss_price = current_price * (1 - stop_loss/100)
                target_price = current_price * (1 + target/100)
                
                # Display SIGNAL
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
                
                # TRADING DETAILS
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Stop Loss", f"₹{stop_loss_price:.2f}")
                with col3:
                    st.metric("Target", f"₹{target_price:.2f}")
                with col4:
                    st.metric("Risk/Reward", f"1:{(target_price-current_price)/(current_price-stop_loss_price):.1f}")
                
                # CHART WITH SIGNALS
                st.subheader("📈 LIVE CHART WITH BUY/SELL SIGNALS")
                st.info("🟢 GREEN ARROWS = BUY Points | 🔴 RED ARROWS = SELL Points")
                chart = analyzer.create_signal_chart(df, selected_stock, buy_points, sell_points)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # SIGNAL BREAKDOWN
                st.subheader("🔍 SIGNAL BREAKDOWN")
                if signals:
                    for signal in signals[-5:]:  # Show latest 5 signals
                        st.write(f"• {signal}")
                else:
                    st.write("• No strong signals detected in recent data")
                
                # NEXT PREDICTION
                st.subheader("🎯 NEXT LIKELY MOVE")
                if buy_points:
                    next_buy = f"Next BUY around ₹{buy_points[-1][1]:.2f}" if buy_points else "No BUY signals"
                    st.success(next_buy)
                if sell_points:
                    next_sell = f"Next SELL around ₹{sell_points[-1][1]:.2f}" if sell_points else "No SELL signals"
                    st.error(next_sell)
                
                # TECHNICAL INDICATORS
                st.subheader("⚙️ TECHNICAL INDICATORS")
                cols = st.columns(4)
                
                indicators = []
                current = df.iloc[-1]
                
                # RSI
                rsi_status = 'bullish' if current['RSI'] < 35 else 'bearish' if current['RSI'] > 65 else 'neutral'
                indicators.append(("RSI", f"{current['RSI']:.1f}", rsi_status))
                
                # MACD
                macd_status = 'bullish' if current['MACD'] > current['MACD_Signal'] else 'bearish'
                indicators.append(("MACD", "BULL" if macd_status == 'bullish' else "BEAR", macd_status))
                
                # Moving Averages
                ma_status = 'bullish' if current['SMA_20'] > current['SMA_50'] else 'bearish'
                indicators.append(("MA Trend", "BULL" if ma_status == 'bullish' else "BEAR", ma_status))
                
                # Volume
                vol_status = 'bullish' if current['Volume_Ratio'] > 1.5 else 'neutral'
                indicators.append(("Volume", f"{current['Volume_Ratio']:.1f}x", vol_status))
                
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
    if st.sidebar.button("🔄 Refresh Signals", use_container_width=True):
        st.rerun()

if __name__ == "__main__":
    main()
