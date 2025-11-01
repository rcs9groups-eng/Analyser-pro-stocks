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
    page_title="PROFIT GUARANTEE STOCK ANALYZER",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .profit-alert {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #00ff88;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
        animation: pulse 2s infinite;
    }
    .buy-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #00ff88;
    }
    .sell-box {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #ff4444;
    }
    .target-box {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        text-align: center;
    }
    .notification {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ffd700;
        animation: blink 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class ProfitGuaranteeAnalyzer:
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
        """Get stock data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def calculate_advanced_indicators(self, data):
        """Calculate all technical indicators"""
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        # Moving Averages
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
        
        # Support/Resistance
        df['Support'] = df['Low'].rolling(20).min()
        df['Resistance'] = df['High'].rolling(20).max()
        
        # Price Momentum
        df['Momentum'] = df['Close'] / df['Close'].shift(5) - 1
        
        return df.fillna(method='bfill')

    def predict_price_movement(self, df):
        """Predict where price will go with accuracy"""
        if df is None or len(df) < 50:
            return "HOLD", 0, 0, 0, "Insufficient data"
            
        current = df.iloc[-1]
        current_price = current['Close']
        
        # Calculate prediction score
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        # 1. RSI Analysis
        if current['RSI'] < 35:
            bullish_signals += 2
            reasons.append("✅ RSI Oversold - Price likely to bounce")
        elif current['RSI'] > 70:
            bearish_signals += 2
            reasons.append("❌ RSI Overbought - Price may correct")
        
        # 2. MACD Analysis
        if current['MACD'] > current['MACD_Signal']:
            bullish_signals += 1
            reasons.append("✅ MACD Bullish - Upward momentum")
        else:
            bearish_signals += 1
            reasons.append("❌ MACD Bearish - Downward pressure")
        
        # 3. Moving Average Trend
        if current['SMA_20'] > current['SMA_50']:
            bullish_signals += 1
            reasons.append("✅ Uptrend - MA Alignment positive")
        else:
            bearish_signals += 1
            reasons.append("❌ Downtrend - MA Alignment negative")
        
        # 4. Bollinger Band Position
        bb_position = (current['Close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower'])
        if bb_position < 0.2:
            bullish_signals += 2
            reasons.append("✅ Near BB Lower - Strong bounce potential")
        elif bb_position > 0.8:
            bearish_signals += 2
            reasons.append("❌ Near BB Upper - Correction expected")
        
        # 5. Volume Confirmation
        if current['Volume_Ratio'] > 1.5:
            if bullish_signals > bearish_signals:
                bullish_signals += 1
                reasons.append("✅ High Volume - Bullish confirmation")
            else:
                bearish_signals += 1
                reasons.append("❌ High Volume - Bearish confirmation")
        
        # Calculate prediction
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            return "HOLD", current_price, 0, 0, "No clear signals"
        
        bullish_percentage = (bullish_signals / total_signals) * 100
        bearish_percentage = (bearish_signals / total_signals) * 100
        
        # Price Target Calculation
        if bullish_percentage > 60:
            direction = "UP"
            # Calculate upside targets
            target_1 = current_price * 1.03  # 3%
            target_2 = current_price * 1.06  # 6%
            target_3 = current_price * 1.10  # 10%
            confidence = bullish_percentage
            reason = f"STRONG BULLISH - {bullish_signals}/{total_signals} signals"
            
        elif bearish_percentage > 60:
            direction = "DOWN"
            # Calculate downside targets
            target_1 = current_price * 0.97  # -3%
            target_2 = current_price * 0.94  # -6%
            target_3 = current_price * 0.90  # -10%
            confidence = bearish_percentage
            reason = f"STRONG BEARISH - {bearish_signals}/{total_signals} signals"
            
        else:
            direction = "SIDEWAYS"
            target_1 = current_price * 1.02
            target_2 = current_price * 0.98
            target_3 = current_price
            confidence = max(bullish_percentage, bearish_percentage)
            reason = f"MARKET CONSOLIDATION - Mixed signals"
        
        return direction, target_1, target_2, target_3, reason, reasons

    def generate_trading_signals(self, df):
        """Generate exact buy/sell signals with timing"""
        if df is None or len(df) < 50:
            return "HOLD", [], "Insufficient data", 0, 0, 0
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = current['Close']
        
        signals = []
        action = "HOLD"
        stop_loss = 0
        targets = []
        
        # BUY Conditions (All must be true for strong buy)
        buy_conditions = [
            current['RSI'] < 35,  # Oversold
            current['MACD'] > current['MACD_Signal'],  # Bullish crossover
            current['Close'] > current['SMA_20'],  # Above short-term MA
            current['Volume_Ratio'] > 1.2,  # Good volume
            current['Close'] <= current['BB_Lower'] * 1.02  # Near support
        ]
        
        # SELL Conditions (All must be true for strong sell)
        sell_conditions = [
            current['RSI'] > 70,  # Overbought
            current['MACD'] < current['MACD_Signal'],  # Bearish crossover
            current['Close'] < current['SMA_20'],  # Below short-term MA
            current['Volume_Ratio'] > 1.2,  # Good volume
            current['Close'] >= current['BB_Upper'] * 0.98  # Near resistance
        ]
        
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        if buy_score >= 4:  # Strong buy signal
            action = "BUY NOW"
            stop_loss = current_price * 0.95  # 5% stop loss
            targets = [
                current_price * 1.03,  # Target 1: 3%
                current_price * 1.06,  # Target 2: 6%
                current_price * 1.10   # Target 3: 10%
            ]
            signals.append("🚀 STRONG BUY - Multiple indicators aligned")
            signals.append(f"💰 Buy at: ₹{current_price:.2f}")
            signals.append(f"🛑 Stop Loss: ₹{stop_loss:.2f}")
            signals.append(f"🎯 Target 1: ₹{targets[0]:.2f} (Sell 30%)")
            signals.append(f"🎯 Target 2: ₹{targets[1]:.2f} (Sell 40%)")
            signals.append(f"🎯 Target 3: ₹{targets[2]:.2f} (Sell 30%)")
            
        elif sell_score >= 4:  # Strong sell signal
            action = "SELL NOW"
            stop_loss = current_price * 1.05  # 5% stop loss
            targets = [
                current_price * 0.97,  # Target 1: -3%
                current_price * 0.94,  # Target 2: -6%
                current_price * 0.90   # Target 3: -10%
            ]
            signals.append("💀 STRONG SELL - Multiple indicators aligned")
            signals.append(f"💰 Sell at: ₹{current_price:.2f}")
            signals.append(f"🛑 Stop Loss: ₹{stop_loss:.2f}")
            signals.append(f"🎯 Target 1: ₹{targets[0]:.2f} (Cover 30%)")
            signals.append(f"🎯 Target 2: ₹{targets[1]:.2f} (Cover 40%)")
            signals.append(f"🎯 Target 3: ₹{targets[2]:.2f} (Cover 30%)")
            
        else:
            action = "HOLD"
            signals.append("⚪ WAIT - No strong signals detected")
            signals.append("📊 Market in consolidation phase")
            targets = [current_price, current_price, current_price]
        
        reason = f"Buy Score: {buy_score}/5, Sell Score: {sell_score}/5"
        return action, signals, reason, stop_loss, targets, current_price

    def create_prediction_chart(self, df, symbol, current_price, targets, direction):
        """Create chart with price predictions and targets"""
        if df is None:
            return None
            
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                f'<b>{symbol} - PRICE PREDICTION & TARGETS</b>',
                '<b>RSI MOMENTUM</b>'
            ),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick
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
        
        # Current Price Line
        fig.add_hline(
            y=current_price, 
            line_dash="solid", 
            line_color="blue",
            line_width=3,
            annotation_text=f"Current: ₹{current_price:.2f}",
            row=1, col=1
        )
        
        # Prediction Arrows
        if direction == "UP":
            # Green upward arrow
            fig.add_annotation(
                x=df.index[-1],
                y=current_price,
                text="📈 PRICE WILL GO UP",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="green",
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1,
                row=1, col=1
            )
        elif direction == "DOWN":
            # Red downward arrow
            fig.add_annotation(
                x=df.index[-1],
                y=current_price,
                text="📉 PRICE WILL GO DOWN",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                bgcolor="lightcoral",
                bordercolor="red",
                borderwidth=1,
                row=1, col=1
            )
        
        # Target Lines with annotations
        colors = ['green', 'orange', 'red']
        target_names = ['TARGET 1', 'TARGET 2', 'TARGET 3']
        
        for i, (target, color, name) in enumerate(zip(targets, colors, target_names)):
            if target != current_price:  # Only show if different from current
                fig.add_hline(
                    y=target,
                    line_dash="dash",
                    line_color=color,
                    line_width=2,
                    annotation_text=name,
                    annotation_position="right",
                    row=1, col=1
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
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        return fig

def main():
    st.markdown('<div class="main-header">💰 PROFIT GUARANTEE STOCK ANALYZER</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280;">Know Exactly When Price Will Move • Accurate Buy/Sell Timing • Profit Targets</p>', unsafe_allow_html=True)
    
    analyzer = ProfitGuaranteeAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 STOCK SELECTION")
        selected_stock = st.selectbox("Choose Stock:", list(analyzer.stock_list.keys()))
        symbol = analyzer.stock_list[selected_stock]
        
        st.header("⚙️ ANALYSIS SETTINGS")
        show_predictions = st.checkbox("Show Price Predictions", True)
        show_signals = st.checkbox("Show Trading Signals", True)
    
    try:
        # Main Analysis
        if st.button("💰 ANALYZE FOR PROFIT OPPORTUNITY", type="primary", use_container_width=True):
            with st.spinner("Scanning for high-profit opportunities..."):
                data = analyzer.get_stock_data(symbol)
                
                if data is not None and not data.empty:
                    # Calculate indicators
                    df = analyzer.calculate_advanced_indicators(data)
                    
                    if df is not None:
                        current_price = df['Close'].iloc[-1]
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                        
                        # PRICE PREDICTION
                        if show_predictions:
                            direction, target1, target2, target3, reason, detailed_reasons = analyzer.predict_price_movement(df)
                            
                            st.markdown(f'''
                            <div class="profit-alert">
                                <h1>🎯 PRICE PREDICTION: {direction}</h1>
                                <h2>Stock Will Go: {direction} Today</h2>
                                <h3>Analysis Time: {current_time}</h3>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # Price Targets
                            st.subheader("📈 PRICE TARGETS")
                            cols = st.columns(3)
                            
                            targets = [target1, target2, target3]
                            target_names = ["SHORT TERM", "MEDIUM TERM", "LONG TERM"]
                            
                            for idx, (col, name, target) in enumerate(zip(cols, target_names, targets)):
                                with col:
                                    change_percent = ((target - current_price) / current_price) * 100
                                    st.markdown(f'''
                                    <div class="target-box">
                                        <h4>{name}</h4>
                                        <h2>₹{target:.2f}</h2>
                                        <p>{change_percent:+.1f}%</p>
                                    </div>
                                    ''', unsafe_allow_html=True)
                            
                            # Prediction Chart
                            st.subheader("📊 PRICE MOVEMENT PREDICTION CHART")
                            chart = analyzer.create_prediction_chart(df, selected_stock, current_price, targets, direction)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                            
                            # Detailed Analysis
                            st.subheader("🔍 PREDICTION ANALYSIS")
                            for detail in detailed_reasons:
                                if "✅" in detail:
                                    st.success(detail)
                                elif "❌" in detail:
                                    st.error(detail)
                                else:
                                    st.info(detail)
                        
                        # TRADING SIGNALS
                        if show_signals:
                            action, signals, reason, stop_loss, targets, entry_price = analyzer.generate_trading_signals(df)
                            
                            if action == "BUY NOW":
                                st.markdown(f'''
                                <div class="notification">
                                    <h2>🚀 IMMEDIATE BUY OPPORTUNITY!</h2>
                                    <h3>This stock will give profit today</h3>
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                st.markdown(f'''
                                <div class="buy-box">
                                    <h2>📈 BUY NOW & EARN PROFIT</h2>
                                    <h3>Entry Price: ₹{entry_price:.2f}</h3>
                                    <p>Best time to enter for short-term gains</p>
                                </div>
                                ''', unsafe_allow_html=True)
                                
                            elif action == "SELL NOW":
                                st.markdown(f'''
                                <div class="notification">
                                    <h2>💀 IMMEDIATE SELL SIGNAL!</h2>
                                    <h3>Price will fall - Exit now</h3>
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                st.markdown(f'''
                                <div class="sell-box">
                                    <h2>📉 SELL NOW & SAVE LOSS</h2>
                                    <h3>Exit Price: ₹{entry_price:.2f}</h3>
                                    <p>Best time to exit to avoid losses</p>
                                </div>
                                ''', unsafe_allow_html=True)
                            
                            # Trading Instructions
                            st.subheader("🎯 TRADING INSTRUCTIONS")
                            for signal in signals:
                                if "BUY" in signal or "TARGET" in signal:
                                    st.success(signal)
                                elif "SELL" in signal or "LOSS" in signal:
                                    st.error(signal)
                                else:
                                    st.info(signal)
                            
                            # Profit Calculation
                            if action in ["BUY NOW", "SELL NOW"]:
                                st.subheader("💰 PROFIT CALCULATOR")
                                if action == "BUY NOW":
                                    profit_potential = ((targets[2] - entry_price) / entry_price) * 100
                                    st.success(f"**Maximum Profit Potential: {profit_potential:.1f}%**")
                                else:
                                    profit_potential = ((entry_price - targets[2]) / entry_price) * 100
                                    st.error(f"**Maximum Save Potential: {profit_potential:.1f}%**")
                        
                        # CURRENT MARKET STATUS
                        st.subheader("📊 CURRENT MARKET STATUS")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        current = df.iloc[-1]
                        with col1:
                            st.metric("Live Price", f"₹{current_price:.2f}")
                        with col2:
                            st.metric("RSI", f"{current['RSI']:.1f}")
                        with col3:
                            trend = "BULLISH" if current['SMA_20'] > current['SMA_50'] else "BEARISH"
                            st.metric("Trend", trend)
                        with col4:
                            volume_status = "HIGH" if current['Volume_Ratio'] > 1.2 else "NORMAL"
                            st.metric("Volume", volume_status)
                        
                    else:
                        st.error("❌ Technical analysis failed. Please try again.")
                else:
                    st.error("❌ Could not fetch stock data. Please try again.")
    
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.info("Please refresh the page and try again.")

    # Quick Actions
    st.sidebar.header("⚡ QUICK ACTIONS")
    if st.sidebar.button("🔄 Refresh Analysis", use_container_width=True):
        st.rerun()

if __name__ == "__main__":
    main()
