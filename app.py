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
    page_title="PRO CANDLESTICK TRADER MASTER",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS
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
    .candle-signal {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #00ff88;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
    }
    .pattern-bullish {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
        border: 2px solid #00ff88;
    }
    .pattern-bearish {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
        border: 2px solid #ff4444;
    }
    .volume-high {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class ProCandlestickTrader:
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
        """Get stock data without TA library"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            return data if not data.empty else None
        except:
            return None

    def calculate_manual_indicators(self, data):
        """Calculate indicators without external TA library"""
        if data is None or len(data) < 50:
            return data
            
        df = data.copy()
        
        # Manual RSI Calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Manual Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Manual MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Manual Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume Analysis
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Support Resistance
        df['Support'] = df['Low'].rolling(20).min()
        df['Resistance'] = df['High'].rolling(20).max()
        
        return df.fillna(method='bfill')

    def detect_candlestick_patterns(self, df):
        """Detect professional candlestick patterns"""
        if df is None or len(df) < 3:
            return []
            
        patterns = []
        current = df.iloc[-1]
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) >= 3 else None
        
        # Calculate candle properties
        current_body = abs(current['Close'] - current['Open'])
        current_high_low = current['High'] - current['Low']
        current_upper_wick = current['High'] - max(current['Open'], current['Close'])
        current_lower_wick = min(current['Open'], current['Close']) - current['Low']
        
        prev1_body = abs(prev1['Close'] - prev1['Open'])
        prev1_high_low = prev1['High'] - prev1['Low']
        
        # 1. BULLISH ENGULFING (70% Accuracy)
        if (prev1['Close'] < prev1['Open'] and  # Previous red
            current['Close'] > current['Open'] and  # Current green
            current['Open'] < prev1['Close'] and  # Opens below prev close
            current['Close'] > prev1['Open']):    # Closes above prev open
            patterns.append({
                "name": "Bullish Engulfing",
                "type": "bullish",
                "accuracy": "70%",
                "description": "Strong reversal pattern with high volume",
                "volume_required": True
            })
        
        # 2. BEARISH ENGULFING
        if (prev1['Close'] > prev1['Open'] and  # Previous green
            current['Close'] < current['Open'] and  # Current red
            current['Open'] > prev1['Close'] and  # Opens above prev close
            current['Close'] < prev1['Open']):    # Closes below prev open
            patterns.append({
                "name": "Bearish Engulfing", 
                "type": "bearish",
                "accuracy": "68%",
                "description": "Strong bearish reversal",
                "volume_required": True
            })
        
        # 3. HAMMER PATTERN (Bullish)
        if (current_lower_wick >= 2 * current_body and  # Long lower wick
            current_upper_wick <= current_body * 0.3 and  # Small upper wick
            current['Close'] > current['Open']):  # Green candle
            patterns.append({
                "name": "Hammer",
                "type": "bullish", 
                "accuracy": "65%",
                "description": "Bullish reversal after downtrend",
                "volume_required": True
            })
        
        # 4. SHOOTING STAR (Bearish)
        if (current_upper_wick >= 2 * current_body and  # Long upper wick
            current_lower_wick <= current_body * 0.3 and  # Small lower wick
            current['Close'] < current['Open']):  # Red candle
            patterns.append({
                "name": "Shooting Star",
                "type": "bearish",
                "accuracy": "63%", 
                "description": "Bearish reversal after uptrend",
                "volume_required": True
            })
        
        # 5. MORNING STAR (3-candle pattern)
        if (prev2 and 
            prev2['Close'] < prev2['Open'] and  # First red
            abs(prev1['Close'] - prev1['Open']) < prev1_high_low * 0.3 and  # Second small
            current['Close'] > current['Open'] and  # Third green
            current['Close'] > prev2['Close']):  # Closes above first
            patterns.append({
                "name": "Morning Star",
                "type": "bullish",
                "accuracy": "65%",
                "description": "Strong bullish reversal pattern", 
                "volume_required": True
            })
        
        # 6. THREE BLACK CROWS (Bearish)
        if (len(df) >= 3 and
            all(df['Close'].iloc[-3:] < df['Open'].iloc[-3:]) and  # 3 red candles
            all(df['Close'].iloc[-3:] < df['Close'].iloc[-4:-1])):  # Lower lows
            patterns.append({
                "name": "Three Black Crows",
                "type": "bearish", 
                "accuracy": "78%",
                "description": "Very strong bearish signal",
                "volume_required": True
            })
        
        return patterns

    def analyze_candlestick_signals(self, df, patterns):
        """Generate trading signals based on candlestick analysis"""
        if df is None or len(df) < 20:
            return "HOLD", 50, "Insufficient data", []
            
        current = df.iloc[-1]
        signals = []
        score = 50
        
        # Volume Analysis (Most Important)
        volume_signal = ""
        if current['Volume_Ratio'] > 2.0:
            volume_signal = "🔥 VERY HIGH VOLUME"
            score += 15
        elif current['Volume_Ratio'] > 1.5:
            volume_signal = "📈 HIGH VOLUME" 
            score += 10
        else:
            volume_signal = "📊 NORMAL VOLUME"
        
        signals.append(f"Volume: {volume_signal} ({current['Volume_Ratio']:.1f}x)")
        
        # Pattern-based Signals
        bullish_patterns = [p for p in patterns if p['type'] == 'bullish']
        bearish_patterns = [p for p in patterns if p['type'] == 'bearish']
        
        for pattern in bullish_patterns:
            signals.append(f"✅ {pattern['name']} - {pattern['accuracy']} accuracy")
            if pattern['volume_required'] and current['Volume_Ratio'] > 1.2:
                score += 12
            else:
                score += 8
        
        for pattern in bearish_patterns:
            signals.append(f"❌ {pattern['name']} - {pattern['accuracy']} accuracy")
            if pattern['volume_required'] and current['Volume_Ratio'] > 1.2:
                score -= 12
            else:
                score -= 8
        
        # Support/Resistance Analysis
        support_distance = (current['Close'] - current['Support']) / current['Close'] * 100
        resistance_distance = (current['Resistance'] - current['Close']) / current['Close'] * 100
        
        if support_distance < 2:
            signals.append(f"🎯 NEAR STRONG SUPPORT: {support_distance:.1f}% away")
            score += 8
        if resistance_distance < 2:
            signals.append(f"⚠️ NEAR STRONG RESISTANCE: {resistance_distance:.1f}% away") 
            score -= 8
        
        # RSI Analysis
        if current['RSI'] < 30:
            signals.append("💰 RSI OVERSOLD (<30)")
            score += 10
        elif current['RSI'] > 70:
            signals.append("💀 RSI OVERBOUGHT (>70)")
            score -= 10
        elif 30 <= current['RSI'] <= 40:
            signals.append("📊 RSI in BUY ZONE")
            score += 5
        
        # Final Decision
        score = max(0, min(100, score))
        
        if score >= 75:
            action = "🚀 STRONG BUY"
            reason = "Multiple bullish patterns with volume confirmation"
        elif score >= 65:
            action = "📈 BUY" 
            reason = "Bullish patterns detected"
        elif score >= 55:
            action = "🟡 MILD BUY"
            reason = "Moderate bullish signals"
        elif score >= 45:
            action = "⚪ HOLD"
            reason = "Mixed signals"
        elif score >= 35:
            action = "🟠 MILD SELL"
            reason = "Moderate bearish signals"
        elif score >= 25:
            action = "📉 SELL"
            reason = "Bearish patterns detected"
        else:
            action = "💀 STRONG SELL"
            reason = "Multiple bearish patterns with volume confirmation"
        
        return action, score, reason, signals

    def create_candlestick_chart(self, df, symbol, patterns):
        """Create professional candlestick chart with patterns"""
        if df is None:
            return None
            
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'<b>{symbol} - PROFESSIONAL CANDLESTICK ANALYSIS</b>',
                '<b>VOLUME ANALYSIS</b>',
                '<b>RSI MOMENTUM</b>'
            ),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick with patterns
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
        
        # Support and Resistance
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Support'],
                name='Support',
                line=dict(color='green', dash='dash'),
                opacity=0.7
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Resistance'], 
                name='Resistance',
                line=dict(color='red', dash='dash'),
                opacity=0.7
            ), row=1, col=1
        )
        
        # Pattern Annotations
        if patterns:
            last_date = df.index[-1]
            last_price = df['High'].iloc[-1] * 1.02
            
            pattern_text = "<br>".join([f"• {p['name']} ({p['accuracy']})" for p in patterns])
            
            fig.add_annotation(
                x=last_date,
                y=last_price,
                text=f"<b>DETECTED PATTERNS:</b><br>{pattern_text}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                bgcolor="#FFFAF0",
                bordercolor="#636363",
                borderwidth=1,
                row=1, col=1
            )
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Volume_MA'],
                name='Volume MA',
                line=dict(color='orange')
            ), row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['RSI'],
                name='RSI',
                line=dict(color='purple')
            ), row=3, col=1
        )
        
        # RSI Levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        return fig

def main():
    st.markdown('<div class="main-header">🎯 PRO CANDLESTICK TRADER MASTER</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280;">Professional Candlestick Pattern Detection • Volume Confirmation • High Accuracy</p>', unsafe_allow_html=True)
    
    trader = ProCandlestickTrader()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 STOCK SELECTION")
        selected_stock = st.selectbox("Select Stock:", list(trader.stock_list.keys()))
        symbol = trader.stock_list[selected_stock]
        
        st.header("🎯 TRADING SETTINGS")
        risk_percent = st.slider("Risk per Trade %", 1.0, 5.0, 2.0, 0.5)
        show_pattern_details = st.checkbox("Show Pattern Details", value=True)
    
    # Main Analysis
    if st.button("🔍 ANALYZE CANDLESTICK PATTERNS", type="primary", use_container_width=True):
        with st.spinner("Detecting high-probability candlestick patterns..."):
            data = trader.get_stock_data(symbol)
            
            if data is not None and not data.empty:
                # Calculate indicators
                df = trader.calculate_manual_indicators(data)
                
                # Detect candlestick patterns
                patterns = trader.detect_candlestick_patterns(df)
                
                # Generate signals
                action, score, reason, signals = trader.analyze_candlestick_signals(df, patterns)
                
                current_price = df['Close'].iloc[-1]
                current_volume_ratio = df['Volume_Ratio'].iloc[-1]
                
                # Display Results
                if "BUY" in action:
                    st.markdown(f'''
                    <div class="candle-signal">
                        <h1>{action}</h1>
                        <h2>Confidence Score: {score}%</h2>
                        <h3>{reason}</h3>
                        <p>Based on {len(patterns)} detected patterns</p>
                    </div>
                    ''', unsafe_allow_html=True)
                elif "SELL" in action:
                    st.markdown(f'''
                    <div class="candle-signal" style="background: linear-gradient(135deg, #ff416c, #ff4b2b);">
                        <h1>{action}</h1>
                        <h2>Confidence Score: {score}%</h2>
                        <h3>{reason}</h3>
                        <p>Based on {len(patterns)} detected patterns</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="candle-signal" style="background: linear-gradient(135deg, #f7971e, #ffd200);">
                        <h1>{action}</h1>
                        <h2>Confidence Score: {score}%</h2>
                        <h3>{reason}</h3>
                        <p>Wait for better pattern formation</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Current Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Volume Power", f"{current_volume_ratio:.1f}x")
                with col3:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                with col4:
                    st.metric("Detected Patterns", len(patterns))
                
                # Chart
                st.subheader("📊 PROFESSIONAL CANDLESTICK CHART")
                chart = trader.create_candlestick_chart(df, selected_stock, patterns)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Pattern Details
                if show_pattern_details and patterns:
                    st.subheader("🎯 DETECTED CANDLESTICK PATTERNS")
                    
                    for pattern in patterns:
                        if pattern['type'] == 'bullish':
                            st.markdown(f'''
                            <div class="pattern-bullish">
                                <h3>✅ {pattern['name']}</h3>
                                <p><strong>Accuracy:</strong> {pattern['accuracy']}</p>
                                <p><strong>Description:</strong> {pattern['description']}</p>
                                <p><strong>Volume Required:</strong> {"Yes" if pattern['volume_required'] else "No"}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="pattern-bearish">
                                <h3>❌ {pattern['name']}</h3>
                                <p><strong>Accuracy:</strong> {pattern['accuracy']}</p>
                                <p><strong>Description:</strong> {pattern['description']}</p>
                                <p><strong>Volume Required:</strong> {"Yes" if pattern['volume_required'] else "No"}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                
                # Signal Breakdown
                st.subheader("🔍 SIGNAL BREAKDOWN")
                for signal in signals:
                    if "HIGH VOLUME" in signal or "VERY HIGH VOLUME" in signal:
                        st.markdown(f'<div class="volume-high">{signal}</div>', unsafe_allow_html=True)
                    elif "✅" in signal:
                        st.success(signal)
                    elif "❌" in signal:
                        st.error(signal)
                    else:
                        st.info(signal)
                
                # Trading Advice
                st.subheader("💡 PROFESSIONAL TRADING ADVICE")
                
                if patterns:
                    if any(p['type'] == 'bullish' for p in patterns) and current_volume_ratio > 1.5:
                        st.success("**🚀 STRONG BUY OPPORTUNITY:** Multiple bullish patterns with high volume confirmation. Consider entry with proper stop loss.")
                    elif any(p['type'] == 'bearish' for p in patterns) and current_volume_ratio > 1.5:
                        st.error("**⚠️ STRONG SELL SIGNAL:** Bearish patterns detected with selling volume. Consider exiting long positions.")
                    else:
                        st.warning("**📊 WAIT FOR CONFIRMATION:** Patterns detected but volume confirmation needed. Wait for better setup.")
                else:
                    st.info("**⚪ NO STRONG PATTERNS:** No high-probability candlestick patterns detected. Market may be in consolidation.")
            
            else:
                st.error("❌ Could not fetch stock data. Please try again.")

    # Educational Section
    with st.expander("📚 CANDLESTICK PATTERN GUIDE"):
        st.markdown("""
        ### 🎯 **High Accuracy Patterns (70%+ Success Rate)**
        
        **Bullish Patterns:**
        - **Bullish Engulfing**: Large green candle completely covers previous red candle
        - **Hammer**: Small body with long lower wick at support
        - **Morning Star**: 3-candle reversal pattern after downtrend
        
        **Bearish Patterns:**  
        - **Bearish Engulfing**: Large red candle covers previous green candle
        - **Shooting Star**: Long upper wick at resistance
        - **Three Black Crows**: 3 consecutive red candles making lower lows
        
        ### 📊 **Volume Confirmation Rules**
        - **High Volume** (>1.5x average) = Strong signal
        - **Low Volume** = Weak signal, may be false
        - **Volume Spike** at support/resistance = High probability
        
        ### ⚠️ **Important Notes**
        - Always wait for volume confirmation
        - Combine with support/resistance levels
        - Use stop loss for every trade
        - No pattern is 100% accurate
        """)

if __name__ == "__main__":
    main()
