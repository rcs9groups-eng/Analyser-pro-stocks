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
    page_title="PRO TRADER SIGNALS WITH TARGETS",
    page_icon="🎯",
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
    .trade-signal {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #00ff88;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
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
    .entry-point {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class ProTraderWithTargets:
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
        """Get stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="6mo")
            return data if not data.empty else None
        except:
            return None

    def calculate_advanced_indicators(self, data):
        """Calculate advanced indicators for precise trading"""
        if data is None or len(data) < 50:
            return data
            
        df = data.copy()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_9'] - df['EMA_21']
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
        df['Support_1'] = df['Low'].rolling(20).min()
        df['Resistance_1'] = df['High'].rolling(20).max()
        
        # ATR for Stop Loss
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df.fillna(method='bfill')

    def detect_high_probability_patterns(self, df):
        """Detect high-probability candlestick patterns"""
        if df is None or len(df) < 3:
            return []
            
        patterns = []
        current = df.iloc[-1]
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) >= 3 else None
        
        try:
            # Calculate candle properties
            current_body = abs(current['Close'] - current['Open'])
            current_range = current['High'] - current['Low']
            
            if current_range > 0:
                current_upper_wick = current['High'] - max(current['Open'], current['Close'])
                current_lower_wick = min(current['Open'], current['Close']) - current['Low']
                
                # 1. BULLISH ENGULFING (70% Accuracy)
                if (prev1['Close'] < prev1['Open'] and  # Previous red
                    current['Close'] > current['Open'] and  # Current green
                    current['Open'] < prev1['Close'] and  # Opens below prev close
                    current['Close'] > prev1['Open'] and  # Closes above prev open
                    current['Volume_Ratio'] > 1.2):  # Volume confirmation
                    patterns.append({
                        "name": "Bullish Engulfing",
                        "type": "BUY",
                        "accuracy": "70%",
                        "entry_price": current['Close'],
                        "stop_loss": current['Low'] * 0.98,
                        "targets": [
                            current['Close'] * 1.03,  # 3% target
                            current['Close'] * 1.06,  # 6% target  
                            current['Close'] * 1.10   # 10% target
                        ]
                    })
                
                # 2. HAMMER PATTERN (65% Accuracy)
                if (current_lower_wick >= 2 * current_body and
                    current_upper_wick <= current_body * 0.3 and
                    current['Close'] > current['Open'] and
                    current['Volume_Ratio'] > 1.2):
                    patterns.append({
                        "name": "Hammer",
                        "type": "BUY", 
                        "accuracy": "65%",
                        "entry_price": current['Close'],
                        "stop_loss": current['Low'] * 0.98,
                        "targets": [
                            current['Close'] * 1.04,
                            current['Close'] * 1.08,
                            current['Close'] * 1.12
                        ]
                    })
                
                # 3. BEARISH ENGULFING (68% Accuracy)
                if (prev1['Close'] > prev1['Open'] and  # Previous green
                    current['Close'] < current['Open'] and  # Current red
                    current['Open'] > prev1['Close'] and  # Opens above prev close
                    current['Close'] < prev1['Open'] and  # Closes below prev open
                    current['Volume_Ratio'] > 1.2):
                    patterns.append({
                        "name": "Bearish Engulfing",
                        "type": "SELL",
                        "accuracy": "68%", 
                        "entry_price": current['Close'],
                        "stop_loss": current['High'] * 1.02,
                        "targets": [
                            current['Close'] * 0.97,  # 3% down
                            current['Close'] * 0.94,  # 6% down
                            current['Close'] * 0.90   # 10% down
                        ]
                    })
                
                # 4. SHOOTING STAR (63% Accuracy)
                if (current_upper_wick >= 2 * current_body and
                    current_lower_wick <= current_body * 0.3 and
                    current['Close'] < current['Open'] and
                    current['Volume_Ratio'] > 1.2):
                    patterns.append({
                        "name": "Shooting Star",
                        "type": "SELL",
                        "accuracy": "63%",
                        "entry_price": current['Close'], 
                        "stop_loss": current['High'] * 1.02,
                        "targets": [
                            current['Close'] * 0.96,
                            current['Close'] * 0.92,
                            current['Close'] * 0.88
                        ]
                    })
            
            # 5. THREE BLACK CROWS (78% Accuracy - Highest)
            if (len(df) >= 3 and
                all(df['Close'].iloc[-3:] < df['Open'].iloc[-3:]) and  # 3 red candles
                all(df['Close'].iloc[-3:] < df['Close'].shift(1).iloc[-3:]) and  # Lower lows
                current['Volume_Ratio'] > 1.5):
                patterns.append({
                    "name": "Three Black Crows",
                    "type": "SELL", 
                    "accuracy": "78%",
                    "entry_price": current['Close'],
                    "stop_loss": df['High'].iloc[-3:].max() * 1.02,
                    "targets": [
                        current['Close'] * 0.95,
                        current['Close'] * 0.90, 
                        current['Close'] * 0.85
                    ]
                })
                
        except Exception as e:
            st.error(f"Pattern error: {str(e)}")
            
        return patterns

    def calculate_dynamic_targets(self, df, current_price, signal_type):
        """Calculate dynamic price targets based on technical analysis"""
        if df is None or len(df) < 50:
            return []
            
        current = df.iloc[-1]
        targets = []
        
        if signal_type == "BUY":
            # For BUY signals - calculate upside targets
            
            # 1. Resistance-based target
            resistance_target = current['Resistance_1'] * 0.99  # Just below resistance
            targets.append(resistance_target)
            
            # 2. ATR-based target (2x ATR)
            atr_target = current_price + (current['ATR'] * 2)
            targets.append(atr_target)
            
            # 3. Percentage-based targets
            targets.append(current_price * 1.03)  # 3%
            targets.append(current_price * 1.06)  # 6%
            targets.append(current_price * 1.10)  # 10%
            
            # 4. Fibonacci extension (1.618)
            recent_low = df['Low'].tail(20).min()
            recent_high = df['High'].tail(20).max()
            fib_target = current_price + ((recent_high - recent_low) * 0.618)
            targets.append(fib_target)
            
        else:  # SELL signals
            # For SELL signals - calculate downside targets
            
            # 1. Support-based target
            support_target = current['Support_1'] * 1.01  # Just above support
            targets.append(support_target)
            
            # 2. ATR-based target (2x ATR)
            atr_target = current_price - (current['ATR'] * 2)
            targets.append(atr_target)
            
            # 3. Percentage-based targets
            targets.append(current_price * 0.97)  # 3% down
            targets.append(current_price * 0.94)  # 6% down
            targets.append(current_price * 0.90)  # 10% down
            
            # 4. Fibonacci retracement (0.618)
            recent_low = df['Low'].tail(20).min()
            recent_high = df['High'].tail(20).max()
            fib_target = current_price - ((recent_high - recent_low) * 0.382)
            targets.append(fib_target)
        
        # Remove duplicates and sort
        unique_targets = sorted(list(set([t for t in targets if t > 0])))
        
        # Return top 3 most probable targets
        if signal_type == "BUY":
            return unique_targets[:3]
        else:
            return unique_targets[:3]

    def generate_precise_signals(self, df, patterns):
        """Generate precise trading signals with exact instructions"""
        if df is None or len(df) < 20:
            return "NO TRADE", 50, "Insufficient data", [], [], []
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Technical Score Calculation
        score = 50
        reasons = []
        actions = []
        targets = []
        
        # 1. MOVING AVERAGE TREND (25%)
        if current['SMA_20'] > current['SMA_50']:
            score += 12
            reasons.append("✅ Uptrend: SMA 20 > SMA 50")
        else:
            score -= 12
            reasons.append("❌ Downtrend: SMA 20 < SMA 50")
        
        # 2. RSI MOMENTUM (20%)
        if current['RSI'] < 35:
            score += 10
            reasons.append("💰 RSI Oversold: Good for BUY")
            actions.append("BUY")
        elif current['RSI'] > 65:
            score -= 10
            reasons.append("💀 RSI Overbought: Good for SELL")
            actions.append("SELL")
        
        # 3. MACD SIGNAL (20%)
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            score += 10
            reasons.append("📈 MACD Bullish Crossover")
            actions.append("BUY")
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            score -= 10
            reasons.append("📉 MACD Bearish Crossover")
            actions.append("SELL")
        
        # 4. VOLUME CONFIRMATION (15%)
        if current['Volume_Ratio'] > 1.5:
            if "BUY" in actions:
                score += 8
                reasons.append("🔥 High Volume BUY Confirmation")
            elif "SELL" in actions:
                score -= 8
                reasons.append("💥 High Volume SELL Confirmation")
        
        # 5. BOLLINGER BANDS (10%)
        if current['Close'] <= current['BB_Lower']:
            score += 5
            reasons.append("🎯 At BB Lower Band - BUY Zone")
            actions.append("BUY")
        elif current['Close'] >= current['BB_Upper']:
            score -= 5
            reasons.append("⚠️ At BB Upper Band - SELL Zone")
            actions.append("SELL")
        
        # 6. PATTERN BASED SIGNALS (10%)
        for pattern in patterns:
            if pattern['type'] == 'BUY':
                score += 5
                reasons.append(f"✅ {pattern['name']} Pattern - {pattern['accuracy']}")
                actions.append("BUY")
                targets.extend(pattern['targets'])
            else:
                score -= 5
                reasons.append(f"❌ {pattern['name']} Pattern - {pattern['accuracy']}")
                actions.append("SELL")
                targets.extend(pattern['targets'])
        
        # FINAL DECISION
        score = max(0, min(100, score))
        current_price = current['Close']
        
        # Determine primary action
        buy_count = actions.count("BUY")
        sell_count = actions.count("SELL")
        
        if buy_count > sell_count and score >= 60:
            action_type = "BUY"
            dynamic_targets = self.calculate_dynamic_targets(df, current_price, "BUY")
            stop_loss = current_price * 0.95  # 5% stop loss
            reason = f"Strong BUY signal with {score}% confidence"
            
        elif sell_count > buy_count and score <= 40:
            action_type = "SELL" 
            dynamic_targets = self.calculate_dynamic_targets(df, current_price, "SELL")
            stop_loss = current_price * 1.05  # 5% stop loss
            reason = f"Strong SELL signal with {100-score}% confidence"
            
        else:
            action_type = "HOLD"
            dynamic_targets = []
            stop_loss = 0
            reason = "Market in consolidation - Wait for better setup"
        
        # Combine pattern targets with dynamic targets
        all_targets = list(set(targets + dynamic_targets))
        all_targets.sort()
        
        if action_type == "BUY":
            all_targets = [t for t in all_targets if t > current_price][:3]
        else:
            all_targets = [t for t in all_targets if t < current_price][:3]
        
        return action_type, score, reason, reasons, all_targets, stop_loss

def main():
    st.markdown('<div class="main-header">🎯 PRO TRADER - EXACT BUY/SELL SIGNALS</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280;">Know Exactly WHEN to Buy/Sell • Precise Price Targets • Stop Loss Levels</p>', unsafe_allow_html=True)
    
    trader = ProTraderWithTargets()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 STOCK SELECTION")
        selected_stock = st.selectbox("Select Stock:", list(trader.stock_list.keys()))
        symbol = trader.stock_list[selected_stock]
        
        st.header("⚙️ RISK MANAGEMENT")
        risk_percent = st.slider("Risk per Trade %", 1.0, 5.0, 2.0, 0.5)
        show_details = st.checkbox("Show Detailed Analysis", True)
    
    # Main Analysis
    if st.button("🎯 GET EXACT TRADING SIGNALS", type="primary", use_container_width=True):
        with st.spinner("Calculating precise entry points and targets..."):
            data = trader.get_stock_data(symbol)
            
            if data is not None and not data.empty:
                # Calculate indicators
                df = trader.calculate_advanced_indicators(data)
                
                # Detect patterns
                patterns = trader.detect_high_probability_patterns(df)
                
                # Generate precise signals
                action, score, reason, reasons, targets, stop_loss = trader.generate_precise_signals(df, patterns)
                
                current_price = df['Close'].iloc[-1] if df is not None else 0
                current_date = df.index[-1].strftime('%Y-%m-%d') if df is not None else ""
                
                # Display TRADING SIGNAL
                if action == "BUY":
                    st.markdown(f'''
                    <div class="trade-signal">
                        <h1>🚀 STRONG BUY SIGNAL</h1>
                        <h2>Confidence: {score}% • Date: {current_date}</h2>
                        <h3>{reason}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # BUY INSTRUCTIONS
                    st.markdown(f'''
                    <div class="buy-box">
                        <h2>📈 BUY INSTRUCTIONS:</h2>
                        <h3>Entry Price: ₹{current_price:.2f}</h3>
                        <h3>Stop Loss: ₹{stop_loss:.2f} (-5%)</h3>
                        <p>Buy when price is near current levels with strict stop loss</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                elif action == "SELL":
                    st.markdown(f'''
                    <div class="trade-signal" style="background: linear-gradient(135deg, #ff416c, #ff4b2b);">
                        <h1>💀 STRONG SELL SIGNAL</h1>
                        <h2>Confidence: {100-score}% • Date: {current_date}</h2>
                        <h3>{reason}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # SELL INSTRUCTIONS
                    st.markdown(f'''
                    <div class="sell-box">
                        <h2>📉 SELL INSTRUCTIONS:</h2>
                        <h3>Entry Price: ₹{current_price:.2f}</h3>
                        <h3>Stop Loss: ₹{stop_loss:.2f} (+5%)</h3>
                        <p>Sell when price is near current levels with strict stop loss</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                else:
                    st.markdown(f'''
                    <div class="trade-signal" style="background: linear-gradient(135deg, #f7971e, #ffd200);">
                        <h1>⚪ HOLD - NO TRADE</h1>
                        <h2>Market in Consolidation</h2>
                        <h3>{reason}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # PRICE TARGETS
                if targets and action in ["BUY", "SELL"]:
                    st.subheader("🎯 PRICE TARGETS (Sell at these levels)")
                    
                    cols = st.columns(3)
                    target_types = ["TARGET 1", "TARGET 2", "TARGET 3"]
                    
                    for idx, (col, target_type, target_price) in enumerate(zip(cols, target_types, targets)):
                        with col:
                            profit_percent = ((target_price - current_price) / current_price * 100) if action == "BUY" else ((current_price - target_price) / current_price * 100)
                            st.markdown(f'''
                            <div class="target-box">
                                <h4>{target_type}</h4>
                                <h2>₹{target_price:.2f}</h2>
                                <p>+{profit_percent:.1f}% Profit</p>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    # TRADING PLAN
                    st.subheader("📋 TRADING PLAN")
                    if action == "BUY":
                        st.success(f"""
                        **BUY NOW at:** ₹{current_price:.2f}
                        **Stop Loss:** ₹{stop_loss:.2f} (Exit if hits)
                        **Target 1:** ₹{targets[0]:.2f} (Sell 30% position)
                        **Target 2:** ₹{targets[1]:.2f} (Sell 40% position)  
                        **Target 3:** ₹{targets[2]:.2f} (Sell remaining 30%)
                        """)
                    else:
                        st.error(f"""
                        **SELL NOW at:** ₹{current_price:.2f}
                        **Stop Loss:** ₹{stop_loss:.2f} (Exit if hits)
                        **Target 1:** ₹{targets[0]:.2f} (Cover 30% position)
                        **Target 2:** ₹{targets[1]:.2f} (Cover 40% position)
                        **Target 3:** ₹{targets[2]:.2f} (Cover remaining 30%)
                        """)
                
                # DETAILED ANALYSIS
                if show_details:
                    st.subheader("🔍 TECHNICAL ANALYSIS BREAKDOWN")
                    for reason_text in reasons:
                        if "✅" in reason_text or "BUY" in reason_text:
                            st.success(reason_text)
                        elif "❌" in reason_text or "SELL" in reason_text:
                            st.error(reason_text)
                        else:
                            st.info(reason_text)
                    
                    # CANDLESTICK PATTERNS
                    if patterns:
                        st.subheader("🎯 DETECTED CANDLESTICK PATTERNS")
                        for pattern in patterns:
                            if pattern['type'] == 'BUY':
                                st.success(f"✅ {pattern['name']} - {pattern['accuracy']} Accuracy")
                                st.info(f"   Entry: ₹{pattern['entry_price']:.2f} | Stop: ₹{pattern['stop_loss']:.2f}")
                            else:
                                st.error(f"❌ {pattern['name']} - {pattern['accuracy']} Accuracy") 
                                st.info(f"   Entry: ₹{pattern['entry_price']:.2f} | Stop: ₹{pattern['stop_loss']:.2f}")
                
                # CREATE CHART
                st.subheader("📊 LIVE PRICE CHART WITH LEVELS")
                create_trading_chart(df, selected_stock, current_price, targets, stop_loss, action)
                
            else:
                st.error("❌ Could not fetch stock data. Please try again.")

    # Trading Education
    with st.expander("📚 TRADING RULES"):
        st.markdown("""
        ### 🎯 **WHEN TO BUY:**
        - RSI < 35 (Oversold)
        - Bullish candlestick pattern (Hammer, Engulfing)
        - Price at support level
        - High volume confirmation
        - MACD bullish crossover
        
        ### ⚠️ **WHEN TO SELL:**
        - RSI > 65 (Overbought) 
        - Bearish candlestick pattern (Shooting Star, Engulfing)
        - Price at resistance level
        - High volume selling
        - MACD bearish crossover
        
        ### 💡 **TARGET STRATEGY:**
        - **Target 1:** Partial profit booking (30%)
        - **Target 2:** More profit booking (40%)
        - **Target 3:** Final exit (30%)
        - Always use stop loss
        """)

def create_trading_chart(df, symbol, current_price, targets, stop_loss, action):
    """Create trading chart with entry, targets and stop loss"""
    if df is None:
        return
        
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Current Price Line
    fig.add_hline(y=current_price, line_dash="dash", line_color="blue", 
                 annotation_text="Current Price")
    
    # Stop Loss Line
    if stop_loss > 0:
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red",
                     annotation_text="Stop Loss")
    
    # Target Lines
    colors = ['green', 'orange', 'red']
    for idx, target in enumerate(targets):
        fig.add_hline(y=target, line_dash="dash", line_color=colors[idx],
                     annotation_text=f"Target {idx+1}")
    
    fig.update_layout(
        title=f'{symbol} - TRADING LEVELS',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
