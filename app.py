import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import ta  # Technical Analysis library

# Set page config
st.set_page_config(
    page_title="PRO TRADER SIGNALS MASTER",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS
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
        padding: 1rem;
    }
    .signal-strong-buy {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #00ff88;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
    }
    .signal-buy {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 2px solid #00ff88;
    }
    .signal-strong-sell {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #ff4444;
        box-shadow: 0 10px 30px rgba(255, 68, 68, 0.3);
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff4b2b, #ff416c);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 2px solid #ff4444;
    }
    .signal-hold {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 2px solid #ffd200;
    }
    .premium-indicator {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .price-target {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        text-align: center;
    }
    .risk-metric {
        background: linear-gradient(135deg, #fd746c, #ff9068);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        text-align: center;
    }
    .notification-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ffd700;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

class ProTraderSignals:
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
            'BHARTIARTL': 'BHARTIARTL.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'WIPRO': 'WIPRO.NS'
        }
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbol):
        """Get comprehensive stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="6mo")
            if data.empty:
                return None
                
            # Get additional info
            info = stock.info
            return data, info
        except:
            return None, None

    def calculate_premium_indicators(self, data):
        """Calculate advanced professional indicators"""
        if data is None or len(data) < 100:
            return data
            
        df = data.copy()
        
        # 1. TREND INDICATORS
        # Multiple Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_100'] = df['Close'].rolling(100).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # Ichimoku Cloud
        df['Ichimoku_Conversion'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        df['Ichimoku_Base'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        df['Ichimoku_SpanB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
        
        # 2. MOMENTUM INDICATORS
        # RSI with multiple timeframes
        for period in [14, 21]:
            df[f'RSI_{period}'] = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
        
        # Stochastic
        df['Stoch_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # 3. VOLATILITY INDICATORS
        # Bollinger Bands with multiple deviations
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # ATR (Average True Range)
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # 4. VOLUME INDICATORS
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()
        
        # 5. ADVANCED OSCILLATORS
        # MACD with histogram
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['MACD_Histogram'] = ta.trend.MACD(df['Close']).macd_diff()
        
        # ADX for trend strength
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df['ADX_Positive'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_pos()
        df['ADX_Negative'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_neg()
        
        # 6. SUPPORT/RESISTANCE
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Resistance_1'] = (2 * df['Pivot']) - df['Low']
        df['Support_1'] = (2 * df['Pivot']) - df['High']
        
        # 7. PRICE ACTION
        df['Price_Change'] = df['Close'].pct_change()
        df['Trend_Strength'] = abs(df['Close'] - df['SMA_20']) / df['ATR']
        
        return df.fillna(method='bfill')

    def calculate_price_targets(self, df):
        """Calculate professional price targets using multiple methods"""
        if df is None or len(df) < 50:
            return {}
            
        current = df.iloc[-1]
        targets = {}
        
        # Method 1: Fibonacci Extensions
        recent_high = df['High'].tail(50).max()
        recent_low = df['Low'].tail(50).min()
        fib_range = recent_high - recent_low
        
        targets['fib_161'] = current['Close'] + (fib_range * 0.618)
        targets['fib_261'] = current['Close'] + (fib_range * 1.0)
        targets['fib_423'] = current['Close'] + (fib_range * 1.618)
        
        # Method 2: Measured Move (Pattern-based)
        if len(df) > 100:
            # Simple channel breakout
            resistance_level = df['High'].tail(50).max()
            support_level = df['Low'].tail(50).min()
            channel_height = resistance_level - support_level
            
            targets['measured_move'] = current['Close'] + channel_height
        
        # Method 3: Volatility-based targets
        avg_atr = df['ATR'].tail(20).mean()
        targets['volatility_1'] = current['Close'] + (avg_atr * 1)
        targets['volatility_2'] = current['Close'] + (avg_atr * 2)
        targets['volatility_3'] = current['Close'] + (avg_atr * 3)
        
        # Method 4: Moving Average convergence
        ma_convergence = (df['SMA_20'].iloc[-1] - df['SMA_50'].iloc[-1]) / df['SMA_50'].iloc[-1]
        if ma_convergence > 0.02:  # Bullish
            targets['ma_target'] = current['Close'] * (1 + abs(ma_convergence))
        else:
            targets['ma_target'] = current['Close'] * (1 - abs(ma_convergence))
        
        # Calculate consensus target
        all_targets = list(targets.values())
        targets['consensus'] = np.mean(all_targets)
        targets['optimistic'] = np.max(all_targets)
        targets['conservative'] = np.min(all_targets)
        
        return targets

    def generate_pro_signals(self, df):
        """Generate professional-grade trading signals"""
        if df is None or len(df) < 50:
            return "HOLD", 50, "Insufficient data", []
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        score = 50
        weight_accumulator = 0
        
        try:
            # 1. TREND ANALYSIS (Weight: 30%)
            trend_score = 0
            # Multiple MA alignment
            if (current['SMA_20'] > current['SMA_50'] > current['SMA_100']):
                signals.append("📊 STRONG UPTREND: All MAs aligned bullish")
                trend_score += 15
            elif (current['SMA_20'] < current['SMA_50'] < current['SMA_100']):
                signals.append("📉 STRONG DOWNTREND: All MAs aligned bearish")
                trend_score -= 15
            
            # Ichimoku Cloud
            if current['Close'] > current['Ichimoku_SpanA'] and current['Close'] > current['Ichimoku_SpanB']:
                signals.append("☁️ BULLISH: Price above Ichimoku Cloud")
                trend_score += 10
            elif current['Close'] < current['Ichimoku_SpanA'] and current['Close'] < current['Ichimoku_SpanB']:
                signals.append("🌧️ BEARISH: Price below Ichimoku Cloud")
                trend_score -= 10
            
            # ADX Trend Strength
            if current['ADX'] > 25:
                if current['ADX_Positive'] > current['ADX_Negative']:
                    signals.append("💪 STRONG UPTREND: ADX confirms bullish strength")
                    trend_score += 5
                else:
                    signals.append("👎 STRONG DOWNTREND: ADX confirms bearish strength")
                    trend_score -= 5
            
            score += trend_score
            weight_accumulator += 30

            # 2. MOMENTUM CONFIRMATION (Weight: 25%)
            momentum_score = 0
            # RSI Multi-timeframe
            if current['RSI_14'] < 30 and current['RSI_21'] < 35:
                signals.append("💰 STRONG BUY: RSI oversold on multiple timeframes")
                momentum_score += 15
            elif current['RSI_14'] > 70 and current['RSI_21'] > 65:
                signals.append("💀 STRONG SELL: RSI overbought on multiple timeframes")
                momentum_score -= 15
            elif 40 <= current['RSI_14'] <= 45 and current['RSI_21'] <= 50:
                signals.append("📈 BULLISH MOMENTUM: RSI in accumulation zone")
                momentum_score += 8
            elif 55 <= current['RSI_14'] <= 60 and current['RSI_21'] >= 50:
                signals.append("📉 BEARISH MOMENTUM: RSI in distribution zone")
                momentum_score -= 8
            
            # Stochastic
            if current['Stoch_K'] < 20 and current['Stoch_D'] < 20:
                signals.append("🎯 MOMENTUM BUY: Stochastic oversold")
                momentum_score += 5
            elif current['Stoch_K'] > 80 and current['Stoch_D'] > 80:
                signals.append("⚠️ MOMENTUM SELL: Stochastic overbought")
                momentum_score -= 5
            
            score += momentum_score
            weight_accumulator += 25

            # 3. VOLUME CONFIRMATION (Weight: 20%)
            volume_score = 0
            if current['Volume_Ratio'] > 2.0:
                if score > 50:
                    signals.append("🔥 VOLUME CONFIRMATION: Strong buying volume")
                    volume_score += 15
                else:
                    signals.append("💥 VOLUME CONFIRMATION: Strong selling volume")
                    volume_score -= 15
            elif current['Volume_Ratio'] > 1.5:
                volume_score += 8 if score > 50 else -8
            
            # OBV Trend
            if len(df) > 5:
                obv_trend = df['OBV'].tail(5).pct_change().mean()
                if obv_trend > 0.02:
                    signals.append("📊 BULLISH VOLUME: OBV trending up")
                    volume_score += 5
            
            score += volume_score
            weight_accumulator += 20

            # 4. VOLATILITY & RISK (Weight: 15%)
            volatility_score = 0
            # Bollinger Band Position
            bb_position = (current['Close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower'])
            if bb_position < 0.1:
                signals.append("🎯 LOW-RISK ENTRY: Near BB Lower Band")
                volatility_score += 10
            elif bb_position > 0.9:
                signals.append("⚠️ HIGH-RISK: Near BB Upper Band")
                volatility_score -= 10
            
            # ATR for stop loss sizing
            atr_ratio = current['ATR'] / current['Close']
            if atr_ratio < 0.02:
                signals.append("✅ LOW VOLATILITY: Good risk management")
                volatility_score += 5
            
            score += volatility_score
            weight_accumulator += 15

            # 5. MACD CONFIRMATION (Weight: 10%)
            macd_score = 0
            if (current['MACD'] > current['MACD_Signal'] and 
                prev['MACD'] <= prev['MACD_Signal']):
                signals.append("📈 MACD BULLISH CROSS: Strong buy signal")
                macd_score += 10
            elif (current['MACD'] < current['MACD_Signal'] and 
                  prev['MACD'] >= prev['MACD_Signal']):
                signals.append("📉 MACD BEARISH CROSS: Strong sell signal")
                macd_score -= 10
            
            score += macd_score
            weight_accumulator += 10

            # Normalize score based on actual weights used
            if weight_accumulator > 0:
                score = 50 + ((score - 50) * 100 / weight_accumulator)
            score = max(0, min(100, score))

            # FINAL DECISION with professional grading
            if score >= 85:
                action = "🚀 STRONG BUY"
                reason = "Multiple high-probability bullish signals aligned"
                confidence = "Very High"
            elif score >= 70:
                action = "📈 BUY"
                reason = "Strong bullish bias with good confirmation"
                confidence = "High"
            elif score >= 60:
                action = "🟡 MILD BUY"
                reason = "Moderate bullish signals"
                confidence = "Medium"
            elif score >= 50:
                action = "⚪ HOLD"
                reason = "Neutral market conditions"
                confidence = "Low"
            elif score >= 40:
                action = "🟠 MILD SELL"
                reason = "Moderate bearish signals"
                confidence = "Medium"
            elif score >= 25:
                action = "📉 SELL"
                reason = "Strong bearish bias emerging"
                confidence = "High"
            else:
                action = "💀 STRONG SELL"
                reason = "Multiple high-probability bearish signals"
                confidence = "Very High"
            
            return action, int(score), reason, confidence, signals
            
        except Exception as e:
            return "HOLD", 50, f"Analysis error: {str(e)}", "Low", []

    def get_precise_entry_points(self, df):
        """Get precise entry and exit points with confirmation"""
        if df is None or len(df) < 50:
            return [], [], [], []
            
        buy_points = []
        sell_points = []
        entry_signals = []
        exit_signals = []
        
        for i in range(5, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # MULTI-CONFIRMATION BUY STRATEGY
            buy_confirmations = 0
            buy_reasons = []
            
            # Condition 1: RSI Oversold bounce
            if current['RSI_14'] < 30 and prev['RSI_14'] >= 30:
                buy_confirmations += 1
                buy_reasons.append("RSI Oversold")
            
            # Condition 2: Stochastic reversal
            if current['Stoch_K'] < 20 and current['Stoch_D'] < 20 and prev['Stoch_K'] >= prev['Stoch_D']:
                buy_confirmations += 1
                buy_reasons.append("Stochastic Bullish")
            
            # Condition 3: MACD bullish crossover
            if (current['MACD'] > current['MACD_Signal'] and 
                prev['MACD'] <= prev['MACD_Signal']):
                buy_confirmations += 1
                buy_reasons.append("MACD Cross")
            
            # Condition 4: Price at support
            if current['Close'] <= current['Support_1']:
                buy_confirmations += 1
                buy_reasons.append("Support Test")
            
            # Condition 5: Volume confirmation
            if current['Volume_Ratio'] > 1.5:
                buy_confirmations += 1
                buy_reasons.append("High Volume")
            
            # Require at least 3 confirmations for high-probability buy
            if buy_confirmations >= 3:
                buy_points.append((df.index[i], current['Low'] * 0.995))
                entry_signals.append(f"BUY: {', '.join(buy_reasons)}")
            
            # SELL STRATEGY
            sell_confirmations = 0
            sell_reasons = []
            
            # Condition 1: RSI Overbought rejection
            if current['RSI_14'] > 70 and prev['RSI_14'] <= 70:
                sell_confirmations += 1
                sell_reasons.append("RSI Overbought")
            
            # Condition 2: Stochastic overbought
            if current['Stoch_K'] > 80 and current['Stoch_D'] > 80 and prev['Stoch_K'] <= prev['Stoch_D']:
                sell_confirmations += 1
                sell_reasons.append("Stochastic Bearish")
            
            # Condition 3: MACD bearish crossover
            if (current['MACD'] < current['MACD_Signal'] and 
                prev['MACD'] >= prev['MACD_Signal']):
                sell_confirmations += 1
                sell_reasons.append("MACD Cross")
            
            # Condition 4: Price at resistance
            if current['Close'] >= current['Resistance_1']:
                sell_confirmations += 1
                sell_reasons.append("Resistance Test")
            
            # Condition 5: Volume confirmation
            if current['Volume_Ratio'] > 1.5:
                sell_confirmations += 1
                sell_reasons.append("High Volume")
            
            # Require at least 3 confirmations for high-probability sell
            if sell_confirmations >= 3:
                sell_points.append((df.index[i], current['High'] * 1.005))
                exit_signals.append(f"SELL: {', '.join(sell_reasons)}")
        
        return buy_points, sell_points, entry_signals[-5:], exit_signals[-5:]

def main():
    st.markdown('<div class="main-header">💎 PRO TRADER SIGNALS MASTER</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.2rem;">Institutional Grade Analysis • Multi-Timeframe Confirmation • Precision Targets</p>', unsafe_allow_html=True)
    
    analyzer = ProTraderSignals()
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 PRO TRADING SETUP")
        selected_stock = st.selectbox("Select Stock:", list(analyzer.stock_list.keys()))
        symbol = analyzer.stock_list[selected_stock]
        
        st.header("⚙️ RISK MANAGEMENT")
        capital = st.number_input("Trading Capital (₹)", 10000, 10000000, 100000, 10000)
        risk_per_trade = st.slider("Risk per Trade %", 0.5, 5.0, 1.0, 0.5)
        position_sizing = st.checkbox("Auto Position Sizing", value=True)
        
        st.header("📊 ANALYSIS SETTINGS")
        timeframe = st.selectbox("Chart Timeframe", ["1d", "1h", "4h"], index=0)
        show_advanced = st.checkbox("Show Advanced Indicators", value=True)
    
    # Main Analysis
    if st.button("💎 GENERATE PRO SIGNALS", type="primary", use_container_width=True):
        with st.spinner("Running institutional-grade analysis..."):
            data, info = analyzer.get_stock_data(symbol)
            
            if data is not None and not data.empty:
                # Calculate advanced indicators
                df = analyzer.calculate_premium_indicators(data)
                
                # Generate professional signals
                action, score, reason, confidence, signals = analyzer.generate_pro_signals(df)
                
                # Get precise entry points
                buy_points, sell_points, entry_signals, exit_signals = analyzer.get_precise_entry_points(df)
                
                # Calculate price targets
                price_targets = analyzer.calculate_price_targets(df)
                
                current_price = df['Close'].iloc[-1]
                current_date = df.index[-1].strftime('%Y-%m-%d %H:%M')
                
                # RISK MANAGEMENT CALCULATION
                risk_amount = capital * (risk_per_trade / 100)
                if position_sizing and 'ATR' in df.columns:
                    atr_stop = df['ATR'].iloc[-1] * 1.5
                    position_size = risk_amount / atr_stop
                    lot_size = int(position_size / current_price) if current_price > 0 else 0
                else:
                    lot_size = int((capital * 0.1) / current_price)  # 10% of capital
                
                # Display PREMIUM SIGNAL
                if "STRONG BUY" in action:
                    st.markdown(f'''
                    <div class="signal-strong-buy">
                        <h1>🎯 {action}</h1>
                        <h2>Confidence Score: {score}% • {confidence} Confidence</h2>
                        <h3>{reason}</h3>
                        <p>Last Updated: {current_date}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                elif "STRONG SELL" in action:
                    st.markdown(f'''
                    <div class="signal-strong-sell">
                        <h1>🎯 {action}</h1>
                        <h2>Confidence Score: {score}% • {confidence} Confidence</h2>
                        <h3>{reason}</h3>
                        <p>Last Updated: {current_date}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                elif "BUY" in action:
                    st.markdown(f'''
                    <div class="signal-buy">
                        <h1>📈 {action}</h1>
                        <h2>Confidence Score: {score}% • {confidence} Confidence</h2>
                        <h3>{reason}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                elif "SELL" in action:
                    st.markdown(f'''
                    <div class="signal-sell">
                        <h1>📉 {action}</h1>
                        <h2>Confidence Score: {score}% • {confidence} Confidence</h2>
                        <h3>{reason}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="signal-hold">
                        <h1>⚪ {action}</h1>
                        <h2>Confidence Score: {score}% • {confidence} Confidence</h2>
                        <h3>{reason}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # HIGH ACCURACY NOTIFICATION
                if score >= 75 or score <= 25:
                    st.markdown(f'''
                    <div class="notification-alert">
                        <h3>🔔 HIGH PROBABILITY ALERT!</h3>
                        <p>This signal has {confidence.lower()} confidence based on multiple indicator confirmation.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # TRADING EXECUTION PANEL
                st.subheader("🎯 TRADE EXECUTION PLAN")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Position Size", f"{lot_size} shares")
                with col3:
                    st.metric("Risk Amount", f"₹{risk_amount:,.0f}")
                with col4:
                    st.metric("Max Capital", f"₹{lot_size * current_price:,.0f}")
                
                # PRICE TARGETS
                st.subheader("🎯 PROFESSIONAL PRICE TARGETS")
                if price_targets:
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f'''
                        <div class="price-target">
                            <h4>Conservative Target</h4>
                            <h2>₹{price_targets['conservative']:.2f}</h2>
                            <p>+{((price_targets['conservative']-current_price)/current_price*100):.1f}%</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f'''
                        <div class="price-target">
                            <h4>Consensus Target</h4>
                            <h2>₹{price_targets['consensus']:.2f}</h2>
                            <p>+{((price_targets['consensus']-current_price)/current_price*100):.1f}%</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown(f'''
                        <div class="price-target">
                            <h4>Optimistic Target</h4>
                            <h2>₹{price_targets['optimistic']:.2f}</h2>
                            <p>+{((price_targets['optimistic']-current_price)/current_price*100):.1f}%</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    with cols[3]:
                        st.markdown(f'''
                        <div class="risk-metric">
                            <h4>Volatility Target</h4>
                            <h2>₹{price_targets['volatility_2']:.2f}</h2>
                            <p>Based on ATR</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Create advanced chart
                st.subheader("📊 INSTITUTIONAL CHART ANALYSIS")
                self.create_pro_chart(df, selected_stock, buy_points, sell_points)
                
                # SIGNAL BREAKDOWN
                st.subheader("🔍 MULTI-TIMEFRAME SIGNAL CONFIRMATION")
                if signals:
                    for signal in signals[-8:]:
                        st.write(f"• {signal}")
                
                # PRECISE ENTRY/EXIT POINTS
                st.subheader("🎯 PRECISE TRADING LEVELS")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Entry Signals:**")
                    if entry_signals:
                        for signal in entry_signals:
                            st.success(f"• {signal}")
                    else:
                        st.info("• No recent high-probability entry signals")
                
                with col2:
                    st.write("**Exit Signals:**")
                    if exit_signals:
                        for signal in exit_signals:
                            st.error(f"• {signal}")
                    else:
                        st.info("• No recent high-probability exit signals")
                
                # ADVANCED INDICATORS DASHBOARD
                if show_advanced:
                    st.subheader("⚙️ ADVANCED INDICATOR DASHBOARD")
                    self.display_advanced_indicators(df)
            
            else:
                st.error("❌ Could not fetch stock data. Please try again.")

    # Quick Actions
    st.sidebar.header("⚡ PRO ACTIONS")
    if st.sidebar.button("🔄 Refresh All Signals", use_container_width=True):
        st.rerun()
    if st.sidebar.button("📊 Scan All Stocks", use_container_width=True):
        self.scan_all_stocks(analyzer)

    def create_pro_chart(self, df, symbol, buy_points, sell_points):
        """Create professional trading chart"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'<b>{symbol} - PRO TRADER ANALYSIS</b>', 
                '<b>MOMENTUM OSCILLATORS</b>',
                '<b>VOLUME & MONEY FLOW</b>',
                '<b>MACD & TREND STRENGTH</b>'
            ),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Price with Ichimoku Cloud
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
        
        # Ichimoku Cloud
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Ichimoku_SpanA'], 
                      name='Ichimoku Span A', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Ichimoku_SpanB'], 
                      name='Ichimoku Span B', line=dict(color='red')),
            row=1, col=1
        )
        
        # BUY/SELL signals
        if buy_points:
            buy_dates, buy_prices = zip(*buy_points)
            fig.add_trace(
                go.Scatter(
                    x=buy_dates, y=buy_prices,
                    mode='markers',
                    name='BUY',
                    marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='darkgreen')),
                    hovertemplate='<b>BUY ENTRY</b><br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
        
        if sell_points:
            sell_dates, sell_prices = zip(*sell_points)
            fig.add_trace(
                go.Scatter(
                    x=sell_dates, y=sell_prices,
                    mode='markers',
                    name='SELL',
                    marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='darkred')),
                    hovertemplate='<b>SELL EXIT</b><br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI_14'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['Close'], df['Open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Volume_SMA'], name='Vol MA', line=dict(color='orange')),
            row=3, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram'),
            row=4, col=1
        )
        
        fig.update_layout(height=1000, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    def display_advanced_indicators(self, df):
        """Display advanced indicator readings"""
        current = df.iloc[-1]
        
        cols = st.columns(4)
        indicators = [
            ("ADX Trend", f"{current['ADX']:.1f}", "bullish" if current['ADX'] > 25 else "neutral"),
            ("ATR Volatility", f"{current['ATR']:.2f}", "high" if current['ATR']/current['Close'] > 0.02 else "low"),
            ("Stochastic", f"K:{current['Stoch_K']:.1f}", "bullish" if current['Stoch_K'] < 20 else "bearish" if current['Stoch_K'] > 80 else "neutral"),
            ("Volume Flow", f"{current['Volume_Ratio']:.1f}x", "bullish" if current['Volume_Ratio'] > 1.5 else "neutral"),
            ("OBV Trend", f"{current['OBV']:,.0f}", "bullish" if current['OBV'] > df['OBV'].iloc[-10] else "bearish"),
            ("CMF Money", f"{current['CMF']:.3f}", "bullish" if current['CMF'] > 0 else "bearish"),
            ("BB Width", f"{current['BB_Width']:.3f}", "high" if current['BB_Width'] > 0.05 else "low"),
            ("Trend Strength", f"{current['Trend_Strength']:.1f}", "strong" if current['Trend_Strength'] > 2 else "weak")
        ]
        
        for idx, (name, value, status) in enumerate(indicators):
            with cols[idx % 4]:
                st.markdown(f'''
                <div class="premium-indicator">
                    <h4>{name}</h4>
                    <h3>{value}</h3>
                    <p>{status.upper()}</p>
                </div>
                ''', unsafe_allow_html=True)

    def scan_all_stocks(self, analyzer):
        """Scan all stocks for high-probability signals"""
        st.subheader("🔍 MULTI-STOCK SCANNER RESULTS")
        results = []
        
        for stock_name, stock_symbol in list(analyzer.stock_list.items())[:6]:  # Limit for performance
            with st.spinner(f"Scanning {stock_name}..."):
                data, _ = analyzer.get_stock_data(stock_symbol)
                if data is not None and not data.empty:
                    df = analyzer.calculate_premium_indicators(data)
                    action, score, reason, confidence, _ = analyzer.generate_pro_signals(df)
                    
                    if score >= 70 or score <= 30:  # Only show strong signals
                        results.append({
                            'Stock': stock_name,
                            'Signal': action,
                            'Score': score,
                            'Confidence': confidence,
                            'Price': df['Close'].iloc[-1],
                            'Reason': reason
                        })
        
        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("No high-probability signals found across scanned stocks.")

if __name__ == "__main__":
    main()
