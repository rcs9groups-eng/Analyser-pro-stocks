import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config FIRST - must be first Streamlit command
st.set_page_config(
    page_title="ULTRA STOCK ANALYZER PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .private-badge {
        background: linear-gradient(45deg, #ef4444, #dc2626);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 25px;
        font-size: 0.9rem;
        margin-left: 1rem;
        font-weight: bold;
    }
    .super-card {
        background: white;
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
        margin: 2rem 0;
        border-left: 8px solid;
        border-right: 3px solid #e5e7eb;
        border-top: 3px solid #e5e7eb;
        border-bottom: 3px solid #e5e7eb;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    .super-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(45deg, #2563eb, #7c3aed);
    }
    .super-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 60px rgba(0,0,0,0.25);
    }
    .ultra-buy {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 50%, #6ee7b7 100%);
    }
    .strong-buy {
        border-left-color: #22c55e;
        background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%);
    }
    .buy {
        border-left-color: #4ade80;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    }
    .strong-sell {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 50%, #f87171 100%);
    }
    .sell {
        border-left-color: #f87171;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    .hold {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    .indicator-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 3px solid #e5e7eb;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .indicator-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .indicator-box:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    .bullish { 
        border-color: #10b981; 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }
    .bullish::before { background: #10b981; }
    .bearish { 
        border-color: #ef4444; 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    .bearish::before { background: #ef4444; }
    .neutral { 
        border-color: #f59e0b; 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    .neutral::before { background: #f59e0b; }
    .trade-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border: none;
    }
    .calculator-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        border: none;
    }
    .scan-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    .profit { color: #10b981; font-weight: bold; }
    .loss { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class UltraStockAnalyzerPro:
    def __init__(self):
        # Reduced stock list for better performance
        self.all_symbols = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'INFOSYS': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'LT': 'LT.NS',
            'ITC': 'ITC.NS',
            'HUL': 'HINDUNILVR.NS'
        }
    
    @st.cache_data(ttl=1800, show_spinner=False)  # Reduced TTL for faster updates
    def get_stock_data(_self, symbol, period="6mo"):  # Reduced period for faster loading
        """Get comprehensive stock data with enhanced error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"⚠️ Data fetch error for {symbol}: {str(e)}")
            return None

    def calculate_advanced_indicators(self, data):
        """Calculate technical indicators with enhanced safety"""
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        try:
            # 1. BASIC PRICE INDICATORS - Only essential ones
            # Moving averages
            for period in [5, 8, 13, 21, 50]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
            # 2. MOMENTUM INDICATORS
            # RSI calculation
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            df['RSI_14'] = calculate_rsi(df['Close'], 14)
            df['RSI_9'] = calculate_rsi(df['Close'], 9)
            
            # MACD
            exp12 = df['Close'].ewm(span=12, adjust=False).mean()
            exp26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # 3. VOLUME INDICATORS
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            
            # 4. VOLATILITY INDICATORS
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
            
            # 5. SUPPORT/RESISTANCE
            df['Support_1'] = df['Low'].rolling(window=20, min_periods=1).min()
            df['Resistance_1'] = df['High'].rolling(window=20, min_periods=1).max()
            
            # Fill NaN values safely
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            st.error(f"Indicator calculation error: {str(e)}")
            # Return basic data if indicator calculation fails
            return data

    def calculate_position_size(self, capital, risk_per_trade, entry_price, stop_loss):
        """Advanced position sizing calculator with risk management"""
        try:
            risk_amount = capital * (risk_per_trade / 100)
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share > 0:
                shares = risk_amount / risk_per_share
                position_value = shares * entry_price
                return int(shares), position_value, risk_amount
            return 0, 0, risk_amount
        except:
            return 0, 0, 0

    def calculate_ai_score(self, df):
        """Enhanced AI-powered scoring with safety checks"""
        if df is None or len(df) < 20:
            return 50, ["Insufficient data for analysis"], {}, 0, 0
            
        try:
            current_price = df['Close'].iloc[-1]
            score = 50
            reasons = []
            signals = {}
            
            # 1. TREND ANALYSIS (40 points)
            trend_score = 0
            bullish_ma_count = 0
            
            # Moving averages analysis
            ma_periods = [5, 8, 13, 21, 50]
            for period in ma_periods:
                ema_col = f'EMA_{period}'
                if ema_col in df and not pd.isna(df[ema_col].iloc[-1]):
                    if current_price > df[ema_col].iloc[-1]:
                        trend_score += 6
                        bullish_ma_count += 1
            
            if bullish_ma_count >= 4:
                trend_score += 15
                reasons.append(f"🚀 STRONG TREND ({bullish_ma_count}/5 EMAs bullish)")
            elif bullish_ma_count >= 3:
                trend_score += 10
                reasons.append(f"📈 BULLISH TREND ({bullish_ma_count}/5 EMAs bullish)")
            
            score += min(trend_score, 40)
            
            # 2. MOMENTUM ANALYSIS (35 points)
            momentum_score = 0
            
            # RSI analysis
            rsi_bullish = 0
            for period in [9, 14]:
                rsi_col = f'RSI_{period}'
                if rsi_col in df and not pd.isna(df[rsi_col].iloc[-1]):
                    rsi_val = df[rsi_col].iloc[-1]
                    if 40 <= rsi_val <= 65:
                        momentum_score += 8
                        rsi_bullish += 1
                        reasons.append(f"🎯 Good RSI {period}: {rsi_val:.1f}")
                    elif rsi_val < 35:
                        momentum_score += 12
                        reasons.append(f"📈 Oversold RSI {period}: {rsi_val:.1f}")
            
            # MACD analysis
            if all(col in df for col in ['MACD', 'MACD_Signal']):
                if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    momentum_score += 15
                    reasons.append("💪 MACD BULLISH - Above signal line")
            
            score += min(momentum_score, 35)
            
            # 3. VOLUME CONFIRMATION (15 points)
            volume_score = 0
            
            if all(col in df for col in ['Volume', 'Volume_SMA_20']):
                volume_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
                if volume_ratio > 2.0:
                    volume_score += 10
                    reasons.append("💰 HIGH VOLUME - Strong interest")
                elif volume_ratio > 1.5:
                    volume_score += 6
                    reasons.append("💰 Good Volume - Participation")
            
            score += min(volume_score, 15)
            
            # 4. VOLATILITY & RISK (10 points)
            risk_score = 0
            
            if 'BB_Position' in df and not pd.isna(df['BB_Position'].iloc[-1]):
                bb_pos = df['BB_Position'].iloc[-1]
                if bb_pos < 0.3:
                    risk_score += 8
                    reasons.append("🎯 Near BB lower band - Good entry")
                elif 0.3 <= bb_pos <= 0.7:
                    risk_score += 4
            
            score += min(risk_score, 10)
            
            # FINAL CONFIDENCE BOOST
            total_reasons = len(reasons)
            if total_reasons >= 6:
                score = min(score + 15, 100)
                reasons.append("🚀 STRONG BULLISH CONVERGENCE")
            elif total_reasons >= 4:
                score = min(score + 8, 100)
                reasons.append("📈 BULLISH BIAS")
            
            bullish_signals = total_reasons
            total_signals = 10  # Estimated total possible signals
            
            return min(score, 100), reasons, signals, bullish_signals, total_signals
            
        except Exception as e:
            st.error(f"AI scoring error: {str(e)}")
            return 50, ["Analysis in progress..."], {}, 0, 0

    def get_trading_signal(self, score):
        """Get detailed trading signal with enhanced categorization"""
        if score >= 90:
            return "🚀 ULTRA STRONG BUY", "ultra-buy", "#10b981", "MAXIMUM CONFIDENCE - IMMEDIATE BUY"
        elif score >= 80:
            return "💎 VERY STRONG BUY", "ultra-buy", "#22c55e", "VERY STRONG BUY - HIGH CONFIDENCE"
        elif score >= 75:
            return "📈 POWERFUL BUY", "strong-buy", "#4ade80", "POWERFUL BUY - STRONG OPPORTUNITY"
        elif score >= 70:
            return "⚡ STRONG BUY", "strong-buy", "#84cc16", "STRONG BUY - GOOD ENTRY"
        elif score >= 65:
            return "💰 ACCUMULATE", "buy", "#a3e635", "ACCUMULATE ON DIPS"
        elif score >= 60:
            return "🔄 HOLD", "hold", "#f59e0b", "HOLD - WAIT FOR CONFIRMATION"
        elif score >= 55:
            return "🔔 CAUTION", "hold", "#f97316", "CAUTION - MONITOR CLOSELY"
        elif score >= 50:
            return "📉 REDUCE", "sell", "#ef4444", "REDUCE POSITION"
        else:
            return "💀 STRONG SELL", "strong-sell", "#dc2626", "STRONG SELL - AVOID"

    def create_advanced_chart(self, df, symbol):
        """Create professional multi-panel chart with safety checks"""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=(
                    f'<b>{symbol} - PRICE ANALYSIS</b>', 
                    '<b>RSI MOMENTUM</b>',
                    '<b>MACD TREND</b>'
                ),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Price Subplot
            fig.add_trace(go.Candlestick(
                x=df.index, 
                open=df['Open'], 
                high=df['High'], 
                low=df['Low'], 
                close=df['Close'],
                name='Price', 
                increasing_line_color='#00C805', 
                decreasing_line_color='#FF0000'
            ), row=1, col=1)
            
            # Key Moving Averages
            for period, color in [(8, '#FF6B35'), (21, '#00B4D8'), (50, '#7209B7')]:
                ema_col = f'EMA_{period}'
                if ema_col in df:
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df[ema_col], 
                        name=f'EMA {period}',
                        line=dict(color=color, width=2),
                        opacity=0.8
                    ), row=1, col=1)
            
            # Bollinger Bands
            if 'BB_Upper' in df and 'BB_Lower' in df:
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['BB_Upper'], 
                    name='BB Upper',
                    line=dict(color='#6B7280', dash='dash', width=1),
                    opacity=0.6
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['BB_Lower'], 
                    name='BB Lower',
                    line=dict(color='#6B7280', dash='dash', width=1),
                    opacity=0.6,
                    fill='tonexty'
                ), row=1, col=1)
            
            # RSI Subplot
            for period, color in [(9, '#8B5CF6'), (14, '#3B82F6')]:
                rsi_col = f'RSI_{period}'
                if rsi_col in df:
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df[rsi_col], 
                        name=f'RSI {period}',
                        line=dict(color=color, width=2)
                    ), row=2, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
            
            # MACD Subplot
            if all(col in df for col in ['MACD', 'MACD_Signal']):
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['MACD'], 
                    name='MACD',
                    line=dict(color='#2563EB', width=2)
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['MACD_Signal'], 
                    name='Signal',
                    line=dict(color='#EF4444', width=2)
                ), row=3, col=1)
                
                fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                template='plotly_white',
                xaxis_rangeslider_visible=False,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Chart creation error: {str(e)}")
            # Return simple price chart if advanced chart fails
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
            fig.update_layout(title=f"Basic Price Chart - {symbol}", height=400)
            return fig

def main():
    # Initialize the analyzer
    app = UltraStockAnalyzerPro()
    
    # Header Section with enhanced design
    st.markdown(
        '<h1 class="main-header">🚀 ULTRA STOCK ANALYZER PRO <span class="private-badge">PRIVATE</span></h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.4rem; color: #6b7280; margin-bottom: 2rem;">'
        '🔒 Professional Use Only • Advanced Indicators • AI-Powered Scoring</p>', 
        unsafe_allow_html=True
    )
    
    # Main Analysis Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 ANALYSIS PARAMETERS")
        
        # Stock selection with enhanced UI
        selected_stock = st.selectbox(
            "Select Stock:", 
            list(app.all_symbols.keys()),
            help="Choose from major Indian stocks and indices"
        )
        symbol = app.all_symbols[selected_stock]
        
        # REMOVED CUSTOM SYMBOL INPUT for better security and performance
        
        # TRADING SETTINGS with enhanced sliders
        st.subheader("💰 TRADING SETTINGS")
        col_a, col_b = st.columns(2)
        
        with col_a:
            stop_loss_percent = st.slider(
                "Stop Loss %", 
                1.0, 20.0, 8.0, 0.5,
                help="Risk management: Maximum loss percentage"
            )
            risk_reward = st.slider(
                "Risk/Reward Ratio", 
                1.0, 5.0, 1.5, 0.1,
                help="Minimum profit to risk ratio"
            )
        
        with col_b:
            target_percent = st.slider(
                "Target %", 
                1.0, 50.0, 15.0, 1.0,
                help="Profit target percentage"
            )
            position_size = st.slider(
                "Position Size (%)", 
                1, 100, 10,
                help="Percentage of capital to allocate"
            )
        
        # ADVANCED CALCULATOR
        st.subheader("🧮 POSITION CALCULATOR")
        capital = st.number_input(
            "Total Capital (₹)", 
            min_value=1000, 
            value=100000, 
            step=1000,
            help="Your total trading capital"
        )
        risk_per_trade = st.number_input(
            "Risk per Trade (%)", 
            min_value=0.1, 
            value=2.0, 
            step=0.1,
            help="Maximum risk per trade as percentage of capital"
        )
        
        # Main Analysis Button
        if st.button("🚀 RUN ADVANCED ANALYSIS", type="primary", use_container_width=True):
            with st.spinner("🔄 Running comprehensive analysis..."):
                data = app.get_stock_data(symbol, "6mo")  # Reduced period for faster loading
                
                if data is not None and not data.empty:
                    df = app.calculate_advanced_indicators(data)
                    
                    if df is not None:
                        score, reasons, signals, bullish_count, total_signals = app.calculate_ai_score(df)
                        signal, signal_class, color, advice = app.get_trading_signal(score)
                        current_price = df['Close'].iloc[-1]
                        
                        # Calculate trading parameters
                        stop_loss_price = current_price * (1 - stop_loss_percent/100)
                        target_price = current_price * (1 + target_percent/100)
                        actual_risk_reward = (target_price - current_price) / (current_price - stop_loss_price) if (current_price - stop_loss_price) > 0 else 0
                        
                        # Position calculation
                        shares, position_value, risk_amount = app.calculate_position_size(
                            capital, risk_per_trade, current_price, stop_loss_price
                        )
                        
                        # RESULTS DISPLAY
                        st.markdown(f'<div class="super-card {signal_class}">', unsafe_allow_html=True)
                        
                        col_s1, col_s2 = st.columns([2, 1])
                        with col_s1:
                            st.subheader(f"{signal}")
                            st.write(f"**AI Confidence Score:** `{score}/100`")
                            st.write(f"**Current Price:** `₹{current_price:.2f}`")
                            st.write(f"**Bullish Signals:** `{bullish_count}/{total_signals}`")
                        with col_s2:
                            st.metric("Signal Strength", f"{score}%", delta=f"{score-50}%")
                        
                        st.write(f"**Expert Advice:** {advice}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # TRADING RECOMMENDATION
                        st.markdown(f'''
                        <div class="trade-box">
                            <h3>💎 PROFESSIONAL TRADING RECOMMENDATION</h3>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                                <div>
                                    <p><strong>Entry Price</strong></p>
                                    <p style="font-size: 1.2rem;">₹{current_price:.2f}</p>
                                </div>
                                <div>
                                    <p><strong>Stop Loss</strong></p>
                                    <p style="font-size: 1.2rem;">₹{stop_loss_price:.2f}</p>
                                    <p style="font-size: 0.9rem;">({stop_loss_percent}% risk)</p>
                                </div>
                                <div>
                                    <p><strong>Target Price</strong></p>
                                    <p style="font-size: 1.2rem;">₹{target_price:.2f}</p>
                                    <p style="font-size: 0.9rem;">(+{target_percent}% gain)</p>
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                                <div>
                                    <p><strong>Risk/Reward</strong></p>
                                    <p style="font-size: 1.2rem;">1:{actual_risk_reward:.1f}</p>
                                </div>
                                <div>
                                    <p><strong>Position Size</strong></p>
                                    <p style="font-size: 1.2rem;">{position_size}%</p>
                                </div>
                                <div>
                                    <p><strong>Signal Confidence</strong></p>
                                    <p style="font-size: 1.2rem;">{score}%</p>
                                </div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # POSITION CALCULATION
                        st.subheader("💰 POSITION CALCULATION")
                        col_p1, col_p2, col_p3 = st.columns(3)
                        
                        with col_p1:
                            st.metric("Shares to Buy", f"{shares:,}")
                        with col_p2:
                            st.metric("Investment", f"₹{position_value:,.2f}")
                        with col_p3:
                            st.metric("Risk Amount", f"₹{risk_amount:,.2f}")
                        
                        # INDICATOR GRID
                        st.subheader("📊 TECHNICAL INDICATORS")
                        cols = st.columns(4)
                        
                        indicators = []
                        
                        # RSI Indicators
                        for period in [9, 14]:
                            rsi_col = f'RSI_{period}'
                            if rsi_col in df:
                                rsi_val = df[rsi_col].iloc[-1]
                                if rsi_val < 35:
                                    rsi_status = 'bullish'
                                elif rsi_val > 65:
                                    rsi_status = 'bearish'
                                else:
                                    rsi_status = 'neutral'
                                indicators.append((f'RSI {period}', f"{rsi_val:.1f}", rsi_status))
                        
                        # MACD
                        if all(col in df for col in ['MACD', 'MACD_Signal']):
                            macd_status = 'bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'bearish'
                            indicators.append(('MACD', 'BULL' if macd_status == 'bullish' else 'BEAR', macd_status))
                        
                        # Volume
                        if all(col in df for col in ['Volume', 'Volume_SMA_20']):
                            vol_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
                            vol_status = 'bullish' if vol_ratio > 1.5 else 'neutral'
                            indicators.append(('Volume', f"{vol_ratio:.1f}x", vol_status))
                        
                        # Bollinger Position
                        if 'BB_Position' in df:
                            bb_pos = df['BB_Position'].iloc[-1]
                            bb_status = 'bullish' if bb_pos < 0.3 else 'bearish' if bb_pos > 0.7 else 'neutral'
                            indicators.append(('BB Position', f"{bb_pos:.2f}", bb_status))
                        
                        # Display indicators in grid
                        for idx, (name, value, status) in enumerate(indicators[:8]):
                            with cols[idx % 4]:
                                st.markdown(f'''
                                <div class="indicator-box {status}">
                                    <h4>{name}</h4>
                                    <h3>{value}</h3>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        # ANALYSIS REASONS
                        with st.expander("📋 View Detailed Analysis Report", expanded=True):
                            for i, reason in enumerate(reasons[:8], 1):
                                st.write(f"{i}. {reason}")
                
                else:
                    st.error("❌ Could not fetch stock data. Please try again later.")
    
    with col2:
        if 'df' in locals() and df is not None:
            st.plotly_chart(app.create_advanced_chart(df, selected_stock), use_container_width=True)
            
            # RISK METRICS
            st.subheader("🛡️ RISK METRICS")
            risk_cols = st.columns(4)
            
            risk_metrics = [
                ("Stop Loss", f"₹{stop_loss_price:.1f}"),
                ("Target Price", f"₹{target_price:.1f}"),
                ("Risk/Reward", f"1:{actual_risk_reward:.1f}"),
                ("ATR", f"₹{df['ATR'].iloc[-1]:.2f}" if 'ATR' in df else "N/A")
            ]
            
            for idx, (name, value) in enumerate(risk_metrics):
                with risk_cols[idx]:
                    st.metric(name, value)

    # SIMPLIFIED MARKET SCANNER
    st.sidebar.header("⚡ SMART ACTIONS")
    
    if st.sidebar.button("🔍 SCAN TOP PERFORMERS", use_container_width=True):
        with st.spinner("Scanning for high-potential stocks..."):
            results = []
            # Scan only first 5 stocks for performance
            for stock_name, stock_symbol in list(app.all_symbols.items())[:5]:
                try:
                    data = app.get_stock_data(stock_symbol, "3mo")
                    if data is not None:
                        df = app.calculate_advanced_indicators(data)
                        if df is not None:
                            score, _, _, _, _ = app.calculate_ai_score(df)
                            if score >= 70:
                                current_price = df['Close'].iloc[-1]
                                results.append({
                                    'symbol': stock_name,
                                    'price': current_price,
                                    'score': score
                                })
                except:
                    continue
            
            if results:
                st.sidebar.subheader("💎 TOP PICKS")
                for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:3]:
                    st.sidebar.markdown(f'''
                    <div class="scan-result">
                        <h4>📈 {stock['symbol']}</h4>
                        <p><strong>AI Score:</strong> {stock['score']}/100</p>
                        <p><strong>Price:</strong> ₹{stock['price']:.2f}</p>
                    </div>
                    ''', unsafe_allow_html=True)

    # INFO SECTION
    st.sidebar.header("🔐 SECURITY")
    
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a, #3730a3); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
        <h4>🚀 PRO FEATURES</h4>
        <ul style="margin-left: -1rem;">
            <li>Advanced Technical Indicators</li>
            <li>AI-Powered Scoring</li>
            <li>Professional Charts</li>
            <li>Risk Management</li>
            <li>Real-time Analysis</li>
        </ul>
        <p><em>For professional use only</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config FIRST - must be first Streamlit command
st.set_page_config(
    page_title="ULTRA STOCK ANALYZER PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .private-badge {
        background: linear-gradient(45deg, #ef4444, #dc2626);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 25px;
        font-size: 0.9rem;
        margin-left: 1rem;
        font-weight: bold;
    }
    .super-card {
        background: white;
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
        margin: 2rem 0;
        border-left: 8px solid;
        border-right: 3px solid #e5e7eb;
        border-top: 3px solid #e5e7eb;
        border-bottom: 3px solid #e5e7eb;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    .super-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(45deg, #2563eb, #7c3aed);
    }
    .super-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 60px rgba(0,0,0,0.25);
    }
    .ultra-buy {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 50%, #6ee7b7 100%);
    }
    .strong-buy {
        border-left-color: #22c55e;
        background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%);
    }
    .buy {
        border-left-color: #4ade80;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    }
    .strong-sell {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 50%, #f87171 100%);
    }
    .sell {
        border-left-color: #f87171;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    .hold {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    .indicator-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 3px solid #e5e7eb;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .indicator-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .indicator-box:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    .bullish { 
        border-color: #10b981; 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }
    .bullish::before { background: #10b981; }
    .bearish { 
        border-color: #ef4444; 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    .bearish::before { background: #ef4444; }
    .neutral { 
        border-color: #f59e0b; 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    .neutral::before { background: #f59e0b; }
    .trade-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border: none;
    }
    .calculator-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        border: none;
    }
    .scan-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    .profit { color: #10b981; font-weight: bold; }
    .loss { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class UltraStockAnalyzerPro:
    def __init__(self):
        # Reduced stock list for better performance
        self.all_symbols = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'INFOSYS': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'LT': 'LT.NS',
            'ITC': 'ITC.NS',
            'HUL': 'HINDUNILVR.NS'
        }
    
    @st.cache_data(ttl=1800, show_spinner=False)  # Reduced TTL for faster updates
    def get_stock_data(_self, symbol, period="6mo"):  # Reduced period for faster loading
        """Get comprehensive stock data with enhanced error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"⚠️ Data fetch error for {symbol}: {str(e)}")
            return None

    def calculate_advanced_indicators(self, data):
        """Calculate technical indicators with enhanced safety"""
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        try:
            # 1. BASIC PRICE INDICATORS - Only essential ones
            # Moving averages
            for period in [5, 8, 13, 21, 50]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
            # 2. MOMENTUM INDICATORS
            # RSI calculation
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            df['RSI_14'] = calculate_rsi(df['Close'], 14)
            df['RSI_9'] = calculate_rsi(df['Close'], 9)
            
            # MACD
            exp12 = df['Close'].ewm(span=12, adjust=False).mean()
            exp26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # 3. VOLUME INDICATORS
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            
            # 4. VOLATILITY INDICATORS
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
            
            # 5. SUPPORT/RESISTANCE
            df['Support_1'] = df['Low'].rolling(window=20, min_periods=1).min()
            df['Resistance_1'] = df['High'].rolling(window=20, min_periods=1).max()
            
            # Fill NaN values safely
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            st.error(f"Indicator calculation error: {str(e)}")
            # Return basic data if indicator calculation fails
            return data

    def calculate_position_size(self, capital, risk_per_trade, entry_price, stop_loss):
        """Advanced position sizing calculator with risk management"""
        try:
            risk_amount = capital * (risk_per_trade / 100)
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share > 0:
                shares = risk_amount / risk_per_share
                position_value = shares * entry_price
                return int(shares), position_value, risk_amount
            return 0, 0, risk_amount
        except:
            return 0, 0, 0

    def calculate_ai_score(self, df):
        """Enhanced AI-powered scoring with safety checks"""
        if df is None or len(df) < 20:
            return 50, ["Insufficient data for analysis"], {}, 0, 0
            
        try:
            current_price = df['Close'].iloc[-1]
            score = 50
            reasons = []
            signals = {}
            
            # 1. TREND ANALYSIS (40 points)
            trend_score = 0
            bullish_ma_count = 0
            
            # Moving averages analysis
            ma_periods = [5, 8, 13, 21, 50]
            for period in ma_periods:
                ema_col = f'EMA_{period}'
                if ema_col in df and not pd.isna(df[ema_col].iloc[-1]):
                    if current_price > df[ema_col].iloc[-1]:
                        trend_score += 6
                        bullish_ma_count += 1
            
            if bullish_ma_count >= 4:
                trend_score += 15
                reasons.append(f"🚀 STRONG TREND ({bullish_ma_count}/5 EMAs bullish)")
            elif bullish_ma_count >= 3:
                trend_score += 10
                reasons.append(f"📈 BULLISH TREND ({bullish_ma_count}/5 EMAs bullish)")
            
            score += min(trend_score, 40)
            
            # 2. MOMENTUM ANALYSIS (35 points)
            momentum_score = 0
            
            # RSI analysis
            rsi_bullish = 0
            for period in [9, 14]:
                rsi_col = f'RSI_{period}'
                if rsi_col in df and not pd.isna(df[rsi_col].iloc[-1]):
                    rsi_val = df[rsi_col].iloc[-1]
                    if 40 <= rsi_val <= 65:
                        momentum_score += 8
                        rsi_bullish += 1
                        reasons.append(f"🎯 Good RSI {period}: {rsi_val:.1f}")
                    elif rsi_val < 35:
                        momentum_score += 12
                        reasons.append(f"📈 Oversold RSI {period}: {rsi_val:.1f}")
            
            # MACD analysis
            if all(col in df for col in ['MACD', 'MACD_Signal']):
                if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    momentum_score += 15
                    reasons.append("💪 MACD BULLISH - Above signal line")
            
            score += min(momentum_score, 35)
            
            # 3. VOLUME CONFIRMATION (15 points)
            volume_score = 0
            
            if all(col in df for col in ['Volume', 'Volume_SMA_20']):
                volume_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
                if volume_ratio > 2.0:
                    volume_score += 10
                    reasons.append("💰 HIGH VOLUME - Strong interest")
                elif volume_ratio > 1.5:
                    volume_score += 6
                    reasons.append("💰 Good Volume - Participation")
            
            score += min(volume_score, 15)
            
            # 4. VOLATILITY & RISK (10 points)
            risk_score = 0
            
            if 'BB_Position' in df and not pd.isna(df['BB_Position'].iloc[-1]):
                bb_pos = df['BB_Position'].iloc[-1]
                if bb_pos < 0.3:
                    risk_score += 8
                    reasons.append("🎯 Near BB lower band - Good entry")
                elif 0.3 <= bb_pos <= 0.7:
                    risk_score += 4
            
            score += min(risk_score, 10)
            
            # FINAL CONFIDENCE BOOST
            total_reasons = len(reasons)
            if total_reasons >= 6:
                score = min(score + 15, 100)
                reasons.append("🚀 STRONG BULLISH CONVERGENCE")
            elif total_reasons >= 4:
                score = min(score + 8, 100)
                reasons.append("📈 BULLISH BIAS")
            
            bullish_signals = total_reasons
            total_signals = 10  # Estimated total possible signals
            
            return min(score, 100), reasons, signals, bullish_signals, total_signals
            
        except Exception as e:
            st.error(f"AI scoring error: {str(e)}")
            return 50, ["Analysis in progress..."], {}, 0, 0

    def get_trading_signal(self, score):
        """Get detailed trading signal with enhanced categorization"""
        if score >= 90:
            return "🚀 ULTRA STRONG BUY", "ultra-buy", "#10b981", "MAXIMUM CONFIDENCE - IMMEDIATE BUY"
        elif score >= 80:
            return "💎 VERY STRONG BUY", "ultra-buy", "#22c55e", "VERY STRONG BUY - HIGH CONFIDENCE"
        elif score >= 75:
            return "📈 POWERFUL BUY", "strong-buy", "#4ade80", "POWERFUL BUY - STRONG OPPORTUNITY"
        elif score >= 70:
            return "⚡ STRONG BUY", "strong-buy", "#84cc16", "STRONG BUY - GOOD ENTRY"
        elif score >= 65:
            return "💰 ACCUMULATE", "buy", "#a3e635", "ACCUMULATE ON DIPS"
        elif score >= 60:
            return "🔄 HOLD", "hold", "#f59e0b", "HOLD - WAIT FOR CONFIRMATION"
        elif score >= 55:
            return "🔔 CAUTION", "hold", "#f97316", "CAUTION - MONITOR CLOSELY"
        elif score >= 50:
            return "📉 REDUCE", "sell", "#ef4444", "REDUCE POSITION"
        else:
            return "💀 STRONG SELL", "strong-sell", "#dc2626", "STRONG SELL - AVOID"

    def create_advanced_chart(self, df, symbol):
        """Create professional multi-panel chart with safety checks"""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=(
                    f'<b>{symbol} - PRICE ANALYSIS</b>', 
                    '<b>RSI MOMENTUM</b>',
                    '<b>MACD TREND</b>'
                ),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Price Subplot
            fig.add_trace(go.Candlestick(
                x=df.index, 
                open=df['Open'], 
                high=df['High'], 
                low=df['Low'], 
                close=df['Close'],
                name='Price', 
                increasing_line_color='#00C805', 
                decreasing_line_color='#FF0000'
            ), row=1, col=1)
            
            # Key Moving Averages
            for period, color in [(8, '#FF6B35'), (21, '#00B4D8'), (50, '#7209B7')]:
                ema_col = f'EMA_{period}'
                if ema_col in df:
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df[ema_col], 
                        name=f'EMA {period}',
                        line=dict(color=color, width=2),
                        opacity=0.8
                    ), row=1, col=1)
            
            # Bollinger Bands
            if 'BB_Upper' in df and 'BB_Lower' in df:
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['BB_Upper'], 
                    name='BB Upper',
                    line=dict(color='#6B7280', dash='dash', width=1),
                    opacity=0.6
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['BB_Lower'], 
                    name='BB Lower',
                    line=dict(color='#6B7280', dash='dash', width=1),
                    opacity=0.6,
                    fill='tonexty'
                ), row=1, col=1)
            
            # RSI Subplot
            for period, color in [(9, '#8B5CF6'), (14, '#3B82F6')]:
                rsi_col = f'RSI_{period}'
                if rsi_col in df:
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df[rsi_col], 
                        name=f'RSI {period}',
                        line=dict(color=color, width=2)
                    ), row=2, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
            
            # MACD Subplot
            if all(col in df for col in ['MACD', 'MACD_Signal']):
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['MACD'], 
                    name='MACD',
                    line=dict(color='#2563EB', width=2)
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['MACD_Signal'], 
                    name='Signal',
                    line=dict(color='#EF4444', width=2)
                ), row=3, col=1)
                
                fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                template='plotly_white',
                xaxis_rangeslider_visible=False,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Chart creation error: {str(e)}")
            # Return simple price chart if advanced chart fails
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
            fig.update_layout(title=f"Basic Price Chart - {symbol}", height=400)
            return fig

def main():
    # Initialize the analyzer
    app = UltraStockAnalyzerPro()
    
    # Header Section with enhanced design
    st.markdown(
        '<h1 class="main-header">🚀 ULTRA STOCK ANALYZER PRO <span class="private-badge">PRIVATE</span></h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.4rem; color: #6b7280; margin-bottom: 2rem;">'
        '🔒 Professional Use Only • Advanced Indicators • AI-Powered Scoring</p>', 
        unsafe_allow_html=True
    )
    
    # Main Analysis Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 ANALYSIS PARAMETERS")
        
        # Stock selection with enhanced UI
        selected_stock = st.selectbox(
            "Select Stock:", 
            list(app.all_symbols.keys()),
            help="Choose from major Indian stocks and indices"
        )
        symbol = app.all_symbols[selected_stock]
        
        # REMOVED CUSTOM SYMBOL INPUT for better security and performance
        
        # TRADING SETTINGS with enhanced sliders
        st.subheader("💰 TRADING SETTINGS")
        col_a, col_b = st.columns(2)
        
        with col_a:
            stop_loss_percent = st.slider(
                "Stop Loss %", 
                1.0, 20.0, 8.0, 0.5,
                help="Risk management: Maximum loss percentage"
            )
            risk_reward = st.slider(
                "Risk/Reward Ratio", 
                1.0, 5.0, 1.5, 0.1,
                help="Minimum profit to risk ratio"
            )
        
        with col_b:
            target_percent = st.slider(
                "Target %", 
                1.0, 50.0, 15.0, 1.0,
                help="Profit target percentage"
            )
            position_size = st.slider(
                "Position Size (%)", 
                1, 100, 10,
                help="Percentage of capital to allocate"
            )
        
        # ADVANCED CALCULATOR
        st.subheader("🧮 POSITION CALCULATOR")
        capital = st.number_input(
            "Total Capital (₹)", 
            min_value=1000, 
            value=100000, 
            step=1000,
            help="Your total trading capital"
        )
        risk_per_trade = st.number_input(
            "Risk per Trade (%)", 
            min_value=0.1, 
            value=2.0, 
            step=0.1,
            help="Maximum risk per trade as percentage of capital"
        )
        
        # Main Analysis Button
        if st.button("🚀 RUN ADVANCED ANALYSIS", type="primary", use_container_width=True):
            with st.spinner("🔄 Running comprehensive analysis..."):
                data = app.get_stock_data(symbol, "6mo")  # Reduced period for faster loading
                
                if data is not None and not data.empty:
                    df = app.calculate_advanced_indicators(data)
                    
                    if df is not None:
                        score, reasons, signals, bullish_count, total_signals = app.calculate_ai_score(df)
                        signal, signal_class, color, advice = app.get_trading_signal(score)
                        current_price = df['Close'].iloc[-1]
                        
                        # Calculate trading parameters
                        stop_loss_price = current_price * (1 - stop_loss_percent/100)
                        target_price = current_price * (1 + target_percent/100)
                        actual_risk_reward = (target_price - current_price) / (current_price - stop_loss_price) if (current_price - stop_loss_price) > 0 else 0
                        
                        # Position calculation
                        shares, position_value, risk_amount = app.calculate_position_size(
                            capital, risk_per_trade, current_price, stop_loss_price
                        )
                        
                        # RESULTS DISPLAY
                        st.markdown(f'<div class="super-card {signal_class}">', unsafe_allow_html=True)
                        
                        col_s1, col_s2 = st.columns([2, 1])
                        with col_s1:
                            st.subheader(f"{signal}")
                            st.write(f"**AI Confidence Score:** `{score}/100`")
                            st.write(f"**Current Price:** `₹{current_price:.2f}`")
                            st.write(f"**Bullish Signals:** `{bullish_count}/{total_signals}`")
                        with col_s2:
                            st.metric("Signal Strength", f"{score}%", delta=f"{score-50}%")
                        
                        st.write(f"**Expert Advice:** {advice}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # TRADING RECOMMENDATION
                        st.markdown(f'''
                        <div class="trade-box">
                            <h3>💎 PROFESSIONAL TRADING RECOMMENDATION</h3>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                                <div>
                                    <p><strong>Entry Price</strong></p>
                                    <p style="font-size: 1.2rem;">₹{current_price:.2f}</p>
                                </div>
                                <div>
                                    <p><strong>Stop Loss</strong></p>
                                    <p style="font-size: 1.2rem;">₹{stop_loss_price:.2f}</p>
                                    <p style="font-size: 0.9rem;">({stop_loss_percent}% risk)</p>
                                </div>
                                <div>
                                    <p><strong>Target Price</strong></p>
                                    <p style="font-size: 1.2rem;">₹{target_price:.2f}</p>
                                    <p style="font-size: 0.9rem;">(+{target_percent}% gain)</p>
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                                <div>
                                    <p><strong>Risk/Reward</strong></p>
                                    <p style="font-size: 1.2rem;">1:{actual_risk_reward:.1f}</p>
                                </div>
                                <div>
                                    <p><strong>Position Size</strong></p>
                                    <p style="font-size: 1.2rem;">{position_size}%</p>
                                </div>
                                <div>
                                    <p><strong>Signal Confidence</strong></p>
                                    <p style="font-size: 1.2rem;">{score}%</p>
                                </div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # POSITION CALCULATION
                        st.subheader("💰 POSITION CALCULATION")
                        col_p1, col_p2, col_p3 = st.columns(3)
                        
                        with col_p1:
                            st.metric("Shares to Buy", f"{shares:,}")
                        with col_p2:
                            st.metric("Investment", f"₹{position_value:,.2f}")
                        with col_p3:
                            st.metric("Risk Amount", f"₹{risk_amount:,.2f}")
                        
                        # INDICATOR GRID
                        st.subheader("📊 TECHNICAL INDICATORS")
                        cols = st.columns(4)
                        
                        indicators = []
                        
                        # RSI Indicators
                        for period in [9, 14]:
                            rsi_col = f'RSI_{period}'
                            if rsi_col in df:
                                rsi_val = df[rsi_col].iloc[-1]
                                if rsi_val < 35:
                                    rsi_status = 'bullish'
                                elif rsi_val > 65:
                                    rsi_status = 'bearish'
                                else:
                                    rsi_status = 'neutral'
                                indicators.append((f'RSI {period}', f"{rsi_val:.1f}", rsi_status))
                        
                        # MACD
                        if all(col in df for col in ['MACD', 'MACD_Signal']):
                            macd_status = 'bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'bearish'
                            indicators.append(('MACD', 'BULL' if macd_status == 'bullish' else 'BEAR', macd_status))
                        
                        # Volume
                        if all(col in df for col in ['Volume', 'Volume_SMA_20']):
                            vol_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
                            vol_status = 'bullish' if vol_ratio > 1.5 else 'neutral'
                            indicators.append(('Volume', f"{vol_ratio:.1f}x", vol_status))
                        
                        # Bollinger Position
                        if 'BB_Position' in df:
                            bb_pos = df['BB_Position'].iloc[-1]
                            bb_status = 'bullish' if bb_pos < 0.3 else 'bearish' if bb_pos > 0.7 else 'neutral'
                            indicators.append(('BB Position', f"{bb_pos:.2f}", bb_status))
                        
                        # Display indicators in grid
                        for idx, (name, value, status) in enumerate(indicators[:8]):
                            with cols[idx % 4]:
                                st.markdown(f'''
                                <div class="indicator-box {status}">
                                    <h4>{name}</h4>
                                    <h3>{value}</h3>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        # ANALYSIS REASONS
                        with st.expander("📋 View Detailed Analysis Report", expanded=True):
                            for i, reason in enumerate(reasons[:8], 1):
                                st.write(f"{i}. {reason}")
                
                else:
                    st.error("❌ Could not fetch stock data. Please try again later.")
    
    with col2:
        if 'df' in locals() and df is not None:
            st.plotly_chart(app.create_advanced_chart(df, selected_stock), use_container_width=True)
            
            # RISK METRICS
            st.subheader("🛡️ RISK METRICS")
            risk_cols = st.columns(4)
            
            risk_metrics = [
                ("Stop Loss", f"₹{stop_loss_price:.1f}"),
                ("Target Price", f"₹{target_price:.1f}"),
                ("Risk/Reward", f"1:{actual_risk_reward:.1f}"),
                ("ATR", f"₹{df['ATR'].iloc[-1]:.2f}" if 'ATR' in df else "N/A")
            ]
            
            for idx, (name, value) in enumerate(risk_metrics):
                with risk_cols[idx]:
                    st.metric(name, value)

    # SIMPLIFIED MARKET SCANNER
    st.sidebar.header("⚡ SMART ACTIONS")
    
    if st.sidebar.button("🔍 SCAN TOP PERFORMERS", use_container_width=True):
        with st.spinner("Scanning for high-potential stocks..."):
            results = []
            # Scan only first 5 stocks for performance
            for stock_name, stock_symbol in list(app.all_symbols.items())[:5]:
                try:
                    data = app.get_stock_data(stock_symbol, "3mo")
                    if data is not None:
                        df = app.calculate_advanced_indicators(data)
                        if df is not None:
                            score, _, _, _, _ = app.calculate_ai_score(df)
                            if score >= 70:
                                current_price = df['Close'].iloc[-1]
                                results.append({
                                    'symbol': stock_name,
                                    'price': current_price,
                                    'score': score
                                })
                except:
                    continue
            
            if results:
                st.sidebar.subheader("💎 TOP PICKS")
                for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:3]:
                    st.sidebar.markdown(f'''
                    <div class="scan-result">
                        <h4>📈 {stock['symbol']}</h4>
                        <p><strong>AI Score:</strong> {stock['score']}/100</p>
                        <p><strong>Price:</strong> ₹{stock['price']:.2f}</p>
                    </div>
                    ''', unsafe_allow_html=True)

    # INFO SECTION
    st.sidebar.header("🔐 SECURITY")
    
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a, #3730a3); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
        <h4>🚀 PRO FEATURES</h4>
        <ul style="margin-left: -1rem;">
            <li>Advanced Technical Indicators</li>
            <li>AI-Powered Scoring</li>
            <li>Professional Charts</li>
            <li>Risk Management</li>
            <li>Real-time Analysis</li>
        </ul>
        <p><em>For professional use only</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
