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
        # Expanded stock list with major indices and stocks
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
            'HUL': 'HINDUNILVR.NS',
            'ASIAN PAINTS': 'ASIANPAINT.NS',
            'MARUTI': 'MARUTI.NS',
            'TITAN': 'TITAN.NS',
            'SUN PHARMA': 'SUNPHARMA.NS',
            'AXIS BANK': 'AXISBANK.NS',
            'KOTAK BANK': 'KOTAKBANK.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'WIPRO': 'WIPRO.NS',
            'HCL TECH': 'HCLTECH.NS',
            'ULTRACEMCO': 'ULTRACEMCO.NS',
            'NESTLE': 'NESTLEIND.NS',
            'POWERGRID': 'POWERGRID.NS',
            'NTPC': 'NTPC.NS',
            'ONGC': 'ONGC.NS',
            'M&M': 'M&M.NS',
            'TATA STEEL': 'TATASTEEL.NS',
            'JSW STEEL': 'JSWSTEEL.NS'
        }
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_stock_data(_self, symbol, period="1y"):
        """Get comprehensive stock data with enhanced error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None
            
            # Add additional info
            info = stock.info
            data['Company'] = info.get('longName', symbol)
            data['Sector'] = info.get('sector', 'N/A')
            
            return data
        except Exception as e:
            st.error(f"⚠️ Data fetch error for {symbol}: {str(e)}")
            return None

    def calculate_advanced_indicators(self, data):
        """Calculate 45+ technical indicators with enhanced accuracy"""
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        # 1. ENHANCED PRICE INDICATORS
        # Fibonacci sequence moving averages
        fib_periods = [5, 8, 13, 21, 34, 55, 89, 144, 200]
        for period in fib_periods:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # 2. ENHANCED MOMENTUM INDICATORS
        # Improved RSI with multiple timeframes
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI_14'] = calculate_rsi(df['Close'], 14)
        df['RSI_21'] = calculate_rsi(df['Close'], 21)
        df['RSI_9'] = calculate_rsi(df['Close'], 9)
        
        # Triple EMA for better trend detection
        ema1 = df['Close'].ewm(span=5, adjust=False).mean()
        ema2 = ema1.ewm(span=8, adjust=False).mean() 
        ema3 = ema2.ewm(span=13, adjust=False).mean()
        df['TEMA'] = (3 * ema1) - (3 * ema2) + ema3
        
        # MACD with enhanced signals
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 3. VOLUME-BASED INDICATORS
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_RSI'] = calculate_rsi(df['Volume'], 14)
        
        # On Balance Volume with smoothing
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=21, adjust=False).mean()
        
        # 4. ADVANCED VOLATILITY INDICATORS
        # Multiple Bollinger Bands
        for dev in [1.5, 2, 2.5]:
            df[f'BB_Middle_{dev}'] = df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=1).std()
            df[f'BB_Upper_{dev}'] = df[f'BB_Middle_{dev}'] + (bb_std * dev)
            df[f'BB_Lower_{dev}'] = df[f'BB_Middle_{dev}'] - (bb_std * dev)
        
        df['BB_Width'] = (df['BB_Upper_2'] - df['BB_Lower_2']) / df['BB_Middle_2']
        df['BB_Position'] = (df['Close'] - df['BB_Lower_2']) / (df['BB_Upper_2'] - df['BB_Lower_2'])
        
        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # Keltner Channel
        df['KC_Middle'] = df['EMA_20']
        df['KC_Upper'] = df['KC_Middle'] + (2 * df['ATR'])
        df['KC_Lower'] = df['KC_Middle'] - (2 * df['ATR'])
        
        # 5. SUPPORT/RESISTANCE INDICATORS
        df['Support_1'] = df['Low'].rolling(window=20, min_periods=1).min()
        df['Resistance_1'] = df['High'].rolling(window=20, min_periods=1).max()
        
        # Pivot Points
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = (2 * df['Pivot']) - df['Low']
        df['S1'] = (2 * df['Pivot']) - df['High']
        
        # 6. TREND STRENGTH INDICATORS - ADX
        def calculate_adx(high, low, close, window=14):
            # +DM and -DM
            high_diff = high.diff()
            low_diff = low.diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Smooth the values
            plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/window).mean() / 
                            pd.Series(true_range).ewm(alpha=1/window).mean())
            minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/window).mean() / 
                             pd.Series(true_range).ewm(alpha=1/window).mean())
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/window).mean()
            
            return adx, plus_di, minus_di
        
        df['ADX'], df['ADX_Pos'], df['ADX_Neg'] = calculate_adx(df['High'], df['Low'], df['Close'])
        
        # 7. MONEY FLOW INDICATORS
        # Money Flow Index
        def calculate_mfi(high, low, close, volume, window=14):
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = (typical_price > typical_price.shift()).astype(int) * money_flow
            negative_flow = (typical_price < typical_price.shift()).astype(int) * money_flow
            
            positive_sum = positive_flow.rolling(window=window, min_periods=1).sum()
            negative_sum = negative_flow.rolling(window=window, min_periods=1).sum()
            
            mfi = 100 - (100 / (1 + positive_sum / negative_sum))
            return mfi
        
        df['MFI'] = calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        
        return df

    def calculate_position_size(self, capital, risk_per_trade, entry_price, stop_loss):
        """Advanced position sizing calculator with risk management"""
        risk_amount = capital * (risk_per_trade / 100)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share > 0:
            shares = risk_amount / risk_per_share
            position_value = shares * entry_price
            return int(shares), position_value, risk_amount
        return 0, 0, risk_amount

    def calculate_ai_score(self, df):
        """Enhanced AI-powered scoring with 45+ indicators"""
        if df is None or len(df) < 50:
            return 0, [], {}, 0, 0
            
        current_price = df['Close'].iloc[-1]
        score = 0
        reasons = []
        signals = {}
        
        try:
            # 1. ENHANCED TREND ANALYSIS (40 points)
            trend_score = 0
            bullish_ma_count = 0
            
            # Fibonacci sequence MAs analysis
            fib_periods = [5, 8, 13, 21, 34, 55, 89, 144]
            for period in fib_periods:
                ema_col = f'EMA_{period}'
                if ema_col in df and not pd.isna(df[ema_col].iloc[-1]):
                    if current_price > df[ema_col].iloc[-1]:
                        trend_score += 3
                        bullish_ma_count += 1
            
            if bullish_ma_count >= 6:
                trend_score += 18
                reasons.append(f"🚀 STRONG FIBONACCI TREND ({bullish_ma_count}/8 EMAs bullish)")
                signals['Fibonacci_Trend'] = 'STRONG_BULLISH'
            elif bullish_ma_count >= 4:
                trend_score += 10
                reasons.append(f"📈 BULLISH FIBONACCI TREND ({bullish_ma_count}/8 EMAs bullish)")
                signals['Fibonacci_Trend'] = 'BULLISH'
            
            # TEMA analysis
            if 'TEMA' in df and not pd.isna(df['TEMA'].iloc[-1]):
                if current_price > df['TEMA'].iloc[-1]:
                    trend_score += 8
                    reasons.append("🎯 TEMA BULLISH - Strong trend confirmation")
                    signals['TEMA'] = 'BULLISH'
            
            score += min(trend_score, 40)
            
            # 2. ENHANCED MOMENTUM (35 points)
            momentum_score = 0
            
            # Multi-timeframe RSI analysis
            rsi_bullish = 0
            for period in [9, 14, 21]:
                rsi_col = f'RSI_{period}'
                if rsi_col in df and not pd.isna(df[rsi_col].iloc[-1]):
                    rsi_val = df[rsi_col].iloc[-1]
                    if 40 <= rsi_val <= 65:
                        momentum_score += 6
                        rsi_bullish += 1
                        reasons.append(f"🎯 Perfect RSI {period}: {rsi_val:.1f}")
                    elif rsi_val < 35:
                        momentum_score += 8
                        reasons.append(f"📈 Oversold RSI {period}: {rsi_val:.1f}")
                        signals[f'RSI_{period}'] = 'OVERSOLD_BULLISH'
            
            # MACD with histogram confirmation
            if all(col in df for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                if (df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and 
                    df['MACD_Histogram'].iloc[-1] > 0):
                    momentum_score += 12
                    reasons.append("💪 MACD STRONG BULLISH - Positive histogram")
                    signals['MACD'] = 'STRONG_BULLISH'
                elif df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    momentum_score += 8
                    reasons.append("📈 MACD BULLISH - Above signal line")
                    signals['MACD'] = 'BULLISH'
            
            score += min(momentum_score, 35)
            
            # 3. VOLUME CONFIRMATION (20 points)
            volume_score = 0
            
            # Volume spike analysis
            if all(col in df for col in ['Volume', 'Volume_SMA_20']):
                volume_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
                if volume_ratio > 2.5:
                    volume_score += 12
                    reasons.append("💰 VOLUME EXPLOSION - Very strong interest")
                    signals['Volume'] = 'EXTREME_BULLISH'
                elif volume_ratio > 1.8:
                    volume_score += 8
                    reasons.append("💰 HIGH VOLUME - Strong participation")
                    signals['Volume'] = 'STRONG_BULLISH'
                elif volume_ratio > 1.2:
                    volume_score += 4
                    signals['Volume'] = 'BULLISH'
            
            # OBV trend confirmation
            if 'OBV_EMA' in df and len(df) > 21:
                if df['OBV'].iloc[-1] > df['OBV_EMA'].iloc[-1]:
                    volume_score += 5
                    reasons.append("📊 OBV above EMA - Accumulation detected")
                    signals['OBV'] = 'BULLISH'
            
            score += min(volume_score, 20)
            
            # 4. VOLATILITY & RISK (15 points)
            risk_score = 0
            
            # Bollinger Band position analysis
            if 'BB_Position' in df and not pd.isna(df['BB_Position'].iloc[-1]):
                bb_pos = df['BB_Position'].iloc[-1]
                if 0.2 <= bb_pos <= 0.8:
                    risk_score += 6
                    reasons.append("📈 Good BB position")
                    signals['Bollinger_Bands'] = 'NEUTRAL'
                elif bb_pos < 0.2:
                    risk_score += 10
                    reasons.append("🎯 Near BB lower band - Excellent entry")
                    signals['Bollinger_Bands'] = 'OVERSOLD_BULLISH'
            
            # Low volatility with strong trend
            if 'ATR' in df and not pd.isna(df['ATR'].iloc[-1]):
                atr_percent = (df['ATR'].iloc[-1] / current_price) * 100
                if atr_percent < 1.8:
                    risk_score += 5
                    reasons.append("🛡️ Low volatility - Reduced risk")
                    signals['ATR'] = 'LOW_VOL'
            
            score += min(risk_score, 15)
            
            # 5. TREND STRENGTH (10 points)
            trend_strength_score = 0
            
            if 'ADX' in df and not pd.isna(df['ADX'].iloc[-1]):
                adx_val = df['ADX'].iloc[-1]
                if adx_val > 25:
                    trend_strength_score += 8
                    reasons.append(f"💪 Strong trend (ADX: {adx_val:.1f})")
                    signals['ADX'] = 'STRONG_TREND'
                elif adx_val > 20:
                    trend_strength_score += 5
                    signals['ADX'] = 'MODERATE_TREND'
            
            score += min(trend_strength_score, 10)
            
            # FINAL CONFIDENCE BOOST
            bullish_signals = sum(1 for s in signals.values() if 'BULLISH' in str(s))
            total_signals = len(signals)
            
            if bullish_signals >= 8:
                score = min(score + 25, 100)
                reasons.append(f"🚀 EXTREME BULLISH CONVERGENCE ({bullish_signals}/{total_signals} signals)")
            elif bullish_signals >= 6:
                score = min(score + 15, 100)
                reasons.append(f"📈 STRONG BULLISH CONVERGENCE ({bullish_signals}/{total_signals} signals)")
            elif bullish_signals >= 4:
                score = min(score + 8, 100)
                reasons.append(f"✅ BULLISH BIAS ({bullish_signals}/{total_signals} signals)")
            
        except Exception as e:
            st.error(f"Error in AI scoring: {str(e)}")
        
        return min(score, 100), reasons, signals, bullish_signals, total_signals

    def get_trading_signal(self, score):
        """Get detailed trading signal with enhanced categorization"""
        if score >= 95:
            return "🚀 QUANTUM STRONG BUY", "ultra-buy", "#059669", "IMMEDIATE BUY - EXTREME CONFIDENCE"
        elif score >= 90:
            return "🎯 ULTRA STRONG BUY", "ultra-buy", "#10b981", "ULTRA STRONG BUY - MAXIMUM CONFIDENCE"
        elif score >= 85:
            return "💎 VERY STRONG BUY", "ultra-buy", "#22c55e", "VERY STRONG BUY - HIGH CONFIDENCE"
        elif score >= 80:
            return "📈 POWERFUL BUY", "strong-buy", "#4ade80", "POWERFUL BUY - STRONG OPPORTUNITY"
        elif score >= 75:
            return "⚡ STRONG BUY", "strong-buy", "#84cc16", "STRONG BUY - GOOD ENTRY"
        elif score >= 70:
            return "💰 ACCUMULATE", "buy", "#a3e635", "ACCUMULATE ON DIPS"
        elif score >= 65:
            return "🔄 HOLD", "hold", "#f59e0b", "HOLD - WAIT FOR CONFIRMATION"
        elif score >= 60:
            return "🔔 CAUTION", "hold", "#f97316", "CAUTION - MONITOR CLOSELY"
        elif score >= 55:
            return "📉 REDUCE", "sell", "#ef4444", "REDUCE POSITION"
        elif score >= 50:
            return "⚠️ SELL", "sell", "#dc2626", "SELL - WEAK OUTLOOK"
        else:
            return "💀 STRONG SELL", "strong-sell", "#991b1b", "STRONG SELL - AVOID COMPLETELY"

    def create_advanced_chart(self, df, symbol):
        """Create professional multi-panel chart with enhanced visuals"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                f'<b>{symbol} - ADVANCED PRICE ANALYSIS</b>', 
                '<b>VOLUME & MONEY FLOW</b>',
                '<b>RSI MOMENTUM ANALYSIS</b>', 
                '<b>MACD TREND STRENGTH</b>'
            ),
            row_heights=[0.4, 0.15, 0.2, 0.25]
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
        
        # Key Fibonacci Moving Averages
        for period, color in [(8, '#FF6B35'), (21, '#00B4D8'), (55, '#7209B7'), (144, '#F72585')]:
            ema_col = f'EMA_{period}'
            if ema_col in df:
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df[ema_col], 
                    name=f'EMA {period}',
                    line=dict(color=color, width=2.5),
                    opacity=0.8
                ), row=1, col=1)
        
        # Bollinger Bands
        if 'BB_Upper_2' in df and 'BB_Lower_2' in df:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['BB_Upper_2'], 
                name='BB Upper',
                line=dict(color='#6B7280', dash='dash', width=1.5),
                opacity=0.7
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['BB_Lower_2'], 
                name='BB Lower',
                line=dict(color='#6B7280', dash='dash', width=1.5),
                opacity=0.7,
                fill='tonexty'
            ), row=1, col=1)
        
        # Volume Subplot with enhanced colors
        colors = ['#00C805' if row['Close'] >= row['Open'] else '#FF0000' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df.index, 
            y=df['Volume'], 
            name='Volume',
            marker_color=colors,
            opacity=0.8
        ), row=2, col=1)
        
        # Volume SMA
        if 'Volume_SMA_20' in df:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['Volume_SMA_20'], 
                name='Vol SMA 20',
                line=dict(color='#FBBF24', width=2.5)
            ), row=2, col=1)
        
        # Multi-Timeframe RSI
        for period, color in [(9, '#8B5CF6'), (14, '#3B82F6'), (21, '#06B6D4')]:
            rsi_col = f'RSI_{period}'
            if rsi_col in df:
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df[rsi_col], 
                    name=f'RSI {period}',
                    line=dict(color=color, width=2.2)
                ), row=3, col=1)
        
        # RSI Levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=2, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=2, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1, row=3, col=1)
        
        # MACD Subplot
        if all(col in df for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['MACD'], 
                name='MACD',
                line=dict(color='#2563EB', width=2.5)
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['MACD_Signal'], 
                name='Signal',
                line=dict(color='#EF4444', width=2.5)
            ), row=4, col=1)
            
            # Enhanced Histogram
            colors_hist = ['#10B981' if x >= 0 else '#EF4444' for x in df['MACD_Histogram']]
            fig.add_trace(go.Bar(
                x=df.index, 
                y=df['MACD_Histogram'], 
                name='Histogram',
                marker_color=colors_hist,
                opacity=0.7
            ), row=4, col=1)
            
            fig.add_hline(y=0, line_color="black", line_width=2, row=4, col=1)
        
        # Update layout with professional styling
        fig.update_layout(
            title=dict(
                text=f'<b>PROFESSIONAL TRADING ANALYSIS - {symbol}</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#1F2937')
            ),
            height=1100,
            showlegend=True,
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            font=dict(family="Arial, sans-serif", size=12, color="#1F2937"),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Update subplot titles
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=14, color='#1F2937')
        
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
        '🔒 Professional Use Only • 45+ Advanced Indicators • AI-Powered Scoring</p>', 
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
        
        # Manual symbol input
        custom_symbol = st.text_input(
            "Or Enter Custom Symbol:", 
            placeholder="e.g., TATAMOTORS.NS, ADANIENT.NS",
            help="Enter any valid Yahoo Finance symbol"
        )
        if custom_symbol:
            symbol = custom_symbol.upper()
        
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
        with st.expander("💰 Open Advanced Calculator", expanded=False):
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
            
            if st.button("📊 Calculate Position Size", use_container_width=True):
                # Get current price for calculation
                data = app.get_stock_data(symbol, "1d")
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    stop_loss = current_price * (1 - stop_loss_percent/100)
                    
                    shares, position_value, risk_amount = app.calculate_position_size(
                        capital, risk_per_trade, current_price, stop_loss
                    )
                    
                    st.markdown(f'''
                    <div class="calculator-box">
                        <h3>💰 ADVANCED POSITION CALCULATION</h3>
                        <p><strong>Current Price:</strong> ₹{current_price:.2f}</p>
                        <p><strong>Stop Loss:</strong> ₹{stop_loss:.2f} (-{stop_loss_percent}%)</p>
                        <p><strong>Target Price:</strong> ₹{current_price * (1 + target_percent/100):.2f} (+{target_percent}%)</p>
                        <p><strong>Shares to Buy:</strong> {shares:,}</p>
                        <p><strong>Position Value:</strong> ₹{position_value:,.2f}</p>
                        <p><strong>Risk Amount:</strong> ₹{risk_amount:,.2f}</p>
                        <p><strong>Risk/Reward:</strong> 1:{risk_reward:.1f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
        
        # Main Analysis Button
        if st.button("🚀 RUN ULTRA ADVANCED ANALYSIS", type="primary", use_container_width=True):
            with st.spinner("🔄 Running comprehensive analysis with 45+ indicators..."):
                data = app.get_stock_data(symbol, "1y")
                
                if data is not None and not data.empty:
                    df = app.calculate_advanced_indicators(data)
                    
                    if df is not None:
                        score, reasons, signals, bullish_count, total_signals = app.calculate_ai_score(df)
                        signal, signal_class, color, advice = app.get_trading_signal(score)
                        current_price = df['Close'].iloc[-1]
                        
                        # Calculate trading parameters
                        stop_loss_price = current_price * (1 - stop_loss_percent/100)
                        target_price = current_price * (1 + target_percent/100)
                        actual_risk_reward = (target_price - current_price) / (current_price - stop_loss_price)
                        
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
                        
                        # ENHANCED INDICATOR GRID
                        st.subheader("📊 ADVANCED INDICATOR SIGNALS")
                        cols = st.columns(4)
                        
                        indicators = []
                        
                        # RSI Indicators
                        for period in [9, 14, 21]:
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
                        
                        # Trend Strength
                        if 'ADX' in df:
                            adx_val = df['ADX'].iloc[-1]
                            trend_status = 'bullish' if adx_val > 25 else 'neutral'
                            indicators.append(('Trend Power', f"{adx_val:.1f}", trend_status))
                        
                        # Bollinger Position
                        if 'BB_Position' in df:
                            bb_pos = df['BB_Position'].iloc[-1]
                            bb_status = 'bullish' if bb_pos < 0.3 else 'bearish' if bb_pos > 0.7 else 'neutral'
                            indicators.append(('BB Position', f"{bb_pos:.2f}", bb_status))
                        
                        # Money Flow
                        if 'MFI' in df:
                            mfi_val = df['MFI'].iloc[-1]
                            mfi_status = 'bullish' if 20 <= mfi_val <= 80 else 'neutral'
                            indicators.append(('MFI', f"{mfi_val:.1f}", mfi_status))
                        
                        # TEMA
                        if 'TEMA' in df:
                            tema_status = 'bullish' if current_price > df['TEMA'].iloc[-1] else 'bearish'
                            indicators.append(('TEMA', 'BULL' if tema_status == 'bullish' else 'BEAR', tema_status))
                        
                        # OBV Trend
                        if 'OBV_EMA' in df:
                            obv_status = 'bullish' if df['OBV'].iloc[-1] > df['OBV_EMA'].iloc[-1] else 'bearish'
                            indicators.append(('OBV Trend', 'RISING' if obv_status == 'bullish' else 'FALLING', obv_status))
                        
                        # Display indicators in grid
                        for idx, (name, value, status) in enumerate(indicators[:8]):
                            with cols[idx % 4]:
                                st.markdown(f'''
                                <div class="indicator-box {status}">
                                    <h4>{name}</h4>
                                    <h3>{value}</h3>
                                </div>
                                ''', unsafe_allow_html=True)
            
            with col2:
                if 'df' in locals() and df is not None:
                    st.plotly_chart(app.create_advanced_chart(df, selected_stock), use_container_width=True)
                    
                    # DETAILED ANALYSIS REPORT
                    st.subheader("🔍 COMPREHENSIVE ANALYSIS REPORT")
                    
                    with st.expander("📋 View Detailed Technical Analysis", expanded=True):
                        for i, reason in enumerate(reasons[:12], 1):
                            st.write(f"{i}. {reason}")
                    
                    # ADVANCED RISK MANAGEMENT
                    st.subheader("🛡️ ADVANCED RISK MANAGEMENT")
                    risk_cols = st.columns(6)
                    
                    risk_metrics = [
                        ("Stop Loss", f"₹{stop_loss_price:.1f}", "#ef4444"),
                        ("Target Price", f"₹{target_price:.1f}", "#10b981"),
                        ("Risk/Reward", f"1:{actual_risk_reward:.1f}", "#f59e0b"),
                        ("ATR", f"₹{df['ATR'].iloc[-1]:.2f}" if 'ATR' in df else "N/A", "#8b5cf6"),
                        ("Support", f"₹{df['Support_1'].iloc[-1]:.1f}" if 'Support_1' in df else "N/A", "#06b6d4"),
                        ("Resistance", f"₹{df['Resistance_1'].iloc[-1]:.1f}" if 'Resistance_1' in df else "N/A", "#f97316")
                    ]
                    
                    for idx, (name, value, color) in enumerate(risk_metrics):
                        with risk_cols[idx]:
                            st.metric(name, value)

    # ENHANCED MARKET SCANNER
    st.sidebar.header("⚡ SMART ACTIONS")
    
    if st.sidebar.button("🔍 SCAN TOP PERFORMERS", use_container_width=True):
        with st.spinner("🔄 Scanning market for high-potential stocks..."):
            results = []
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            for i, (stock_name, stock_symbol) in enumerate(app.all_symbols.items()):
                try:
                    status_text.text(f"Analyzing {stock_name}...")
                    data = app.get_stock_data(stock_symbol, "6mo")
                    if data is not None and not data.empty:
                        df = app.calculate_advanced_indicators(data)
                        if df is not None:
                            score, _, _, bullish_count, _ = app.calculate_ai_score(df)
                            if score >= 75:  # Only show high-confidence picks
                                current_price = df['Close'].iloc[-1]
                                prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                                change_percent = ((current_price - prev_price) / prev_price) * 100
                                
                                results.append({
                                    'symbol': stock_name,
                                    'price': current_price,
                                    'change': change_percent,
                                    'score': score,
                                    'bullish': bullish_count
                                })
                    progress_bar.progress((i + 1) / len(app.all_symbols))
                except Exception as e:
                    continue
            
            if results:
                st.subheader("💎 TOP MARKET PICKS")
                st.info(f"Found {len(results)} high-potential stocks with AI score ≥ 75")
                
                for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:6]:
                    change_color = "profit" if stock['change'] >= 0 else "loss"
                    change_icon = "📈" if stock['change'] >= 0 else "📉"
                    
                    st.markdown(f'''
                    <div class="scan-result">
                        <h4>{change_icon} {stock['symbol']}</h4>
                        <p><strong>AI Score:</strong> {stock['score']}/100</p>
                        <p><strong>Current Price:</strong> ₹{stock['price']:.2f}</p>
                        <p><strong>Daily Change:</strong> <span class="{change_color}">{stock['change']:+.2f}%</span></p>
                        <p><strong>Bullish Signals:</strong> {stock['bullish']}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.warning("No high-confidence stocks found in current market conditions.")

    # Quick Actions for Popular Stocks
    st.sidebar.header("🎯 QUICK ANALYSIS")
    popular_stocks = ['NIFTY 50', 'BANK NIFTY', 'RELIANCE', 'TCS', 'HDFC BANK', 'INFOSYS']
    
    for stock in popular_stocks:
        if st.sidebar.button(f"📊 {stock}", use_container_width=True):
            # This would trigger analysis for the selected stock
            st.session_state.quick_stock = stock
            st.rerun()

    # Security & Info Section
    st.sidebar.header("🔐 SECURITY")
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a, #3730a3); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
        <h4>🔒 PRIVATE ACCESS</h4>
        <p><strong>Features:</strong></p>
        <ul style="margin-left: -1rem;">
            <li>45+ Advanced Indicators</li>
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
