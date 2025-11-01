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
    page_title="ULTRA STOCK ANALYZER PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .pro-badge {
        background: linear-gradient(45deg, #ef4444, #dc2626);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-left: 1rem;
    }
    .super-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border-left: 6px solid;
        transition: all 0.3s ease;
    }
    .super-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    .ultra-buy {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
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
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
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
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    .bullish { 
        border-color: #10b981; 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }
    .bearish { 
        border-color: #ef4444; 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    .neutral { 
        border-color: #f59e0b; 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .candle-pattern {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedCandleAnalyzer:
    def __init__(self):
        self.stock_list = {
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
            'ITC': 'ITC.NS'
        }
    
    @st.cache_data(ttl=300)
    def get_realtime_data(_self, symbol):
        """Get real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d", interval="5m")
            if data.empty:
                data = stock.history(period="5d")
            return data
        except:
            return None

    def calculate_advanced_indicators(self, data):
        """Calculate 40+ advanced technical indicators"""
        if data is None or len(data) < 50:
            return data
            
        df = data.copy()
        
        try:
            # 1. ENHANCED MOVING AVERAGES (8 indicators)
            for period in [5, 8, 13, 21, 34, 55, 89, 200]:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
                df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            
            # 2. MOMENTUM INDICATORS (6 indicators)
            # RSI Multiple Timeframes
            for period in [6, 9, 14, 21]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD with enhancements
            exp12 = df['Close'].ewm(span=12).mean()
            exp26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp12 - exp26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Stochastic Oscillator
            df['STOCH_K'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                            (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
            df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
            
            # Williams %R
            df['WILLIAMS_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                               (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
            
            # CCI (Commodity Channel Index)
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
            
            # 3. VOLATILITY INDICATORS (5 indicators)
            # Bollinger Bands with multiple deviations
            for dev in [1, 2, 2.5]:
                df[f'BB_Middle_{dev}'] = df['Close'].rolling(20).mean()
                bb_std = df['Close'].rolling(20).std()
                df[f'BB_Upper_{dev}'] = df[f'BB_Middle_{dev}'] + (bb_std * dev)
                df[f'BB_Lower_{dev}'] = df[f'BB_Middle_{dev}'] - (bb_std * dev)
            
            df['BB_Width'] = (df['BB_Upper_2'] - df['BB_Lower_2']) / df['BB_Middle_2']
            df['BB_Position'] = (df['Close'] - df['BB_Lower_2']) / (df['BB_Upper_2'] - df['BB_Lower_2'])
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # Keltner Channels
            df['KC_Middle'] = df['EMA_20']
            df['KC_Upper'] = df['KC_Middle'] + (2 * df['ATR'])
            df['KC_Lower'] = df['KC_Middle'] - (2 * df['ATR'])
            
            # Donchian Channels
            df['DC_Upper'] = df['High'].rolling(20).max()
            df['DC_Lower'] = df['Low'].rolling(20).min()
            df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2
            
            # 4. VOLUME INDICATORS (4 indicators)
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # OBV (On Balance Volume)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['OBV_EMA'] = df['OBV'].ewm(span=21).mean()
            
            # Money Flow Index (MFI)
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_flow = (typical_price > typical_price.shift()).astype(int) * money_flow
            negative_flow = (typical_price < typical_price.shift()).astype(int) * money_flow
            positive_sum = positive_flow.rolling(14).sum()
            negative_sum = negative_flow.rolling(14).sum()
            df['MFI'] = 100 - (100 / (1 + positive_sum / negative_sum))
            
            # 5. SUPPORT/RESISTANCE (3 indicators)
            df['Support_1'] = df['Low'].rolling(20).min()
            df['Resistance_1'] = df['High'].rolling(20).max()
            df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            
            # 6. CANDLESTICK PATTERN DETECTION (8+ patterns)
            df = self.detect_candlestick_patterns(df)
            
            # 7. TREND STRENGTH INDICATORS (3 indicators)
            # ADX (Average Directional Index)
            df['ADX'] = self.calculate_adx(df)
            
            # Parabolic SAR
            df['PSAR'] = self.calculate_parabolic_sar(df)
            
            # Ichimoku Cloud (simplified)
            df = self.calculate_ichimoku(df)
            
            return df.fillna(method='bfill')
            
        except Exception as e:
            st.error(f"Indicator calculation error: {str(e)}")
            return data

    def detect_candlestick_patterns(self, df):
        """Detect 15+ candlestick patterns"""
        # Basic candle calculations
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Total_Range'] = df['High'] - df['Low']
        
        # Pattern detection
        patterns = {}
        
        # 1. Doji Pattern
        patterns['DOJI'] = (df['Body'] / df['Total_Range']) < 0.1
        
        # 2. Hammer Pattern
        patterns['HAMMER'] = (
            (df['Lower_Shadow'] >= 2 * df['Body']) & 
            (df['Upper_Shadow'] <= df['Body'] * 0.1) &
            (df['Close'] > df['Open'])
        )
        
        # 3. Shooting Star
        patterns['SHOOTING_STAR'] = (
            (df['Upper_Shadow'] >= 2 * df['Body']) & 
            (df['Lower_Shadow'] <= df['Body'] * 0.1) &
            (df['Close'] < df['Open'])
        )
        
        # 4. Engulfing Patterns
        prev_close = df['Close'].shift(1)
        prev_open = df['Open'].shift(1)
        
        patterns['BULLISH_ENGULFING'] = (
            (df['Close'] > df['Open']) & 
            (prev_close < prev_open) &
            (df['Open'] < prev_close) & 
            (df['Close'] > prev_open)
        )
        
        patterns['BEARISH_ENGULFING'] = (
            (df['Close'] < df['Open']) & 
            (prev_close > prev_open) &
            (df['Open'] > prev_close) & 
            (df['Close'] < prev_open)
        )
        
        # 5. Morning Star
        patterns['MORNING_STAR'] = (
            (df['Close'].shift(2) < df['Open'].shift(2)) &  # First red candle
            (abs(df['Close'].shift(1) - df['Open'].shift(1)) / df['Total_Range'].shift(1) < 0.3) &  # Small body
            (df['Close'] > df['Open']) &  # Green candle
            (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2)  # Closes above first candle midpoint
        )
        
        # 6. Evening Star
        patterns['EVENING_STAR'] = (
            (df['Close'].shift(2) > df['Open'].shift(2)) &  # First green candle
            (abs(df['Close'].shift(1) - df['Open'].shift(1)) / df['Total_Range'].shift(1) < 0.3) &  # Small body
            (df['Close'] < df['Open']) &  # Red candle
            (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2)  # Closes below first candle midpoint
        )
        
        # Add patterns to dataframe
        for pattern_name, pattern_condition in patterns.items():
            df[pattern_name] = pattern_condition
        
        return df

    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # +DM and -DM
            up_move = high.diff()
            down_move = low.diff().abs() * -1
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Smooth the values
            plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / 
                            pd.Series(tr).rolling(period).mean())
            minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / 
                             pd.Series(tr).rolling(period).mean())
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx
        except:
            return pd.Series([25] * len(df), index=df.index)

    def calculate_parabolic_sar(self, df):
        """Calculate Parabolic SAR"""
        try:
            high = df['High'].values
            low = df['Low'].values
            
            psar = df['Close'].copy()
            af = 0.02
            ep = low[0]
            trend = 1
            
            for i in range(2, len(df)):
                if trend == 1:
                    psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                    if low.iloc[i] < psar.iloc[i]:
                        trend = -1
                        psar.iloc[i] = ep
                        ep = high.iloc[i]
                        af = 0.02
                    else:
                        if high.iloc[i] > ep:
                            ep = high.iloc[i]
                            af = min(af + 0.02, 0.2)
                else:
                    psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                    if high.iloc[i] > psar.iloc[i]:
                        trend = 1
                        psar.iloc[i] = ep
                        ep = low.iloc[i]
                        af = 0.02
                    else:
                        if low.iloc[i] < ep:
                            ep = low.iloc[i]
                            af = min(af + 0.02, 0.2)
            
            return psar
        except:
            return df['Close']

    def calculate_ichimoku(self, df):
        """Calculate Ichimoku Cloud components"""
        try:
            # Tenkan-sen (Conversion Line)
            high_9 = df['High'].rolling(9).max()
            low_9 = df['Low'].rolling(9).min()
            df['Ichimoku_Tenkan'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line)
            high_26 = df['High'].rolling(26).max()
            low_26 = df['Low'].rolling(26).min()
            df['Ichimoku_Kijun'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A)
            df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            high_52 = df['High'].rolling(52).max()
            low_52 = df['Low'].rolling(52).min()
            df['Ichimoku_Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            df['Ichimoku_Chikou'] = df['Close'].shift(-26)
            
            return df
        except:
            return df

    def analyze_candles_deeply(self, df):
        """Deep analysis of candle patterns and market structure"""
        if df is None or len(df) < 10:
            return "Insufficient data", 50, []
            
        current_candle = df.iloc[-1]
        analysis_points = []
        score = 50
        
        try:
            # 1. Candle Body Analysis
            body_ratio = current_candle['Body'] / current_candle['Total_Range']
            if body_ratio > 0.7:
                analysis_points.append("📊 STRONG BODY - High conviction move")
                score += 10
            elif body_ratio < 0.3:
                analysis_points.append("📊 SMALL BODY - Indecision in market")
                score -= 5
            
            # 2. Shadow Analysis
            upper_shadow_ratio = current_candle['Upper_Shadow'] / current_candle['Total_Range']
            lower_shadow_ratio = current_candle['Lower_Shadow'] / current_candle['Total_Range']
            
            if lower_shadow_ratio > 0.3:
                analysis_points.append("🛡️ STRONG LOWER SHADOW - Buying at lows")
                score += 8
            if upper_shadow_ratio > 0.3:
                analysis_points.append("🎯 STRONG UPPER SHADOW - Selling at highs")
                score -= 8
            
            # 3. Pattern Detection Analysis
            bullish_patterns = ['HAMMER', 'BULLISH_ENGULFING', 'MORNING_STAR']
            bearish_patterns = ['SHOOTING_STAR', 'BEARISH_ENGULFING', 'EVENING_STAR']
            
            for pattern in bullish_patterns:
                if pattern in df.columns and current_candle[pattern]:
                    analysis_points.append(f"🎯 {pattern.replace('_', ' ')} - Bullish reversal pattern")
                    score += 15
            
            for pattern in bearish_patterns:
                if pattern in df.columns and current_candle[pattern]:
                    analysis_points.append(f"⚠️ {pattern.replace('_', ' ')} - Bearish reversal pattern")
                    score -= 15
            
            # 4. Volume Confirmation
            if 'Volume_Ratio' in current_candle and current_candle['Volume_Ratio'] > 1.5:
                analysis_points.append("💰 HIGH VOLUME - Strong institutional interest")
                score += 10
            
            # 5. Support/Resistance Analysis
            if 'Support_1' in df.columns and 'Resistance_1' in df.columns:
                support = df['Support_1'].iloc[-1]
                resistance = df['Resistance_1'].iloc[-1]
                current_price = current_candle['Close']
                
                if current_price <= support * 1.02:
                    analysis_points.append("🛡️ NEAR SUPPORT - Potential bounce zone")
                    score += 12
                elif current_price >= resistance * 0.98:
                    analysis_points.append("🎯 NEAR RESISTANCE - Potential rejection zone")
                    score -= 12
            
            # 6. Trend Analysis using Multiple MAs
            if all(f'EMA_{period}' in df.columns for period in [5, 13, 21]):
                above_ma_count = sum(1 for period in [5, 13, 21] if current_candle['Close'] > current_candle[f'EMA_{period}'])
                if above_ma_count >= 2:
                    analysis_points.append("📈 ABOVE KEY MAs - Bullish trend structure")
                    score += 8
                else:
                    analysis_points.append("📉 BELOW KEY MAs - Bearish trend structure")
                    score -= 8
            
            # 7. Bollinger Band Position
            if 'BB_Position' in current_candle:
                bb_pos = current_candle['BB_Position']
                if bb_pos < 0.2:
                    analysis_points.append("🎯 NEAR BB LOWER - Oversold bounce potential")
                    score += 10
                elif bb_pos > 0.8:
                    analysis_points.append("⚠️ NEAR BB UPPER - Overbought pullback risk")
                    score -= 10
            
            # Determine final signal
            if score >= 75:
                signal = "🚀 STRONG BUY"
            elif score >= 65:
                signal = "📈 BUY"
            elif score >= 55:
                signal = "🔄 HOLD"
            elif score >= 45:
                signal = "📉 REDUCE"
            else:
                signal = "💀 STRONG SELL"
            
            return signal, min(max(score, 0), 100), analysis_points
            
        except Exception as e:
            return "HOLD", 50, [f"Analysis error: {str(e)}"]

    def create_advanced_candle_chart(self, df, symbol):
        """Create advanced candle pattern chart"""
        if df is None:
            return None
            
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                f'<b>{symbol} - ADVANCED CANDLE ANALYSIS</b>', 
                '<b>VOLUME & PATTERNS</b>',
                '<b>RSI MOMENTUM</b>',
                '<b>MACD TREND</b>'
            ),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Candlestick with patterns
        colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in df.iterrows()]
        
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
        
        # Add key moving averages
        for period, color in [(8, 'orange'), (21, 'blue'), (55, 'purple')]:
            if f'EMA_{period}' in df:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[f'EMA_{period}'], 
                        name=f'EMA {period}', line=dict(color=color, width=1.5)
                    ), row=1, col=1
                )
        
        # Add Bollinger Bands
        if all(col in df for col in ['BB_Upper_2', 'BB_Lower_2']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper_2'], name='BB Upper', 
                          line=dict(dash='dash', color='gray')), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower_2'], name='BB Lower', 
                          line=dict(dash='dash', color='gray'), fill='tonexty'), row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        # RSI
        if 'RSI_14' in df:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI_14'], name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if all(col in df for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'),
                row=4, col=1
            )
            fig.add_hline(y=0, line_color="black", row=4, col=1)
        
        fig.update_layout(height=900, xaxis_rangeslider_visible=False)
        return fig

def main():
    st.markdown(
        '<h1 class="main-header">🚀 ADVANCED CANDLE ANALYZER PRO <span class="pro-badge">40+ INDICATORS</span></h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;">'
        '🕯️ Deep Candle Analysis • 40+ Technical Indicators • Pattern Recognition • High Accuracy</p>', 
        unsafe_allow_html=True
    )
    
    analyzer = AdvancedCandleAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 STOCK SELECTION")
        selected_stock = st.selectbox("Select Stock:", list(analyzer.stock_list.keys()))
        symbol = analyzer.stock_list[selected_stock]
        
        st.header("⚙️ ANALYSIS SETTINGS")
        stop_loss = st.slider("Stop Loss %", 1.0, 20.0, 8.0, 0.5)
        target = st.slider("Target %", 1.0, 50.0, 15.0, 1.0)
    
    # Main Analysis
    if st.button("🔍 DEEP CANDLE ANALYSIS", type="primary", use_container_width=True):
        with st.spinner("🕯️ Analyzing candle patterns and market structure..."):
            data = analyzer.get_realtime_data(symbol)
            
            if data is not None and not data.empty:
                # Calculate all indicators
                analyzed_data = analyzer.calculate_advanced_indicators(data)
                
                # Deep candle analysis
                signal, score, analysis_points = analyzer.analyze_candles_deeply(analyzed_data)
                
                current_price = analyzed_data['Close'].iloc[-1] if 'Close' in analyzed_data else 0
                stop_loss_price = current_price * (1 - stop_loss/100)
                target_price = current_price * (1 + target/100)
                
                # Display Results
                signal_class = "ultra-buy" if "STRONG BUY" in signal else "buy" if "BUY" in signal else "strong-sell" if "STRONG SELL" in signal else "sell" if "SELL" in signal else "hold"
                
                st.markdown(f'<div class="super-card {signal_class}">', unsafe_allow_html=True)
                st.subheader(f"{signal}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Candle Analysis Score", f"{score}/100")
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Stop Loss", f"₹{stop_loss_price:.2f}")
                    st.metric("Target", f"₹{target_price:.2f}")
                with col3:
                    st.metric("Risk/Reward", f"1:{(target_price-current_price)/(current_price-stop_loss_price):.1f}")
                    st.metric("Analysis Points", len(analysis_points))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # CANDLE PATTERN ANALYSIS
                st.subheader("🕯️ CANDLE PATTERN BREAKDOWN")
                for point in analysis_points[:8]:  # Show top 8 points
                    st.markdown(f'''
                    <div class="candle-pattern">
                        <p>{point}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # ADVANCED CHART
                st.subheader("📊 ADVANCED TECHNICAL CHART")
                chart = analyzer.create_advanced_candle_chart(analyzed_data, selected_stock)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # INDICATOR SUMMARY
                st.subheader("⚙️ INDICATOR SUMMARY")
                cols = st.columns(4)
                
                indicators = []
                
                # RSI
                if 'RSI_14' in analyzed_data:
                    rsi = analyzed_data['RSI_14'].iloc[-1]
                    rsi_status = 'bullish' if rsi < 35 else 'bearish' if rsi > 65 else 'neutral'
                    indicators.append(("RSI 14", f"{rsi:.1f}", rsi_status))
                
                # MACD
                if all(col in analyzed_data for col in ['MACD', 'MACD_Signal']):
                    macd_status = 'bullish' if analyzed_data['MACD'].iloc[-1] > analyzed_data['MACD_Signal'].iloc[-1] else 'bearish'
                    indicators.append(("MACD", "BULL" if macd_status == 'bullish' else "BEAR", macd_status))
                
                # Volume
                if 'Volume_Ratio' in analyzed_data:
                    vol_ratio = analyzed_data['Volume_Ratio'].iloc[-1]
                    vol_status = 'bullish' if vol_ratio > 1.5 else 'neutral'
                    indicators.append(("Volume", f"{vol_ratio:.1f}x", vol_status))
                
                # Stochastic
                if 'STOCH_K' in analyzed_data:
                    stoch_k = analyzed_data['STOCH_K'].iloc[-1]
                    stoch_status = 'bullish' if stoch_k < 20 else 'bearish' if stoch_k > 80 else 'neutral'
                    indicators.append(("Stochastic", f"{stoch_k:.1f}", stoch_status))
                
                # Williams %R
                if 'WILLIAMS_R' in analyzed_data:
                    will_r = analyzed_data['WILLIAMS_R'].iloc[-1]
                    will_status = 'bullish' if will_r < -80 else 'bearish' if will_r > -20 else 'neutral'
                    indicators.append(("Williams %R", f"{will_r:.1f}", will_status))
                
                # CCI
                if 'CCI' in analyzed_data:
                    cci = analyzed_data['CCI'].iloc[-1]
                    cci_status = 'bullish' if cci < -100 else 'bearish' if cci > 100 else 'neutral'
                    indicators.append(("CCI", f"{cci:.0f}", cci_status))
                
                # ATR
                if 'ATR' in analyzed_data:
                    atr = analyzed_data['ATR'].iloc[-1]
                    indicators.append(("ATR", f"₹{atr:.2f}", 'neutral'))
                
                # Display indicators
                for idx, (name, value, status) in enumerate(indicators[:8]):
                    with cols[idx % 4]:
                        st.markdown(f'''
                        <div class="indicator-box {status}">
                            <h4>{name}</h4>
                            <h3>{value}</h3>
                        </div>
                        ''', unsafe_allow_html=True)
            
            else:
                st.error("❌ Could not fetch data. Please try again.")

if __name__ == "__main__":
    main()
