import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config - MUST BE FIRST
st.set_page_config(
    page_title="ULTRA STOCK ANALYZER PRO",
    page_icon="📈",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563eb;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #2563eb;
    }
    .buy { border-left-color: #10b981; background: #f0fdf4; }
    .sell { border-left-color: #ef4444; background: #fef2f2; }
    .hold { border-left-color: #f59e0b; background: #fffbeb; }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
    }
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
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'LT': 'LT.NS'
        }
    
    @st.cache_data(ttl=1800)
    def get_stock_data(_self, symbol):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="6mo")
            return data if not data.empty else None
        except Exception as e:
            st.error(f"Data error: {str(e)}")
            return None

    def calculate_indicators(self, data):
        if data is None or len(data) < 20:
            return data
            
        df = data.copy()
        
        # Moving Averages
        for period in [20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Support & Resistance
        df['Support'] = df['Low'].rolling(20).min()
        df['Resistance'] = df['High'].rolling(20).max()
        
        return df.fillna(method='bfill')

    def calculate_ai_score(self, df):
        if df is None or len(df) < 50:
            return 50, ["Insufficient data"], 0, 0
            
        current_price = df['Close'].iloc[-1]
        score = 50
        reasons = []
        bullish_signals = 0
        
        try:
            # RSI Analysis
            if 'RSI' in df:
                rsi = df['RSI'].iloc[-1]
                if rsi < 30:
                    score += 20
                    reasons.append("RSI Oversold - Strong Buy Signal")
                    bullish_signals += 1
                elif rsi < 45:
                    score += 10
                    reasons.append("RSI Bullish")
                    bullish_signals += 1
                elif rsi > 70:
                    score -= 20
                    reasons.append("RSI Overbought - Caution")
            
            # Moving Average Analysis
            if all(col in df for col in ['SMA_20', 'SMA_50']):
                if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
                    score += 15
                    reasons.append("Short-term trend bullish")
                    bullish_signals += 1
                else:
                    score -= 10
                    reasons.append("Short-term trend bearish")
            
            # MACD Analysis
            if all(col in df for col in ['MACD', 'MACD_Signal']):
                if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    score += 15
                    reasons.append("MACD Bullish")
                    bullish_signals += 1
                else:
                    score -= 10
                    reasons.append("MACD Bearish")
            
            # Price Position
            if 'SMA_20' in df:
                if current_price > df['SMA_20'].iloc[-1]:
                    score += 10
                    reasons.append("Price above 20-day average")
                    bullish_signals += 1
            
            # Volume Analysis (simplified)
            if 'Volume' in df and len(df) > 20:
                avg_volume = df['Volume'].tail(20).mean()
                if df['Volume'].iloc[-1] > avg_volume * 1.5:
                    score += 5
                    reasons.append("High volume - Strong interest")
                    bullish_signals += 1
            
            return min(max(score, 0), 100), reasons, bullish_signals, 8
            
        except Exception as e:
            return 50, [f"Analysis error: {str(e)}"], 0, 0

    def get_trading_signal(self, score):
        if score >= 80:
            return "🚀 STRONG BUY", "buy"
        elif score >= 70:
            return "📈 BUY", "buy"
        elif score >= 60:
            return "🔄 HOLD", "hold"
        elif score >= 50:
            return "📉 REDUCE", "sell"
        else:
            return "💀 STRONG SELL", "sell"

    def create_advanced_chart(self, df, symbol):
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price', 'RSI', 'MACD'),
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
        for period, color in [(20, 'orange'), (50, 'red')]:
            if f'SMA_{period}' in df:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df[f'SMA_{period}'], 
                        name=f'SMA {period}',
                        line=dict(color=color)
                    ), row=1, col=1
                )
        
        # Bollinger Bands
        if all(col in df for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(dash='dash'), fill='tonexty'),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in df:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD
        if all(col in df for col in ['MACD', 'MACD_Signal']):
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
    st.markdown('<div class="main-header">🚀 ULTRA STOCK ANALYZER PRO</div>', unsafe_allow_html=True)
    
    analyzer = StockAnalyzerPro()
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 ANALYSIS PARAMETERS")
        selected_stock = st.selectbox("Select Stock:", list(analyzer.stock_list.keys()))
        symbol = analyzer.stock_list[selected_stock]
        
        st.header("💰 TRADING SETTINGS")
        col1, col2 = st.columns(2)
        with col1:
            stop_loss = st.slider("Stop Loss %", 1.0, 20.0, 8.0)
        with col2:
            target = st.slider("Target %", 1.0, 50.0, 15.0)
        
        capital = st.number_input("Capital (₹)", 1000, 10000000, 100000)
        risk_per_trade = st.slider("Risk per Trade %", 0.1, 10.0, 2.0)
    
    # Main Analysis
    if st.button("🚀 RUN ADVANCED ANALYSIS", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing with advanced indicators..."):
            data = analyzer.get_stock_data(symbol)
            
            if data is not None and not data.empty:
                df = analyzer.calculate_indicators(data)
                score, reasons, bullish_count, total_signals = analyzer.calculate_ai_score(df)
                signal, signal_class = analyzer.get_trading_signal(score)
                current_price = df['Close'].iloc[-1]
                
                # Trading calculations
                stop_loss_price = current_price * (1 - stop_loss/100)
                target_price = current_price * (1 + target/100)
                risk_amount = capital * (risk_per_trade/100)
                risk_per_share = abs(current_price - stop_loss_price)
                shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                investment = shares * current_price
                
                # Display Results
                st.markdown(f'<div class="card {signal_class}">', unsafe_allow_html=True)
                st.subheader(f"{signal}")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("AI Score", f"{score}/100")
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col_b:
                    st.metric("Stop Loss", f"₹{stop_loss_price:.2f}")
                    st.metric("Target", f"₹{target_price:.2f}")
                with col_c:
                    st.metric("Bullish Signals", f"{bullish_count}/{total_signals}")
                    st.metric("Risk/Reward", f"1:{(target_price-current_price)/(current_price-stop_loss_price):.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Position Calculator
                st.subheader("💰 POSITION CALCULATOR")
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                with col_p1:
                    st.metric("Shares", f"{shares:,}")
                with col_p2:
                    st.metric("Investment", f"₹{investment:,.2f}")
                with col_p3:
                    st.metric("Risk Amount", f"₹{risk_amount:,.2f}")
                with col_p4:
                    st.metric("Risk per Share", f"₹{risk_per_share:.2f}")
                
                # Technical Chart
                st.subheader("📊 ADVANCED TECHNICAL CHART")
                st.plotly_chart(analyzer.create_advanced_chart(df, selected_stock), use_container_width=True)
                
                # Analysis Reasons
                st.subheader("🔍 ANALYSIS BREAKDOWN")
                for i, reason in enumerate(reasons, 1):
                    st.write(f"{i}. {reason}")
                
                # Technical Indicators
                st.subheader("⚙️ TECHNICAL INDICATORS")
                tech_cols = st.columns(4)
                indicators = []
                
                if 'RSI' in df:
                    rsi_val = df['RSI'].iloc[-1]
                    indicators.append(("RSI", f"{rsi_val:.1f}"))
                
                if all(col in df for col in ['MACD', 'MACD_Signal']):
                    macd_val = df['MACD'].iloc[-1]
                    indicators.append(("MACD", f"{macd_val:.4f}"))
                
                if 'BB_Width' in df:
                    bb_width = df['BB_Width'].iloc[-1]
                    indicators.append(("BB Width", f"{(bb_width*100):.1f}%"))
                
                if 'Support' in df:
                    support = df['Support'].iloc[-1]
                    indicators.append(("Support", f"₹{support:.1f}"))
                
                if 'Resistance' in df:
                    resistance = df['Resistance'].iloc[-1]
                    indicators.append(("Resistance", f"₹{resistance:.1f}"))
                
                for idx, (name, value) in enumerate(indicators):
                    with tech_cols[idx % 4]:
                        st.markdown(f'<div class="metric-box"><strong>{name}</strong><br>{value}</div>', unsafe_allow_html=True)
                        
            else:
                st.error("❌ Could not fetch stock data. Please try again.")

    # Quick Analysis Buttons
    st.sidebar.header("⚡ QUICK ACTIONS")
    if st.sidebar.button("📈 Market Overview", use_container_width=True):
        st.info("Market overview feature - Check major indices performance")
    
    if st.sidebar.button("🔍 Top Performers", use_container_width=True):
        with st.spinner("Scanning for top stocks..."):
            results = []
            for stock_name, stock_symbol in list(analyzer.stock_list.items())[:5]:
                try:
                    data = analyzer.get_stock_data(stock_symbol)
                    if data is not None:
                        df = analyzer.calculate_indicators(data)
                        if df is not None:
                            score, _, _, _ = analyzer.calculate_ai_score(df)
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
                st.success("💎 TOP PERFORMING STOCKS")
                for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:3]:
                    st.write(f"**{stock['symbol']}** - Score: {stock['score']}/100 - Price: ₹{stock['price']:.2f}")

if __name__ == "__main__":
    main()
