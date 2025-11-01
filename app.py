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
    page_title="LIVE TRADING SIGNALS WITH CHART",
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
    .live-price {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .buy-signal {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #00ff88;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
    }
    .sell-signal {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        border: 3px solid #ff4444;
        box-shadow: 0 10px 30px rgba(255, 68, 68, 0.3);
    }
    .target-box {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        text-align: center;
    }
    .signal-marker {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        margin: 0.5rem;
        display: inline-block;
    }
    .buy-marker { background: #10b981; color: white; }
    .sell-marker { background: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)

class LiveTradingSignals:
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
    
    @st.cache_data(ttl=60)  # 1 minute cache for live data
    def get_live_data(_self, symbol):
        """Get live stock data with current price"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get historical data for analysis
            hist_data = stock.history(period="1mo")
            
            # Get live price
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', hist_data['Close'].iloc[-1] if not hist_data.empty else 0))
            
            return hist_data, current_price, info
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            return None, 0, None

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        if data is None or len(data) < 20:
            return data
            
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
        
        return df.fillna(method='bfill')

    def detect_buy_sell_signals(self, df, current_price):
        """Detect buy/sell signals with exact levels"""
        if df is None or len(df) < 20:
            return "HOLD", [], [], 50, "Insufficient data"
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        buy_signals = []
        sell_signals = []
        score = 50
        reasons = []
        
        # 1. MOVING AVERAGE CROSSOVER (BUY/SELL)
        if current['SMA_20'] > current['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            buy_signals.append(("MA Crossover", current_price, "SMA 20 crossed above SMA 50"))
            score += 20
            reasons.append("✅ MA Golden Cross - STRONG BUY")
        
        if current['SMA_20'] < current['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            sell_signals.append(("MA Crossover", current_price, "SMA 20 crossed below SMA 50"))
            score -= 20
            reasons.append("❌ MA Death Cross - STRONG SELL")
        
        # 2. RSI SIGNALS
        if current['RSI'] < 30:
            buy_signals.append(("RSI Oversold", current_price, "RSI below 30 - Oversold"))
            score += 15
            reasons.append("💰 RSI Oversold - BUY Opportunity")
        elif current['RSI'] > 70:
            sell_signals.append(("RSI Overbought", current_price, "RSI above 70 - Overbought"))
            score -= 15
            reasons.append("💀 RSI Overbought - SELL Opportunity")
        
        # 3. MACD CROSSOVER
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            buy_signals.append(("MACD Bullish", current_price, "MACD crossed above Signal"))
            score += 15
            reasons.append("📈 MACD Bullish Cross - BUY")
        
        if current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            sell_signals.append(("MACD Bearish", current_price, "MACD crossed below Signal"))
            score -= 15
            reasons.append("📉 MACD Bearish Cross - SELL")
        
        # 4. BOLLINGER BANDS
        if current_price <= current['BB_Lower']:
            buy_signals.append(("BB Support", current_price, "Price at BB Lower Band"))
            score += 10
            reasons.append("🎯 At BB Lower Band - BUY Zone")
        
        if current_price >= current['BB_Upper']:
            sell_signals.append(("BB Resistance", current_price, "Price at BB Upper Band"))
            score -= 10
            reasons.append("⚠️ At BB Upper Band - SELL Zone")
        
        # 5. VOLUME CONFIRMATION
        if current['Volume_Ratio'] > 1.5:
            if len(buy_signals) > len(sell_signals):
                score += 10
                reasons.append("🔥 High Volume BUY Confirmation")
            else:
                score -= 10
                reasons.append("💥 High Volume SELL Confirmation")
        
        # FINAL DECISION
        score = max(0, min(100, score))
        
        if len(buy_signals) > len(sell_signals) and score >= 60:
            action = "BUY"
            final_reason = f"STRONG BUY - {len(buy_signals)} signals confirming"
        elif len(sell_signals) > len(buy_signals) and score <= 40:
            action = "SELL"
            final_reason = f"STRONG SELL - {len(sell_signals)} signals confirming"
        else:
            action = "HOLD"
            final_reason = "Wait for better signals - Market consolidating"
        
        return action, buy_signals, sell_signals, score, reasons

    def calculate_price_targets(self, df, current_price, action):
        """Calculate price targets for buy/sell"""
        if df is None or len(df) < 20:
            return []
            
        current = df.iloc[-1]
        targets = []
        
        if action == "BUY":
            # For BUY - calculate upside targets
            
            # Target 1: Recent resistance
            recent_high = df['High'].tail(10).max()
            target1 = min(recent_high * 0.99, current_price * 1.08)  # Max 8% up
            targets.append(("Target 1", target1, f"Near resistance: ₹{target1:.2f}"))
            
            # Target 2: Bollinger Band upper
            target2 = current['BB_Upper'] * 0.98
            targets.append(("Target 2", target2, f"BB Upper: ₹{target2:.2f}"))
            
            # Target 3: ATR based (2x ATR)
            atr = df['High'].tail(14).std()  # Simple ATR approximation
            target3 = current_price + (atr * 2)
            targets.append(("Target 3", target3, f"Volatility target: ₹{target3:.2f}"))
            
        else:  # SELL targets
            # For SELL - calculate downside targets
            
            # Target 1: Recent support
            recent_low = df['Low'].tail(10).min()
            target1 = max(recent_low * 1.01, current_price * 0.92)  # Max 8% down
            targets.append(("Target 1", target1, f"Near support: ₹{target1:.2f}"))
            
            # Target 2: Bollinger Band lower
            target2 = current['BB_Lower'] * 1.02
            targets.append(("Target 2", target2, f"BB Lower: ₹{target2:.2f}"))
            
            # Target 3: ATR based (2x ATR)
            atr = df['High'].tail(14).std()
            target3 = current_price - (atr * 2)
            targets.append(("Target 3", target3, f"Volatility target: ₹{target3:.2f}"))
        
        return targets

    def create_live_chart_with_signals(self, df, symbol, current_price, buy_signals, sell_signals, targets, action):
        """Create chart with buy/sell signals and real-time price"""
        if df is None:
            return None
            
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                f'<b>{symbol} - LIVE TRADING SIGNALS</b>',
                '<b>RSI MOMENTUM</b>'
            ),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
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
            annotation_position="top left",
            row=1, col=1
        )
        
        # BUY Signals (Green triangles)
        if buy_signals:
            buy_dates = [df.index[-1]] * len(buy_signals)  # Current time
            buy_prices = [signal[1] for signal in buy_signals]
            buy_texts = [signal[2] for signal in buy_signals]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers+text',
                    name='BUY SIGNALS',
                    marker=dict(
                        symbol='triangle-up',
                        size=20,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    text=[f"BUY<br>{text}" for text in buy_texts],
                    textposition="top center",
                    hovertemplate='<b>BUY SIGNAL</b><br>%{text}<br>Price: ₹%{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
        
        # SELL Signals (Red triangles)
        if sell_signals:
            sell_dates = [df.index[-1]] * len(sell_signals)
            sell_prices = [signal[1] for signal in sell_signals]
            sell_texts = [signal[2] for signal in sell_signals]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers+text',
                    name='SELL SIGNALS',
                    marker=dict(
                        symbol='triangle-down',
                        size=20,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    text=[f"SELL<br>{text}" for text in sell_texts],
                    textposition="bottom center",
                    hovertemplate='<b>SELL SIGNAL</b><br>%{text}<br>Price: ₹%{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
        
        # Price Targets
        if targets:
            target_dates = [df.index[-1]] * len(targets)
            target_prices = [target[1] for target in targets]
            target_names = [target[0] for target in targets]
            
            fig.add_trace(
                go.Scatter(
                    x=target_dates,
                    y=target_prices,
                    mode='markers+text',
                    name='TARGETS',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='orange',
                        line=dict(width=2, color='darkorange')
                    ),
                    text=target_names,
                    textposition="middle right",
                    hovertemplate='<b>TARGET</b><br>%{text}<br>Price: ₹%{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
        
        # Moving Averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                      line=dict(dash='dash', color='gray', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                      line=dict(dash='dash', color='gray', width=1)),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            title_x=0.5
        )
        
        return fig

def main():
    st.markdown('<div class="main-header">🎯 LIVE TRADING SIGNALS WITH CHART</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280;">Real-time Buy/Sell Signals • Live Price Display • Chart Markers</p>', unsafe_allow_html=True)
    
    trader = LiveTradingSignals()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 STOCK SELECTION")
        selected_stock = st.selectbox("Select Stock:", list(trader.stock_list.keys()))
        symbol = trader.stock_list[selected_stock]
        
        st.header("⚙️ TRADING SETTINGS")
        auto_refresh = st.checkbox("Auto Refresh Every 30s", value=False)
        show_details = st.checkbox("Show Technical Details", value=True)
    
    # Auto refresh
    if auto_refresh:
        st.rerun()
    
    # Main Analysis
    if st.button("🔄 GET LIVE SIGNALS", type="primary", use_container_width=True):
        with st.spinner("Fetching live market data..."):
            # Get live data
            hist_data, current_price, info = trader.get_live_data(symbol)
            
            if hist_data is not None and current_price > 0:
                # Calculate indicators
                df = trader.calculate_technical_indicators(hist_data)
                
                # Detect signals
                action, buy_signals, sell_signals, score, reasons = trader.detect_buy_sell_signals(df, current_price)
                
                # Calculate targets
                targets = trader.calculate_price_targets(df, current_price, action)
                
                # Get current time
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Display LIVE PRICE
                st.markdown(f'''
                <div class="live-price">
                    <h2>💰 LIVE PRICE: ₹{current_price:.2f}</h2>
                    <p>Last Updated: {current_time}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Display TRADING SIGNAL
                if action == "BUY":
                    st.markdown(f'''
                    <div class="buy-signal">
                        <h1>🚀 STRONG BUY SIGNAL</h1>
                        <h2>Confidence Score: {score}%</h2>
                        <h3>{len(buy_signals)} BUY Signals Detected</h3>
                        <p>Best time to enter long position</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # BUY Instructions
                    st.subheader("📈 BUY INSTRUCTIONS")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Entry Price", f"₹{current_price:.2f}")
                    with col2:
                        stop_loss = current_price * 0.95
                        st.metric("Stop Loss", f"₹{stop_loss:.2f}")
                    with col3:
                        st.metric("Risk", "5%")
                
                elif action == "SELL":
                    st.markdown(f'''
                    <div class="sell-signal">
                        <h1>💀 STRONG SELL SIGNAL</h1>
                        <h2>Confidence Score: {100-score}%</h2>
                        <h3>{len(sell_signals)} SELL Signals Detected</h3>
                        <p>Best time to exit or short position</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # SELL Instructions
                    st.subheader("📉 SELL INSTRUCTIONS")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Exit Price", f"₹{current_price:.2f}")
                    with col2:
                        stop_loss = current_price * 1.05
                        st.metric("Stop Loss", f"₹{stop_loss:.2f}")
                    with col3:
                        st.metric("Risk", "5%")
                
                else:
                    st.info("""
                    ## ⚪ HOLD SIGNAL
                    **Market in consolidation phase. Wait for clearer signals.**
                    """)
                
                # PRICE TARGETS
                if targets and action in ["BUY", "SELL"]:
                    st.subheader("🎯 PRICE TARGETS")
                    
                    cols = st.columns(3)
                    for idx, (col, target) in enumerate(zip(cols, targets)):
                        with col:
                            profit_percent = ((target[1] - current_price) / current_price * 100) if action == "BUY" else ((current_price - target[1]) / current_price * 100)
                            st.markdown(f'''
                            <div class="target-box">
                                <h4>{target[0]}</h4>
                                <h2>₹{target[1]:.2f}</h2>
                                <p>{profit_percent:+.1f}%</p>
                                <small>{target[2]}</small>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    # TRADING PLAN
                    st.subheader("📋 TRADING PLAN")
                    if action == "BUY":
                        st.success(f"""
                        **EXECUTION PLAN:**
                        - **BUY NOW** at: ₹{current_price:.2f}
                        - **Stop Loss:** ₹{stop_loss:.2f} (Exit if price drops 5%)
                        - **Target 1:** ₹{targets[0][1]:.2f} (Sell 30% of position)
                        - **Target 2:** ₹{targets[1][1]:.2f} (Sell 40% of position)  
                        - **Target 3:** ₹{targets[2][1]:.2f} (Sell remaining 30%)
                        """)
                    else:
                        st.error(f"""
                        **EXECUTION PLAN:**
                        - **SELL NOW** at: ₹{current_price:.2f}
                        - **Stop Loss:** ₹{stop_loss:.2f} (Exit if price rises 5%)
                        - **Target 1:** ₹{targets[0][1]:.2f} (Cover 30% of position)
                        - **Target 2:** ₹{targets[1][1]:.2f} (Cover 40% of position)
                        - **Target 3:** ₹{targets[2][1]:.2f} (Cover remaining 30%)
                        """)
                
                # LIVE CHART WITH SIGNALS
                st.subheader("📊 LIVE CHART WITH BUY/SELL SIGNALS")
                chart = trader.create_live_chart_with_signals(df, selected_stock, current_price, buy_signals, sell_signals, targets, action)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    st.info("""
                    **📈 CHART LEGEND:**
                    - 🟢 **GREEN ARROW** = BUY Signal
                    - 🔴 **RED ARROW** = SELL Signal  
                    - 🟠 **ORANGE STAR** = Price Target
                    - 🔵 **BLUE LINE** = Current Price
                    """)
                
                # TECHNICAL ANALYSIS BREAKDOWN
                if show_details:
                    st.subheader("🔍 TECHNICAL ANALYSIS BREAKDOWN")
                    
                    # Display signals
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if buy_signals:
                            st.subheader("✅ BUY SIGNALS")
                            for signal in buy_signals:
                                st.markdown(f'<div class="signal-marker buy-marker">{signal[2]}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        if sell_signals:
                            st.subheader("❌ SELL SIGNALS")
                            for signal in sell_signals:
                                st.markdown(f'<div class="signal-marker sell-marker">{signal[2]}</div>', unsafe_allow_html=True)
                    
                    # Technical Reasons
                    st.subheader("📊 TECHNICAL INDICATORS")
                    for reason in reasons:
                        if "BUY" in reason or "✅" in reason:
                            st.success(reason)
                        elif "SELL" in reason or "❌" in reason:
                            st.error(reason)
                        else:
                            st.info(reason)
                
            else:
                st.error("❌ Could not fetch live data. Please try again.")

    # Quick Refresh Button
    st.sidebar.header("⚡ QUICK ACTIONS")
    if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

if __name__ == "__main__":
    main()
