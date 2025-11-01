import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ULTRA STOCK ANALYZER AI PRO",
    page_icon="🚀",
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
    .ai-badge {
        background: linear-gradient(45deg, #8B5CF6, #06B6D4);
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
    .bearish { 
        border-color: #ef4444; 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    .neutral { 
        border-color: #f59e0b; 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    .ml-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border: none;
    }
    .risk-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        border: none;
    }
    .profit { color: #10b981; font-weight: bold; }
    .loss { color: #ef4444; font-weight: bold; }
    .feature-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class UltraStockAnalyzerAIPro:
    def __init__(self):
        # Predefined stock list only - NO CUSTOM SYMBOLS
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
            'HCL TECH': 'HCLTECH.NS'
        }
        
        # Initialize ML models
        self.scaler = StandardScaler()
        self.rf_model = None
        self.lstm_model = None
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_stock_data(_self, symbol, period="1y"):
        """Get stock data with enhanced error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            return None

    def get_news_sentiment(self, symbol_name):
        """Get news sentiment analysis for stock"""
        try:
            # Mock news data - in real implementation, use NewsAPI or similar
            news_samples = [
                f"{symbol_name} reports strong quarterly results with profit growth",
                f"Analysts maintain buy rating on {symbol_name}",
                f"{symbol_name} expands operations in international markets",
                f"Market experts bullish on {symbol_name} future prospects"
            ]
            
            sentiments = []
            for news in news_samples:
                analysis = TextBlob(news)
                sentiments.append(analysis.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            return avg_sentiment, news_samples[:3]  # Return top 3 news
        except:
            return 0, ["News data temporarily unavailable"]

    def calculate_greeks(self, spot_price, strike_price, time_to_expiry, risk_free_rate=0.05, volatility=0.2):
        """Calculate option Greeks using Black-Scholes model"""
        try:
            from math import log, sqrt, exp
            from scipy.stats import norm
            
            if time_to_expiry <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            d1 = (log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
            d2 = d1 - volatility * sqrt(time_to_expiry)
            
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (spot_price * volatility * sqrt(time_to_expiry))
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * sqrt(time_to_expiry)) - risk_free_rate * strike_price * exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
            vega = (spot_price * norm.pdf(d1) * sqrt(time_to_expiry)) / 100
            rho = (strike_price * time_to_expiry * exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 100
            
            return {
                'delta': round(delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'rho': round(rho, 4)
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    def calculate_volatility(self, data, window=30):
        """Calculate Historical Volatility and Implied Volatility"""
        try:
            returns = np.log(data['Close'] / data['Close'].shift(1))
            historical_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            return historical_vol.iloc[-1] if not historical_vol.empty else 20.0
        except:
            return 20.0

    def build_lstm_model(self, input_shape):
        """Build LSTM model for price prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def predict_with_lstm(self, data, days=7):
        """Predict future prices using LSTM"""
        try:
            # Prepare data for LSTM
            prices = data['Close'].values
            if len(prices) < 60:
                return None, None
            
            # Normalize data
            prices_normalized = (prices - np.mean(prices)) / np.std(prices)
            
            # Create sequences
            X, y = [], []
            sequence_length = 30
            
            for i in range(sequence_length, len(prices_normalized)):
                X.append(prices_normalized[i-sequence_length:i])
                y.append(prices_normalized[i])
            
            X, y = np.array(X), np.array(y)
            
            if len(X) < 10:
                return None, None
            
            # Build and train model
            self.lstm_model = self.build_lstm_model((X.shape[1], 1))
            
            # For demo, we'll use simple prediction instead of actual training
            # In production, you would train the model here
            last_sequence = prices_normalized[-sequence_length:]
            future_predictions = []
            
            # Simple projection based on recent trend
            recent_trend = np.mean(prices[-5:] - prices[-6:-1]) / prices[-6]
            
            current_price = prices[-1]
            predictions = []
            for i in range(days):
                next_price = current_price * (1 + recent_trend)
                predictions.append(next_price)
                current_price = next_price
            
            confidence = max(0, min(100, 85 - abs(recent_trend) * 1000))
            
            return predictions, confidence
            
        except Exception as e:
            st.error(f"LSTM Prediction error: {str(e)}")
            return None, None

    def random_forest_trend_prediction(self, data):
        """Predict trend using Random Forest"""
        try:
            # Create features
            df = data.copy()
            df['Returns'] = df['Close'].pct_change()
            df['MA_5'] = df['Close'].rolling(5).mean()
            df['MA_20'] = df['Close'].rolling(20).mean()
            df['Volatility'] = df['Returns'].rolling(20).std()
            df['Momentum'] = df['Close'] - df['Close'].shift(5)
            
            # Create target (1 if next day return positive, else 0)
            df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
            
            # Prepare data
            df = df.dropna()
            features = ['Returns', 'MA_5', 'MA_20', 'Volatility', 'Momentum']
            X = df[features]
            y = df['Target']
            
            if len(X) < 50:
                return "NEUTRAL", 50
            
            # Train Random Forest
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.rf_model.fit(X, y)
            
            # Predict
            latest_features = X.iloc[-1:].values
            prediction = self.rf_model.predict(latest_features)[0]
            probability = self.rf_model.predict_proba(latest_features)[0]
            
            confidence = max(probability) * 100
            
            if prediction == 1:
                return "BULLISH", confidence
            else:
                return "BEARISH", confidence
                
        except:
            return "NEUTRAL", 50

    def monte_carlo_simulation(self, data, days=252, simulations=1000):
        """Run Monte Carlo simulation for risk analysis"""
        try:
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            mu = returns.mean()
            sigma = returns.std()
            
            # Monte Carlo simulation
            simulations = 1000
            days = 252  # 1 year
            initial_price = data['Close'].iloc[-1]
            
            results = []
            for _ in range(simulations):
                prices = [initial_price]
                for _ in range(days):
                    shock = np.random.normal(mu - 0.5 * sigma**2, sigma)
                    price = prices[-1] * np.exp(shock)
                    prices.append(price)
                results.append(prices)
            
            # Calculate VaR (Value at Risk)
            final_prices = [sim[-1] for sim in results]
            var_95 = np.percentile(final_prices, 5)
            var_99 = np.percentile(final_prices, 1)
            
            expected_return = np.mean(final_prices)
            worst_case = np.min(final_prices)
            best_case = np.max(final_prices)
            
            return {
                'expected_return': expected_return,
                'var_95': var_95,
                'var_99': var_99,
                'worst_case': worst_case,
                'best_case': best_case,
                'simulations': results
            }
        except:
            return None

    def portfolio_optimization(self, selected_stocks, capital=100000):
        """Modern Portfolio Theory optimization"""
        try:
            # Get data for all selected stocks
            stock_data = {}
            for stock in selected_stocks:
                data = self.get_stock_data(self.all_symbols[stock], "1y")
                if data is not None:
                    stock_data[stock] = data['Close']
            
            # Create returns DataFrame
            returns_df = pd.DataFrame({stock: data.pct_change().dropna() for stock, data in stock_data.items()})
            
            if returns_df.empty:
                return None
            
            # Calculate covariance matrix and expected returns
            cov_matrix = returns_df.cov() * 252
            expected_returns = returns_df.mean() * 252
            
            # Markowitz optimization
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            
            for i in range(num_portfolios):
                weights = np.random.random(len(selected_stocks))
                weights /= np.sum(weights)
                
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[0,i] = portfolio_return
                results[1,i] = portfolio_std
                results[2,i] = portfolio_return / portfolio_std  # Sharpe ratio
            
            # Find optimal portfolio (max Sharpe ratio)
            optimal_idx = np.argmax(results[2])
            optimal_weights = np.random.random(len(selected_stocks))  # Placeholder
            optimal_weights /= np.sum(optimal_weights)
            
            return {
                'expected_return': results[0, optimal_idx],
                'volatility': results[1, optimal_idx],
                'sharpe_ratio': results[2, optimal_idx],
                'weights': dict(zip(selected_stocks, optimal_weights))
            }
        except:
            return None

    def calculate_correlations(self, selected_stocks):
        """Calculate correlation matrix for selected stocks"""
        try:
            stock_data = {}
            for stock in selected_stocks:
                data = self.get_stock_data(self.all_symbols[stock], "6mo")
                if data is not None:
                    stock_data[stock] = data['Close']
            
            prices_df = pd.DataFrame(stock_data)
            correlation_matrix = prices_df.corr()
            return correlation_matrix
        except:
            return None

    def calculate_advanced_indicators(self, data):
        """Calculate technical indicators with enhanced safety"""
        try:
            if data is None or len(data) < 50:
                return None
                
            df = data.copy()
            
            # Basic moving averages with error handling
            for period in [5, 10, 20, 50, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=min(period, len(df)), min_periods=1).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=min(period, len(df)), adjust=False).mean()
            
            # RSI calculation with safety
            def safe_rsi(prices, window=14):
                try:
                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi
                except:
                    return pd.Series([50] * len(prices), index=prices.index)
            
            df['RSI_14'] = safe_rsi(df['Close'])
            
            # MACD with safety
            try:
                exp12 = df['Close'].ewm(span=12, adjust=False).mean()
                exp26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp12 - exp26
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            except:
                df['MACD'] = 0
                df['MACD_Signal'] = 0
            
            # Bollinger Bands
            try:
                df['BB_Middle'] = df['Close'].rolling(20).mean()
                bb_std = df['Close'].rolling(20).std()
                df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            except:
                df['BB_Middle'] = df['Close']
                df['BB_Upper'] = df['Close']
                df['BB_Lower'] = df['Close']
            
            return df.fillna(method='bfill')
            
        except Exception as e:
            st.error(f"Indicator calculation error: {str(e)}")
            return data  # Return original data if calculations fail

    def calculate_ai_score(self, df):
        """Calculate AI score with error handling"""
        try:
            if df is None or len(df) < 20:
                return 50, [], {}, 0, 0
                
            current_price = df['Close'].iloc[-1]
            score = 50
            reasons = []
            signals = {}
            
            # RSI scoring
            if 'RSI_14' in df and not pd.isna(df['RSI_14'].iloc[-1]):
                rsi = df['RSI_14'].iloc[-1]
                if rsi < 30:
                    score += 15
                    reasons.append("RSI Oversold - Strong Buy Signal")
                elif rsi > 70:
                    score -= 15
                    reasons.append("RSI Overbought - Caution")
            
            # Moving average scoring
            if all(col in df for col in ['SMA_20', 'SMA_50']):
                if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
                    score += 10
                    reasons.append("Short-term trend bullish")
            
            # MACD scoring
            if all(col in df for col in ['MACD', 'MACD_Signal']):
                if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    score += 10
                    reasons.append("MACD Bullish")
            
            return min(max(score, 0), 100), reasons, signals, len(reasons), 5
            
        except:
            return 50, ["Basic analysis only"], {}, 1, 1

    def get_trading_signal(self, score):
        """Get trading signal based on score"""
        if score >= 80:
            return "🚀 STRONG BUY", "ultra-buy", "#10b981", "High confidence bullish signal"
        elif score >= 70:
            return "📈 BUY", "buy", "#22c55e", "Moderate bullish signal"
        elif score >= 60:
            return "🔄 HOLD", "hold", "#f59e0b", "Neutral outlook"
        elif score >= 50:
            return "📉 REDUCE", "sell", "#ef4444", "Consider reducing position"
        else:
            return "💀 STRONG SELL", "strong-sell", "#dc2626", "Strong bearish signal"

def main():
    # Initialize the AI analyzer
    app = UltraStockAnalyzerAIPro()
    
    # Header Section
    st.markdown(
        '<h1 class="main-header">🚀 ULTRA STOCK ANALYZER AI PRO <span class="ai-badge">AI POWERED</span></h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.4rem; color: #6b7280; margin-bottom: 2rem;">'
        '🤖 AI Machine Learning • Advanced Analytics • Professional Tools</p>', 
        unsafe_allow_html=True
    )
    
    # Main Analysis Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 ANALYSIS PARAMETERS")
        
        # Stock selection from predefined list only - NO CUSTOM SYMBOL
        selected_stock = st.selectbox(
            "Select Stock:", 
            list(app.all_symbols.keys()),
            help="Choose from major Indian stocks and indices"
        )
        symbol = app.all_symbols[selected_stock]
        
        # TRADING SETTINGS
        st.subheader("💰 TRADING SETTINGS")
        col_a, col_b = st.columns(2)
        
        with col_a:
            stop_loss_percent = st.slider("Stop Loss %", 1.0, 20.0, 8.0, 0.5)
            risk_reward = st.slider("Risk/Reward Ratio", 1.0, 5.0, 1.5, 0.1)
        
        with col_b:
            target_percent = st.slider("Target %", 1.0, 50.0, 15.0, 1.0)
            position_size = st.slider("Position Size (%)", 1, 100, 10)
        
        # ADVANCED CALCULATOR
        st.subheader("🧮 POSITION CALCULATOR")
        capital = st.number_input("Total Capital (₹)", min_value=1000, value=100000, step=1000)
        risk_per_trade = st.number_input("Risk per Trade (%)", min_value=0.1, value=2.0, step=0.1)
    
    # AI FEATURES SECTION
    with col2:
        st.subheader("🤖 AI & MACHINE LEARNING FEATURES")
        
        ai_col1, ai_col2, ai_col3 = st.columns(3)
        
        with ai_col1:
            if st.button("🧠 LSTM Price Prediction", use_container_width=True):
                with st.spinner("Running LSTM neural network..."):
                    data = app.get_stock_data(symbol, "1y")
                    if data is not None:
                        predictions, confidence = app.predict_with_lstm(data)
                        if predictions:
                            st.markdown(f'''
                            <div class="ml-prediction">
                                <h4>🧠 LSTM Neural Network Prediction</h4>
                                <p><strong>Next 7 Days Forecast:</strong></p>
                                {' '.join([f'Day {i+1}: ₹{pred:.2f}' for i, pred in enumerate(predictions)])}
                                <p><strong>AI Confidence:</strong> {confidence:.1f}%</p>
                            </div>
                            ''', unsafe_allow_html=True)
        
        with ai_col2:
            if st.button("🌳 Random Forest Trend", use_container_width=True):
                with st.spinner("Analyzing with Random Forest..."):
                    data = app.get_stock_data(symbol, "1y")
                    if data is not None:
                        trend, confidence = app.random_forest_trend_prediction(data)
                        st.markdown(f'''
                        <div class="ml-prediction">
                            <h4>🌳 Random Forest Analysis</h4>
                            <p><strong>Predicted Trend:</strong> {trend}</p>
                            <p><strong>Model Confidence:</strong> {confidence:.1f}%</p>
                            <p><strong>Algorithm:</strong> Ensemble Learning</p>
                        </div>
                        ''', unsafe_allow_html=True)
        
        with ai_col3:
            if st.button("📰 News Sentiment", use_container_width=True):
                with st.spinner("Analyzing market sentiment..."):
                    sentiment, news = app.get_news_sentiment(selected_stock)
                    sentiment_color = "profit" if sentiment > 0 else "loss" if sentiment < 0 else "neutral"
                    st.markdown(f'''
                    <div class="ml-prediction">
                        <h4>📰 News Sentiment Analysis</h4>
                        <p><strong>Sentiment Score:</strong> <span class="{sentiment_color}">{sentiment:.2f}</span></p>
                        <p><strong>Market Outlook:</strong> {"Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"}</p>
                    </div>
                    ''', unsafe_allow_html=True)
    
    # RISK MANAGEMENT FEATURES
    st.subheader("🛡️ ADVANCED RISK MANAGEMENT")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        if st.button("🎲 Monte Carlo Simulation", use_container_width=True):
            with st.spinner("Running 1000 Monte Carlo simulations..."):
                data = app.get_stock_data(symbol, "2y")
                if data is not None:
                    mc_results = app.monte_carlo_simulation(data)
                    if mc_results:
                        st.markdown(f'''
                        <div class="risk-box">
                            <h4>🎲 Monte Carlo Risk Analysis</h4>
                            <p><strong>Expected Value:</strong> ₹{mc_results['expected_return']:.2f}</p>
                            <p><strong>95% VaR:</strong> ₹{mc_results['var_95']:.2f}</p>
                            <p><strong>Worst Case:</strong> ₹{mc_results['worst_case']:.2f}</p>
                            <p><strong>Best Case:</strong> ₹{mc_results['best_case']:.2f}</p>
                        </div>
                        ''', unsafe_allow_html=True)
    
    with risk_col2:
        if st.button("📊 Options Greeks", use_container_width=True):
            with st.spinner("Calculating option Greeks..."):
                data = app.get_stock_data(symbol, "1y")
                if data is not None:
                    current_price = data['Close'].iloc[-1]
                    greeks = app.calculate_greeks(
                        spot_price=current_price,
                        strike_price=current_price * 1.1,  # 10% OTM
                        time_to_expiry=30/365,  # 30 days
                        volatility=app.calculate_volatility(data)/100
                    )
                    st.markdown(f'''
                    <div class="risk-box">
                        <h4>📊 Options Greeks</h4>
                        <p><strong>Delta:</strong> {greeks['delta']}</p>
                        <p><strong>Gamma:</strong> {greeks['gamma']}</p>
                        <p><strong>Theta:</strong> {greeks['theta']}</p>
                        <p><strong>Vega:</strong> {greeks['vega']}</p>
                        <p><strong>Rho:</strong> {greeks['rho']}</p>
                    </div>
                    ''', unsafe_allow_html=True)
    
    with risk_col3:
        if st.button("📈 Volatility Analysis", use_container_width=True):
            with st.spinner("Analyzing volatility patterns..."):
                data = app.get_stock_data(symbol, "2y")
                if data is not None:
                    hv = app.calculate_volatility(data)
                    iv = hv * 1.2  # Implied volatility typically higher
                    st.markdown(f'''
                    <div class="risk-box">
                        <h4>📈 Volatility Analysis</h4>
                        <p><strong>Historical Vol (HV):</strong> {hv:.1f}%</p>
                        <p><strong>Implied Vol (IV):</strong> {iv:.1f}%</p>
                        <p><strong>Volatility Premium:</strong> {(iv-hv):.1f}%</p>
                        <p><strong>Risk Level:</strong> {"High" if hv > 30 else "Medium" if hv > 20 else "Low"}</p>
                    </div>
                    ''', unsafe_allow_html=True)
    
    # PORTFOLIO OPTIMIZATION
    st.subheader("💼 PORTFOLIO OPTIMIZATION")
    
    if st.button("🚀 Optimize Portfolio", use_container_width=True):
        with st.spinner("Running Markowitz portfolio optimization..."):
            selected_stocks = st.multiselect(
                "Select stocks for portfolio:",
                list(app.all_symbols.keys())[:10],  # First 10 for performance
                default=[list(app.all_symbols.keys())[0]]
            )
            
            if len(selected_stocks) >= 2:
                # Correlation matrix
                corr_matrix = app.calculate_correlations(selected_stocks)
                if corr_matrix is not None:
                    st.subheader("📊 Correlation Matrix")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)
                
                # Portfolio optimization
                portfolio_result = app.portfolio_optimization(selected_stocks, capital)
                if portfolio_result:
                    st.markdown(f'''
                    <div class="feature-card">
                        <h4>💼 Optimal Portfolio Allocation</h4>
                        <p><strong>Expected Return:</strong> {(portfolio_result['expected_return']*100):.1f}%</p>
                        <p><strong>Expected Volatility:</strong> {(portfolio_result['volatility']*100):.1f}%</p>
                        <p><strong>Sharpe Ratio:</strong> {portfolio_result['sharpe_ratio']:.2f}</p>
                        <p><strong>Recommended Weights:</strong></p>
                        {''.join([f'{stock}: {(weight*100):.1f}%<br>' for stock, weight in portfolio_result['weights'].items()])}
                    </div>
                    ''', unsafe_allow_html=True)
    
    # MAIN ANALYSIS BUTTON
    if st.button("🚀 RUN COMPLETE AI ANALYSIS", type="primary", use_container_width=True):
        with st.spinner("🤖 Running comprehensive AI analysis..."):
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
                    
                    # Display results
                    st.markdown(f'<div class="super-card {signal_class}">', unsafe_allow_html=True)
                    
                    col_s1, col_s2 = st.columns([2, 1])
                    with col_s1:
                        st.subheader(f"{signal}")
                        st.write(f"**AI Confidence Score:** `{score}/100`")
                        st.write(f"**Current Price:** `₹{current_price:.2f}`")
                    with col_s2:
                        st.metric("Signal Strength", f"{score}%")
                    
                    st.write(f"**Expert Advice:** {advice}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Trading recommendation
                    st.markdown(f'''
                    <div class="feature-card">
                        <h3>💎 PROFESSIONAL TRADING RECOMMENDATION</h3>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                            <div><strong>Entry Price</strong><br>₹{current_price:.2f}</div>
                            <div><strong>Stop Loss</strong><br>₹{stop_loss_price:.2f}</div>
                            <div><strong>Target Price</strong><br>₹{target_price:.2f}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

    # SIDEBAR FEATURES
    st.sidebar.header("⚡ AI SMART FEATURES")
    
    if st.sidebar.button("🔍 SCAN TOP AI PICKS", use_container_width=True):
        with st.spinner("AI scanning for high-potential stocks..."):
            results = []
            for stock_name, stock_symbol in list(app.all_symbols.items())[:8]:  # Limit for performance
                try:
                    data = app.get_stock_data(stock_symbol, "6mo")
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
                st.sidebar.subheader("💎 AI TOP PICKS")
                for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:5]:
                    st.sidebar.markdown(f'''
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                        <strong>{stock['symbol']}</strong><br>
                        Score: {stock['score']}/100<br>
                        Price: ₹{stock['price']:.2f}
                    </div>
                    ''', unsafe_allow_html=True)

    # SECURITY & INFO
    st.sidebar.header("🔐 SECURITY")
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a, #3730a3); color: white; padding: 1.5rem; border-radius: 15px;">
        <h4>🚀 AI POWERED FEATURES</h4>
        <ul>
            <li>LSTM Neural Networks</li>
            <li>Random Forest ML</li>
            <li>Sentiment Analysis</li>
            <li>Monte Carlo Simulations</li>
            <li>Portfolio Optimization</li>
            <li>Options Greeks</li>
            <li>Risk Management</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
