import streamlit as st
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit import session_state as state
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
    "XRPUSDT", "DOTUSDT", "UNIUSDT", "BCHUSDT", "LTCUSDT",
    "LINKUSDT", "VETUSDT", "XLMUSDT", "ETCUSDT", "THETAUSDT",
    "FILUSDT", "TRXUSDT", "XMRUSDT", "EOSUSDT", "AAVEUSDT"
]

def get_crypto_news(symbol, days=7):
    """
    Fetch recent news for a given cryptocurrency symbol.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "apiKey": "15e21bc9efa94401ac1365a311a20b23"  
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        return []

def analyze_sentiment(text):
    """
    Perform sentiment analysis on the given text.
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def crypto_news_sentiment(symbol):
    """
    Get recent news and perform sentiment analysis for a cryptocurrency.
    """
    news = get_crypto_news(symbol)
    results = []
    
    for article in news:
        title = article["title"]
        description = article["description"]
        sentiment = analyze_sentiment(title + " " + description)
        results.append({
            "title": title,
            "description": description,
            "sentiment": sentiment,
            "url": article["url"],
            "publishedAt": article["publishedAt"]
        })
    
    return results

def get_crypto_price(symbol):
    """
    Fetch current price for a given cryptocurrency symbol in different currencies.
    """
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        usd_price = float(data['price'])
        
        # Get exchange rates
        exchange_rates = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()['rates']
        
        return {
            'USD': usd_price,
            'EUR': usd_price * exchange_rates['EUR'],
            'GBP': usd_price * exchange_rates['GBP'],
            'JPY': usd_price * exchange_rates['JPY'],
            'CNY': usd_price * exchange_rates['CNY'],
            'INR': usd_price * exchange_rates['INR']  # Added Indian Rupee
        }
    else:
        return None

def get_market_status(positive_percentage, negative_percentage):
    """
    Determine market status based on sentiment percentages.
    """
    if positive_percentage > negative_percentage + 10:
        return "Bullish"
    elif negative_percentage > positive_percentage + 10:
        return "Bearish"
    else:
        return "Neutral"

def get_crypto_details(symbol):
    """
    Fetch additional details for a cryptocurrency.
    """
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "symbol": data["symbol"],
            "priceChange": float(data["priceChange"]),
            "priceChangePercent": float(data["priceChangePercent"]),
            "volume": float(data["volume"]),
            "highPrice": float(data["highPrice"]),
            "lowPrice": float(data["lowPrice"])
        }
    else:
        return None

def get_historical_prices(symbol, days=31):
    """
    Fetch historical prices for a given cryptocurrency symbol.
    """
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&startTime={start_time}&endTime={end_time}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices = [float(item[4]) for item in data]  # Close price
        dates = [datetime.fromtimestamp(item[0] / 1000) for item in data]
        return dates, prices
    else:
        return None, None

def generate_word_cloud(news_sentiment):
    """
    Generate a word cloud from news titles and descriptions.
    """
    text = " ".join([item['title'] + " " + item['description'] for item in news_sentiment])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def calculate_technical_indicators(prices):
    df = pd.DataFrame(prices, columns=['close'])
    
    # SMA
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()
    
    # EMA
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def predict_price(prices, days=30):
    if len(prices) < days + 1:
        return None  # Not enough data to make a prediction

    df = pd.DataFrame(prices, columns=['close'])
    df['prediction'] = df['close'].shift(-days)
    
    X = np.array(df.drop('prediction', axis=1))[:-days]
    y = np.array(df['prediction'].dropna())
    
    if len(X) < 2 or len(y) < 2:
        return None  # Not enough data for train-test split
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    last_days = np.array(df.drop('prediction', axis=1))[-days:]
    prediction = model.predict(last_days)
    
    return prediction[-1]  # Return only the last predicted value

def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Cryptocurrency News and Analysis")

    # Initialize session state
    if 'crypto_symbol' not in state:
        state.crypto_symbol = CRYPTO_SYMBOLS[0]
    if 'compare_symbol' not in state:
        state.compare_symbol = CRYPTO_SYMBOLS[1]
    if 'comparison_data' not in state:
        state.comparison_data = None

    # Sidebar
    st.sidebar.header("Settings")
    state.crypto_symbol = st.sidebar.selectbox("Select cryptocurrency:", CRYPTO_SYMBOLS, index=CRYPTO_SYMBOLS.index(state.crypto_symbol))
    
    if st.sidebar.button("Analyze", key="analyze_button"):
        try:
            news_sentiment = crypto_news_sentiment(state.crypto_symbol)
            crypto_details = get_crypto_details(state.crypto_symbol)
            
            # Main content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Fetch and display current price in different currencies
                current_prices = get_crypto_price(state.crypto_symbol)
                if current_prices:
                    st.subheader("💰 Current Price")
                    col3, col4 = st.columns(2)
                    items = list(current_prices.items())
                    mid = len(items) // 2
                    
                    with col3:
                        for currency, price in items[:mid]:
                            st.write(f"{currency}: {price:.2f}")
                    
                    with col4:
                        for currency, price in items[mid:]:
                            st.write(f"{currency}: {price:.2f}")
                else:
                    st.warning("No current prices available.")
                # Display market status
                total_articles = len(news_sentiment)
                if total_articles > 0:
                    positive_count = sum(1 for item in news_sentiment if item['sentiment'] == "Positive")
                    negative_count = sum(1 for item in news_sentiment if item['sentiment'] == "Negative")
                    neutral_count = sum(1 for item in news_sentiment if item['sentiment'] == "Neutral")
                    
                    positive_percentage = (positive_count / total_articles) * 100
                    negative_percentage = (negative_count / total_articles) * 100
                    neutral_percentage = (neutral_count / total_articles) * 100
                    
                    market_status = get_market_status(positive_percentage, negative_percentage)
                    st.subheader("🏛️ Market Status")
                    st.subheader(market_status, divider='blue')
                    
                    # Calculate sentiment score
                    sentiment_score = (positive_count - negative_count) / total_articles
                else:
                    st.warning("No news articles found for sentiment analysis.")
                
                # Fetch historical prices and calculate technical indicators
                dates, prices = get_historical_prices(state.crypto_symbol)
                if dates and prices:
                    df = calculate_technical_indicators(prices)
                    
                    # Create price chart with technical indicators
                    st.subheader("📊 Price Chart with Technical Indicators")
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.1, row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Scatter(x=dates, y=df['close'], mode='lines', name='Close Price'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=dates, y=df['SMA20'], mode='lines', name='SMA20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=dates, y=df['SMA50'], mode='lines', name='SMA50'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=dates, y=df['EMA20'], mode='lines', name='EMA20'), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=dates, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
                    
                    fig.update_layout(height=600, title_text="Price and Technical Indicators")
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
                    fig.update_yaxes(title_text="RSI", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No historical prices available.")
                
                # Generate and display word cloud
                st.subheader("🔤 News Word Cloud")
                wordcloud_plt = generate_word_cloud(news_sentiment)
                st.pyplot(wordcloud_plt)
                
                st.header(f"📰 Recent News for {state.crypto_symbol}")
                
                # Create a scrollable container for news
                news_container = st.container()
                with news_container:
                    for item in news_sentiment:
                        with st.expander(item['title']):
                            st.write(item['description'])
                            st.write(f"Sentiment: {item['sentiment']}")
                            recommendation = "Buy" if item['sentiment'] == "Positive" else "Sell" if item['sentiment'] == "Negative" else "Hold"
                            st.write(f"Recommendation: {recommendation}")
                            st.write(f"Published at: {item['publishedAt']}")
                            st.write(f"[Read more]({item['url']})")

            with col2:
                if total_articles > 0:
                    st.header("📊 Sentiment Analysis")
                    
                    fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative', 'Neutral'],
                                                 values=[positive_percentage, negative_percentage, neutral_percentage],
                                                 marker_colors=['#00FF00', '#FF0000', '#FFFF00'])])
                    fig.update_layout(title_text="Sentiment Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write(f"Positive: {positive_percentage:.2f}%")
                    st.write(f"Negative: {negative_percentage:.2f}%")
                    st.write(f"Neutral: {neutral_percentage:.2f}%")
                else:
                    st.warning("Insufficient data for sentiment analysis.")
                
                # Add trading volume chart
                if crypto_details:
                    st.subheader("📊 Trading Volume")
                    volume = crypto_details['volume']
                    fig_volume = go.Figure(data=[go.Bar(x=[state.crypto_symbol], y=[volume])])
                    fig_volume.update_layout(title_text="24h Trading Volume", yaxis_title="Volume")
                    st.plotly_chart(fig_volume, use_container_width=True)
                else:
                    st.warning("No trading volume data available.")
                
                # Add price change percentage
                if crypto_details:
                    st.subheader("📈 24h Price Change")
                    price_change = crypto_details['priceChangePercent']
                    color = 'green' if price_change >= 0 else 'red'
                    st.markdown(f"<h1 style='text-align: center; color: {color};'>{price_change:.2f}%</h1>", unsafe_allow_html=True)
                else:
                    st.warning("No price change data available.")

        except Exception as e:
            st.warning("Please try again later or select a different cryptocurrency.")

    # Multi-Crypto Comparison
    st.sidebar.header("Crypto Comparison")
    compare_symbols = st.sidebar.multiselect("Select cryptocurrencies to compare", CRYPTO_SYMBOLS, default=[CRYPTO_SYMBOLS[0], CRYPTO_SYMBOLS[1]])
    
    if st.sidebar.button("Compare"):
        st.header("🔍 Cryptocurrency Comparison")
        comparison_data = []
        for symbol in compare_symbols:
            price = get_crypto_price(symbol)['USD']
            news = crypto_news_sentiment(symbol)
            positive_sentiment = sum(1 for item in news if item['sentiment'] == "Positive") / len(news) * 100
            comparison_data.append({"Symbol": symbol, "Price": price, "Positive Sentiment": positive_sentiment})
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Comparison chart
        fig = go.Figure()
        for symbol in compare_symbols:
            dates, prices = get_historical_prices(symbol)
            fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name=symbol))
        
        fig.update_layout(title="Price Comparison", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
