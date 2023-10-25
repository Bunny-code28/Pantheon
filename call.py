import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from scipy.stats import norm
import numpy as np
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def get_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=86cb92c4ecd7421f824696ad49afab58"
    response = requests.get(url)
    news_data = response.json()

    if news_data["status"] == "ok":
        return [article["title"] for article in news_data["articles"]]
    else:
        print("Error fetching news data.")
        return []
    

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    textblob_score = TextBlob(text).sentiment.polarity
    return (sentiment_score + textblob_score) / 2

def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return delta, gamma, theta, vega

def get_greeks_data(ticker):
    option_type = 'call'
    strike_price = 100
    expiration_time = 30 / 365
    risk_free_rate = 0.01
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period="1y")
    greeks_data = []
    for date, row in hist_data.iterrows():
        S = row['Close']
        sigma = hist_data['Close'].pct_change().std() * np.sqrt(252)
        delta, gamma, theta, vega = calculate_greeks(option_type, S, strike_price, expiration_time, risk_free_rate, sigma)
        greeks_data.append([date, delta, gamma, theta, vega])
    greeks_df = pd.DataFrame(greeks_data, columns=['Date', 'Delta', 'Gamma', 'Theta', 'Vega'])
    greeks_df.set_index('Date', inplace=True)
    return greeks_df

def preprocess_data(stock_data, greeks_data):
    combined_data = pd.concat([stock_data, greeks_data], axis=1).dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)
    X, y = [], []
    for i in range(60, len(scaled_data) - 1):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i + 1, 0]) # 'Close' price is the first column
    return np.array(X), np.array(y), scaler

def build_and_train_model(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=64, epochs=100)
    return model

def main():
    ticker = input("Please enter the stock ticker: ").strip().upper()
    stock_data = yf.Ticker(ticker).history(period="1y")
    news_text = get_news(ticker)
    sentiment_score = analyze_sentiment(news_text)
    stock_data['Sentiment'] = sentiment_score
    greeks_data = get_greeks_data(ticker)
    X, y, scaler = preprocess_data(stock_data[['Close', 'Sentiment']], greeks_data)
    model = build_and_train_model(X, y)
    # Add prediction and further analysis

if __name__ == "__main__":
    main()
