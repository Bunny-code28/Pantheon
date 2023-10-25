from bs4 import BeautifulSoup
from datetime import date
from fredapi import Fred
from keras.layers import LSTM, Dense
from keras.models import Sequential
from nsepy import get_history
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def get_news_articles(stock_symbol):
    # URL of Yahoo Finance news for the stock
    url = f"https://finance.yahoo.com/quote/{stock_symbol}/news?p={stock_symbol}"

    # Send HTTP request to the URL
    response = requests.get(url)

    # Parse the response content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the news articles on the page
    news_divs = soup.find_all('div', attrs={'class': 'Ov(h) Pend(44px) Pstart(25px)'})
    
    # Create a list to hold the news articles
    articles = []

    # Loop through the news articles and get the text
    for news in news_divs:
        news_text = news.get_text(separator=' ')
        articles.append(news_text)

    return articles

def calculate_sentiment_score(articles):
    analyzer = SentimentIntensityAnalyzer()

    textblob_scores = []
    vader_scores = []

    for article in articles:
        # Calculate TextBlob sentiment score
        tb_score = TextBlob(article).sentiment.polarity
        textblob_scores.append(tb_score)

        # Calculate VADER sentiment score
        vs_score = analyzer.polarity_scores(article)['compound']
        vader_scores.append(vs_score)

    # Calculate average sentiment scores
    avg_textblob_score = sum(textblob_scores) / len(textblob_scores)
    avg_vader_score = sum(vader_scores) / len(vader_scores)

    return avg_textblob_score, avg_vader_score

    return sentiment_score

def get_fred_data(fred_api_key):
  # Use the fredapi library to get data from the FRED API
  fred = Fred(api_key=fred_api_key)
  interest_rates = fred.get_series('api key')
  inflation_rates = fred.get_series('api key')
  gdp_growth_rate = fred.get_series('api key')
  return interest_rates, inflation_rates, gdp_growth_rate

def fetch_nse_data(stock_symbol):
  # Set the start date for fetching historical data
  start_date = date(1990, 1, 1)
  end_date = date.today()

    # Fetch historical stock data from NSE using NSEpy
    hist = get_history(symbol=stock_symbol, start=start_date, end=end_date)

    return hist

def preprocess_data(hist):
  # Get only the 'Close' column from the historical stock data
  data = hist['Close'].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create training and testing datasets
    x_data, y_data = [], []
    for i in range(60, len(data)):
        x_data.append(data[i-60:i, 0])
        y_data.append(data[i, 0])
    
    x_data, y_data = np.array(x_data), np.array(y_data)
    
    # Reshape the data to the format expected by the LSTM model
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    return x_data, y_data, scaler

# Prompt user for stock symbol
stock_symbol = input('Enter stock symbol: ')

# Use longer period of historical data
period = 'max'

# Fetch historical stock data using yfinance
stock_data = yf.Ticker(stock_symbol)
hist = stock_data.history(period=period)

x_data, y_data, scaler = preprocess_data(hist)

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Get news articles related to the stock symbol
articles = get_news_articles(stock_symbol)

# Calculate sentiment score of the news articles using TextBlob and VADER
sentiment_score = calculate_sentiment_score(articles)

# Get data from FRED API
fred_api_key = 'a88dd0ec71faa8e1b3cf0aef55c83af7'
interest_rates, inflation_rates, gdp_growth_rate = get_fred_data(fred_api_key)

# Make predictions using the trained model
predictions = model.predict(x_test)

# Inverse transform the predicted and actual values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Print the predicted stock prices
print(predictions)
from fredapi import Fred

    # Get the FRED API URL
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={fred_api_key}"

    # Remove the space character from the URL
    url = url.replace(" ", "%20")

    # Get the economic data from the FRED API
    interest_rates = fred.get_series(url)

    return interest_rates


def train_model_on_all_stocks():
  # Get a list of all the available stocks
  stocks = yf.Tickers('SPY').symbols

    # Train the model on each stock
    for stock in stocks:
        model = train_model(stock)

    return model

# Train the model on all the available stocks
model = train_model_on_all_stocks()
