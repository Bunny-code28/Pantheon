from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import cryptocompare
import requests
from pycoingecko import CoinGeckoAPI
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from newsapi import NewsApiClient
import numpy as np

# Initialize News API
newsapi = NewsApiClient(api_key='86cb92c4ecd7421f824696ad49afab58')

# Initialize CoinGecko API
cg = CoinGeckoAPI()

# Function to fetch data from CoinMarketCap
def fetch_coinmarketcap_data(symbol):
    url = f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?symbol={symbol}"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '13fbfe19-d236-4684-89a0-4008793ec458',
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return pd.DataFrame(data['data'][0]['quote'])

# Function to fetch data from CoinGecko
def fetch_coingecko_data(symbol):
    data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency='usd', days=30)
    return pd.DataFrame(data['prices'], columns=['time', 'price'])

# Function to fetch cryptocurrency data from Cryptocompare
def fetch_crypto_data(crypto, limit=100):
    return pd.DataFrame(cryptocompare.get_historical_price_day(crypto, currency='USD', limit=limit))

# Function to fetch news and perform sentiment analysis
def fetch_news_sentiment(crypto):
    news = newsapi.get_everything(q=crypto, language='en', sort_by='relevancy', page_size=10)
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(article['title'])['compound'] for article in news['articles']]
    return np.mean(scores)

# Get user input for cryptocurrency symbols
crypto_symbols = input("Enter the cryptocurrency symbols separated by commas (e.g., BTC,ETH): ").split(',')

# Initialize an empty DataFrame to store combined data
combined_df = pd.DataFrame()

for symbol in crypto_symbols:
    # Fetch data from different sources
    df_cryptocompare = fetch_crypto_data(symbol.strip())
    df_coinmarketcap = fetch_coinmarketcap_data(symbol.strip())
    df_coingecko = fetch_coingecko_data(symbol.lower().strip())
    
    # Combine and average the data for more accuracy (you can also use weighted average)
    df_combined = pd.concat([df_cryptocompare, df_coinmarketcap, df_coingecko]).groupby(level=0).mean()df_coingecko]).groupby(level=0).mean()df['price'].iloc[-1]
    print(f"Last closing price for {symbol.strip()}: ${last_closing_price}")

    comments = fetch_investor_comments()
    investor_sentiment = analyze_sentiment(comments)

    df['moving_avg_7d'] = df['close'].rolling(window=7).mean() if 'close' in df.columns else df['price'].rolling(window=7).mean()
    df['news_sentiment'] = fetch_news_sentiment(symbol.strip())
    df['investor_sentiment'] = investor_sentiment
    df = df.dropna()
    print(f"Data after dropna for {symbol}:")  # Debugging line
    print(df.head())  # Debugging line

    df['crypto'] = symbol.strip()
    combined_df = pd.concat([combined_df, df])

print("Combined data:")  # Debugging line
print(combined_df.head())  # Debugging line
# Feature and target variables
X = combined_df[['crypto', 'volumeto', 'moving_avg_7d', 'news_sentiment', 'investor_sentiment']]
y = combined_df['close'] if 'close' in combined_df.columns else combined_df['price']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_imputed, y_train)

# Make predictions
y_pred = model.predict(X_test_imputed)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Square Error: {rmse}')
print(f"Last closing price for {symbol.strip()}: ${last_closing_price}")
