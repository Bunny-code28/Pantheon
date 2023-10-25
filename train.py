import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from fredapi import Fred

def get_fred_data(fred_api_key):
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
