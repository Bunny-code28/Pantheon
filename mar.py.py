import requests
import pandas as pd
import time
import csv
from io import StringIO

def download_option_chain():
    url = "https://www.nseindia.com/option-chain"  # Replace with the actual URL to download CSV
    headers = {"User-Agent": "Mozilla/5.0"}  # Some websites require a user-agent header
    response = requests.get(url, headers=headers)
    csv_content = response.content.decode('utf-8')
    csv_data = StringIO(csv_content)
    df = pd.read_csv(csv_data, skiprows=1)  # Skip the first row as it doesn't contain data
    return df

def evaluate_strike_price(df, buy_strike, sell_strike):
    # Implement your logic for evaluating which strike price is favorable
    # For demonstration, let's find the strike price with the highest difference between Calls_OI and Puts_OI
    df['OI_Difference'] = df['Calls_OI'] - df['Puts_OI']
    favorable_strike = df.loc[df['OI_Difference'].idxmax()]['Strike_Price']
    
    # Calculate the initial cost for a Bull Call Spread for the favorable strike price
    ltp_buy = df.loc[df['Strike_Price'] == favorable_strike, 'Calls_LTP'].values[0]
    ltp_sell = df.loc[df['Strike_Price'] == favorable_strike, 'Puts_LTP'].values[0]
    initial_cost = ltp_buy - ltp_sell
    
    # Minimum capital required
    min_capital = initial_cost * 40  # Assuming lot size of 40
    
    # Check profitability (for demonstration, let's assume profitable if initial_cost < 0)
    is_profitable = "Yes" if initial_cost < 0 else "No"
    
    message = f"The most favorable strike price is: {favorable_strike}. " \
              f"Recommended strategy: Bull Call Spread. " \
              f"Minimum capital required: ₹{min_capital}. " \
              f"Is it profitable? {is_profitable}"
    
    return message

while True:
    # Take user input for the strike prices they have chosen for buying and selling
    buy_strike = float(input("Enter the strike price you have chosen for buying: "))
    sell_strike = float(input("Enter the strike price you have chosen for selling: "))
    
    df = download_option_chain()
    message = evaluate_strike_price(df, buy_strike, sell_strike)
    print(message)
    
    time.sleep(1800)  # Wait for 30 minutes before the next iteration
