import yfinance as yf
from datetime import datetime
import pandas as pd

# Collecting S&P 500 Index data

ticker = "^GSPC"

now_time = datetime.now()
start_time = datetime(now_time.year - 8, now_time.month, now_time.day)

stock_data = yf.download(ticker, start=start_time, end=now_time)

stock_data.to_csv(
    'C:\\Users\\kedar\\Desktop\\Projects\\CIVE 7100 Stock Market Time Series Project\\datasets\\s_and_p_500_index_data.csv')

# Collecting Sector Index Data
indexes = {"Healthcare": "^SP500-35", "Communication Services": "^SP500-50", "Consumer Discretionary": "^SP500-25",
           "Consumer Staples": "^SP500-30", "Energy": "^SP500-1010", "Financials": "^SP500-40",
           "Industrials": "^SP500-20", "Materials": "^SP500-15", "Utilities": "^SP500-55", "Real Estate": "^SP500-60",
           "Information Technology": "^SP500-45"}
df = pd.DataFrame()

for key, value in indexes.items():
    stock_data = yf.download(value, start=start_time, end=now_time)
    stock_data['Sector'] = key
    df = pd.concat([df, stock_data])

df.to_csv(
    'C:\\Users\\kedar\\Desktop\\Projects\\CIVE 7100 Stock Market Time Series Project\\datasets\\sector_index_data.csv')
