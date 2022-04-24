import os
import glob
import pandas as pd

os.chdir(
    "C:\\Users\\kedar\\Desktop\\Projects\\CIVE 7100 Stock Market Time Series Project\\datasets\\individual_stock\\")

# PART 1: Creating a dataframe which contains all stock data
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

# export to csv
path = "C:\\Users\\kedar\\Desktop\\Projects\\CIVE 7100 Stock Market Time Series Project\\datasets\\"
combined_csv.to_csv(path + "S_and_P500_data.csv", index=False, encoding='utf-8-sig')
path = "C:\\Users\\kedar\\Desktop\\Projects\\CIVE 7100 Stock Market Time Series Project\\datasets\\individual_stock\\"

# PART 2: Creating a dataframe with only closing price of stock data
df = pd.DataFrame()
for file in all_filenames:
    stock_df = pd.read_csv(path + file)  # Read each csv
    stock_df.index = pd.DatetimeIndex(stock_df.Date)  # Definition of "Date" as index
    name = stock_df['Name'].iloc[0]
    df[name] = stock_df['Adj Close']
df = df.sort_index(axis=1)
path2 = "C:\\Users\\kedar\\Desktop\\Projects\\CIVE 7100 Stock Market Time Series Project\\datasets\\"
df.to_csv(path2 + 's_and_p_500_closing.csv')
