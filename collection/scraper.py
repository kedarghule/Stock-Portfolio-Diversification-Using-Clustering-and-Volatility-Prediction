from datetime import datetime
from concurrent import futures
import pandas as pd
import pandas_datareader.data as web

path = 'C:\\Users\\kedar\\Desktop\\Projects\\CIVE 7100 Stock Market Time Series Project\\datasets\\individual_stock\\'
master_path = 'C:\\Users\\kedar\\Desktop\\Projects\\CIVE 7100 Stock Market Time Series Project\\datasets\\'
dataframe_list = []


def download_stock(stock):
    """Download and return individual stock data"""
    try:
        print(stock)
        stock_df = web.DataReader(stock, 'yahoo', start_time, now_time)
        stock_df['Name'] = stock
        output_name = stock + '.csv'
        stock_df.to_csv(path+output_name)
        dataframe_list.append(output_name)
    except:
        bad_names.append(stock)
        print('bad: %s' % (stock))


def get_ticker_details():
    """Function to return a list of stocks currently in the S&P 500 Index"""
    ticker_list = []
    sector_list = []
    sub_sector_list = []
    security_list = []
    symbols_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]
    for ticker in symbols_table['Symbol']:
        ticker_list.append(ticker.strip())

    for name in symbols_table['Security']:
        security_list.append(name.strip())

    for name in symbols_table['GICS Sector']:
        sector_list.append(name.strip())

    for name in symbols_table['GICS Sub-Industry']:
        sub_sector_list.append(name.strip())

    return ticker_list, security_list, sector_list, sub_sector_list


if __name__ == '__main__':

    # We get 8 years of stock data from the present day. Here we define the current time and the start time
    now_time = datetime.now()
    start_time = datetime(now_time.year - 8, now_time.month, now_time.day)

    s_and_p, company_list, c_sector_list, c_sub_sector_list = get_ticker_details()

    # Create Constituents Dataset
    constituents_dict = {'Symbol': s_and_p, 'Name': company_list, 'Sector': c_sector_list,
                         'Sub-Sector': c_sub_sector_list}
    constituents_df = pd.DataFrame(constituents_dict)
    constituents_df.to_csv(master_path+'constituents.csv', index=False)

    # Getting Individual Stock Data
    bad_names = []  # to keep track of failed queries

    # here we use the concurrent.futures module's ThreadPoolExecutor to speed up the downloads buy doing them
    # in parallel as opposed to sequentially

    # set the maximum thread number
    max_workers = 50

    workers = min(max_workers, len(s_and_p))  # in case a smaller number of stocks than threads was passed in
    with futures.ThreadPoolExecutor(workers) as executor:
        res = executor.map(download_stock, s_and_p)

    bad_names = [name.replace('.', '-') for name in bad_names]
    with futures.ThreadPoolExecutor(workers) as executor:
        res = executor.map(download_stock, bad_names)
    # Save failed queries to a text file to retry
    # if len(bad_names) > 0:
    #     with open('failed_queries.txt', 'w') as outfile:
    #         for name in bad_names:
    #             outfile.write(name + '\n')

    # timing:
    finish_time = datetime.now()
    duration = finish_time - now_time
    minutes, seconds = divmod(duration.seconds, 60)
    print('getSandP_threaded.py')
    print(f'The threaded script took {minutes} minutes and {seconds} seconds to run.')
