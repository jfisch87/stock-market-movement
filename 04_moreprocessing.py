import numpy as np
import pandas as pd
import os
import datetime





def process(company, tweets, resamp, ticker):
    #resample
    stock = company.drop(columns=['date', 'time']).copy()
    twit = tweets[['vader_sentiment']].copy() # only care about this column
    
    stock['mean_price'] = stock['price']
    stock = stock.resample(resamp).agg({'vol' : 'sum', 'price' : 'ohlc', 'mean_price' : 'mean'})
    twit = twit.resample(resamp).mean()
    
    # ewm tweet score
    hrs = 18 # amount of time to average
    if resamp == 's':
        span = 3600*hrs #hours in seconds.  
    elif resamp == 'm':
        span = 60*hrs #hours in minutes
    else: span = hrs
    # ewm tweet score
    twit['sent_mean'] = twit.ewm(span).mean() #weighed mean
    twit.drop(columns='vader_sentiment', inplace=True)
    
    # cleaning cols up before merge
    stock.columns = [' '.join(col).strip() for col in stock.columns.values]
    stock.rename(columns={'vol vol' : 'vol', 'mean_price mean_price' : 'mean_price'}, inplace=True)
    
    # Merge
    df = pd.merge(stock, twit, left_index=True, right_index=True, how='left')
    df = remove_closed(df)
    df = clean_frame(df)
    
    # export
    df['ticker'] = ticker
    df.to_csv(f'./datasets/dow_clean/{resamp}/{ticker}.csv', index=True)

def clean_frame(df):
    # Fill NaNs and pct changes
    # forward fill NAs in price
    df['price high'].fillna(method='ffill', inplace=True)
    df['mean_price'].fillna(method='ffill', inplace=True)
    df['vol'].fillna(0, inplace=True)

    intervals = [1, 2, 3, 4, 5, 10, 15, 30, 60]
    for i in intervals:
        df[f'high_px_{i}'] = df['price high'].pct_change(i)
        df[f'mean_px_{i}'] = df['mean_price'].pct_change(i)
    
#     # cleaning up column names
#     df.columns = [' '.join(col).strip() for col in df.columns.values]
#     df.rename(columns={'vol vol' : 'vol', 'mean_price mean_price' : 'mean_price'}, inplace=True)
    
    # filling times with no trades with previous prices
    df['price open'].fillna(method='ffill', inplace=True)
    df['price high'].fillna(method='ffill', inplace=True)
    df['price low'].fillna(method='ffill', inplace=True)
    df['price close'].fillna(method='ffill', inplace=True)
    df['mean_price'].fillna(method='ffill', inplace=True)
    return df
    
def remove_closed(df):
    # Removed closed hours
    # dropping weekends
    #https://stackoverflow.com/questions/37803040/remove-non-business-days-rows-from-pandas-dataframe
    df = df[df.index.dayofweek<5] 
    # dropping Thanksgiving/Christmas
    # https://stackoverflow.com/questions/3240458/how-to-increment-a-datetime-by-one-day
    # https://stackoverflow.com/questions/41513324/python-pandas-drop-rows-of-a-timeserie-based-on-time-range
    thanksgiving = pd.to_datetime('2019-11-28')
    christmas = pd.to_datetime('2019-12-25')
    tgdrop = pd.date_range(thanksgiving, thanksgiving+datetime.timedelta(days=1), freq='ms')
    chrismasdrop = pd.date_range(christmas, christmas+datetime.timedelta(days=1), freq='ms')
    df = df[~df.index.isin(tgdrop)]
    df = df[~df.index.isin(chrismasdrop)]
    
    # Dropping hours not between 9:30am and 4pm
    df.index = df.index.to_timestamp()
    df = df.between_time('9:30', '16:00')
    return df

def get_data(directory, ind):
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    all_data = {}
    for file in files:
        df = pd.read_csv(directory + file)
        df[ind] = pd.to_datetime(df[ind], 
                                         format='%Y-%m-%d %H:%M:%S')
        df.set_index(ind, inplace=True)
        df.index = pd.DatetimeIndex(df.index).to_period('ms')
        df.sort_index(inplace=True)
        try:
            ticker = df['ticker'][0]
        except:
            ticker=df['company'][0]
        all_data[ticker] = df
#         print(f'loaded {ticker} from {file}')
#         print(f'rows {df.shape[0]} imported')
    return all_data


stocks = get_data('./datasets/russia site/cleaned', 'date_time')
twits = get_data('./datasets/twitters/withsent/', 'date')

counter = 0
for company in stocks.keys():
    process(stocks[company], twits[company], 's', company)
    process(stocks[company], twits[company], 'min', company)
    counter += 1
    print(f'finished processing {company}. {counter}/{len(stocks.keys())}')
