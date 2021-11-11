from settings.default import COMMODITIES_TICKERS
from settings.default import RAW_DATA_FOLDER
from typing import List
from typing import Tuple
from tqdm import tqdm
import os
import pandas as pd
import yfinance as yf

def pull_yahoo_sample_data(ticker: str) -> Tuple[pd.DataFrame]:
    data_source = yf.Ticker(ticker)
    mkt_data = data_source.history(period="max")[["Close"]].copy()
    mkt_data.reset_index(inplace = True)
    
    name = ticker.split('=')[0]
    info_data = pd.DataFrame.from_dict({'ticker': [name]})
    try:
        info_data['description'] = data_source.info['shortName']
        info_data['description'] = info_data['description'].apply(lambda x: ' '.join(x.split(' ')[:-2]))
    except:
        info_data['description'] = None
    mkt_data = mkt_data.rename(columns={"Close": "price", "Date": "date"})

    start_year = mkt_data['date'][0].year
    info_data['start_year'] = start_year

    return mkt_data, info_data


def pull_yahoo_sample_data_multiple(tickers: List[str]) -> pd.DataFrame:
    error_logs = []
    mkt_full_data = []
    info_full_data = []
    mkt_list_data = []
    
    for ticker in tqdm(tickers, total = len(tickers)):
        try:
            name = ticker.split('=')[0]
            mkt_data, info_data = pull_yahoo_sample_data(ticker)

            mkt_list_data.append((name, mkt_data.copy()))
            mkt_data['ticker'] = name
            mkt_full_data.append(mkt_data)
            info_full_data.append(info_data)
        except:
            error_logs.append(ticker)
    mkt_full_data = pd.concat(mkt_full_data)
    info_full_data = pd.concat(info_full_data)

    print('Finish fetching data!!')
    if len(error_logs) > 0:
        print(f'Error Tickers are {",".join(error_logs)}')
    
    return mkt_full_data, info_full_data, mkt_list_data

if __name__ == '__main__':
    if not os.path.exists(RAW_DATA_FOLDER):
        os.makedirs(RAW_DATA_FOLDER)
    
    mkt_data, info_data, mkt_list_data = pull_yahoo_sample_data_multiple(COMMODITIES_TICKERS)
    mkt_file_path = os.path.join(RAW_DATA_FOLDER, 'futures.csv')
    info_file_path = os.path.join(RAW_DATA_FOLDER, 'futures_info.csv')

    mkt_data.to_csv(mkt_file_path, index = False)
    info_data.to_csv(info_file_path, index = False)
    for ticker, data in mkt_list_data:
        mkt_file_path = os.path.join(RAW_DATA_FOLDER, f'{ticker}.csv')
        data.to_csv(mkt_file_path, index = False)

    
