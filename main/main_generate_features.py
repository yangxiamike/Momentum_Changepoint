from finance_data.dmn_data import DMNData
from settings.default import RAW_DATA_FILE
from settings.default import FEATURES_DIR_DEFAULT
import os
import pandas as pd

if __name__ == '__main__':

    data = pd.read_csv(RAW_DATA_FILE, parse_dates = True)
    dmndata = DMNData()
    df_asset = dmndata.make_all_features(data)

    if not os.path.exists(FEATURES_DIR_DEFAULT):
        os.makedirs(FEATURES_DIR_DEFAULT)
        os.makedirs(os.path.join(FEATURES_DIR_DEFAULT, 'single_asset'))
    for ticker, data in df_asset.groupby('ticker'):
        del data['ticker']
        file_path = os.path.join(FEATURES_DIR_DEFAULT, 'single_asset', f'{ticker}.csv')
        data.to_csv(file_path, index = False)
    file_path = os.path.join(FEATURES_DIR_DEFAULT, 'features.csv')
    df_asset.to_csv(file_path, index = False)
