import pandas as pd
import os


class BaseData(object):
    def __init__(self, data = None, is_moving = True, window_size = 20):
        """
        data: (pd.DataFrame), date x asset_names 
        """
        self.data = data
        self.ret_data = None
        self.is_moving = is_moving
        self.is_window_size = window_size
        self.num_samples = None

        if self.data:
            self._set_attribute()
            self._cal_return()
    
    def load_from_dir(self, dir_path):
        """
        Require all files saved in csv format
        Data format: date x price
        """
        data_all = pd.DataFrame()
        files = os.listdir(dir_path)
        files = [file for file in files if '.xlsx' in file]
        for file in files:
            path = os.path.join(dir_path, file)
            title = file.split('.')[0]
            data = pd.read_excel(path)
            data['asset'] = title
            data['date'] = pd.to_datetime(data['date'])
            data_all = data_all.append(data)
        data_all = pd.pivot_table(data_all, values = 'price', index = 'date', columns = 'asset')
        self.data = data_all
        self._set_attribute()
        self._cal_return()

    def get_asset_list(self):
        return self.asset_names
    
    def get_date_list(self):
        return self.dates
    
    def get_price(self, name = None):
        if name is None:
            return self.data
        else:
            return self.data[name]
    
    def get_return(self, name = None):
        if name is None:
            return self.ret_data
        else:
            return self.ret_data[name]

    def _cal_return(self):
        """
        Calculate return based on each asset, pct change/log ret etc.
        """
        self.ret_data = self.data.apply(lambda x: x.pct_change())
    
    def _set_attribute(self):
        self.asset_names = self.data.columns
        self.dates = self.data.index
        self.min_date = self.dates[0]
        self.max_date = self.dates[-1]
        
    def __getitem__(self, idx):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented



    