import argparse
import os
import pandas as pd
from typing import List

# python main/main_get_prod_data.py --input_dir_path data/tmp_data --output_dir_path data/prod_data -c 'date' 'open'

def main(
    input_dir_path: str, output_dir_path: str, 
    use_cols: List, col_names: List
):
    files = os.listdir(input_dir_path)
    files = [file for file in files if '.csv' in file]
    input_paths = [f'{input_dir_path}/{file}' for file in files]
    output_paths = [f'{output_dir_path}/{file}' for file in files]
    
    for input_path, output_path in zip(input_paths, output_paths):
        data = pd.read_csv(input_path, parse_dates = True)
        data = data[use_cols]
        data.columns = col_names
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values('date', inplace = True)
        data = data[data['price'] > 1e-8]
        data.to_csv(output_path, index = False)

if __name__ == '__main__':

    def get_args():
        parser = argparse.ArgumentParser(description="Run clean raw data for prod data")
        parser.add_argument('-i', '--input_dir_path', type = str, required = True, help = 'Input directory path')
        parser.add_argument('-o', '--output_dir_path', type = str, required = True, help = 'Output directory path')
        parser.add_argument('-c', '--use_columns', type = str, default = ['date', 'price'], 
                            nargs = '+', help = 'Use columns for price data')
        parser.add_argument('-n', '--names', type = str, default = ['date', 'price'],
                            nargs = '+', help = 'Column name for price and date')

        arg_res = parser.parse_args()
        return (arg_res.input_dir_path, arg_res.output_dir_path, arg_res.use_columns, arg_res.names)
    
    main(*get_args())

