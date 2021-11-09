from finance_data.base_data import BaseData
from moment_strategy.moment_strat import MomentStrat
from moment_strategy.long_strat import LongStrat
from moment_strategy.macd_strat import MACDStrat
from moment_strategy.dmn_strat import DMNStrat
import numpy as np
import pandas as pd
import pytz
from matplotlib import pyplot as plt
import pyfolio as pf

# Init FinanceData and load data
# data_dir = 'data/prod_data'
# Data = FinanceData(is_moving = True, window_size = 21)
# Data.load_from_dir(data_dir)
ret_data = pd.read_csv('data/prod_data/futures.csv')
ret_data.set_index('date', inplace=True)
ret_data.index = pd.to_datetime(ret_data.index)
price_data = np.cumprod(1 + ret_data)

# Calculate momentum and get PnL
# print('Momentum: \n')
# Strat = MomentStrat(price_data)
# signal = Strat.cal_signal(0.0)
# pnl, pnl_cum = Strat.get_pnl(volatility_scaling=True)
# Strat.print_stats_log()
# Strat.plot_stats(is_save = True, save_path='data/output_data/long_moment')
# print('=' * 100)
# # # Calcuate PnL Curve

# print('Long: \n')
# Strat = LongStrat(price_data)
# signal = Strat.cal_signal()
# pnl, pnl_cum = Strat.get_pnl(volatility_scaling=True)
# Strat.print_stats_log()
# Strat.plot_stats(is_save = True, save_path='data/output_data/long')
# print('=' * 100)

# print('MACD: \n')
# Strat = MACDStrat(price_data, [(8, 24), (16, 48), (32, 96)])
# signal = Strat.cal_signal()
# pnl, pnl_cum = Strat.get_pnl(volatility_scaling=True)
# Strat.print_stats_log()
# Strat.plot_stats(is_save = True, save_path = 'data/output_data/macd')
# print('=' * 100)

Strat = DMNStrat(price_data)
Strat.make_features()