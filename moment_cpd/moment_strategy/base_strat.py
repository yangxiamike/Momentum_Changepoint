from abc import ABCMeta
from abc import abstractmethod
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import empyrical
import os
import pyfolio as pf
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import cm

VOL_LOOKBACK = 60
VOL_TARGET = 0.15

class BaseStrat(metaclass=ABCMeta):
    def __init__(self, price_data):
        """
        Parameter:
            price_data: (DataFrame), date x asset_name
            ret_data: (DataFrame), date x asset_name
            window_sigma: (int), window for calculating sigma 
                                 on return for normalizatoin
            sigma_tgt: (float), target sigma for PnL comparison
        """
        self.price_data = price_data
        self.price_data.fillna(method = 'ffill', inplace = True)
        self.signal = None
    
    @abstractmethod
    def cal_signal(self, w: float) -> pd.DataFrame:
        raise NotImplemented
    
    @staticmethod
    def calc_returns(srs: pd.DataFrame, day_offset: int = 1) -> pd.DataFrame:
        """
        for each element of a pandas time-series srs,
        calculates the returns over the past number of days
        specified by offset
        Args:
            srs (pd.DataFrame): time-series of prices
            day_offset (int, optional): number of days to calculate returns over. Defaults to 1.
        Returns:
            pd.Series: pd.DataFrame
        """
        returns = srs / srs.shift(day_offset) - 1.0
        return returns
    
    @staticmethod
    def calc_daily_vol(daily_returns):
        return (
            daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
            .std()
            .fillna(method="bfill")
        )
    
    @staticmethod
    def calc_vol_scaled_returns(daily_returns, daily_vol=pd.Series(None)):
        """calculates volatility scaled returns for annualised VOL_TARGET of 15%
        with input of pandas series daily_returns"""
        if not len(daily_vol):
            daily_vol = BaseStrat.calc_daily_vol(daily_returns)
        annualised_vol = daily_vol * np.sqrt(252)  # annualised
        return daily_returns * VOL_TARGET / annualised_vol.shift(1)
    
    @staticmethod
    def calc_normalised_returns(price, daily_vol, day_offset):
        return (
            BaseStrat.calc_returns(price, day_offset)
            / daily_vol
            / np.sqrt(day_offset)
        )
    
    def get_pnl(self, volatility_scaling=True):
        ## todo: Add transaction cost
        daily_returns = BaseStrat.calc_returns(self.price_data)
        next_day_returns = (
            BaseStrat.calc_vol_scaled_returns(daily_returns).shift(-1)
            if volatility_scaling
            else daily_returns.shift(-1)
        )
        pnl_asset = self.signal * next_day_returns

        self.pnl_asset = pnl_asset
        self.pnl_mean = pnl_asset.mean(axis = 1)
        self.pnl_cum = np.cumprod(1+self.pnl_mean)
        return self.pnl_mean, self.pnl_cum

    def print_stats_log(self):
        pnl = self.pnl_mean.values
        print("Annualized Sharpe Ratio = ", empyrical.sharpe_ratio(pnl, period='daily'))
        print("Annualized Mean Returns = ", empyrical.annual_return(pnl, period='daily'))
        print("Annualized Standard Deviations = ", empyrical.annual_volatility(pnl, period='daily'))
        print("Max Drawdown = ", empyrical.max_drawdown(pnl))
        print("Calmar ratio = ", empyrical.calmar_ratio(pnl, period='daily'))

    def plot_stats(self, is_save = False, save_path = ''):
        plt.figure(figsize=(12, 8))
        pf.plot_drawdown_underwater(self.pnl_mean)
        plt.legend()
        if is_save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'underwater.jpg'))

        fig, ax = plt.subplots(1, 1, figsize = (12, 8))
        (1 + self.pnl_mean.dropna()).cumprod().plot(ax = ax, logy = True);
        ax.set_title("Cummulative Excess Return, " + \
                    "\ntarget vol = " + str(VOL_TARGET) + ", look back = " + \
             str(VOL_LOOKBACK) + " months");
        if is_save:
            plt.savefig(os.path.join(save_path, 'cum_return.jpg'))

        # tmp = pd.DataFrame(self.pnl_mean.dropna(), columns = ['pnl'])
        # index = tmp.index
        # tmp['month'] = index.month
        # tmp['year'] = index.year
        # tmp = np.round(tmp, 6)
        # res = tmp.pivot('year', 'month', 'pnl')
        # res['total'] = np.sum(res, axis=1)
        # fig, ax = plt.subplots(figsize=(20,20));
        # sns.heatmap(res.fillna(0) * 100,
        #             annot=True,
        #             annot_kws={
        #                 "size": 13},
        #             alpha=1.0,
        #             center=0.0,
        #             cbar=True,
        #             cmap=cm.PiYG,
        #             linewidths=.5,
        #             ax = ax); 
        # ax.set_ylabel('Year');
        # ax.set_xlabel('Month');
        # ax.set_title("Monthly Returns (%), " + \
        #             "\ntarget vol = " + str(VOL_TARGET) + ", look back = " + \
        #             str(VOL_LOOKBACK) + " months");
        # if is_save:
        #     plt.savefig(os.path.join(save_path, 'heatmaps.jpg'))

    