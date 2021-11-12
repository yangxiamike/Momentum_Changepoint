from pandas.core.frame import DataFrame
from finance_data.base_data import BaseData
from moment_strategy.base_strat import BaseStrat
from moment_strategy.macd_strat import MACDStrat
import numpy as np
import os
import pandas as pd

from moment_strategy.macd_strat import MACDStrat

HALFLIFE_WINSORISE = 252
VOL_THRESHOLD = 5
TREND_COMBINATIONS = [(8, 24), (16, 48), (32, 96)]


class DMNData(BaseData):

    def make_all_features(self, df_assets: pd.DataFrame) -> pd.DataFrame:
        """
        df_assets: [date, price, ticker]
        """
        df_features = []
        for asset, df_asset in df_assets.groupby('ticker'):
            df_feature = self.make_single_features(df_asset)
            df_features.append(df_feature)

        df_features = pd.concat(df_features)
        return df_features

    @staticmethod
    def read_changepoint_results_and_fill_na(
        file_path: str, lookback_window_length: int
    ) -> pd.DataFrame:
        """Read output data from changepoint detection module into a dataframe.
        For rows where the module failed, information for changepoint location and severity is
        filled using the previous row.


        Args:
            file_path (str): the file path of the csv containing the results
            lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

        Returns:
            pd.DataFrame: changepoint severity and location information
        """

        return (
            pd.read_csv(file_path, index_col=0, parse_dates=True)
            .fillna(method="ffill")
            .dropna()  # if first values are na
            .assign(
                cp_location_norm=lambda row: (row["t"] - row["cp_location"])
                / lookback_window_length
            )  # fill by assigning the previous cp and score, then recalculate norm location
        )

    @staticmethod
    def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
        """Read output data from changepoint detection module for all assets into a dataframe.


        Args:
            file_path (str): the folder path containing csvs with the CPD the results
            lookback_window_length (int): lookback window length

        Returns:
            pd.DataFrame: changepoint severity and location information for all assets
        """

        return pd.concat(
            [
                DMNData.read_changepoint_results_and_fill_na(
                    os.path.join(folder_path, f), lookback_window_length
                ).assign(ticker=os.path.splitext(f)[0])
                for f in os.listdir(folder_path)
            ]
        )

    @staticmethod
    def include_changepoint_features(
        features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
    ) -> pd.DataFrame:
        """combine CP features and DMN featuress

        Args:
            features (pd.DataFrame): features
            cpd_folder_name (pd.DataFrame): folder containing CPD results
            lookback_window_length (int): LBW used for the CPD

        Returns:
            pd.DataFrame: features including CPD score and location
        """
        features = features.merge(
            DMNData.prepare_cpd_features(cpd_folder_name, lookback_window_length)[
                ["ticker", "cp_location_norm", "cp_score"]
            ]
            .rename(
                columns={
                    "cp_location_norm": f"cp_rl_{lookback_window_length}",
                    "cp_score": f"cp_score_{lookback_window_length}",
                }
            )
            .reset_index(),  # for date column
            on=["date", "ticker"],
        )

        features.index = features["date"]

        return features

    @staticmethod
    def make_single_features(df_asset: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare single asset's input features for deep learning model
        Args:
            df_asset (pd.DataFrame): time-series for single asset with column "price", index -> "date"
        Returns:
            pd.DataFrame: input features, date x feature_columns
        """
        df_asset = df_asset[
        ~df_asset["price"].isna()
        | ~df_asset["price"].isnull()
        | (df_asset["price"] > 1e-8)  # price is zero
        ].copy()

        # winsorize using rolling 5X standard deviations to remove outliers
        ewm = df_asset["price"].ewm(halflife=HALFLIFE_WINSORISE)
        means = ewm.mean()
        stds = ewm.std()
        df_asset["price"] = np.minimum(df_asset["price"], means + VOL_THRESHOLD * stds)
        df_asset["price"] = np.maximum(df_asset["price"], means - VOL_THRESHOLD * stds) 
        
        df_asset["daily_returns"] = BaseStrat.calc_returns(df_asset["price"])
        df_asset["daily_vol"] = BaseStrat.calc_daily_vol(df_asset["daily_returns"])
        # vol scaling and shift to be next day returns
        df_asset["target_returns"] = BaseStrat.calc_vol_scaled_returns(
            df_asset["daily_returns"], df_asset["daily_vol"]
        ).shift(-1)

        price = pd.DataFrame(df_asset['price'])
        daily_vol = df_asset[['daily_vol']]
        df_asset["norm_daily_return"] = BaseStrat.calc_normalised_returns(price, daily_vol, 1)
        df_asset["norm_monthly_return"] = BaseStrat.calc_normalised_returns(price, daily_vol, 21)
        df_asset["norm_quarterly_return"] = BaseStrat.calc_normalised_returns(price, daily_vol, 63)
        df_asset["norm_biannual_return"] = BaseStrat.calc_normalised_returns(price, daily_vol, 126)
        df_asset["norm_annual_return"] = BaseStrat.calc_normalised_returns(price, daily_vol, 252)
        
        # Calculate MACD feature
        macd_strat = MACDStrat(price)
        for short_window, long_window in TREND_COMBINATIONS:
            col_name = f"macd_{short_window}_{long_window}"
            df_asset[col_name] = macd_strat.cal_single_signal(short_window, long_window)
        
        # date features
        # df_asset["day_of_week"] = df_asset.index.isocalendar().day
        # df_asset["day_of_month"] = df_asset.index.map(lambda d: d.day)
        # df_asset["week_of_year"] = df_asset.index.isocalendar().week
        # df_asset["month_of_year"] = df_asset.index.map(lambda d: d.month)
        # df_asset["year"] = df_asset.index.isocalendar().year
        # df_asset["date"] = df_asset.index  # duplication but sometimes makes life easier
        columns = ['daily_returns', 'norm_daily_return', 'norm_monthly_return',
       'norm_quarterly_return', 'norm_biannual_return', 'norm_annual_return',
       'macd_8_24', 'macd_16_48', 'macd_32_96', 
       'target_returns', 'daily_vol', 'date', 'ticker']

        df_asset = df_asset[columns]
        return df_asset.dropna()
    
    
    

