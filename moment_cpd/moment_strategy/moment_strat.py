from moment_strategy.base_strat import BaseStrat
import numpy as np
import pandas as pd

class MomentStrat(BaseStrat):

    def cal_signal(self, w: float) -> pd.DataFrame:
        """Calculate intermediate strategy
        Args:
            srs (pd.DataFrame): series of prices
            w (float): weight, w=0 is Moskowitz TSMOM
            volatility_scaling (bool, optional): [description]. Defaults to True.
        Returns:
            pd.DataFrame: series of captured returns
        """
        srs = self.price_data
        monthly_returns = BaseStrat.calc_returns(srs, 21)
        annual_returns = BaseStrat.calc_returns(srs, 252)

        signal = w * np.sign(monthly_returns) + (1 - w) * np.sign(annual_returns)
        self.signal = signal

        return signal