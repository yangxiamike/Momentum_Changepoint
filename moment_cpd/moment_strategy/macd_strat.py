from moment_strategy.base_strat import BaseStrat
import numpy as np
import pandas as pd
from typing import List
from typing import Tuple

class MACDStrat(BaseStrat):
    def __init__(self, price_data, trend_combinations: List[Tuple[float, float]] = None):
        """
        Used to calculated the combined MACD signal for a multiple short/signal combinations,
        as described in https://arxiv.org/pdf/1904.04912.pdf
        Args:
            trend_combinations (List[Tuple[float, float]], optional): short/long trend combinations. Defaults to None.
        """
        super(MACDStrat, self).__init__(price_data)
        if trend_combinations is None:
            self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trend_combinations = trend_combinations

    def cal_single_signal(self, short_timescale: int, long_timescale: int) -> float:
        """Calculate MACD signal for a signal short/long timescale combination
        Args:
            short_timescale ([type]): short timescale
            long_timescale ([type]): long timescale
        Returns:
            float: MACD signal
        """
        macd = (
            self.price_data.ewm(halflife=self.cal_halflife(short_timescale)).mean()
            - self.price_data.ewm(halflife=self.cal_halflife(long_timescale)).mean()
        )
        q = macd / self.price_data.rolling(63).std().fillna(method="bfill")
        q =  q / q.rolling(252).std().fillna(method="bfill")
        return q
    
    @staticmethod
    def cal_halflife(timescale):
        return np.log(0.5) / np.log(1 - 1 / timescale)

    @staticmethod
    def scale_signal(y):
        return y * np.exp(-(y ** 2) / 4) / 0.89

    def cal_signal(self) -> float:
        """Combined MACD signal
        Returns:
            float: MACD combined signal
        """
        signal =  sum([self.cal_single_signal(S, L) for S, L in self.trend_combinations]) \
                    / len(self.trend_combinations)
        self.signal = signal
        return signal