from moment_strategy.base_strat import BaseStrat
import numpy as np
import pandas as pd

class LongStrat(BaseStrat):

    def cal_signal(self) -> pd.DataFrame:
        signal = self.price_data.copy()
        signal = 1
        self.signal = signal
        return signal