from moment_strategy.base_strat import BaseStrat
from moment_strategy.macd_strat import MACDStrat
import pandas as pd

class DMNStrat(BaseStrat):
    
    def cal_signal(self, w: float) -> pd.DataFrame:
        raise NotImplemented



