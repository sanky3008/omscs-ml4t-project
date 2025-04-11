import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import TheoreticallyOptimalStrategy as TOS
from strategy_evaluation.marketsimcode import compute_portvals, compute_stats
import indicators
from strategy_evaluation.ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import random as rand
import time


def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

if __name__ == "__main__":
    rand.seed(904081199)
    np.random.seed(904081199)
    symbol = "JPM"
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    # Run Manual Strategy
    manualstrategy = ManualStrategy(verbose=False, commission=commission, impact=impact)
    is_pv, os_pv, b_pv, os_b_pv = manualstrategy.run(symbol, is_sd=in_sample_sd, is_ed=in_sample_ed, os_sd=out_sample_sd, os_ed=out_sample_ed, sv=sv)



