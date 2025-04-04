import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import TheoreticallyOptimalStrategy as TOS
from marketsimcode import compute_portvals, compute_stats
import indicators
from strategy_evaluation.ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner


def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

if __name__ == "__main__":
    symbol = "JPM"
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    manualstrategy = ManualStrategy()
    strategylearner = StrategyLearner(impact=impact, commission=commission)
    # trades = manualstrategy.testpolicy(symbol="JPM", sd=in_sample_sd, ed=in_sample_ed)
    tt = strategylearner.add_evidence(symbol="JPM", sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    trades = strategylearner.testPolicy(symbol="JPM", sd=in_sample_sd, ed=in_sample_ed, sv=sv)

    print(trades.equals(tt))

    portvals = compute_portvals(trades, start_val=sv, commission=commission, impact=impact)

    print(compute_stats(portvals))







