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
import experiment1
import experiment2


def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

if __name__ == "__main__":
    rand.seed(904081199)
    np.random.seed(904081199)

    symbol = "JPM"
    is_sd = dt.datetime(2008, 1, 1)
    is_ed = dt.datetime(2009, 12, 31)
    os_sd = dt.datetime(2010, 1, 1)
    os_ed = dt.datetime(2011, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    # Run Manual Strategy
    manualstrategy = ManualStrategy(verbose=False, commission=commission, impact=impact)
    ms_results = manualstrategy.run(symbol, is_sd=is_sd, is_ed=is_ed, os_sd=os_sd, os_ed=os_ed, sv=sv)
    m_is_pv, m_os_pv, is_b_pv, os_b_pv = ms_results

    # Run Experiment 1
    s_is_pv, s_os_pv = experiment1.run(
        symbol=symbol,
        is_sd=is_sd,
        is_ed=is_ed,
        os_sd=os_sd,
        os_ed=os_ed,
        sv=sv,
        ms_results=ms_results,
        commission=commission,
        impact=impact
    )



    # ROUGH

    # strategylearner = StrategyLearner(verbose=False, impact=impact, commission=commission)
    # strategylearner.add_evidence(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)
    #
    # s_is_trades = strategylearner.testPolicy(symbol, sd=is_sd, ed=is_ed, sv=sv)
    # s_os_trades = strategylearner.testPolicy(symbol, sd=os_sd, ed=os_ed, sv=sv)
    # s_is_pv = compute_portvals(s_is_trades, start_val=sv, impact=impact, commission=commission)
    # s_os_pv = compute_portvals(s_os_trades, start_val=sv, impact=impact, commission=commission)
    #
    # b_trades = pd.DataFrame(0, index=s_is_trades.index, columns=[symbol])
    # b_trades.iloc[0] = 1000
    # b_pv = compute_portvals(b_trades, commission=commission, impact=impact, start_val=sv)
    #
    # print(symbol, compute_stats(s_is_pv)[0], compute_stats(b_pv)[0], compute_stats(s_os_pv)[0])