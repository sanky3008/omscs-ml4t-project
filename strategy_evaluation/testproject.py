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
    # rand.seed(904081199)
    # np.random.seed(904081199)
    #
    symbol = "AAPL"
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)
    sv = 100000
    # commission = 9.95
    # impact = 0.005
    commission = 9.95
    impact = 0.005
    #
    # manualstrategy = ManualStrategy()
    # strategylearner = StrategyLearner(impact=impact, commission=commission, verbose=False)
    #
    # m_trades = manualstrategy.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
    #
    # strategylearner.add_evidence(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    # s_trades = strategylearner.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
    #
    # b_trades = pd.DataFrame(0, index=m_trades.index, columns=[symbol])
    # b_trades.iloc[0] = 1000
    #
    # m_pv = compute_portvals(m_trades, start_val=sv, commission=commission, impact=impact)
    # s_pv = compute_portvals(s_trades, start_val=sv, commission=commission, impact=impact)
    # b_pv = compute_portvals(b_trades, start_val=sv, commission=commission, impact=impact)
    #
    # print("\n\nOut of Sample:")
    # print(compute_stats(s_pv))
    # print("\n")
    # print(strategylearner.statespace)
    #
    # print("\n\nIn-sample:")
    # m_is = manualstrategy.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    # s_is = strategylearner.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    # # print(s_is)
    # print("\nManual: ", compute_stats(compute_portvals(m_is, start_val=sv, commission=commission, impact=impact)))
    # print("\nStrategy: ", compute_stats(compute_portvals(s_is, start_val=sv, commission=commission, impact=impact)))
    # print("\nBenchmark: ", compute_stats(compute_portvals(b_trades, start_val=sv, commission=commission, impact=impact)))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.plot(m_pv, label='Manual Strategy')
    # ax.plot(s_pv, label="Strategy Learner")
    # ax.plot(b_pv, label="Benchmark")
    # ax.legend()
    # plt.show()

    for symbol in ["ML4T-220", "AAPL", "SINE_FAST_NOISE", "UNH", "GNW", "SINE_SLOW_NOISE", "SNV"]:
        rand.seed(904081199)
        np.random.seed(904081199)
        strategylearner = StrategyLearner(impact=impact, commission=commission, verbose=False)
        st = time.time()
        strategylearner.add_evidence(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
        et = time.time()
        is_trades = strategylearner.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
        os_trades = strategylearner.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
        b_trades = pd.DataFrame(0, index=is_trades.index, columns=[symbol])
        b_trades.iloc[0] = 1000
        b_pv = compute_portvals(b_trades, start_val=sv, commission=commission, impact=impact)
        is_pv = compute_portvals(is_trades, start_val=sv, commission=commission, impact=impact)
        os_pv = compute_portvals(os_trades, start_val=sv, commission=commission, impact=impact)
        print(symbol, compute_stats(is_pv)[0]*100, compute_stats(os_pv)[0]*100, compute_stats(b_pv)[0]*100, et-st)

