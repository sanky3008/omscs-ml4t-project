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

    manualstrategy = ManualStrategy()
    strategylearner = StrategyLearner(impact=impact, commission=commission)

    m_trades = manualstrategy.testpolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed)

    strategylearner.add_evidence(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    s_trades = strategylearner.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)

    b_trades = pd.DataFrame(0, index=m_trades.index, columns=[symbol])
    b_trades.iloc[0] = 1000

    m_pv = compute_portvals(m_trades, start_val=sv, commission=commission, impact=impact)
    s_pv = compute_portvals(s_trades, start_val=sv, commission=commission, impact=impact)
    b_pv = compute_portvals(b_trades, start_val=sv, commission=commission, impact=impact)

    print(compute_stats(s_pv))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(m_pv, label='Manual Strategy')
    ax.plot(s_pv, label="Strategy Learner")
    ax.plot(b_pv, label="Benchmark")
    ax.legend()
    plt.show()







