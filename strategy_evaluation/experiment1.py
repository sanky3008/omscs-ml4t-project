import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


from marketsimcode import compute_portvals, compute_stats
import indicators
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import random as rand
import time


def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

def normalise(df):
    return df/df.iloc[0]

def run(
        symbol,
        is_sd,
        is_ed,
        os_sd,
        os_ed,
        sv,
        ms_results,
        commission,
        impact
    ):
    m_is_pv, m_os_pv, is_b_pv, os_b_pv = ms_results

    strategylearner = StrategyLearner(verbose=False, impact=impact, commission=commission)
    strategylearner.add_evidence(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)

    s_is_trades = strategylearner.testPolicy(symbol, is_sd, is_ed, sv)
    s_os_trades = strategylearner.testPolicy(symbol, os_sd, os_ed, sv)

    s_is_pv = compute_portvals(s_is_trades, start_val=sv, impact=impact, commission=commission)
    s_os_pv = compute_portvals(s_os_trades, start_val=sv, impact=impact, commission=commission)

    is_b_pv = normalise(is_b_pv)
    os_b_pv = normalise(os_b_pv)

    m_is_pv = normalise(m_is_pv)
    m_os_pv = normalise(m_os_pv)

    s_is_pv = normalise(s_is_pv)
    s_os_pv = normalise(s_os_pv)

    fig, (insample, outsample) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    insample.plot(is_b_pv, color = "purple", label="Benchmark")
    insample.plot(m_is_pv, color = "red", label="Manual Strategy")
    insample.plot(s_is_pv, color = "orange", label = "Strategy Learner")
    insample.grid()
    insample.legend()
    insample.set(xlabel='Date', ylabel='Normalised Value', title='JPM In-Sample Portfolio Values')

    outsample.plot(os_b_pv, color="purple", label="Benchmark")
    outsample.plot(m_os_pv, color="red", label="Manual Strategy")
    outsample.plot(s_os_pv, color="orange", label="Strategy Learner")
    outsample.grid()
    outsample.legend()
    outsample.set(xlabel='Date', ylabel='Normalised Value', title='JPM Out-Sample Portfolio Values')

    fig.savefig("images/experiment1.png")
    plt.close()

    return s_is_pv, s_os_pv