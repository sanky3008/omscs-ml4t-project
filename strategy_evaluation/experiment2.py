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
from util import get_data


def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

def run(
    symbol,
    is_sd,
    is_ed,
    sv,
    s_is_trades,
    results
):
    commission = 0
    learner_1 = StrategyLearner(verbose=False, impact=0, commission=commission)
    learner_2 = StrategyLearner(verbose=False, impact=0.005, commission=commission)
    learner_3 = StrategyLearner(verbose=False, impact=0.01, commission=commission)

    learner_1.add_evidence(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)
    learner_2.add_evidence(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)
    learner_3.add_evidence(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)

    trades_1 = learner_1.testPolicy(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)
    trades_2 = learner_2.testPolicy(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)
    trades_3 = learner_3.testPolicy(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)

    pv1 = compute_portvals(trades_1, start_val=sv, commission=commission, impact=0)
    pv2 = compute_portvals(trades_2, start_val=sv, commission=commission, impact=0.005)
    pv3 = compute_portvals(trades_3, start_val=sv, commission=commission, impact=0.01)

    npv1 = pv1 / pv1.iloc[0]
    npv2 = pv2 / pv2.iloc[0]
    npv3 = pv3 / pv3.iloc[0]

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax1.plot(npv1, label='Impact = 0')
    ax1.plot(npv2, label='Impact = 0.005')
    ax1.plot(npv3, label='Impact = 0.01')
    ax1.legend()
    ax1.grid()
    ax1.set(xlabel='Date', ylabel='Normalised Portfolio Value', title='JPM Strategy Learner with various Impacts')

    fig.savefig("images/experiment2.png")
    plt.close()

    results.write("\nTrades Placed")
    results.write(f"\nBenchmark (Commission = 9.95, Impact = 0.005): {s_is_trades[s_is_trades.iloc[:, 0] != 0].shape[0]}")
    results.write(f"\nImpact = 0, Commission = 0: {trades_1[trades_1.iloc[:, 0] != 0].shape[0]}")
    results.write(f"\nImpact = 0.005, Commission = 0: {trades_2[trades_2.iloc[:, 0] != 0].shape[0]}")
    results.write(f"\nImpact = 0.01, Commission = 0: {trades_3[trades_3.iloc[:, 0] != 0].shape[0]}")
    results.write("\n")

    return pv1, pv2, pv3
