import numpy as np
from util import get_data
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

import TheoreticallyOptimalStrategy as tos
from marketsimcode import compute_portvals, compute_stats
from indicators import Indicators
import math

def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

def plot_tos(symbol, sd, ed, sv):
    df_trades = tos.testpolicy(symbol,sd,ed,sv)
    portvals = compute_portvals(df_trades, start_val=100000)
    df_benchmark = df_trades.copy()
    df_benchmark.iloc[0] = 1000
    df_benchmark.iloc[1:-1] = 0
    # print(df_benchmark)
    benchmark = compute_portvals(df_benchmark)
    # print(benchmark)

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    ax.plot(portvals / portvals.iloc[0], color='red', label="Theoretically Optimal Strategy")
    ax.plot(benchmark / benchmark.iloc[0], color='purple', label="Benchmark")
    ax.set(title="Theoretically Optimal Strategy v/s Benchmark", xlabel="Date", ylabel="Normalised Value")
    ax.grid()
    ax.legend(loc='upper left')
    plt.savefig("images/tos.png")
    plt.close(fig)

    results = open("p6_results.txt", "w")

    results.write("Theoretically Optimal Strategy Stats:\n\n")
    cr, adr, sddr, sr = compute_stats(portvals)
    bcr, badr, bsddr, bsr = compute_stats(benchmark)

    results.write(f"Sharpe Ratio of TOS: {sr}\n")
    results.write(f"Sharpe Ratio of Benchmark : {bsr}\n")
    results.write("\n")
    results.write(f"Cumulative Return of TOS: {cr}\n")
    results.write(f"Cumulative Return of Benchmark : {bcr}\n")
    results.write("\n")
    results.write(f"Standard Deviation of TOS: {sddr}\n")
    results.write(f"Standard Deviation of Benchmark : {bsddr}\n")
    results.write("\n")
    results.write(f"Average Daily Return of TOS: {adr}\n")
    results.write(f"Average Daily Return of Benchmark : {badr}\n")
    results.write("\n")
    results.write(f"Final Portfolio Value: {portvals[-1]}\n")

    results.close()



if __name__ == "__main__":
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    sv = 100000
    plot_tos("JPM", sd, ed, sv)
    indicator = Indicators(symbol, pd.date_range(sd,ed))
    indicator.run()
