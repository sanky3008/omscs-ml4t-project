import numpy as np
from util import get_data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

def testpolicy(
    symbol = "AAPL",
    sd = dt.datetime(2010, 1, 1),
    ed = dt.datetime(2011, 12, 31),
    sv = 100000
):
    """
    Look into the future and give a trade dataframe in output which will yield max returns
    """
    prices_master_data = get_data([symbol], pd.date_range(sd, ed), addSPY=True, colname="Adj Close")
    prices = prices_master_data[symbol]
    date_range = prices.index.tolist()

    df_trades = pd.DataFrame(0, index=date_range, columns=[symbol])
    one_day_ret = prices.shift(-1) - prices
    # print(prices)
    # print(one_day_ret)

    net_position = 0

    for date in date_range:
        if one_day_ret.loc[pd.to_datetime(date)] > 0:
            df_trades.loc[pd.to_datetime(date)] = 1000 - net_position
        elif one_day_ret.loc[pd.to_datetime(date)] < 0:
            df_trades.loc[pd.to_datetime(date)] = -1000 - net_position

        net_position += df_trades.loc[pd.to_datetime(date), symbol]
        # print(net_position)
        # print(one_day_ret.loc[pd.to_datetime(date)], df_trades.loc[pd.to_datetime(date)], net_position)
        # print("\n")

    # print(df_trades)
    return df_trades


if __name__ == "__main__":
    testpolicy(
        symbol = "AAPL",
        sd = dt.datetime(2010, 1, 1),
        ed = dt.datetime(2011, 12, 31),
        sv = 100000
    )
