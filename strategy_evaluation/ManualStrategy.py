import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import TheoreticallyOptimalStrategy as TOS
from marketsimcode import compute_portvals, compute_stats
import indicators
from util import get_data

class ManualStrategy(object):
    def __init__(self, verbose=False, commission=0, impact=0):
        self.verbose = verbose
        self.commission = commission
        self.impact = impact

    def author():
        return "sphadnis9"

    def study_group():
        return "sphadnis9"

    def add_evidence(
        self,
        symbol='IBM',
        sd = dt.datetime(2008, 1, 1, 0, 0),
        ed = dt.datetime(2009, 1, 1, 0, 0),
        sv=100000
    ):
        pass

    def testpolicy(
            self,
            symbol='IBM',
            sd = dt.datetime(2008, 1, 1, 0, 0),
            ed = dt.datetime(2009, 1, 1, 0, 0),
            sv=100000
    ):
        """
        Gives out trades based on a manual trading strategy.

        Strategy
            Buy when:
                BBP <= 0
                RSI <= 30
                MACD - Signal goes from -ve to +ve
            Sell when:
                BBP >= 1
                RSI >= 70
                MACD - Signal goes from +ve to -ve

        Parameters
            symbol (str) – The stock symbol that you trained on on
            sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) – The starting value of the portfolio
        Returns
            A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        """
        indicator = indicators.Indicators(symbol, pd.date_range(sd, ed))
        bbp = indicator.get_bbp(window = 20)
        rsi = indicator.get_rsi(window = 14)
        macd = indicator.get_macd().loc[sd:] # My version of indicators.py is also returning values 9 market days prior to sd

        # Calculate MACD crossover
        macd_prev = macd.shift(1)
        macd_prev.iloc[0] = macd_prev.iloc[1]
        conditions = [
            (macd_prev <= 0) & (macd > 0),
            (macd_prev >= 0) & (macd < 0)
        ]
        values = [1, -1]
        # Documentation: https://numpy.org/doc/2.1/reference/generated/numpy.select.html
        macd_crossover = pd.DataFrame({'result': np.select(conditions, values, default=0)}, index=macd.index, columns=["result"])

        # Generate signals
        signals_cond = [
            ((bbp <= 0) & (rsi <= 30)) | ((bbp <= 0) & (macd_crossover["result"] == 1)) | ((rsi <= 30) & (macd_crossover["result"] == 1)),
            ((bbp >= 100) & (rsi >= 70)) | ((bbp >= 100) & (macd_crossover["result"] == 1)) | ((rsi >= 70) & (macd_crossover["result"] == 1))
        ]
        signal_values = [1, -1]
        signals = pd.DataFrame({'result': np.select(signals_cond, signal_values, default=0)}, index=bbp.index)

        # Based on indicators on T day, we will place orders on T+1 day
        signals = signals.shift(1)
        signals = signals.fillna(0)
        signals.drop(signals.index[-1])

        # Generate trades
        net_position = 0
        manual_trades = pd.DataFrame(0, index=bbp.index, columns=[symbol])

        for date in signals.index:
            if signals.loc[date].iloc[0] == 1:
                manual_trades.loc[date, "signal"] = 1000 - net_position
            elif signals.loc[date].iloc[0] == -1:
                manual_trades.loc[date] = -1000 - net_position

            net_position += manual_trades.loc[pd.to_datetime(date), symbol]

        if self.verbose:
            print("Manual Trade File\n", manual_trades)

        return manual_trades

