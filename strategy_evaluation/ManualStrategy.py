import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

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

    def testPolicy(
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
        ppo = indicator.get_ppo()

        # Generate signals
        signals_cond = [
            ((bbp <= 0) & (rsi <= 30)) | ((bbp <= 0) & (ppo > 1)) | ((rsi <= 30) & (ppo > 1)),
            ((bbp >= 100) & (rsi >= 70)) | ((bbp >= 100) & (ppo < -1)) | ((rsi >= 70) & (ppo < -1))
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
                manual_trades.loc[date] = 1000 - net_position
            elif signals.loc[date].iloc[0] == -1:
                manual_trades.loc[date] = -1000 - net_position

            net_position += manual_trades.loc[pd.to_datetime(date), symbol]

        if self.verbose:
            print("Manual Trade File\n", manual_trades)

        return manual_trades

    def run(self, symbol, is_sd, is_ed, os_sd, os_ed, sv):
        is_trades = self.testPolicy(symbol=symbol, sd=is_sd, ed=is_ed, sv=sv)
        os_trades = self.testPolicy(symbol=symbol, sd=os_sd, ed=os_ed, sv=sv)
        b_trades = pd.DataFrame(0, index=is_trades.index, columns=[symbol])
        os_b_trades = pd.DataFrame(0, index=os_trades.index, columns=[symbol])
        b_trades.iloc[0] = 1000
        os_b_trades.iloc[0] = 1000

        is_pv = compute_portvals(is_trades, commission=self.commission, impact=self.impact, start_val=sv)
        os_pv = compute_portvals(os_trades, commission=self.commission, impact=self.impact, start_val=sv)
        b_pv = compute_portvals(b_trades, commission=self.commission, impact=self.impact, start_val=sv)
        os_b_pv = compute_portvals(os_b_trades, commission=self.commission, impact=self.impact, start_val=sv)

        is_pv_norm = is_pv/is_pv.iloc[0]
        os_pv_norm = os_pv/os_pv.iloc[0]
        b_pv_norm = b_pv/b_pv.iloc[0]
        os_b_pv_norm = os_b_pv / os_b_pv.iloc[0]

        is_long_dates = is_trades.loc[is_trades[symbol] >= 1000].index
        is_short_dates = is_trades.loc[is_trades[symbol] <= -1000].index

        os_long_dates = os_trades.loc[os_trades[symbol] >= 1000].index
        os_short_dates = os_trades.loc[os_trades[symbol] <= -1000].index

        fig, (insample, outsample) = plt.subplots(figsize=(18, 8), ncols = 2, nrows = 1)
        insample.plot(is_pv_norm, color='red', label='Manual Strategy')
        insample.plot(b_pv_norm, color='purple', label='Benchmark')

        i = 0
        for date in is_long_dates:
            if i == 0:
                insample.axvline(date, color='blue', linestyle='--', label='Long')
                i = 1
            else:
                insample.axvline(date, color='blue', linestyle='--')

        i = 0
        for date in is_short_dates:
            if i == 0:
                insample.axvline(date, color='black', linestyle='--', label='Short')
                i = 1
            else:
                insample.axvline(date, color='black', linestyle='--')

        insample.legend()
        insample.grid()
        insample.set(xlabel="Date", ylabel="Normalised Portfolio Value", title="In-Sample Manual Strategy")

        outsample.plot(os_pv_norm, color='red', label='Manual Strategy')
        outsample.plot(os_b_pv_norm, color='purple', label='Benchmark')

        i = 0
        for date in os_long_dates:
            if i == 0:
                outsample.axvline(date, color='blue', linestyle='--', label='Long')
                i = 1
            else:
                outsample.axvline(date, color='blue', linestyle='--')

        i = 0
        for date in os_short_dates:
            if i == 0:
                outsample.axvline(date, color='black', linestyle='--', label='Short')
                i = 1
            else:
                outsample.axvline(date, color='black', linestyle='--')

        outsample.legend()
        outsample.grid()
        outsample.set(xlabel="Date", ylabel="Normalised Portfolio Value", title="Out-Sample Manual Strategy")

        fig.tight_layout()
        fig.savefig("images/manual_strategy.png")
        plt.close()

        return is_pv, os_pv, b_pv, os_b_pv

