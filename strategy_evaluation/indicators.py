import numpy as np
from util import get_data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

class Indicators:
    def __init__(self, symbol, date_range):
        self.prices_master = get_data([symbol], pd.date_range(date_range[0]-dt.timedelta(90), date_range[-1]+dt.timedelta(90)))
        self.market_days = self.prices_master.index
        self.date_range = self.__build_date_range(date_range)
        self.symbol = symbol

    def __build_date_range(self, date_range):
        sd = date_range[0]
        ed = date_range[-1]

        while sd not in self.market_days:
            sd = sd + dt.timedelta(1)

        while ed not in self.market_days:
            ed = ed - dt.timedelta(1)

        return pd.date_range(sd, ed)

    def __get_prev_market_day(self, sd, window):
        return self.market_days[self.market_days.get_loc(sd) - window]

    """
    Bollinger Band %age - Indicator #1 + Helper Functions
    """
    def get_bbp(self, window):
        date_range = self.date_range
        prices_bbp = get_data([self.symbol], date_range).loc[:, self.symbol]
        sma = self.__get_sma(window, date_range)
        std = self.__get_std(window, date_range)
        upper = sma + 2*std
        lower = sma - 2*std

        bbp = (prices_bbp - lower)/(upper - lower)

        return bbp*100

    def __get_sma(self, window, date_range):
        sd = date_range[0]
        sd_sma = self.__get_prev_market_day(sd, window) #Need values before start date to avoid NaN for first window observations
        ed = date_range[-1]
        prices_sma = self.prices_master.loc[sd_sma:ed, self.symbol]
        sma = prices_sma.rolling(window).mean()
        return sma.loc[sd:]

    def __get_std(self, window, date_range):
        sd = date_range[0]
        sd_std = self.__get_prev_market_day(sd, window)  # Need values before start date to avoid NaN for first window observations
        ed = date_range[-1]
        prices_std = self.prices_master.loc[sd_std:ed, self.symbol]
        std = prices_std.rolling(window).std()
        return std.loc[sd:]


    """
    RSI - Indicator #2 + Helper Functions
    """
    def get_rsi(self, window):
        date_range = self.date_range
        sd = date_range[0]
        sd_rsi = self.__get_prev_market_day(sd, window)
        ed = date_range[-1]
        dc = self.__get_dc(pd.date_range(sd_rsi, ed))
        gain = dc.clip(lower=0) # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.clip.html
        loss = dc.clip(upper=0) # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.clip.html

        avg_gain = gain.rolling(window).mean().loc[sd:]
        avg_loss = np.abs(loss.rolling(window).mean().loc[sd:])

        rsi = 100 - (100/(1 + (avg_gain/avg_loss)))
        rsi[rsi == np.inf] = 100
        return rsi

    def __get_dc(self, date_range):
        sd = date_range[0]
        sd_dc = self.__get_prev_market_day(sd, 1)
        ed = date_range[-1]

        prices_dc = self.prices_master.loc[sd_dc:ed, self.symbol]
        dc = prices_dc - prices_dc.shift(1)

        return dc.loc[sd:]


    """
    MACD - Indicator #3 + Helper Functions
    """
    def get_macd(self, fast_window=12, slow_window=26):
        date_range = self.date_range
        sd = date_range[0]
        sd_macd = self.__get_prev_market_day(sd, 9) # We need 9 periods prior data for signal
        ed = date_range[-1]
        fast_ema = self.__get_ema(fast_window, pd.date_range(sd_macd, ed))
        slow_ema = self.__get_ema(slow_window, pd.date_range(sd_macd, ed))
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        cont_histogram = macd-signal

        return cont_histogram


    def __get_ema(self, window, date_range):
        sd = date_range[0]
        sd_ema = self.__get_prev_market_day(sd, window)
        ed = date_range[-1]
        prices_ema = self.prices_master.loc[sd_ema:ed, self.symbol]
        ema = prices_ema.ewm(span=window, adjust=False).mean() # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
        return ema.loc[sd:]


    """
    Rate of Change (ROC) - Indicator #4 + Helper Functions
    """
    def get_roc(self, window):
        date_range = self.date_range
        sd = date_range[0]
        sd_roc = self.__get_prev_market_day(sd, window)
        ed = date_range[-1]
        prices_roc = self.prices_master.loc[sd_roc:ed, self.symbol]
        roc = (prices_roc/prices_roc.shift(window)) - 1
        return roc*100


    """
    Percentage Price Oscillator (PPO) - Indicator #5 + Helper Functions
    """
    def get_ppo(self, fast_window=9, slow_window=26):
        date_range = self.date_range
        fast_ema = self.__get_ema(fast_window, date_range)
        slow_ema = self.__get_ema(slow_window, date_range)

        ppo = fast_ema/slow_ema - 1
        return ppo*100

    """
    Run function for testproject.py
    """
    def __run_bbp(self, prices):
        # Bollinger Bands Plots
        window = 20
        bbp = self.get_bbp(window)
        sma = self.__get_sma(window, self.date_range)
        std = self.__get_std(window, self.date_range)

        upper_band = sma + 2 * std
        lower_band = sma - 2 * std

        # https://www.geeksforgeeks.org/how-to-change-the-figure-size-with-subplots-in-matplotlib/
        fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

        ax.plot(prices, color="blue", label="Price")
        ax.plot(sma, color="red", label="20 day SMA")
        ax.plot(upper_band, color="purple", label="Upper Bollinger Band", linestyle='--')
        ax.plot(lower_band, color="violet", label="Lower Bollinger Band", linestyle='--')
        ax.grid()
        ax.set(xlabel="Date", ylabel="Price", title="Bollinger Band Percentage")
        ax.legend(loc="best")

        ax1.plot(bbp, color="blue", label="Bollinger Band %age")
        ax1.axhline(y=100, color="red", label="Overbought", linestyle='--')
        ax1.axhline(y=0, color="green", label="Oversold", linestyle='--')
        ax1.grid()
        ax1.set(xlabel="Date", ylabel="BBP")
        ax1.legend(loc="upper left")

        fig.tight_layout()
        plt.savefig("images/bbp.png", bbox_inches='tight')
        plt.close(fig)

    def __run_rsi(self, prices):
        window = 14
        rsi = self.get_rsi(window)

        # https://www.geeksforgeeks.org/how-to-change-the-figure-size-with-subplots-in-matplotlib/
        fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        ax.plot(prices, color="blue", label="Price")
        ax.grid()
        ax.legend(loc="best")
        ax.set(xlabel="Date", ylabel="Price", title="Price")

        ax1.plot(rsi, color="blue", label="14-day RSI")
        ax1.axhline(y=70, color="red", label="Overbought", linestyle='--')
        ax1.axhline(y=30, color="green", label="Oversold", linestyle='--')
        ax1.grid()
        ax1.set(xlabel="Date", ylabel="RSI", title="14-day RSI")
        ax1.legend(loc="best", bbox_to_anchor=(1,1))

        fig.tight_layout()
        plt.savefig("images/rsi.png")
        plt.close(fig)

    def __run_macd(self, prices):
        fast_window = 12
        slow_window = 26
        cont_histogram = self.get_macd(12, 26)
        fast_ema = self.__get_ema(fast_window, self.date_range)
        slow_ema = self.__get_ema(slow_window, self.date_range)
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=9, adjust=False).mean()

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,12), gridspec_kw={'height_ratios': [2, 1, 1]})
        ax1.plot(prices, color="blue", label="Price")
        ax1.plot(fast_ema, color="red", label="Fast EMA", linestyle='--')
        ax1.plot(slow_ema, color="orange", label="Slow EMA", linestyle='--')
        ax1.set(xlabel="Date", ylabel="Price", title="Price, 12 day and 26 day EMA")
        ax1.legend(loc="best")
        ax1.grid()

        ax2.plot(macd, color="blue", label="MACD")
        ax2.plot(signal, color="red", label="Signal")
        ax2.set(xlabel="Date", ylabel="MACD/Signal", title="MACD & Signal Line")
        ax2.legend(loc="best")
        ax2.grid()

        ax3.plot(cont_histogram, color="blue", label="MACD - Signal")
        ax3.axhline(y=0, color="red", linestyle='--', label="Delta = 0")
        ax3.set(xlabel="Date", ylabel="Delta", title="MACD-Signal: Continuous Delta")
        ax3.legend(loc="best")
        ax3.grid()

        fig.tight_layout()
        plt.savefig("images/macd.png")
        plt.close(fig)

    def __run_roc(self, prices):
        roc = self.get_roc(14)

        fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

        ax.plot(prices, color="blue", label="Price")
        ax.grid()
        ax.legend(loc="best")
        ax.set(xlabel="Date", ylabel="Price", title="Price")

        ax1.plot(roc, color="blue", label="Rate of Change")
        ax1.axhline(y=0, color="red", label="0 change", linestyle='--')
        ax1.grid()
        ax1.set(xlabel="Date", ylabel="%age", title="14-day Momentum Rate of Change")
        ax1.legend(loc="best", bbox_to_anchor=(1, 1))

        fig.tight_layout()
        plt.savefig("images/roc.png")
        plt.close(fig)

    def __run_ppo(self, prices):
        ppo = self.get_ppo()
        fast_window = 9
        slow_window = 26
        fast_ema = self.__get_ema(fast_window, self.date_range)
        slow_ema = self.__get_ema(slow_window, self.date_range)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

        ax1.plot(prices, color="blue", label="Price")
        ax1.plot(fast_ema, color="red", label="Fast EMA", linestyle='--')
        ax1.plot(slow_ema, color="orange", label="Slow EMA", linestyle='--')
        ax1.set(xlabel="Date", ylabel="Price", title="Price, 9 day and 26 day EMA")
        ax1.legend(loc="best")
        ax1.grid()

        ax2.plot(ppo, color="blue", label="PPO")
        ax2.axhline(y=0, color="red", label="0 change", linestyle='--')
        ax2.grid()
        ax2.set(xlabel="Date", ylabel="%age", title="Percentage Price Oscillator")
        ax2.legend(loc="best", bbox_to_anchor=(1, 1))

        fig.tight_layout()
        plt.savefig("images/ppo.png")
        plt.close(fig)

    def run(self):
        prices = self.prices_master.loc[self.date_range[0]:self.date_range[-1], self.symbol]
        self.__run_bbp(prices)
        self.__run_rsi(prices)
        self.__run_macd(prices)
        self.__run_roc(prices)
        self.__run_ppo(prices)


