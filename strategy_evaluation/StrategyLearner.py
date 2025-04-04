"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Sankalp Phadnis  	   		 	 	 			  		 			     			  	 
GT User ID: sphadnis9  		 	 	 			  		 			     			  	 
GT ID: 904081199	  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import util as ut
import QLearner as ql

import TheoreticallyOptimalStrategy as TOS
from marketsimcode import compute_portvals, compute_stats
import indicators
from util import get_data
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
class StrategyLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # constructor  		  	   		 	 	 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.impact = impact  		  	   		 	 	 			  		 			     			  	 
        self.commission = commission
        self.epochs = 500
        self.learner = ql.QLearner(
            num_states=332,
            num_actions=3, # 0 -> do nothing, 1 -> buy, 2 -> sell
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
        )

    # this method gives a single state for given indicators
    def discretize(self, bbp, rsi, macd_crossover):
        dbbp = min(bbp, 99) // 10
        drsi = min(rsi, 99) // 10

        return dbbp * 10 * 3 + drsi * 3 + macd_crossover

    # this method fetch price data and generates states
    def get_indicators(self, symbol, sd, ed):
        indicator = indicators.Indicators(symbol, pd.date_range(sd, ed))
        bbp = indicator.get_bbp(window=20)
        rsi = indicator.get_rsi(window=14)
        macd = indicator.get_macd().loc[sd:]  # My version of indicators.py is also returning values 9 market days prior to sd

        # Calculate MACD crossover
        macd_prev = macd.shift(1)
        macd_prev.iloc[0] = macd_prev.iloc[1]
        conditions = [
            (macd_prev <= 0) & (macd > 0),
            (macd_prev >= 0) & (macd < 0)
        ]
        values = [1, 2] # 0 -> do nothing, 1 -> buy, 2 -> sell

        # Documentation: https://numpy.org/doc/2.1/reference/generated/numpy.select.html
        macd_crossover = pd.DataFrame({'result': np.select(conditions, values, default=0)}, index=macd.index,
                                      columns=["result"])

        return bbp, rsi, macd_crossover

    # this method will get the price data for the given symbol
    def get_price(self, symbol, sd, ed):
        prices = get_data(symbol, pd.date_range(sd, ed))

        return prices[symbol]

    # update position based on action
    def update_pos(self, pos, action):
        if action == 0:
            return pos
        elif action == 1:
            return 1000
        elif action == 2:
            return -1000

    # compute daily returns
    def get_dr(self, price):
        dr = (price / price.shift(1)) - 1
        dr.iloc[0] = 0
        return dr

    def get_reward(self, action, date, price, dr, net_position, symbol):
        yesterday_price = price.shift(1)
        if action == 0: # do nothing
            return net_position * dr.loc[date, symbol]
        elif action == 1: # buy
            return -(self.impact + self.commission/price.loc[date, symbol])
        elif action == 2: # sell
            return -(self.impact + self.commission/price.loc[date, symbol])

    # this method should create a QLearner, and train it for trading
    def add_evidence(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):
        """  		  	   		 	 	 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  	 	   		 	 	 			  		 			     			  	 
        """

        # add your code to do learning here
        bbp, rsi, macd_crossover = self.get_indicators(symbol, sd, ed)
        price = self.get_price([symbol], sd, ed)
        dr = self.get_dr(price)
        signals = pd.DataFrame(0, index=price.index, columns=["signal"])

        state = int(self.discretize(bbp.iloc[0], rsi.iloc[0], macd_crossover.iloc[0][0]))
        action = self.learner.querysetstate(state)
        net_position = self.update_pos(0, action)
        scores = []
        for count in range(0, 1000):
            scores.append(0)
            for date in price.index.tolist()[1:]:
                reward = self.get_reward(action, date, price, dr, net_position, symbol) # Accommodate for reward & commission
                scores[-1] = scores[-1]*(1+reward)
                state = int(self.discretize(bbp.loc[date], rsi.loc[date], macd_crossover.loc[date][0]))
                action = self.learner.query(state, reward)
                net_position = self.update_pos(net_position, action)
                signals.loc[date, symbol] = action

            print("\nEPOCH: ", count)
            print("\nReward: ", scores[-1])
            print("\nScores & Count: ", scores, count)
            print("\n")

            if len(scores) >= 3 and (abs(scores[-2]/scores[-1] - 1) <= 0.01) and (abs(scores[-3]/scores[-2] - 1) <= 0.01):
                break

        return
  		  	   		 	 	 			  		 			     			  	 
    # for a set of signals, this method generates it into a trades df
    def get_trades(self, signals, symbol):
        net_position = 0
        trades = pd.DataFrame(0, index=signals.index, columns=[symbol])

        for date in signals.index:
            if signals.loc[date].iloc[0] == 1:
                trades.loc[date] = 1000 - net_position
            elif signals.loc[date].iloc[0] == 2:
                trades.loc[date] = -1000 - net_position

            net_position += trades.loc[pd.to_datetime(date), symbol]

        if self.verbose:
            print("Trade File\n", trades)

        return trades

    # this method should use the existing policy and test it against new data
    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	 	 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	 	 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	 	 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	 	 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
        """
        bbp, rsi, macd_crossover = self.get_indicators(symbol, sd, ed)
        price = self.get_price([symbol], sd, ed)
        dr = self.get_dr(price)
        signals = pd.DataFrame(0, index=price.index, columns=[symbol])

        state = int(self.discretize(bbp.iloc[0], rsi.iloc[0], macd_crossover.iloc[0][0]))
        action = self.learner.querysetstate(state)
        net_position = self.update_pos(0, action)
        for date in price.index.tolist()[1:]:
            reward = int(dr.loc[date, symbol] * net_position)
            state = int(self.discretize(bbp.loc[date], rsi.loc[date], macd_crossover.loc[date][0]))
            action = self.learner.query(state, reward)
            net_position = self.update_pos(net_position, action)
            signals.loc[date, symbol] = action

        print(signals)
        return self.get_trades(signals, symbol)

  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		 	 	 			  		 			     			  	 
