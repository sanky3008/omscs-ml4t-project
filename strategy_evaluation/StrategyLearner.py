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
            num_states=27,
            num_actions=3, # 0 -> do nothing, 1 -> buy, 2 -> sell
            alpha=0.01,
            gamma=0.99,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
        )
        self.statespace = np.zeros(27)
        self.add_evidence_trades = 0

    def author():
        return "sphadnis9"

    def study_group():
        return "sphadnis9"

    # this method gives a single state for given indicators
    def discretize(self, bbp, rsi, ppo):
        # dbbp = min(bbp, 99) // 34
        # drsi = min(rsi, 99) // 34

        dbbp = 1
        if bbp <= 20:
            dbbp = 0
        elif bbp >= 80:
            dbbp = 2

        drsi = 1
        if rsi <= 30: #prev: 40
            drsi = 0
        elif rsi >= 70: #prev: 60
            drsi = 2

        dppo = 0
        if ppo < -1:
            dppo = 1
        elif ppo > 1:
            dppo = 2

        return dbbp * 3 * 3 + drsi * 3 + dppo

    # this method fetch price data and generates states
    def get_indicators(self, symbol, sd, ed):
        indicator = indicators.Indicators(symbol, pd.date_range(sd, ed))
        bbp = indicator.get_bbp(window=20)
        rsi = indicator.get_rsi(window=14)
        ppo = indicator.get_ppo()

        return bbp, rsi, ppo

    # this method will get the price data for the given symbol
    def get_price(self, symbol, sd, ed):
        prices = get_data(symbol, pd.date_range(sd, ed))

        return prices[symbol]

    def get_reward(self, action, date, price, holdings):
        yesterday_price = price.shift(1)

        costs = 0
        if action != 0:
            costs = self.impact + self.commission/yesterday_price.loc[date] # Prev this was commission/price.loc[date]

        reward = holdings*(price.loc[date]/yesterday_price.loc[date] - 1) - costs

        # print("\ndate: ", date)
        # print("yday's price: ", yesterday_price.loc[date].iloc[0])
        # print("today's price: ", price.loc[date].iloc[0])
        # print("action: ", action)
        # print("holdings before: ", holdings)
        # print("reward: ", reward.iloc[0])

        if action == 1:
            holdings = 1000
        elif action == 2:
            holdings = -1000

        # print("holdings after: ", holdings)

        return reward, holdings

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
        # bbp, rsi, macd_crossover = self.get_indicators(symbol, sd, ed)
        bbp, rsi, ppo = self.get_indicators(symbol, sd, ed)
        price = self.get_price([symbol], sd, ed)
        scores = []
        for count in range(0, self.epochs):
            holdings = 0
            state = int(self.discretize(bbp.iloc[0], rsi.iloc[0], ppo.iloc[0]))
            self.statespace[state] += 1
            action = self.learner.querysetstate(state)
            signals = pd.DataFrame(0, index=price.index, columns=[symbol])
            signals.iloc[0] = action
            # portvals = pd.DataFrame(0, index=price.index, columns=["portvals"])

            for date in price.index.tolist()[1:]:
                signals.loc[date, symbol] = action
                reward, holdings = self.get_reward(action, date, price, holdings) # Accommodate for reward & commission
                reward = reward.loc[symbol]
                # reward = ((holdings[0] * price.loc[date] + holdings[1]) / sv - 1) * 100

                if bbp.loc[date] >= 80:
                    pass

                state = int(self.discretize(bbp.loc[date], rsi.loc[date], ppo.loc[date]))
                self.statespace[state] += 1
                action = self.learner.query(state, reward)
                # portvals.loc[date].iloc[0] = holdings * price.loc[date] + holdings[1]

            trades = self.get_trades(signals, symbol)
            pv_ms = compute_portvals(orders_df=trades, start_val=sv, commission=self.commission, impact=self.impact)
            scores.append(compute_stats(pv_ms)[0])

            if self.verbose:
                print("\nEPOCH: ", count)
                print("\nCum Ret in this EPOCH: ", scores[-1])
                print("\nLast 3 Scores & Count: ", scores[-3:], count)
                print("\nrar: ", self.learner.rar)
                print("\n")

            if len(scores) >= 3 and (abs(scores[-2]/scores[-1] - 1) <= 0.000001) and (abs(scores[-3]/scores[-2] - 1) <= 0.000001):
                self.add_evidence_trades = trades
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
        # print("\nq1-table coverage: ", (self.learner.q == 0).sum().sum()/self.learner.q.size)
        # print("\nmax: ", self.learner.q.max())
        # print("\nmin: ", self.learner.q.min())
        # print("\nmedian: ", np.median(self.learner.q))
        # print("\nmean: ", self.learner.q.mean())
        # bbp, rsi, macd_crossover = self.get_indicators(symbol, sd, ed)
        bbp, rsi, ppo = self.get_indicators(symbol, sd, ed)
        price = self.get_price([symbol], sd, ed)
        state = int(self.discretize(bbp.iloc[0], rsi.iloc[0], ppo.iloc[0]))
        action = self.learner.querysetstate(state)
        signals = pd.DataFrame(0, index=price.index, columns=[symbol])
        signals.iloc[0] = action

        for date in price.index.tolist()[1:]:
            signals.loc[date, symbol] = action
            state = int(self.discretize(bbp.loc[date], rsi.loc[date], ppo.loc[date]))
            action = self.learner.querysetstate(state)

        # print(signals)
        return self.get_trades(signals, symbol)

  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		 	 	 			  		 			     			  	 
