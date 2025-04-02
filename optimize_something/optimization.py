""""""  		  	   		 	 	 			  		 			     			  	 
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data
import math
import scipy.optimize as spo

def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			     			  	 
# The student must update this code to properly implement the functionality  		  	   		 	 	 			  		 			     			  	 
def optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	 	 			  		 			     			  	 
    gen_plot=False,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			     			  	 
    statistics.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
    :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
    :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			     			  	 
        symbol in the data directory)  		  	   		 	 	 			  		 			     			  	 
    :type syms: list  		  	   		 	 	 			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			     			  	 
        code with gen_plot = False.  		  	   		 	 	 			  		 			     			  	 
    :type gen_plot: bool  		  	   		 	 	 			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			     			  	 
    :rtype: tuple  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
  		  	   		 	 	 			  		 			     			  	 
    # find the allocations for the optimal portfolio  		  	   		 	 	 			  		 			     			  	 
    n = len(syms)
    init_allocs = np.full(n, 1/n)

    # define constrains & bounds for optimize - https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.optimize.minimize.html
    cons = {'type':'eq', 'fun': lambda allocs: np.sum(allocs)-1} # the sum of allocations should be 1
    bnds = ((0,1),)*n # for each stock, the bounds are 0 to 1

    # call scipy optimize
    result = spo.minimize(
        get_sharpe_for_optimize,
        init_allocs,
        args=(prices, ),
        method='SLSQP',
        bounds=bnds,
        constraints=cons,
        options = {'disp':True }
    )

    allocs = result.x

    port_val = get_port_val(allocs, prices)
    cr, adr, sddr, sr = compute_stats(port_val) # add code here to compute stats

    # Get daily portfolio value
    normed_SPY = prices_SPY/prices_SPY.iloc[0]  # add code here to compute daily portfolio values of SPY
  		  	   		 	 	 			  		 			     			  	 
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	 	 			  		 			     			  	 
    if gen_plot:  		  	   		 	 	 			  		 			     			  	 
        # add code to plot here  		  	   		 	 	 			  		 			     			  	 
        df_temp = pd.concat(  		  	   		 	 	 			  		 			     			  	 
            [port_val, normed_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        ax = df_temp['Portfolio'].plot(title='Daily Portfolio Value and SPY', label = 'Portfolio')
        df_temp['SPY'].plot(label='SPY', ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left')

        # add watermark (reference: https://labex.io/tutorials/python-add-watermark-to-matplotlib-plot-49030):
        # ax.text(0.5, 0.5, 'sphadnis9@gatech.edu',
        #         transform=ax.transAxes, fontsize=30, color='gray',
        #         alpha=0.5, ha='center', va='center')

        ax.grid()
        plt.savefig('images/Figure1.png')
        plt.close()
        pass  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    return allocs, cr, adr, sddr, sr  		  	   		 	 	 			  		 			     			  	 

def get_sharpe_for_optimize(allocs, prices):
    port_val = get_port_val(allocs, prices)
    cr, adr, sddr, sr = compute_stats(port_val)
    return -sr # we need to return -sharpe ratio so that minimizer will give us the max sharpe ratio

def get_port_val(allocs, prices):
    # calculate portfolio value per day as per allocs
    normalised = prices/prices.iloc[0]
    allocated = normalised * allocs
    portfolio_value = allocated.sum(axis=1)

    return portfolio_value

def compute_stats(portfolio_value):
    daily_return = get_daily_return(portfolio_value)
    cr = (portfolio_value[-1] / portfolio_value[0]) - 1
    adr = daily_return.mean()
    sddr = daily_return.std()
    sr = math.sqrt(252) * adr / sddr # k=252 for daily samples; risk free rate = 0

    return cr, adr, sddr, sr

def get_daily_return(portfolio_value):
    daily_return = (portfolio_value/portfolio_value.shift(1)) - 1
    daily_return.iloc[0] = 0
    return daily_return
  		  	   		 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # start_date = dt.datetime(2009, 1, 1)
    # end_date = dt.datetime(2010, 1, 1)
    # symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]


# Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Print statistics  		  	   		 	 	 			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return: {cr}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			     			  	 
    # Do not assume that it will be called  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
