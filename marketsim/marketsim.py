""""""  		  	   		 	 	 			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
import os  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data
# from optimization import compute_stats

def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"
  		  	   		 	 	 			  		 			     			  	 
def compute_portvals(  		  	   		 	 	 			  		 			     			  	 
    orders_file="./orders/orders-01.csv",
    start_val=1000000,  		  	   		 	 	 			  		 			     			  	 
    commission=9.95,  		  	   		 	 	 			  		 			     			  	 
    impact=0.005,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Computes the portfolio values.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 	 	 			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 	 	 			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
    :type start_val: int  		  	   		 	 	 			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	 	 			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is the function the autograder will call to test your code  		  	   		 	 	 			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	 	 			  		 			     			  	 
    # code should work correctly with either input
    orders = pd.read_csv(orders_file, index_col = 'Date', parse_dates = True, na_values = ['nan'])
    symbols = orders['Symbol'].unique()
    start_date = orders.index.min() # starting date is the earliest date in orders
    end_date = orders.index.max() # ending date is the max date in orders

    prices = get_data(symbols, pd.date_range(start_date, end_date)) # get price data
    date_range = prices.index.tolist() # get all trading days from start date to end date
    prices['CASH'] = np.ones(len(date_range))  # add price of 1 dollar
    print("PRICES:", prices)

    # Add CASH to symbols for further dataframes
    df_columns = np.append(symbols.copy(), "CASH")  # https://www.w3schools.com/python/python_lists_copy.asp

    # Initialize obligation, holdings and portvals
    obligation = pd.DataFrame(0, index=date_range, columns=df_columns)
    holdings = pd.DataFrame(0, index=date_range, columns=df_columns)

    # Add an extra row for starting value of cash in holdings
    minus_date = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    holdings.loc[minus_date] = np.append(np.zeros(len(symbols)), start_val)

    # ensure all dfs are sorted
    holdings = holdings.sort_index()
    obligation = obligation.sort_index()

    # iterate across dates and fill obligation and holdings based on orders data
    for date in date_range:
        if date not in orders.index:
            pass
        else:
            # print(day_orders, day_orders.empty)
            day_orders = orders.loc[[pd.to_datetime(date)]]
            # print(day_orders)
            # print(day_orders.shape, day_orders.dtypes)
            # print("\n\n")
            for index, order in day_orders.iterrows():
                # print(order)
                if order['Order'] == 'BUY':
                    obligation.loc[date, order['Symbol']] += order['Shares']
                    obligation.loc[date, 'CASH'] -= order['Shares'] * (prices.loc[date, order['Symbol']]*(1 + impact)) + commission
                else:
                    obligation.loc[date, order['Symbol']] -= order['Shares']
                    obligation.loc[date, 'CASH'] += order['Shares'] * (prices.loc[date, order['Symbol']]*(1 - impact)) - commission

        if holdings.shift(1).loc[date].isnull().values.any():
            pass
        holdings.loc[date] = holdings.shift(1).loc[date] + obligation.loc[date]

    interim = holdings*prices
    portvals = interim.sum(axis=1)
    portvals = portvals.drop(minus_date)

    # print(obligation)
    # print("\n\n")
    # print(holdings)
    # print("\n\n")
    # print(interim)
    # print("\n\n")
    # print(portvals)

    return portvals
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Helper function to test code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			     			  	 
    # Define input parameters  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    of = "./additional_orders/orders.csv"
    sv = 1000000  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Process orders  		  	   		 	 	 			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	 	 			  		 			     			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 			  		 			     			  	 
    else:  		  	   		 	 	 			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Get portfolio stats  		  	   		 	 	 			  		 			     			  	 
    # Here we just fake the data. you should use your code from previous assignments.
    orders = pd.read_csv(of, index_col='Date', parse_dates=True, na_values=['nan'])
    symbols = orders['Symbol'].unique()
    start_date = orders.index.min()  # starting date is the earliest date in orders
    end_date = orders.index.max()  # ending date is the max date in orders
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		 	 	 			  		 			     			  	 
        0.2,  		  	   		 	 	 			  		 			     			  	 
        0.01,  		  	   		 	 	 			  		 			     			  	 
        0.02,  		  	   		 	 	 			  		 			     			  	 
        1.5,  		  	   		 	 	 			  		 			     			  	 
    ]  		  	   		 	 	 			  		 			     			  	 

    # Compare portfolio against $SPX  		  	   		 	 	 			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
