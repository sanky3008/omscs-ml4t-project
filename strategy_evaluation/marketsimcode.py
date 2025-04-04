import numpy as np
from util import get_data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import math

def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"

def compute_portvals(
        orders_df,
        start_val=100000,
        commission=0,
        impact=0,
):
    orders = orders_df
    symbols = orders.columns
    symbol = symbols[0]
    start_date = orders.index.min()  # starting date is the earliest date in orders
    end_date = orders.index.max()  # ending date is the max date in orders

    prices = get_data(symbols, pd.date_range(start_date, end_date))  # get price data
    date_range = prices.index.tolist()  # get all trading days from start date to end date
    prices['CASH'] = np.ones(len(date_range))  # add price of 1 dollar
    # print("PRICES:", prices)

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
    for index, row in obligation.iterrows():
        date = index
        if date not in orders.index:
            pass
        else:
            day_order = orders.loc[date, symbol]
            if day_order > 0:
                obligation.loc[date, symbol] += np.abs(day_order)
                obligation.loc[date, 'CASH'] -= day_order*(prices.loc[date,symbol]*(1+impact)) + commission
            elif day_order < 0:
                obligation.loc[date, symbol] -= np.abs(day_order)
                obligation.loc[date, 'CASH'] += np.abs(day_order) * (prices.loc[date, symbol] * (1 - impact)) - commission

        if holdings.shift(1).loc[date].isnull().values.any():
            pass
        holdings.loc[date] = holdings.shift(1).loc[date] + obligation.loc[date]

    # print(holdings)
    interim = holdings * prices
    portvals = interim.sum(axis=1)
    portvals = portvals.drop(minus_date)

    return portvals

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