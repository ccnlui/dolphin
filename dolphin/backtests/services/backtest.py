#        File: backtest.py
#        Date: 2021-10-04
#          By: Calvin
#       Brief: Backtest app.
# Requirement: Python 3.8

from backtests.services.market_data import get_symbol_list_daily_split_adjusted_df_list, get_daily_split_adjusted_df
from backtests.constants import cboe_holidays

from numba import jit
from numba.typed import Dict
from numba.typed import List
from numba.core import types

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import stats

from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta

from collections import namedtuple
import code
import os

import empyrical as em
import pyfolio.timeseries as pt

import multiprocessing as mp

import csv


class pandas_algo_turtle(object):
    #--------------------------------------------------------------------------
    # Parameters.
    #--------------------------------------------------------------------------
    TURTLE_PERIOD_ENTRY = 20
    TURTLE_PERIOD_EXIT = 10
    ATR_PERIOD = 20
    VOL_PERIOD = 20
    MOMENTUM_WINDOW = 125
    YEARLY_TRADING_DAYS = 252
    PORTFOLIO_NUM_STOCK = 10
    SINGLE_DAY_VOLATILITY_FILTER_DAYS = 90
    SINGLE_DAY_VOLATILITY_FILTER_PCT = 15

    INITIAL_CAPITAL = 10000

    # Replaced by inverse volatility.
    DOLLAR_VOL_PCT = 2

    # Exponential model: y = a * exp(b*x)
    EXP_MODEL_GUESS_A = 4
    EXP_MODEL_GUESS_B = 0.1

    # Round ATR.
    MODEL_PRECISION = 4

    #--------------------------------------------------------------------------
    # Variables.
    #--------------------------------------------------------------------------
    START_DATE = "2010-01-01"
    END_DATE = "2020-12-31"
    INTERVAL = "1day"

    VERBOSE = True

    #--------------------------------------------------------------------------
    # Constants.
    #--------------------------------------------------------------------------
    MARKET_DATA_ROOT_PATH = "/home/calvin/source/data/alpha_vantage/backup/2021-01-16/data"
    GRAPH_ROOT_PATH       = "/home/calvin/source/python/pandas/graph"
    LOG_ROOT_PATH         = "/home/calvin/source/python/pandas/log"
    CSV_ROOT_PATH         = "/home/calvin/source/python/pandas/csv"
    LISTING_ROOT_PATH     = "/home/calvin/source/data/alpha_vantage/listing"

    REGULAR_TRADING_HOURS_START_TIME = time.fromisoformat("09:30:00")
    REGULAR_TRADING_HOURS_END_TIME = time.fromisoformat("16:00:00")

    ATR_SMOOTHING_FACTOR = 1 / ATR_PERIOD

    #--------------------------------------------------------------------------
    # Constructor.
    #--------------------------------------------------------------------------
    def __init__(self):
        #----------------------------------------------------------------------
        # Members.
        #----------------------------------------------------------------------
        # self.symbol_universe = ["AAPL", "AMD", "NVDA"]
        # self.symbol_universe = ["AAPL", "FB", "AMZN", "GOOGL", "TSLA"]
        # self.symbol_universe = ["AAPL", "AMD", "NVDA", "PTON", "FSLY", "OSTK", "BIGC", "SHOP"]
        # self.symbol_universe = ["XELB", "ACS", "CODA", "AAPL", "AMD", "NVDA"]
        # self.symbol_universe = ["CODA"]
        self.symbol_universe = os.listdir(self.MARKET_DATA_ROOT_PATH)
        self.symbol_universe.sort()
        if "raw" in self.symbol_universe:
            self.symbol_universe.remove('raw')

        self.curr_split_factor = None

        self.curr_date = None
        self.prev_date = None

        self.df = None

        return

    #--------------------------------------------------------------------------
    # Methods.
    #--------------------------------------------------------------------------
    def load_market_data(self, symbol_universe, start_date_str, end_date_str, interval):
        #----------------------------------------------------------------------
        # Initialize dates.
        #----------------------------------------------------------------------
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        prefetch_start_date = start_date - relativedelta(months=6)

        start_year = prefetch_start_date.year
        end_year = end_date.year
        curr_year = start_year
        next_year = None

        self.df = pd.DataFrame()

        #----------------------------------------------------------------------
        # Read csv file for each symbol.
        #----------------------------------------------------------------------
        for symbol in self.symbol_universe:

            df_symbol = pd.DataFrame()

            if interval == "1day":
                #------------------------------------------------------------------------------
                # Load price data one year at a time.
                #------------------------------------------------------------------------------
                curr_year = start_year

                while curr_year <= end_year:

                    next_year = curr_year + 1

                    #------------------------------------------------------------------
                    # Check if historical data exists.
                    #------------------------------------------------------------------
                    symbol_data_filepath = "{}/{}/{}/Alpha_Vantage_{}_{}_{}_adjusted.csv".format(
                                            self.MARKET_DATA_ROOT_PATH,
                                            symbol,
                                            interval,
                                            symbol,
                                            interval,
                                            curr_year)
                    if not os.path.exists(symbol_data_filepath):
                        print("[ERROR] {} does not exist.".format(symbol_data_filepath))
                        # Next iteration.
                        curr_year = next_year
                        continue


                    #------------------------------------------------------------------------------
                    # Build dataframe.
                    #------------------------------------------------------------------------------
                    # Read current year csv.
                    df_curr_year = pd.read_csv(symbol_data_filepath)

                    # Filter date range.
                    df_curr_year = df_curr_year.loc[ (df_curr_year.timestamp >= prefetch_start_date.isoformat()) & (df_curr_year.timestamp <= end_date.isoformat()) ]

                    # Concatenate current year dataframe.
                    df_symbol = pd.concat([df_symbol, df_curr_year], ignore_index=True)


                    #--------------------------------------------------------------------------
                    # Next iteration.
                    #--------------------------------------------------------------------------
                    curr_year = next_year

                #--------------------------------------------------------------------------
                # Calculate split-adjusted close.
                #--------------------------------------------------------------------------
                if not df_symbol.empty:
                    total_split_coefficient = df_symbol.split_coefficient.product()

                    df_symbol["symbol"] = symbol
                    df_symbol["split_coefficient_inv"] = df_symbol.split_coefficient.rdiv(1)
                    df_symbol["split_coefficient_inv_cum"] = df_symbol.split_coefficient_inv.cumprod() * total_split_coefficient
                    df_symbol["split_adjusted_open"] = df_symbol.open / df_symbol.split_coefficient_inv_cum
                    df_symbol["split_adjusted_high"] = df_symbol.high / df_symbol.split_coefficient_inv_cum
                    df_symbol["split_adjusted_low"] = df_symbol.low / df_symbol.split_coefficient_inv_cum
                    df_symbol["split_adjusted_close"] = df_symbol.close / df_symbol.split_coefficient_inv_cum

                    #--------------------------------------------------------------------------
                    # Append symbol dataframe.
                    #--------------------------------------------------------------------------
                    # Keep useful columns, drop the rest.
                    df_symbol = df_symbol.loc[:, ["timestamp", "symbol", "split_adjusted_open", "split_adjusted_high", "split_adjusted_low", "split_adjusted_close"]]

                    # Rename column timestamp to date.
                    df_symbol = df_symbol.rename(columns={"timestamp": "date"})

                    # Append.
                    if self.df.empty:
                        self.df = self.df.append(df_symbol)
                    else:
                        # Append symbol dataframe on existing columns.
                        self.df = self.df.append(df_symbol, ignore_index=True)

                        # Append symbol dataframe as new columns.
                    # self.df = self.df.join(df_symbol.set_index("date"), on="date")
                else:
                    print("[ERROR] {} dataframe has no price data.".format(symbol))

        #--------------------------------------------------------------------------
        # Convert date column to type numpy datetime64.
        #--------------------------------------------------------------------------
        self.df.date = pd.to_datetime(self.df.date)

        return


    def load_market_data_from_db(self, symbol_universe, start_date_str, end_date_str, interval):

        print("[{}] [INFO] Loading market data from database...".format(datetime.now().isoformat()))

        #----------------------------------------------------------------------
        # Initialize dates.
        #----------------------------------------------------------------------
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        prefetch_start_date = start_date - relativedelta(months=6)

        #----------------------------------------------------------------------
        # Read raw daily adjusted from database.
        #----------------------------------------------------------------------
        df_list = get_symbol_list_daily_split_adjusted_df_list(symbol_universe, prefetch_start_date.isoformat(), end_date.isoformat())

        #--------------------------------------------------------------------------
        # Generate symbol indicators.
        #--------------------------------------------------------------------------
        # for df_symbol in df_list:
        #     self.generate_symbol_indicators(df_symbol)

        #--------------------------------------------------------------------------
        # Generate symbol indicators in parallel.
        #--------------------------------------------------------------------------
        pool = mp.Pool(mp.cpu_count()-2)
        df_list = pool.map(self.generate_symbol_indicators, df_list)
        pool.close()

        #--------------------------------------------------------------------------
        # Combine all symbol dataframes together.
        #--------------------------------------------------------------------------
        self.df = pd.concat(df_list, ignore_index=True)

        #--------------------------------------------------------------------------
        # Convert date column to type numpy datetime64.
        #--------------------------------------------------------------------------
        self.df.date = pd.to_datetime(self.df.date)

        #--------------------------------------------------------------------------
        # Write to csv.
        #--------------------------------------------------------------------------
        turtle.df.to_csv("{}/algo_turtle_indicators.csv".format(turtle.CSV_ROOT_PATH), index=False)

        return


    def load_market_data_from_csv(self, csv_fullpath):

        print("[{}] [INFO] Loading market data from csv...".format(datetime.now().isoformat()))

        self.df = pd.read_csv(csv_fullpath)

        #--------------------------------------------------------------------------
        # Convert date column to type numpy datetime64.
        #--------------------------------------------------------------------------
        self.df.date = pd.to_datetime(self.df.date)

        return


    def check_listing_status(self, start_date_str, end_date_str):

        #----------------------------------------------------------------------
        # Initialize dates.
        #----------------------------------------------------------------------
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        prefetch_start_date = start_date - relativedelta(months=6)

        curr_date = prefetch_start_date

        #----------------------------------------------------------------------
        # Iterate date range.
        #----------------------------------------------------------------------
        while curr_date <= end_date:

            # Create pandas datetime object.
            curr_date_pd = pd.to_datetime(curr_date)

            #----------------------------------------------------------------------
            # Read csv file.
            #----------------------------------------------------------------------
            listing_fullpath = "{}/{}/Alpha_Vantage_{}_listing_status.csv".format(self.LISTING_ROOT_PATH, curr_date.year, curr_date)
            df_listing = pd.read_csv(listing_fullpath)

            # Check listing status.
            self.df.loc[ self.df.date == curr_date_pd, "listed" ] = self.df.loc[ self.df.date == curr_date_pd, "symbol" ].isin(df_listing.symbol)

            #------------------------------------------------------------------
            # Next iteration.
            #------------------------------------------------------------------
            next_date = curr_date + timedelta(1)
            curr_date = next_date


    def check_sp500_constituents(self, start_date_str, end_date_str):

        #----------------------------------------------------------------------
        # Initialize dates.
        #----------------------------------------------------------------------
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        prefetch_start_date = start_date - relativedelta(months=6)

        curr_date = prefetch_start_date

        #----------------------------------------------------------------------
        # Iterate date range.
        #----------------------------------------------------------------------
        while curr_date <= end_date:

            # Create pandas datetime object.
            curr_date_pd = pd.to_datetime(curr_date)

            #----------------------------------------------------------------------
            # Fetch S&P500 list.
            #----------------------------------------------------------------------
            sp500_list = get_sp500(curr_date.isoformat()).split(',')
            sp500_arr = np.array(sp500_list)

            df_sp500 = pd.DataFrame(sp500_arr, columns=["symbol"])

            # Check if symbol is in S&P500.
            self.df.loc[ self.df.date == curr_date_pd, "sp500" ] = self.df.loc[ self.df.date == curr_date_pd, "symbol" ].isin(df_sp500.symbol)

            #------------------------------------------------------------------
            # Next iteration.
            #------------------------------------------------------------------
            next_date = curr_date + timedelta(1)
            curr_date = next_date


    @staticmethod
    def momentum_score(time_series):
        # Exponential model: y = a * e^(b*x)
        # Linear model: y = a * x + b

        # Requires data of previous rolling window, 125 days by default.
        x = np.arange(pandas_algo_turtle.MOMENTUM_WINDOW)

        #----------------------------------------------------------------------
        # Sanity check: Scipy curve fit.
        #----------------------------------------------------------------------
        try:
            popt, pcov = curve_fit(lambda x, a, b: a*np.exp(b*x), x, time_series, p0=(pandas_algo_turtle.EXP_MODEL_GUESS_A, pandas_algo_turtle.EXP_MODEL_GUESS_B))
            a = popt[0]
            b = popt[1]
            fitted_curve = lambda x: a*np.exp(b*x)

            #----------------------------------------------------------------------
            # Sanity check: Numpy polyfit with weights.
            # Log exponential model: ln y = ln a + b*x
            # Rewrite as y' = a' + b*x
            #----------------------------------------------------------------------
            # log_ts = np.log(time_series)
            # weights = np.sqrt(time_series)
            # b, a = np.polyfit(x, log_ts, 1, w=weights)
            # plt.figure()
            # plt.plot(x, (lambda x, a, b: np.exp(a) * np.exp(b*x))(x, a, b), label="polyfit weights")
            # plt.legend()
            # plt.show()

            #----------------------------------------------------------------------
            # Sanity check: Numpy polyfit without weights.
            #----------------------------------------------------------------------
            # log_ts = np.log(time_series)
            # weights = np.sqrt(time_series)
            # b, a = np.polyfit(x, log_ts, 1)
            # plt.figure()
            # plt.plot(x, (lambda x, a, b: np.exp(a) * np.exp(b*x))(x, a, b), label="polyfit")
            # plt.legend()
            # plt.show()

            #----------------------------------------------------------------------
            # Calculate coefficient of determination.
            #----------------------------------------------------------------------
            y = np.fromfunction(fitted_curve, x.shape)

            ss_res = np.sum((time_series - y) ** 2)
            ss_tot = np.sum((time_series - np.mean(time_series)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            #----------------------------------------------------------------------
            # Momentum score.
            #----------------------------------------------------------------------
            if r2 < 0:
                # Discard regression result where coefficient of determination is negative.
                score = np.nan
            else:
                score = (np.power(1+b, pandas_algo_turtle.YEARLY_TRADING_DAYS) - 1) * 100 * r2

            #----------------------------------------------------------------------
            # Sanity check: Animate the change of curve.
            #----------------------------------------------------------------------
            # fig = plt.figure()
            # plt.plot(x, time_series, '.', label="close")
            # plt.plot(x, (lambda x, a, b: a * np.exp(b*x))(x, a, b), label="curve fit")
            # plt.legend()
            # plt.title("a:{:.2f} b:{:.4f} r2:{:.2f} momentum:{:.2f}".format(a, b, r2, score))
            # plt.show()
            # plt.close(fig)

            return score
        except:
            return np.nan


    #--------------------------------------------------------------------------
    # Assumption:
    #   1. A symbol can only have at max 1 signal at any point in time
    #   2. Margin account that allows negative cash.
    #--------------------------------------------------------------------------
    @staticmethod
    # @jit(nopython=True)
    def simulate_trading(date, symbol, split_adjusted_open, split_adjusted_close, close_entry_rolling_max, close_exit_rolling_min, atr, turtle_rank, weights, initial_capital, start_date, end_date, portfolio_num_stock, verbose):
        #--------------------------------------------------------------------------
        # Initialize columns.
        #--------------------------------------------------------------------------
        length = date.shape[0]

        # Trade specific columns.
        stop_loss = np.full(length, np.nan)
        last_fill = np.full(length, np.nan)
        unit_cnt_long = np.zeros(length, dtype=np.uint32)
        unit_qty_long = np.zeros(length, dtype=np.uint32)

        cashflow = np.zeros(length)
        book_value = np.zeros(length)
        market_value = np.zeros(length)
        avg_price = np.zeros(length)
        trade_pnl = np.zeros(length)

        # Account specific columns.
        cash = np.zeros(length)
        equity = np.zeros(length)
        account_pnl = np.zeros(length)

        cash_bod = np.zeros(length)
        equity_bod = np.zeros(length)
        account_pnl_bod = np.zeros(length)

        #--------------------------------------------------------------------------
        # Initialize variables.
        #--------------------------------------------------------------------------
        curr_date = None

        # Running account variables.
        daily_cash = initial_capital
        daily_equity = daily_cash
        daily_account_pnl = 0

        daily_equity_bod = 0
        daily_cash_bod = 0
        daily_account_pnl_bod = 0

        #--------------------------------------------------------------------------
        # Keep track of symbol variables.
        #--------------------------------------------------------------------------
        symbol_prev_idx = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        symbol_curr_idx = Dict.empty(key_type=types.unicode_type, value_type=types.int64)

        symbol_entry_signal = []
        symbol_exit_signal = []
        portfolio_symbol = []
        rebalance = False

        #--------------------------------------------------------------------------
        # Process tick data.
        #--------------------------------------------------------------------------
        for idx in range(0, length):

            #------------------------------------------------------------------
            # Show progress.
            #------------------------------------------------------------------
            if date[idx] != curr_date:
                print("[DEBUG] Processing date {}...".format(date[idx]))

            #------------------------------------------------------------------
            # Reset.
            #------------------------------------------------------------------
            trading = False

            curr_date = date[idx]

            #------------------------------------------------------------------
            # Read in symbol index.
            #------------------------------------------------------------------
            # Convert unichr array into unicode strings.
            symbol_str = str(symbol[idx])
            symbol_curr_idx[symbol_str] = idx

            #------------------------------------------------------------------
            # Trading.
            #------------------------------------------------------------------
            # Read in all symbols for a date before trading.
            if idx+1 == length or date[idx+1] > curr_date:
                trading = True

            if trading:
                #------------------------------------------------------------------
                # First pass: Assume open position liquidated when halted or delisted.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol:

                    if curr_symbol not in symbol_curr_idx:

                        # Get previous symbol index.
                        prev_idx = symbol_prev_idx[curr_symbol]

                        # Assume trade liquidated at previous close.
                        liquidated_cashflow = split_adjusted_close[prev_idx] * unit_qty_long[prev_idx]
                        liquidated_book_value = book_value[prev_idx] - liquidated_cashflow

                        # Close remaining book value as trade profit and loss.
                        liquidated_trade_pnl = liquidated_book_value * -1
                        liquidated_book_value = 0

                        # Keep track of running account variables.
                        daily_cash += liquidated_cashflow
                        daily_account_pnl = daily_equity - initial_capital

                        # Remove from portfolio.
                        portfolio_symbol.remove(curr_symbol)

                        # Remove from entry and exit maps.
                        if curr_symbol in symbol_entry_signal:
                            symbol_entry_signal.remove(curr_symbol)
                        if curr_symbol in symbol_exit_signal:
                            symbol_exit_signal.remove(curr_symbol)

                        print("------------------------------------------------")
                        print("[WARNING] Liquidated trade: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                                curr_date,
                                curr_symbol,
                                0,
                                split_adjusted_close[prev_idx],
                                liquidated_cashflow,
                                liquidated_book_value,
                                0,
                                0,
                                daily_cash,
                                daily_equity,
                                daily_account_pnl,
                                liquidated_trade_pnl))
                        print("------------------------------------------------")

                #------------------------------------------------------------------
                # Second pass: Mark-to-market at BOD.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol:

                    # Get symbol index.
                    curr_idx = symbol_curr_idx[curr_symbol]
                    prev_idx = symbol_prev_idx[curr_symbol]

                    # Carry over existing trade columns.
                    stop_loss[curr_idx] = stop_loss[prev_idx]
                    last_fill[curr_idx] = last_fill[prev_idx]
                    unit_cnt_long[curr_idx] = unit_cnt_long[prev_idx]
                    unit_qty_long[curr_idx] = unit_qty_long[prev_idx]
                    
                    book_value[curr_idx] = book_value[prev_idx]
                    avg_price[curr_idx] = avg_price[prev_idx]

                    # Mark-to-market at BOD.
                    market_value[curr_idx] = unit_qty_long[curr_idx] * split_adjusted_open[curr_idx]
                    trade_pnl[curr_idx] = market_value[curr_idx] - book_value[curr_idx]
                    
                    # Keep track of running account variables.
                    daily_equity = daily_equity - market_value[prev_idx] + market_value[curr_idx]
                    daily_account_pnl = daily_equity - initial_capital

                    print("[DEBUG] BOD Mark-to-market: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                            curr_date,
                            curr_symbol,
                            unit_qty_long[curr_idx],
                            split_adjusted_open[curr_idx],
                            cashflow[curr_idx],
                            book_value[curr_idx],
                            avg_price[curr_idx],
                            market_value[curr_idx],
                            daily_cash,
                            daily_equity,
                            daily_account_pnl,
                            trade_pnl[curr_idx]))

                #------------------------------------------------------------------
                # Third pass: Exit trades.
                #------------------------------------------------------------------
                for curr_symbol in symbol_exit_signal:

                    # Get symbol index.
                    curr_idx = symbol_curr_idx[curr_symbol]
                    prev_idx = symbol_prev_idx[curr_symbol]

                    # Exit trade.
                    stop_loss[curr_idx] = np.nan
                    last_fill[curr_idx] = split_adjusted_open[curr_idx]
                    unit_cnt_long[curr_idx] = 0
                    unit_qty_long[curr_idx] = 0

                    cashflow[curr_idx] = split_adjusted_open[curr_idx] * unit_qty_long[prev_idx]
                    book_value[curr_idx] = book_value[prev_idx] - cashflow[curr_idx]
                    avg_price[curr_idx] = 0
                    market_value[curr_idx] = 0

                    # Close remaining book value as trade profit and loss.
                    trade_pnl[curr_idx] = book_value[curr_idx] * -1
                    book_value[curr_idx] = 0

                    # Keep track of running account variables.
                    daily_cash += cashflow[curr_idx]
                    daily_account_pnl = daily_equity - initial_capital

                    # Remove from portfolio.
                    portfolio_symbol.remove(curr_symbol)

                    print("         [INFO] Exit trade: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                            curr_date,
                            curr_symbol,
                            unit_qty_long[curr_idx],
                            split_adjusted_open[curr_idx],
                            cashflow[curr_idx],
                            book_value[curr_idx],
                            avg_price[curr_idx],
                            market_value[curr_idx],
                            daily_cash,
                            daily_equity,
                            daily_account_pnl,
                            trade_pnl[curr_idx]))

                #------------------------------------------------------------------
                # Fourth pass: Rebalance existing position.
                #------------------------------------------------------------------
                if rebalance:

                    for curr_symbol in portfolio_symbol:

                        # Get symbol index.
                        curr_idx = symbol_curr_idx[curr_symbol]
                        prev_idx = symbol_prev_idx[curr_symbol]

                        # Rebalance trade.
                        target_qty_long = np.floor(daily_equity * weights[prev_idx] / split_adjusted_open[curr_idx])
                        delta_qty_long = target_qty_long - unit_qty_long[curr_idx]

                        # Add to position.
                        if delta_qty_long > 0:

                            last_fill[curr_idx] = split_adjusted_open[curr_idx]
                            unit_qty_long[curr_idx] += delta_qty_long

                            cashflow[curr_idx] = split_adjusted_open[curr_idx] * delta_qty_long * -1
                            book_value[curr_idx] -= cashflow[curr_idx]
                            avg_price[curr_idx] = book_value[curr_idx] / unit_qty_long[curr_idx]
                            market_value[curr_idx] -= cashflow[curr_idx]
                        
                        # Remove from position.
                        if delta_qty_long < 0:

                            last_fill[curr_idx] = split_adjusted_open[curr_idx]
                            unit_qty_long[curr_idx] += delta_qty_long

                            # Remove entire position.
                            if unit_qty_long[curr_idx] == 0:

                                stop_loss[curr_idx] = np.nan
                                unit_cnt_long[curr_idx] = 0

                                cashflow[curr_idx] = split_adjusted_open[curr_idx] * unit_qty_long[prev_idx]
                                book_value[curr_idx] = book_value[prev_idx] - cashflow[curr_idx]
                                avg_price[curr_idx] = 0
                                market_value[curr_idx] = 0

                                # Close remaining book value as trade profit and loss.
                                trade_pnl[curr_idx] = book_value[curr_idx] * -1
                                book_value[curr_idx] = 0

                                # Remove from portfolio.
                                portfolio_symbol.remove(curr_symbol)
                            else:
                                cashflow[curr_idx] = split_adjusted_open[curr_idx] * delta_qty_long * -1
                                book_value[curr_idx] -= cashflow[curr_idx]
                                avg_price[curr_idx] = book_value[curr_idx] / unit_qty_long[curr_idx]
                                market_value[curr_idx] -= cashflow[curr_idx]

                                # Keep track of trade profit and loss.
                                trade_pnl[curr_idx] = market_value[curr_idx] - book_value[curr_idx]
                    
                        # Keep track of running account variables.
                        daily_cash += cashflow[curr_idx]
                        daily_account_pnl = daily_equity - initial_capital

                        print("          [INFO] Rebalance: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                                curr_date,
                                curr_symbol,
                                unit_qty_long[curr_idx],
                                split_adjusted_open[curr_idx],
                                cashflow[curr_idx],
                                book_value[curr_idx],
                                avg_price[curr_idx],
                                market_value[curr_idx],
                                daily_cash,
                                daily_equity,
                                daily_account_pnl,
                                trade_pnl[curr_idx]))

                daily_equity_bod = daily_equity
                daily_cash_bod = daily_cash
                daily_account_pnl = daily_account_pnl_bod

                #------------------------------------------------------------------
                # Fifth pass: Enter trades.
                #------------------------------------------------------------------
                for curr_symbol in symbol_entry_signal:

                    if curr_symbol not in symbol_curr_idx:
                        print("------------------------------------------------")
                        print("[WARNING] Invalid entry signal for non-trading symbol {}.".format(curr_symbol))
                        print("------------------------------------------------")
                        continue

                    # Get symbol index.
                    curr_idx = symbol_curr_idx[curr_symbol]
                    prev_idx = symbol_prev_idx[curr_symbol]

                    target_qty_long = np.floor(daily_equity * weights[prev_idx] / split_adjusted_open[curr_idx])

                    # Enter trade.
                    if target_qty_long > 0:

                        stop_loss[curr_idx] = split_adjusted_open[curr_idx] - 2*atr[prev_idx]
                        last_fill[curr_idx] = split_adjusted_open[curr_idx]
                        unit_cnt_long[curr_idx] = unit_cnt_long[prev_idx] + 1
                        unit_qty_long[curr_idx] = target_qty_long

                        cashflow[curr_idx] = split_adjusted_open[curr_idx] * unit_qty_long[curr_idx] * -1
                        book_value[curr_idx] = book_value[prev_idx] - cashflow[curr_idx]
                        avg_price[curr_idx] = book_value[curr_idx] / unit_qty_long[curr_idx]
                        market_value[curr_idx] = book_value[curr_idx]

                        # Keep track of running account variables.
                        daily_cash += cashflow[curr_idx]
                        daily_account_pnl = daily_equity - initial_capital

                        # Add to portfolio if not exist.
                        if portfolio_symbol.count(curr_symbol) == 0:
                            portfolio_symbol.append(curr_symbol)

                        print("        [INFO] Enter trade: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                                curr_date,
                                curr_symbol,
                                unit_qty_long[curr_idx],
                                split_adjusted_open[curr_idx],
                                cashflow[curr_idx],
                                book_value[curr_idx],
                                avg_price[curr_idx],
                                market_value[curr_idx],
                                daily_cash,
                                daily_equity,
                                daily_account_pnl,
                                trade_pnl[curr_idx]))
                    else:
                        print("------------------------------------------------")
                        print("[WARNING] Buying symbol {} with target quantity 0 on {}. ({} * {} / {})".format(curr_symbol, date[curr_idx], daily_equity, weights[prev_idx], split_adjusted_open[curr_idx]))
                        print("------------------------------------------------")

                #------------------------------------------------------------------
                # Sixth pass: Mark-to-market at EOD.
                #------------------------------------------------------------------
                # Reset.
                daily_equity = daily_cash
                daily_account_pnl = 0

                for curr_symbol in portfolio_symbol:

                    # Get symbol index.
                    curr_idx = symbol_curr_idx[curr_symbol]
                    prev_idx = symbol_prev_idx[curr_symbol]

                    # Mark-to-market at EOD.
                    market_value[curr_idx] = unit_qty_long[curr_idx] * split_adjusted_close[curr_idx]
                    trade_pnl[curr_idx] = market_value[curr_idx] - book_value[curr_idx]
                    
                    # Keep track of running account variables.
                    daily_equity += market_value[curr_idx]

                daily_account_pnl = daily_equity - initial_capital

                #------------------------------------------------------------------
                # Seventh pass: Process all symbols read in a day.
                #------------------------------------------------------------------
                # Reset.
                symbol_entry_signal.clear()
                symbol_exit_signal.clear()
                rebalance = False

                for curr_symbol, curr_idx in symbol_curr_idx.items():

                    cash_bod[curr_idx] = daily_cash_bod
                    equity_bod[curr_idx] = daily_equity_bod
                    account_pnl_bod[curr_idx] = daily_account_pnl_bod

                    # Fill daily account columns.
                    cash[curr_idx] = daily_cash
                    equity[curr_idx] = daily_equity
                    account_pnl[curr_idx] = daily_account_pnl

                    # Process date range only.
                    if start_date <= date[curr_idx] and date[curr_idx] <= end_date:

                        # Entry and exit signals.
                        if split_adjusted_close[curr_idx] >= close_entry_rolling_max[curr_idx] and unit_cnt_long[curr_idx] == 0 and turtle_rank[curr_idx] <= portfolio_num_stock:
                            symbol_entry_signal.append(curr_symbol)
                        elif split_adjusted_close[curr_idx] <= close_exit_rolling_min[curr_idx] and unit_cnt_long[curr_idx] > 0:
                            symbol_exit_signal.append(curr_symbol)
                        elif split_adjusted_close[curr_idx] < stop_loss[curr_idx] and unit_cnt_long[curr_idx] > 0:
                            symbol_exit_signal.append(curr_symbol)
                        elif unit_cnt_long[curr_idx] > 0 and turtle_rank[curr_idx] > portfolio_num_stock:
                            symbol_exit_signal.append(curr_symbol)

                        # Rebalance existing entry before new entry.
                        if portfolio_symbol and curr_symbol in symbol_entry_signal:
                            rebalance = True

                    # Store symbol index.
                    symbol_prev_idx[curr_symbol] = curr_idx

                # Reset symbol index map for next day.
                symbol_curr_idx.clear()


        if verbose:
            result = [stop_loss, last_fill, unit_cnt_long, unit_qty_long, cashflow, book_value, avg_price, market_value, cash, equity, account_pnl, trade_pnl, cash_bod, equity_bod, account_pnl_bod]
        else:
            result = [stop_loss, last_fill, unit_cnt_long, unit_qty_long, cashflow, book_value, avg_price, market_value, cash, equity, account_pnl, trade_pnl]

        return result


    def generate_indicators(self, symbol):
        #--------------------------------------------------------------------------
        # Get dataframe.
        #--------------------------------------------------------------------------
        df = self.df

        #--------------------------------------------------------------------------
        # Calculate rolling max/min of close.
        #--------------------------------------------------------------------------
        # Exit.
        df.loc[ df.symbol == symbol, "close_exit_rolling_max" ] = df.loc[ df.symbol == symbol, "split_adjusted_close" ].rolling(self.TURTLE_PERIOD_EXIT).max()
        df.loc[ df.symbol == symbol, "close_exit_rolling_min" ] = df.loc[ df.symbol == symbol, "split_adjusted_close" ].rolling(self.TURTLE_PERIOD_EXIT).min()

        # Entry.
        df.loc[ df.symbol == symbol, "close_entry_rolling_max" ] = df.loc[ df.symbol == symbol, "split_adjusted_close" ].rolling(self.TURTLE_PERIOD_ENTRY).max()
        df.loc[ df.symbol == symbol, "close_entry_rolling_min" ] = df.loc[ df.symbol == symbol, "split_adjusted_close" ].rolling(self.TURTLE_PERIOD_ENTRY).min()

        #--------------------------------------------------------------------------
        # Calculate true range + ATR.
        #--------------------------------------------------------------------------
        # Range 1: High - low.
        range_1 = df.loc[ df.symbol == symbol, "split_adjusted_high" ] - df.loc[ df.symbol == symbol, "split_adjusted_low" ]

        # Range 2: High - previous close.
        range_2 = df.loc[ df.symbol == symbol, "split_adjusted_high" ] - df.loc[ df.symbol == symbol, "split_adjusted_close" ].shift(1)

        # Range 3: Previous close - low.
        range_3 = df.loc[ df.symbol == symbol, "split_adjusted_close" ].shift(1) - df.loc[ df.symbol == symbol, "split_adjusted_low" ]

        # True range.
        df.loc[ df.symbol == symbol, "true_range" ] = pd.concat([range_1, range_2, range_3], axis=1).max(axis=1)

        # Calculate ATR using exponentially moving window.
        df.loc[ df.symbol == symbol, "atr" ] = df.loc[ df.symbol == symbol, "true_range" ].ewm(alpha=self.ATR_SMOOTHING_FACTOR, min_periods=self.ATR_PERIOD).mean()

        # Inverse ATR.
        df.loc[ df.symbol == symbol, "inv_atr" ] = df.loc[ df.symbol == symbol, "atr" ].rdiv(1)

        #--------------------------------------------------------------------------
        # Exponential regression.
        #--------------------------------------------------------------------------
        try:
            df.loc[ df.symbol == symbol, "momentum_score" ] = df.loc[ df.symbol == symbol, "split_adjusted_close" ].rolling(self.MOMENTUM_WINDOW).apply(pandas_algo_turtle.momentum_score, raw=True)
        except:
            code.interact(local=locals())

        return


    def generate_symbol_indicators(self, df_symbol):

        print("[{}] [INFO] Generating indicators for symbol: {}".format(datetime.now().isoformat(), df_symbol.symbol.iloc[0]))

        #--------------------------------------------------------------------------
        # Calculate rolling max/min of close.
        #--------------------------------------------------------------------------
        # Exit.
        df_symbol["close_exit_rolling_max" ] = df_symbol["split_adjusted_close"].rolling(self.TURTLE_PERIOD_EXIT).max()
        df_symbol["close_exit_rolling_min" ] = df_symbol["split_adjusted_close"].rolling(self.TURTLE_PERIOD_EXIT).min()


        # Entry.
        df_symbol["close_entry_rolling_max"] = df_symbol["split_adjusted_close"].rolling(self.TURTLE_PERIOD_ENTRY).max()
        df_symbol["close_entry_rolling_min"] = df_symbol["split_adjusted_close"].rolling(self.TURTLE_PERIOD_ENTRY).min()

        #--------------------------------------------------------------------------
        # Calculate true range + ATR.
        #--------------------------------------------------------------------------
        # Range 1: High - low.
        range_1 = df_symbol["split_adjusted_high"] - df_symbol["split_adjusted_low"]

        # Range 2: High - previous close.
        range_2 = df_symbol["split_adjusted_high"] - df_symbol["split_adjusted_close"].shift(1)

        # Range 3: Previous close - low.
        range_3 = df_symbol["split_adjusted_close"].shift(1) - df_symbol["split_adjusted_low"]

        # True range.
        df_symbol["true_range"] = pd.concat([range_1, range_2, range_3], axis=1).max(axis=1)

        # Calculate ATR using exponentially moving window.
        df_symbol["atr"] = df_symbol["true_range"].ewm(alpha=self.ATR_SMOOTHING_FACTOR, min_periods=self.ATR_PERIOD).mean()

        # Inverse ATR.
        df_symbol["inv_atr"] = df_symbol["atr"].rdiv(1)
        df_symbol["inv_atr"] = df_symbol["inv_atr"].replace(np.inf, np.nan)

        #--------------------------------------------------------------------------
        # Calculate standard deviation.
        #--------------------------------------------------------------------------
        # df_symbol["std"] = df_symbol["split_adjusted_close"].rolling(self.VOL_PERIOD).std()
        df_symbol["std"] = df_symbol["split_adjusted_close"].pct_change().rolling(self.VOL_PERIOD).std()

        # Inverse standard deviation.
        df_symbol["inv_std"] = df_symbol["std"].rdiv(1)
        df_symbol["inv_std"] = df_symbol["inv_std"].replace(np.inf, np.nan)

        #--------------------------------------------------------------------------
        # Calculate rolling max of single day absolute percent change.
        #--------------------------------------------------------------------------
        df_symbol["abs_pct_rolling_max"] = df_symbol["split_adjusted_close"].pct_change().abs().mul(100).rolling(self.SINGLE_DAY_VOLATILITY_FILTER_DAYS).max()

        #--------------------------------------------------------------------------
        # Exponential regression.
        #--------------------------------------------------------------------------
        try:
            df_symbol["momentum_score"] = df_symbol["split_adjusted_close"].rolling(self.MOMENTUM_WINDOW).apply(pandas_algo_turtle.momentum_score, raw=True)
            df_symbol["momentum_score"] = df_symbol["momentum_score"].replace(np.inf, np.nan)
        except:
            print("[{}] [ERROR] Cannot calculate momentum score for symbol: {}.".format(datetime.now().isoformat(), df_symbol.symbol.iloc[0]))
            raise

        #--------------------------------------------------------------------------
        # Disqualify filter.
        #--------------------------------------------------------------------------
        # Disqualify symbols trading under $1.00.
        df_symbol["disqualify_penny"] = df_symbol["split_adjusted_close"] < 1

        # Disqualify symbols with a single day move exceeding 15% in the past 90 days.
        df_symbol["disqualify_volatile"] = df_symbol["abs_pct_rolling_max"] > self.SINGLE_DAY_VOLATILITY_FILTER_PCT

        return df_symbol


    def generate_trading_data(self, symbol, start_date_str, end_date_str):
        #--------------------------------------------------------------------------
        # Get dataframe.
        #--------------------------------------------------------------------------
        df = self.df

        #--------------------------------------------------------------------------
        # Generate trading data.
        #--------------------------------------------------------------------------
        # Convert column date to numpy array for numba.
        date = df.loc[ df.symbol == symbol, "date" ].to_numpy()

        start_date = np.datetime64(start_date_str)
        end_date = np.datetime64(end_date_str)

        # Convert symbol series to numpy array for numba.
        split_adjusted_open = df.loc[ df.symbol == symbol, "split_adjusted_open" ].values
        split_adjusted_close = df.loc[ df.symbol == symbol, "split_adjusted_close" ].values
        close_entry_rolling_max = df.loc[ df.symbol == symbol, "close_entry_rolling_max" ].values
        close_exit_rolling_min = df.loc[ df.symbol == symbol, "close_exit_rolling_min" ].values
        atr = df.loc[ df.symbol == symbol, "atr" ].values
        turtle_rank = df.loc[ df.symbol == symbol, "turtle_rank" ].values
        weights = df.loc[ df.symbol == symbol, "weights" ].values

        # Calculate positions, trade profit and loss.
        result = pandas_algo_turtle.simulate_trading(date, split_adjusted_open, split_adjusted_close, close_entry_rolling_max, close_exit_rolling_min, atr, turtle_rank, weights, self.INITIAL_CAPITAL, start_date, end_date)

        # Unpack result.
        df.loc[ df.symbol == symbol, "stop_loss" ] = result[0]
        df.loc[ df.symbol == symbol, "last_fill" ] = result[1]
        df.loc[ df.symbol == symbol, "unit_cnt_long" ] = result[2]
        df.loc[ df.symbol == symbol, "unit_qty_long" ] = result[3]
        df.loc[ df.symbol == symbol, "cashflow" ] = result[4]
        df.loc[ df.symbol == symbol, "book_value" ] = result[5]
        df.loc[ df.symbol == symbol, "avg_price" ] = result[6]
        df.loc[ df.symbol == symbol, "cash" ] = result[7]
        df.loc[ df.symbol == symbol, "market_value" ] = result[8]
        df.loc[ df.symbol == symbol, "equity" ] = result[9]
        df.loc[ df.symbol == symbol, "account_pnl" ] = result[10]
        df.loc[ df.symbol == symbol, "trade_pnl" ] = result[11]

        #--------------------------------------------------------------------------
        # Calculate long + short exposure data.
        #--------------------------------------------------------------------------
        df.loc[ df.symbol == symbol, "long_exposure" ] = np.where(df.loc[ df.symbol == symbol, "unit_cnt_long" ] > 0,
                                                                    df.loc[ df.symbol == symbol, "unit_cnt_long" ] * df.loc[ df.symbol == symbol,
                                                                    "split_adjusted_close"], np.nan)

        #--------------------------------------------------------------------------
        # Calculate daily return in percentage.
        #--------------------------------------------------------------------------
        df.loc[ df.symbol == symbol, "split_adjusted_close_pct" ] = df.loc[ df.symbol == symbol, "split_adjusted_close" ].pct_change()
        df.loc[ df.symbol == symbol, "algo_turtle_pct" ] = np.where(df.loc[ df.symbol == symbol, "unit_cnt_long" ] > 0,
                                                                    df.loc[ df.symbol == symbol, "split_adjusted_close_pct" ],
                                                                    0)

        #--------------------------------------------------------------------------
        # Calculate total return in percentage.
        #--------------------------------------------------------------------------
        # TODO.
        # Dollar term.
        baseline_dollar = df.loc[ df.symbol == symbol, "split_adjusted_close" ].head(1).iloc[0]

        df.loc[ (df.symbol == symbol) & (df.date >= start_date_str), "split_adjusted_close_return" ] = df.loc[ df.symbol == symbol, "split_adjusted_close_pct" ].add(1).cumprod().mul(baseline_dollar)

        #--------------------------------------------------------------------------
        # Calculate total profit and loss in dollars.
        #--------------------------------------------------------------------------
        # TODO.
        # Dollar term.
        baseline_dollar = df.loc[ (df.symbol == symbol) & (df.date >= start_date_str), "split_adjusted_close" ].head(1).iloc[0]

        df.loc[ (df.symbol == symbol) & (df.date >= start_date_str), "algo_turtle_return" ] = df.loc[ df.symbol == symbol, "algo_turtle_pct" ].add(1).cumprod().mul(baseline_dollar)
        df.loc[ (df.symbol == symbol) & (df.date >= start_date_str), "algo_turtle_equity" ] = df.loc[ df.symbol == symbol, "equity" ].div(self.INITIAL_CAPITAL).mul(baseline_dollar)

        return


    def generate_all_trading_data(self, start_date_str, end_date_str):
        #--------------------------------------------------------------------------
        # Get dataframe.
        #--------------------------------------------------------------------------
        df = self.df

        df = df.sort_values(by=["date", "symbol"])

        #--------------------------------------------------------------------------
        # Generate trading data.
        #--------------------------------------------------------------------------
        # Convert column date to numpy date array for numba.
        date = df.date.to_numpy(dtype='datetime64[D]')

        start_date = np.datetime64(start_date_str)
        end_date = np.datetime64(end_date_str)

        # Convert symbol series to numpy array for numba.
        symbol = df.symbol.values.astype(str)
        split_adjusted_open = df.split_adjusted_open.values
        split_adjusted_close = df.split_adjusted_close.values
        close_entry_rolling_max = df.close_entry_rolling_max.values
        close_exit_rolling_min = df.close_exit_rolling_min.values
        atr = df.atr.values
        turtle_rank = df.turtle_rank.values
        weights = df.weights.values

        # Calculate positions, trade profit and loss.
        result = pandas_algo_turtle.simulate_trading(date,
                                                    symbol,
                                                    split_adjusted_open,
                                                    split_adjusted_close,
                                                    close_entry_rolling_max,
                                                    close_exit_rolling_min,
                                                    atr,
                                                    turtle_rank,
                                                    weights,
                                                    self.INITIAL_CAPITAL,
                                                    start_date,
                                                    end_date,
                                                    self.PORTFOLIO_NUM_STOCK,
                                                    self.VERBOSE)

        # Testing. Delete me.
        print("[{}] [DEBUG] Add trading data to dataframe.".format(datetime.now().isoformat()))

        # Unpack result.
        df["stop_loss"] = result[0]
        df["last_fill"] = result[1]
        df["unit_cnt_long"] = result[2]
        df["unit_qty_long"] = result[3]
        df["cashflow"] = result[4]
        df["book_value"] = result[5]
        df["avg_price"] = result[6]
        df["market_value"] = result[7]
        df["cash"] = result[8]
        df["equity"] = result[9]
        df["account_pnl"] = result[10]
        df["trade_pnl"] = result[11]

        if self.VERBOSE:
            df["cash_bod"] = result[12]
            df["equity_bod"] = result[13]
            df["account_pnl_bod"] = result[14]

        """
        # TODO.
        # Move to a separate function.
        #----------------------------------------------------------------------
        # Calculate portfolio returns.
        #----------------------------------------------------------------------
        df = df.sort_index()
        for symbol in self.symbol_universe:
            #--------------------------------------------------------------------------
            # Calculate long + short exposure data.
            #--------------------------------------------------------------------------
            df.loc[ df.symbol == symbol, "long_exposure" ] = np.where(df.loc[ df.symbol == symbol, "unit_cnt_long" ] > 0,
                                                                        df.loc[ df.symbol == symbol, "unit_cnt_long" ] * df.loc[ df.symbol == symbol, "split_adjusted_close"],
                                                                        np.nan)

            #--------------------------------------------------------------------------
            # Calculate daily return in percentage.
            #--------------------------------------------------------------------------
            df.loc[ df.symbol == symbol, "split_adjusted_close_pct" ] = df.loc[ df.symbol == symbol, "split_adjusted_close" ].pct_change()
            df.loc[ df.symbol == symbol, "algo_turtle_pct" ] = df.loc[ df.symbol == symbol, "equity" ].pct_change()

            #--------------------------------------------------------------------------
            # Calculate total return in percentage.
            #--------------------------------------------------------------------------
            # TODO.
            # Dollar term.
            baseline_dollar = df.loc[ df.symbol == symbol, "split_adjusted_close" ].head(1).iloc[0]

            df.loc[ (df.symbol == symbol) & (df.date >= start_date_str), "split_adjusted_close_return" ] = df.loc[ df.symbol == symbol, "split_adjusted_close_pct" ].add(1).cumprod().mul(baseline_dollar)

            #--------------------------------------------------------------------------
            # Calculate total profit and loss in dollars.
            #--------------------------------------------------------------------------
            # TODO.
            # Dollar term.
            baseline_dollar = df.loc[ (df.symbol == symbol) & (df.date >= start_date_str), "split_adjusted_close" ].head(1).iloc[0]

            df.loc[ (df.symbol == symbol) & (df.date >= start_date_str), "algo_turtle_return" ] = df.loc[ df.symbol == symbol, "algo_turtle_pct" ].add(1).cumprod().mul(baseline_dollar)

        # Calculate equity returns.
        df = df.sort_values(by=["date", "symbol"])
        df.loc[ df.date >= start_date_str, "algo_turtle_equity" ] = df.equity.div(self.INITIAL_CAPITAL)
        """

        #--------------------------------------------------------------------------
        # Set dataframe.
        #--------------------------------------------------------------------------
        self.df = df

        return


    def backtest_turtle_rules(self, start_date_str, end_date_str):
        #--------------------------------------------------------------------------
        # Generate symbol indicators.
        #--------------------------------------------------------------------------
        # print("[{}] Generating symbol indicators...".format(datetime.now().isoformat()))
        # for symbol in self.symbol_universe:

        #     self.generate_indicators(symbol)

        df = self.df
        #--------------------------------------------------------------------------
        # Rank qualified stocks by momentum.
        #--------------------------------------------------------------------------
        print("[{}] [INFO] Ranking qualified stock universe by momentum...".format(datetime.now().isoformat()))

        # Rank only qualified stocks.
        # df["turtle_rank"] = df.where(~df.disqualify_penny & ~df.disqualify_volatile).groupby("date")["momentum_score"].rank(ascending=False)
        df["turtle_rank"] = df.loc[ :, ["date", "symbol", "momentum_score"]].where(~df.disqualify_penny & ~df.disqualify_volatile).groupby("date")["momentum_score"].rank(ascending=False)

        # Rank all stocks.
        # df["turtle_rank"] = df.groupby("date")["momentum_score"].rank(ascending=False)

        #--------------------------------------------------------------------------
        # Calculate stock weights.
        #--------------------------------------------------------------------------
        print("[{}] [INFO] Calculating stock weights...".format(datetime.now().isoformat()))

        # Testing. Delete me.
        # Bias cheap symbols.
        # df["weights"] = df.loc[ df.turtle_rank <= self.PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.inv_atr / group.inv_atr.sum())

        # Proper.
        df["weights"] = df.loc[ df.turtle_rank <= self.PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.inv_std / group.inv_std.sum())

        # Equal weights.
        # df["weights"] = df.loc[ df.turtle_rank <= self.PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.turtle_rank / group.turtle_rank  / group.shape[0])

        #--------------------------------------------------------------------------
        # Generate symbol trading data.
        #--------------------------------------------------------------------------
        print("[{}] [INFO] Generating trading data...".format(datetime.now().isoformat()))
        self.generate_all_trading_data(start_date_str, end_date_str)

        # for symbol in self.symbol_universe:

        #     self.generate_trading_data(symbol, start_date_str, end_date_str)

        #--------------------------------------------------------------------------
        # Generate returns.
        #--------------------------------------------------------------------------

        return


    def generate_backtest_graph(self, start_date_str, end_date_str):
        print("[{}] [INFO] Generating backtest graph...".format(datetime.now().isoformat()))

        #--------------------------------------------------------------------------
        # Calculate returns.
        #--------------------------------------------------------------------------
        df_turtle = self.generate_df_returns_one_pass(self.df, start_date_str, end_date_str)
        df_spy = get_daily_split_adjusted_df("SPY", start_date_str, end_date_str)
        df_spy["returns"] = df_spy.split_adjusted_close.pct_change()

        df_turtle["cum_returns"] = (df_turtle["returns"] + 1).cumprod()
        df_spy["cum_returns"] = (df_spy["returns"] + 1).cumprod()


        #--------------------------------------------------------------------------
        # Get exchange holidays.
        #--------------------------------------------------------------------------
        holiday_list = [ date.isoformat() for date in cboe_holidays.keys() ]

        #--------------------------------------------------------------------------
        # Make graph with Plotly.
        #--------------------------------------------------------------------------
        fig = make_subplots(rows=1, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0,
                            row_heights=[1])

        fig.add_trace(go.Scatter(x=df_turtle.date, y=df_turtle.cum_returns, name="turtle"),
                        row=1, col=1)

        fig.add_trace(go.Scatter(x=df_spy.date, y=df_spy.cum_returns, name="spy"),
                        row=1, col=1)

        # Close.
        # fig.add_trace(go.Scatter(x=df.date, y=df.split_adjusted_close, name=symbol),
        #                 row=1, col=1)

        # # Open, high, low, close.
        # fig.add_trace(go.Candlestick(x=df.date,
        #                                 open=df.split_adjusted_open,
        #                                 high=df.split_adjusted_high,
        #                                 low=df.split_adjusted_low,
        #                                 close=df.split_adjusted_close,
        #                                 name=symbol),
        #                 row=1, col=1)


        fig.update_layout(autosize=True,
                            hovermode="x unified")

        # Sync x axis during zoom and pan.
        fig.update_xaxes(matches='x')

        # Set x axis range.
        fig.update_xaxes(range=[df_turtle['date'].iloc[0], df_turtle['date'].iloc[-1]])

        # Fixed y axis range.
        # fig.update_yaxes(fixedrange=True)

        # Hide range slider from OHLC chart.
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Hide weekends and holidays.
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["Saturday", "Monday"]),
                dict(values=holiday_list),
            ],
        )

        # Title.
        fig.update_layout(title="equity curve")

        # Set theme.
        fig.update_layout(template="plotly_dark")

        # Overlay bars.
        fig.update_layout(barmode="overlay")

        fig.show()

        # Save to HTML file.
        # fig.write_html("{}/{}_{}-{}.html".format(self.GRAPH_ROOT_PATH, self.symbol, self.start_date_str, self.end_date_str))

        return


    def generate_symbol_graph(self, symbol, start_date_str, end_date_str):
        #--------------------------------------------------------------------------
        # Get exchange holidays.
        #--------------------------------------------------------------------------
        holiday_list = [ date.isoformat() for date in cboe_holidays.keys() ]

        #--------------------------------------------------------------------------
        # Graph only data in range.
        #--------------------------------------------------------------------------
        df = get_daily_split_adjusted_df(symbol, start_date_str, end_date_str)

        #--------------------------------------------------------------------------
        # Make graph with Plotly.
        #--------------------------------------------------------------------------
        fig = make_subplots(rows=1, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0,
                            row_heights=[1])

        # Close.
        fig.add_trace(go.Scatter(x=df.date, y=df.split_adjusted_close, name=symbol),
                        row=1, col=1)

        # Open, high, low, close.
        fig.add_trace(go.Candlestick(x=df.date,
                                    open=df.split_adjusted_open,
                                    high=df.split_adjusted_high,
                                    low=df.split_adjusted_low,
                                    close=df.split_adjusted_close,
                                    name=symbol),
                        row=1, col=1)


        fig.update_layout(autosize=True, hovermode="x unified")

        # Sync x axis during zoom and pan.
        fig.update_xaxes(matches='x')

        # Set x axis range.
        fig.update_xaxes(range=[df['date'].iloc[0], df['date'].iloc[-1]])

        # Fixed y axis range.
        # fig.update_yaxes(fixedrange=True)

        # Hide range slider from OHLC chart.
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Hide weekends and holidays.
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["Saturday", "Monday"]),
                dict(values=holiday_list),
            ],
        )

        # Title.
        fig.update_layout(title=symbol)

        # Set theme.
        fig.update_layout(template="plotly_dark")

        # Overlay bars.
        fig.update_layout(barmode="overlay")

        fig.show()

        # Save to HTML file.
        # fig.write_html("{}/{}_{}-{}.html".format(self.GRAPH_ROOT_PATH, self.symbol, self.start_date_str, self.end_date_str))

        return


    def generate_trade_summary(self):

        print("[{}] [INFO] Generating trade summary...".format(datetime.now().isoformat()))

        # Dictionary of list. Key: symbol. Value: [ entry date, last open trade ]
        trade_info = {}
        df_trade_summary_values = []

        #--------------------------------------------------------------------------
        # Get dataframe.
        #--------------------------------------------------------------------------
        df = self.df

        #--------------------------------------------------------------------------
        # Create named tuple trade.
        #--------------------------------------------------------------------------
        TradeSummaryTuple = namedtuple("TradeSummaryTuple", "symbol date_entry date_exit avg_price exit_price book_value trade_pnl")

        #--------------------------------------------------------------------------
        # Process entry and exit only.
        #--------------------------------------------------------------------------
        for row in df.itertuples(index=False):

            # Skip rows without cashflow.
            if pd.isnull(row.cashflow) or row.cashflow == 0:
                continue

            # Store trade info into dictionary.
            if row.symbol not in trade_info and row.cashflow < 0:

                trade_info[row.symbol] = [row.date, row]

            # Update last open trade.
            elif row.symbol in trade_info and row.unit_cnt_long == 1:

                trade_info[row.symbol][1] = row

            # Pop entry and store both entry and exit to list.
            elif row.symbol in trade_info and row.unit_cnt_long == 0:

                # Unpack list.
                symbol_trade_info = trade_info.pop(row.symbol)
                entry_date, last_open_trade = symbol_trade_info
                exit_trade = row

                df_trade_summary_values.append(TradeSummaryTuple(last_open_trade.symbol,
                                                            entry_date,
                                                            last_open_trade.date,
                                                            last_open_trade.avg_price,
                                                            exit_trade.split_adjusted_open,
                                                            last_open_trade.book_value,
                                                            exit_trade.trade_pnl))

        #--------------------------------------------------------------------------
        # Add outstanding trades.
        #--------------------------------------------------------------------------
        for symbol_trade_info in trade_info.values():

            # Unpack list.
            entry_date, last_open_trade = symbol_trade_info

            df_trade_summary_values.append(TradeSummaryTuple(last_open_trade.symbol,
                                                        entry_date,
                                                        np.nan,
                                                        last_open_trade.avg_price,
                                                        np.nan,
                                                        last_open_trade.book_value,
                                                        np.nan))

        #--------------------------------------------------------------------------
        # Convert to dataframe.
        #--------------------------------------------------------------------------
        df_trade_summary = pd.DataFrame(df_trade_summary_values)

        #--------------------------------------------------------------------------
        # Add trade profit and loss percentage.
        #--------------------------------------------------------------------------
        df_trade_summary["trade_pnl_pct"] = df_trade_summary.trade_pnl / df_trade_summary.book_value

        #--------------------------------------------------------------------------
        # Write to csv.
        #--------------------------------------------------------------------------
        df_trade_summary.to_csv("{}/algo_turtle_trade_summary.csv".format(turtle.CSV_ROOT_PATH), index=False)

        #--------------------------------------------------------------------------
        # Trade statistics.
        #--------------------------------------------------------------------------
        trades = df_trade_summary
        winning_trades = trades.loc[ trades.trade_pnl > 0 ]
        losing_trades = trades.loc[ trades.trade_pnl < 0 ]

        print("----------------------------------------------------------------------")
        print("  All trades:")
        print("----------------------------------------------------------------------")
        print(trades[["trade_pnl", "trade_pnl_pct"]].describe().to_string())

        print("----------------------------------------------------------------------")
        print("  Winning trades:")
        print("----------------------------------------------------------------------")
        print(winning_trades[["trade_pnl", "trade_pnl_pct"]].describe().to_string())

        print("----------------------------------------------------------------------")
        print("  Losing trades:")
        print("----------------------------------------------------------------------")
        print(losing_trades[["trade_pnl", "trade_pnl_pct"]].describe().to_string())


    #--------------------------------------------------------------------------
    # Assume df has at least two columns: date and equity
    #--------------------------------------------------------------------------
    def generate_df_returns_one_pass(self, df, start_date_str, end_date_str):

        # Sanity check. Consumes more memory.
        # df = df.loc[ df.date >= self.START_DATE ]
        # df = df.groupby(df.date, as_index=False).apply(lambda g: g.iloc[-1])
        # df["returns"] = df.equity.pct_change()

        prev_row = None
        return_values = []

        start_date_ts = pd.Timestamp(start_date_str)
        end_date_ts = pd.Timestamp(end_date_str)

        #--------------------------------------------------------------------------
        # Create named tuple trade.
        #--------------------------------------------------------------------------
        ReturnTuple = namedtuple("ReturnTuple", "date equity")
        
        for row in df.itertuples():

            # Add to list if we reach a new date.
            if row.Index > 0 and row.date > prev_row.date and prev_row.date >= start_date_ts and prev_row.date <= end_date_ts:
                return_values.append(ReturnTuple(prev_row.date, prev_row.equity))

            # Add to list if we reach the end of dataframe.
            if row.Index+1 == df.shape[0] and row.date >= start_date_ts and row.date <= end_date_ts:
                return_values.append(ReturnTuple(row.date, row.equity))

            # Next iteration.
            prev_row = row

        #--------------------------------------------------------------------------
        # Convert to dataframe.
        #--------------------------------------------------------------------------
        df_returns = pd.DataFrame(return_values)
        df_returns["returns"] = df_returns.equity.pct_change()

        return df_returns



    def generate_tear_sheet(self):

        print("[{}] [INFO] Generating tear sheet...".format(datetime.now().isoformat()))

        #--------------------------------------------------------------------------
        # Calculate returns.
        #--------------------------------------------------------------------------
        df_turtle = self.generate_df_returns_one_pass(self.df, self.START_DATE, self.END_DATE)
        df_spy = get_daily_split_adjusted_df("SPY", self.START_DATE, self.END_DATE)
        df_spy["returns"] = df_spy.split_adjusted_close.pct_change()

        #--------------------------------------------------------------------------
        # Calculate tear sheet statistics.
        #--------------------------------------------------------------------------
        annual_return = em.annual_return(df_turtle.returns)
        cum_return = em.cum_returns(df_turtle.returns).iloc[-1]
        annual_volatility = em.annual_volatility(df_turtle.returns)
        sharpe_ratio = em.sharpe_ratio(df_turtle.returns)
        calmar_ratio = em.calmar_ratio(df_turtle.returns)
        stability = em.stability_of_timeseries(df_turtle.returns)
        max_drawdown = em.max_drawdown(df_turtle.returns)
        omega_ratio = em.omega_ratio(df_turtle.returns)
        sortino_ratio = em.sortino_ratio(df_turtle.returns)
        skew = df_turtle.returns.skew()
        kurtosis = df_turtle.returns.kurtosis()
        tail_ratio = em.tail_ratio(df_turtle.returns)
        common_sense_ratio = pt.common_sense_ratio(df_turtle.returns)
        daily_value_at_risk = pt.value_at_risk(df_turtle.returns)
        alpha = em.alpha(df_turtle.returns, df_spy.returns)
        beta = em.beta(df_turtle.returns, df_spy.returns)

        print("----------------------------------------------------------------------")
        print("  Tear sheet:")
        print("----------------------------------------------------------------------")
        print("        annual_return: {:.4f}".format(annual_return))
        print("           cum_return: {:.4f}".format(cum_return))
        print("    annual_volatility: {:.4f}".format(annual_volatility))
        print("         sharpe_ratio: {:.4f}".format(sharpe_ratio))
        print("         calmar_ratio: {:.4f}".format(calmar_ratio))
        print("            stability: {:.4f}".format(stability))
        print("         max_drawdown: {:.4f}".format(max_drawdown))
        print("          omega_ratio: {:.4f}".format(omega_ratio))
        print("        sortino_ratio: {:.4f}".format(sortino_ratio))
        print("                 skew: {:.4f}".format(skew))
        print("             kurtosis: {:.4f}".format(kurtosis))
        print("           tail_ratio: {:.4f}".format(tail_ratio))
        print("   common_sense_ratio: {:.4f}".format(common_sense_ratio))
        print("  daily_value_at_risk: {:.4f}".format(daily_value_at_risk))
        print("                alpha: {:.4f}".format(alpha))
        print("                 beta: {:.4f}".format(beta))

        return


    def dump_weights_entries(self):

        print("[{}] [INFO] Dumping rows with weights...".format(datetime.now().isoformat()))

        df = self.df

        #----------------------------------------------------------------------
        # Create a subset dataframe if there is enough memory.
        #----------------------------------------------------------------------
        try:
            df.loc[ df.weights > 0 ].to_csv("{}/algo_turtle_weights.csv".format(turtle.CSV_ROOT_PATH), index=False)
        except:
            #------------------------------------------------------------------
            # Replace NaN with empty string.
            #------------------------------------------------------------------
            # df.fillna('', inplace=True)

            #------------------------------------------------------------------
            # Convert date column to string.
            #------------------------------------------------------------------
            df.date = df.date.dt.strftime("%Y-%m-%d")

            with open("{}/algo_turtle_weights.csv".format(turtle.CSV_ROOT_PATH), 'w', newline='') as f:

                # CSV writer, defaults uses "\r\n". 
                cw = csv.writer(f, lineterminator='\n')
                
                # Write headers.
                cw.writerow(list(df))

                # Write rows.
                for row in df.itertuples(index=False):
                    if row.weights > 0:
                        cw.writerow(row)

            #------------------------------------------------------------------
            # Convert date column to type numpy datetime64.
            #------------------------------------------------------------------
            df.date = pd.to_datetime(df.date)

        return


    def dump_cashflow_entries(self):

        print("[{}] [INFO] Dumping rows with cashflow...".format(datetime.now().isoformat()))

        df = self.df

        #----------------------------------------------------------------------
        # Create a subset dataframe if there is enough memory.
        #----------------------------------------------------------------------
        try:
            df.loc[ ~df.cashflow.isna() ].to_csv("{}/algo_turtle_cashflow.csv".format(turtle.CSV_ROOT_PATH), index=False)
        except:
            #------------------------------------------------------------------
            # Replace NaN with empty string.
            #------------------------------------------------------------------
            # df.fillna('', inplace=True)

            #------------------------------------------------------------------
            # Convert date column to string.
            #------------------------------------------------------------------
            df.date = df.date.dt.strftime("%Y-%m-%d")

            with open("{}/algo_turtle_cashflow.csv".format(turtle.CSV_ROOT_PATH), 'w', newline='') as f:

                # CSV writer, defaults uses "\r\n". 
                cw = csv.writer(f, lineterminator='\n')
                
                # Write headers.
                cw.writerow(list(df))

                # Write rows.
                for row in df.itertuples(index=False):
                    if row.cashflow > 0 or row.cashflow < 0:
                        cw.writerow(row)

            #------------------------------------------------------------------
            # Convert date column to type numpy datetime64.
            #------------------------------------------------------------------
            df.date = pd.to_datetime(df.date)

        return


    def generate_trade_summary_graph(self, trade_summary_full_path):

        #----------------------------------------------------------------------
        # Read in trade summary csv file.
        #----------------------------------------------------------------------
        df_trade_summary = pd.read_csv(trade_summary_full_path)

        #----------------------------------------------------------------------
        # Graph pnl.
        #----------------------------------------------------------------------
        # quantile_25 = df_trade_summary.trade_pnl.quantile(.25)
        # quantile_75 = df_trade_summary.trade_pnl.quantile(.75)
        # bin_width = 2 * (quantile_75-quantile_25) / ( df_trade_summary.trade_pnl.count() ** (1/3) )

        # bins = int((df_trade_summary.trade_pnl.max() - df_trade_summary.trade_pnl.min()) / bin_width)
        # plt.hist(df_trade_summary.trade_pnl.values, bins=bins)
        # plt.show()

        #----------------------------------------------------------------------
        # Graph pnl percentage.
        #----------------------------------------------------------------------
        quantile_25 = df_trade_summary.trade_pnl_pct.quantile(.25)
        quantile_75 = df_trade_summary.trade_pnl_pct.quantile(.75)
        bin_width = 2 * (quantile_75-quantile_25) / ( df_trade_summary.trade_pnl_pct.count() ** (1/3) )

        bins = int((df_trade_summary.trade_pnl_pct.max() - df_trade_summary.trade_pnl_pct.min()) / bin_width)
        plt.hist(df_trade_summary.trade_pnl_pct.values, bins=bins)
        plt.show()

        return
