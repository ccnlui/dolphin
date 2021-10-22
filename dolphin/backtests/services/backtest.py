#        File: backtest.py
#        Date: 2021-10-04
#          By: Calvin
#       Brief: Backtest app.
# Requirement: Python 3.8

from backtests.services.market_data import (
    get_symbol_list_daily_split_adjusted_df_list,
    get_daily_split_adjusted_df,
    get_sp500_symbols_list,
)
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
import bisect

from backtests.constants import (
    ATR_PERIOD,
    ATR_SMOOTHING_FACTOR,
    BASIS_POINT_DP,
    CSV_ROOT_PATH,
    EXPENSIVE_PRICE,
    EXP_MODEL_GUESS_A,
    EXP_MODEL_GUESS_B,
    INITIAL_CAPITAL,
    MARKET_DATA_ROOT_PATH,
    MARKET_TREND_FILTER_DAYS,
    MIN_MOMENTUM_SCORE,
    MOMENTUM_WINDOW,
    PENNY_PRICE,
    PORTFOLIO_NUM_STOCK,
    PREFETCH_NUM_MONTH,
    SINGLE_DAY_VOLATILITY_FILTER_DAYS,
    SINGLE_DAY_VOLATILITY_FILTER_PCT,
    START_DATE,
    END_DATE,
    TURTLE_PERIOD_ENTRY,
    TURTLE_PERIOD_EXIT,
    VOL_PERIOD,
    YEARLY_TRADING_DAYS,
)

class pandas_algo_turtle(object):

    #--------------------------------------------------------------------------
    # Constructor.
    #--------------------------------------------------------------------------
    def __init__(self):
        #----------------------------------------------------------------------
        # Members.
        #----------------------------------------------------------------------
        # self.symbol_universe = ["AAPL", "AMD", "NVDA"]
        # self.symbol_universe = ["AAPL", "FB", "AMZN", "GOOGL", "TSLA"]
        # self.symbol_universe = ["AAPL", "AMD", "NVDA", "PTON", "FSLY", "OSTK", "BIGC", "SHOP", "QUSA", "THTX", "GOOGL", "BRNC"]
        # self.symbol_universe = ["XELB", "ACS", "CODA", "AAPL", "AMD", "NVDA"]
        # self.symbol_universe = ["CODA"]

        # self.symbol_universe = os.listdir(self.MARKET_DATA_ROOT_PATH)
        # self.symbol_universe.sort()
        # if "raw" in self.symbol_universe:
        #     self.symbol_universe.remove('raw')

        self.symbol_universe = get_sp500_symbols_list()
        self.df = None

    #--------------------------------------------------------------------------
    # Methods.
    #--------------------------------------------------------------------------
    def load_market_data(self, symbol_universe, start_date_str, end_date_str, interval):
        #----------------------------------------------------------------------
        # Initialize dates.
        #----------------------------------------------------------------------
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        prefetch_start_date = start_date - relativedelta(months=PREFETCH_NUM_MONTH)

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
                                            MARKET_DATA_ROOT_PATH,
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
        prefetch_start_date = start_date - relativedelta(months=PREFETCH_NUM_MONTH)

        #----------------------------------------------------------------------
        # Read raw daily adjusted from database.
        #----------------------------------------------------------------------
        df_list = get_symbol_list_daily_split_adjusted_df_list(symbol_universe, prefetch_start_date.isoformat(), end_date.isoformat())

        #--------------------------------------------------------------------------
        # Index close + market trend filter.
        #--------------------------------------------------------------------------
        symbol_index = 'SPY'
        df_index = get_daily_split_adjusted_df(symbol_index, prefetch_start_date.isoformat(), end_date.isoformat())
        df_index = df_index.loc[ :, ['date', 'split_adjusted_close']]
        df_index.rename(columns={'split_adjusted_close': 'index_close'}, inplace=True)
        df_index.set_index('date', inplace=True)
        df_index['index_close_sma'] = df_index.index_close.rolling(MARKET_TREND_FILTER_DAYS).mean()
        df_index['market_trend_filter'] = (df_index.index_close > df_index.index_close_sma).astype(int)
        df_list = [ df.join(df_index, on='date') for df in df_list ]

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
        self.df.to_csv("{}/algo_turtle_indicators.csv".format(CSV_ROOT_PATH), index=False)

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
        prefetch_start_date = start_date - relativedelta(months=PREFETCH_NUM_MONTH)

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
            listing_fullpath = "{}/{}/Alpha_Vantage_{}_listing_status.csv".format(LISTING_ROOT_PATH, curr_date.year, curr_date)
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
        prefetch_start_date = start_date - relativedelta(months=PREFETCH_NUM_MONTH)

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
        x = np.arange(MOMENTUM_WINDOW)

        #----------------------------------------------------------------------
        # Sanity check: Scipy curve fit.
        #----------------------------------------------------------------------
        try:
            popt, pcov = curve_fit(lambda x, a, b: a*np.exp(b*x), x, time_series, p0=(EXP_MODEL_GUESS_A, EXP_MODEL_GUESS_B))
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
                score = (np.power(1+b, YEARLY_TRADING_DAYS) - 1) * 100 * r2

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
    #--------------------------------------------------------------------------
    @staticmethod
    # @jit(nopython=True)
    def simulate_trading(
        date,
        symbol,
        split_adjusted_open,
        split_adjusted_close,
        close_entry_rolling_max,
        close_exit_rolling_min,
        market_trend_filter,
        atr,
        momentum_score,
        turtle_rank,
        weights,
        initial_capital,
        start_date,
        end_date,
        portfolio_num_stock
    ):
        #--------------------------------------------------------------------------
        # Initialize columns.
        #--------------------------------------------------------------------------
        length = date.shape[0]

        # Trade specific columns.
        trade_id = np.full(length, np.nan)
        cnt_long = np.full(length, np.nan)
        qty_long = np.full(length, np.nan)
        stop_loss = np.full(length, np.nan)
        last_fill = np.full(length, np.nan)
        avg_price = np.full(length, np.nan)
        cashflow = np.full(length, np.nan)
        book_value = np.full(length, np.nan)
        market_value = np.full(length, np.nan)
        trade_pnl = np.full(length, np.nan)

        # Account specific columns.
        cash = np.full(length, np.nan)
        equity = np.full(length, np.nan)
        account_pnl = np.full(length, np.nan)

        #--------------------------------------------------------------------------
        # Initialize variables.
        #--------------------------------------------------------------------------
        trading = False
        curr_trade_id = 0
        curr_date = None
        curr_cash = initial_capital
        curr_equity = curr_cash
        curr_account_pnl = 0
        equity_bod = curr_equity

        #--------------------------------------------------------------------------
        # Keep track of symbol variables.
        #--------------------------------------------------------------------------
        symbol_prev_idx = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        symbol_curr_idx = Dict.empty(key_type=types.unicode_type, value_type=types.int64)

        portfolio_symbol = []
        curr_watchlist = []
        prev_watchlist = []

        #--------------------------------------------------------------------------
        # Inner helper.
        #--------------------------------------------------------------------------
        def new_trade_id():
            nonlocal curr_trade_id
            curr_trade_id += 1
            return curr_trade_id


        def buy_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):

            curr_idx = symbol_curr_idx[curr_symbol]
            prev_idx = symbol_prev_idx[curr_symbol]

            # Market trend filter.
            if not market_trend_filter[curr_idx]:
                print("[DEBUG]  Market trend down: {} Not buying {}.".format(
                    curr_date,
                    curr_symbol,
                ))
                return False

            if cnt_long[curr_idx] > 0:
                return False

            # Don't buy if it is sold earlier the day.
            if not np.isnan(trade_id[curr_idx]):
                return False

            # Minimum momentum.
            if momentum_score[curr_idx] < MIN_MOMENTUM_SCORE:
                return False

            # Not rank.
            if np.isnan(turtle_rank[prev_idx]) or turtle_rank[prev_idx] > portfolio_num_stock:
                return False

            # Turtle entry.
            if True:
            # if curr_price >= close_entry_rolling_max[prev_idx]:
                return True

            return False

        def sell_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):

            curr_idx = symbol_curr_idx[curr_symbol]
            prev_idx = symbol_prev_idx[curr_symbol]

            if cnt_long[curr_idx] == 0:
                return False

            # Turtle exit and stop loss.
            if False:
            # if curr_price <= close_exit_rolling_min[prev_idx]:
                return True
            if False:
            # if curr_price <= stop_loss[prev_idx]:
                return True

            # Minimum momentum.
            if momentum_score[curr_idx] < MIN_MOMENTUM_SCORE:
                return True

            # Penny stock.
            if curr_price < PENNY_PRICE:
                return True

            # Expensive stock.
            if curr_price > EXPENSIVE_PRICE:
                return True

            # Not rank.
            if np.isnan(turtle_rank[prev_idx]) or turtle_rank[prev_idx] > portfolio_num_stock:
                return True

            # Volatile.
            # TODO.

            return False


        def liquidate(curr_date, curr_symbol, symbol_prev_idx, portfolio_symbol, curr_cash, curr_equity, curr_account_pnl, prev_watchlist):

            # Don't change previous entry.
            prev_idx = symbol_prev_idx[curr_symbol]

            # Assume position liquidated at previous close.
            liquidate_cashflow = split_adjusted_close[prev_idx] * qty_long[prev_idx]
            liquidate_trade_pnl = liquidate_cashflow - book_value[prev_idx]
            curr_cash += liquidate_cashflow

            portfolio_symbol.remove(curr_symbol)

            print("------------------------------------------------")
            print("[WARNING] Liquidated trade: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                curr_date,
                curr_symbol,
                0,
                split_adjusted_close[prev_idx],
                liquidate_cashflow,
                0,
                0,
                0,
                curr_cash,
                curr_equity,
                curr_account_pnl,
                liquidate_trade_pnl
            ))
            print("------------------------------------------------")

            return curr_cash, curr_equity, curr_account_pnl


        def mark_to_market(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl):

            curr_idx = symbol_curr_idx[curr_symbol]
            prev_idx = symbol_prev_idx[curr_symbol]

            # Carry over if at open.
            trade_id[curr_idx] = trade_id[prev_idx] if np.isnan(trade_id[curr_idx]) else trade_id[curr_idx]
            cnt_long[curr_idx] = cnt_long[prev_idx] if np.isnan(cnt_long[curr_idx]) else cnt_long[curr_idx]
            qty_long[curr_idx] = qty_long[prev_idx] if np.isnan(qty_long[curr_idx]) else qty_long[curr_idx]
            stop_loss[curr_idx] = stop_loss[prev_idx] if np.isnan(stop_loss[curr_idx]) else stop_loss[curr_idx]
            last_fill[curr_idx] = last_fill[prev_idx] if np.isnan(last_fill[curr_idx]) else last_fill[curr_idx]
            avg_price[curr_idx] = avg_price[prev_idx] if np.isnan(avg_price[curr_idx]) else avg_price[curr_idx]
            cashflow[curr_idx] = 0 if np.isnan(cashflow[curr_idx]) else cashflow[curr_idx]
            book_value[curr_idx] = book_value[prev_idx] if np.isnan(book_value[curr_idx]) else book_value[curr_idx]
            market_value[curr_idx] = market_value[prev_idx] if np.isnan(market_value[curr_idx]) else market_value[curr_idx]
            trade_pnl[curr_idx] = trade_pnl[prev_idx] if np.isnan(trade_pnl[curr_idx]) else trade_pnl[curr_idx]

            # Trade columns.
            # trade_id[curr_idx]
            # cnt_long[curr_idx]
            # qty_long[curr_idx]
            # stop_loss[curr_idx]
            # last_fill[curr_idx]
            # avg_price[curr_idx]

            # cashflow[curr_idx]
            # book_value[curr_idx]
            prev_market_value = market_value[curr_idx]
            market_value[curr_idx] = curr_price * qty_long[curr_idx]
            trade_pnl[curr_idx] = market_value[curr_idx] - book_value[curr_idx]

            # Account.
            curr_equity = curr_equity - prev_market_value + market_value[curr_idx]
            curr_account_pnl = curr_equity - initial_capital

            # Account columns.
            cash[curr_idx] = curr_cash
            equity[curr_idx] = curr_equity
            account_pnl[curr_idx] = curr_account_pnl

            print("[DEBUG]  Mark-to-market {}: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                curr_tick,
                curr_date,
                curr_symbol,
                qty_long[curr_idx],
                curr_price,
                cashflow[curr_idx],
                book_value[curr_idx],
                avg_price[curr_idx],
                market_value[curr_idx],
                curr_cash,
                curr_equity,
                curr_account_pnl,
                trade_pnl[curr_idx]
            ))

            return curr_cash, curr_equity, curr_account_pnl


        def buy(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl, equity_bod):

            curr_idx = symbol_curr_idx[curr_symbol]
            prev_idx = symbol_prev_idx[curr_symbol]

            target_qty_long = np.floor(equity_bod * weights[prev_idx] / curr_price)

            if target_qty_long == 0:
                print("------------------------------------------------")
                print("[WARNING] Buying symbol {} with target quantity 0 on {}. ({} * {} / {})".format(
                    curr_symbol,
                    curr_date,
                    equity_bod,
                    weights[prev_idx],
                    curr_price
                ))
                print("------------------------------------------------")

            elif target_qty_long > 0:

                # Initialize columns.
                cashflow[curr_idx] = 0 if np.isnan(cashflow[curr_idx]) else cashflow[curr_idx]
                book_value[curr_idx] = 0 if np.isnan(book_value[curr_idx]) else book_value[curr_idx]
                market_value[curr_idx] = 0 if np.isnan(market_value[curr_idx]) else market_value[curr_idx]
                trade_pnl[curr_idx] = 0 if np.isnan(trade_pnl[curr_idx]) else trade_pnl[curr_idx]
                trade_id[curr_idx] = new_trade_id() if np.isnan(trade_id[curr_idx]) else trade_id[curr_idx]
                cnt_long[curr_idx] = 0 if np.isnan(cnt_long[curr_idx]) else cnt_long[curr_idx]
                qty_long[curr_idx] = 0 if np.isnan(qty_long[curr_idx]) else qty_long[curr_idx]
                stop_loss[curr_idx] = 0 if np.isnan(stop_loss[curr_idx]) else stop_loss[curr_idx]
                last_fill[curr_idx] = 0 if np.isnan(last_fill[curr_idx]) else last_fill[curr_idx]
                avg_price[curr_idx] = 0 if np.isnan(avg_price[curr_idx]) else avg_price[curr_idx]

                # Account.
                curr_cash -= curr_price * target_qty_long

                # Account columns.
                cash[curr_idx] = curr_cash
                equity[curr_idx] = curr_equity
                account_pnl[curr_idx] = curr_account_pnl

                # Trade columns.
                cashflow[curr_idx] -= curr_price * target_qty_long
                book_value[curr_idx] += curr_price * target_qty_long
                market_value[curr_idx] += curr_price * target_qty_long
                trade_pnl[curr_idx] = market_value[curr_idx] - book_value[curr_idx]

                cnt_long[curr_idx] = 1
                qty_long[curr_idx] += target_qty_long
                stop_loss[curr_idx] = curr_price - 2*atr[prev_idx]
                last_fill[curr_idx] = curr_price
                avg_price[curr_idx] = book_value[curr_idx] / qty_long[curr_idx]

                # Add to portfolio if not exist.
                if portfolio_symbol.count(curr_symbol) == 0:
                    # portfolio_symbol.append(curr_symbol)
                    bisect.insort(portfolio_symbol, curr_symbol)


                print("[INFO]      Enter trade {}: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                    curr_tick,
                    curr_date,
                    curr_symbol,
                    qty_long[curr_idx],
                    curr_price,
                    cashflow[curr_idx],
                    book_value[curr_idx],
                    avg_price[curr_idx],
                    market_value[curr_idx],
                    curr_cash,
                    curr_equity,
                    curr_account_pnl,
                    trade_pnl[curr_idx]
                ))

            else:
                print("------------------------------------------------")
                print("[ERROR] Unexpected buying condition for {} on {}".format(
                    curr_symbol,
                    curr_date,
                ))
                print("------------------------------------------------")

            return curr_cash, curr_equity, curr_account_pnl


        def sell(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, curr_cash, curr_equity, curr_account_pnl):

            curr_idx = symbol_curr_idx[curr_symbol]

            # Account.
            curr_cash += curr_price * qty_long[curr_idx]

            # Trade columns.
            cashflow[curr_idx] += curr_price * qty_long[curr_idx]
            book_value[curr_idx] -= curr_price * qty_long[curr_idx]
            market_value[curr_idx] = 0

            # Close remaining book value as trade profit and loss.
            trade_pnl[curr_idx] = book_value[curr_idx] * -1
            book_value[curr_idx] = 0

            # trade_id[curr_idx]
            cnt_long[curr_idx] = 0
            qty_long[curr_idx] = 0
            stop_loss[curr_idx] = 0
            last_fill[curr_idx] = curr_price
            avg_price[curr_idx] = 0

            # Account columns.
            cash[curr_idx] = curr_cash
            equity[curr_idx] = curr_equity
            account_pnl[curr_idx] = curr_account_pnl

            portfolio_symbol.remove(curr_symbol)

            print("[INFO]       Exit trade {}: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                curr_tick,
                curr_date,
                curr_symbol,
                qty_long[curr_idx],
                curr_price,
                cashflow[curr_idx],
                book_value[curr_idx],
                avg_price[curr_idx],
                market_value[curr_idx],
                curr_cash,
                curr_equity,
                curr_account_pnl,
                trade_pnl[curr_idx]
            ))

            return curr_cash, curr_equity, curr_account_pnl


        def rebalance(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl, equity_bod):

            curr_idx = symbol_curr_idx[curr_symbol]
            prev_idx = symbol_prev_idx[curr_symbol]

            # Rebalance trade.
            target_qty_long = np.floor(equity_bod * weights[prev_idx] / curr_price)
            delta_qty_long = target_qty_long - qty_long[curr_idx]

            # Market trend filter.
            if not market_trend_filter[curr_idx]:
                print("[DEBUG]  Market trend down: {} Not buying {}.".format(
                    curr_date,
                    curr_symbol,
                ))
                return curr_cash, curr_equity, curr_account_pnl

            # Add to position.
            if delta_qty_long > 0:

                # Trade columns.
                cashflow[curr_idx] -= curr_price * delta_qty_long
                book_value[curr_idx] += curr_price * delta_qty_long
                market_value[curr_idx] += curr_price * delta_qty_long
                trade_pnl[curr_idx] = market_value[curr_idx] - book_value[curr_idx]

                # trade_id[curr_idx]
                # cnt_long[curr_idx]
                qty_long[curr_idx] += delta_qty_long
                # stop_loss[curr_idx]
                last_fill[curr_idx] = curr_price
                avg_price[curr_idx] = book_value[curr_idx] / qty_long[curr_idx]

            if delta_qty_long < 0:

                # Sell position.
                if target_qty_long == 0:
                    # Trade columns.
                    cashflow[curr_idx] -= curr_price * delta_qty_long
                    book_value[curr_idx] += curr_price * delta_qty_long
                    market_value[curr_idx] = 0

                    # Close remaining book value as trade profit and loss.
                    trade_pnl[curr_idx] = book_value[curr_idx] * -1
                    book_value[curr_idx] = 0

                    # trade_id[curr_idx]
                    cnt_long[curr_idx] = 0
                    qty_long[curr_idx] = 0
                    stop_loss[curr_idx] = 0
                    last_fill[curr_idx] = curr_price
                    avg_price[curr_idx] = 0

                    portfolio_symbol.remove(curr_symbol)

                # Remove from position.
                else:
                    # Trade columns.
                    cashflow[curr_idx] -= curr_price * delta_qty_long
                    book_value[curr_idx] += curr_price * delta_qty_long
                    market_value[curr_idx] += curr_price * delta_qty_long
                    trade_pnl[curr_idx] = market_value[curr_idx] - book_value[curr_idx]

                    # trade_id[curr_idx]
                    # cnt_long[curr_idx]
                    qty_long[curr_idx] += delta_qty_long
                    # stop_loss[curr_idx]
                    last_fill[curr_idx] = curr_price
                    avg_price[curr_idx] = book_value[curr_idx] / qty_long[curr_idx]

            # Account.
            curr_cash += cashflow[curr_idx]

            # Account columns.
            cash[curr_idx] = curr_cash
            equity[curr_idx] = curr_equity
            account_pnl[curr_idx] = curr_account_pnl

            print("[INFO]        Rebalance {}: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                curr_tick,
                curr_date,
                curr_symbol,
                qty_long[curr_idx],
                curr_price,
                cashflow[curr_idx],
                book_value[curr_idx],
                avg_price[curr_idx],
                market_value[curr_idx],
                curr_cash,
                curr_equity,
                curr_account_pnl,
                trade_pnl[curr_idx]
            ))

            return curr_cash, curr_equity, curr_account_pnl

        #--------------------------------------------------------------------------
        # Process tick data.
        #--------------------------------------------------------------------------
        for idx in range(0, length):

            # New date.
            if date[idx] != curr_date:

                print("[DEBUG] Processing date {}...".format(date[idx]))

                #------------------------------------------------------------------
                # Reset.
                #------------------------------------------------------------------
                trading = False
                curr_date = date[idx]

                # Store previous day's symbols.
                for curr_symbol, curr_idx in symbol_curr_idx.items():
                    symbol_prev_idx[curr_symbol] = curr_idx

                prev_watchlist = curr_watchlist.copy()

                symbol_curr_idx.clear()
                curr_watchlist.clear()

            #------------------------------------------------------------------
            # Read in symbol data.
            #------------------------------------------------------------------
            # Convert unichr array into unicode strings.
            symbol_str = str(symbol[idx])
            symbol_curr_idx[symbol_str] = idx

            # Watchlist.
            if turtle_rank[idx] <= portfolio_num_stock:
                # curr_watchlist.append(symbol_str)
                bisect.insort(curr_watchlist, symbol_str)

            #------------------------------------------------------------------
            # Trading.
            #------------------------------------------------------------------
            # Read in all symbols for a day before trading.
            if (idx+1 == length or date[idx+1] > curr_date) and start_date <= curr_date and curr_date <= end_date:
                trading = True

            if trading:

                #------------------------------------------------------------------
                # Pre-market: liquidate halted/delisted symbols.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol.copy():
                    if curr_symbol not in symbol_curr_idx:
                        curr_cash, curr_equity, acurr_ccount_pnl = liquidate(curr_date, curr_symbol, symbol_prev_idx, portfolio_symbol, curr_cash, curr_equity, curr_account_pnl, prev_watchlist)

                for curr_symbol in prev_watchlist.copy():
                    if curr_symbol not in symbol_curr_idx:
                        print("------------------------------------------------")
                        print("[WARNING] Watchlist symbol: {} not available for trading on {}.".format(curr_symbol, curr_date))
                        print("------------------------------------------------")
                        prev_watchlist.remove(curr_symbol)

                #------------------------------------------------------------------
                # Open: Mark-to-market.
                #------------------------------------------------------------------
                equity_bod = curr_equity
                for curr_symbol in portfolio_symbol:
                    curr_tick = 'O'
                    curr_price = split_adjusted_open[symbol_curr_idx[curr_symbol]]
                    curr_cash, curr_equity, curr_account_pnl = mark_to_market(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl)
                    equity_bod = curr_equity

                #------------------------------------------------------------------
                # Open: Sell.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol.copy():
                    curr_tick = 'O'
                    curr_price = split_adjusted_open[symbol_curr_idx[curr_symbol]]
                    if sell_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):
                        curr_cash, curr_equity, curr_account_pnl = sell(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, curr_cash, curr_equity, curr_account_pnl)

                #------------------------------------------------------------------
                # Open: Rebalance.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol:
                    curr_tick = 'O'
                    curr_price = split_adjusted_open[symbol_curr_idx[curr_symbol]]
                    curr_cash, curr_equity, curr_account_pnl = rebalance(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl, equity_bod)

                #------------------------------------------------------------------
                # Open: Buy.
                #------------------------------------------------------------------
                for curr_symbol in prev_watchlist.copy():
                    curr_tick = 'O'
                    curr_price = split_adjusted_open[symbol_curr_idx[curr_symbol]]
                    if buy_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):
                        curr_cash, curr_equity, curr_account_pnl = buy(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl, equity_bod)

                '''
                #------------------------------------------------------------------
                # High: Mark-to-market.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol:
                    curr_tick = 'H'
                    curr_price = split_adjusted_high[symbol_curr_idx[curr_symbol]]
                    curr_cash, curr_equity, curr_account_pnl = mark_to_market(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl)

                #------------------------------------------------------------------
                # High: Sell.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol.copy():
                    curr_tick = 'H'
                    curr_price = split_adjusted_high[symbol_curr_idx[curr_symbol]]
                    if sell_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):
                        curr_cash, curr_equity, curr_account_pnl = sell(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, curr_cash, curr_equity, curr_account_pnl)

                #------------------------------------------------------------------
                # High: Buy.
                #------------------------------------------------------------------
                for curr_symbol in prev_watchlist.copy():
                    curr_tick = 'H'
                    curr_price = split_adjusted_high[symbol_curr_idx[curr_symbol]]
                    if buy_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):
                        curr_cash, curr_equity, curr_account_pnl = buy(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl, equity_bod)

                #------------------------------------------------------------------
                # Low: Mark-to-market.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol:
                    curr_tick = 'L'
                    curr_price = split_adjusted_low[symbol_curr_idx[curr_symbol]]
                    curr_cash, curr_equity, curr_account_pnl = mark_to_market(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl)

                #------------------------------------------------------------------
                # Low: Sell.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol.copy():
                    curr_tick = 'L'
                    curr_price = split_adjusted_low[symbol_curr_idx[curr_symbol]]
                    if sell_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):
                        curr_cash, curr_equity, curr_account_pnl = sell(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, curr_cash, curr_equity, curr_account_pnl)

                #------------------------------------------------------------------
                # Low: Buy.
                #------------------------------------------------------------------
                for curr_symbol in prev_watchlist.copy():
                    curr_tick = 'L'
                    curr_price = split_adjusted_close[symbol_curr_idx[curr_symbol]]
                    if buy_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):
                        curr_cash, curr_equity, curr_account_pnl = buy(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl, equity_bod)
                '''

                #------------------------------------------------------------------
                # Close: Mark-to-market.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol:
                    curr_tick = 'C'
                    curr_price = split_adjusted_close[symbol_curr_idx[curr_symbol]]
                    curr_cash, curr_equity, curr_account_pnl = mark_to_market(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl)

                #------------------------------------------------------------------
                # Close: Sell.
                #------------------------------------------------------------------
                for curr_symbol in portfolio_symbol.copy():
                    curr_tick = 'C'
                    curr_price = split_adjusted_close[symbol_curr_idx[curr_symbol]]
                    if sell_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):
                        curr_cash, curr_equity, curr_account_pnl = sell(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, curr_cash, curr_equity, curr_account_pnl)

                #------------------------------------------------------------------
                # Close: Buy.
                #------------------------------------------------------------------
                for curr_symbol in prev_watchlist.copy():
                    curr_tick = 'C'
                    curr_price = split_adjusted_close[symbol_curr_idx[curr_symbol]]
                    if buy_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx):
                        curr_cash, curr_equity, curr_account_pnl = buy(curr_tick, curr_date, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, curr_cash, curr_equity, curr_account_pnl, equity_bod)

        return trade_id, cnt_long, qty_long, stop_loss, last_fill, avg_price, cashflow, book_value, market_value, trade_pnl, cash, equity, account_pnl


    def generate_symbol_indicators(self, df_symbol):

        print("[{}] [INFO] Generating indicators for symbol: {}".format(datetime.now().isoformat(), df_symbol.symbol.iloc[0]))

        #--------------------------------------------------------------------------
        # Calculate rolling max/min of close.
        #--------------------------------------------------------------------------
        # Exit.
        df_symbol["close_exit_rolling_max" ] = df_symbol["split_adjusted_close"].rolling(TURTLE_PERIOD_EXIT).max()
        df_symbol["close_exit_rolling_min" ] = df_symbol["split_adjusted_close"].rolling(TURTLE_PERIOD_EXIT).min()

        # Entry.
        df_symbol["close_entry_rolling_max"] = df_symbol["split_adjusted_close"].rolling(TURTLE_PERIOD_ENTRY).max()
        df_symbol["close_entry_rolling_min"] = df_symbol["split_adjusted_close"].rolling(TURTLE_PERIOD_ENTRY).min()

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
        df_symbol["atr"] = df_symbol["true_range"].ewm(alpha=ATR_SMOOTHING_FACTOR, min_periods=ATR_PERIOD).mean()

        # Inverse ATR.
        df_symbol["inv_atr"] = df_symbol["atr"].rdiv(1)
        df_symbol["inv_atr"] = df_symbol["inv_atr"].replace(np.inf, np.nan)

        #--------------------------------------------------------------------------
        # Calculate standard deviation.
        #--------------------------------------------------------------------------
        # df_symbol["std"] = df_symbol["split_adjusted_close"].rolling(VOL_PERIOD).std()
        df_symbol["std"] = df_symbol["split_adjusted_close"].pct_change().rolling(VOL_PERIOD).std()
        df_symbol["std"] = df_symbol["std"].round(BASIS_POINT_DP)

        # Inverse standard deviation.
        df_symbol["inv_std"] = df_symbol["std"].rdiv(1)
        df_symbol["inv_std"] = df_symbol["inv_std"].replace(np.inf, np.nan)

        #--------------------------------------------------------------------------
        # Calculate rolling max of single day absolute percent change.
        #--------------------------------------------------------------------------
        df_symbol["abs_pct_rolling_max"] = df_symbol["split_adjusted_close"].pct_change().abs().mul(100).rolling(SINGLE_DAY_VOLATILITY_FILTER_DAYS).max()

        #--------------------------------------------------------------------------
        # Exponential regression.
        #--------------------------------------------------------------------------
        try:
            df_symbol["momentum_score"] = df_symbol["split_adjusted_close"].rolling(MOMENTUM_WINDOW).apply(pandas_algo_turtle.momentum_score, raw=True)
            df_symbol["momentum_score"] = df_symbol["momentum_score"].replace(np.inf, np.nan)
        except:
            print("[{}] [ERROR] Cannot calculate momentum score for symbol: {}.".format(datetime.now().isoformat(), df_symbol.symbol.iloc[0]))
            raise

        #--------------------------------------------------------------------------
        # Disqualify filter.
        #--------------------------------------------------------------------------
        # Disqualify symbols trading under $1.00.
        df_symbol["disqualify_penny"] = (df_symbol["split_adjusted_close"] < PENNY_PRICE).astype(int)

        # Disqualify symbols trading above $1000.00.
        df_symbol["disqualify_expensive"] = (df_symbol["split_adjusted_close"] > EXPENSIVE_PRICE).astype(int)

        # Disqualify symbols with a single day move exceeding 15% in the past 90 days.
        df_symbol["disqualify_volatile"] = (df_symbol["abs_pct_rolling_max"] > SINGLE_DAY_VOLATILITY_FILTER_PCT).astype(int)

        # Disqualify symbols with 0 standard deviation.
        df_symbol["disqualify_stale"] = (df_symbol["std"] == 0).astype(int)

        return df_symbol


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
        market_trend_filter = df.market_trend_filter.values
        atr = df.atr.values
        momentum_score = df.momentum_score.values
        turtle_rank = df.turtle_rank.values
        weights = df.weights.values

        # Calculate positions, trade profit and loss.
        result = pandas_algo_turtle.simulate_trading(
            date,
            symbol,
            split_adjusted_open,
            split_adjusted_close,
            close_entry_rolling_max,
            close_exit_rolling_min,
            market_trend_filter,
            atr,
            momentum_score,
            turtle_rank,
            weights,
            INITIAL_CAPITAL,
            start_date,
            end_date,
            PORTFOLIO_NUM_STOCK
        )

        # Testing. Delete me.
        print("[{}] [DEBUG] Add trading data to dataframe.".format(datetime.now().isoformat()))

        # Unpack result.
        df["trade_id"] = result[0]
        df["cnt_long"] = result[1]
        df["qty_long"] = result[2]
        df["stop_loss"] = result[3]
        df["last_fill"] = result[4]
        df["avg_price"] = result[5]
        df["cashflow"] = result[6]
        df["book_value"] = result[7]
        df["market_value"] = result[8]
        df["trade_pnl"] = result[9]
        df["cash"] = result[10]
        df["equity"] = result[11]
        df["account_pnl"] = result[12]

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
            df.loc[ df.symbol == symbol, "long_exposure" ] = np.where(df.loc[ df.symbol == symbol, "cnt_long" ] > 0,
                                                                        df.loc[ df.symbol == symbol, "cnt_long" ] * df.loc[ df.symbol == symbol, "split_adjusted_close"],
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
        df.loc[ df.date >= start_date_str, "algo_turtle_equity" ] = df.equity.div(INITIAL_CAPITAL)
        """

        #--------------------------------------------------------------------------
        # Set dataframe.
        #--------------------------------------------------------------------------
        self.df = df

        return


    def backtest_turtle_rules(self, start_date_str, end_date_str):

        df = self.df
        #--------------------------------------------------------------------------
        # Rank qualified stocks by momentum.
        #--------------------------------------------------------------------------
        print("[{}] [INFO] Ranking qualified stock universe by momentum...".format(datetime.now().isoformat()))

        # Rank only qualified stocks.
        df["turtle_rank"] = df.loc[ :, ["date", "symbol", "momentum_score"]].where(~df.in_sp500_start.isna()).groupby("date")["momentum_score"].rank(ascending=False)
        # df["turtle_rank"] = df.loc[ :, ["date", "symbol", "momentum_score"]].where(
        #     (df.disqualify_penny == 0)
        #     & (df.disqualify_expensive == 0)
        #     & (df.disqualify_volatile == 0)
        #     & (df.disqualify_stale == 0)
        #     & (~df.in_sp500_start.isna())
        # ).groupby("date")["momentum_score"].rank(ascending=False)

        # Rank all stocks.
        # df["turtle_rank"] = df.groupby("date")["momentum_score"].rank(ascending=False)

        #--------------------------------------------------------------------------
        # Calculate stock weights.
        #--------------------------------------------------------------------------
        print("[{}] [INFO] Calculating stock weights...".format(datetime.now().isoformat()))

        # Testing. Delete me.
        # Bias cheap symbols.
        # df["weights"] = df.loc[ df.turtle_rank <= PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.inv_atr / group.inv_atr.sum())

        # Proper.
        df["weights"] = df.loc[ df.turtle_rank <= PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.inv_std / group.inv_std.sum())

        # Equal weights.
        # df["weights"] = df.loc[ df.turtle_rank <= PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.turtle_rank / group.turtle_rank  / group.shape[0])

        #--------------------------------------------------------------------------
        # Generate symbol trading data.
        #--------------------------------------------------------------------------
        print("[{}] [INFO] Generating trading data...".format(datetime.now().isoformat()))
        self.generate_all_trading_data(start_date_str, end_date_str)

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
        # fig.write_html("{}/{}_{}-{}.html".format(GRAPH_ROOT_PATH, self.symbol, self.start_date_str, self.end_date_str))

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
        # fig.write_html("{}/{}_{}-{}.html".format(GRAPH_ROOT_PATH, self.symbol, self.start_date_str, self.end_date_str))

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
            elif row.symbol in trade_info and row.cnt_long == 1:

                trade_info[row.symbol][1] = row

            # Pop entry and store both entry and exit to list.
            elif row.symbol in trade_info and row.cnt_long == 0:

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
        df_trade_summary.to_csv("{}/algo_turtle_trade_summary.csv".format(CSV_ROOT_PATH), index=False)

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
        # df = df.loc[ df.date >= START_DATE ]
        # df = df.groupby(df.date, as_index=False).apply(lambda g: g.iloc[-1])
        # df["returns"] = df.equity.pct_change()

        prev_row = None
        return_values = []

        start_date_ts = pd.Timestamp(start_date_str)
        end_date_ts = pd.Timestamp(end_date_str)

        # #--------------------------------------------------------------------------
        # # Create named tuple trade.
        # #--------------------------------------------------------------------------
        # ReturnTuple = namedtuple("ReturnTuple", "date equity")

        # for row in df.itertuples():

        #     # Add to list if we reach a new date.
        #     if row.Index > 0 and row.date > prev_row.date and prev_row.date >= start_date_ts and prev_row.date <= end_date_ts:
        #         return_values.append(ReturnTuple(prev_row.date, prev_row.equity))

        #     # Add to list if we reach the end of dataframe.
        #     if row.Index+1 == df.shape[0] and row.date >= start_date_ts and row.date <= end_date_ts:
        #         return_values.append(ReturnTuple(row.date, row.equity))

        #     # Next iteration.
        #     prev_row = row

        # #--------------------------------------------------------------------------
        # # Convert to dataframe.
        # #--------------------------------------------------------------------------
        # df_returns = pd.DataFrame(return_values)
        # df_returns["returns"] = df_returns.equity.pct_change()

        # Last row of every group.
        df_returns = df.loc[~df.equity.isna()].sort_values(by=['date', 'symbol']).groupby(['date']).last()
        df_returns.reset_index(inplace=True)
        df_returns = df_returns.loc[:,['date', 'equity']]
        df_returns["returns"] = df_returns.equity.pct_change()

        return df_returns



    def generate_tear_sheet(self):

        print("[{}] [INFO] Generating tear sheet...".format(datetime.now().isoformat()))

        #--------------------------------------------------------------------------
        # Calculate returns.
        #--------------------------------------------------------------------------
        df_turtle = self.generate_df_returns_one_pass(self.df, START_DATE, END_DATE)
        df_spy = get_daily_split_adjusted_df("SPY", START_DATE, END_DATE)
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
            df.loc[ df.weights > 0 ].to_csv("{}/algo_turtle_weights.csv".format(CSV_ROOT_PATH), index=False)
        except:
            #------------------------------------------------------------------
            # Replace NaN with empty string.
            #------------------------------------------------------------------
            # df.fillna('', inplace=True)

            #------------------------------------------------------------------
            # Convert date column to string.
            #------------------------------------------------------------------
            df.date = df.date.dt.strftime("%Y-%m-%d")

            with open("{}/algo_turtle_weights.csv".format(CSV_ROOT_PATH), 'w', newline='') as f:

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
            df.loc[ ~df.cashflow.isna() ].to_csv("{}/algo_turtle_cashflow.csv".format(CSV_ROOT_PATH), index=False)
        except:
            #------------------------------------------------------------------
            # Replace NaN with empty string.
            #------------------------------------------------------------------
            # df.fillna('', inplace=True)

            #------------------------------------------------------------------
            # Convert date column to string.
            #------------------------------------------------------------------
            df.date = df.date.dt.strftime("%Y-%m-%d")

            with open("{}/algo_turtle_cashflow.csv".format(CSV_ROOT_PATH), 'w', newline='') as f:

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
