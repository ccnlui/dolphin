#        File: backtest.py
#        Date: 2021-10-26
#          By: Calvin
#       Brief: Backtest class with states.

import pandas as pd
import numpy as np

import bisect

from backtests.constants import (
    INITIAL_CAPITAL,
    TRADE_DAILY,
    TRADE_MONTHLY,
    TRADE_WEEKLY_WEDNESDAY,
)


class Backtest:

    def __init__(self, df, algo, start_date_str, end_date_str):

        self.df = df
        self.algo = algo
        self.start_date = pd.to_datetime(start_date_str)
        self.end_date = pd.to_datetime(end_date_str)

        # Simulation variables.
        self.curr_trade_id = 0
        self.curr_date = None
        self.curr_cash = INITIAL_CAPITAL
        self.curr_equity = self.curr_cash
        self.curr_account_pnl = 0
        self.equity_bod = self.curr_equity
        self.prev_trading_date = None

        self.symbol_prev_idx = {}
        self.symbol_curr_idx = {}
        self.portfolio_symbol = []
        self.curr_watchlist = []
        self.prev_watchlist = []

        # Initialize columns.
        df['trade_id'] = np.nan
        df['cnt_long'] = np.nan
        df['qty_long'] = np.nan
        df['stop_loss'] = np.nan
        df['last_fill'] = np.nan
        df['avg_price'] = np.nan
        df['cashflow'] = np.nan
        df['book_value'] = np.nan
        df['market_value'] = np.nan
        df['trade_pnl'] = np.nan
        df['cash'] = np.nan
        df['equity'] = np.nan
        df['account_pnl'] = np.nan


    def in_backtest_period(self):
        return self.start_date <= self.curr_date and self.curr_date <= self.end_date


    def is_trading_day(self):

        algo = self.algo

        if algo.get_trade_frequency() == TRADE_DAILY:
            return True

        if algo.get_trade_frequency() == TRADE_WEEKLY_WEDNESDAY:
            if (
                self.curr_date.isocalendar().weekday == 3
                or (self.prev_trading_date is not None and (self.curr_date - self.prev_trading_date).days > 7)
            ):
                return True

        if algo.get_trade_frequency() == TRADE_MONTHLY:
            if (
                self.prev_trading_date is None
                or self.curr_date.month != self.prev_trading_date.month
            ):
                return True

        return False


    def new_trade_id(self):
        self.curr_trade_id += 1
        return self.curr_trade_id


    def liquidate(self, symbol):
        """
        Assume position was liquidated at the previous close if symbol is not available for trading today.
        Run every day.
        """

        df = self.df

        # Don't change previous entry.
        prev_idx = self.symbol_prev_idx[symbol]

        # Assume position liquidated at previous close.
        liquidate_cashflow = df.split_adjusted_close[prev_idx] * df.qty_long[prev_idx]
        liquidate_trade_pnl = liquidate_cashflow - df.book_value[prev_idx]
        self.curr_cash += liquidate_cashflow

        self.portfolio_symbol.remove(symbol)

        print("------------------------------------------------")
        print("[WARNING] Liquidated trade: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
            self.curr_date.date().isoformat(),
            symbol,
            0,
            df.split_adjusted_close[prev_idx],
            liquidate_cashflow,
            0,
            0,
            0,
            self.curr_cash,
            self.curr_equity,
            self.curr_account_pnl,
            liquidate_trade_pnl
        ))
        print("------------------------------------------------")


    def mark_to_market(self, tick, symbol, price):
        """
        Mark to market at every tick.
        Carry over previous day's values at open.
        """

        df = self.df

        curr_idx = self.symbol_curr_idx[symbol]
        prev_idx = self.symbol_prev_idx[symbol]

        # Carry over if rows are empty at open.
        if tick == 'O':
            df.trade_id.iat[curr_idx] = df.trade_id[prev_idx]
            df.cnt_long.iat[curr_idx] = df.cnt_long[prev_idx]
            df.qty_long.iat[curr_idx] = df.qty_long[prev_idx]
            df.stop_loss.iat[curr_idx] = df.stop_loss[prev_idx]
            df.last_fill.iat[curr_idx] = df.last_fill[prev_idx]
            df.avg_price.iat[curr_idx] = df.avg_price[prev_idx]
            df.cashflow.iat[curr_idx] = 0
            df.book_value.iat[curr_idx] = df.book_value[prev_idx]
            df.market_value.iat[curr_idx] = df.market_value[prev_idx]
            df.trade_pnl.iat[curr_idx] = df.trade_pnl[prev_idx]

        # Trade columns.
        # df.trade_id.iat[curr_idx]
        # df.cnt_long.iat[curr_idx]
        # df.qty_long.iat[curr_idx]
        # df.stop_loss.iat[curr_idx]
        # df.last_fill.iat[curr_idx]
        # df.avg_price.iat[curr_idx]

        # df.cashflow.iat[curr_idx]
        # df.book_value.iat[curr_idx]
        prev_market_value = df.market_value.iat[curr_idx]
        df.market_value[curr_idx] = price * df.qty_long.iat[curr_idx]
        df.trade_pnl[curr_idx] = df.market_value[curr_idx] - df.book_value.iat[curr_idx]

        # Account.
        self.curr_equity = self.curr_equity - prev_market_value + df.market_value.iat[curr_idx]
        self.curr_account_pnl = self.curr_equity - INITIAL_CAPITAL

        # Account columns.
        df.cash.iat[curr_idx] = self.curr_cash
        df.equity.iat[curr_idx] = self.curr_equity
        df.account_pnl.iat[curr_idx] = self.curr_account_pnl

        print("[DEBUG]  Mark-to-market {}: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
            tick,
            self.curr_date.date().isoformat(),
            symbol,
            df.qty_long[curr_idx],
            price,
            df.cashflow[curr_idx],
            df.book_value[curr_idx],
            df.avg_price[curr_idx],
            df.market_value[curr_idx],
            self.curr_cash,
            self.curr_equity,
            self.curr_account_pnl,
            df.trade_pnl[curr_idx]
        ))


    def buy(self, tick, symbol, price):
        """
        Enter position using latest indicators calculated using prices until last close.

        Current limitations:
            - Cannot buy multiple times in 1 day
            - Cannot buy if sold symbol earlier the same day
        """

        df = self.df
        algo = self.algo

        curr_idx = self.symbol_curr_idx[symbol]
        prev_idx = self.symbol_prev_idx[symbol]

        target_qty_long = np.floor(self.equity_bod * df.weights[prev_idx] / price)

        if target_qty_long == 0:
            print("------------------------------------------------")
            print("[WARNING] Buying symbol {} with target quantity 0 on {}. ({} * {} / {})".format(
                symbol,
                self.curr_date.date().isoformat(),
                self.equity_bod,
                df.weights[prev_idx],
                price
            ))
            print("------------------------------------------------")

        elif target_qty_long > 0:

            # Initialize columns in case they are empty.
            if np.isnan(df.trade_id[curr_idx]):
                df.cashflow.iat[curr_idx] = 0
                df.book_value.iat[curr_idx] = 0
                df.market_value.iat[curr_idx] = 0
                df.trade_pnl.iat[curr_idx] = 0
                df.trade_id.iat[curr_idx] = self.new_trade_id()
                df.cnt_long.iat[curr_idx] = 0
                df.qty_long.iat[curr_idx] = 0
                df.stop_loss.iat[curr_idx] = 0
                df.last_fill.iat[curr_idx] = 0
                df.avg_price.iat[curr_idx] = 0

            # Account.
            self.curr_cash -= price * target_qty_long

            df.cash.iat[curr_idx] = self.curr_cash
            df.equity.iat[curr_idx] = self.curr_equity
            df.account_pnl.iat[curr_idx] = self.curr_account_pnl

            # Trade columns.
            df.cashflow.iat[curr_idx] -= price * target_qty_long
            df.book_value.iat[curr_idx] += price * target_qty_long
            df.market_value.iat[curr_idx] += price * target_qty_long
            df.trade_pnl[curr_idx] = df.market_value[curr_idx] - df.book_value.iat[curr_idx]

            df.cnt_long.iat[curr_idx] = 1
            df.qty_long.iat[curr_idx] += target_qty_long
            df.stop_loss.iat[curr_idx] = algo.calculate_stop_loss(symbol, price, self.symbol_curr_idx, self.symbol_prev_idx, self.df)
            df.last_fill.iat[curr_idx] = price
            df.avg_price[curr_idx] = df.book_value[curr_idx] / df.qty_long.iat[curr_idx]

            # Add to portfolio if not exist.
            if self.portfolio_symbol.count(symbol) == 0:
                # portfolio_symbol.append(symbol)
                bisect.insort(self.portfolio_symbol, symbol)


            print("[INFO]      Enter trade {}: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
                tick,
                self.curr_date.date().isoformat(),
                symbol,
                df.qty_long[curr_idx],
                price,
                df.cashflow[curr_idx],
                df.book_value[curr_idx],
                df.avg_price[curr_idx],
                df.market_value[curr_idx],
                self.curr_cash,
                self.curr_equity,
                self.curr_account_pnl,
                df.trade_pnl[curr_idx]
            ))

        else:
            print("------------------------------------------------")
            print("[ERROR] Unexpected buying condition for {} on {}".format(
                symbol,
                self.curr_date.date().isoformat(),
            ))
            print("------------------------------------------------")


    def sell(self, tick, symbol, price):

        df = self.df

        curr_idx = self.symbol_curr_idx[symbol]

        # Account.
        self.curr_cash += price * df.qty_long[curr_idx]

        # Trade columns.
        df.cashflow.iat[curr_idx] += price * df.qty_long[curr_idx]
        df.book_value.iat[curr_idx] -= price * df.qty_long[curr_idx]
        df.market_value.iat[curr_idx] = 0

        # Close remaining book value as trade profit and loss.
        df.trade_pnl.iat[curr_idx] = df.book_value[curr_idx] * -1
        df.book_value.iat[curr_idx] = 0

        # df.trade_id.iat[curr_idx]
        df.cnt_long.iat[curr_idx] = 0
        df.qty_long.iat[curr_idx] = 0
        df.stop_loss.iat[curr_idx] = 0
        df.last_fill.iat[curr_idx] = price
        df.avg_price.iat[curr_idx] = 0

        # Account columns (no change).
        df.cash.iat[curr_idx] = self.curr_cash
        df.equity.iat[curr_idx] = self.curr_equity
        df.account_pnl.iat[curr_idx] = self.curr_account_pnl

        self.portfolio_symbol.remove(symbol)

        print("[INFO]       Exit trade {}: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
            tick,
            self.curr_date.date().isoformat(),
            symbol,
            df.qty_long[curr_idx],
            price,
            df.cashflow[curr_idx],
            df.book_value[curr_idx],
            df.avg_price[curr_idx],
            df.market_value[curr_idx],
            self.curr_cash,
            self.curr_equity,
            self.curr_account_pnl,
            df.trade_pnl[curr_idx]
        ))


    def rebalance(self, tick, symbol, price):

        df = self.df

        curr_idx = self.symbol_curr_idx[symbol]
        prev_idx = self.symbol_prev_idx[symbol]

        # Rebalance trade.
        target_qty_long = np.floor(self.equity_bod * df.weights[prev_idx] / price)
        delta_qty_long = target_qty_long - df.qty_long[curr_idx]

        # Market trend filter.
        if not df.market_trend_filter[prev_idx]:
            print("[DEBUG]  Market trend down: {} Not buying {}.".format(
                self.curr_date.date().isoformat(),
                symbol,
            ))
            return self.curr_cash, self.curr_equity, self.curr_account_pnl

        # Add to position.
        if delta_qty_long > 0:

            # Trade columns.
            df.cashflow.iat[curr_idx] -= price * delta_qty_long
            df.book_value.iat[curr_idx] += price * delta_qty_long
            df.market_value.iat[curr_idx] += price * delta_qty_long
            df.trade_pnl.iat[curr_idx] = df.market_value[curr_idx] - df.book_value[curr_idx]

            # df.trade_id.iat[curr_idx]
            # df.cnt_long.iat[curr_idx]
            df.qty_long.iat[curr_idx] += delta_qty_long
            # df.stop_loss.iat[curr_idx]
            df.last_fill.iat[curr_idx] = price
            df.avg_price.iat[curr_idx] = df.book_value[curr_idx] / df.qty_long[curr_idx]

        # Sell position.
        if delta_qty_long < 0:

            if target_qty_long == 0:
                # Trade columns.
                df.cashflow.iat[curr_idx] -= price * delta_qty_long
                df.book_value.iat[curr_idx] += price * delta_qty_long
                df.market_value.iat[curr_idx] = 0

                # Close remaining book value as trade profit and loss.
                df.trade_pnl.iat[curr_idx] = df.book_value[curr_idx] * -1
                df.book_value.iat[curr_idx] = 0

                # df.trade_id.iat[curr_idx]
                df.cnt_long.iat[curr_idx] = 0
                df.qty_long.iat[curr_idx] = 0
                df.stop_loss.iat[curr_idx] = 0
                df.last_fill.iat[curr_idx] = price
                df.avg_price.iat[curr_idx] = 0

                self.portfolio_symbol.remove(symbol)

            else:
                # Trade columns.
                df.cashflow.iat[curr_idx] -= price * delta_qty_long
                df.book_value.iat[curr_idx] += price * delta_qty_long
                df.market_value.iat[curr_idx] += price * delta_qty_long
                df.trade_pnl.iat[curr_idx] = df.market_value[curr_idx] - df.book_value[curr_idx]

                # df.trade_id.iat[curr_idx]
                # df.cnt_long.iat[curr_idx]
                df.qty_long.iat[curr_idx] += delta_qty_long
                # df.stop_loss.iat[curr_idx]
                df.last_fill.iat[curr_idx] = price
                df.avg_price.iat[curr_idx] = df.book_value[curr_idx] / df.qty_long[curr_idx]

        # Account.
        self.curr_cash += df.cashflow[curr_idx]

        # Account columns.
        df.cash.iat[curr_idx] = self.curr_cash
        df.equity.iat[curr_idx] = self.curr_equity
        df.account_pnl.iat[curr_idx] = self.curr_account_pnl

        print("[INFO]        Rebalance {}: {} {} {}@{:.4f} shares, cashflow {:.4f}, book value {:.4f}, avg price {:.4f}, market value {:.4f}, cash {:.4f}, equity {:.4f}, acount pnl {:.4f}, trade pnl {:.4f}".format(
            tick,
            self.curr_date.date().isoformat(),
            symbol,
            df.qty_long[curr_idx],
            price,
            df.cashflow[curr_idx],
            df.book_value[curr_idx],
            df.avg_price[curr_idx],
            df.market_value[curr_idx],
            self.curr_cash,
            self.curr_equity,
            self.curr_account_pnl,
            df.trade_pnl[curr_idx]
        ))


    def simulate_trading(self):

        algo = self.algo
        df = self.df

        # Process tick data.
        length = len(df)
        for idx in range(0, length):

            # New date.
            if df.date[idx] != self.curr_date:

                print("[DEBUG] Processing date {}...".format(df.date[idx]))

                # Reset.
                self.curr_date = df.date[idx]

                # Store previous day's symbols.
                for s, i in self.symbol_curr_idx.items():
                    self.symbol_prev_idx[s] = i

                self.prev_watchlist = self.curr_watchlist.copy()
                self.symbol_curr_idx.clear()
                self.curr_watchlist.clear()

            #------------------------------------------------------------------
            # Read in symbol data.
            #------------------------------------------------------------------
            symbol_str = str(df.symbol[idx])
            self.symbol_curr_idx[symbol_str] = idx

            # Insert to watchlist.
            if df.ranking[idx] <= algo.get_portfolio_num_stock():
                # self.curr_watchlist.append(symbol_str)
                bisect.insort(self.curr_watchlist, symbol_str)

            # Finish reading entire day's data before trading.
            if idx+1 < length and df.date[idx+1] <= self.curr_date:
                continue

            #------------------------------------------------------------------
            # Trading.
            #------------------------------------------------------------------
            if self.in_backtest_period():

                #------------------------------------------------------------------
                # Pre-market: liquidate symbols not available for trading.
                #------------------------------------------------------------------
                for symbol in self.portfolio_symbol.copy():
                    if symbol not in self.symbol_curr_idx:
                        self.liquidate(symbol)

                for symbol in self.prev_watchlist.copy():
                    if symbol not in self.symbol_curr_idx:
                        print("------------------------------------------------")
                        print("[WARNING] Watchlist symbol: {} not available for trading on {}.".format(symbol, self.curr_date))
                        print("------------------------------------------------")
                        self.prev_watchlist.remove(symbol)

                #------------------------------------------------------------------
                # Open: Mark-to-market.
                #------------------------------------------------------------------
                self.equity_bod = self.curr_equity
                for symbol in self.portfolio_symbol:
                    tick = 'O'
                    price = df.split_adjusted_open[self.symbol_curr_idx[symbol]]
                    self.mark_to_market(tick, symbol, price)
                    self.equity_bod = self.curr_equity

                #------------------------------------------------------------------
                # Open: Sell, rebalance, buy.
                #------------------------------------------------------------------
                if self.is_trading_day():
                    for symbol in self.portfolio_symbol.copy():
                        tick = 'O'
                        price = df.split_adjusted_open[self.symbol_curr_idx[symbol]]
                        if algo.sell_signal(symbol, price, self.symbol_curr_idx, self.symbol_prev_idx, df):
                            self.sell(tick, symbol, price)

                    for symbol in self.portfolio_symbol:
                        tick = 'O'
                        price = df.split_adjusted_open[self.symbol_curr_idx[symbol]]
                        self.rebalance(tick, symbol, price)

                    for symbol in self.prev_watchlist.copy():
                        tick = 'O'
                        price = df.split_adjusted_open[self.symbol_curr_idx[symbol]]
                        if algo.buy_signal(symbol, price, self.symbol_curr_idx, self.symbol_prev_idx, df):
                            self.buy(tick, symbol, price)

                #------------------------------------------------------------------
                # Close: Mark-to-market.
                #------------------------------------------------------------------
                for symbol in self.portfolio_symbol:
                    tick = 'C'
                    price = df.split_adjusted_close[self.symbol_curr_idx[symbol]]
                    self.mark_to_market(tick, symbol, price)

                #------------------------------------------------------------------
                # Close: Sell, buy.
                #------------------------------------------------------------------
                if self.is_trading_day():
                    for symbol in self.portfolio_symbol.copy():
                        tick = 'C'
                        price = df.split_adjusted_close[self.symbol_curr_idx[symbol]]
                        if algo.sell_signal(symbol, price, self.symbol_curr_idx, self.symbol_prev_idx, df):
                            self.sell(tick, symbol, price)

                    for symbol in self.prev_watchlist.copy():
                        tick = 'C'
                        price = df.split_adjusted_close[self.symbol_curr_idx[symbol]]
                        if algo.buy_signal(symbol, price, self.symbol_curr_idx, self.symbol_prev_idx, df):
                            self.buy(tick, symbol, price)

                #------------------------------------------------------------------
                # After market.
                #------------------------------------------------------------------
                if self.is_trading_day():
                    self.prev_trading_date = self.curr_date

        return df
