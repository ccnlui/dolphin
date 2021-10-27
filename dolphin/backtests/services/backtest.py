#        File: backtest.py
#        Date: 2021-10-04
#          By: Calvin
#       Brief: Backtest app.
# Requirement: Python 3.8

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta

from collections import namedtuple

import empyrical as em
import pyfolio.timeseries as pt

import bisect

from backtests.constants import (
    CSV_ROOT_PATH,
    MARKET_DATA_ROOT_PATH,
    START_DATE,
    END_DATE,
    PREFETCH_NUM_MONTH,
    INITIAL_CAPITAL,
    TRADE_DAILY,
    TRADE_WEEKLY_WEDNESDAY,
    TRADE_MONTHLY,
    cboe_holidays,
)

from backtests.services.market_data import (
    get_symbol_list_daily_split_adjusted_df_list,
    get_daily_split_adjusted_df,
    get_sp500_symbols_list,
)

from backtests.backtests.backtest import Backtest


class BacktestService(object):
    """
    Stateless service to perform algo trading backtest.
    """

    #--------------------------------------------------------------------------
    # Methods.
    #--------------------------------------------------------------------------
    def load_symbol_universe_data_from_db(self, symbol_universe, start_date_str, end_date_str):
        """
        Fetch daily adjusted close from database within backtest period for all symbols.

        Returns:
        DataFrame
        """

        print("[{}] [INFO] Loading symbol universe data from database...".format(datetime.now().isoformat()))

        #----------------------------------------------------------------------
        # Initialize dates.
        #----------------------------------------------------------------------
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        prefetch_start_date = start_date - relativedelta(months=PREFETCH_NUM_MONTH)

        #----------------------------------------------------------------------
        # Read raw daily adjusted from database.
        #----------------------------------------------------------------------
        df_symbol_list = get_symbol_list_daily_split_adjusted_df_list(symbol_universe, prefetch_start_date.isoformat(), end_date.isoformat())

        return df_symbol_list


    def load_symbol_universe_data_from_csv(self, csv_fullpath):
        """
        Fetch daily adjusted close from CSV file.

        Returns:
        DataFrame
        """

        print("[{}] [INFO] Loading symbol universe data from csv...".format(datetime.now().isoformat()))

        df = pd.read_csv(csv_fullpath)

        #--------------------------------------------------------------------------
        # Convert date column to type numpy datetime64.
        #--------------------------------------------------------------------------
        df.date = pd.to_datetime(df.date)

        return df


    def load_market_benchmark_data_from_db(self, market_benchmark, start_date_str, end_date_str):
        """
        Fetch daily adjusted close from database within backtest period for market benchmark.

        Returns:
        DataFrame
        """

        print("[{}] [INFO] Loading market benchmark data from database...".format(datetime.now().isoformat()))

        #----------------------------------------------------------------------
        # Initialize dates.
        #----------------------------------------------------------------------
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        prefetch_start_date = start_date - relativedelta(months=PREFETCH_NUM_MONTH)

        #----------------------------------------------------------------------
        # Read raw daily adjusted from database.
        #----------------------------------------------------------------------
        return get_daily_split_adjusted_df(market_benchmark, prefetch_start_date.isoformat(), end_date.isoformat())


    def prepare_data_for_backtest(self, algo, df_symbol_list, df_market):

        df_market = algo.generate_market_indicators(df_market)
        df_symbol_list = algo.append_market_indicators_to_symbol(df_symbol_list, df_market)
        df_symbol_universe = algo.generate_all_symbol_indicators(df_symbol_list)
        df_symbol_universe = algo.rank_symbols(df_symbol_universe)
        df_symbol_universe = algo.calculate_symbol_weights(df_symbol_universe)

        df_symbol_universe.sort_values(by=["date", "symbol"], inplace=True)
        df_symbol_universe.reset_index(inplace=True)

        return df_symbol_universe

    
    def backtest_algo(self, algo_type, start_date_str=None, end_date_str=None):
        
        algo = algo_type()
        df_symbol_list = self.load_symbol_universe_data_from_db(algo.symbol_universe, start_date_str, end_date_str)
        df_market = self.load_market_benchmark_data_from_db(algo.market_benchmark, start_date_str, end_date_str)
        df = self.prepare_data_for_backtest(algo, df_symbol_list, df_market)

        backtest = Backtest(df, algo, start_date_str, end_date_str)
        df = backtest.simulate_trading()

        self.dump_trading_data(df, algo.get_portfolio_num_stock())

        return df


    def dump_trading_data(self, df, portfolio_num_stock):
        df = df.loc[ (df.ranking <= portfolio_num_stock) | (~df.market_value.isna()) | (~df.cashflow.isna()) ]
        df.to_csv("{}/backtest_systematic_momentum.csv".format(CSV_ROOT_PATH), index=False)


    def generate_backtest_graph(self, df, start_date_str, end_date_str):
        print("[{}] [INFO] Generating backtest graph...".format(datetime.now().isoformat()))

        #--------------------------------------------------------------------------
        # Calculate returns.
        #--------------------------------------------------------------------------
        df_turtle = self.generate_df_returns_one_pass(df, start_date_str, end_date_str)
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
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0,
            row_heights=[1]
        )

        fig.add_trace(
            go.Scatter(x=df_turtle.date, y=df_turtle.cum_returns, name="turtle"),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(x=df_spy.date, y=df_spy.cum_returns, name="spy"),
            row=1,
            col=1
        )

        # Close.
        # fig.add_trace(
        #     go.Scatter(x=df.date, y=df.split_adjusted_close, name=symbol),
        #     row=1,
        #     col=1
        # )

        # Open, high, low, close.
        # fig.add_trace(
        #     go.Candlestick(
        #         x=df.date,
        #         open=df.split_adjusted_open,
        #         high=df.split_adjusted_high,
        #         low=df.split_adjusted_low,
        #         close=df.split_adjusted_close,
        #         name=symbol
        #     ),
        #     row=1,
        #     col=1
        # )


        fig.update_layout(
            autosize=True,
            hovermode="x unified"
        )

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


    def generate_trade_summary(self, df):
        """
        Analyze trades from backtest period and write to CSV file.
        """

        print("[{}] [INFO] Generating trade summary...".format(datetime.now().isoformat()))

        # Dictionary of list. Key: symbol. Value: [ entry date, last open trade ]
        trade_info = {}
        df_trade_summary_values = []

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

                df_trade_summary_values.append(TradeSummaryTuple(
                    last_open_trade.symbol,
                    entry_date,
                    last_open_trade.date,
                    last_open_trade.avg_price,
                    exit_trade.split_adjusted_open,
                    last_open_trade.book_value,
                    exit_trade.trade_pnl
                ))

        #--------------------------------------------------------------------------
        # Add outstanding trades.
        #--------------------------------------------------------------------------
        for symbol_trade_info in trade_info.values():

            # Unpack list.
            entry_date, last_open_trade = symbol_trade_info

            df_trade_summary_values.append(TradeSummaryTuple(
                last_open_trade.symbol,
                entry_date,
                np.nan,
                last_open_trade.avg_price,
                np.nan,
                last_open_trade.book_value,
                np.nan
            ))

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



    def generate_tear_sheet(self, df):
        """
        Analyze returns from backtest period.
        """

        print("[{}] [INFO] Generating tear sheet...".format(datetime.now().isoformat()))

        #--------------------------------------------------------------------------
        # Calculate returns.
        #--------------------------------------------------------------------------
        df_turtle = self.generate_df_returns_one_pass(df, START_DATE, END_DATE)
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
