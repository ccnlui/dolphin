#  File: systematic_momentum.py
#  Date: 2021-10-25
#    By: Calvin
# Brief: Systematic momentum trading algo.

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import stats
import numpy as np

from backtests.algos.algo import Algo
from backtests.services.market_data import get_sp500_symbols_list
from backtests.constants import (
    TRADE_DAILY,
    YEARLY_TRADING_DAYS,
)

#--------------------------------------------------------------------------
# Algo params.
#--------------------------------------------------------------------------
ATR_PERIOD                        = 20
ATR_SMOOTHING_FACTOR              = 1 / ATR_PERIOD
EXP_MODEL_GUESS_A                 = 4
EXP_MODEL_GUESS_B                 = 0.1
MOMENTUM_WINDOW                   = 125
PORTFOLIO_NUM_STOCK               = 10
SINGLE_DAY_VOLATILITY_FILTER_DAYS = 90
SINGLE_DAY_VOLATILITY_FILTER_PCT  = 15
TURTLE_PERIOD_ENTRY               = 20
TURTLE_PERIOD_EXIT                = 10
VOL_PERIOD                        = 20
# Round std. Minimum is 0.1%.
BASIS_POINT_DP                    = 4
TRADE_FREQUENCY                   = TRADE_DAILY


#--------------------------------------------------------------------------
# Indicators.
#--------------------------------------------------------------------------
def generate_turtle_indicators(df_ohlc):
    """
    Create turtle indicators as additional columns on symbol dataframe.
    """

    # Calculate rolling max/min of close.
    df_ohlc["close_entry_rolling_max"] = df_ohlc["split_adjusted_close"].rolling(TURTLE_PERIOD_ENTRY).max()
    df_ohlc["close_entry_rolling_min"] = df_ohlc["split_adjusted_close"].rolling(TURTLE_PERIOD_ENTRY).min()
    df_ohlc["close_exit_rolling_max" ] = df_ohlc["split_adjusted_close"].rolling(TURTLE_PERIOD_EXIT).max()
    df_ohlc["close_exit_rolling_min" ] = df_ohlc["split_adjusted_close"].rolling(TURTLE_PERIOD_EXIT).min()

    # Calculate true range + ATR.
    range_1 = df_ohlc["split_adjusted_high"] - df_ohlc["split_adjusted_low"]
    range_2 = df_ohlc["split_adjusted_high"] - df_ohlc["split_adjusted_close"].shift(1)
    range_3 = df_ohlc["split_adjusted_close"].shift(1) - df_ohlc["split_adjusted_low"]
    df_ohlc["true_range"] = pd.concat([range_1, range_2, range_3], axis=1).max(axis=1)

    # Calculate ATR using exponentially moving window.
    df_ohlc["atr"] = df_ohlc["true_range"].ewm(alpha=ATR_SMOOTHING_FACTOR, min_periods=ATR_PERIOD).mean()

    # Inverse ATR.
    df_ohlc["inv_atr"] = df_ohlc["atr"].rdiv(1)
    df_ohlc["inv_atr"] = df_ohlc["inv_atr"].replace(np.inf, np.nan)


def generate_price_pct_change_std_and_inv_std(df_ohlc):
    """
    Create std and inv_std as additional columns on symbol dataframe.
    """
    # Calculate standard deviation.
    df_symbol["std"] = df_symbol["split_adjusted_close"].pct_change().rolling(VOL_PERIOD).std()
    df_symbol["std"] = df_symbol["std"].round(BASIS_POINT_DP)

    # Inverse standard deviation.
    df_symbol["inv_std"] = df_symbol["std"].rdiv(1)
    df_symbol["inv_std"] = df_symbol["inv_std"].replace(np.inf, np.nan)


def momentum_score(time_series):
    #--------------------------------------------------------------------------
    # Exponential regression.
    #   - Exponential model: y = a * e^(b*x)
    #   - Linear model: y = a * x + b
    # Time series is the size of rolling window, 125 days by default.
    #--------------------------------------------------------------------------
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


def generate_momentum_score(df_ohlc):
    """
    Create momentum score as additional columns on symbol dataframe.
    """
    try:
        df_symbol["momentum_score"] = df_symbol["split_adjusted_close"].rolling(MOMENTUM_WINDOW).apply(momentum_score, raw=True)
        df_symbol["momentum_score"] = df_symbol["momentum_score"].replace(np.inf, np.nan)
    except:
        print("[{}] [ERROR] Cannot calculate momentum score for symbol: {}.".format(datetime.now().isoformat(), df_ohlc.symbol.iloc[0]))
        raise


def generate_volatile_filter(df_ohlc):
    """
    Create volatile filter as additional columns on symbol dataframe.
    """
    # Calculate rolling max of single day absolute percent change.
    df_ohlc["abs_pct_rolling_max"] = df_ohlc["split_adjusted_close"].pct_change().abs().mul(100).rolling(SINGLE_DAY_VOLATILITY_FILTER_DAYS).max()

    # Disqualify symbols with a single day move exceeding 15% in the past 90 days.
    df_ohlc["disqualify_volatile"] = (df_ohlc["abs_pct_rolling_max"] > SINGLE_DAY_VOLATILITY_FILTER_PCT).astype(int)


def generate_penny_filter(df_ohlc):
    # Disqualify symbols trading under $1.00.
    df_ohlc["disqualify_penny"] = (df_ohlc["split_adjusted_close"] < PENNY_PRICE).astype(int)


def generate_expensive_filter(df_ohlc):
    # Disqualify symbols trading above $1000.00.
    df_ohlc["disqualify_expensive"] = (df_ohlc["split_adjusted_close"] > EXPENSIVE_PRICE).astype(int)


def generate_stale_filter(df_ohlc):
    # Disqualify symbols with 0 standard deviation.
    df_ohlc["disqualify_stale"] = (df_ohlc["std"] == 0).astype(int)


def generate_market_trend_filter(df_ohlc):
    df_ohlc = df_ohlc.loc[ :, ['date', 'split_adjusted_close']]
    df_ohlc.rename(columns={'split_adjusted_close': 'market_close'}, inplace=True)
    df_ohlc.set_index('date', inplace=True)
    df_ohlc['market_close_sma'] = df_ohlc.market_close.rolling(MARKET_TREND_FILTER_DAYS).mean()
    df_ohlc['market_trend_filter'] = (df_ohlc.market_close > df_ohlc.market_close_sma).astype(int)


class SystematicMomentum(Algo):

    #--------------------------------------------------------------------------
    # Class fields for base Algo class.
    #--------------------------------------------------------------------------
    # symbol_universe = get_sp500_symbols_list()
    symbol_universe = ["AAPL", "FB", "AMZN", "GOOGL", "TSLA"]
    market_benchmark = 'SPY'
    symbol_indicator_list = [
        generate_price_pct_change_std_and_inv_std,
        generate_momentum_score,
        generate_turtle_indicators,
        generate_volatile_filter,
    ]
    market_indicator_list = [
        generate_market_trend_filter,
    ]

    #--------------------------------------------------------------------------
    # Methods.
    #--------------------------------------------------------------------------
    def __init__(self):
        super().__init__()


    def buy_signal(self, curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, df):

        curr_idx = symbol_curr_idx[curr_symbol]
        prev_idx = symbol_prev_idx[curr_symbol]

        # Market trend filter.
        if not df.market_trend_filter[prev_idx]:
            print("[DEBUG]  Market trend down: {} Not buying {}.".format(
                df.date[curr_idx],
                curr_symbol,
            ))
            return False

        if df.cnt_long[curr_idx] > 0:
            return False

        # Don't buy if it is sold earlier the day.
        if not np.isnan(df.trade_id[curr_idx]):
            return False

        # Minimum momentum.
        if df.momentum_score[curr_idx] < MIN_MOMENTUM_SCORE:
            return False

        # Not rank.
        if np.isnan(df.rank[prev_idx]) or df.rank[prev_idx] > PORTFOLIO_NUM_STOCK:
            return False

        # Turtle entry.
        if True:
        # if curr_price >= df.close_entry_rolling_max[prev_idx]:
            return True

        return False


    def sell_signal(curr_symbol, curr_price, symbol_curr_idx, symbol_prev_idx, df):

            curr_idx = symbol_curr_idx[curr_symbol]
            prev_idx = symbol_prev_idx[curr_symbol]

            if df.cnt_long[curr_idx] == 0:
                return False

            # Turtle exit and stop loss.
            if False:
            # if curr_price <= df.close_exit_rolling_min[prev_idx]:
                return True
            if False:
            # if curr_price <= df.stop_loss[prev_idx]:
                return True

            # Minimum momentum.
            if df.momentum_score[curr_idx] < MIN_MOMENTUM_SCORE:
                return True

            # Penny stock.
            if curr_price < PENNY_PRICE:
                return True

            # Expensive stock.
            if curr_price > EXPENSIVE_PRICE:
                return True

            # Not rank.
            if np.isnan(df.rank[prev_idx]) or df.rank[prev_idx] > PORTFOLIO_NUM_STOCK:
                return True

            # Volatile.
            # TODO.

            return False


    def get_trade_frequency(self):
        return TRADE_FREQUENCY


    def rank_symbols(self, df):
        """
        Params:
        df (DataFrame): All symbols with indicators in 1 dataframe.
        """

        # Rank qualified stocks by momentum.
        print("[{}] [INFO] Ranking qualified stock universe by momentum...".format(datetime.now().isoformat()))

        # Rank only qualified stocks.
        df["rank"] = df.loc[ :, ["date", "symbol", "momentum_score"]].where(~df.in_sp500_start.isna()).groupby("date")["momentum_score"].rank(ascending=False)
        # df["rank"] = df.loc[ :, ["date", "symbol", "momentum_score"]].where(
        #     (df.disqualify_penny == 0)
        #     & (df.disqualify_expensive == 0)
        #     & (df.disqualify_volatile == 0)
        #     & (df.disqualify_stale == 0)
        #     & (~df.in_sp500_start.isna())
        # ).groupby("date")["momentum_score"].rank(ascending=False)

        # Rank all stocks.
        # df["rank"] = df.groupby("date")["momentum_score"].rank(ascending=False)
        return df


    def calculate_symbol_weights(self, df):
        """
        Params:
        df (DataFrame): All symbols with indicators in 1 dataframe.
        """

        # Calculate stock weights.
        print("[{}] [INFO] Calculating stock weights...".format(datetime.now().isoformat()))

        # Bias cheap symbols.
        # df["weights"] = df.loc[ df.turtle_rank <= PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.inv_atr / group.inv_atr.sum())

        # Proper.
        df["weights"] = df.loc[ df.turtle_rank <= PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.inv_std / group.inv_std.sum())

        # Equal weights.
        # df["weights"] = df.loc[ df.turtle_rank <= PORTFOLIO_NUM_STOCK ].groupby("date", group_keys=False).apply(lambda group: group.turtle_rank / group.turtle_rank  / group.shape[0])

        return df
