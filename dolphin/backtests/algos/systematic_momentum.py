#  File: systematic_momentum.py
#  Date: 2021-10-25
#    By: Calvin
# Brief: Systematic momentum trading algo.


from backtests.algos.algo import Algo
from backtests.services.market_data import get_sp500_symbols_list


#--------------------------------------------------------------------------
# Algo params.
#--------------------------------------------------------------------------
ATR_PERIOD                        = 20
ATR_SMOOTHING_FACTOR              = 1 / ATR_PERIOD
SINGLE_DAY_VOLATILITY_FILTER_DAYS = 90
SINGLE_DAY_VOLATILITY_FILTER_PCT  = 15
TURTLE_PERIOD_ENTRY               = 20
TURTLE_PERIOD_EXIT                = 10
VOL_PERIOD                        = 20
# Round std. Minimum is 0.1%.
BASIS_POINT_DP                    = 4

#--------------------------------------------------------------------------
# Indicators.
#--------------------------------------------------------------------------
def generate_turtle_indicators(df_ohlc):
    """
    Create turtle indicators as additional columns on symbol dataframe.
    """

    #--------------------------------------------------------------------------
    # Calculate rolling max/min of close.
    #--------------------------------------------------------------------------
    df_ohlc["close_entry_rolling_max"] = df_ohlc["split_adjusted_close"].rolling(TURTLE_PERIOD_ENTRY).max()
    df_ohlc["close_entry_rolling_min"] = df_ohlc["split_adjusted_close"].rolling(TURTLE_PERIOD_ENTRY).min()
    df_ohlc["close_exit_rolling_max" ] = df_ohlc["split_adjusted_close"].rolling(TURTLE_PERIOD_EXIT).max()
    df_ohlc["close_exit_rolling_min" ] = df_ohlc["split_adjusted_close"].rolling(TURTLE_PERIOD_EXIT).min()

    #--------------------------------------------------------------------------
    # Calculate true range + ATR.
    #--------------------------------------------------------------------------
    # Range 1: High - low.
    range_1 = df_ohlc["split_adjusted_high"] - df_ohlc["split_adjusted_low"]

    # Range 2: High - previous close.
    range_2 = df_ohlc["split_adjusted_high"] - df_ohlc["split_adjusted_close"].shift(1)

    # Range 3: Previous close - low.
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
    #--------------------------------------------------------------------------
    # Calculate standard deviation.
    #--------------------------------------------------------------------------
    # df_symbol["std"] = df_symbol["split_adjusted_close"].rolling(VOL_PERIOD).std()
    df_symbol["std"] = df_symbol["split_adjusted_close"].pct_change().rolling(VOL_PERIOD).std()
    df_symbol["std"] = df_symbol["std"].round(BASIS_POINT_DP)

    # Inverse standard deviation.
    df_symbol["inv_std"] = df_symbol["std"].rdiv(1)
    df_symbol["inv_std"] = df_symbol["inv_std"].replace(np.inf, np.nan)


def generate_volatile_filter(df_ohlc):
    #--------------------------------------------------------------------------
    # Calculate rolling max of single day absolute percent change.
    #--------------------------------------------------------------------------
    df_ohlc["abs_pct_rolling_max"] = df_ohlc["split_adjusted_close"].pct_change().abs().mul(100).rolling(SINGLE_DAY_VOLATILITY_FILTER_DAYS).max()

    # Disqualify symbols with a single day move exceeding 15% in the past 90 days.
    df_ohlc["disqualify_volatile"] = (df_ohlc["abs_pct_rolling_max"] > SINGLE_DAY_VOLATILITY_FILTER_PCT).astype(int)





class SystematicMomentum(Algo):

    #--------------------------------------------------------------------------
    # Class fields for base Algo class.
    #--------------------------------------------------------------------------
    symbol_universe = get_sp500_symbols_list()
    market_benchmark = 'SPY'
    symbol_indicator_list = [
        generate_turtle_indicators,
        generate_price_pct_change_std_and_inv_std,
    ]
    market_indicator_list = []

    #--------------------------------------------------------------------------
    # Methods.
    #--------------------------------------------------------------------------
    def __init__(self):
        super().__init__()


