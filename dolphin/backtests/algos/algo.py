#  File: algo.py
#  Date: 2021-10-25
#    By: Calvin
# Brief: Base class of a trading algo.


from backtests.exceptions import NotImplementedError


class Algo:
    """
    Abstract base class of a trading algo.

    Every algo, on a high level, should specify the following:
        1. Symbol universe
        2. Entry + Exit
        3. Position size.

    Algo needs to do the following before a backtest can be run:
        1. Generate symbol indicators.
        2. Generate market indicators.
        3. Rank symbols.
        4. Assign weights.

    The following should be defined as class fields in the subclass:
        - symbol_universe (List[str])
        - market_benchmark (str)
        - symbol_indicator_list (List[function])
        - market_indicator_list (List[function])

    Functions in indicator lists will be used to generate indicators
    on either the symbol level or the market level every day, in the form
    of additional columns.
    """


    def __init__(self):
        if not hasattr(self, "symbol_universe"):
            raise NotImplementedError("Need to define class field symbol_universe in subclass.")

        if not hasattr(self, "market_benchmark"):
            raise NotImplementedError("Need to define class field market_benchmark in subclass.")

        if not hasattr(self, "symbol_indicator_list"):
            raise NotImplementedError("Need to define class field symbol_indicator_list in subclass.")

        if not hasattr(self, "market_indicator_list"):
            raise NotImplementedError("Need to define class field indicator_list in subclass.")


    #--------------------------------------------------------------------------
    # Abstract methods.
    #--------------------------------------------------------------------------
    def buy_signal(self):
        raise NotImplementedError("Need to implement buy_signal() method.")


    def sell_signal(self):
        raise NotImplementedError("Need to implement sell_signal() method.")


    def calculate_symbol_weights(self):
        raise NotImplementedError("Need to implement calculate_symbol_weights() method.")


    #--------------------------------------------------------------------------
    # Methods.
    #--------------------------------------------------------------------------
    def generate_symbol_indicators(self, df_symbol):

        print("[{}] [INFO] Generating indicators for symbol: {}".format(datetime.now().isoformat(), df_symbol.symbol.iloc[0]))

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

        # Disqualify symbols with 0 standard deviation.
        df_symbol["disqualify_stale"] = (df_symbol["std"] == 0).astype(int)

        return df_symbol
