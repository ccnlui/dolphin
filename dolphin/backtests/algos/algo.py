#  File: algo.py
#  Date: 2021-10-25
#    By: Calvin
# Brief: Base class of a trading algo.

from datetime import datetime
from backtests.exceptions import NotImplementedError


class Algo:
    """
    Abstract base class of a trading algo.

    Every algo, on a high level, should specify the following:
        1. Symbol universe
        2. Entry + Exit
        3. Position size.

    An algo needs to execute the following steps before a backtest:
        1. Generate market indicators.
        2. Append market indicators to symbol dataframe.
        1. Generate symbol indicators.
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


    def get_trade_frequency(self):
        raise NotImplementedError("Need to implement get_trade_frequency() method.")


    def rank_symbols(self):
        raise NotImplementedError("Need to implement rank_symbols() method.")


    def calculate_symbol_weights(self):
        raise NotImplementedError("Need to implement calculate_symbol_weights() method.")


    #--------------------------------------------------------------------------
    # Methods.
    #--------------------------------------------------------------------------
    def generate_market_indicators(self, df_market):

        print("[{}] [INFO] Generating market indicators: {}".format(datetime.now().isoformat(), df_market.symbol.iloc[0]))
        import ipdb; ipdb.set_trace()
        for indicator_method in self.market_indicator_list:
            indicator_method(df_market)

        return df_market


    def append_market_indicators_to_symbol(self, df_symbol_list, df_market):
        df_symbol_list = [ df.join(df_market, on='date') for df in df_symbol_list ]
        return df_symbol_list


    def generate_symbol_indicators(self, df_symbol):

        print("[{}] [INFO] Generating symbol indicators: {}".format(datetime.now().isoformat(), df_symbol.symbol.iloc[0]))
        for indicator_method in self.symbol_indicator_list:
            indicator_method(df_symbol)

        return df_symbol


    def generate_all_symbol_indicators(self, df_symbol_list):

        # Generate symbol indicators in parallel.
        pool = mp.Pool(mp.cpu_count()-2)
        df_symbol_list = pool.map(self.generate_symbol_indicators, df_symbol_list)
        pool.close()

        # Combine all symbol dataframes together.
        df_symbol_universe = pd.concat(df_symbol_list, ignore_index=True)
        df_symbol_universe.date = pd.to_datetime(df_symbol_universe.date)

        # Write to csv.
        df_symbol_universe.to_csv("{}/algo_indicators.csv".format(CSV_ROOT_PATH), index=False)

        return df_symbol_universe


    def prepare_for_backtest(self, df_symbol_list, df_market):

        df_market = self.generate_market_indicators(df_market)
        df_symbol_list = self.append_market_indicators_to_symbol(df_symbol_list, df_market)
        df_symbol_universe = self.generate_all_symbol_indicators(df_symbol_list)
        df_symbol_universe = self.rank_symbols(df_symbol_universe)
        df_symbol_universe = self.calculate_symbol_weights(df_symbol_universe)

        df_symbol_universe.sort_values(by=["date", "symbol"], inplace=True)

        return df_symbol_universe






