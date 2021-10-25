#  File: algo.py
#  Date: 2021-10-25
#    By: Calvin
# Brief: Base class of a trading algo.


from backtests.exceptions import NotImplementedError


class Algo:
    """
    Abstract base class of a trading algo.

    Every algo should specify the following:
        1. Symbol universe
        2. Market benchmark
        3. Symbol indicators
        4. Market indicators
        5. Symbol weights
        6. Entry + Exit

    The following should be defined as class fields in the subclass:
        - symbol_universe (List[str])
        - market_benchmark (str)
        - symbol_indicator_list (List[function])
        - market_indicator_list (List[function])

    Indicator lists will be used to generate indicators
    on either the symbol level or the market level every day. Every indicator
    will be created as a new column on the DataFrame.

    calculate_symbol_weights() will be run after all indicators are generated.
    """

    def __init__(self):
        if not hasattr(self, "symbol_indicator_list"):
            raise NotImplementedError("Need to define class field symbol_indicator_list in subclass.")

        if not hasattr(self, "market_indicator_list"):
            raise NotImplementedError("Need to define class field indicator_list in subclass.")


    def buy_signal(self):
        raise NotImplementedError("Need to implement buy_signal() method.")


    def sell_signal(self):
        raise NotImplementedError("Need to implement sell_signal() method.")


    def calculate_symbol_weights(self):
        raise NotImplementedError("Need to implement calculate_symbol_weights() method.")
