#  File: systematic_momentum.py
#  Date: 2021-10-25
#    By: Calvin
# Brief: Systematic momentum trading algo.


from backtests.algos.algo import Algo
from backtests.services.market_data import get_sp500_symbols_list


class SystematicMomentum(Algo):

    #--------------------------------------------------------------------------
    # Class fields for base Algo class.
    #--------------------------------------------------------------------------
    symbol_universe = get_sp500_symbols_list()
    market_benchmark = 'SPY'
    symbol_indicator_list = []
    market_indicator_list = []

    #--------------------------------------------------------------------------
    # Methods.
    #--------------------------------------------------------------------------
    def __init__(self):
        super().__init__()


