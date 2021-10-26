from datetime import datetime

from backtests.constants import (
    START_DATE,
    END_DATE,
    CSV_ROOT_PATH,
)
from backtests.services.backtest import BacktestService
from backtests.algos.systematic_momentum import SystematicMomentum


#------------------------------------------------------------------------------
# Entry point.
#------------------------------------------------------------------------------
if __name__ == '__main__':
    #--------------------------------------------------------------------------
    # Measure execution time.
    #--------------------------------------------------------------------------
    start_time = datetime.now()

    #--------------------------------------------------------------------------
    # Build database and simulate trading.
    #--------------------------------------------------------------------------
    backtest_service = BacktestService()
    df = backtest_service.backtest_algo(SystematicMomentum, START_DATE, END_DATE)

    #--------------------------------------------------------------------------
    # Performance analysis.
    #--------------------------------------------------------------------------
    backtest_service.generate_trade_summary(df)
    backtest_service.generate_tear_sheet(df)
    backtest_service.generate_backtest_graph(df, START_DATE, END_DATE)

    #--------------------------------------------------------------------------
    # Graph symbol daily adjusted.
    #--------------------------------------------------------------------------
    # backtest_service.generate_symbol_graph("MJRC", START_DATE, END_DATE)

    #--------------------------------------------------------------------------
    # Measure execution time.
    #--------------------------------------------------------------------------
    end_time = datetime.now()
    elapsed_sec = (end_time - start_time).total_seconds()

    print("--- Time elapsed: {:.6f} seconds ---".format(elapsed_sec))