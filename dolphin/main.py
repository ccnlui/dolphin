from datetime import datetime

from backtests.constants import (
    START_DATE,
    END_DATE,
    PORTFOLIO_NUM_STOCK,
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

    # Write only relevant rows to CSV file.
    df = df.loc[ (df.rank <= PORTFOLIO_NUM_STOCK) | (~df.market_value.isna()) | (~df.cashflow.isna()) ]
    df.to_csv("{}/backtest_systematic_momentum.csv".format(CSV_ROOT_PATH), index=False)

    #--------------------------------------------------------------------------
    # Performance analysis.
    #--------------------------------------------------------------------------
    turtle.generate_trade_summary(df)
    turtle.generate_tear_sheet(df)
    turtle.generate_backtest_graph(df, START_DATE, END_DATE)

    #--------------------------------------------------------------------------
    # Graph symbol daily adjusted.
    #--------------------------------------------------------------------------
    # turtle.generate_symbol_graph("MJRC", turtle.START_DATE, turtle.END_DATE)

    #--------------------------------------------------------------------------
    # Generate symbol indicators.
    #--------------------------------------------------------------------------
    # df_symbol = get_daily_split_adjusted_df("AAPL", turtle.START_DATE, turtle.END_DATE)
    # turtle.generate_symbol_indicators(df_symbol)

    #--------------------------------------------------------------------------
    # Measure execution time.
    #--------------------------------------------------------------------------
    end_time = datetime.now()
    elapsed_sec = (end_time - start_time).total_seconds()

    print("--- Time elapsed: {:.6f} seconds ---".format(elapsed_sec))