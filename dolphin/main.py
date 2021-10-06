from backtests.services.backtest import pandas_algo_turtle
from datetime import datetime


#------------------------------------------------------------------------------
# Entry point.
#------------------------------------------------------------------------------
if __name__ == '__main__':
    #--------------------------------------------------------------------------
    # Measure execution time.
    #--------------------------------------------------------------------------
    start_time = datetime.now()

    turtle = pandas_algo_turtle()

    # print("[{}] Checking SP500 constituents...".format(datetime.now().isoformat()))
    # turtle.check_sp500_constituents(turtle.START_DATE, turtle.END_DATE)

    #--------------------------------------------------------------------------
    # Build database and simulate trading.
    #--------------------------------------------------------------------------
    turtle.load_market_data_from_db(turtle.symbol_universe, turtle.START_DATE, turtle.END_DATE, turtle.INTERVAL)
    turtle.backtest_turtle_rules(turtle.START_DATE, turtle.END_DATE)
    turtle.df.to_csv("{}/algo_turtle.csv".format(turtle.CSV_ROOT_PATH), index=False)

    #--------------------------------------------------------------------------
    # Simulate trading only.
    #--------------------------------------------------------------------------
    # turtle.load_market_data_from_csv("/home/calvin/source/python/pandas/csv/algo_turtle_indicators.csv")
    # turtle.backtest_turtle_rules(turtle.START_DATE, turtle.END_DATE)
    # turtle.df.to_csv("{}/algo_turtle.csv".format(turtle.CSV_ROOT_PATH), index=False)

    #--------------------------------------------------------------------------
    # Performance analysis.
    #--------------------------------------------------------------------------
    turtle.generate_trade_summary()
    turtle.generate_tear_sheet()

    turtle.dump_weights_entries()
    turtle.dump_cashflow_entries()
    
    turtle.generate_backtest_graph(turtle.START_DATE, turtle.END_DATE)

    # turtle.generate_trade_summary_graph("/home/calvin/source/python/pandas/csv/atr/algo_turtle_trade_summary.csv")

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