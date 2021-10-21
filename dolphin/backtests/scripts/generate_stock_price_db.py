#  File: generate_stock_price_db.py
#  Date: 2021-10-04
#    By: Calvin
# Brief: Create stock price table in database using Alpha Vantage data.


import configparser
from mysql.connector import connect, Error

import pandas as pd

from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta

import plotly.express as px
import os


#------------------------------------------------------------------------------
# Constants.
#------------------------------------------------------------------------------
MARKET_DATA_ROOT_PATH = "/home/calvin/source/data/alpha_vantage/backup/2021-01-16/data"
CREDENTIALS_FULLPATH  = "/media/calvin/Backup Plus/backup/source/credentials/credentials.ini"
PREFETCH_NUM_MONTH = 12
START_DATE = "2010-01-01"
END_DATE = "2020-12-31"
INTERVAL = "1day"

#------------------------------------------------------------------------------
# Functions.
#------------------------------------------------------------------------------
def generate_daily_adjusted_database(symbol_universe_path, start_date_str, end_date_str, interval, overwrite=False):
    #----------------------------------------------------------------------
    # Initialize dates.
    #----------------------------------------------------------------------
    start_date = date.fromisoformat(start_date_str)
    end_date = date.fromisoformat(end_date_str)
    prefetch_start_date = start_date - relativedelta(months=PREFETCH_NUM_MONTH)

    start_year = prefetch_start_date.year
    end_year = end_date.year
    curr_year = start_year
    next_year = None

    #--------------------------------------------------------------------------
    # Connect to MariaDB.
    #--------------------------------------------------------------------------
    config = configparser.ConfigParser()
    config.read(CREDENTIALS_FULLPATH)
    db_user = config["MariaDB"]["db_user"]
    db_passwd = config["MariaDB"]["db_passwd"]

    try:
        connection = connect(host="localhost", user=db_user, passwd=db_passwd)
    except Error as e:
        print("[ERROR] {}".format(e))

    # Cursor is used to execute SQL statements.
    cursor = connection.cursor()

    #--------------------------------------------------------------------------
    # Create database and table.
    #--------------------------------------------------------------------------
    sql_query = ""

    if (overwrite):
        sql_query = "DROP DATABASE IF EXISTS alpha_vantage"
        cursor.execute(sql_query)

    sql_query = "CREATE DATABASE IF NOT EXISTS alpha_vantage"
    cursor.execute(sql_query)

    # Select databse.
    sql_query = "USE alpha_vantage"
    cursor.execute(sql_query)

    if (overwrite):
        sql_query = "DROP TABLE IF EXISTS daily_adjusted"
        cursor.execute(sql_query)

    sql_query = """
        CREATE TABLE daily_adjusted (
            id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            timestamp DATE,
            symbol VARCHAR(16),
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adjusted_close DOUBLE,
            volume BIGINT UNSIGNED,
            dividend_amount DOUBLE,
            split_coefficient DOUBLE
        )
    """
    cursor.execute(sql_query)

    #--------------------------------------------------------------------------
    # Create indexes.
    #--------------------------------------------------------------------------
    sql_query = """
        CREATE INDEX ix_symbol ON daily_adjusted(symbol)
    """
    cursor.execute(sql_query)

    sql_query = """
        CREATE UNIQUE INDEX ix_timestamp_symbol ON daily_adjusted(timestamp, symbol)
    """
    cursor.execute(sql_query)

    #--------------------------------------------------------------------------
    # Create symbol universe list.
    #--------------------------------------------------------------------------
    symbol_universe = os.listdir(symbol_universe_path)
    # symbol_universe = ["AAPL", "FB", "AMZN", "GOOGL", "TSLA"]
    symbol_universe.sort()

    if "raw" in symbol_universe:
        symbol_universe.remove("raw")

    #--------------------------------------------------------------------------
    # Read csv file for each symbol.
    #--------------------------------------------------------------------------
    for symbol in symbol_universe:

        print("[{}] [INFO] Reading in market data for symbol: {}".format(datetime.today().isoformat(), symbol))

        df_symbol = pd.DataFrame()

        if interval == "1day":
            #------------------------------------------------------------------
            # Load price data one year at a time.
            #------------------------------------------------------------------
            curr_year = start_year

            while curr_year <= end_year:

                next_year = curr_year + 1

                #--------------------------------------------------------------
                # Check if historical data exists.
                #--------------------------------------------------------------
                symbol_data_filepath = "{}/{}/{}/Alpha_Vantage_{}_{}_{}_adjusted.csv".format(
                                        MARKET_DATA_ROOT_PATH,
                                        symbol,
                                        interval,
                                        symbol,
                                        interval,
                                        curr_year)
                if not os.path.exists(symbol_data_filepath):
                    # print("[ERROR] {} does not exist.".format(symbol_data_filepath))
                    # Next iteration.
                    curr_year = next_year
                    continue


                #--------------------------------------------------------------
                # Build dataframe.
                #--------------------------------------------------------------
                # Read current year csv.
                df_curr_year = pd.read_csv(symbol_data_filepath, dtype=str)

                # Filter date range.
                df_curr_year = df_curr_year.loc[ (df_curr_year.timestamp >= prefetch_start_date.isoformat()) & (df_curr_year.timestamp <= end_date.isoformat()) ]

                # Concatenate current year dataframe.
                df_symbol = pd.concat([df_symbol, df_curr_year], ignore_index=True)

                #--------------------------------------------------------------
                # Next iteration.
                #--------------------------------------------------------------
                curr_year = next_year


            #------------------------------------------------------------------
            # Append symbol dataframe.
            #------------------------------------------------------------------
            if df_symbol.empty:
                # print("[ERROR] Cannot insert empty {} dataframe to database.".format(symbol))
                continue

            # Keep useful columns, drop the rest.
            df_symbol["symbol"] = symbol
            df_symbol = df_symbol.loc[:, ["timestamp", "symbol", "open", "high", "low", "close", "adjusted_close", "volume", "dividend_amount", "split_coefficient"]]

            #------------------------------------------------------------------
            # Insert to database.
            #------------------------------------------------------------------
            df_symbol_values = []

            for row in df_symbol.itertuples(index=False):
                df_symbol_values.append(row)

            sql_query = """
                INSERT INTO daily_adjusted (timestamp, symbol, open, high, low, close, adjusted_close, volume, dividend_amount, split_coefficient)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            try:
                cursor.executemany(sql_query, df_symbol_values)
            except:
                print("[ERROR] Cannot insert into database: {} {}".format(sql_query, df_symbol_values))
                raise

    connection.commit()

    return


#------------------------------------------------------------------------------
# Entry point.
#------------------------------------------------------------------------------
if __name__ == "__main__":
    generate_daily_adjusted_database(MARKET_DATA_ROOT_PATH, START_DATE, END_DATE, INTERVAL, True)
