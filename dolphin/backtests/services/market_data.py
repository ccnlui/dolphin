#        File: market_data.py
#        Date: 2021-10-05
#          By: Calvin
#       Brief: Service that provides market data.


import configparser
from mysql.connector import connect, Error

import pandas as pd

from datetime import datetime, date, time, timedelta

#------------------------------------------------------------------------------
# Constants.
#------------------------------------------------------------------------------
SYMBOL_UNIVERSE_PATH = "/home/calvin/source/data/alpha_vantage/backup/2021-01-16/data"
CREDENTIALS_FULLPATH = "/media/calvin/Backup Plus/backup/source/credentials/credentials.ini"

#------------------------------------------------------------------------------
# Functions.
#------------------------------------------------------------------------------
def get_daily_split_adjusted_df(symbol, start_date_str, end_date_str):

    print("[{}] [INFO] Get daily split adjusted dataframe for {} between {} and {}".format(datetime.today().isoformat(), symbol, start_date_str, end_date_str))

    #--------------------------------------------------------------------------
    # Connect to MariaDB.
    #--------------------------------------------------------------------------
    config = configparser.ConfigParser()
    config.read(CREDENTIALS_FULLPATH)
    db_user = config["MariaDB"]["db_user"]
    db_passwd = config["MariaDB"]["db_passwd"]

    try:
        # connection = connect(host="localhost", user=db_user, passwd=db_passwd, database="alpha_vantage")
        connection = connect(host="localhost", user=db_user, passwd=db_passwd)
    except Error as e:
        print("[ERROR] {}".format(e))

    # Cursor is used to execute SQL statements.
    cursor = connection.cursor()

    #--------------------------------------------------------------------------
    # Read from database.
    #--------------------------------------------------------------------------
    # sql_query = """
    #     SELECT
    #         timestamp,
    #         symbol,
    #         open,
    #         high,
    #         low,
    #         close,
    #         adjusted_close,
    #         volume,
    #         dividend_amount,
    #         split_coefficient,
    #         EXISTS(
    #             SELECT 1
    #             FROM sp500.membership m
    #             WHERE
    #                 m.start_date <= da.timestamp
    #                 AND ( m.end_date IS NULL or da.timestamp < m.end_date )
    #         ) AS in_sp500
    #     FROM alpha_vantage.daily_adjusted da
    #     WHERE symbol = "{}" AND "{}" <= timestamp AND timestamp <= "{}"
    # """.format(symbol, start_date_str, end_date_str)
    sql_query = """
        SELECT
            da.timestamp,
            da.symbol,
            da.open,
            da.high,
            da.low,
            da.close,
            da.adjusted_close,
            da.volume,
            da.dividend_amount,
            da.split_coefficient,
            m.start_date in_sp500_start,
            m.end_date in_sp500_end
        FROM alpha_vantage.daily_adjusted da
        LEFT JOIN sp500.membership m
        ON
            da.symbol = m.symbol
            AND m.start_date <= da.timestamp
            AND (m.end_date IS NULL OR da.timestamp < m.end_date)
        WHERE da.symbol = "{}" AND "{}" <= da.timestamp AND da.timestamp <= "{}"
    """.format(symbol, start_date_str, end_date_str)
    cursor.execute(sql_query)
    result = cursor.fetchall()

    #--------------------------------------------------------------------------
    # Create dataframe.
    #--------------------------------------------------------------------------
    df = pd.DataFrame(result, columns=["timestamp", "symbol", "open", "high", "low", "close", "adjusted_close", "volume", "dividend_amount", "split_coefficient", "in_sp500_start", "in_sp500_end"])

    #--------------------------------------------------------------------------
    # Calculate split-adjusted close.
    #--------------------------------------------------------------------------
    if not df.empty:
        total_split_coefficient = df.split_coefficient.product()

        df["split_coefficient_inv"] = df.split_coefficient.rdiv(1)
        df["split_coefficient_inv_cum"] = df.split_coefficient_inv.cumprod() * total_split_coefficient
        df["split_adjusted_open"] = df.open / df.split_coefficient_inv_cum
        df["split_adjusted_high"] = df.high / df.split_coefficient_inv_cum
        df["split_adjusted_low"] = df.low / df.split_coefficient_inv_cum
        df["split_adjusted_close"] = df.close / df.split_coefficient_inv_cum

        # Keep useful columns, drop the rest.
        df = df.loc[:, ["timestamp", "symbol", "split_adjusted_open", "split_adjusted_high", "split_adjusted_low", "split_adjusted_close", "in_sp500_start", "in_sp500_end"]]

        # Rename column timestamp to date.
        df = df.rename(columns={"timestamp": "date"})
    else:
        print("[{}] [WARNING] daily split adjusted dataframe for symbol: {} is empty.".format(datetime.now().isoformat(), symbol))

    return df


def get_symbol_list_daily_split_adjusted_df_list(symbol_list, start_date_str, end_date_str):

    df_list = []

    for symbol in symbol_list:
        df_symbol = get_daily_split_adjusted_df(symbol, start_date_str, end_date_str)
        if not df_symbol.empty:
            df_list.append(df_symbol)

    return df_list


def get_sp500_symbols_list():

    print("[INFO] Get all symbols ever been in sp500...")
    import ipdb; ipdb.set_trace()

    #--------------------------------------------------------------------------
    # Connect to MariaDB.
    #--------------------------------------------------------------------------
    config = configparser.ConfigParser()
    config.read(CREDENTIALS_FULLPATH)
    db_user = config["MariaDB"]["db_user"]
    db_passwd = config["MariaDB"]["db_passwd"]

    try:
        # connection = connect(host="localhost", user=db_user, passwd=db_passwd, database="alpha_vantage")
        connection = connect(host="localhost", user=db_user, passwd=db_passwd)
    except Error as e:
        print("[ERROR] {}".format(e))

    # Cursor is used to execute SQL statements.
    cursor = connection.cursor()

    #--------------------------------------------------------------------------
    # Read from database.
    #--------------------------------------------------------------------------
    sql_query = "SELECT DISTINCT symbol FROM sp500.membership;"
    cursor.execute(sql_query)
    result = cursor.fetchall()

    # Flatten list of tuples.
    result = [ symbol for row in result for symbol in row ]
    
    if not result:
        print("[{}] [WARNING] sp500 list is empty.")

    return result
