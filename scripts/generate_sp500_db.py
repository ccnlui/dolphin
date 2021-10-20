#  File: generate_sp500_db.py
#  Date: 2021-10-19
#    By: Calvin
# Brief: Create sp500 membership table in database using sp500-master csv file.


import pandas as pd
from collections import defaultdict
import configparser
from mysql.connector import connect, Error


#------------------------------------------------------------------------------
# Constants.
#------------------------------------------------------------------------------
SP500_DATA_PATH = "/home/calvin/source/data/sp500"
SP500_DATA_FILENAME = "sp500_2021-10-18.csv"
CREDENTIALS_FULLPATH = "/media/calvin/Backup Plus/backup/source/credentials/credentials.ini"


#------------------------------------------------------------------------------
# Functions.
#------------------------------------------------------------------------------
def generate_sp500_database(sp500_data_fullpath, overwrite=False):

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
        sql_query = "DROP DATABASE IF EXISTS sp500"
        cursor.execute(sql_query)

    sql_query = "CREATE DATABASE IF NOT EXISTS sp500"
    cursor.execute(sql_query)

    # Select database.
    sql_query = "USE sp500"
    cursor.execute(sql_query)

    if (overwrite):
        sql_query = "DROP TABLE IF EXISTS membership"
        cursor.execute(sql_query)

    sql_query = """
        CREATE TABLE membership (
            id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(16) NOT NULL,
            start_date DATE,
            end_date DATE
        )
    """
    cursor.execute(sql_query)

    #--------------------------------------------------------------------------
    # Create indexes.
    #--------------------------------------------------------------------------
    sql_query = """
        CREATE INDEX ix_symbol ON membership(symbol)
    """
    cursor.execute(sql_query)

    #------------------------------------------------------------------------------
    # Generate table data.
    #------------------------------------------------------------------------------
    df_sp500 = pd.read_csv(sp500_data_fullpath)

    intervals = defaultdict(list)
    start_dates = {}

    for row in df_sp500.itertuples():
        next_date = row.date
        tickers = row.tickers.split(',')
        next_tickers = set(tickers)
        curr_tickers = set(start_dates.keys())

        # Interval ends.
        for symbol in curr_tickers - next_tickers:
            curr_date = start_dates[symbol]
            intervals[symbol].append([curr_date, next_date])
            start_dates.pop(symbol)

        # Interval starts.
        for symbol in next_tickers - curr_tickers:
            start_dates[symbol] = next_date

    # Current sp500 members.
    for symbol, start_date in start_dates.items():
        intervals[symbol].append([start_date, None])

    #------------------------------------------------------------------------------
    # Insert into database.
    #------------------------------------------------------------------------------
    values = []
    for symbol, itvs in intervals.items():
        for itv in itvs:
            values.append((symbol, itv[0], itv[1]))
    values.sort()

    sql_query = """
        INSERT INTO membership (symbol, start_date, end_date)
        VALUES (%s, %s, %s)
    """

    try:
        cursor.executemany(sql_query, values)
    except:
        print("[ERROR] Cannot insert into database: {} {}".format(sql_query, df_symbol_values))
        raise

    connection.commit()

    return


#------------------------------------------------------------------------------
# Entry point.
#------------------------------------------------------------------------------
if __name__ == "__main__":
    sp500_data_fullpath = "{}/{}".format(SP500_DATA_PATH, SP500_DATA_FILENAME)
    generate_sp500_database(sp500_data_fullpath, overwrite=True)
