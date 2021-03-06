#        File: constants.py
#        Date: 2021-10-05
#          By: Calvin
#       Brief: Constants used by backtest app.


from datetime import date, time

#--------------------------------------------------------------------------
# Backtest params.
#--------------------------------------------------------------------------
INITIAL_CAPITAL = 100000

START_DATE = "2011-10-26"
END_DATE = "2020-12-31"
INTERVAL = "1day"
PREFETCH_NUM_MONTH = 12

REGULAR_TRADING_HOURS_START_TIME = time.fromisoformat("09:30:00")
REGULAR_TRADING_HOURS_END_TIME = time.fromisoformat("16:00:00")

#--------------------------------------------------------------------------
# Constants.
#--------------------------------------------------------------------------
TRADE_DAILY = "trade_daily"
TRADE_WEEKLY_WEDNESDAY = "trade_weekly_wednesday"
TRADE_MONTHLY = "trade_monthly"

#--------------------------------------------------------------------------
# File paths.
#--------------------------------------------------------------------------
MARKET_DATA_ROOT_PATH = "/home/calvin/source/data/alpha_vantage/backup/2021-01-16/data"
LISTING_ROOT_PATH     = "/home/calvin/source/data/alpha_vantage/backup/2021-01-16/listing"
GRAPH_ROOT_PATH       = "/home/calvin/source/dolphin/graph"
LOG_ROOT_PATH         = "/home/calvin/source/dolphin/log"
CSV_ROOT_PATH         = "/home/calvin/source/dolphin/csv"
SP500_DATA_PATH       = "/home/calvin/source/data/sp500"
SP500_DATA_FILENAME   = "sp500_2021-10-18.csv"
CREDENTIALS_FULLPATH  = "/media/calvin/Backup Plus/backup/sandisk-extreme/source/credentials/credentials.ini"

#--------------------------------------------------------------------------
# Holidays
#--------------------------------------------------------------------------
# Reference.
# https://en.wikipedia.org/wiki/Trading_day
cboe_holidays = {
    date(2020, 1, 1)    : "New Year's Day",
    date(2020, 1, 20)   : "Martin Luther King Jr. Day",
    date(2020, 2, 17)   : "President's Day",
    date(2020, 4, 10)   : "Good Friday",
    date(2020, 5, 25)   : "Memorial Day",
    date(2020, 7, 3)    : "Independence Day (Observed)",
    date(2020, 9, 7)    : "Labor Day",
    date(2020, 11, 26)  : "Thanksgiving Day",
    date(2020, 12, 25)  : "Christmas Day",

    date(2019, 1, 1)    : "New Year's Day",
    date(2019, 1, 21)   : "Martin Luther King Jr. Day",
    date(2019, 2, 18)   : "President's Day",
    date(2019, 4, 19)   : "Good Friday",
    date(2019, 5, 27)   : "Memorial Day",
    date(2019, 7, 4)    : "Independence Day",
    date(2019, 9, 2)    : "Labor Day",
    date(2019, 11, 28)  : "Thanksgiving Day",
    date(2019, 12, 25)  : "Christmas Day",

    date(2018, 1, 1)    : "New Year's Day",
    date(2018, 1, 15)   : "Martin Luther King Jr. Day",
    date(2018, 2, 19)   : "President's Day",
    date(2018, 3, 30)   : "Good Friday",
    date(2018, 5, 28)   : "Memorial Day",
    date(2018, 7, 4)    : "Independence Day",
    date(2018, 9, 3)    : "Labor Day",
    date(2018, 11, 22)  : "Thanksgiving Day",
    date(2018, 12, 5)   : "National Day of Mourning for George H.W. Bush",
    date(2018, 12, 25)  : "Christmas Day",
}