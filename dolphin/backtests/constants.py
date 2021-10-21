#        File: constants.py
#        Date: 2021-10-05
#          By: Calvin
#       Brief: Constants used by backtest app.


from datetime import date

#--------------------------------------------------------------------------
# Turtle algo parameters.
#--------------------------------------------------------------------------
TURTLE_PERIOD_ENTRY = 20
TURTLE_PERIOD_EXIT = 10
SINGLE_DAY_VOLATILITY_FILTER_DAYS = 90
SINGLE_DAY_VOLATILITY_FILTER_PCT = 15
ATR_PERIOD = 20
ATR_SMOOTHING_FACTOR = 1 / ATR_PERIOD
VOL_PERIOD = 20
YEARLY_TRADING_DAYS = 252
MOMENTUM_WINDOW = 125
MIN_MOMENTUM_SCORE = 40
PORTFOLIO_NUM_STOCK = 30

INITIAL_CAPITAL = 100000

# Replaced by inverse volatility.
DOLLAR_VOL_PCT = 2

# Exponential model: y = a * exp(b*x)
EXP_MODEL_GUESS_A = 4
EXP_MODEL_GUESS_B = 0.1

# Round ATR.
MODEL_PRECISION = 4

# Round std. Minimum is 1%.
BASIS_POINT_DP = 2

PENNY_PRICE = 1
EXPENSIVE_PRICE = 10000

MARKET_TREND_FILTER_DAYS = 200

#--------------------------------------------------------------------------
# Backtest period.
#--------------------------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = "2020-12-31"
INTERVAL = "1day"
PREFETCH_NUM_MONTH = 12

REGULAR_TRADING_HOURS_START_TIME = time.fromisoformat("09:30:00")
REGULAR_TRADING_HOURS_END_TIME = time.fromisoformat("16:00:00")

#--------------------------------------------------------------------------
# File paths.
#--------------------------------------------------------------------------
MARKET_DATA_ROOT_PATH = "/home/calvin/source/data/alpha_vantage/backup/2021-01-16/data"
LISTING_ROOT_PATH     = "/home/calvin/source/data/alpha_vantage/backup/2021-01-16/listing"
GRAPH_ROOT_PATH       = "/home/calvin/source/dolphin/graph"
LOG_ROOT_PATH         = "/home/calvin/source/dolphin/log"
CSV_ROOT_PATH         = "/home/calvin/source/dolphin/csv"

CREDENTIALS_FULLPATH = "/media/calvin/Backup Plus/backup/source/credentials/credentials.ini"

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