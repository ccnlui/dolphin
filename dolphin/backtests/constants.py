#        File: constants.py
#        Date: 2021-10-05
#          By: Calvin
#       Brief: Constants used by backtest app.


from datetime import date

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