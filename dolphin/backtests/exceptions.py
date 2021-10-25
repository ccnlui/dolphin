#  File: exceptions.py
#  Date: 2021-10-25
#    By: Calvin
# Brief: Common exceptions for the backtest app.


class NotImplementedError(Exception):
    def __init__(self, message):
        self.message = message
