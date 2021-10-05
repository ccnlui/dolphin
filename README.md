# Setup

## Install MariaDB
1. Refer to steps from howtos.


## Create virtual environment using Anaconda
1. `conda create -n dolphin-env python=3.9`
2. `conda activate dolphin-env`
3. `pip install -r requirements.txt`


## Load Alpha Vantage stock data into database
1. Make sure `credentials.ini` exists in `CREDENTIALS_FULLPATH`
2. Make sure daily adjusted close data exists in `SYMBOL_UNIVERSE_PATH`
3. `cd scripts`
4. `python generate_stock_price_db.py`
