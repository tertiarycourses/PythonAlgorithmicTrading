"""
Id:             data_hub.py
Copyright:      2018 xiaokang.guan All rights reserved.
Description:    Data hub to download data from web.
"""

import pandas_datareader.data as web
import logging
import datetime
import numpy as np

DATA_SOURCE_YAHOO = 'yahoo'
DATA_SOURCE_GOOGLE = 'google'


class DataHub:
    def __init__(self):
        pass

    def _downloadData(self, startDate=datetime.date(2017, 1, 1), endDate=datetime.date.today(), symbols=['AAPL', 'SPY'], dataSource=DATA_SOURCE_YAHOO):
        """
        Downland stock historical data from Yahoo finance, histPanel is a Panel, already in ascending order, e.g:
        Dimensions: 6 (items) x 2 (major_axis) x 2 (minor_axis)
        Items axis(0): Open to Adj Close
        Major_axis axis(1): 2016-10-11 00:00:00 to 2016-10-12 00:00:00
        Minor_axis axis(2): SPY to ^N225

        We now use dict of DataFrames to replace panel as Panel is depreciated in Pandas.
        key: symbol
        value: DataFrame with dates as index and "Open" etc as columns

        Now we allow different indexes across different symbol DataFrames
        And we will simply remove all 0 or NaN in every DataFrame
        """
        symbolData = dict()
        for symbol in symbols:
            try:
                df = web.DataReader(symbol, dataSource, startDate, endDate)
            except:
                logging.error('DataHub: _downloadData: Cannot download historical data for symbol={}'.format(symbol))
                continue
            symbolData[symbol] = df

        # Cleanse data: remove dates where there is NaN or 0
        for symbol, df in symbolData.iteritems():
            df = df.replace(0, np.nan)
            df = df.dropna()
            # Remove duplicated date index
            df = df[~df.index.duplicated(keep='first')]
            symbolData[symbol] = df.sort_index(ascending=True)

        logging.info('============================================================')
        logging.info('DataHub: downlaodData: Completed startDate={}, endDate={}'.format(startDate, endDate))
        logging.info('============================================================')
        return symbolData

    def downloadDataFromYahoo(self, startDate, endDate, symbols):
        return self._downloadData(startDate, endDate, symbols, DATA_SOURCE_YAHOO)

    def downloadDataFromGoogle(self, startDate, endDate, symbols):
        return self._downloadData(startDate, endDate, symbols, DATA_SOURCE_GOOGLE)