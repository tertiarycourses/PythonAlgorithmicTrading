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