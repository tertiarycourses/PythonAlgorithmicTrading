"""
Id:             correlation.py
Copyright:      2018 xiaokang.guan All rights reserved.
Description:    Utility functions for correlations.
"""

import pandas as pd
from data_hub import DataHub
import datetime
import logging

def calculateCorrelation(ts1, ts2):
    """
    Calculate the correlation between the two given time series.
    :param ts1: Time series 1 in pandas Series format
    :param ts2: Time series 2 in pandas Series format
    :return: correlation value
    """

    # Combine 2 time series and only keep those common indices
    df = pd.concat([ts1, ts2], axis=1, join='inner')
    return df.corr()


def calculateBeta(ts, tsMarket):
    """
    Calculate the Beta for the given time series
    :param ts: The time series to calculate Beta for, in pandas.Series format
    :param tsMarket: The Market time series, in pandas.Series format
    :return: Beta value
    """
    pass

def main():
    """
    Main entry point.
    """
    dataHub = DataHub()
    startDate = datetime.date(2018,1,1)
    endDate = datetime.date.today()
    symbols = ['AAPL', 'SPY']
    data = dataHub.downloadDataFromYahoo(startDate, endDate, symbols)
    ts1 = data.values()[0].loc[:, 'Close'].rename(data.keys()[0])
    ts2 = data.values()[1].loc[:, 'Close'].rename(data.keys()[1])

    corr = calculateCorrelation(ts1, ts2)
    logging.info('Correlation = %.2f', corr)
