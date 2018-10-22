"""
Id:             arima.py
Copyright:      2018 xiaokang.guan All rights reserved.
Description:    AutoRegressive Integrated Moving Average method for Time Series analysis.
"""

import pandas_datareader.data as web
from datetime import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

TRAIN_SIZE = 0.2
FORECAST_STEPS = 20
STOCKS_PREDICT = ['SPY', 'AAPL']


def downloadData(startDate, endDate):
    '''
    Downland stock historical data from Yahoo finance, histPanel is a Panel, already in ascending order, e.g:
    Dimensions: 6 (items) x 2 (major_axis) x 2 (minor_axis)
    Items axis: Open to Adj Close
    Major_axis axis: 2016-10-11 00:00:00 to 2016-10-12 00:00:00
    Minor_axis axis: SPY to ^N225
    '''
    histPanel = web.DataReader(STOCKS_PREDICT, 'google', startDate, endDate)

    # Cleanse data: remove dates where there is NaN and 0
    for item, df in histPanel.iteritems():
        for minor, ts in df.iteritems():
            for major, value in ts.iteritems():
                if float(value) == 0:
                    histPanel[item][minor][major] = 'NaN'

    histPanel = histPanel.dropna(axis=1, how='any')

    return histPanel

def doAnalysis(panel):
    '''Perform analysis on the ARIMA model'''
    closeDF = panel['Close']
    ts = closeDF['AAPL']
    ts1 = ts.shift(1).dropna()
    diff = (ts1 - ts).dropna()

    # Analyze TimeSeries after 1st differencing
    #diff.plot()
    #pyplot.subplot(311)
    #pyplot.title('Auto Correlation after 1st Differencing')
    autocorrelation_plot(diff)
    pyplot.show()

    # Fit ARIMA model
    model = ARIMA(ts, order=(1,1,0))
    modelFit = model.fit(disp=0)
    print modelFit.summary()

    # Analyze model residuals
    resid = modelFit.resid
    print resid.describe()
    pyplot.subplot(311)
    pyplot.title('modelFit Residuals')
    resid.plot()

    pyplot.subplot(312)
    pyplot.title('modelFit Residuals Distribution')
    resid.plot(kind='kde')

    # Analyze model forecast
    data = ts.tolist()
    trainSize = int(ts.size * TRAIN_SIZE)
    trainData, testData = data[:trainSize], data[trainSize:]
    forecast, _, _ = modelFit.forecast(steps=FORECAST_STEPS)
    pyplot.subplot(313)
    pyplot.title('10-day Forecast vs Expected')
    pyplot.plot(range(FORECAST_STEPS), forecast.tolist(), 'r^', testData[:FORECAST_STEPS], 'bo')
    #pyplot.axis([0, 20, 200, 270])

    pyplot.tight_layout()
    pyplot.show()

def doForecast(panel):
    '''Perform forecast using ARIMA model'''
    closeDF = panel['Close']
    ts = closeDF['SPY']
    tsWeekly = ts.resample('W-MON').last()
    values = tsWeekly.tolist()
    trainSize = int(ts.size * TRAIN_SIZE)

    train, test = values[0:trainSize], values[trainSize:ts.size]
    history = train
    predictions = list()
    errorPcts = list()

    for t in range(len(test)):
        model = ARIMA(history, order=(2, 1, 0))
        modelFit = model.fit(disp=0)
        output = modelFit.forecast()
        predicted = output[0][0] #output contains forecast, stdError, confInt
        predictions.append(predicted)

        observed = test[t]
        history.append(observed)

        errorPct = (predicted-observed)/observed * 100
        errorPcts.append(errorPct)

        print 'date={0}, predicted={1}, expected={2}, error pct={3:.2f}%'.format(tsWeekly.keys()[len(history)-1],
                                                                                 predicted, observed, errorPct)

    error = mean_squared_error(test, predictions)
    print 'Test MSE: {0:.3f}'.format(error)

    pyplot.subplot(211)
    pyplot.title('Predicted vs Expected')
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.subplot(212)
    pyplot.title('Prediction Error Percentage')
    pyplot.plot(errorPcts)
    pyplot.tight_layout()
    pyplot.show()

def main():
    startDate = datetime(2016, 1, 1)
    endDate = datetime.today()

    panel = downloadData(startDate, endDate)

    #doAnalysis(panel)
    doForecast(panel)

if __name__ == '__main__':
    main()