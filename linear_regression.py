"""
Id:             linear_regression.py
Copyright:      2018 xiaokang.guan All rights reserved.
Description:    Linear Regression example using sklearn.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from data_hub import DataHub
import datetime

def getTrainAndTestData(startDate=datetime.date(2018,1,1), endDate=datetime.date.today(), xSymbols=['SPY'], ySymbol=['AAPL'], trainingSize=0.7):
    """
    For given period and symbols, generate the training and test data set.
    :param startDate: Start date for the entire data set
    :param endDate: End date for the entire data set
    :param xSymbols: Independent variables symbol tickers
    :param ySymbol: Depdendent variable symbol ticker
    :param trainingSize: The proportion of data set for training
    :return: (xTrain, yTrain, xTest, yTest), in 2-d array format
    """
    dataHub = DataHub()
    historicalDataBySymbol = dataHub.downloadDataFromYahoo(startDate, endDate, xSymbols+ySymbol)
    df = pd.concat([historicalDataBySymbol[symbol].loc[:,'Close'] for symbol in xSymbols+ySymbol], axis=1, join='inner')

    splitIdx = int(df.shape[0] * trainingSize)
    xTrain = df.iloc[:splitIdx, :-1].values
    yTrain = df.iloc[:splitIdx, -1].values
    xTest = df.iloc[splitIdx:, :-1].values
    yTest = df.iloc[splitIdx:, -1].values

    return xTrain, yTrain, xTest, yTest


def performLinearRegression():
    """
    Main function to perform Linear Regression
    """

    xTrain, yTrain, xTest, yTest = getTrainAndTestData()

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(xTrain, yTrain)

    # Make predictions using the testing set
    yPredict = regr.predict(xTest)

    # The coefficients
    print('Coefficients: %.4f' % regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(yTest, yPredict))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(yTest, yPredict))

    # Plot outputs
    plt.scatter(xTest, yTest,  color='black')
    plt.plot(xTest, yPredict, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def __main__():
    """
    Entry point.
    """
    performLinearRegression()