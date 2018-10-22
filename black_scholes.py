"""
Id:             black_scholes.py
Copyright:      2018 xiaokang.guan All rights reserved.
Description:    Black-Scholes utility functions.
"""

import numpy as np
import math
import logging


def ncdf(x):
    """
    Cumulative distribution function for the standard normal distribution.
    Alternatively, we can use below:
    from scipy.stats import norm
    norm.cdf(x)
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def npdf(x):
    """
    Probability distribution function for the standard normal distribution.
    Alternatively, we can use below:
    from scipy.stats import norm
    norm.pdf(x)
    """
    return np.exp(-np.square(x) / 2) / np.sqrt(2 * np.pi)


def blackScholesOptionPrice(callPut, spot, strike, tenor, rate, sigma):
    """
    Black-Scholes option pricing
    tenor is float in years. e.g. tenor for 6 month is 0.5
    """
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
    d2 = d1 - sigma * np.sqrt(tenor)

    if callPut == 'Call':
        return spot * ncdf(d1) - strike * np.exp(-rate * tenor) * ncdf(d2)
    elif callPut == 'Put':
        return -spot * ncdf(-d1) + strike * np.exp(-rate * tenor) * ncdf(-d2)


def blackScholesVega(callPut, spot, strike, tenor, rate, sigma):
    """ Black-Scholes vega """
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
    return spot * np.sqrt(tenor) * npdf(d1)


def blackScholesDelta(callPut, spot, strike, tenor, rate, sigma):
    """ Black-Scholes delta """
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
    if callPut == 'Call':
        return ncdf(d1)
    elif callPut == 'Put':
        return ncdf(d1) - 1


def blackScholesGamma(callPut, spot, strike, tenor, rate, sigma):
    """" Black-Scholes gamma """
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
    return npdf(d1) / (spot * sigma * np.sqrt(tenor))


def blackScholesSolveImpliedVol(targetPrice, callPut, spot, strike, tenor, rate):
    """" Solve for implied volatility using Black-Scholes """
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5

    sigma = 0.5
    i = 0
    while i < MAX_ITERATIONS:
        optionPrice = blackScholesOptionPrice(callPut, spot, strike, tenor, rate, sigma)
        vega = blackScholesVega(callPut, spot, strike, tenor, rate, sigma)
        diff = targetPrice - optionPrice
        logging.debug('blackScholesSolveImpliedVol: iteration={}, sigma={}, diff={}'.format(i, sigma, diff))

        if abs(diff) < PRECISION:
            return sigma

        sigma = sigma + diff/vega
        i = i + 1

    logging.debug('blackScholesSolveImpliedVol: After MAX_ITERATIONS={}, best sigma={}'.format(MAX_ITERATIONS, sigma))
    return sigma