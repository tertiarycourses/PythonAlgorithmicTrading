"""
Id:             binomial_tree.py
Copyright:      2018 xiaokang.guan All rights reserved.
Description:    Binomial Tree model utility functions.
"""

import numpy as np


def binomialTree(callPut, spot, strike, rate, sigma, tenor, N=2000, american=True):

    # Each time step period
    deltaT = float(tenor) / N
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
    a = np.exp(rate * deltaT)
    p = (a - d) / (u - d)
    oneMinusP = 1.0 - p

    # Initialize the arrays
    fs = np.asarray([0.0 for i in xrange(N + 1)])

    # Stock tree for calculations of expiration values
    fs2 = np.asarray([(spot * u ** j * d ** (N - j)) for j in xrange(N + 1)])

    # Vectorize the strikes to speed up expiration check
    fs3 = np.asarray([float(strike) for i in xrange(N + 1)])

    # Compute the Binomial Tree leaves, f_{N, j}
    if callPut == 'Call':
        fs[:] = np.maximum(fs2 - fs3, 0.0)
    else:
        fs[:] = np.maximum(-fs2 + fs3, 0.0)

    # Calculate backward the option prices
    for i in xrange(N - 1, -1, -1):
        fs[:-1] = np.exp(-rate * deltaT) * (p * fs[1:] + oneMinusP * fs[:-1])
        fs2[:] = fs2[:] * u

        if american:
            # Simply check if the option is worth more alive or dead
            if callPut == 'Call':
                fs[:] = np.maximum(fs[:], fs2[:] - fs3[:])
            else:
                fs[:] = np.maximum(fs[:], -fs2[:] + fs3[:])

    return fs[0]
