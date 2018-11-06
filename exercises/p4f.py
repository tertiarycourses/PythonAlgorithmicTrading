import numpy as np
import math
from math import exp,sqrt
import logging
from scipy import log,exp,sqrt,stats
import networkx as nx
import matplotlib.pyplot as plt

# Black-Scholes Model

def bs_call(S,X,T,rf,sigma):
	d1=(log(S/X)+(rf+sigma*sigma/2.)*T)/(sigma*sqrt(T))
	d2 = d1-sigma*sqrt(T)
	return S*stats.norm.cdf(d1)-X*exp(-rf*T)*stats.norm.cdf(d2)

def callAndPut(S,X,T,r,sigma,type='C'):
    d1=(log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T)) 
    d2 = d1-sigma*sqrt(T)
    if type.upper()=='C':
        c=S*stats.norm.cdf(d1)-X*exp(-r*T)*stats.norm.cdf(d2)
        return c
    else:
        p=X*exp(-r*T)*stats.norm.cdf(-d2)-S*stats.norm.cdf(-d1)
        return p

def ncdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def npdf(x):
    return np.exp(-np.square(x) / 2) / np.sqrt(2 * np.pi)

def blackScholesOptionPrice(callPut, spot, strike, tenor, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
    d2 = d1 - sigma * np.sqrt(tenor)

    if callPut == 'Call':
        return spot * ncdf(d1) - strike * np.exp(-rate * tenor) * ncdf(d2)
    elif callPut == 'Put':
        return -spot * ncdf(-d1) + strike * np.exp(-rate * tenor) * ncdf(-d2)

# Binomial Option Price Model        

def binomial_grid(n):
	G=nx.Graph()
	for i in range(0,n+1):
		for j in range(1,i+2):
			if i<n:
				G.add_edge((i,j),(i+1,j))
				G.add_edge((i,j),(i+1,j+1))
	posG={} #dictionary with nodes position
	for node in G.nodes():
		posG[node]=(node[0],n+2+node[0]-2*node[1])
	nx.draw(G,pos=posG)

def delta_call(S,X,T,rf,sigma):
	d1=(log(S/X)+(rf+sigma*sigma/2.)*T)/(sigma*sqrt(T))
	return(stats.norm.cdf(d1))

def delta_put(S,X,T,rf,sigma):
	d1=(log(S/X)+(rf+sigma*sigma/2.)*T)/(sigma*sqrt(T))
	return(stats.norm.cdf(d1)-1)

def binomialCallEuropean(s,x,T,r,sigma,n=100):
    deltaT = T /n
    u = exp(sigma * sqrt(deltaT)) 
    d = 1.0 / u
    a = exp(r * deltaT)
    p = (a - d) / (u - d)
    v = [[0.0 for j in np.arange(i + 1)]  for i in np.arange(n + 1)] 
    for j in np.arange(n+1):
        v[n][j] = max(s * u**j * d**(n - j) - x, 0.0) 
    for i in np.arange(n-1, -1, -1):
        for j in np.arange(i + 1):
            v[i][j]=exp(-r*deltaT)*(p*v[i+1][j+1]+(1.0-p)*v[i+1][j]) 
    return v[0][0]


def binomialAmericanCall(s,x,T,r,sigma,n=100):
    deltaT = T /n
    u = exp(sigma * sqrt(deltaT)) 
    d = 1.0 / u
    a = exp(r * deltaT)
    p = (a - d) / (u - d)
    v = [[0.0 for j in np.arange(i + 1)] for i in np.arange(n + 1)] 
    for j in np.arange(n+1):
        v[n][j] = max(s * u**j * d**(n - j) - x, 0.0) 
    for i in np.arange(n-1, -1, -1):
        for j in np.arange(i + 1):
            v1=exp(-r*deltaT)*(p*v[i+1][j+1]+(1.0-p)*v[i+1][j]) 
            v2=max(v[i][j]-x,0)	       # early exercise 
            v[i][j]=max(v1,v2)
    return v[0][0]

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

# Greek Options

def blackScholesVega(callPut, spot, strike, tenor, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
    return spot * np.sqrt(tenor) * npdf(d1)


def blackScholesDelta(callPut, spot, strike, tenor, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
    if callPut == 'Call':
        return ncdf(d1)
    elif callPut == 'Put':
        return ncdf(d1) - 1


def blackScholesGamma(callPut, spot, strike, tenor, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
    return npdf(d1) / (spot * sigma * np.sqrt(tenor))

# Implied Volitity

def blackScholesSolveImpliedVol(targetPrice, callPut, spot, strike, tenor, rate):
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
