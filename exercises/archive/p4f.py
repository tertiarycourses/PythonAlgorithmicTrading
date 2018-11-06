def bs_call(S,X,T,rf,sigma):
	from scipy import log,exp,sqrt,stats
	d1=(log(S/X)+(rf+sigma*sigma/2.)*T)/(sigma*sqrt(T))
	d2 = d1-sigma*sqrt(T)
	return S*stats.norm.cdf(d1)-X*exp(-rf*T)*stats.norm.cdf(d2)

def binomial_grid(n):
	import networkx as nx
	import matplotlib.pyplot as plt
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
	from scipy import log,exp,sqrt,stats
	d1=(log(S/X)+(rf+sigma*sigma/2.)*T)/(sigma*sqrt(T))
	return(stats.norm.cdf(d1))

def delta_put(S,X,T,rf,sigma):
	from scipy import log,exp,sqrt,stats
	d1=(log(S/X)+(rf+sigma*sigma/2.)*T)/(sigma*sqrt(T))
	return(stats.norm.cdf(d1)-1)

def binomialCallEuropean(s,x,T,r,sigma,n=100):
    from math import exp,sqrt
    import numpy as np
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
    from math import exp,sqrt
    import numpy as np
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
