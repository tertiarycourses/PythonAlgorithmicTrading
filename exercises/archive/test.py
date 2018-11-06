import black_scholes as bs

call = bs.blackScholesOptionPrice('Call', 50, 50, 0.5, 0.0366, 0.62)
put = bs.blackScholesOptionPrice('Put', 50, 50, 0.5, 0.0366, 0.62)

print(call)
print(put)