import numpy as np
import mpmath

from scipy import interpolate

gamma = -1.01

# 1) Build  a spline w/ 1000 points (10^-6 -> 10^6)

# check if making avals bigger affects anything ?
avals = 10**np.linspace(-6, 6., 1000)
avals2 = 10**np.linspace(-6, 6., 100000)
avals3 = 10**np.linspace(-6, 6., 999)
numer = np.array([float(mpmath.gammainc(gamma, a=iEE)) for iEE in avals]) # expensive
# np.save and then load from file
test = np.random.rand(1000)

spline = interpolate.splrep(avals, numer)

print(len(numer))

#2a) 100 calls to splev of 1000 (x) values 
def calls_100():
    for i in range(100):
         val = interpolate.splev(avals3, spline)
         #print(val)
        



#2b) 1 call of 100 x 1000 (x) values
def calls_1():
    val = interpolate.splev(avals2, spline)
    # call avals
    # 

for i in range(100):
    calls_100()
    calls_1()

#3) Use scipy instead of spline

# find a new interpolater and try make it run faster than these specific calls (down to 0.10s)

'''
#TIMEPROFILING
python -m cProfile -o spline_tests.prof spline_tests.py



'''

