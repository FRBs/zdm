import numpy as np
import mpmath

from scipy import interpolate
from torch import embedding

from fast_interp import interp1d

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

log_avals = np.linspace(-6, 6., 1000)
h = log_avals[1]-log_avals[0]

linear = interpolate.interp1d(avals, numer)

fast = interp1d(-6., 6., h, numer)

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

#3) Use scipy linear

def call_linear_1():
    val = linear(avals2)

#3) Fast interpolator
# Compile numba
val = fast(np.log10(avals3[0:1]))
from IPython import embed
def call_fast_100():
    # Run
    for i in range(100):
        val = fast(np.log10(avals3))

# find a new interpolater and try make it run faster than these specific calls (down to 0.10s)

 
# Run now

for i in range(100):
    calls_100()
    calls_1()
    call_linear_1()
    call_fast_100()


'''
#TIMEPROFILING
python -m cProfile -o interpolation_tests.prof interpolation_tests.py



'''

