import numpy as np
import mpmath
from matplotlib import pyplot as plt

from zdm import iteration as it
from zdm.craco import loading

from scipy import interpolate

gamma = -1.01

# 1) Build  a spline w/ 1000 points (10^-6 -> 10^6)
avals = 10**np.linspace(-6, 6., 1000)
numer = np.array([float(mpmath.gammainc(gamma, a=iEE)) for iEE in avals])

#2a) 100 calls to spev of 1000 (x) values 
def calls_100(spline):
    for i in range(100):
        interpolate.splev(avals, ([1],[1],[1]))


calls_100(gamma)

#2b) 1 call of 100 x 1000 (x) values
def calls_1(spline):
    pass


#3) Use scipy instead of spline

'''
#TIMEPROFILING
python -m cProfile -o spline_tests.prof spline_tests.py


'''

