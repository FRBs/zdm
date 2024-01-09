""" 
This script produces plots to test spline accuracy,
and test the time taken to create and evaluate splines.

"""

from zdm import energetics


def main():
    
    energetics.test_spline_accuracy()
    energetics.time_splines(Nreps=10)


main()
