"""
Generates a plot of the peak likelihood of Ptot.

Must have run "analyse_repeat_dists.py" on multiple
values of FC (fraction of CHIME single FRBs due to repeaters)
"""

import numpy as np
from matplotlib import pyplot as plt

import matplotlib
#matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)



def main():
    
    infile = 'peak_likelihood.txt'
    data = np.loadtxt(infile)
    print(data.shape)
    
    FC = data[:10,0]
    peak = 10**data[:10,1]
    
    plt.figure()
    plt.ylim(0,0.09)
    plt.xlim(0.1,1.0)
    plt.xlabel('$F_{\\rm single}$')# [fraction of single bursts due to repeaters]')
    plt.ylabel('Max $[P_{\\rm tot}(\\gamma_r, R_{\\rm max})] (F_{\\rm single})$')
    plt.plot(FC,peak,linewidth=3)
    plt.tight_layout()
    plt.savefig('Ptot_FC.pdf')
    plt.close()


main()
