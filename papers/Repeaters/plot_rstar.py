"""
Makes the plot of Rstar vs f
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    nsets=5
    
    for i in np.arange(nsets):
        infile = 'Rfitting/rstar_set_'+str(i)+'__output.npz'
        data = np.load(infile)
        fs = data['arr_0']
        Rstar = data['arr_1']
        
        outfile = 'Rstar/set_'+str(i)+'_rstar.pdf'
        
        plt.figure()
        plt.xlabel('$f_{\\rm rep}$')
        plt.ylabel('$R^*$')
        plt.plot(fs,Rstar)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
    
    
    
    



main()


