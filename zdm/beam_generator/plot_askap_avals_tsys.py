"""
Simple program to plot the avals and Tsys of askap,
which are taken from Hotan et al.

"""

import numpy as np
from matplotlib import pyplot as plt


def main():
    xs = np.linspace(-4,4,101)
    ys,x,y = get_avals(xs)
    
    plt.figure()
    plt.plot(xs,ys)
    plt.plot(x,y,linestyle="",marker="+")
    plt.tight_layout()
    plt.savefig("avals.png")
    plt.close()
    
    freqs = np.linspace(700,1800,101)
    get_tsys(freqs,plot=True)

def get_avals(thetas):
    """
    Values extrated from Hotan et al.
    """
    
    data = np.loadtxt("avals.dat")
    x = data[:,0]
    y = data[:,1]
    newx = np.concatenate([x,-x[::-1]])
    newy = np.concatenate([y,y[::-1]])
    avals = np.interp(thetas,newx,newy)
    return avals,newx,newy

    
def get_tsys(freq,plot=False):
    """
    reads in interpolated data from Hotan et al
    
    Freq: frequency in MHz
    """   
    
    data = np.loadtxt("askap_tsys.dat")
    x = data[:,0]
    y = data[:,1]
    deg = 10
    coeffs = np.polyfit(x,y,deg)
    
    
    if plot:
        xvals = np.linspace(700,1800,201)
        yvals = np.polyval(coeffs,xvals)
        plt.figure()
        plt.plot(xvals,yvals,color='grey')
        plt.plot(x,y,linestyle="",color='red',marker='o')
        plt.ylim(0,130)
        plt.savefig("askap_tsys.png")
        plt.close()
    
    Tsys = np.polyval(coeffs,freq)
    return Tsys
    
main()

