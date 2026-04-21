"""
New file to plot generated MC FRBs
"""


from zdm import loading
from zdm import states
from zdm import optical as opt
from zdm import optical_params as op
from zdm import figures

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from frb.dm import igm

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    """
    Main program
    """
    
    frbs = pd.read_csv("craco_900_mc_sample.csv")
    make_scatter_plots(frbs)
    

def make_scatter_plots(frbs):
    """
    Makes scatter plots of the generated frbs
    """
    # generate scatter plot of z and mr
    plt.figure()
    plt.scatter(frbs["z"],frbs["m_r"],s=1,c=frbs["DMeg"], cmap='gnuplot2_r')
    cbar = plt.colorbar()
    cbar.set_label("DM$_{\\rm EG}$")
    plt.xlabel("Redshift, $z$")
    plt.ylabel("Apparent magnitude, $m_r$")
    plt.xlim(0,2.5)
    plt.ylim(10,32)
    plt.tight_layout()
    plt.savefig("mr_z_plot.png")
    plt.close()
    
    plt.figure()
    plt.scatter(frbs["z"],frbs["DMeg"],s=3,c=frbs["m_r"], cmap='gnuplot2_r')
    cbar = plt.colorbar()
    plt.xlim(0,2.5)
    plt.ylim(0,2000)
    plt.xlabel("Redshift, $z$")
    cbar.set_label("Host $m_r$")
    plt.ylabel("Extragalactic DM, DM$_{\\rm EG}$")
    plt.tight_layout()
    plt.savefig("dmeg_z_plot.png")
    plt.close()


main()
