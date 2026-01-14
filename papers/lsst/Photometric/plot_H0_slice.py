"""
Generates a plot of previously calculated slice through H0
"""

import argparse
import numpy as np
import os

from zdm import figures
from zdm import iteration as it

from zdm import parameters
from zdm import repeat_grid as zdm_repeat_grid
from zdm import MCMC
from zdm import survey
from zdm import states
from astropy.cosmology import Planck18

from numpy import random
import matplotlib.pyplot as plt
import time

import matplotlib
defaultsize=18
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    """
    Main routine to create plots and extract characteristic parameters
    """
    ll_lists=np.load("ll_lists.npy")
    vals = np.load("h0vals.npy")
    
    nh,ns = ll_lists.shape
    
    plt.figure()
    linestyles=["-","--",":","-."]
    # up to the user to get thius order right! Use e.g.
    # python run_H0_slice.py -n 10 --min=50 --max=100 -f Spectroscopic Smeared zFrac Smeared_and_zFrac
    
    s_names=["All hosts, spec-zs","$\\sigma_z=0.035$","$m_r^{\\rm lim}=24.7$","$\\sigma_z=0.035,~m_r^{\\rm lim}=24.7$"]
    plt.clf()
    llsum = np.zeros(ll_lists.shape[0])
    FWHM=[]
    for i in np.arange(ns):
        
        lls = ll_lists[:, i]
        
        lls[lls < -1e10] = -np.inf
        lls[np.argwhere(np.isnan(lls))] = -np.inf
        
        llsum += lls
        
        lls = lls - np.max(lls)
        lls=10**lls
        index1=np.where(lls>=0.5)[0][0]
        index2=np.where(lls>=0.5)[0][-1]
        root1=vals[index1-1]-(0.5-lls[index1-1])*(vals[index1]-vals[index1-1])/(lls[index1]-lls[index1-1])
        root2=vals[index2]-(0.5-lls[index2])*(vals[index2+1]-vals[index2])/(lls[index2+1]-lls[index2])
        FWHM.append(root2-root1)
        # plt.figure()
        # plt.clf()
        plt.plot(vals, lls, label=s_names[i],ls=linestyles[i])
        plt.xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=14)
        plt.ylabel('$\\frac{\\mathcal{L}}{max(\\mathcal{L})}$',fontsize=18)
        # plt.savefig(os.path.join(outdir, s.name + ".pdf"))
        #print("Max H0:",vals[np.where(lls==1.0)[0][0]])
    
    plt.minorticks_on() 
    plt.tick_params(axis='y', which='major', labelsize=14)    # To set tick label fontsize
    plt.tick_params(axis='y', which='major', length=9)        # To set tick size
    plt.tick_params(axis='y', which='minor', length=4.5)      # To set tick size
    plt.tick_params(axis='y', which='both',direction='in',right='on', top='on')  

    plt.tick_params(axis='x', which='major', labelsize=14)    # To set tick label fontsize 
    plt.tick_params(axis='x', which='major', length=9)        # To set tick size 
    plt.tick_params(axis='x', which='minor', length=4.5)      # To set tick size
    plt.tick_params(axis='x', which='both',direction='in',right='on', top='on') 
    
    #peak=vals[np.argwhere(llsum == np.max(llsum))[0]]
    plt.xlim(60,90)
    plt.ylim(0,1)
    
    
    plt.plot([70.63,70.63],[0,1],color="black",linestyle=":")
    plt.text(69,0.06,"$H_0^{\\rm sim}$",rotation=90,fontsize=14)
    
    #plt.axvline(peak,ls='--')
    #plt.legend(loc='upper left')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("H0_scan_linear.png")
    percentage=(FWHM/FWHM[0]-1)*100
    for i,name in enumerate(s_names):
        print(name," FWHM is ",FWHM[i]," frac is ",percentage[i])
    #print("FWHM:Spectroscopic,Photometric,zFrac,Photometric+zfrac\n",FWHM,percentage)
    

main()
