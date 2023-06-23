"""
This file is intended to load a grid, and generate a single
slice of likelihoods through H0

"""
#
import pytest

#from pkg_resources import resource_filename
#import os


from zdm import iteration as it
from zdm.craco import loading
from zdm import misc_functions
import matplotlib.pyplot as plt
#from zdm import io
#from zdm.tests import tstutils

#from IPython import embed

#import time
import numpy as np

def main():
    """
    Write stuff here
    """
    # create new grids directly from updates state parameters in grid   
    
    frb_names = ["181112", "190611", "190711", "191228", "210117", "210320", "210407", "210912", "220501", "230526"]
    # frb_names = ["180924","181112", "190102", "190608", "190611", "190711", "190714", "191228", "210117", "210214", "210320", "210912", "211117"]
    # frb_names = ["190711"]
    edir='/fred/oz002/jhoffmann/FRB_library/zdm/zdm/data/Efficiencies/'
    sdir='/fred/oz002/jhoffmann/FRB_library/zdm/zdm/data/Surveys/Hoffmann2023/'

    H0s = np.linspace(50,100,50)
    llsum_total = np.zeros(len(H0s), dtype=float)
    llsum_total_exact = np.zeros(len(H0s), dtype=float)

    # plt.figure()
    # plt.xlabel(r"H_0")
    # plt.ylabel(r"log likelihood")
    # fig, axes = plt.subplots(1,2, sharey=True)
    for name in frb_names:
        # Normal calculation
        s,g = loading.survey_and_grid(survey_name=name,sdir=sdir,NFRB=None,model='Quadrature',edir=edir) 
        # Calculation with efficiencies
        s_exact,g_exact = loading.survey_and_grid(survey_name=name,sdir=sdir,NFRB=None,model=name,edir=edir)

        # nozlist=[]
        # DMmax=4000
        # zmax=3
  
        # # Plotting difference in grids
        # plt_rates = np.abs((g.rates - g2.rates)) / (g.rates + g2.rates) * 2
        # plt_rates[np.isfinite(plt_rates)==False] = 0.0
        # plt_rates[g.rates==0] = 0.0
        # plt_rates[g2.rates==0] = 0.0

        # plt.rc('font', size=16)
        # plt.figure(figsize=(9,6))
        # plt.pcolormesh(g.zvals, g.dmvals, plt_rates.T, shading='nearest')
        # cbar = plt.colorbar()
        # cbar.set_label(r"$\frac{2|p_{Cordes} - p_{Numeric}|}{p_{Cordes} + p_{Numeric}}$", fontsize=20)
        # plt.xlabel("z")
        # plt.ylabel(r"DM$_{\mathrm{IGM}}$")
        # plt.show()
        # plt.close()

        # misc_functions.plot_grid_2(plt_rates,g.zvals,g.dmvals,
        # name=name+'_diff.pdf',norm=3,log=True,
        # label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        # project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        # zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state)

        # misc_functions.plot_grid_2(g2.rates,g2.zvals,g2.dmvals,
        # name=name+'_jordan_quad.pdf',norm=3,log=True,
        # label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        # project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        # zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g2.state)

        # varies H0
        llsum=[]
        llsum_exact=[]
        for H0 in H0s:
            # updates grid to new parameter values
            vparams = {}
            vparams['H0'] = H0
            g.update(vparams)
            # returns log-likelihood sum for this survey and grid
            llsum.append(get_likelihood(s,g))

            g_exact.update(vparams)
            # returns log-likelihood sum for this survey and grid
            llsum_exact.append(get_likelihood(s_exact,g_exact))
        
        llsum_total += np.array(llsum)
        llsum_total_exact += np.array(llsum_exact)

        fig, axes = plt.subplots(1,1)
        axes.plot(H0s, llsum, label="Quadrature")
        axes.plot(H0s, llsum_exact - np.max(llsum_exact) + np.max(llsum), label="Exact")
        axes.set_xlabel(r"$H_0$")
        axes.set_ylabel(r"Normalised log(p($H_0$))")
        axes.legend()

        plt.savefig("../Figures/ll_" + name + '.png', format='png', bbox_inches='tight')
        plt.close()

    # axes[0].plot(H0s, llsum_total - np.max(llsum_total), label="Total")
    # axes[0].legend()
    # axes[0].set_title('Quadrature')
    # axes[0].set_xlabel(r"$H_0$")
    # axes[0].set_ylabel(r"Normalised log likelihood")

    # axes[1].plot(H0s, llsum_total_exact - np.max(llsum_total_exact), label="Total")
    # axes[1].legend()
    # axes[1].set_title('Exact')
    # axes[1].set_xlabel(r"$H_0$")

    # plt.show()
    # plt.close()

    fig, axes = plt.subplots(1,1)
    print(np.max(llsum_total))
    print(np.max(llsum_total_exact))
    axes.plot(H0s, llsum_total - np.max(llsum_total), label="Quadrature")
    axes.plot(H0s, llsum_total_exact - np.max(llsum_total_exact), label="Exact")
    axes.set_xlabel(r"$H_0$")
    axes.set_ylabel(r"Normalised log(p($H_0$))")
    axes.legend()
    plt.savefig("../Figures/ll_total.png", format='png', bbox_inches='tight')
    plt.close()

def get_likelihood(s,g,norm=True,Pn=False,psnr=True,dolist=0):
    """
    Returns total ikelihood for a single survey s and grid g
    
    I am turning Pn off now becuse for a single survey it's useless info
    """
    # we only return the total log-likelihood, not separated into components
    
    
    if s.nD==1:
        llsum = it.calc_likelihoods_1D(
            g,s,norm=norm,psnr=psnr,dolist=dolist,Pn=Pn)
    elif s.nD==2:
        llsum = it.calc_likelihoods_2D(
            g,s,norm=norm,psnr=psnr,dolist=dolist,Pn=Pn)
    elif s.nD==3:
        # mixture of 1 and 2D samples. NEVER calculate Pn twice!
        llsum = it.calc_likelihoods_1D(
            g,s,norm=norm,psnr=psnr,dolist=dolist,Pn=Pn)
        # must always have Pn being false for one of these two
        llsum += it.calc_likelihoods_2D(
            g,s,norm=norm,psnr=psnr,dolist=dolist,Pn=False)
    return llsum


main() 
