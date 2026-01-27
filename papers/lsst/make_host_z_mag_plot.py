"""
Plots the expected magnitude-z relation for various FRB host galaxy models

Adds points corresponding to known hosts

"""

#standard Python imports
import os
import numpy as np
from matplotlib import pyplot as plt

import galaxies as g

# imports from the "FRB" series
from zdm import optical as opt
from zdm import optical_params as op
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading
import galaxies as g
import astropath.priors as pathpriors
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def make_zmr_plots():
    """
    Loops over all ICS FRBs
    """
    
    # loops over different FRB host models
    opdir="zmr/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # model 1: simple model
    model1 = opt.simple_host_model()
    absprior = [0,0,0.0705, 0.839, 0.328, 0.001,0,0,0,0]
    model1.init_args(absprior)
    
    
    # this is from an initial estimate. Currently, no way to enter this into the opstate. To do.
    
    #loudas model with SFR=0
    model2=opt.loudas_model()
    model2.init_args(0.)
    
    # loudas model with SFR=1
    model3=opt.loudas_model()
    model3.init_args(1.)
    
    # model from Lachlan
    model4=opt.marnoch_model()
    
    models=[model1,model2,model3,model4]
    labels=["simple","sfr0","sfr1","marnoch"]
    
    for i,model in enumerate(models):
        opfile = opdir+labels[i]+"_zmr.png"
        make_host_plot(model,opfile)
    

def make_host_plot(model,opfile):
    """
    generates a plot showing the magnitude and redshift of a bunch of FRB host galaxies
    
    Args:
        model: optical model class instance
        opfile: string labelling the plotfile
    """
    
    nz=200
    zmax=2
    zmin = zmax/nz
    zvals = np.linspace(zmin,zmax,nz)
    mrbins = np.linspace(0,40,401)
    mrvals = (mrbins[1:] + mrbins[:-1])/2.
    
    medians = np.zeros([nz])
    sig1ds = np.zeros([nz])
    sig1us = np.zeros([nz])
    sig2ds = np.zeros([nz])
    sig2us = np.zeros([nz])
    
    for i,z in enumerate(zvals):
        pmr = model.get_pmr_gz(mrbins,z)
        
        cpmr = np.cumsum(pmr)
        cpmr /= cpmr[-1]
        median = np.where(cpmr > 0.5)[0][0]
        sig1d = np.where(cpmr < 0.165)[0][-1]
        sig1u = np.where(cpmr > 0.835)[0][0]
        sig2d = np.where(cpmr < 0.025)[0][-1]
        sig2u = np.where(cpmr > 0.975)[0][0]
        
        medians[i] = mrvals[median]
        sig1ds[i] = mrvals[sig1d]
        sig1us[i] = mrvals[sig1u]
        sig2ds[i] = mrvals[sig2d]
        sig2us[i] = mrvals[sig2u]
    
    plt.figure()
    plt.plot(zvals,medians,linestyle="-",color="red",label="Mean $m_r$")
    plt.plot(zvals,sig1ds,linestyle="--",color=plt.gca().lines[-1].get_color(),label="67% C.I.")
    plt.plot(zvals,sig1us,linestyle="--",color=plt.gca().lines[-1].get_color())
    plt.plot(zvals,sig2ds,linestyle=":",color=plt.gca().lines[-1].get_color(),label="95% C.I.")
    plt.plot(zvals,sig2us,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.xlabel("z")
    plt.ylabel("$m$")
    
    z,mr,w = g.read_craft()
    OK = np.where(w>= 0.5)[0]
    plt.scatter(z[OK],mr[OK],marker='d',label="CRAFT ICS",s=20)
    
    z,mr,w = g.read_meerkat()
    OK = np.where(w>= 0.5)[0]
    plt.scatter(z[OK],mr[OK],marker='+',label="MeerTRAP coherent",s=20)
    
    z,mr,w = g.read_dsa()
    OK = np.where(w>= 0.5)[0]
    plt.scatter(z[OK],mr[OK],marker='o',label="DSA",s=20)
    
    zmax=2
    plt.ylim(10,30)
    plt.xlim(0,zmax)
    
    Rlim1=24.7
    Rlim2=27.5
    plt.plot([0,zmax],[Rlim1,Rlim1],linestyle=":",color="black")
    plt.plot([0,zmax],[Rlim2,Rlim2],linestyle=":",color="black")
    plt.text(0.1,Rlim1+0.2,"$m_r^{\\rm lim}=$"+str(Rlim1))
    plt.text(0.1,Rlim2+0.2,"$m_r^{\\rm lim}=$"+str(Rlim2))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(opfile)
    plt.close()


if __name__ == "__main__":
    
    make_zmr_plots()

    
    
