"""
Plots Figure 10 ('CRACO') analysis

Produces plots for each parameter, even though only H0
was shown in the paper.

"""

import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt

def main():
    
    if not os.path.exists("Figure10/"):
        os.mkdir("Figure10")
    
    CubeFile='/Users/cjames/CRAFT/Paper/H0/CRACO/craco_3rd_full_cube-001.npz'
    if os.path.exists(CubeFile):
        data=np.load(CubeFile)
    else:
        print("Missing cube file ",CubeFile," please download")
        exit()
    
    data=np.load(CubeFile)
    
    lst = data.files
    lldata=data["ll"]
    params=data["params"]
    # builds uvals list
    uvals=[]
    for param in params:
        uvals.append(data[param])
        
    alpha_dim = np.where(data["params"] == "alpha")[0][0]
    alpha_mean=0.65
    alpha_sigma=0.3
    
    wlls = ac.apply_gaussian_prior(data["ll"],alpha_dim,data["alpha"],alpha_mean,alpha_sigma)
    deprecated,vectors,wvectors=ac.get_bayesian_data(data["ll"],wlls)
    
    latexnames=['$\\log_{10} E_{\\rm max}$','$H_0$','$\\alpha$','$\\gamma$','$n_{\\rm sfr}$','$\\mu_{\\rm host}$','$\\sigma_{\\rm host}$']
    units=['[erg]','[km/s/Mpc]','','','','$[\\log_{10} {\\rm DM}]','']
    
    truth=[41.4,67.66,0.65,-1.01,0.73,2.18,0.48]
    #ac.do_single_plots(uvals,vectors,wvectors,params,tag="prior_",truth=truth,dolevels=True,latexnames=latexnames)
    ac.do_single_plots(uvals,vectors,None,params,tag="Figure10_",truth=truth,dolevels=True,latexnames=latexnames,units=units)

main()
