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
    
    CubeFile='Cubes/craco_full_cube.npz'
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
    
    deprecated,vectors,wvectors=ac.get_bayesian_data(data["ll"])
    
    latexnames=[
        "H_0",
        "\\mu_{\\rm host}",
        "\\sigma_{\\rm host}",
        "\\log_{10} F",
    ]
    units=[
        "km/s/Mpc",
        "",
        "",
        "",
    ]

    # ['[erg]','[km/s/Mpc]','','','','$[\\log_{10} {\\rm DM}]','']
    
    truth=[67.66,2.16,.51,-0.49]
    #ac.do_single_plots(uvals,vectors,wvectors,params,tag="prior_",truth=truth,dolevels=True,latexnames=latexnames)
    ac.do_single_plots(uvals,vectors,None,params,tag="Figure10_",truth=truth,dolevels=True,latexnames=latexnames,units=units)

main()
