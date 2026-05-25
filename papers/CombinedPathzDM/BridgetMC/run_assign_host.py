"""
Based on notebook 
https://astropath.readthedocs.io/en/latest/nb/Simulate_Generate_FRBs.html

This takes generated FRBs, and gives them a "true: host galaxy

While the catalogue used to generate hosts is not complete,
this doesn't really matter too much. The issue is how to deal with
FRBs generated at very low z with very bright galaxies.
Something to think about later...

E.g. typical error message:

"Ran out of bright galaxies at iteration 67
  Brightest remaining galaxy: m_r = 12.71
  Faintest remaining FRB: m_r = 12.14"


"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.patches import Ellipse


def main(outfile,localisation,seed):
    
    # Set up plotting style
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    from astropath.simulations import assign_host as assign
    
    #os.environ['FRB_APATH'] = "./"
    catalogue = pd.read_parquet("combined_HSC_DECaLs_HECATE_galaxies_hecatecut.parquet")
    
    frbs = pd.read_csv("craco_900_mc_sample.csv")
    scale=0.5
    
    #mag_range=[17.,28]
    
    assignments = assign.assign_frbs_to_hosts(frb_df=frbs, galaxy_catalog=catalogue,
                                        localization=localisation,
                                        scale=scale, seed=seed)
    
    assignments.to_csv(outfile,index=False)


seed=9609572
localisation=(0.5,0.5,0.)
#main("craco_assigned_galaxies.csv",localisation,seed)

seed=1057248
localisation=(30.,30.,0.)
main("loc30_craco_assigned_galaxies.csv",localisation,seed)
