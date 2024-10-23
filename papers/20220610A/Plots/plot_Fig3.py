""" 
This script plots Figure 3 of Ryder et al.

It produces output which is similar to
plot_220610.py
but only for the standard parameetr set.
"""
import os

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import matplotlib
matplotlib.rcParams['image.interpolation'] = None

defaultsize=12
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    # in case you wish to switch to another output directory
    opdir = "Extremes/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    
    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    names = ['CRAFT_ICS','CRAFT_ICS_892','CRAFT_ICS_1632']
    sdir = "../../zdm/data/Surveys/"
    
    # approximate best-fit values from recent analysis
    vparams = {}
    vparams["H0"] = 67.4
    vparams["lEmax"] = 41.63
    vparams["gamma"] = -0.948
    vparams["alpha"] = 1.03
    vparams["sfr_n"] = 1.15
    vparams["lmean"] = 2.22
    vparams["lsigma"] = 0.57
    
    opfile=opdir+"newE.pdf"
    zvals,std_pzgdm=plot_expectations(names,sdir,vparams,opfile)
    
    
def plot_expectations(names,sdir,vparams,opfile,intermediate=False,sumit=True):
    # Initialise surveys and grids
    # if True, this generates a summed histogram of all the surveys, weighted by
    # the observation time
    #sumit = True
    
    zvals = []
    dmvals = []
    grids = []
    surveys = []
    nozlist = []
    for i, name in enumerate(names):
        s, g = loading.survey_and_grid(
            survey_name=name, NFRB=None, sdir=sdir, lum_func=2
        )  # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        grids.append(g)
        surveys.append(s)
        
        # set up new parameters
        g.update(vparams)

        # gets cumulative rate distribution
        if i == 0:
            rtot = np.copy(g.rates) * s.TOBS
        else:
            rtot += g.rates * s.TOBS
        
        if s.zlist is not None:
            for iFRB in s.zlist:
                zvals.append(s.Zs[iFRB])
                dmvals.append(s.DMEGs[iFRB])
                
        if s.nozlist is not None:
            for dm in s.DMEGs[s.nozlist]:
                nozlist.append(dm)
        
        if intermediate:
            ############# do 2D plots ##########
            misc_functions.plot_grid_2(
                g.rates,
                g.zvals,
                g.dmvals,
                name=opdir + name + ".pdf",
                norm=3,
                log=True,
                #label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]",
                label="$\\log_{10}$ (detection probability)",
                project=False,
                FRBDM=s.DMEGs,
                FRBZ=s.frbs["Z"],
                Aconts=[0.01, 0.1, 0.5],
                zmax=1.5,
                DMmax=1500
            )  # ,DMlines=s.DMEGs[s.nozlist])
    DMEG220610=1458-31-50
    Z220610=1.0153
    
    DMEG190520=1204.7-60-50
    Z190520=0.241
    
    if sumit:
        # does the final plot of all data
        frbzvals = np.array(zvals)
        frbdmvals = np.array(dmvals)
        print("Plotting frb dm vals ",frbdmvals)
        ############# do 2D plots ##########
        misc_functions.plot_grid_2(
            rtot,
            g.zvals,
            g.dmvals,
            name=opfile,
            norm=3,
            log=True,
            #label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]",
            label="$\\log_{10}$ (detection probability)",
            project=False,
            FRBDM=frbdmvals,
            FRBZ=frbzvals,
            Aconts=[0.01, 0.1, 0.5],
            zmax=2,
            DMmax=3000,
            #DMlines=nozlist,
            special=[[DMEG220610,Z220610,'white','*'],[DMEG190520,Z190520,'black','+']]
        )
    # does plot of p(DM|z)
    ddm=g.dmvals[1]-g.dmvals[0]
    dz=g.zvals[1]-g.zvals[0]
    idm=int(DMEG220610/ddm)
    pzgdm = rtot[:,idm]/np.sum(rtot[:,idm])/dz
    return g.zvals,pzgdm

Planck_H0 = 67.4
def read_extremes(infile='planck_extremes.dat',H0=Planck_H0):
    """
    reads in extremes of parameters from a get_extremes_from_cube
    """
    f = open(infile)
    
    sets=[]
    
    for pset in np.arange(6):
        # reads the 'getting' line
        line=f.readline()
        
        pdict={}
        # gets parameter values
        for i in np.arange(6):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        sets.append(pdict)
        
        pdict={}
        # gets parameter values
        for i in np.arange(6):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        sets.append(pdict)
    return sets
main()
