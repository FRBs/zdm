""" 
This script creates zdm grids and plots localised FRBs

It can also generate a summed histogram from all CRAFT data

"""
import os

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import time

def main():

    # in case you wish to switch to another output directory
    opdir = "reps/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)

    # Initialise surveys and grids

    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    # names = ['private_CRAFT_ICS','private_CRAFT_ICS_892','private_CRAFT_ICS_1632']
    

    # Public CRAFT FRBs
    # names = ["CRAFT_ICS", "CRAFT_ICS_892", "CRAFT_ICS_1632"]

    # Examples for other FRB surveys
    # names = ["FAST", "Arecibo", "parkes_mb_class_I_and_II"]
    
    sdir = "../data/Surveys/"
    rep_sdir = "../data/Surveys/CHIME/"
    edir = "../data/Efficiencies/"
    names = ["FAST", "CRAFT_class_I_and_II"]
    rep_names=None
    # names = ["DSA","FAST","FAST2","CRAFT_class_I_and_II","private_CRAFT_ICS_892","private_CRAFT_ICS_1300","private_CRAFT_ICS_1632","parkes_mb_class_I_and_II"]
    # rep_names = ["CHIME_decbin_0_of_6","CHIME_decbin_1_of_6","CHIME_decbin_2_of_6","CHIME_decbin_3_of_6","CHIME_decbin_4_of_6","CHIME_decbin_5_of_6"]
    # names = ["parkes_mb_class_I_and_II", "DSA3", "private_CRAFT_ICS_1632"]

    # if True, this generates a summed histogram of all the surveys, weighted by
    # the observation time
    sumit=False
    
    # approximate best-fit values from 220610 analysis
    # vparams = {}
    # vparams['H0'] = 73
    # vparams['lEmax'] = 41.3
    # vparams['gamma'] = -0.9
    # vparams['alpha'] = 1
    # vparams['sfr_n'] = 1.15
    # vparams['lmean'] = 2.25
    # vparams['lsigma'] = 0.55
    # vparams = {
    #     'logF': -1.464148638577786, 
    #     'sfr_n': 0.6256539113545991, 
    #     'alpha': 2.569072338364329, 
    #     'lmean': 1.6016037127430354, 
    #     'lsigma': 0.7369492794796966, 
    #     'lEmax': 41.76406466597176, 
    #     'gamma': -0.04810183607795404, 
    #     'H0': 45.63834807349863, 
    #     # 'lEmin': 38.9278537549956, 
    #     'DMhalo': 84.8982781888609, 
    #     'lRmin': -1.870300691808065, 
    #     'lRmax': 0.9269620788119024, 
    #     'Rgamma': -3.633111972331566}

    vparams = {'sfr_n': 1.8342406907210933, 'alpha': 1.8333854125000386, 'lmean': 2.192135707972721, 'lsigma': 0.38705389085781317, 'lEmax': 41.27201813996426, 'lEmin': 39.22622840814371, 'gamma': -1.32556116225346, 'H0': 65.93010516823207}
    
    zvals=[]
    dmvals=[]
    grids=[]
    surveys=[]
    nozlist=[]
    
    # writs the Macquart relation - temporary!
    if False:
        from frb.dm import igm
        zmax=1.4
        nz=1000
        DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
        for i,DM in enumerate(DMbar):
            print(zeval[i],DM)

    t0 = time.time()    
    if names is not None:
        surveys, grids = loading.surveys_and_grids(survey_names = names, repeaters=False, sdir=sdir, edir=edir)
    else:
        surveys = []
        grids = []
    
    if rep_names is not None:
        rep_surveys, rep_grids = loading.surveys_and_grids(survey_names = rep_names, repeaters=True, sdir=rep_sdir, edir=edir)
        for s,g in zip(rep_surveys, rep_grids):
            surveys.append(s)
            grids.append(g)

    t1 = time.time()

    llsum = 0
    for i,name in enumerate(names):
        g = grids[i]
        s = surveys[i]

        # set up new parameters
        g.update(vparams)
        llsum += it.get_log_likelihood(g,s, Pn=False)

    t2 = time.time()

    # print("Initialising: " + str(t1 - t0))
    # print("Updating: " + str(t2 - t1))
    print("llsum = ", llsum)

    exit()
        # gets cumulative rate distribution
        # if i == 0:
        #     rtot = np.copy(g.rates) * s.TOBS
        # else:
        #     rtot += g.rates * s.TOBS

        # if name == "Arecibo":
        #     # remove high DM vals from rates as per ALFA survey limit
        #     delete=np.where(g.dmvals > 2038)[0]
        #     g.rates[:,delete]=0.
        
        # # print("meow",s.nozlist, s.nozreps, s.nozsingles)
        # # print("meow 2", s.zlist, s.zreps, s.zsingles)
        # # print("meow 3", s.nD, s.nDr, s.nDs)
        
        # if s.zlist is not None:
        #     for iFRB in s.zlist:
        #         zvals.append(s.Zs[iFRB])
        #         dmvals.append(s.DMEGs[iFRB])
        # if s.nozlist is not None:
        #     for dm in s.DMEGs[s.nozlist]:
        #         nozlist.append(dm)
        #     #print("nolist is now ",nozlist)
        # print("About to plot")
        # # print(g.rates)
        # ############# do 2D plots ##########
        
        # misc_functions.plot_grid_2(
        #     g.rates,
        #     g.zvals,
        #     g.dmvals,
        #     name=opdir + name + ".pdf",
        #     norm=3,
        #     log=True,
        #     label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]",
        #     project=False,
        #     FRBDM=s.DMEGs,
        #     FRBZ=s.frbs["Z"],
        #     Aconts=[0.01, 0.1, 0.5],
        #     zmax=1.5,
        #     DMmax=4000,
        # )  # ,DMlines=s.DMEGs[s.nozlist])

    if sumit:
        # does the final plot of all data
        frbzvals = np.array(zvals)
        frbdmvals = np.array(dmvals)
        ############# do 2D plots ##########
    
        misc_functions.plot_grid_2(
            g.rates,
            g.zvals,
            g.dmvals,
            name=opdir + "combined_localised_FRBs.pdf",
            norm=3,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]",
            project=False,
            FRBDM=frbdmvals,
            FRBZ=frbzvals,
            Aconts=[0.01, 0.1, 0.5],
            zmax=1.5,
            DMmax=2000,
            DMlines=nozlist,
        )
        
        misc_functions.plot_grid_2(rtot,g.zvals,g.dmvals,
            name=opdir+'combined_localised_FRBs.pdf',norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
            project=False,FRBDM=frbdmvals,FRBZ=frbzvals,Aconts=[0.01,0.1,0.5],
            zmax=2.0,DMmax=2000)
        
        DMhost=80.
        misc_functions.plot_grid_2(g.grid,g.zvals,g.dmvals,
            name=opdir+'pdmIGMgz_localised_FRBs.pdf',norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm IGM}|z)$ [a.u.]',
            ylabel='${\\rm DM}_{\\rm cosmic}$',
            project=False,FRBDM=frbdmvals-DMhost,FRBZ=frbzvals,
            zmax=1.2,DMmax=1400,
            pdmgz=[0.05,0.5,0.95])
        
        misc_functions.plot_grid_2(g.grid,g.zvals,g.dmvals,
            name=opdir+'pdmIGMgz_localised_FRBs_alt.pdf',norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm IGM}|z)$ [a.u.]',
            ylabel='${\\rm DM}_{\\rm cosmic}$',
            project=False,FRBDM=frbdmvals-DMhost,FRBZ=frbzvals,
            zmax=1.2,DMmax=1400,cmap="Oranges",data_clr='black',
            pdmgz=[0.05,0.5,0.95])
        
        
        misc_functions.plot_grid_2(g.grid,g.zvals,g.dmvals,
            name=opdir+'pdmEGgz_localised_FRBs.pdf',norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm EG}|z)$ [a.u.]',
            project=False,FRBDM=frbdmvals,FRBZ=frbzvals,
            zmax=1.2,DMmax=1400,
            pdmgz=[0.05,0.5,0.95])
        
        misc_functions.plot_grid_2(g.smear_grid,g.zvals,g.dmvals,
            name=opdir+'pdmEGgz_localised_FRBs_alt.pdf',norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm EG}|z)$ [a.u.]',
            project=False,FRBDM=frbdmvals,FRBZ=frbzvals,
            zmax=1.2,DMmax=1400,cmap="Oranges",data_clr='black',
            pdmgz=[0.05,0.5,0.95])

main()
