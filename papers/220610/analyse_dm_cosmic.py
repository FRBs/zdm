""" 
This script creates zdm grids.

It steps through different effects, beginning with the
intrinsic zdm, applying various cuts.

It generates the following plots in opdir:

dm_cosmic.pdf: shows only intrinsic p(dm_cosmic|z)
dm_eg.pdf: shows intrinsic p(dm_host + dm_cosmic|z)

pEG_luminosity.pdf: shows p(DM,z) including
    - source evolution
    - FRB luminosity function
    - cosmological volume calculation

pEG_luminosity_eff.pdf: shows p(DM,z) as pEG_luminosity, adding in
    - detection efficiency losses from FRB width, DM smearing etc

pEG_luminosity_beam.pdf: shows p(DM,z) as pEG_luminosity, adding in
    - antenna beamshape

pEG_luminosity_eff_beam.pdf: shows p(DM,z) as pEG_luminosity, adding in
    - detection efficiency losses from FRB width, DM smearing etc
    - telescope beamshape
    (i.e. this is the full calculation of FRB rates)

pEG_luminosity_eff_beam_FRBs.pdf: as above, but shows detected FRBs


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
from zdm import analyze_cube as ac

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

def main():
    
    H0=80
    F=0.32
    
    
    name = '220610_only'
    sdir= "../../zdm/data/Surveys/"
    
    # approximate best-fit values from recent analysis
    vparams = {}
    vparams["H0"] = 67.4
    vparams["lEmax"] = 41.63
    vparams["gamma"] = -0.948
    vparams["alpha"] = 1.03
    vparams["sfr_n"] = 1.15
    vparams["lmean"] = 2.22
    vparams["lsigma"] = 0.57
    
    nozlist=[]
    s,g = loading.survey_and_grid(
        survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # remove effects of host galaxy
    orig_lmean = vparams['lmean']
    vparams['lmean'] = -5
    # set up new parameters
    g.update(vparams)
    
    # extracts data for that redshift
    Z220610=1.0153
    iz = np.where(Z220610 < g.zvals)[0][0]
    dz = g.zvals[1]-g.zvals[0]
    kz1 = (g.zvals[iz]-Z220610)/dz
    kz2 = 1.-kz1
    
    for array in [g.grid,g.rates]:
        v02,v12,k0,k1=ac.extract_limits(g.dmvals,array[iz,:],(1.-0.68)/2.,method=2)
        v01,v11,k0,k1=ac.extract_limits(g.dmvals,array[iz-1,:],(1.-0.68)/2.,method=2)
    
        v0 = kz1*v01 + kz2*v02
        v1 = kz1*v11 + kz2*v12
        print("One sigma intrinsic limits are ",v0,v1)
        cs = np.cumsum(array[iz,:])
        cs /= cs[-1]
        imid2 = np.where(cs > 0.5)[0][0]
    
        cs = np.cumsum(array[iz-1,:])
        cs /= cs[-1]
        imid1 = np.where(cs > 0.5)[0][0]
    
        dmmid = kz1*g.dmvals[imid1] + kz2*g.dmvals[imid2]
    
        print("Median is at ",dmmid)
    exit()
    #v0,v1,k0,k1=ac.extract_limits(g.dmvals,g.rates[iz,:],(1.-0.68)/2.)
    #print("One sigma effective limits are ",v0,v1)
    
    plt.figure()
    plt.xlim(0,1200)
    plt.plot(g.dmvals,g.grid[iz,:]/np.sum(g.grid[iz,:]/dz),label='before bias')
    plt.plot(g.dmvals,g.rates[iz,:]/np.sum(g.rates[iz,:]/dz),label='after bias')
    plt.tight_layout()
    plt.savefig('dm_cosmic.pdf')
    plt.close()
    exit()
    
    # set limits for plots - will be LARGE!   
    DMmax=4000
    zmax=3
    
    ####### plots the p(DMcosmic|z) grid - intrinsic distribution
    misc_functions.plot_grid_2(g.grid,g.zvals,g.dmvals,
        name=opdir+'pcosmic.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM}|z)$ [a.u.]',
        project=False, ylabel='${\\rm DM}_{\\rm IGM}$',
        zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state)
    exit()
    # restore host galaxy contribution   
    vparams['lmean'] = orig_lmean
    g.update(vparams)
            
    # plots the p(DMEG (host + cosmic)|z) grid
    misc_functions.plot_grid_2(g.smear_grid,g.zvals,g.dmvals,
        name=opdir+'pEG.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}|z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state)
    
    ######## plots while ignoring beam, bias against DM ######
    
    # saves some data
    orig_efficiencies=g.efficiencies
    orig_beam_b=g.beam_b
    orig_beam_o=g.beam_o
    orig_weights=g.weights
    
    
    # sets up fake "efficiencies". Equivalent of no penalty to detection
    # due to DM smearing, FRB width, etc
    # not quite 'perfect' - "1" is relative to a width of 1ms, while
    # in theory an FRB with width 0.01 ms would be detectable at
    # efficiency of 10 (1% of noise, same fluence) by the proper system
    fake=np.full([g.dmvals.size],1.)
    weights=None
    g.calc_thresholds(s.meta['THRESH'],fake)
    
    # calculates the detection probability assuming a 'perfect' beam
    # i.e. 100% of beamshape has 100% of max sensitivity
    g.calc_pdv(beam_b=np.array([1.]),beam_o=np.array([1.]))
    g.calc_rates()
    
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+'pEG_luminosity.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state)
    
    # somehow this gave errors, not yet debugged
    fixed=False
    if fixed:
        # calculates a probability of DM given z
        pz=np.sum(g.rates,axis=1)
        pdm_given_z=g.rates.T / pz
        # will try to do better...
        misc_functions.plot_grid_2(temp.T,g.zvals,g.dmvals,
            name=opdir+'pEG_luminosity_only_dmgz.pdf',norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}|z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
            zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state)
                
    # adds in effect of FRB width and detection efficiency
    # does a false grid with no penalty to DM efficiency
    #zmax=1.5
    #DMmax=2000
    
    ####### calculates full beamshape effects #######
    g.calc_pdv(beam_b=orig_beam_b,beam_o=orig_beam_o)
    g.calc_rates()
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+'pEG_luminosity_beam.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state)
    
    
    ##### plots the p(z,DM) grid with correct efficiencies, but ignoring beamshape #####
    g.calc_thresholds(s.meta['THRESH'],orig_efficiencies,weights=orig_weights)
    g.calc_pdv(beam_b=np.array([1.]),beam_o=np.array([1.]))
    g.calc_rates()
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+'pEG_luminosity_eff.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state)
    
    ######## calculates full zdm grid including everything ########
    g.calc_pdv(beam_b=orig_beam_b,beam_o=orig_beam_o)
    g.calc_rates()
    
    # plots without showing FRBs
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+'pEG_luminosity_eff_beam.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state)
    
    # plots while showing FRBs  
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+'pEG_luminosity_eff_beam_FRBs.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,DMlines=nozlist,Macquart=g.state,
        FRBDM=s.DMEGs,FRBZ=s.frbs["Z"])

    
main()
