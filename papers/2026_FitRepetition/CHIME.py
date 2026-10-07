'''
Plots total, single, and repeater z,DM distributions for
the CHIME Catalogue 1 survey

'''
# standard python imports
import numpy as np
import scipy.stats as st
from astropy.cosmology import Planck18
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})

import os
from zdm import states
import importlib.resources as resources

# zdm imports
from zdm import loading
from zdm import parameters
from zdm import figures

def main():
    '''
    Main program to evaluate log0-likelihoods and predictions for
    repeat grids
    '''
    
    opname="CHIME"
    opdir = opname+'/'
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # sets basic state and cosmology
    state = states.load_state("HoffmannRepeaters26Prep",scat="updated")
    
    # defines CHIME grids to load
    NDECBINS=6
    names=[]
    for i in np.arange(NDECBINS):
        name="CHIME_decbin_"+str(i)+"_of_6"
        names.append(name)
    sdir = resources.files('zdm').joinpath('data/Surveys/CHIME')
    ss,gs = loading.surveys_and_grids(survey_names=names, init_state=state, rand_DMG=False,sdir = sdir, repeaters=True)
    
    # compiles sums over all six declination bins
    gs[0].calc_constant(verbose=True)
    rates = gs[0].rates * 10**gs[0].state.FRBdemo.lC * ss[0].TOBS
    reps = gs[0].exact_reps * gs[0].state.rep.RC
    singles = gs[0].exact_singles * gs[0].state.rep.RC
    
    for i,g in enumerate(gs):
        s = ss[i]
        if i ==0:
            continue
        else:
            rates += g.rates * 10**g.state.FRBdemo.lC * s.TOBS
            reps += g.exact_reps * g.state.rep.RC
            singles += g.exact_singles * g.state.rep.RC

    np.save(opdir+opname+"_bursts.npy", rates)
    np.save(opdir+opname+"_reps.npy", reps)
    np.save(opdir+opname+"_singles.npy", singles)

    # Plot dm and z idstributions
    # plot_dm_z(gs[0], rates, reps, singles, opdir, opname)

    # Plot cumulative dm distributions
    plot_cumulative(gs[0], ss, reps+singles, reps, singles, opdir, opname, 10**state.host.lmean)


'''
===================================================================================================
Function: plot_dm_z
Plots the DM-z distribution for the CHIME survey, including total, single, and repeater distributions. 
It also calculates and prints the probability of z being greater than a specified value

    g: The grid object containing the DM-z distribution data.
    rates: The 2D z-DM grid of total rates.
    reps: The 2D z-DM grid of repeater rates.
    singles: The 2D z-DM grid of single rates.
    opdir: The output directory where the plots will be saved.
    opname: The name of the output files for the plots.

    output: Saves the plots of the DM-z distributions and the probability density functions for 
            z and DM to the specified output directory.
===================================================================================================
'''
def plot_dm_z(g, rates, reps, singles, opdir, opname):
    # set limits for plots   
    DMmax=3000
    zmax=3.

    # p(z > x)
    z0 = 1.0
    pz = np.sum(rates,axis=1)
    dz = g.zvals[1]-g.zvals[0]
    pz /= np.sum(pz)*dz  # Normalize to make it a probability density function
    pz_gt_z0 = np.sum(pz[np.where(g.zvals>z0)])*dz
    print("p(z > "+str(z0)+") = ",pz_gt_z0)
    
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    figures.plot_grid(rates,g.zvals,g.dmvals,
        name=opdir+opname+"total_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
    
    figures.plot_grid(reps,g.zvals,g.dmvals,
        name=opdir+opname+"reps_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
        
    figures.plot_grid(singles,g.zvals,g.dmvals,
        name=opdir+opname+"singles_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
        
    tags = [" (all bursts)"," (singles)"," (repeaters)"]
    plot_dm_z(g,ax1,ax2,rates,opname+" (all bursts)")
    plot_dm_z(g,ax1,ax2,singles,opname+" (singles)")
    plot_dm_z(g,ax1,ax2,reps,opname+" (repeaters)")

    plt.sca(ax1)
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0,1.5)
    # plt.ylim(0,1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+opname+"_pz.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.xlabel("DM")
    plt.ylabel("p(DM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+opname+"_pdm.pdf")
    plt.close()

'''
===================================================================================================
Function: plot_cumulative
Plots the cumulative distribution functions (CDFs) for the DM distributions of the CHIME survey, 
including total, single, and repeater distributions. It also performs the Kolmogorov-Smirnov (KS) 
test to compare the empirical CDFs with the model CDFs and prints the KS statistics and p-values.

    g: The grid object containing the DM distribution data.
    ss: A list of survey objects containing the DM data for each declination bin.
    rates: The 2D z-DM grid of total rates.
    reps: The 2D z-DM grid of repeater rates.
    singles: The 2D z-DM grid of single rates.
    opdir: The directory where the output files will be saved.
    opname: The name of the output file.
    muhost: The mean DM of the host galaxy.

    output: Saves the plot of the cumulative distribution functions to the specified output 
            directory and prints the KS statistics and p-values for the comparisons.
===================================================================================================
'''
def plot_cumulative(g,ss,rates,reps,singles, opdir, opname, muhost=0.0):

    # Model predictions
    cumModel = np.cumsum(np.sum(rates,axis=0))/np.sum(rates)
    cumReps = np.cumsum(np.sum(reps,axis=0))/np.sum(reps)
    cumSingles = np.cumsum(np.sum(singles,axis=0))/np.sum(singles)

    # Cat 1
    DMs = []
    DMSingles = []
    DMReps = []
    for s in ss:
        DMs.append(s.DMEGs)
        DMSingles.append(s.DMEGs[s.singleslist])
        DMReps.append(s.DMEGs[s.replist])
    DMs = np.concatenate(DMs)
    DMSingles = np.concatenate(DMSingles)
    DMReps = np.concatenate(DMReps)
    # print("DMs, Singles, Reps:", DMs, DMSingles, DMReps)

    DMs = np.sort(DMs)
    cumData = np.arange(1, len(DMs)+1) / len(DMs)
    DMs = np.append(DMs, 3000)
    cumData = np.append(cumData, 1.0)
    cdf_interp = np.interp(DMs, g.dmvals, cumModel)
    stat, p = st.ks_2samp(cumData, cdf_interp)
    print("Cat 1 KS statistic:", stat)
    print("Cat 1 p-value:", p)

    stat, p = st.ks_2samp(cumData[np.where(DMs>200)], cdf_interp[np.where(DMs>200)])
    print("Cat 1 KS statistic (DM>200):", stat)
    print("Cat 1 p-value (DM>200):", p)

    DMSingles = np.sort(DMSingles)
    cumDataSingles = np.arange(1, len(DMSingles)+1) / len(DMSingles)
    cumModelSingles = np.cumsum(np.sum(singles,axis=0))/np.sum(singles)
    cdf_interp_singles = np.interp(DMSingles, g.dmvals + muhost, cumModelSingles)
    stat_singles, p_singles = st.ks_2samp(cumDataSingles, cdf_interp_singles)
    print("Cat 1 KS statistic (singles):", stat_singles)
    print("Cat 1 p-value (singles):", p_singles)

    DMReps = np.sort(DMReps)
    cumDataReps = np.arange(1, len(DMReps)+1) / len(DMReps)
    cumModelReps = np.cumsum(np.sum(reps,axis=0))/np.sum(reps)
    cdf_interp_reps = np.interp(DMReps, g.dmvals + muhost, cumModelReps)
    stat_reps, p_reps = st.ks_2samp(cumDataReps, cdf_interp_reps)
    print("Cat 1 KS statistic (repeaters):", stat_reps)
    print("Cat 1 p-value (repeaters):", p_reps)

    # Catalogue 2
    cat2 = np.load(resources.files('zdm').joinpath('data/Surveys/chimefrbcat2.npy'))
    print("Catalogue 2 bursts:", len(cat2))

    DMs2reps = []
    DMs2singles = []
    repnames = []
    for i in range(len(cat2)):
        if cat2[i]['sub_num'] == 0:
            if cat2[i]['repeater_name'] != '':
                if cat2[i]['repeater_name'] not in repnames:
                    repnames.append(cat2[i]['repeater_name'])
                    DMs2reps.append(cat2[i]['dm_exc_ne2001']-ss[0].DMhalo)
            else:
                DMs2singles.append(cat2[i]['dm_exc_ne2001']-ss[0].DMhalo)

    # DMs2reps = [cat2[i]['dm_exc_ne2001'] for i in range(len(cat2)) if cat2[i]['repeater_name'] != '']
    # DMs2singles = [cat2[i]['dm_exc_ne2001'] for i in range(len(cat2)) if cat2[i]['repeater_name'] == '']
    DMs2 = DMs2reps + DMs2singles

    print(len(DMs2), len(DMs2reps), len(DMs2singles))

    DMs2reps = np.sort(DMs2reps)
    DMs2singles = np.sort(DMs2singles)
    cumData2reps = np.arange(1, len(DMs2reps)+1) / len(DMs2reps)
    cumData2singles = np.arange(1, len(DMs2singles)+1) / len(DMs2singles)

    DMs2 = np.sort(DMs2)
    cumData2 = np.arange(1, len(DMs2)+1) / len(DMs2)
    cdf_interp2 = np.interp(DMs2, g.dmvals, cumModel)
    stat, p = st.ks_2samp(cumData2, cdf_interp2)
    print("Cat 2 KS statistic:", stat)
    print("Cat 2 p-value:", p)

    stat, p = st.ks_2samp(cumData2[np.where(DMs2>200)], cdf_interp2[np.where(DMs2>200)])
    print("Cat 2 KS statistic (DM>200):", stat)
    print("Cat 2 p-value (DM>200):", p)

    cdf_interp2_1 = np.interp(DMs, DMs2, cumData2)
    stat, p = st.ks_2samp(cumData, cdf_interp2_1)
    print("Cat 1 vs 2 KS statistic:", stat)
    print("Cat 1 vs 2 p-value:", p)

    plt.figure()

    plt.plot(g.dmvals,cumModel,label="CHIME model", linewidth=3, color="red", linestyle='--', zorder=10)
    # plt.plot(g.dmvals,cumReps,label="CHIME model (repeaters)", linewidth=3, color="red", linestyle=':', zorder=10)
    # plt.plot(g.dmvals,cumSingles,label=opname+" model (singles)", linewidth=3, color="tab:purple", linestyle='--', zorder=10)

    plt.plot(DMs,cumData,label="CHIME cat 1", linewidth=3, color='orange')
    plt.plot(DMs2,cumData2,label="CHIME cat 2", linewidth=3, color="tab:blue")

    # plt.plot(DMReps,cumDataReps,label="CHIME cat 1 repeaters", linewidth=3, color="orange", linestyle=':')
    # plt.plot(DMSingles,cumDataSingles,label="CHIME cat 1 singles", linewidth=3, color="tab:purple")

    # plt.plot(DMs2reps,cumData2reps,label="CHIME cat 2 repeaters", linewidth=3, color="tab:blue", linestyle=':')
    # plt.plot(DMs2singles,cumData2singles,label="CHIME cat 2 singles", linewidth=3, color="tab:purple")

    # plt.plot(g.dmvals + 10**state.host.lmean,cumModelSingles,label="CHIME model (singles)", linewidth=2)
    # plt.plot(DMSingles,cumDataSingles,label="CHIME data (singles)", linewidth=2)

    # plt.plot(g.dmvals + 10**state.host.lmean,cumModelReps,label="CHIME model (repeaters)", linewidth=2)
    # plt.plot(DMReps,cumDataReps,label="CHIME data (repeaters)", linewidth=2)

    plt.xlabel(r"DM$_{\rm EG}$ (pc cm$^{-3}$)")
    plt.ylabel("CDF")
    plt.xscale("log")
    plt.xlim(50,3000)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+opname+"_pdm_cumulative.pdf")
    plt.close()

def plot_dm_z(g,ax1,ax2,array,tag):
    pz = np.sum(array,axis=1)
    pz /= np.sum(pz)*(g.zvals[1]-g.zvals[0])  # Normalize to make it a probability density function
    #pz /= np.max(pz)
    ax1.plot(g.zvals,pz,label=tag)
    
    pdm = np.sum(array,axis=0)
    pdm /= np.sum(pdm)*(g.dmvals[1]-g.dmvals[0])  # Normalize to make it a probability density function
    #pdm /= np.max(pdm)
    ax2.plot(g.dmvals,pdm,label=tag)

main()
