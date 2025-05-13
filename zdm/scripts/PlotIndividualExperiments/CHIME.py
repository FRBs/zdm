'''
Plots total, single, and repeater z,DM distributions for
the CHIME Catalogue 1 survey

'''
# standard python imports
import numpy as np
from astropy.cosmology import Planck18
from matplotlib import pyplot as plt
import os
from pkg_resources import resource_filename

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
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    
    # shortest method to load CHIME grids. Slightly expanded below
    #ss,rgs,all_rates, all_singles, all_reps = loading.load_CHIME(state=state)
    
    # defines CHIME grids to load
    NDECBINS=6
    names=[]
    for i in np.arange(NDECBINS):
        name="CHIME_decbin_"+str(i)+"_of_6"
        names.append(name)
    survey_dir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/CHIME/')
    ss,gs = loading.surveys_and_grids(survey_names=names, init_state=state, rand_DMG=False,sdir = survey_dir, repeaters=True)
    
    # compiles sums over all six declination bins
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
    
    # set limits for plots   
    DMmax=3000
    zmax=3.
    
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    s=ss[0]
    g=gs[0]
    name = names[0]
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
    plt.xlim(0.01,3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+opname+"_pz.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.xlabel("DM")
    plt.ylabel("p(DM)")
    plt.xlim(0,4000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+opname+"_pdm.pdf")
    plt.close()

def plot_dm_z(g,ax1,ax2,array,tag):
    pz = np.sum(array,axis=1)
    #pz /= np.max(pz)
    ax1.plot(g.zvals,pz,label=tag)
    
    pdm = np.sum(array,axis=0)
    #pdm /= np.max(pdm)
    ax2.plot(g.dmvals,pdm,label=tag)

main()
