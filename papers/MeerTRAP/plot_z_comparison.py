""" 
This script creates a redshifty comparison figure of MeerTRAP,
ASKAP/CRACO (estimates), DSA, and CHIME


"""
import os

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from zdm import optical as opt

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
from pkg_resources import resource_filename

import matplotlib
import cmasher as cmr
from astropy import units


defaultsize=12
ds=4
font = {
        # 'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    # in case you wish to switch to another output directory
    
    opdir='zcomparison/'
    
    # approximate best-fit values from recent analysis
    # best-fit from Jordan et al
    if True:
        # approximate best-fit values from recent analysis
        param_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37, 
                'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,
                'lC': -7.61, 'min_lat': 0.0}
        
    else:
        # best fit from James et al
        param_dict={'sfr_n': 1.13, 'alpha': 0.99, 'lmean': 2.27, 'lsigma': 0.55, 'lEmax': 41.26, 
                    'lEmin': 32, 'gamma': -0.95, 'H0': 73, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,
                    'lC': -0.76, 'min_lat': 0.0}
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    names=["MeerTRAPcoherent","DSA","CRAFT_ICS_1300"]
    
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    ############ Save arrays ##########
    # np.save("MeerTRAP_zDM", g.rates)
    # np.save("MeerTRAP_zvals", g.zvals)
    # np.save("MeerTRAP_dmvals", g.dmvals)

    # set limits for plots - will be LARGE!   
    
    ######### Loads public FRBs ####### -- Currently using zDM surveys instead
    # from frb.galaxies import utils as frb_gal_u
    
    # # Load up the hosts
    # host_tbl, _ = frb_gal_u.build_table_of_hosts() #attrs=['redshift']
    
    # # Cut
    # POx_min = 0.9
    # host_tbl = host_tbl[host_tbl['P_Ox'] > POx_min]
    
    # # DMs
    # DM_FRB = units.Quantity([frb.DM for frb in host_tbl.FRBobj.values])
    # DM_ISM = units.Quantity([frb.DMISM for frb in host_tbl.FRBobj.values])
    # DM_MWhalo = state.MW.DMhalo * units.pc / units.cm**3
    
    # DM_EG = DM_FRB - DM_ISM - DM_MWhalo

    # # zs
    # z = units.Quantity([frb.z for frb in host_tbl.FRBobj.values])

    ########### Get CHIME info ###########
    
    # defines CHIME grids to load
    NDECBINS=6
    cnames=[]
    for i in np.arange(NDECBINS):
        cname="CHIME_decbin_"+str(i)+"_of_6"
        cnames.append(cname)
    survey_dir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/CHIME/')
    css,cgs = loading.surveys_and_grids(survey_names=cnames, init_state=state, rand_DMG=False,sdir = survey_dir, repeaters=True)
    
    # compiles sums over all six declination bins
    crates = cgs[0].rates * 10**cgs[0].state.FRBdemo.lC * css[0].TOBS
    creps = cgs[0].exact_reps * cgs[0].state.rep.RC
    csingles = cgs[0].exact_singles * cgs[0].state.rep.RC
    
    for i,g in enumerate(cgs):
        s = css[i]
        if i ==0:
            continue
        else:
            crates += g.rates * 10**g.state.FRBdemo.lC * s.TOBS
            creps += g.exact_reps * g.state.rep.RC
            csingles += g.exact_singles * g.state.rep.RC


    ###### Get list of z and dm for DSA, CRAFT and CHIME localised FRBs #####
    ICS_names=["CRAFT_ICS_892", "CRAFT_ICS_1632"]
    ics_ss, ics_gs = loading.surveys_and_grids(survey_names=ICS_names, init_state=state)

    dsa_Zs = ss[1].Zs[ss[1].zlist]
    dsa_DMs = ss[1].DMEGs[ss[1].zlist]

    ics_Zs = np.array([ss[2].Zs[ss[2].zlist].tolist() + ics_ss[0].Zs[ics_ss[0].zlist].tolist() + ics_ss[1].Zs[ics_ss[1].zlist].tolist()])
    ics_DMs = np.array([ss[2].DMEGs[ss[2].zlist].tolist() + ics_ss[0].DMEGs[ics_ss[0].zlist].tolist() + ics_ss[1].DMEGs[ics_ss[1].zlist].tolist()])
    

    ###### plots MeerTRAP zDM figure ###########
    Zs = [np.array([2.148]), dsa_Zs, None, ics_Zs]
    DMs = [np.array([2398.03]), dsa_DMs, None, ics_DMs]
    point_labels = ["FRB 20240304B", None, None, None]

    # Set colours and styles for plotting contours and FRBs
    cmap = cmr.arctic
    data_clrs = cmap(np.linspace(0.0, 0.7, 4))
    temp = data_clrs[1].copy()
    data_clrs[1] = data_clrs[2]
    data_clrs[2] = temp
    cont_clrs = data_clrs
    markers=["*", "o", "o", "x"]
    markersize = [10, 4, 4, 5]


    plt_dicts = []
    for i in range(len(data_clrs)):
        styles = {
            'color': data_clrs[i],
            'marker': markers[i],
            'markersize': markersize[i],
            'label': point_labels[i]
        }
        plt_dicts.append(styles)


    s=ss[0]
    g=gs[0]
    name = names[0]

    # Do the plotting
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+name+"_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=3,DMmax=3000, FRBZs=Zs, FRBDMs=DMs, 
        # point_labels=point_labels, data_clrs=data_clrs, markersize=5, data_styles=markers,
        plt_dicts=plt_dicts,
        Aconts=[0.1],othergrids=[gs[1].rates,crates,gs[2].rates],
        othernames = ["MeerKAT","DSA","CHIME","ASKAP"], 
        cmap=cmr.prinsenvlag_r, cont_colours=cont_clrs)
        #0.01, 0.1,0.5
    
    
    ############ Plots z projection ##########
    plt.figure()
    
    names = ["MeerTRAP coherent", "DSA 110", "ASKAP ICS"]
    styles=["-","--","-."]

    for i,g in enumerate(gs):
        s=ss[i]
        
        # Calc pz
        pz = np.sum(g.rates,axis=1)
        pz = pz / np.sum(pz)

        # Do plotting
        plt.plot(g.zvals,pz,label=names[i],linestyle=styles[i],linewidth=2)

        # Calculate z0 at which P(z < z0) = 0.95
        pz_cum = np.cumsum(pz) 
        i_one_percent = np.where(pz_cum>0.95)[0][0]
        one_percent = g.zvals[i_one_percent]
        print(s.name, one_percent, pz_cum[i_one_percent])

        # Calculate P(z > 2)
        i_z_two = np.where(g.zvals>2)[0][0]
        print(pz_cum[i_z_two])
    
    # adds CHIME
    pz = np.sum(crates,axis=1)
    pz = pz / np.sum(pz)

    # Calculate z0 at which P(z < z0) = 0.95
    pz_cum = np.cumsum(pz)
    i_one_percent = np.where(pz_cum>0.95)[0][0]
    one_percent = g.zvals[i_one_percent]
    print("CHIME", one_percent, pz_cum[i_one_percent])
    
    # Calculate P(z > 2)
    i_z_two = np.where(g.zvals>2)[0][0]
    print(pz_cum[i_z_two])
    
    plt.plot(g.zvals,pz,label="CHIME",linestyle=":",linewidth=2)
    
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0.,3)
    plt.ylim(0,1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(opdir+"pz_comparison.pdf")
    plt.close()
    

    
main()
