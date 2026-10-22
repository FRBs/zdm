""" 
This script creates a redshifty comparison figure of MeerTRAP,
ASKAP/CRACO (estimates), DSA, and CHIME


"""
import os

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import figures
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
    names=["MeerTRAPcoherent","MeerTRAPincoherent","DSA","CRAFT_ICS_1300", "FAST"]
    
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    ############ Save arrays ##########
    # np.save("MeerTRAP_zDM", g.rates)
    # np.save("MeerTRAP_zvals", g.zvals)
    # np.save("MeerTRAP_dmvals", g.dmvals)

    if False:
        #### plots p(z|DM) for the MeerTRAP FRB ####
        DMfrb = 2398
        iDM = np.where(DMfrb < gs[0].dmvals)[0][0]
        pzgdm = gs[0].rates[:,iDM]
        
        pzgdm /= np.sum(pzgdm)
        plt.figure()
        plt.xlabel("z")
        plt.ylabel("p(z|DMEG = 2398)")
        plt.plot(gs[0].zvals,pzgdm)
        plt.tight_layout()
        plt.savefig("pzgdm.png")
        plt.close()
    
        np.save("pzgdm.npy",pzgdm)
        np.save("zvals.npy",gs[0].zvals)
    

    # this is howto use the FRB library to load up known hosts
    if False:
        from frb.galaxies import utils as frb_gal_u
        
        # Load up the hosts
        host_tbl, _ = frb_gal_u.build_table_of_hosts(attrs=['redshift'])
        
        # Cut
        host_tbl = host_tbl[host_tbl['P_Ox'] > POx_min]
        
        # DMs
        DM_FRB = units.Quantity([frb.DM for frb in host_tbl.FRBobj.values])
        DM_ISM = units.Quantity([frb.DMISM for frb in host_tbl.FRBobj.values])
        DM_EG = DM_FRB - DM_ISM - DM_MWhalo
    
    ########### Get CHIME info ###########
    
    # defines CHIME grids to load
    NDECBINS=6
    cnames=[]
    chime_Zs = np.array([])
    chime_DMs = np.array([])
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
        chime_Zs = np.append(chime_Zs, css[i].Zs[css[i].zlist])
        chime_DMs = np.append(chime_DMs, css[i].DMEGs[css[i].zlist])
    print("CHIME_Zs", chime_Zs)
    print("CHIME_DMs", chime_DMs)

    ###### Get list of z and dm for DSA, CRAFT and CHIME localised FRBs #####
    ICS_names=["CRAFT_ICS_892", "CRAFT_ICS_1632"]
    ics_ss, ics_gs = loading.surveys_and_grids(survey_names=ICS_names, init_state=state)

    dsa_Zs = ss[2].Zs[ss[2].zlist]
    dsa_DMs = ss[2].DMEGs[ss[2].zlist]

    ics_Zs = np.array([ss[3].Zs[ss[3].zlist].tolist() + ics_ss[0].Zs[ics_ss[0].zlist].tolist() + ics_ss[1].Zs[ics_ss[1].zlist].tolist()])
    ics_DMs = np.array([ss[3].DMEGs[ss[3].zlist].tolist() + ics_ss[0].DMEGs[ics_ss[0].zlist].tolist() + ics_ss[1].DMEGs[ics_ss[1].zlist].tolist()])
    

    ###### plots MeerTRAP zDM figure ###########
    Zs = [np.array([2.148]), None, None, dsa_Zs, ics_Zs]
    DMs = [np.array([2398.03]), None, None, dsa_DMs, ics_DMs]
    point_labels = ["FRB 20240304B", None, None, "DSA", "ASKAP"]
    # point_labels = [None, None, None, None]

    # Set colours and styles for plotting contours and FRBs
    cmap = cmr.arctic
    data_clrs = cmap(np.linspace(0.0, 0.7, 5))
    # temp = data_clrs[1].copy()
    # data_clrs[1] = data_clrs[2]
    # data_clrs[2] = temp
    markers=["*", ".", "+", "o", "x"]
    markersize = [10, 4, 4, 4, 5]
    ewidths = [1,1,1,1,1]

    plt_dicts = []
    cont_dicts = []
    for i in range(len(data_clrs)):
        plt_styles = {
            'color': data_clrs[i],
            'marker': markers[i],
            'markersize': markersize[i],
            'label': None,
            'markeredgewidth': ewidths[i]
        }
        plt_dicts.append(plt_styles)

        # cont_styles = {
        #     'color': data_clrs[i],
        #     'label': point_labels[i]
        # }
        # cont_dicts.append(cont_styles)
    plt_dicts[0]['label'] = point_labels[0]
    # plt_dicts = None
    cont_dicts = None

    s=ss[0]
    g=gs[0]
    name = names[0]
    
    # Do the plotting
    # First plot the theoretical values
    zvals = g.zvals
    dmvals = g.dmvals
    ndm = dmvals.size
    nz = zvals.size
    dz = zvals[1] - zvals[0]
    ddm = dmvals[1] - dmvals[0]

    # l_meerkat = {'color': cont_clrs[0], 'linestyle': "--"}
    # l_dsa = {'color': cont_clrs[2], 'linestyle': "-.", 'marker': 'o', 'markeredgewidth': 1, 'markersize': 4}
    # l_chime = {'color': cont_clrs[1], 'linestyle': ":", 'markeredgewidth': 1, 'markersize': 4}
    # l_askap = {'color': cont_clrs[3], 'linestyle': "-", 'marker': 'x', 'markeredgewidth': 1, 'markersize': 5}
    # l_cont_dicts = [l_meerkat, l_chime, l_dsa, l_askap]
    
    # figures.plot_grid(g.rates,zvals,dmvals,
    #     name=opdir+name+"_zDM_combined.pdf",norm=3,log=True,
    #     label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$',
    #     project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
    #     zmax=4.5,DMmax=3500, FRBZs=Zs, FRBDMs=DMs, 
    #     plt_dicts=plt_dicts, cont_dicts=cont_dicts,
    #     Aconts=[0.1],othergrids=[gs[1].rates, gs[2].rates, crates, gs[3].rates],
    #     othernames = ["Coherent", "Incoherent","DSA", "CHIME", "ASKAP"], 
    #     cmap=cmr.prinsenvlag_r)
    #     #0.01, 0.1,0.5
    #     #Zs, DMs
    
    # plt.figure()
    # plt.clf()
    # muDMhost = np.log(10 ** state.host.lmean)
    # sigmaDMhost = np.log(10 ** state.host.lsigma)
    # meanHost = np.exp(muDMhost + sigmaDMhost ** 2 / 2.0)
    
    # plt.ylim(0, 3000)
    # plt.xlim(0, 2.8)
    # zmax = zvals[-1]
    # nz = zvals.size
    # # DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
    # DM_cosmic = pcosmic.get_mean_DM(zvals, state)

    # # idea is that 1 point is 1, hence...
    # # zeval = zvals / dz
    # DMEG_mean = (DM_cosmic + meanHost/(1+zvals))
    # # DMEG_mean[0] = 0.0
    # plt.plot(
    #     zvals,
    #     DMEG_mean,
    #     color="blue",
    #     linewidth=2,
    #     label="Macquart relation (mean)",
    # )
    # print(Zs, DMs)
    # for i in range(len(Zs)):
    #     plt.scatter(Zs[i], DMs[i], color=data_clrs[i], marker=markers[i], s=markersize[i] ** 2, label=point_labels[i])

    # plt.ylabel("${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$")
    # plt.xlabel("z")
    # plt.legend(loc="upper left", fontsize=12)
    # plt.savefig(opdir + "data_Macquart.pdf")


    # ############ Plots z projection ##########
    # plt.figure()
    
    # names = ["MeerTRAP coherent", "DSA 110", "ASKAP ICS"]
    # styles=["-","--","-."]

    for i,g in enumerate(gs):
        s=ss[i]
        
        # Calc pz
        pz = np.sum(g.rates,axis=1)
        dz = g.zvals[1] - g.zvals[0]
        pz = pz / np.sum(pz*dz)

        # Do plotting
        plt.plot(g.zvals,pz,label=names[i],linewidth=2)

        # Calculate z0 at which P(z < z0) = 0.95
        pz_cum = np.cumsum(pz) 
        i_one_percent = np.where(pz_cum>0.95)[0][0]
        one_percent = g.zvals[i_one_percent]
        print(s.name, one_percent, pz_cum[i_one_percent])

        # Calculate P(z > 2)
        i_z_two = np.where(g.zvals>2)[0][0]
        print("z>2", s.name, pz_cum[i_z_two])
        print(pz_cum[-1])
    
    # # adds CHIME
    # pz = np.sum(crates,axis=1)
    # pz = pz / np.sum(pz)

    # # Calculate z0 at which P(z < z0) = 0.95
    # pz_cum = np.cumsum(pz)
    # i_one_percent = np.where(pz_cum>0.95)[0][0]
    # one_percent = g.zvals[i_one_percent]
    # print("CHIME", one_percent, pz_cum[i_one_percent])
    
    # # Calculate P(z > 2)
    # i_z_two = np.where(g.zvals>2)[0][0]
    # print(pz_cum[i_z_two])
    
    # plt.plot(g.zvals,pz,label="CHIME",linestyle=":",linewidth=2)
    
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0.,5)
    plt.ylim(bottom=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(opdir+"pz_comparison.pdf")
    plt.close()
    

    
main()
