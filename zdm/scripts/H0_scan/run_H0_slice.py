import argparse
import numpy as np
import os

from zdm import figures
from zdm import iteration as it

from zdm import parameters
from zdm import repeat_grid as zdm_repeat_grid
from zdm import MCMC
from zdm import survey
from astropy.cosmology import Planck18

from numpy import random
import matplotlib.pyplot as plt
import time

def main():
    
    t0 = time.time()
    parser = argparse.ArgumentParser()
    #parser.add_argument(dest='param',type=str,help="Parameter to do the slice in")
    parser.add_argument(dest='min',type=float,help="Min value")
    parser.add_argument(dest='max',type=float,help="Max value")
    parser.add_argument('-f', '--files', default=None, nargs='+', type=str, help="Survey file names")
    parser.add_argument('-s', '--smeared_surveys', default=None, nargs='+', type=str, help="Surveys with smeared z-vals")
    parser.add_argument('-z', '--frac_surveys', default=None, nargs='+', type=str, help="Surveys with  z-fraction")
    parser.add_argument('-n',dest='n',type=int,default=50,help="Number of values")   
    # parser.add_argument('-r',dest='repeaters',default=False,action='store_true',help="Surveys are repeater surveys")   
    args = parser.parse_args()

    vals = np.linspace(args.min, args.max, args.n)

    # Set state
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    # param_dict={'sfr_n': 1.13, 'alpha': 1.5, 'lmean': 2.27, 'lsigma': 0.55, 
    #         'lEmax': 41.26, 'lEmin': 39.5, 'gamma': -0.95, 'H0': 73,
    #         'min_lat': 0.0,  'sigmaDMG': 0.0, 'sigmaHalo': 20.0}
    # param_dict={'sfr_n': 0.8808527057055584, 'alpha': 0.7895161131856694, 
    #             'lmean': 2.1198711983468064, 'lsigma': 0.44944780033763343, 
    #             'lEmax': 41.18671139482926, 'lEmin': 39.81049090314043, 'gamma': -1.1558450520609953, 
    #             'H0': 54.6887137195215, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0, 'min_lat': 30.0}
    param_dict={'sfr_n': 1.7294049204398037, 'alpha': 1.4859524003747502, 
                 'lmean': 2.3007428869522486, 'lsigma': 0.396300210604263, 
                 'lEmax': 41.0, 'lEmin': 38.35533894604933, 'gamma': 0.6032500201815869, 
                 'H0': 70.51322705185869, 'DMhalo': 39.800465306883666}
    # param_dict={'lEmax': 40.578551786703116}
    state.update_params(param_dict)

    state.update_param('Rgamma', -2.2)
    state.update_param('lRmax', 3.0)
    state.update_param('lRmin', -4.0)
    #state.update_param('min_lat', 30.0)


    # Initialise surveys
    surveys_sep = [[], []]
    s_surveys_sep=[[],[]]

    grid_params = {}
    grid_params['dmmax'] = 7000.0
    grid_params['ndm'] = 1400
    grid_params['nz'] = 500
    ddm = grid_params['dmmax'] / grid_params['ndm']
    dmvals = (np.arange(grid_params['ndm']) + 1) * ddm
    
    if args.files is not None:
        for survey_name in args.files:
            s = survey.load_survey(survey_name, state, dmvals)
            surveys_sep[0].append(s)
    t1 = time.time()
    print("Step 1: ", str(t1-t0), flush=True)

    # state.update_param('halo_method', 1)
    # state.update_param(args.param, vals[0])
    
    outdir = 'cube/' + 'H0' + '/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ll_lists = []
    for val in vals:
        print("val:", val)
        param = {"H0": {'min': -np.inf, 'max': np.inf}}
        ll=0
        ll_list=[]
        for i, surveys in enumerate(args.files):
            state.photo.smearing=False
            surveys_sep[0][i].survey_data.observing.Z_FRACTION=None
            if args.smeared_surveys is not None:
                if surveys in args.smeared_surveys:
                    state.photo.smearing=True
            if args.frac_surveys is not None:
                if surveys in args.frac_surveys:
                    surveys_sep[0][i].survey_data.observing.Z_FRACTION="lsst_24.7"
            sll, sll_list = MCMC.calc_log_posterior([val], state, param,[[surveys_sep[0][i]],[]], grid_params, ind_surveys=True)#,psnr=True)
            for s in sll_list:
                ll_list.append(s)
            ll+=sll
        print(ll, ll_list)
        ll_lists.append(ll_list)
        t2 = time.time()
        print("Step 2: ", str(t2-t1), flush=True)
        t1 = t2
    print(ll_lists)
    ll_lists = np.asarray(ll_lists)

    plt.figure()
    linestyles=["-","--",":","-."]
    s_names=["None","Photometric","$F_z$","Both"]
    plt.clf()
    llsum = np.zeros(ll_lists.shape[0])
    surveys = surveys_sep[0] + surveys_sep[1]
    FWHM=[]
    for i in range(len(surveys)):
        s = surveys[i]
        lls = ll_lists[:, i]
        
        lls[lls < -1e10] = -np.inf
        lls[np.argwhere(np.isnan(lls))] = -np.inf
        
        llsum += lls
        
        lls = lls - np.max(lls)
        lls=10**lls
        index1=np.where(lls>=0.5)[0][0]
        index2=np.where(lls>=0.5)[0][-1]
        root1=vals[index1-1]-(0.5-lls[index1-1])*(vals[index1]-vals[index1-1])/(lls[index1]-lls[index1-1])
        root2=vals[index2]-(0.5-lls[index2])*(vals[index2+1]-vals[index2])/(lls[index2+1]-lls[index2])
        FWHM.append(root2-root1)
        # plt.figure()
        # plt.clf()
        plt.plot(vals, lls, label=s_names[i],ls=linestyles[i])
        plt.xlabel('$H_0 (km s^{-1} Mpc^{-1})$',fontsize=14)
        plt.ylabel('$\\frac{\\mathcal{L}}{max(\\mathcal{L})}$',fontsize=14)
        # plt.savefig(os.path.join(outdir, s.name + ".pdf"))
        print("Max H0:",vals[np.where(lls==1.0)[0][0]])
    
    plt.minorticks_on() 
    plt.tick_params(axis='y', which='major', labelsize=14)    # To set tick label fontsize
    plt.tick_params(axis='y', which='major', length=9)        # To set tick size
    plt.tick_params(axis='y', which='minor', length=4.5)      # To set tick size
    plt.tick_params(axis='y', which='both',direction='in',right='on', top='on')  

    plt.tick_params(axis='x', which='major', labelsize=14)    # To set tick label fontsize 
    plt.tick_params(axis='x', which='major', length=9)        # To set tick size 
    plt.tick_params(axis='x', which='minor', length=4.5)      # To set tick size
    plt.tick_params(axis='x', which='both',direction='in',right='on', top='on') 
    
    print(vals)
    print(llsum)
    peak=vals[np.argwhere(llsum == np.max(llsum))[0]]
    print("peak", peak,)
    #plt.axvline(peak,ls='--')
    plt.legend(fontsize=14,loc='upper left')
    plt.savefig(outdir + "H0" + ".pdf")
    percentage=(FWHM/FWHM[0]-1)*100
    print("FWHM:Spectroscopic,Photometric,zFrac,Photometric+zfrac\n",FWHM,percentage)

    # llsum = llsum - np.max(llsum)
    # llsum[llsum < -1e10] = -np.inf
    plt.figure()
    plt.clf()
    plt.plot(vals, llsum, label='Total')
    plt.axvline(peak,ls='--')
    # plt.plot(vals, llsum2)
    plt.xlabel('H0')
    plt.ylabel('log likelihood')
    plt.legend()
    plt.savefig(outdir + "H0"+ "_sum.pdf")

#==============================================================================
"""
Function: plot_grids
Date: 10/01/2024
Purpose:
    Plot grids. Adapted from zdm/scripts/plot_pzdm_grid.py

Imports:
    grids = list of grids
    surveys = list of surveys
    outdir = output directory
    val = parameter value for this grid
"""
def plot_grids(grids, surveys, outdir, val):
    for g,s in zip(grids, surveys):
        zvals=[]
        dmvals=[]
        nozlist=[]
        
        if s.zlist is not None:
            for iFRB in s.zlist:
                zvals.append(s.Zs[iFRB])
                dmvals.append(s.DMEGs[iFRB])
        if s.nozlist is not None:
            for dm in s.DMEGs[s.nozlist]:
                nozlist.append(dm)
        
        frbzvals = np.array(zvals)
        frbdmvals = np.array(dmvals)

        figures.plot_grid(
            g.rates,
            g.zvals,
            g.dmvals,
            name=outdir + s.name + "_" + str(val) + ".pdf",
            norm=3,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]",
            project=False,
            FRBDM=frbdmvals,
            FRBZ=frbzvals,
            Aconts=[0.01, 0.1, 0.5],
            zmax=1.5,
            DMmax=3000,
            # DMlines=nozlist,
        )

#==============================================================================
"""
Function: commasep
Date: 23/08/2022
Purpose:
    Turn a string of variables seperated by commas into a list

Imports:
    s = String of variables

Exports:
    List conversion of s
"""
def commasep(s):
    return list(map(str, s.split(',')))

#==============================================================================

main()
