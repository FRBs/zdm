import argparse
import numpy as np
import os

from zdm import figures
from zdm import iteration as it

from zdm import parameters
from zdm import repeat_grid as zdm_repeat_grid
from zdm import MCMC
from zdm import survey
from zdm import misc_functions

from astropy.cosmology import Planck18

import matplotlib.pyplot as plt
import time
from pkg_resources import resource_filename

def main():
    
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='param',type=str,help="Parameter to do the slice in")
    parser.add_argument(dest='min',type=float,help="Min value")
    parser.add_argument(dest='max',type=float,help="Max value")
    parser.add_argument('-f', '--files', default=None, nargs='+', type=str, help="Survey file names")
    parser.add_argument('-r', '--rep_surveys', default=None, nargs='+', type=str, help="Surveys to consider repeaters in")
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
    # param_dict={'sfr_n': 3.1, 'alpha': 1.4859524003747502, 
    #              'lmean': 2.3007428869522486, 'lsigma': 0.396300210604263, 
    #              'lEmax': 40.5, 'lEmin': 39, 'gamma': -1.12, 
    #              'H0': 70.51322705185869, 'DMhalo': 39.800465306883666}
    param_dict={'sfr_n': 2.8727580728334483, 'alpha': 1.4311162666594126, 
            'lmean': 2.182113926164531, 'lsigma': 0.43672819419999337, 
            'lEmax': 40.91165578515364, 'lEmin': 38.394926807403984, 
            'gamma': -1.1268723802877352, 'H0': 70.6408065355808, 'DMhalo': 61.038340637162705,
            'halo_method': 0, 'sigmaDMG': 0.2, 'sigmaHalo': 15.0, 'min_lat': 20.0}
    # param_dict={'lEmax': 40.578551786703116}
    state.update_params(param_dict)

    state.update_param('Rgamma', -2.2)
    state.update_param('lRmax', 3.0)
    state.update_param('lRmin', -4.0)
    # state.update_param('min_lat', 30.0)

    # Initialise surveys
    surveys_sep = [[], []]

    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic', 
        datdir=resource_filename('zdm', 'GridData'))
    
    if args.files is not None:
        for survey_name in args.files:
            s = survey.load_survey(survey_name, state, dmvals, zvals)
            surveys_sep[0].append(s)
    
    if args.rep_surveys is not None:
        for survey_name in args.rep_surveys:
            s = survey.load_survey(survey_name, state, dmvals, zvals)
            surveys_sep[1].append(s)

    # state.update_param('halo_method', 1)
    # state.update_param(args.param, vals[0])
    
    outdir = 'cube/' + args.param + '/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ll_lists = []
    for val in vals:
        print("val:", val)
        param = {args.param: {'min': -np.inf, 'max': np.inf}}

        ll, ll_list = MCMC.calc_log_posterior([val], state, param, surveys_sep, ind_surveys=True, Pn=True, pNreps=True)
        print(ll, ll_list)
        ll_lists.append(ll_list)
    print(ll_lists)
    ll_lists = np.asarray(ll_lists)

    plt.figure()
    plt.clf()

    llsum = np.zeros(ll_lists.shape[0])
    surveys = surveys_sep[0] + surveys_sep[1]
    for i in range(len(surveys)):
        s = surveys[i]
        lls = ll_lists[:, i]
        
        lls[lls < -1e10] = -np.inf
        lls[np.argwhere(np.isnan(lls))] = -np.inf
        
        llsum += lls

        lls = lls - np.max(lls)

        # plt.figure()
        # plt.clf()
        plt.plot(vals, lls, label=s.name)
        plt.xlabel(args.param)
        plt.ylabel('log likelihood')
        # plt.savefig(os.path.join(outdir, s.name + ".pdf"))
    
    print(vals)
    print(llsum)
    peak=vals[np.argwhere(llsum == np.max(llsum))[0]]
    print("peak", peak)
    plt.axvline(peak)
    plt.legend()
    plt.savefig(outdir + args.param + ".pdf")

    # llsum = llsum - np.max(llsum)
    # llsum[llsum < -1e10] = -np.inf
    plt.figure()
    plt.clf()
    plt.plot(vals, llsum, label='Total')
    plt.axvline(peak)
    # plt.plot(vals, llsum2)
    plt.xlabel(args.param)
    plt.ylabel('log likelihood')
    plt.legend()
    plt.savefig(outdir + args.param + "_sum.pdf")

    np.save(outdir + args.param + "_vals.npy", vals)
    np.save(outdir + args.param + "_lls2.npy", llsum)

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
