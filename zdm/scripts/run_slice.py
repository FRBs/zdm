import argparse
import numpy as np
import os

from zdm import loading
from zdm import misc_functions
from zdm import iteration as it

from zdm import parameters
from astropy.cosmology import Planck18

import matplotlib.pyplot as plt

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='names',type=commasep,help='Survey names')
    parser.add_argument(dest='param',type=str,help="Parameter to do the slice in")
    parser.add_argument(dest='min',type=float,help="Min value")
    parser.add_argument(dest='max',type=float,help="Max value")
    parser.add_argument('-n',dest='n',type=int,default=50,help="Number of values")   
    parser.add_argument('-r',dest='repeaters',default=False,action='store_true',help="Surveys are repeater surveys")   
    args = parser.parse_args()

    vals = np.linspace(args.min, args.max, args.n)

    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    # state = loading.set_state()
    # state.update_param('Rgamma', -1.5)
    # state.update_param('lRmax', 1.0)
    # state.update_param('lRmin', -4.0)
    # state.update_param('luminosity_function', 2)
    # state.update_param('H0', 100.0)
    # state.update_param('sfr_n', 1.18)
    # state.update_param('alpha', 1.17)
    # state.update_param('lmean', 2.16)
    # state.update_param('lsigma', 0.5)
    # state.update_param('lEmax', 41.05)
    # state.update_param('gamma', -0.995)
    # state.update_param('sigmaDMG', 0.0)
    # state.update_params({'sfr_n': 0.6858762799998724, 'alpha': 1.7665198706279686, 'lmean': 2.074825172832976, 'lsigma': 0.4003714831421404, 'lEmax': 41.13739600201252, 'lEmin': 39.551691554143936, 'gamma': -1.0348224611860115, 'H0': 61.22965004043496})
    # state.update_params({'sfr_n': 0.8806591144921403, 'alpha': 1.0451512509567609, 'lmean': 2.0411626762512824, 'lsigma': 0.4285714684532393, 'lEmax': 41.45631839060552, 'lEmin': 39.52262703306915, 'gamma': -1.1856556240866645, 'H0': 57.59867790323104})
    # param_dict={'sfr_n': 0.8808527057055584, 'alpha': 0.7895161131856694, 'lmean': 2.1198711983468064, 'lsigma': 0.44944780033763343, 'lEmax': 41.18671139482926, 
    #             'lEmin': 39.81049090314043, 'gamma': -1.1558450520609953, 'H0': 54.6887137195215, 'halo_method': 2}
    param_dict={'sfr_n': 0.8808527057055584, 'alpha': 0.7895161131856694, 'lmean': 2.1198711983468064, 'lsigma': 0.44944780033763343, 'lEmax': 41.18671139482926, 
                'lEmin': 39.81049090314043, 'gamma': -1.1558450520609953, 'H0': 54.6887137195215, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0}
    state.update_params(param_dict)
    # state.update_param('halo_method', 1)
    state.update_param(args.param, vals[0])

    surveys, grids = loading.surveys_and_grids(survey_names = args.names, repeaters=args.repeaters, init_state=state)
    
    outdir = 'cube/' + args.param
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    llsum = np.zeros(len(vals))
    lls_list = [[] for _ in range(len(grids))]

    for val in vals:
        print("val:", val)
        vparams = {}
        vparams[args.param] = val

        newC, llC = it.minimise_const_only(vparams, grids, surveys,  update=True)

        ll=0
        for i, g in enumerate(grids):
            g.state.FRBdemo.lC = newC

            # if isinstance(g, zdm_repeat_grid.repeat_Grid):
            #     g.calc_constant()
            
            # try:
            ll = it.get_log_likelihood(g, surveys[i], Pn=True, psnr=True)
            # except ValueError:
            #     ll = -np.inf

            # plot_grids(grids, surveys, outdir, val)
                
            lls_list[i].append(ll)
            print("ll", ll)

    plt.figure()
    plt.clf()
    for s, lls in zip(surveys, lls_list):
        lls = np.array(lls)
        lls[lls < -1e10] = -np.inf
        lls[np.argwhere(np.isnan(lls))] = -np.inf
        # print(lls, len(s.DMs))
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
    plt.savefig(outdir + ".pdf")

    # llsum = llsum - np.max(llsum)
    # llsum[llsum < -1e10] = -np.inf
    plt.figure()
    plt.clf()
    plt.plot(vals, llsum, label='Total')
    # plt.plot(vals, llsum2)
    plt.xlabel(args.param)
    plt.ylabel('log likelihood')
    plt.legend()
    plt.savefig(outdir + "_sum.pdf")

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

        misc_functions.plot_grid_2(
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
