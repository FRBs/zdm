""" Run on Emax """

######
# first run this to generate surveys and parameter sets, by 
# setting NewSurveys=True NewGrids=True
# Then set these to False and run with command line arguments
# to generate *many* outputs
#####

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import pandas

from zdm import iteration as it
from zdm.craco import loading
from zdm import io
from zdm import real_loading

from IPython import embed

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

#import igm
defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(pargs):


    ############## Load up ##############
    input_dict=io.process_jfile('best_James22.json')

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    # State
    state = real_loading.set_state()
    state.update_param_dict(state_dict)

    ############## Initialise ##############
    surveys, grids = real_loading.surveys_and_grids(init_state=state)

    pvals = np.linspace(pargs.min, pargs.max, pargs.nstep)
    vparams = {}
    param = 'lEmax'
    vparams[param] = None
    vparams['lC'] = -0.9
    vparams["H0"] = 67.4
    #vparams["lEmax"] = 41.3
    vparams["gamma"] = -0.948
    vparams["alpha"] = 1.03
    vparams["sfr_n"] = 1.15
    vparams["lmean"] = 2.22
    vparams["lsigma"] = 0.57

    # DEBUGGING
    #embed(header='71 of constrain')

    final_lls = [] 
    final_dicts = []
    nterms = []  # LL term related to norm (i.e. rates)
    pvterms = []  # LL term related to norm (i.e. rates)
    pvvals = []  # 
    wzvals = []  # 
    
    lls = np.zeros_like(pvals)

    norm = True
    psnr = True
    odict = dict()

    for tt, pval in enumerate(pvals):
        vparams[param] = pval
        C, llC = it.minimise_const_only(
                    vparams,grids,surveys, Verbose=False)
        # Set lC
        vparams['lC']=C
        for igrid in grids:
            igrid.state.FRBdemo.lC = C

        # LL
        ll=0.
        longlistsum=np.array([0.,0.,0.,0.])
        alistsum=np.array([0.,0.,0.])
        for j,s in enumerate(surveys):
            # Update
            grids[j].update(vparams)
            if s.nD==1:
                lls[j],alist,expected,longlist = it.calc_likelihoods_1D(
                    grids[j],s,norm=norm,psnr=psnr,dolist=5)
            elif s.nD==2:
                lls[j],alist,expected,longlist = it.calc_likelihoods_2D(
                    grids[j],s,norm=norm,psnr=psnr,dolist=5)
            elif s.nD==3:
                # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                llsum1,alist1,expected1,longlist1 = it.calc_likelihoods_1D(
                    grids[j],s,norm=norm,psnr=psnr,dolist=5)
                llsum2,alist2,expected2, longlist2 = it.calc_likelihoods_2D(
                    grids[j],s,norm=norm,psnr=psnr,dolist=5,Pn=False)
                lls[j] = llsum1+llsum2
                # adds log-likelihoods for psnrs, pzdm, pn
                # however, one of these Pn *must* be zero by setting Pn=False
                alist = [alist1[0]+alist2[0], alist1[1]+alist2[1], alist1[2]+alist2[2]] #messy!
                expected = expected1 #expected number of FRBs ignores how many are localsied
                longlist = [longlist1[0]+longlist2[0], longlist1[1]+longlist2[1], 
                            longlist1[2]+longlist2[2],
                            longlist1[3]+longlist2[3]] #messy!
            else:
                raise ValueError("Unknown code ",s.nD," for dimensions of survey")
            # these are slow operations but negligible in the grand scheme of things
            
            # accumulate the 'alist' of pn,s,zdm and 'long list' of pzdm factors over multiple surveys
            longlistsum += np.array(longlist)
            alistsum += np.array(alist)
            
            # save information for individual surveys
            odict['lls'+str(j)] = lls[j]
            odict['P_zDM'+str(j)] = alist[0]
            odict['P_n'+str(j)] = alist[1]
            odict['P_s'+str(j)] = alist[2]
            odict['N'+str(j)] = expected
        
                    # save accumulated information
        ll=np.sum(lls)
        odict['lls'] = ll
        odict['P_zDM'] = alistsum[0]
        odict['P_n'] = alistsum[1]
        odict['P_s'] = alistsum[2]
        
        # More!!
        odict['p_zgDM'] = longlistsum[0]
        odict['p_DM'] = longlistsum[1]
        odict['p_DMgz'] = longlistsum[2]
        odict['p_z'] = longlistsum[3]

        # Hold
        final_lls.append(ll)
        final_dicts.append(odict)
        print(f'{param}: pval={pval}, C={C}, lltot={ll}')

    # Write
    tdf = pandas.DataFrame(final_dicts)
    tdf.to_csv(pargs.opfile.replace('.png','.csv'), index=False)
    
    # Max
    imx = np.nanargmax(final_lls)
    print(f"Max LL at {param}={pvals[imx]}")

    # Plot
    plt.clf()
    ax = plt.gca()
    ax.plot(pvals, final_lls, 'o')
    # Nan
    bad = np.isnan(lls)
    nbad = np.sum(bad)
    if nbad > 0:
        ax.plot(pvals[bad], [np.nanmin(final_lls)]*nbad, 'x', color='r')
    ax.set_xlabel(param)
    ax.set_ylabel('LL')
    # Max
    ax.axvline(pvals[imx], color='g', ls='--', label=f'max={pvals[imx]}')
    ax.legend()
    # Save?
    if pargs.opfile is not None:
        plt.savefig(pargs.opfile, dpi=300)
        print(f"Wrote: {pargs.opfile}")
    else:
        plt.show()
    plt.close()


# command-line arguments here
parser = argparse.ArgumentParser()
parser.add_argument('min',type=float,help="minimum value")
parser.add_argument('max',type=float,help="maximum value")
parser.add_argument('--nstep',type=int,default=10,required=False,help="number of steps")
parser.add_argument('-o','--opfile',type=str,required=False,help="Output file for the data")
parser.add_argument('--lum_func',type=int,default=0, required=False,help="Luminosity function (0=power-law, 1=gamma)")
pargs = parser.parse_args()


main(pargs)

'''
python py/constrain_Emax.py 40.5 43.5 --nstep 30  -o James2022_Emax.png
#
'''