
"""
This generates 1000 MC samples from CRACO.

It's written slightly differently from the standard mc_statistics.py
to make it easier to iterate through different surveys, i.e.
it processes the surveys one at a time.

The main output is the print to screen. This gives the (default 1000)
FRBs and measured redshifts. This has been copied to
CRAFT_CRACO_MC_alpha1_1000.dat (power law)
CRAFT_CRACO_MC_alpha1_gamma_1000.dat (gamma function)

It then also removes the redshift info for all FRBs with DMEG > 1000
and sets z=-1 (z < 0 is code for "unlocalised"), and re-prints the
same list. See
CRAFT_CRACO_MC_alpha1_1000.dat
CRAFT_CRACO_MC_alpha1_gamma_1000.dat

Finally, it also takes one third of all FRBs (irrespective of DM)
and *also* sets z=-1 to simulate FRBs that have not yet been
localised. See
CRAFT_CRACO_MC_alpha1_1000_missing.dat
CRAFT_CRACO_MC_alpha1_gamma_1000_missing.dat


"""

from zdm.MC_sample import loading
from pkg_resources import resource_filename
import os
import copy
import pickle
import sys
import scipy as sp
import numpy as np
import time
import argparse

from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import iteration as it
from zdm import beams
from zdm import misc_functions

from IPython import embed

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib

from zdm import io

from matplotlib.ticker import NullFormatter

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(N=100,plots=False):
    "created N*survey.NFRBs mock FRBs for all surveys"
    "Input N : creates N*survey.NFRBs"
    "Output : Sample(list) and Surveys(list)"
    
    ############## Load up ##############
    input_dict=io.process_jfile('../../papers/H0_I/Analysis/Cubes/craco_H0_Emax_cube.json')

    # Deconstruct the input_dict
    state, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    ######## choose which surveys to do an MC for #######
    
    # choose which survey to create an MC for
    names = ['CRAFT/FE', 'CRAFT/ICS', 'CRAFT/ICS892', 'PKS/Mb','CRAFT_CRACO_MC_alpha1_gamma_1000']
    dirnames=['ASKAP_FE','ASKAP_ICS','ASKAP_ICS892','Parkes_Mb','CRACO']
    # select which to perform an MC for
    which=1
    N=10000
    
    do_mc(state,names[which],N,which)
    
def do_mc(state_dict, survey_name, Nsamples, which_survey,plots=False):
 
    ############## Initialise survey and grid ##############
    s,g = loading.survey_and_grid(
        state_dict=state_dict,
        survey_name=survey_name, NFRB=9,sdir="/Users/cjames/CRAFT/FRB_library/Git/H0paper/zdm/data/Surveys/")
    
    ############## Initialise surveys ##############
    #saves everything in this directory
    outdir='SNRbias/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,zmax=2,DMmax=2000,
        name=os.path.join(outdir,'test_ICS.pdf'),
        norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5])
    
    samples = []
    
    name = str(which_survey)
    savefile=outdir+'mc_sample_'+name+'_alpha_'+str(g.state.FRBdemo.alpha_method)+str(Nsamples)+'.npy'
    
    try:
        sample=np.load(savefile)
        print("Loading ",sample.shape[0]," samples from file ",savefile)
    except:
        print("Generating ",Nsamples," samples from survey/grid ",which_survey)
        sample=g.GenMCSample(Nsamples)
        sample=np.array(sample)
        np.save(savefile,sample)
    print("Shape of samples is ",samples)
    samples.append(sample)
    
    print("########### Set 1: complete ########")
    for j in np.arange(Nsamples):
        DMG=35
        DMEG=sample[j,1]
        DMtot=DMEG+DMG+g.state.MW.DMhalo
        SNRTHRESH=9
        SNR=SNRTHRESH*sample[j,4]
        z=sample[j,0]
        w=sample[j,3]
        
        string="FRB "+str(j)+'  {:6.1f}  35   {:6.1f}  {:5.3f}   {:5.1f}  {:5.1f}'.format(DMtot,DMEG,z,SNR,w)
        #print("FRB ",i,DMtot,SNR,DMEG,w)
        print(string)
        
    print("\n\n\n\n########### Set 2: losing 50% below SNR 14 ########")
    removed=True
    for j in np.arange(Nsamples):
        DMG=35
        DMEG=sample[j,1]
        DMtot=DMEG+DMG+g.state.MW.DMhalo
        SNRTHRESH=9
        SNR=SNRTHRESH*sample[j,4]
        if SNR < 14.:
            if removed:
                removed=False
                continue
            else:
                removed=True
        z=sample[j,0]
        w=sample[j,3]
        
        string="FRB "+str(j)+'  {:6.1f}  35   {:6.1f}  {:5.3f}   {:5.1f}  {:5.1f}'.format(DMtot,DMEG,z,SNR,w)
        print(string)
    
    print("\n\n\n\n########### Set 3: SNR thresh 14 ########")
    
    for j in np.arange(Nsamples):
        DMG=35
        DMEG=sample[j,1]
        DMtot=DMEG+DMG+g.state.MW.DMhalo
        SNRTHRESH=9
        SNR=SNRTHRESH*sample[j,4]
        if SNR < 14.:
            continue
        z=sample[j,0]
        w=sample[j,3]
        
        string="FRB "+str(j)+'  {:6.1f}  35   {:6.1f}  {:5.3f}   {:5.1f}  {:5.1f}'.format(DMtot,DMEG,z,SNR,w)
        print(string)
    
main()
