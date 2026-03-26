""" 
This script plots an example of an MC calculation for specified Rmax, Rgamma

"""
import os
from pkg_resources import resource_filename
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.MC_sample import loading
from zdm import io
from zdm import repeat_grid as rep

import utilities as ute
import states as st

import pickle
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp
from scipy.stats import poisson

import matplotlib
import time
from zdm import beams
beams.beams_path = '/Users/cjames/CRAFT/FRB_library/Git/H0paper/papers/Repeaters/BeamData/'

Planck_H0=67.4

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(Nbin=6):
    
    # gets the possible states for evaluation
    states,names=st.get_states()
    state = states[0]
    
    
    ############## loads CHIME surveys and grids #############
    
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/'+bdir)
    bounds = np.load(bdir+'bounds.npy')
    solids = np.load(bdir+'solids.npy')
    
    ####### initialises basic surveys and grids #########
    ss=[]
    gs=[]
    nrs=[]
    nss=[]
    irs=[]
    iss=[]
    NR=0
    NS=0
    # we initialise surveys and grids
    Cnreps = np.array([])
    for ibin in np.arange(Nbin):
        
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
            #with open(savefile, 'wb') as output:
            #    pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)
            #    pickle.dump(g, output, pickle.HIGHEST_PROTOCOL)
        
        ss.append(s)
        gs.append(g)
        
        ir = np.where(s.frbs['NREP']>1)[0]
        nr=len(ir)
        irs.append(ir)
        nreps = s.frbs['NREP'][ir]
        Cnreps = np.concatenate((Cnreps,nreps))
        i_s = np.where(s.frbs['NREP']==1)[0]
        ns=len(i_s)
        iss.append(i_s)
        
        NR += nr
        NS += ns
        
        nrs.append(nr)
        nss.append(ns)
    
    ############## Set up results grid ##############
    
    # we now have an initial estmate for Rstar. From hereon, we make a grid in Rmin, Rmax, and
    # optimise gamma for each one
    
    Rmin = 1e-2
    Rmax = 100
    Rgamma = -2.1
    
    perform_mc_simulation(ss,gs,Rmin,Rmax,Rgamma,F=0.35)
      
    ####### now inside repeater loop ###### - new function to loop over rep properties
def perform_mc_simulation(ss,gs,Rmin,Rmax,Rgamma,Nbin=6,F=1):
    """
    """
    
    numbers = np.array([])
    irzs=np.array([])
    irdms=np.array([])
    for ibin in np.arange(Nbin):
        # calculates repetition info
        g=gs[ibin]
        s=ss[ibin]
        g.state.rep.Rmin = Rmin
        g.state.rep.Rmax = Rmax
        g.state.rep.Rgamma = Rgamma
        
        # calculate both exact and MC grids
        rg = rep.repeat_Grid(g,Tfield=s.TOBS*F,Nfields=1,opdir=None,bmethod=2,\
            Exact=True,MC=1)
        if ibin ==0:
            all_reps = rg.exact_reps
            all_s = rg.exact_singles
        else:
            all_reps += rg.exact_reps
            all_s += rg.exact_singles
        
        irz,irdm = np.where(rg.MC_reps > 0)
        
        irzs = np.concatenate((irzs,irz))
        irdms = np.concatenate((irdms,irdm))
    
    NS = np.sum(all_s)
    NR = np.sum(all_reps)
    NMC = irzs.size
    print("We have estimated ",NS,NR,NMC," singles, reps, MC reps")
    
    irzs = irzs.astype('int')
    irdms = irdms.astype('int')
    g=gs[0]
    zvals=g.zvals
    dmvals = g.dmvals
    rzs = zvals[irzs]
    rdms = dmvals[irdms]
    
    misc_functions.plot_grid_2(all_reps,zvals,dmvals,FRBZ=rzs,FRBDM=rdms,\
        name="TEST/example_mc.pdf",norm=3,log=True,\
        label='$\\log_{10} p_{\\rm rep}({\\rm DM}_{\\rm EG},z)$  [a.u.]',\
        project=True,Aconts=[0.01,0.1,0.5],zmax=1,DMmax=3000)
    
    return


def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            init_state=None,
            state_dict=None, iFRB:int=0,
               alpha_method=1, NFRB:int=100, 
               lum_func:int=2,sdir=None):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        init_state (State, optional):
            Initial state
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        iFRB (int, optional): Starting index for the FRBs.  Defaults to 0
        lum_func (int, optional): Flag for the luminosity function. 
            0=power-law, 1=gamma.  Defaults to 0.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters

    Raises:
        IOError: [description]

    Returns:
        tuple: Survey, Grid objects
    """
    
    # Init state
    if init_state is None:
        state = loading.set_state(alpha_method=alpha_method)
        # Addiitonal updates
        if state_dict is None:
            state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
            state.energy.luminosity_function = lum_func
        state.update_param_dict(state_dict)
    else:
        state = init_state
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'),
        zlog=False,nz=500)

    ############## Initialise surveys ##############
    if sdir is not None:
        print("Searching for survey in directory ",sdir)
    else:
        sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    
    
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, Nbeams=5,
                                 iFRB=iFRB)
    
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]


    
main()
