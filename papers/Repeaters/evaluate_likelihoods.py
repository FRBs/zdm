""" 
This script creates example plots for a combination
of FRB surveys and repeat bursts

"""
import os
from pkg_resources import resource_filename
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io
from zdm import repeat_grid as rep
from zdm import old_working_repeat_grid as orep

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp

import matplotlib
import time
from zdm import beams
beams.beams_path = '/Users/cjames/CRAFT/FRB_library/Git/H0paper/papers/Repeaters/BeamData/'
    

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    #defines lists of repeater properties, in order Rmin,Rmax,r
    # units of Rmin and Rmax are "per day above 10^39 erg"
    #Rset={"Rmin":1e-4,"Rmax":10,"Rgamma":-2.2}
    Rset={"Rmin":3.9,"Rmax":4,"Rgamma":-1.1}
    #sets=[Rset1,Rset2]
    
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    print("sdir ",sdir)
    Nbin=6
    surveys=[]
    grids=[]
    reps=[]
    state=set_state()
    t0=time.time()
    
    for ibin in np.arange(Nbin):
        name = "CHIME_decbin"+str(ibin)
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        t1=time.time()
        # calculates repetition info
        g.state.rep.Rmin = Rset["Rmin"]
        g.state.rep.Rmax = Rset["Rmax"]
        g.state.rep.Rgamma = Rset["Rgamma"]
        
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        org = orep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        
        t2=time.time()
        surveys.append(s)
        grids.append(g)
        reps.append(rg)
        Times=s.TOBS #here, TOBS is actually soid angle in steradians, since beamfile contains time factor
        print("Took total of ",t1-t0,t2-t1," seconds for init of survey ",ibin)
        t0=t2
        plot=True
        if plot:
            # collapses CHIME dm distribution for repeaters and once-off burts
            rdm = np.sum(rg.exact_reps,axis=0)
            sdm = np.sum(rg.exact_singles,axis=0)
            rbdm = np.sum(rg.exact_rep_bursts,axis=0)
            adm = np.sum(g.rates,axis=0)*s.TOBS*10**(g.state.FRBdemo.lC)
            
            ordm = np.sum(org.exact_reps,axis=0)
            osdm = np.sum(org.exact_singles,axis=0)
            orbdm = np.sum(org.exact_rep_bursts,axis=0)
            
            #gets histogram of CHIME bursts
            nreps=s.frbs['NREP']
            ireps=np.where(nreps>1)
            isingle=np.where(nreps==1)[0]
            
            # normalises to singles
            bins=np.linspace(0,4000,21)
            db = bins[1]-bins[0]
            tot=np.sum(osdm)#*db/(g.dmvals[1]-g.dmvals[0])
            nsingles = len(isingle)
            norm = nsingles/tot*db/(g.dmvals[1]-g.dmvals[0])
            print("Norm factor is ",norm,nsingles,tot)
            
            plt.figure()
            plt.plot(g.dmvals,sdm*norm,label='singles')
            plt.plot(g.dmvals,rdm*norm,label='repeaters')
            plt.plot(g.dmvals,(rbdm+sdm)*norm,label='all from rg')
            
            plt.plot(g.dmvals,osdm*norm,label='orig singles',linestyle='--')
            plt.plot(g.dmvals,ordm*norm,label='orig repeaters',linestyle='--')
            plt.plot(g.dmvals,(orbdm+osdm)*norm,label='orig all from rg',linestyle='--')
            
            plt.plot(g.dmvals,adm*norm,label='all from grid',linestyle=':')
            
            plt.hist(s.DMEGs[isingle],bins=bins,alpha=0.5,label='CHIME once-off bursts')
            if len(ireps)>0:
                plt.hist(s.DMEGs[ireps],bins=bins,alpha=0.5,label='CHIME repeaters')
            plt.xlim(0,3000)
            plt.xlabel('${\\rm DM}_{\\rm EG}$')
            plt.ylabel('p(DM)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('DecbinFigs/bin_'+str(ibin)+'.pdf')
            plt.close()
            
            
            plt.figure()
            
            
            plt.ylim(0,5)
            plt.plot(g.dmvals,(rbdm+sdm)/adm,label='Ratio: all')
            plt.plot(g.dmvals,sdm/osdm,label='singles')
            plt.plot(g.dmvals,rdm/ordm,label='repeaters')
            plt.xlim(0,3000)
            plt.xlabel('dm')
            plt.ylabel('ratio new/old')
            plt.legend()
            plt.tight_layout()
            plt.savefig('ratio_repeat_grids.pdf')
            exit()
    
def set_state():
    """
    Sets the state parameters
    """
    
    state = loading.set_state(alpha_method=1)
    state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
    state.energy.luminosity_function = 2 # this is Schechter
    state.update_param_dict(state_dict)
    # changes the beam method to be the "exact" one, otherwise sets up for FRBs
    state.beam.Bmethod=3
    state.width.Wmethod=0
    state.width.Wbias="CHIME"
    
    # updates to most recent best-fit values
    state.cosmo.H0 = 67.4
    state.energy.lEmax = 41.63
    state.energy.gamma = -0.948
    state.energy.alpha = 1.03
    state.FRBdemo.sfr_n = 1.15
    state.host.lmean = 2.22
    state.host.lsigma = 0.57
    
    state.FRBdemo.lC = 1.963
    
    print("Making alpha crazy for testing purposes")
    state.energy.alpha = 4
    
    return state


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
