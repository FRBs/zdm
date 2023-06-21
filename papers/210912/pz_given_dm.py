""" Calculate p(z|DM) for a given DM and survey
"""

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import imp
import numpy as np
import os

from zdm import iteration as it
from zdm import io
from zdm.craco import loading
from pkg_resources import resource_filename
from IPython import embed


def main(pargs):
    
    import numpy as np
    from matplotlib import pyplot as plt
    
    from frb import mw
    from frb.figures.utils import set_fontsize 

    from zdm import survey
    from zdm import parameters
    from zdm import cosmology as cos
    from zdm import misc_functions
    
    limits = (2.5, 97.5)
    
    if pargs.DM_ISM:
        DM_ISM = pargs.DM_ISM
    else:
        # Deal with coord
        from linetools import utils as ltu
        from linetools.scripts.utils import coord_arg_to_coord
        icoord = ltu.radec_to_coord(coord_arg_to_coord(pargs.coord))
        
        # NE 2001
        DM_ISM = mw.ismDM(icoord)
        DM_ISM = DM_ISM.value
        print("")
        print("-----------------------------------------------------")
        print(f"NE2001 = {DM_ISM:.2f}")
    
    
    # DM EG
    DM_EG = pargs.DM_FRB - DM_ISM - pargs.DM_HALO
    
    # Cosmology
    states=get_states(newEmax=True,width=pargs.width)
    
    #all states have identical cosmology
    cos.set_cosmology(states[0])
    cos.init_dist_measures()
    
    # save file
    all_pzgdm = np.zeros([len(states),500])
    all_pzgdm_SNR = np.zeros([len(states),500])
    
    # Load Survey
    
    for i,state in enumerate(states):
        sdir = os.path.join(resource_filename('zdm','data'),'Surveys')
        
        if pargs.SNR is not None:
            SNR1=pargs.SNR
            s,igrid = survey_and_grid(survey_name=pargs.survey,NFRB=None,sdir=sdir,
                init_state=state,SNR=SNR1)
            SNR2=SNR1+1.
            s2,igrid2 = survey_and_grid(survey_name=pargs.survey,NFRB=None,sdir=sdir,
                init_state=state,SNR=SNR2)
            PDM_z = igrid.rates - igrid2.rates # z, DM
            
        else:
            s,igrid = survey_and_grid(survey_name=pargs.survey,NFRB=None,sdir=sdir,init_state=state)
            PDM_z = igrid.rates # z, DM
        
        
        
        # get the grid of p(DM|z)
        dmvals = igrid.dmvals
        zvals = igrid.zvals
        
        
        # Fuss
        iDM = np.argmin(np.abs(dmvals - DM_EG))
        PzDM = PDM_z[:, iDM] / np.sum(PDM_z[:, iDM]) / (zvals[1]-zvals[0])
        
        all_pzgdm[i,:] = PzDM
        
        if pargs.redshift is not None:
            cPzDM = np.cumsum(PzDM)
            cPzDM /= cPzDM[-1]
            iz = np.argmin(np.abs(zvals - pargs.redshift))
            print("set ",i," cumulative p is ",cPzDM[iz])
        
        
    np.savez(pargs.output+".npz",zvals=zvals,all_pzgdm = all_pzgdm)
    
    
def james_fit():
    """
    Returns best-fit parameters from James et al 2022 (Hubble paper)
    NOT updated with larger Emax from Science paper
    """
    
    pset={}
    pset["lEmax"] = 41.63
    pset["alpha"] = -1.03
    pset["gamma"] = -0.948
    pset["sfr_n"] = 1.15
    pset["lmean"] = 2.22
    pset["lsigma"] = 0.57
    pset["lC"] = 1.963
    
    return pset

def parse_args(options=None):
    # test for command-line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="Coordinates, e.g. J081240.7+320809 or 122.223,-23.2322 or 07:45:00.47,34:17:31.1 or FRB name (FRB180924)")
    parser.add_argument("-i", '--DM_ISM', type=float, help="Estimate for Galactic ISM contribution")
    parser.add_argument('-d','--DM_FRB', type=float, help="FRB DM (pc/cm^3)")
    parser.add_argument('-s','--survey',type=str, default='CRAFT/ICS', help="Name of survey [CRAFT/ICS, PKS/Mb]")
    parser.add_argument('-H',"--DM_HALO", type=float, default=40., help="Assumed DM contribution from the Milky Way Halo (ISM is calculated from NE2001). Default = 40")
    parser.add_argument('-o','--output',type=str, default=None, help="Name of output image file")
    parser.add_argument('-S','--SNR',type=float, default=None, help="Signal-to-noise of burst")
    parser.add_argument('-z','--redshift',type=float, default=None, help="Actual redshift of the FRB")
    parser.add_argument('-w','--width',type=float, default=None, help="Total width of FRB, including scattering, excluding DM smearing")
    
    args = parser.parse_args()
    return args



def get_states(newEmax=False,width=None):  
    """
    Gets the states corresponding to plausible fits to single CHIME data
    """
    psets=read_extremes()
    psets.insert(0,james_fit())
    
    # loop over chime-compatible state
    for i,pset in enumerate(psets):
        
        state=set_state(pset,chime_response=False,newEmax=newEmax)
        
        if width is not None:
            state.width.Wbins=1
            state.Wlogmean = np.log(width)
            state.Wmethod = 0 # sets Wbins to unity anyway, uses only the mean above
        if i==0:
            states=[state]
        else:
            states.append(state)
    return states


def read_extremes(infile='planck_extremes.dat',H0=67.4):
    """
    reads in extremes of parameters from a get_extremes_from_cube
    """
    f = open(infile)
    
    sets=[]
    
    for pset in np.arange(6):
        # reads the 'getting' line
        line=f.readline()
        
        pdict={}
        # gets parameter values
        for i in np.arange(7):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        pdict["alpha"] = -pdict["alpha"] # alpha is reversed!
        sets.append(pdict)
        
        pdict={}
        # gets parameter values
        for i in np.arange(7):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        pdict["alpha"] = -pdict["alpha"] # alpha is reversed!
        sets.append(pdict)
    return sets


def james_fit():
    """
    Returns best-fit parameters from James et al 2022 (Hubble paper)
    NOT updated with larger Emax from Science paper
    """
    
    pset={}
    pset["lEmax"] = 41.63
    pset["alpha"] = -1.03
    pset["gamma"] = -0.948
    pset["sfr_n"] = 1.15
    pset["lmean"] = 2.22
    pset["lsigma"] = 0.57
    pset["lC"] = 1.963
    
    return pset




def set_state(pset,chime_response=True,newEmax=False):
    """
    Sets the state parameters
    """
    
    state = loading.set_state(alpha_method=1)
    state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
    state.energy.luminosity_function = 2 # this is Schechter
    state.update_param_dict(state_dict)
    
    
    
    # updates to most recent best-fit values
    state.cosmo.H0 = 67.4
    
    if newEmax:
        oldEmax = 41.62
        diff = pset['lEmax'] - oldEmax
        state.energy.lEmax = 41.7 + diff
    else:
        state.energy.lEmax = pset['lEmax']
    
    state.energy.gamma = pset['gamma']
    state.energy.alpha = pset['alpha']
    state.FRBdemo.sfr_n = pset['sfr_n']
    state.host.lsigma = pset['lsigma']
    state.host.lmean = pset['lmean']
    state.FRBdemo.lC = pset['lC']
    
    return state


def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            init_state=None,
            state_dict=None, iFRB:int=0,
               alpha_method=1, NFRB:int=100, 
               lum_func:int=0,sdir=None,SNR=None):
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
    from zdm import cosmology as cos
    from zdm import misc_functions
    from zdm import survey
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
    
    if SNR is not None:
        increase = SNR/isurvey.meta['SNRTHRESH']
        isurvey.meta['THRESH'] = isurvey.meta['THRESH']*increase
        print("Increasing by ",increase)
    
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]


def run():
    pargs = parse_args()
    main(pargs)

run()
'''
# Test
python py/pz_given_dm.py -n 1 -m 100 -o tmp.out --clobber

# 
python py/pz_given_dm.py -n 1 -m 250 -o Cubes/craco_H0_Emax_cube0.out --clobber

'''
