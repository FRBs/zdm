'''
File to test repetition modelling in zDM

'''
# standard python imports
import numpy as np
from astropy.cosmology import Planck18
from matplotlib import pyplot as plt

# zdm imports
from zdm import loading
from zdm import parameters
from zdm import iteration

def main():
    '''
    Main program to evaluate log-likelihoods and predictions for
    repeat grids
    '''
    
    # sets basic state and cosmology
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    param_dict={'sfr_n': 0.8808527057055584, 'alpha': 0.7895161131856694,
                'lmean': 2.1198711983468064, 'lsigma': 0.44944780033763343,
                'lEmax': 41.18671139482926, 'lEmin': 39.81049090314043,
                'gamma': -1.1558450520609953, 'H0': 54.6887137195215,
                'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -7.61}
    state.update_params(param_dict)
    print("The relevant repeater parameters are currently:")
    print("    Rmin: ",state.rep.lRmin)
    print("    Rmax: ",state.rep.lRmax)
    print("    Rgamma: ",state.rep.Rgamma)
    
    # selects one of the CHIME declination bins
    names = ["CHIME_decbin_3_of_6"]
    
    survey_dir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/CHIME/')
    ss,gs = loading.surveys_and_grids(survey_names=names, init_state=state,
                                rand_DMG=False,sdir = survey_dir, repeaters=True)
    
    print_grid_vals(gs,ss)
    
    s=ss[0]
    
    DMs=s.DMEGs[s.replist]
    Nreps = s.frbs["NREP"][s.replist]
    g=gs[0]
    cs=0.
    print("\n\nExample FRB at redshift 0.05")
    for Nreps in np.arange(2,10):
        rel_p = g.calc_exact_repeater_probability(Nreps,s.DMEGs[s.replist][0],z=0.05)
        cs += rel_p
        print("rel p is ",rel_p," cumulative is ",cs)
    
    cs=0.
    print("\n\nExample FRB with unknown redshift")
    for Nreps in np.arange(2,10):
        rel_p = g.calc_exact_repeater_probability(Nreps,s.DMEGs[s.replist][0])
        cs += rel_p
        print("rel p is ",rel_p," cumulative is ",cs)
    
def print_grid_vals(gs,ss): 
    '''
    Print relative number of bursts expected for these grids and surveys
    '''
    for i,g in enumerate(gs):
        s=ss[i]
        Ntot=np.sum(g.rates * 10**g.state.FRBdemo.lC * s.TOBS)
        Nreps=np.sum(g.exact_reps)
        Nsingle=np.sum(g.exact_singles)
        Nbursts = np.sum(g.exact_rep_bursts)
        print("CHIME decbin i predicts ")
        print("   Ntot ",Ntot)
        print("   Nreps ",Nreps*g.Rc, " true value ",s.NORM_REPS)
        print("   Nsingle ",Nsingle*g.Rc," true value ",s.NORM_SINGLES)
        print("   Nrep bursts ",Nbursts*g.Rc)
    
   
   
main()
