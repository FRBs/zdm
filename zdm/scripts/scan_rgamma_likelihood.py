'''
Illustrates how to calculate likelihoods for FGRB surveys with repeaters (i.e., CHIME),
and scans a likelihood through Rgamma

This simply re-initialises the surveys - it does not test the update method

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
    Main program to evaluate log0-likelihoods and predictions for
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
    
    #name="CRAFT_ICS_1300"
    #ss,gs = loading.surveys_and_grids(survey_names=[name], init_state=state, rand_DMG=False, repeaters=False)
    #print("Number of observed FRBs ",ss[0].NFRB)
    #print("Number predicted ",np.sum(gs[0].rates) * 10**gs[0].state.FRBdemo.lC * ss[0].TOBS)
    #exit()
    
    
    # defines CHIME grids to load
    NDECBINS=6
    names=[]
    for i in np.arange(NDECBINS):
        name="CHIME_decbin_"+str(i)+"_of_6"
        names.append(name)
    survey_dir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/CHIME/')
    
    # we now loop through Rgamma
    Rgammas = np.linspace(-3.01,-1.01,11)
    Rgammaplot = np.linspace(-3,-1,11)
    llsums=np.zeros([11])
    for i,Rgamma in enumerate(Rgammas):
        state.rep.Rgamma = Rgamma
        
        # creates new grids - we do not update for now to be safe
        ss,gs = loading.surveys_and_grids(survey_names=names, init_state=state, rand_DMG=False,sdir = survey_dir, repeaters=True)
        
        llsum = get_total_likelihoods(gs,ss,printit=False)
        
        llsums[i] = llsum
        
        # prints out expected values of things
        #print_grid_vals(gs,ss)
        
        # calculates the likelihood over all grids
    
    # we now create a slice through Rmax
    
    plt.figure()
    plt.plot(Rgammaplot,llsums)
    plt.xlabel("$R_{\\gamma}$")
    plt.ylabel("Log likelihood")
    plt.tight_layout()
    plt.savefig("rgamma_likelihood.pdf")
    plt.close()
    
    #s = ss[0]
    #g = gs[0]
    #
    #iteration.get_log_likelihood(g,s)

def get_total_likelihoods(gs,ss,printit=False):
    '''
    Gets the likelihood of getting these results
    '''
    llsum = 0
    for i,g in enumerate(gs):
        s=ss[i]
        ll=iteration.get_log_likelihood(g, s)
        if printit:
            print("For decbin ",i," the likelihood is ",ll)
        llsum += ll
        print("llsum is ",llsum)
    return llsum
        
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
        print("   Nrep biursts ",Nbursts*g.Rc)
    
   
main()
