
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters

def test_make_grids():
    pass

    ############## Initialise cosmology ##############
    cos.init_dist_measures()
    
    # get the grid of p(DM|z). See function for default values.
    # set new to False once this is already initialised
    zDMgrid, zvals,dmvals,H0 = misc_functions.get_zdm_grid(
        new=True, plot=False, method='analytic')
    # NOTE: if this is new, we also need new surveys and grids!
    
    # constants of beam method
    thresh=0
    method=2
    
    
    # sets which kind of source evolution function is being used
    source_evolution=0 # SFR^n scaling
    #source_evolution=1 # (1+z)^(2.7n) scaling
    
    
    # sets the nature of scaling with the 'spectral index' alpha
    #alpha_method=0 # spectral index interpretation: includes k-correction. Slower to update
    alpha_method=1 # rate interpretation: extra factor of (1+z)^alpha in source evolution
    
    ############## Initialise surveys ##############
    
    # constants of intrinsic width distribution
    Wlogmean=1.70267
    Wlogsigma=0.899148
    DMhalo=50
    
    #These surveys combine time-normalised and time-unnormalised samples 
    NewSurveys=True
    #sprefix='Full' # more detailed estimates. Takes more space and time
    sprefix='Std' # faster - fine for max likelihood calculations, not as pretty
    