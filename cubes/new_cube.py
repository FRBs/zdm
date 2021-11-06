""" JXP VERSION """

'''
Questions for Clancy:

1. lC vs. C
2. Order of update_grid()
3. Survey is unique to a grid, right?
4. Order of grid parameters
5. Why 90s per iteration?
6. Is it ok to get a NAN??   And why is it ignored??
7. https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#faster-integration-using-low-level-callback-functions
'''

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
import os
import matplotlib

from zdm import survey
from zdm import parameters
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import pcosmic
from zdm import iteration as it
from zdm import io

from IPython import embed

import pickle

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

def main(Cube):
    
    ############## Initialise cosmology ##############
    # Location for maximisation output
    outdir='Cube/'

    #psetmins,psetmaxes,nvals=misc_functions.process_pfile(Cube[2])
    input_dict=io.process_jfile(Cube[2])
    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    ############## Initialise parameters ##############
    state = parameters.State()
    state.update_param_dict(state_dict)

    # alpha-method
    state.FRBdemo.alpha_method = 1

    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    #parser.add_argument(", help
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir='../zdm/GridData')
    # NOTE: if this is new, we also need new surveys and grids!
    
    ############## Initialise surveys ##############
    
    '''
    # constants of beam method
    thresh=0
    method=2
    
    # constants of intrinsic width distribution
    Wlogmean=1.70267
    Wlogsigma=0.899148
    DMhalo=50
    
    NewSurveys=False
    
    Wbins=5
    Wscale=3.5
    Nbeams=[5,5,10]
    '''
    prefix='Cube'
    
    surveys = []
    #names = ['CRAFT/FE', 'CRAFT/ICS', 'CRAFT/ICS892', 'PKS/Mb']
    names = ['CRAFT/FE', 'CRAFT/ICS', 'PKS/Mb'] # Match x_cube.py
    #names = ['CRAFT/FE'] # For debugging
    for survey_name in names:
        surveys.append(survey.load_survey(survey_name, state, dmvals))
    '''
    # Five surveys: we need to distinguish between those with and without a time normalisation
    if NewSurveys:
        # contains both normalised and unnormalised Tobs FRBs
        FE1=survey.survey()
        FE1.process_survey_file('Surveys/CRAFT_class_I_and_II.dat')
        FE1.init_DMEG(DMhalo)
        FE1.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
        pwidths,pprobs=survey.make_widths(FE1,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        efficiencies=FE1.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        # load ICS data
        ICS=survey.survey()
        ICS.process_survey_file('Surveys/CRAFT_ICS.dat')
        ICS.init_DMEG(DMhalo)
        ICS.init_beam(nbins=Nbeams[1],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
        pwidths,pprobs=survey.make_widths(ICS,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        efficiencies=ICS.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        # load Parkes data
        p1=survey.survey()
        p1.process_survey_file('Surveys/parkes_mb_class_I_and_II.dat')
        p1.init_DMEG(DMhalo)
        p1.init_beam(nbins=Nbeams[2],method=2,plot=False,thresh=thresh) # need more bins for Parkes!
        pwidths,pprobs=survey.make_widths(p1,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        efficiencies=p1.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        names=['CRAFT/FE','CRAFT/ICS','PKS/Mb']
    
        surveys=[FE1,ICS,p1]
        with open('Pickle/'+prefix+'surveys.pkl', 'wb') as output:
            pickle.dump(surveys, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading ",'Pickle/'+prefix+'surveys.pkl')
        with open('Pickle/'+prefix+'surveys.pkl', 'rb') as infile:
            surveys=pickle.load(infile)
            names=pickle.load(infile)
            FE1,ICS,p1=surveys
    '''
    print("Initialised surveys ",names)
    
    
    '''
    # initial parameter values. SHOULD BE LOGSIGMA 0.75! (WAS 0.25!?!?!?)
    # these are meaningless btw - but the program is set up to require
    # a parameter set when first initialising grids
    lEmin=30.
    lEmax=42.
    gamma=-0.7
    alpha=1.5
    sfr_n=1.
    lmean=np.log10(50)
    lsigma=0.5
    C=0.
    pset=[lEmin,lEmax,alpha,gamma,sfr_n,lmean,lsigma,C]
    '''
    
    # generates zdm grids for initial parameter set
    # when submitting a job, make sure this is all pre-generated once
    #if state.analysis.NewGrids:
    if True:
        grids = misc_functions.initialise_grids(
            surveys,zDMgrid, zvals, dmvals, state, wdist=True)
        # Write to disk
        if not os.path.isdir('Pickle'):
            os.mkdir('Pickle')
        with open('Pickle/'+prefix+'grids.pkl', 'wb') as output:
            pickle.dump(grids, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open('Pickle/'+prefix+'grids.pkl', 'rb') as infile:
            grids=pickle.load(infile)
        #gFE1,gFE2,gICS,gp1,gp2=grids
    print("Initialised grids")
    #embed(header='175 of new_cube')
    
    if Cube is not None:
        # hard-coded cloning ability. This is now out-dated.
        #clone=[-1,0,-1,-1,3] # if > 0, will clone that grid
        # i.e. grid 1 is a clone of grid 0,
        # grid 4 is a clone of grid 3, grid 0,2,3 do not
        # clone anything. This is an approx speedup of 40%
        # This is because these grids are identical except
        # for the NFRB <=> Tobs likelihood estimate
        clone=None
        
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        run=Cube[0]
        howmany=Cube[1]
        opfile=Cube[3]
        
        # checks to see if the file is already there, and how many iterations have been performed
        starti=it.check_cube_opfile(run,howmany,opfile)
        print("starti is ",starti)
        if starti==howmany:
            print("Done everything!")
            pass
        # this takes a while...
        #it.cube_likelihoods(grids,surveys,psetmins,psetmaxes,
        #              nvals,run,howmany,opfile,
        #              starti=starti,clone=clone)
        # Check cosmology
        print(f"cosmology: {cos.cosmo}")
        #
        it.cube_likelihoods(grids,surveys, vparam_dict, cube_dict,
                      run,howmany,opfile, starti=starti,clone=clone)
        


# test for command-line arguments here
parser = argparse.ArgumentParser()
parser.add_argument('-n','--number',type=int,required=False,help="nth iteration, beginning at 0")
parser.add_argument('-m','--howmany',type=int,required=False,help="number m to iterate at once")
parser.add_argument('-p','--pfile',type=str,required=False,help="File defining parameter ranges")
parser.add_argument('-o','--opfile',type=str,required=False,help="Output file for the data")
parser.add_argument('--clobber', default=False, action='store_true',
                    help="Clobber output file?")
args = parser.parse_args()


if args.number is not None and args.howmany is not None and args.pfile is not None and args.opfile is not None:
    if args.number is None or args.howmany is None or args.pfile is None or args.opfile is None:
        print("We require some or all values of the arguments!")
        exit()
    Cube=[args.number,args.howmany,args.pfile,args.opfile]
    #mins,maxs,Ns=misc_functions.process_pfile(args.pfile)
    # Clobber?
    if args.clobber and os.path.isfile(args.opfile):
        os.remove(args.opfile)
else:
    Cube=None


main(Cube)

'''
python new_cube.py -n 1 -m 3 -p all_params.json -o tmp.out --clobber
python new_cube.py -n 1 -m 3 -p H0_params.json -o H0.out --clobber
starti is  0
cosmology: CosmoParams(H0=67.66, Omega_k=0.0, Omega_lambda=0.6888463055445441, Omega_m=0.30966, Omega_b=0.04897, Omega_b_h2=0.0224178568132, fixed_H0=67.66, fix_Omega_b_h2=True)
FIX THIS!!!!!
Starting at time  25.224615935
Testing  0  of  3  begin at  0
vparams: {'lEmin': 30.0, 'lEmax': 41.4, 'alpha': 1.0, 'gamma': -0.5, 'sfr_n': 0.0, 'lmean': 0.5, 'lsigma': 0.2, 'lC': -0.911}
survey=CRAFT_class_I_and_II, lls=47.359007883041286
/home/xavier/Projects/FRB_Software/zdm/zdm/iteration.py:723: RuntimeWarning: divide by zero encountered in log10
  longlist += np.log10(wzpsnr)
survey=CRAFT_ICS, lls=nan
'''
