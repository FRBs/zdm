""" JXP VERSION """

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

    ############## Initialise parameters ##############
    state = parameters.State()

    # Variable parameters
    vparams = {}
    vparams['cosmo'] = {}
    vparams['cosmo']['H0'] = 67.74
    vparams['cosmo']['Omega_lambda'] = 0.685
    vparams['cosmo']['Omega_m'] = 0.315
    vparams['cosmo']['Omega_b'] = 0.044
    
    #cos.set_cosmology(Omega_m=1.2) setup for cosmology
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
    names = ['CRAFT/FE', 'CRAFT/ICS', 'CRAFT/ICS892', 'PKS/Mb']
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
        
        #psetmins,psetmaxes,nvals=misc_functions.process_pfile(Cube[2])
        vparam_dict=misc_functions.process_jfile(Cube[2])
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
        it.cube_likelihoods(grids,surveys, vparam_dict,
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