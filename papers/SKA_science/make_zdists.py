""" 
This script creates plots of p(z) for different SKA configs

It first loads in the simulation info from the script
"sim_SKA_configs.py", and generates plots for the best cases.

"""
import os
import emcee
from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import io
from zdm import misc_functions as mf
from zdm import grid as zdm_grid
from zdm import survey

import numpy as np
import copy
from matplotlib import pyplot as plt
from pkg_resources import resource_filename

def main():
    """
    Main program to loop over SKA frequency ranges and deployment milestones
    
    """
    
    #### Initialises general zDM stuff #####
    state = parameters.State()
    # approximate best-fit values from recent analysis
    state.set_astropy_cosmo(Planck18)
    
    zDMgrid, zvals, dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    datdir=resource_filename('zdm', 'GridData'))
    
    
    ####### sample MCMC parameter sets ######
    infile = resource_filename('zdm', 'scripts/MCMC')+"/H0_prior10.h5"
    nsets=100
    sample, params, pconfig = get_samples(infile,nsets)
    
    ####### Loop over input files #########
    # these set the frequencies in MHz and bandwidths in MHz
    names = ["SKA_mid","SKA_mid","SKA_low"]
    freqs = [865,1400,190]
    bws = [300,300,120]
    for i,tel in enumerate(["Band1", "Band2", "Low"]):
        # sets frequency and bandwidth for each instrument
        freq = freqs[i]
        bw = bws[i]
        survey_name = names[i]
        for telconfig in ["AA4","AAstar"]:
            infile = "inputs/"+tel+telconfig+"_ID_radius_AonT_FoVdeg2"
            label = tel+"_"+telconfig
            generate_sensitivity_plot(infile,state,zDMgrid, zvals, dmvals, label, freq, bw, survey_name,
                                sample, params, pconfig)
            
    
def generate_sensitivity_plot(infile,state,zDMgrid, zvals, dmvals, label, freq, bw, survey_name,
                            samples, params, pconfig,
                        opdir = "outputs/", plotdir = "plotdir/"):
    """
    generates a plot of FRB rate vs Nelements used
    
    Args:
        infile: input file from Evan Keane relating sensitivity to FOV
        
        state: zDM parameters.state class
        
        zDMgrid (np.ndarray, 2D): intrinsic grid of p(DM|z)
        
        zvals (np.ndarray): redshift values
        
        dmvals (np.ndarray): dispersion measure values
        
        label (string): string label to add to outputs
        
        freq (float): mean frequency in MHz
        
        bw (float): bandwidth in MHz
        
        sample: set of MCMC values
        
        params: parameter names for ordered values
        
        pconfig: configuration dict specifying values of other variables
    """
    
    if pconfig is not None:
        state.update_params(pconfig)
    
    data=np.loadtxt(infile,dtype=str)
    radius = data[:,1].astype(float)
    sense_m2K = data[:,2].astype(float)
    fov = data[:,3].astype(float)
    
    Ngrids = radius.size
    
    # defines name of output files
    opfile1 = opdir+label+"_N.npy"
    opfile2 = opdir+label+"_pdm.npy"
    opfile3 = opdir+label+"_pz.npy"
    opfile4 = opdir+label+"_Nz.npy"
    opfile5 = opdir+label+"_Tobs.npy"
    opfile6 = opdir+label+"_thresh.npy"
    oldNs = np.load(opfile1)
    #pdms = np.load(opfile2)
    #pzs = np.load(opfile3)
    #Nizs = np.load(opfile4)
    TOBS = np.load(opfile5)
    thresh_Jyms = np.load(opfile6)
    
    # finds the best
    survey_name='SKA_mid'
    ibest = np.argmax(oldNs)
    survey_dict = {"THRESH": thresh_Jyms[ibest], "TOBS": TOBS[ibest], "FBAR": freq, "BW": bw}
    
    
    Nsamples = samples.shape[0]
    Nz = zvals.size
    Ndm = dmvals.size
    
    pdms = np.zeros([Nsamples,Ndm])
    pzs = np.zeros([Nsamples,Nz])
    Ns = np.zeros([Nsamples])
    
    opdir = "sys_outputs/"
    opfile7 = opdir+label+"_sys_N.npy"
    opfile8 = opdir+label+"_sys_pz.npy"
    opfile9 = opdir+label+"_sys_pdm.npy"
    
    
    
    np.save("sysplotdir/zvals.npy",zvals)
    np.save("sysplotdir/dmvals.npy",dmvals)
    
    #load=True
    load=False
    
    verbose=True
    for i in range(samples.shape[0]):
        if load:
            continue
        
        if verbose:
            print("Sampling parameter set ",i,". Params: ")
        vals = samples[i,:]
        
        dict = {}
        for j in range(len(vals)):
            dict[params[j]] = vals[j]
            if verbose:
                print("     ",params[j],": ",vals[j])
        
        state.update_params(dict)
        
        mask = pcosmic.get_dm_mask(dmvals, (state.host.lmean, state.host.lsigma), zvals, plot=False)
        
        # normalise number of FRBs to the CRAFT Fly's Eye survey
        s = survey.load_survey("CRAFT_class_I_and_II", state, dmvals, zvals=zvals)
        g = zdm_grid.Grid(s, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist=True)
        # we expect np.sum(g.rates)*s.TOBS * C = s.NORM_FRB
        norm = s.NORM_FRB/(s.TOBS*np.sum(g.rates))
        
        print("norm is ",norm,s.NORM_FRB,s.TOBS)
        s = survey.load_survey(survey_name, state, dmvals, zvals=zvals, survey_dict=survey_dict)
        
        g = zdm_grid.Grid(s, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist=True)
        scale = TOBS[ibest] * norm
        
        if verbose:
            print("Finished iteration ",i," norm is ",norm)
        
        pz = np.sum(g.rates,axis=1)*scale
        pdm = np.sum(g.rates,axis=0)*scale
        N = np.sum(pdm)
        
        Ns[i] = N
        pzs[i,:] = pz
        pdms[i,:] = pdm
    
    if load:
        Ns = np.load(opfile7)
        pzs = np.load(opfile8)
        pdms = np.load(opfile9)
    
    else:
        np.save(opfile7,Ns)
        np.save(opfile8,pzs)
        np.save(opfile9,pdms)
    
    print("######### Finished ",label," #########")
    plotdir = "sysplotdir/"
    make_pz_plots(zvals,pzs,plotdir+label)


def make_pz_plots(zvals,pzs,label):
    """
    
    """
    
    Nparams,NZ = pzs.shape
    
    scale = 1./(zvals[1]-zvals[0])
    
    mean = np.sum(pzs,axis=0)/Nparams
    
    # make un-normalised plots
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("N(z)")
    plt.xlim(0,5)
    
    for i in np.arange(Nparams):
        plt.plot(zvals,pzs[i,:],color="grey",linestyle="-")
    
    plt.plot(zvals,mean,color="black",linestyle="-",linewidth=2)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(label+"_pz.png")
    plt.close()
    
    
    

def get_samples(infile,nsets):
    # Read in the MCMC results without the burnin
    #infile = os.path.join(args.directory, args.infile)
    reader = emcee.backends.HDFBackend(infile)
    sample = reader.get_chain(discard=500, flat=True)

    # Thin the results
    step = len(sample) // nsets
    
    sample = sample[::step,:]

    # Get the corresponding parameters
    # If there is a corresponding .out file, it will contain all the necessary information that was used during that run,
    # otherwise parameters must be specified manually
    if os.path.exists(infile + '.out'):
        with open(infile + '.out', 'r') as f:
            # Get configuration
            line = f.readline()
            
            while not line.startswith('Config') and line:
                line = f.readline()
            if not line:
                raise ValueError("No 'Config' line found in the .out file.")
            config = json.loads(line[9:].replace("'", '"'))

            # Read down to parameter lines
            while ('priors' not in line) and line:
                line = f.readline()
            
            # Read parameters
            params = []
            while ('priors' in line) and line:
                vals = line.split()
                params.append(vals[0])
                line = f.readline()

    # If there is no .out file, then the parameters must be specified manually
    else:
        params = ["sfr_n", "alpha", "lmean", "lsigma", "lEmax", "lEmin", "gamma", "H0"]
        config = None
        
    return sample, params, config

main()
