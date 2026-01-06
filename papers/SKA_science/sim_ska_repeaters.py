""" 
This script creates plots of p(z) and p(dm) for different SKA configs

"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import importlib.resources as resources
from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import loading
from zdm import io
from zdm import misc_functions as mf
from zdm import grid as zdm_grid
from zdm import figures
from zdm import states
import copy

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    """
    Plots outputs of simulations
    
    """
    
    ####### Loop over input files #########
    # these set the frequencies in MHz and bandwidths in MHz
    names = ["SKA_mid_band1_AA4","SKA_mid_band2_AA4"] #,"SKA_mid_band5a_AA4","SKA_mid_band5b_AA4"]
    # don't have Aeff/Tsys for these ones. The above are based on how far out we beamform
    Tobs = np.array([100.]) #[1.,10.,100.,1000.]) # hr per pointing
    Tobs /= 24. # converts to day
    
    outdir = "FRBAstrophysics/"
    
    # ensures we keep the same state
    case = "b"
    state = states.load_state("HoffmannEmin25",scat="updated",rep=case)
    
    ss,gs = loading.surveys_and_grids(init_state=state,survey_names=["CRAFT_class_I_and_II"],repeaters=False)
    s=ss[0]
    g=gs[0]
    
    # we expect np.sum(g.rates)*s.TOBS * C = s.NORM_FRB
    #norm = s.NORM_FRB/(s.TOBS*np.sum(g.rates))
    #logC = np.log10(norm)
    #print("logC is ",logC)
    
    for i,name in enumerate(names):
        # sets frequency and bandwidth for each instrument
        zvals,plow,pmid,phigh = make_plots(name,outdir,state,Tobs,tag=name+"_case_"+case)
        
    
def make_plots(survey_name,outdir,state,Tobss,tag=""):
    """
    
    Args:
        label (string): string label identifying the band and config
            of the SKA data to load, and tag to apply to the
            output files
        
    """
    state = parameters.State()
    survey_dict={}
    survey_dict["Telescope"]={}
    survey_dict["TOBS"] = 365 # This is one year
    survey_dict["NORM_FRB"] = 0
    survey_dict["NORM_REPS"] = 0 # fake
    survey_dict["NORM_SINGLES"] = 0 #fake
    sdir = resources.files('zdm').joinpath('data/Surveys/SKA/')
    
    
    for i,Tobs in enumerate(Tobss):
        
        survey_dict["TFIELD"] = Tobs #OK, this is time per field
        ss,gs = loading.surveys_and_grids(init_state=state,survey_names=[survey_name],repeaters=True,
                    survey_dict=survey_dict,sdir=sdir)
        s=ss[0]
        g=gs[0]
        figures.plot_repeaters_zdist(g,prefix=tag)
        exit()
    
    pzs = np.load(datadir+label+"_sys_pz.npy")
    pdms = np.load(datadir+label+"_sys_pdm.npy")
    
    plow,pmid,phigh = make_pz_plots(zvals,pzs,plotdir+label)
    make_pdm_plots(dmvals,pdms,plotdir+label)

    return zvals,plow,pmid,phigh
    
def make_pz_plots(zvals,pzs,label):
    """
    Make plots of p(z) for each systematic simulation
    """
    
    Nparams,NZ = pzs.shape
    
    # this scales from the "per z bin" to "per z",
    # i.e. to make the units N per year per dz
    scale = 1./(zvals[1]-zvals[0])
    
    mean = np.sum(pzs,axis=0)/Nparams
    
    # total estimates
    Ntots = np.sum(pzs,axis=1)
    Nordered = np.sort(Ntots)
    Nbar = np.sum(Ntots)/Nparams
    sigma1 = Nordered[15]
    sigma2 = Nordered[83]
    print("Range for Ntotal is ",sigma1-Nbar,Nbar,sigma2-Nbar)
    
    # constructs intervals - does this on a per-z basis
    # first sorts over the axis of different simulations
    zordered = np.sort(pzs,axis=0)
    pzlow = zordered[15,:]
    pzhigh = zordered[83,:]
    
    # make un-normalised plots
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("N(z) per year")
    plt.xlim(0,5)
    
    themax = np.max(pzs)
    plt.ylim(0,themax*scale)
    for i in np.arange(Nparams):
        plt.plot(zvals,pzs[i,:]*scale,color="grey",linestyle="-")
    
    plt.plot(zvals,mean*scale,color="black",linestyle="-",linewidth=2,label="Simulation mean")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(label+"_pz.png")
    plt.close()
    
    # prints some summary statistics in z-space
    # calculates mean z
    zbar = zvals * mean / np.sum(mean)
    last=0.
    tot = np.sum(mean)
    for z in np.arange(5)+1:
        OK = np.where(zvals < z)
        Nthis = np.sum(mean[OK])
        N = Nthis -last
        print("FRBs from ",z-1," to ",z,": ",N/tot," %")
        last = Nthis
    
    return pzlow*scale,mean*scale,pzhigh*scale
    
def make_pdm_plots(dmvals,pdms,label):
    """
    Make plots of p(DM) for each systematic simulation
    """
    
    Nparams,NDM = pdms.shape
    
    # this scales from the "per z bin" to "per z",
    # i.e. to make the units N per year per dz
    scale = 1./(dmvals[1]-dmvals[0])
    
    mean = np.sum(pdms,axis=0)/Nparams
    
    # make un-normalised plots
    plt.figure()
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("N(DM) per year")
    plt.xlim(0,5000)
    
    themax = np.max(pdms)
    plt.ylim(0,themax*scale)
    
    for i in np.arange(Nparams):
        plt.plot(dmvals,pdms[i,:]*scale,color="grey",linestyle="-")
    
    plt.plot(dmvals,mean*scale,color="black",linestyle="-",linewidth=2,label="Simulation mean")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(label+"_pdm.png")
    plt.close()   

main()
