""" 
This script creates a zDM plot for SKA_Mid

It also estimates the raction of SKA bursts that will have
unseen hosts by a VLT-like optical obeservation
"""
import os

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
    
    
    """
    
    #### Initialises general zDM stuff #####
    state = parameters.State()
    # approximate best-fit values from recent analysis
    state.set_astropy_cosmo(Planck18)
    
    zDMgrid, zvals, dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    datdir=resource_filename('zdm', 'GridData'))
    
    ####### Loop over input files #########
    
    #indir = "inputs/"
    #infiles = os.listdir(indir)
    # loop over all files
    freqs = [865,1400,190]
    bws = [300,300,120]
    
    
    for i,tel in enumerate(["Band1", "Band2", "Low"]):
        # sets frequency and bandwidth for each instrument
        freq = freqs[i]
        bw = bws[i]
        for config in ["AA4","AAstar"]:
            infile = "inputs/"+tel+config+"_ID_radius_AonT_FoVdeg2"
            label = tel+"_"+config
            generate_sensitivity_plot(infile,state,zDMgrid, zvals, dmvals, label, freq, bw)
    
def generate_sensitivity_plot(infile,state,zDMgrid, zvals, dmvals, label, freq, bw,
                        opdir = "outputs/", plotdir = "plotdir/"):
    """
    generates a plot of FRB rate vs Nelements used
    """
    
    data=np.loadtxt(infile,dtype=str)
    radius = data[:,1].astype(float)
    sense_m2K = data[:,2].astype(float)
    fov = data[:,3].astype(float)
    
    Ngrids = radius.size
    
    # now converts sensitivity to Jy ms threshold
    Tint = 1e-3 # normalise sensitiivty at 1 ms
    Nsamps = 2*Tint*bw*1e6 # number of independent samples. bw in MHz
    Nsigma = 10. # need 10 sigma detection
    kboltzmann = 1.380649e-23
    
    ######## scale inputs to relevant zDM ones #######
    
    ## calculates threshold in Jy ms #########
    SEFD = 2*kboltzmann/sense_m2K
    thresh_Jyms = 1e26 * Nsigma * SEFD/Nsamps**0.5
    
    # calculates effective observation time
    # We begin by characterising a "full beam" as being
    # the FOV for a single element. This is a Gaussian
    # we then set that Tobs to be one year. Then, scale that down
    # as FOV decreases
    TOBS0 = 365.25 #calendar year
    TOBS = fov*TOBS0/fov[0]
    
    if True:
        # produces a plot of sensitivity
        plt.figure()
        p1,=plt.plot(thresh_Jyms,label="Thresh",color="blue")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel("Threshold [Jy ms]")
    
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.set_yscale("log")
        p2,=ax2.plot(fov,label="FOV",color="orange")
        ax2.set_ylabel("Field of view [deg 2]")
        
        naive = fov * thresh_Jyms**-1.5
        p3,=ax.plot(naive,label="Euclidean rate",color="red")
        
        plt.legend(labels = ["Thresh","FOV","Euclidean rate"], handles = [p1,p2,p3])
        
        plt.tight_layout()
        plt.savefig(label+"_thresh_fov.png")
        plt.close()
    
    
    #########
    # determines z-values of interest
    
    zis = np.array([1.,2.,3.,4.])
    izs = []
    for i,z in enumerate(zis):
        iz = np.where(zvals > z)[0][0]
        izs.append(iz)
    izs = np.array(izs)
    ########## speedups ############
    
    #set survey path
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    # we use SKA mid, but actually we will over-ride may attributes here
    survey_name='SKA_mid'
    
    # we can keep this constant - it smears DM due to host DM
    mask = pcosmic.get_dm_mask(dmvals, (state.host.lmean, state.host.lsigma), zvals, plot=False)
    
    prev_grid = None
    
    Nizs = np.zeros([Ngrids,zis.size])
    Ns = np.zeros([Ngrids])
    pzs = np.zeros([Ngrids,zvals.size])
    pdms = np.zeros([Ngrids,dmvals.size])
    
    # defines name of output files
    opfile1 = opdir+label+"_N.npy"
    opfile2 = opdir+label+"_pdm.npy"
    opfile3 = opdir+label+"_pz.npy"
    opfile4 = opdir+label+"_Nz.npy"
    opfile5 = opdir+label+"_Tobs.npy"
    opfile6 = opdir+label+"_thresh.npy"
    
    if os.path.exists(opfile1):
        load=True
    else:
        np.save(opfile5,TOBS)
        np.save(opfile6,thresh_Jyms)
    
    
    ######### Loop over all options ##########
    for i in np.arange(Ngrids):
        if load:
            continue
            # ignore if we are loading
        survey_dict = {"THRESH": thresh_Jyms[i], "TOBS": TOBS[i], "FBAR": freq, "BW": bw}
        s = survey.load_survey(survey_name, state, dmvals, zvals=zvals, survey_dict=survey_dict)
        print("iteration ",i,"thresh ",s.meta['THRESH'],"tobs ",s.meta['TOBS'],s.meta['FBAR'])
        g = zdm_grid.Grid(s, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist=True, prev_grid=prev_grid)
        scale = TOBS[i] * 10**(state.FRBdemo.lC)
        
        pz = np.sum(g.rates,axis=1)*scale
        pdm = np.sum(g.rates,axis=0)*scale
        N = np.sum(pdm)
        
        Ns[i] = N
        pzs[i,:] = pz
        pdms[i,:] = pdm
        
        # calculates number of FRBs beyond a certain redshift
        for j,iz in enumerate(izs):
            Niz = np.sum(pz[iz:])*scale
            Nizs[i,j] = Niz
        
        print("Did ",i," of ",Ngrids)
    
    if load:
        Ns = np.load(opfile1)
        pdms = np.load(opfile2)
        pzs = np.load(opfile3)
        Nizs = np.load(opfile4)
        
    else:
        #save outputs
        np.save(opfile1,Ns)
        np.save(opfile2,pdms)
        np.save(opfile3,pzs)
        np.save(opfile4,Nizs)
    
    # generates plots
    plt.figure()
    plt.xlabel("Number of elements used in tied beamforming")
    plt.ylabel("Relative number of FRBs detected")
    plt.plot(Ns/np.max(Ns),label = "All FRBs")
    for iz, zi in enumerate(zis):
        plt.plot(Nizs[:,iz]/np.max(Nizs[:,iz]),label="$z_{\\rm FRB} > "+str(zi)[0]+"$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir+label+"_PN.png")
    plt.close()
    
    
main()
