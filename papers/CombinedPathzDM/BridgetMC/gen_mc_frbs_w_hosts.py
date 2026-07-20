import os

from zdm import loading
from zdm import states
from zdm import survey_data as sd
from zdm import optical as opt
from zdm import optical_params as op
from zdm import figures

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
from frb.dm import igm

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(prefix=""):
    """
    Generates a Monte Carlo sample of FRB properties, including host properties
    
    """
    
    opdir = "MC_Generation_Plots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # creates ASKAP grid
    name = "CRAFT_CRACO_900"
    state = states.load_state("HoffmannHalo25") # old scattering
    state.width.WNbins = 100
    state.width.WNInternalBins = 1000
    state.MW.sigmaHalo=0.
    state.MW.sigmaDM=0.
    state.energy.lEmin = 30
    
    # default value of fsfr is 0.5!!! Change it to 1.5
    
    #survey_state = sd.SurveyData()
    #survey_state.telescope.NBINS = 30
    survey_dict = {}
    survey_dict["NBINS"] = 50
    
    # generate or load surveys and grids as appropriate
    
    pklname = 'survey_and_grid.pkl'
    
    if os.path.exists(pklname):
        with open(pklname, 'rb') as pklfile:
            surveys = pickle.load(pklfile)
            grids = pickle.load(pklfile)
    else:
        surveys, grids = loading.surveys_and_grids(survey_names = [name],repeaters=False,
                                                init_state=state,survey_dict = survey_dict)
        
        with open(pklname, 'ab') as pklfile:
            pickle.dump(surveys, pklfile)
            pickle.dump(grids, pklfile)
    
    s = surveys[0]
    g = grids[0]
    
    plot_prediction(g,opdir)
    
    # Do the below for some hard-core tests - but only if you perhaps don't
    # want to assign host galaxy magnitudes, which takes longer
    #NMC = 1000000
    #savefile="1M_craco_900_mc_sample.csv"
    
    # standard run. Only 1000 get used in MCMC
    NMC = 1000
    savefile=prefix+"craco_900_mc_sample.csv"
    
    if os.path.exists(savefile):
        frbs = pd.read_csv(savefile)
    else:
        frbs = gen_mc_frbs(g,NMC)
        
        # adds m_r values to the FRBs
        gen_hosts(g,frbs,fsfr=1.5)
    
        frbs.to_csv(savefile,index=False)
    
    # makes scatter plots of generated FRBs
    make_scatter_plots(frbs,opdir)
    
    exit()
    # Do the below for tests, typically using 100,000 FRBs at least.
    # I did this once for N=1,000,000. It works!
    compare_rates(g,frbs,opdir,downsample=10)
    
    # loads fake survey according to 
    compare_b_w_dists(g,s,frbs,opdir)
    
    
    
def compare_b_w_dists(g,s,frbs,opdir):
    """
    Compares beam and width distributions of FRBs
    """
    
    
    # dimensions: nz, ndm, nbeam
    bf = g.b_fractions
    nz,ndm,nb = bf.shape
    NFRB = len(frbs)
    
    brate = np.zeros([nb])
    for i in np.arange(nb):
        #nzxndm giving volume * fraction
        bv = np.multiply(bf[:,:,i].T, g.dV).T
        brate[i] = np.sum(bv * g.sfr_smear * s.dm_mask)
    
    # dimensions: nz, ndm, nbeam
    wf = g.w_fractions
    nz,ndm,nw = wf.shape
    
    wrate = np.zeros([nw])
    for i in np.arange(nw):
        #nzxndm giving volume * fraction
        wv = np.multiply(wf[:,:,i].T, g.dV).T
        wrate[i] = np.sum(wv * g.sfr_smear * s.dm_mask)
        #wrate[i] = np.sum(wf[:,:,i])
    
    
    ###### get MC values #####
    
    # also does MC init calculation
    #print("initting MC")
    #g.initMC()
    #pwb = g.MCpwb.reshape([nb,nw])
    #MCpw = np.sum(pwb,axis=0)
    #MCpb = np.sum(pwb,axis=1)
    
    ##### Calculates width from data #######
    
    width_hist = np.zeros([nw])
    iws1, iws2, dkws1, dkws2 = s.get_w_coeffs(frbs["w"])
    for i in np.arange(NFRB):
        width_hist[iws1[i]] += dkws1[i]
        width_hist[iws2[i]] += dkws2[i]
    
    # sums over DM axis
    ebar = np.sum(g.eff_table,axis=1)
    
    plt.figure()
    plt.plot(width_hist/np.sum(width_hist),label="Generated")
    plt.plot(wrate/np.sum(wrate),label="Predicted")
    #plt.plot(g.eff_weights/np.sum(g.eff_weights),label="weights")
    plt.plot(ebar/np.sum(ebar),label="efficiencies")
    #plt.plot(MCpw/np.sum(MCpw),label="MC prediction")
    plt.legend()
    plt.ylabel("$P(w)$")
    plt.xlabel("Width bin")
    plt.tight_layout()
    plt.savefig(opdir+"width_comparison.png")
    plt.close()
    
    ###### calculates bvals for data #####
    s.init_frb_bvals(frbs["B"])
    bweights = np.sum(s.frb_bweights,axis=0)
    
    plt.figure()
    plt.plot(bweights/np.sum(bweights),label="Generated")
    plt.plot(brate/np.sum(brate),label="Predicted")
    #plt.plot(MCpb/np.sum(MCpb),label="MC prediction")
    plt.legend()
    plt.ylabel("$P(B)$")
    plt.xlabel("Beam bin")
    plt.tight_layout()
    plt.savefig(opdir+"beam_comparison.png")
    plt.close()
    
    # calls survey for width-making
    s.make_widths()
    
def plot_prediction(g,opdir):
    """
    Makes 2d histogram of generated FRBs for comparison with predictions
    """
    
    # predicted grid of rates
    rates=g.get_rates()
    figures.plot_grid(rates,g.zvals,g.dmvals,
            name=opdir+"predicted_zdm.png",norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm EG}$',
            zmax=2.5,DMmax=2000.)
    
    

def compare_rates(g,frbs,opdir,downsample=10):
    """
    Makes 2d histogram of generated FRBs for comparison with predictions
    """
    
    # predicted grid of rates
    rates=g.get_rates()
    nz,ndm = rates.shape
    NFRB = len(frbs)
    
    dz = g.zvals[1]-g.zvals[0]
    dDM = g.dmvals[1] - g.dmvals[0]
    NZ = g.zvals.size
    NDM = g.dmvals.size
    zbins = np.linspace(g.zvals[0] - dz/2., g.zvals[-1] + dz/2.,NZ+1)
    DMbins = np.linspace(g.dmvals[0] - dDM/2., g.dmvals[-1] + dDM/2.,NDM+1)
    
    hist,xb,yb = np.histogram2d(frbs["z"],frbs["DMeg"],bins=[zbins,DMbins])
    
    
    figures.plot_grid(hist,g.zvals,g.dmvals,
            name=opdir+"mc_zdm.png",norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm EG}$',
            zmax=3.,DMmax=3000.)
    
    # downsamples original data
    new_nz = int(nz/downsample)
    new_ndm = int(ndm/downsample)
    
    new_rates = np.zeros([new_nz,new_ndm])
    new_hist = np.zeros([new_nz,new_ndm])
    new_zvals = 0.5*(g.zvals[::downsample] + g.zvals[downsample-1::downsample])
    new_dmvals = 0.5*(g.dmvals[::downsample] + g.dmvals[downsample-1::downsample])
    for i in np.arange(downsample):
        for j in np.arange(downsample):
            new_rates += rates[i::downsample,j::downsample]
            new_hist += hist[i::downsample,j::downsample]
    
    figures.plot_grid(new_rates,new_zvals,new_dmvals,
            name=opdir+"downsampled_predicted_zdm.png",norm=0,log=False,
            label='$N_{\\rm FRB}({\\rm DM}_{\\rm EG},z)$',
            project=False,ylabel='${\\rm DM}_{\\rm EG}$',
            zmax=3.,DMmax=3000.)
    
    figures.plot_grid(new_hist,new_zvals,new_dmvals,
            name=opdir+"downsampled_mc_zdm.png",norm=0,log=False,
            label='$N_{\\rm FRB}({\\rm DM}_{\\rm EG},z)$',
            project=False,ylabel='${\\rm DM}_{\\rm EG}$',
            zmax=3.,DMmax=3000.)
    
    # estimate N per cell. This is total number, time probability per cell
    
    norm = np.sum(new_rates)
    expectation = new_rates/norm * NFRB
    
    sigma = expectation**0.5
    
    # rounds up to unity - can't be less
    shape = sigma.shape
    toolow = np.where(sigma.flatten() < 1)[0]
    sigma=sigma.flatten()
    sigma[toolow] = 1.
    sigma = sigma.reshape(shape)
    
    deviation = (new_hist - expectation)/sigma
    
    # calculates contour level
    figures.plot_grid(deviation,new_zvals,new_dmvals,
            name=opdir+"downsampled_sigma_deviation.png",norm=0,log=False,
            label='$\\sigma$',
            project=False,ylabel='${\\rm DM}_{\\rm EG}$',
            zmax=3.,DMmax=3000.,othergrids = [sigma],Aconts=[0.99],alevels=[6.],
            other_alevels=[[1.]],othernames=["",""],cmap="bwr",clim=[-3,3])
    
    
def gen_hosts(g,frbs,fsfr=1.5):
    """
    Generate absolute and apparent magnitudes for FRB hosts
    
    args:
        frb (pandas dataframe): output of MC generation, containing
                                frb["z"] redshifts
    """
    opstate = op.OpticalState()
    opstate.loudas.fSFR=fsfr
    model = opt.loudas_model(opstate)
    wrapper = opt.model_wrapper(model,g.zvals)
    mrs = wrapper.gen_mc_mr(np.array(frbs["z"]))
    frbs["m_r"] = mrs
    
    frbs["M_r"] = opt.SimpleAbsoluteMags(frbs["m_r"],frbs["z"])
    
    # wrapper has p(z|m) distributions. Fundamentally, has p(m|z)!

def gen_mc_frbs(g,NMC):
    """
    generate MC FRBs
    
    Args:
        g [zdm grid object]: grid from which to generate MC FRBs
        NFRB [in] : number of FRBs to generate
    
    Returns:
        pandas dataframe containing FRB data
    """
    
    frbs = g.GenMCSample(NMC)
    df = pd.DataFrame({
        'DMeg': frbs[:,1],
        'z': frbs[:,0],
        's': frbs[:,3],
        'B': frbs[:,2],
        'w': frbs[:,4]
    })
    return df

def make_scatter_plots(frbs,opdir):
    """
    Makes scatter plots of the generated frbs
    """
    # generate scatter plot of z and mr
    plt.figure()
    plt.scatter(frbs["z"],frbs["m_r"],s=1,c=frbs["DMeg"], cmap='gnuplot2_r')
    cbar = plt.colorbar()
    cbar.set_label("DM$_{\\rm EG}$")
    plt.xlabel("Redshift, $z$")
    plt.ylabel("Apparent magnitude, $m_r$")
    plt.xlim(0,2.5)
    plt.ylim(10,32)
    plt.tight_layout()
    plt.savefig(opdir+"mr_z_plot.png")
    plt.close()
    
    plt.figure()
    plt.scatter(frbs["z"],frbs["DMeg"],s=3,c=frbs["m_r"], cmap='gnuplot2_r')
    cbar = plt.colorbar()
    plt.xlim(0,2.5)
    plt.ylim(0,2000)
    plt.xlabel("Redshift, $z$")
    cbar.set_label("Host $m_r$")
    plt.ylabel("Extragalactic DM, DM$_{\\rm EG}$")
    plt.tight_layout()
    plt.savefig(opdir+"dmeg_z_plot.png")
    plt.close()


    # snr selection
    OK1 = np.where(frbs["z"]> 0.1)[0]
    OK2 = np.where(frbs["z"]< 0.2)[0]
    OK3 = np.intersect1d(OK1,OK2)
    OK4 = np.where(frbs["B"]> 0.5)[0]
    OK5 = np.where(frbs["B"]< 1.0)[0]
    OK6  = np.intersect1d(OK4,OK5)
    OK7 = np.intersect1d(OK3,OK6)

    
    plt.figure()
    
    ### bins data ###
    bins=np.linspace(0,3,31)
    bcs = bins[0:-1]+0.5*(bins[1]-bins[0])
    ls = np.log10(frbs["s"])
    h,b = np.histogram(ls,bins=bins)
    h2,b = np.histogram(ls[OK7],bins=bins)
    lh = np.log10(h)
    lh2 = np.log10(h2)
    plt.plot(bcs,lh-lh[0],label="Generated: all parameter space")
    plt.plot(bcs,lh2-lh2[0],label="Selected subset")

    ## does a fit ####
    slope,intercept = np.polyfit(bcs, lh, 1)
    #print("Fit slope was ",slope)
    fitted = intercept+slope*bcs
    plt.plot(bcs,fitted-fitted[0],label="fit: slope = "+str(slope)[0:5])
    
    plt.plot(bcs,bcs*-1.5-bcs[0]*-1.5,label="Cartesian",color="black",linestyle="--")
    
    from zdm import energetics
    diff_lf = np.log10(energetics.vector_diff_gamma(10**bcs,1.,100,-1.12))
    
    diff_lf += bcs
    Es = np.logspace(0,3,1000)
    #diff_lf = energetics.vector_diff_gamma(10**bcs,1.,100,-1.12)
    plt.plot(bcs,diff_lf-diff_lf[0] ,label="Schechter, slope = -1.12")
    
    plt.ylim(-5,0)
    plt.xlabel("$\\log_{10}$SNR/SNR$_{\\rm th}$, s")
    #plt.yscale("log")
    plt.ylabel("$\\log_{10}$ Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"snr_histogram.png")
    plt.close()

prefix=""
main(prefix)
