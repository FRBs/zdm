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

from frb.dm import igm


import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


#params 
NMC = 10 
nu = 1. # GHz
t_samp = 1.182  # ms
bandwidth = 1.0 # MHz
snr_thresh = 4. # SNR threshold
mean_DM_MW = 80. # pc cm^-3
disp_DM_MW = 50. # pc cm^-3



#df_frbs = pd.DataFrame({
#        'DMeg': dm_samples,
#        'z': zs,
#        'M_r': Mr_samples,
#        'm_r': mr_samples
#    })


def main():
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
    
    #survey_state = sd.SurveyData()
    #survey_state.telescope.NBINS = 30
    survey_dict = {}
    survey_dict["NBINS"] = 50
    surveys, grids = loading.surveys_and_grids(survey_names = [name],repeaters=False,
                                                init_state=state,survey_dict = survey_dict)
    s = surveys[0]
    g = grids[0]
    
    plot_prediction(g)
    
    NMC = 10000
    frbs = gen_mc_frbs(g,NMC)
    
    # I did this once for N=100,000
    compare_rates(g,frbs,opdir,downsample=10)
    
    # adds m_r values to the FRBs
    gen_hosts(g,frbs)
    
    frbs.to_csv("craco_900_mc_sample.csv",index=False)

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
            name=opdir+"downsampled_predicted_zdm.png",norm=3,log=False,
            label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm EG}$',
            zmax=3.,DMmax=3000.)
    
    figures.plot_grid(new_hist,new_zvals,new_dmvals,
            name=opdir+"downsampled_mc_zdm.png",norm=3,log=False,
            label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm EG}$',
            zmax=3.,DMmax=3000.)
    
    
    
def gen_hosts(g,frbs):
    """
    Generate absolute and apparent magnitudes for FRB hosts
    
    args:
        frb (pandas dataframe): output of MC generation, containing
                                frb["z"] redshifts
    """
    opstate = op.OpticalState()
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


main()
