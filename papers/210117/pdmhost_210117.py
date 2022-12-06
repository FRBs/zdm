""" 
This script creates zdm grids and plots localised FRBs for FRB 20220610A,
using the exact conditions at which 20220610A was observed.

    - A set of plots giving p(dmhost):
        dmhost.pdf [included in the Science paper] (Fr)
        p_dmhost.pdf (p)
        210117_dmhost.pdf (FAr)
        210117_FAST_p_dmhost.pdf  (FpA)
        FAST_p_dmhost.pdf  (Fp)
        rest_frame_FAST_p_dmhost.pdf  (Frp)
        rest_frame_210117_FAST_p_dmhost.pdf (FpAr)
        
       with the following inclusions:
        includes estimates also for FAST FRB 20190520B 'F'
        includes estimates also for ASKAP FRB 20210117 'A'
        includes posterior estimates for host galaxy contribution 'p'
        plotted in the rest-frame of the host galaxies 'r'
    
"""
import os

from zdm import analyze_cube as ac
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    
    
    from astropy.cosmology import Planck18
    
    # in case you wish to switch to another output directory
    opdir = "220610/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    names = '220610_only'
    sdir = "../../zdm/data/Surveys/"
    
    labels=["lEmax","alpha","gamma","sfr_n","lmean","lsigma"]
    
    
    vparams = {}
    vparams["H0"] = 67.4
    vparams["lEmax"] = 41.26
    vparams["gamma"] = -0.948
    vparams["alpha"] = 1.03
    vparams["sfr_n"] = 1.15
    vparams["lmean"] = 2.22
    vparams["lsigma"] = 0.57
    
    opfile=opdir+"Planck_standard.pdf"
    zvals,std_pzgdm=plot_expectations(names,sdir,vparams,opfile)
        
def plot_expectations(name,sdir,vparams,opfile):
    '''
    Initialise surveys and grids for specified parameters
    Plots zdm distribution
    Gets p(z|DM) for 20220610A
    
    Args:
        Name (string): survey name to load
        sdir (string): place to load survey from
        vparams (dict): parameters for the grid
        opfile (string): output name for the plot
        dmhost (bool): if True, performs p(z|dm) analysis for this parameter set
            The result is a series of plots giving p(DM_host) data.
    
    Returns:
        zvals (np.ndarray) z values for the below
        p(z|dm) (np.ndarray) Probability of redshift z given the DM of 20220610a
    
    '''
    # hard-coded values for this FRB
    DMEG220610=1458-31-50
    Z220610=1.0153
    
    zvals = []
    dmvals = []
    nozlist = []
    #state = set_special_state()
    s, g = loading.survey_and_grid(
        survey_name=name, NFRB=None, sdir=sdir, lum_func=2
        )  # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set up new parameters
    g.update(vparams)
    # gets cumulative rate distribution
    
    # does plot of p(DM|z)
    ddm=g.dmvals[1]-g.dmvals[0]
    dz=g.zvals[1]-g.zvals[0]
    idm=int(DMEG220610/ddm)
    iz=int(Z220610/dz)
    pzgdm = g.rates[:,idm]/np.sum(g.rates[:,idm])/dz
    
    logmean=g.state.host.lmean
    
    
    # probability distribution
    dmdist = g.grid[iz,:]
    #dm vals
    dmhost = (DMEG220610-g.dmvals)
    iOK=np.where(dmhost > 0.)[0]
    dmhost = dmhost[iOK]
    dmdist = dmdist[iOK]
    dmdist /= np.sum(dmdist) * ddm #normalisation to a probability distribution
    if True:
        
        # prior
        #args=[(g.state.host.lmean-np.log10(1.+Z220610))/0.4342944619,g.state.host.lsigma/0.4342944619]
        #*(1+Z220610)
        priors=pcosmic.linlognormal_dlin(dmhost*(1.+Z220610),2.2/0.4342944619,0.5/0.4342944619)
        norm_priors=np.sum(priors)*ddm
        priors /= norm_priors
        
        convolve = priors * dmdist
        convolve /= np.sum(convolve)*ddm
        
        imax = np.argmax(dmdist)
        dmmax = dmhost[imax]
        print("Peak DM host is ",dmmax*(1+Z220610))
        
        #v1,v2,k1,k2=ac.extract_limits(dmhost,dmdist,0.16) # 0.16 = (1-68%)/2
        #print("1 sigma bounds are ",v1*(1+Z220610),v2*(1+Z220610))
        #v1,v2,k1,k2=ac.extract_limits(dmhost,dmdist,0.48) # 0.49 ~ (1-0)/2
        #print("Median is ",v1*(1+Z220610),v2*(1+Z220610))
        
        # compare to the other one!
        # 20190520B
        zFAST = 0.241
        DMFAST = 1204.7-50.-60.
        izFAST = int(zFAST/dz)
        
        # probability distribution
        dmdistFAST = g.grid[izFAST,:]
        #dm vals
        dmhostFAST = (DMFAST-g.dmvals)
        iOK=np.where(dmhostFAST > 0.)[0]
        dmhostFAST = dmhostFAST[iOK]
        dmdistFAST = dmdistFAST[iOK]
        dmdistFAST /= np.sum(dmdistFAST) * ddm #normalisation to a probability distribution
        
        #v1,v2,k1,k2=ac.extract_limits(dmhostFAST,dmdistFAST,0.16) # 0.16 = (1-68%)/2
        #print("FAST: 1 sigma bounds are ",v1*(1+zFAST),v2*(1+zFAST))
        #v1,v2,k1,k2=ac.extract_limits(dmhostFAST,dmdistFAST,0.42) # 0.49 ~ (1-0)/2
        #print("FAST: Median is ",v1*(1+zFAST),v2*(1+zFAST))
        
        priorsFAST=pcosmic.linlognormal_dlin(dmhostFAST*(1.+zFAST),2.2/0.4342944619,0.5/0.4342944619)
        norm_priors=np.sum(priorsFAST)*ddm
        priorsFAST /= norm_priors
        
        convolveFAST = priorsFAST * dmdistFAST
        convolveFAST /= np.sum(convolveFAST)*ddm
        
        z210117=0.214
        DM210117=730-50.-34.4
        iz210117 = int(z210117/dz)
        
        # probability distribution
        dmdist210117 = g.grid[iz210117,:]
        #dm vals
        dmhost210117 = (DM210117-g.dmvals)
        iOK=np.where(dmhost210117 > 0.)[0]
        dmhost210117 = dmhost210117[iOK]
        dmdist210117 = dmdist210117[iOK]
        dmdist210117 /= np.sum(dmdist210117) * ddm #normalisation to a probability distribution
        
        # gets 1 sigma limits
        from zdm import analyze_cube as ac
        xvals=dmhost210117*(1.+z210117)
        l0,l1,k0,k1=ac.extract_limits(xvals,dmdist210117,(1.-0.6827)/2.)
        print("68% limits are ",l0,l1)
        imax=np.argmax(dmdist210117)
        print("max is ",dmhost210117[imax]*(1.+z210117))
        cdf=np.cumsum(dmdist210117)
        cdf /= cdf[-1]
        icdf = np.where(cdf>0.5)[0][0]
        print("median is ",dmhost210117[icdf]*(1.+z210117))
        
        #v1,v2,k1,k2=ac.extract_limits(dmhost210117,dmdist210117,0.16) # 0.16 = (1-68%)/2
        #print("210117: 1 sigma bounds are ",v1*(1+z210117),v2*(1+z210117))
        #v1,v2,k1,k2=ac.extract_limits(dmhost210117,dmdist210117,0.42) # 0.49 ~ (1-0)/2
        #print("210117: Median is ",v1*(1+z210117),v2*(1+z210117))
        
        priors210117=pcosmic.linlognormal_dlin(dmhost210117*(1.+z210117),2.2/0.4342944619,0.5/0.4342944619)
        norm_priors=np.sum(priors210117)*ddm
        priors210117 /= norm_priors
        
        convolve210117 = priors210117 * dmdist210117
        convolve210117 /= np.sum(convolve210117)*ddm
        
        ### plotting! ###
        plt.figure()
        plt.ylim(0,0.015)
        plt.xlim(0,2000)
        plt.xlabel('Rest frame ${\\rm DM}_{\\rm host}~[{\\rm pc\\,cm}^{-3}]$')
        plt.ylabel('$p({\\rm DM}_{\\rm host})$')
        plt.plot(dmhost210117*(1.+z210117),dmdist210117/(1.+z210117),label="FRB 20210117",linestyle='-')
        plt.plot(dmhost*(1.+Z220610),dmdist/(1.+Z220610),label="FRB 20220610A",linestyle=':')
        plt.plot(dmhostFAST*(1.+zFAST),dmdistFAST/(1.+zFAST),label="FRB 20190520B",linestyle='--')
        plt.plot(dmhost*(1.+Z220610),priors/(1.+Z220610),label="Fit to population",linestyle='-.')
        #plt.plot(dmhost*(1.+Z220610),dmdist/(1.+Z220610),label="FRB 20220610A")
        #plt.plot(dmhostFAST*(1.+zFAST),dmdistFAST/(1.+zFAST),label="FRB 20190520B",linestyle='--')
        #plt.plot(dmhost*(1.+Z220610),priors/(1.+Z220610),label="Expectation",linestyle=":")
        #plt.plot(dmhost210117*(1.+z210117),dmdist210117/(1.+z210117),label="FRB 20210117",linestyle='-.')
        plt.legend(loc='upper right')
        #plt.legend()
        #plt.legend(loc=[0.45,0.4])
        plt.tight_layout()
        plt.savefig('210117_dmhost.pdf')
        plt.close()
        
    return g.zvals,pzgdm

#from pkg_resources import resource_filename
    
from astropy.cosmology import Planck15, Planck18


main()
