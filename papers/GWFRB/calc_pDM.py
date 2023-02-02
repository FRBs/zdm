"""

Script produced by Clancy James (clancy.james@curtin.edu.au)

This script calculates p(DM|z) and hence the associated value of PDM

It requires the followinging libraries from GitHub:
https://github.com/FRBs/zdm 
https://github.com/FRBs/FRB
https://github.com/FRBs/ne2001
as well as several standard public libraries.

Run with:
python3 gwfrb.py

Inputs:
    - CHIME FRB data, in CHIME/,
        Downloaded from https://www.chime-frb.ca/catalog
        536 FRBs total
    - Limits on FRB host galaxy properties, from Macquart et al:
        https://ui.adsabs.harvard.edu/abs/2020Natur.581..391M/abstract
        Data extracted from paper pdf due to loss of original data
    - GW data in GW/
        From https://www.gw-openscience.org/
    

Outputs: all placed in "Outputs/"
    - Prediction p(DM|z_GW190425A), plotted in 
    Rank of FRB 

"""


# this uses the zdm, ne2001, and frb libraries on github
# standard Python functionality
import argparse
import os
import sys
import time
import scipy as sp
import numpy as np
import pickle
from pkg_resources import resource_filename

# import zdm modules
from zdm import cosmology as cos
from zdm import survey
from zdm import grid
from zdm import pcosmic
from zdm.misc_functions import *
from zdm import beams
from zdm import iteration as it

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import NullFormatter
from zdm.craco import loading

print("Successful import!")

#import igm
defaultsize=16
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

global DM_Halo
DM_Halo=50 # pc/cm3. Assumed value for DM halo contribution. Same as Macquart et al.

global outdir
outdir='pDMOutputs/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def main():
    
    if not os.path.exists('pDMOutputs/'):
        os.mkdir('Outputs/')
    
    global DM_Halo, outdir
    
    # should default to Planck H0
    state = loading.set_state()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get starting grid - mostly don't need this functionality, but
    # it's got routines to combine cosmological and host galaxy DM
    s,g = survey_and_grid(nz=70,zmin=0.001,zmax=0.07,ndm=500,dmmax=1000.)
    
    ######## CHIME Data #######
    
    # extract data from CHIME, get extragalctic DM
    names,DMEGne,DMEGym=get_chime_data()
    index=names.index("FRB20190425A")
    dmeg_20190425A_1=DMEGne[index]
    dmeg_20190425A_2=DMEGym[index]
    
    ############ Load best-fit values of host galaxy DM properties #######
    # here, we get the integration values of galaxy mean DM
    # return values are:
    # values of mu (mean host DM)
    # mu_ps mean probability for that value (do not use this)
    # weighted probability (reflects span in dx, i.e. it is p(x) dx )
    mu_values,mu_ps,mu_weights=get_equivalent_values('Macquart/mu_host.dat',int(5),plot=outdir+'mu',plot2=1,lognorm=True)
    # and equivalent for log-standard deviation
    sigma_values,sigma_ps,sigma_weights=get_equivalent_values('Macquart/sigma_host.dat',int(5),plot=outdir+'sigma',plot2=2)
    
    ######## load p(z) from GW data #######
    zs,zweights=get_redshift_data()
    
    ################################################################################
    ######################### Create expected p(DM) distribution ###################
    ################################################################################
    
    # this will contain all the data  
    wfile=outdir+'weighted_expected.npy'
    
    grid=gv# begins initialised
    
    # either load or create new wexpected distribution
    if os.path.exists(wfile):
        wexpected=np.load(wfile)
    else:
        print("About to iterate over distributions - this will take O~30 minutes")
        t0=time.time()
        wexpected=np.zeros([g.dmvals.size])
        sumweight=0 
        # loops over mean of log host galaxy mean DM values
        for i,mu in enumerate(mu_values):
            # loops over sigma of log host galaxy DM values
            for j,sigma in enumerate(sigma_values):
                pset=[lEmin,lEmax,alpha,gamma,sfr_n,mu,sigma,C]
                
                # weights are probabilities of these mu and sigma
                weight=mu_weights[i]*sigma_weights[j]
                
                # calculate the expected p(DM) distribution for this combination of mu and sigma,
                # given a distribution of z
                state.host.lmean = mu
                state.host.lsigma = sigma
                expected,grid=get_p_value(zs,zweights,pset,state,grid=grid)#,plot='GWFRB/')
                
                # sum this with weights
                wexpected += expected*weight
                sumweight += weight # checks this sums to 1 - it does
        print("Sumweight comes to ",sumweight," (should be unity)")
        np.save(wfile,wexpected)
        t1=time.time()
        print("Calculated distribution of DMs in ",t1-t0," seconds")
    
    ####################################################################################
    ############################### Generate the main plot! ############################
    ####################################################################################
    
    ############## Main plot for paper #############
    # plots weighted expectation value
    # against histogram of CHIME FRBs
    # and the measured extragalactic DM for 20190425A
    # NOTE: DMEG calculations peviously do NOT include
    #   halo contributions! Need to add DM_Halo in.
    
    total=np.sum(wexpected)
    wexpected = wexpected/np.sum(wexpected)/(g.dmvals[1]-g.dmvals[0]) # sums to 1, means sums to ddm when integrated
    plt.figure()
    
    ax=plt.gca()
    ax2 = ax.twinx()
    
    plt.sca(ax)
    hpdm = plt.plot(g.dmvals+DM_Halo,wexpected,linewidth=3,label='p(DM)')
    plt.sca(ax2)
    plt.plot(g.dmvals+DM_Halo,wexpected-100,linewidth=3,label='$p({\\rm DM}-{\\rm DM}_{\\rm MWISM})$',
        color=ax.lines[-1].get_color())
    plt.sca(ax)
    plt.xlabel('${\\rm DM}-{\\rm DM}_{\\rm MWISM} ({\\rm pc\\,cm}^{-3}$)')
    plt.ylabel('$p({\\rm DM}-{\\rm DM}_{\\rm MWISM})$')
    
    #ax.set_aspect(10)
    plt.yticks(ticks=[0,0.0025,0.005,0.0075,0.01])
    
    #plt.text(DMEGne[index]+DM_Halo+5,0.002,'FRB 20190425A',rotation=90,fontsize=12)
    plt.xlim(0,1000)
    plt.ylim(0,0.01)
    hfrb = plt.plot([DMEGne[index]+DM_Halo,DMEGne[index]+50],[0,0.009],linestyle='--',color='orange',linewidth=3,label='FRB 20190425A')
    
    #ax2 = ax.twinx()
    plt.sca(ax2)
    
    hfrb = plt.plot([DMEGne[index]+DM_Halo,DMEGne[index]+50],[-100,0.009-100],linestyle='--',color='orange',linewidth=3,label='FRB 20190425A')
    
    
    ax2.set_ylabel('$N_{\\rm FRB}$')
    ax2.set_ylim(0,50)
    bins=np.linspace(50,3000,61)
    plt.hist(DMEGne+50,color='green',edgecolor='black',alpha=0.3,density=False,bins=bins,label='All CHIME FRBs')
    leg=plt.legend(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(outdir+'weighted_p_dm.png')
    plt.savefig(outdir+'weighted_p_dm.pdf')
    plt.close()
    
    ####################################################################################
    ############################### Evaluate Probabilities #############################
    ####################################################################################
    
    # index here is the index of FRB 20190425A
    NMC=1000 # number of Monte Carlo samples
    
    ntot_real=final_evaluate(DMEGne,DMEGym,g.dmvals,wexpected,index,-1,verbose=False)
    print("Compared to other FRBs, 20190425A is the ", ntot_real," to best match")
    # we also send both NE2001 and YMW estimates of Galactic ISM contributions
    ntot_mc=final_evaluate(DMEGne,DMEGym,g.dmvals,wexpected,index,NMC,verbose=False)#-1,-2, or 1000)
    # returns 4.281  greater than average, out of 474
    print("When randomising GalacticDM, 20190425A has an average rank of ", ntot_mc)

def final_evaluate(DMEGne,DMEGym,dmvals,expected,index,NDMtrials,verbose=False):
    """
    Performs NDM MC trials by randomly deviating Galactic DM
    estimates from measured values.
    
    If NDMtrials == -1:
        Just use actual measured values
    Else:
        Perform NDM random samplings
    """
    
    
    ddm=dmvals[1]-dmvals[0]
    NFRB=DMEGne.size
    DMEGmean=(DMEGne+DMEGym)/2.
    DMEGsigma=np.abs(DMEGne-DMEGym)
    
    if NDMtrials==-1:
        NDMtrials=1
        DMEGsigma[:]=0
        DMEGmean=DMEGne
    elif NDMtrials==-2:
        NDMtrials=1
        DMEGsigma[:]=0
        DMEGmean=DMEGym
    
    rands=np.random.normal(0,1,size=[NFRB,NDMtrials])
    ntot=0.
    
    
    for j in np.arange(NDMtrials):
        DMEGs=DMEGmean+rands[:,j]*DMEGsigma
        lt0=np.where(DMEGs <= 0)[0]
        if len(lt0) > 0:
            DMEGs[lt0]=0.1
        
        OK=np.where(DMEGs < 500)[0]
        THIS=np.where(OK==index)[0]
        
        DMEGs=DMEGs[np.where(DMEGs < 500)[0]]
        THISN=DMEGs.size
        
        # make this faster - get indices faster
        # extract count equal to index, do not re-evaluate
        idm1s=(DMEGs/ddm).astype('int')+1
        idm2s=idm1s+1
        kdm2s=DMEGs/ddm-(idm1s+1)
        kdm1s=1.-kdm2s
        
        pvals=kdm1s*expected[idm1s]+kdm2s*expected[idm2s]
        thisp=pvals[THIS]
        order=np.where(pvals >= thisp)[0]
        ngreater=len(order)
        if verbose:
            print("Trial ",j," finds ",ngreater," greater or equal")
        ntot += ngreater
    ntot /= NDMtrials
    return ntot
    
def get_p_value(zs,zweights,pset,state,grid=None,plot=None):
    
    global DM_Halo
    
    zvals=grid.zvals
    dmvals = grid.dmvals
    
    gprefix='gwfrb'
    savefile='Outputs/'+gprefix+'grids.pkl'
    if grid is not None:
        # update grid to smear the intrinsic p(DM|z) distribution
        smear_mean=pset[5]
        smear_sigma=pset[6]
        #mask=pcosmic.get_dm_mask(grid.dmvals,[smear_mean,smear_sigma],grid.zvals)
        mask=pcosmic.get_dm_mask(dmvals,(state.host.lmean,state.host.lsigma),zvals)
        
        grid.smear_dm(mask)
    else:
        print("wtf is the grid???")
#    else:
#        print("Grid is None - initialising...")
#        # constants of beam method
#        thresh=0
#        method=2
#        
#        # constants of intrinsic width distribution
#        Wbins=5
#        Wscale=3.5
#        Wlogmean=1.70267
#        Wlogsigma=0.899148
#        Wbins=10
#        Wscale=2
#        Nbeams=[20]
#        
#        # we need a survey object to base the grid off
#        # details are not important
        # the grid object is only used to call functions
        # to update DM smearing anyways
#        lat50=survey.OldSurvey()
#        
#        sdir = os.path.join(resource_filename('zdm','data'), 'Surveys')
#        lat50.process_survey_file(sdir+'/CRAFT_class_II.dat') # anything would do here
#        lat50.init_DMEG(DM_Halo)
#        lat50.init_beam(method=method,thresh=thresh) # tells the survey to use the beam file
#        pwidths,pprobs=survey.make_widths(lat50,state)
#        efficiencies=lat50.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
#        
#        grids=initialise_grids([lat50],zDMgrid,zvals,dmvals,state,wdist=True)
#        grid=grids[0]
    
    ##### sets up an integral over the redshift distribution ####
    
    # integer valus of redshifts on the grid
    NZ=zs.size
    dz=zvals[1]-zvals[0]
    
    iz1s=(zs/dz).astype('int')-1 # because 0 is in fact dz, not o
    iz1s[np.where(iz1s <=0)[0]]=0 # correction in low regime, assumes linear scaling
    iz2s=iz1s+1
    
    kz2s=zs/dz-(iz1s+1) # since iz1=0 equates to dz, should be
    kz1s=1.-kz2s
    
    # setting some constant values
    ddm=dmvals[1]-dmvals[0]
    
    expected=grid.smear_grid[iz1s,:].T*kz1s + grid.smear_grid[iz2s,:].T*kz2s
    wexpected=np.matmul(expected,zweights)
    
    if plot is not None:
        # plot p(DM|z) for this particular parameter set over
        # every 10th z value
        plt.figure()
        every=10
        for i,z in enumerate(zs):
            if i % every ==0:
                plt.plot(dmvals,expected[:,i],label="z="+str(z)[0:6])
        plt.xlabel('DM [pc.cm3]')
        plt.ylabel('p(DM|z)')
        plt.legend(fontsize=6)
        plt.xlim(0,200)
        plt.tight_layout()
        plt.savefig(plot+'pdm_z.pdf')
        plt.close()
    
        # plots expected distribution for particular redshift of GW event
        iz=int(0.034/dz)
        best_expected=grid.smear_grid[iz,:]
        pexpected = best_expected/np.max(best_expected)
        plt.figure()
        plt.plot(dmvals,pexpected,label='Macquart et al.')
        plt.plot([79.4,79.4],[0,1],label='FRB20190425A')
        plt.text(83,0,'FRB20190425A',rotation=90)
        plt.legend()
        plt.xlabel('DM (pc cm$^{-3}$)')
        plt.ylabel('$p({\\rm DM}-{\\rm DM}_{\\rm MW ISM}|z=0.034)$')
        plt.xlim(0,1000)
        plt.tight_layout()
        plt.savefig(plot+'pdm_given_z.pdf')
        plt.close()
    
    return wexpected,grid

    
def get_chime_data():
    
    global DM_Halo
    # read in the CHIME FRB data
    chime_dmeg=np.zeros([474])
    names=[]
    with open("CHIME/chime_singlefrbs_ne.dat") as f:
        lines=f.readlines()
        for i,l in enumerate(lines):
            data=l.split()
            names.append(data[0])
            dm_eg=float(data[1])
            dm_eg -= DM_Halo # subtracts assumed halo contribution from Macquart et al fit
            chime_dmeg[i]=dm_eg
            
    chime_ygdmeg=np.zeros([474])
    with open("CHIME/chime_allfrbs_ym.dat") as f:
        lines=f.readlines()
        j=0
        for i,l in enumerate(lines):
            data=l.split()
            if data[0]==names[j]:
                chime_ygdmeg[j]=float(data[1])-DM_Halo
                j += 1
                print(l)
    return names,chime_dmeg,chime_ygdmeg

def get_redshift_data():
    """
    Returns z and p(z) dz for the distance estimates from the GW pipeline
    Does this by interpolating the GW data with splines.
    
    Automatically produces the plot 
    """
    global outdir
    zs=[]
    pzs=[]
    with open("GW/extracted_redshift.dat") as f:
        lines=f.readlines()
        for i,l in enumerate(lines):
            data=l.split()
            z=float(data[0])
            pz=float(data[1])
            zs.append(z)
            pzs.append(pz)
    zs=np.array(zs)
    pzs=np.array(pzs)
    nz=40
    from scipy.interpolate import interp1d
    fit=interp1d(zs,pzs,kind='cubic')    
    zmin=np.min(zs)
    zmax=np.max(zs)
    linzs=np.linspace(zmin,zmax,nz)
    plinzs=fit(linzs)
    weights=plinzs/np.sum(plinzs)
    
    
    plt.figure()
    plt.plot(zs,pzs,label='extracted data',linestyle='',marker='x')
    plt.plot(linzs,plinzs,label='spline fit')
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+'redshift_fit.pdf')
    plt.close()
    
    plt.figure()
    plt.plot(linzs,plinzs,label='spline fit',color='black')
    plt.xlabel('z')
    plt.ylabel('p(z)')
    #plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+'pz.png')
    plt.close()
    
    # we now normalise it numerically to sum to unity, rather
    # than normalise such that \int p(z) dz = 1
    plinzs /= np.sum(plinzs)
    return linzs,plinzs

def get_equivalent_values(filename,every,plot=None,plot2=None,lognorm=False):
    """
    Loads data taken from Macquart et al, and converts
        it to equivalent values in the notation of the
        zDM code.
    """
    
    global outdir
    
    # marginalises over the mean 
    means=np.loadtxt(filename)
    
    n_mu_p,ncol=means.shape
    if lognorm:
        means[:,0]=np.log10(means[:,0])
    else:
        means[:,0]=means[:,0]
    mu_probs=means[:,1]
    mu_vals=means[:,0]
    
    args=np.argsort(mu_vals)
    mu_vals=mu_vals[args]
    mu_probs=mu_probs[args]
    
    dvs=mu_vals[1:]-mu_vals[:-1]
    mean_ps=(mu_probs[1:]+mu_probs[:-1])/2.
    
    mean_vals=(mu_vals[1:]+mu_vals[:-1])/2.
    # does this for every value
    every=int(5)
    
    integrate=mean_ps*dvs
    norm=np.sum(integrate)
    
    ennp=n_mu_p/every
    mus=[]
    dps=[]
    meanps=[]
    for i in np.arange(ennp):
        i=int(i)
        # calculates the probabalistic weight
        
        # protection against the final value
        start=i*every
        stop=min((i+1)*every,dvs.size)
        if start==stop:
            break
        
        dp=np.sum(dvs[start:stop]*mean_ps[start:stop])
        dp /= norm
        
        meanp=np.sum(mean_vals[start:stop]*mean_ps[start:stop])/np.sum(mean_vals[start:stop])
        
        # gets weighted mean value of ps
        mu=np.sum(mean_vals[start:stop]*mean_ps[start:stop])/np.sum(mean_ps[start:stop])
        
        mus.append(mu)
        dps.append(dp)
        meanps.append(meanp)
    mus=np.array(mus)
    dps=np.array(dps)
    meanps=np.array(meanps)
    plot_dps = dps/every
    
    if plot is not None:
        plt.figure()
        plt.plot(means[:,0],means[:,1],label='orig data')
        plt.plot(mus,meanps,marker='x',linestyle='',label='my interpolation')
        plt.plot(mus,plot_dps*40,marker='x',linestyle='',label='integration weight')
        plt.legend()
        plt.xlabel('$\\mu$')
        plt.ylabel('$p(\\mu)$')
        plt.savefig(plot+'_check.pdf')
        plt.close()
    
    if plot2 is not None:
        plt.figure()
        if lognorm:
            means[:,0]=10**(means[:,0])
        #plt.plot(means[:,0],means[:,1],label='orig data')
        plt.plot(means[:,0],means[:,1],linestyle='-',color='black',label='orig data')
        #plt.plot(mus,plot_dps*40,marker='x',linestyle='',label='integration weight')
        #plt.legend()
        plt.ylim(0,1.1)
        if plot2==1:
            plt.xlabel('$\\exp(\\mu_{\\rm host})\\, [{\\rm pc\\, cm}^{-3}]$')
            plt.ylabel('$p(\\mu_{\\rm host})$')
            plt.tight_layout()
            plt.savefig(outdir+'pmu.png')
        else:
            plt.xlabel('$\\sigma_{\\rm host}$')
            plt.ylabel('$p(\\sigma_{\\rm host})$')
            plt.tight_layout()
            plt.savefig(outdir+'psigma.png')
        
        plt.close()
    
    # the three arrays we need are:
    mu_values=mus
    mu_ps=meanps
    mu_weights=dps
    return mu_values,mu_ps,mu_weights

def survey_and_grid(survey_name:str='CRAFT_class_I_and_II',
            init_state=None,
            state_dict=None, iFRB:int=0,
               alpha_method=1, NFRB:int=10, 
               lum_func:int=2,sdir=None,
               nz=500,zmin=0.01,zmax=5,ndm=1400,dmmax=7000.,
               nbins=5):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        init_state (State, optional):
            Initial state
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        iFRB (int, optional): Starting index for the FRBs.  Defaults to 0
        lum_func (int, optional): Flag for the luminosity function. 
            0=power-law, 1=gamma, 2=gamma+spline.  Defaults to 0.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters

    Raises:
        IOError: [description]

    Returns:
        tuple: Survey, Grid objects
    """
    # Init state
    if init_state is None:
        state = loading.set_state(alpha_method=alpha_method)
        # Addiitonal updates
        if state_dict is None:
            state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
            state.energy.luminosity_function = lum_func
        state.update_param_dict(state_dict)
    else:
        state = init_state
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'),
        zlog=False,nz=nz,zmin=zmin,zmax=zmax,ndm=ndm,dmmax=dmmax,)

    ############## Initialise surveys ##############
    if sdir is not None:
        print("Searching for survey in directory ",sdir)
    else:
        sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, nbins=nbins,
                                 iFRB=iFRB)
    
    # generates zdm grid
    grids = initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]

main()
