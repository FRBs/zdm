"""
Examines the effect of handling of Galactic DM
Uses Monte Carlo data for ICS generated for
    DM_MW = 0,100,200,500 pc/cm3
    (generate using ics_dmg_mc.py)
Runs agaist H0
Also outputs component likelihoods (p(z,DM),p(z|DM), etc)

Plots results.

"""
import pytest

from zdm import io
from zdm.craco import loading
from pkg_resources import resource_filename
import os
import copy
import pickle
import numpy as np
from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import iteration as it

from IPython import embed
from matplotlib import pyplot as plt
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def make_grids(gamma):
    if gamma==0:
        raise ValueError("WARNING: gamma=0 is not currently implemented")
    ############## Load up ##############
    input_dict=io.process_jfile('../../papers/H0_I/Analysis/Cubes/craco_H0_Emax_cube.json')

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    surveys = []
    if (state_dict['energy']['luminosity_function'] == 1):
        names = ['CRAFT_ICS_g1a1_GMG_0','CRAFT_ICS_g1a1_GMG_100',
        'CRAFT_ICS_g1a1_GMG_200','CRAFT_ICS_g1a1_GMG_500',
        'CRAFT_ICS_g1a1_GMG_0_200','CRAFT_ICS_g1a1_GMG_0_500',
        'CRAFT_ICS_g1a1_GMG_500_as_0','CRAFT_ICS_g1a1_GMG_200_as_0','CRAFT_ICS_g1a1_GMG_100_as_0']
        
        sdm=[0.,0.,0.,0.,0.,0.,500.,200.,100.]
        nfrb=[1000,1000,1000,1000,2000,2000,1000,1000,1000]
        savefile="DMGtests/all_ICS_DMG_lls_H0.npy"
    else:
        raise ValueError("Sorry, not done gamma=0 yet...")
    
    ############## Initialise survey and grids ##############
    #NOTE: grid will be identical for all three, only bother to update one!
    surveys=[]
    grids=[]
    for i,name in enumerate(names):
        s,g = survey_and_grid(
            state_dict=state_dict,
            survey_name=name,NFRB=nfrb[i],subDM=sdm[i])
        surveys.append(s)
        grids.append(g)
    
    nH0=76
    nsurveys=len(names)
    lls=np.zeros([nsurveys,nH0])
    pzdmsarray=np.zeros([nsurveys,nH0,4])
    lllists=np.zeros([nsurveys,nH0,3])
    H0s=np.linspace(60,75,nH0)
    
    trueH0 = grids[0].state.cosmo.H0
    
    
    
    # Let's update H0 (barely) and find the constant for fun too as part of the test
    for ih,H0 in enumerate(H0s):
        
        vparams = {}
        vparams['H0'] = H0
        
        for i,s in enumerate(surveys):
            grid=grids[i]
            grid.update(vparams)
            if s.nD==1:
                llsum,lllist,expected,pzdms=it.calc_likelihoods_1D(grid,s,psnr=True,dolist=5,Pn=False)
            elif s.nD==2:
                llsum,lllist,expected,pzdms=it.calc_likelihoods_2D(grid,s,psnr=True,dolist=5,Pn=False)
            elif s.nD==3:
                # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                llsum1,lllist1,expected,pzdms1=it.calc_likelihoods_1D(grid,s,psnr=True,dolist=5,Pn=False)
                llsum2,lllist2,expected,pzdms2=it.calc_likelihoods_2D(grid,s,psnr=True,dolist=5,Pn=False)
                llsum = llsum1+llsum2
                pzdms=np.array(pzdms1)+np.array(pzdms2)
            lls[i,ih] = llsum
            pzdms=np.array(pzdms)
            pzdmsarray[i,ih,:]=pzdms
            lllists[i,ih,:]=np.array(lllist)
            print("H0",ih," Survey ",i," we have :")
            print(llsum)
            print(lllist)
            print(pzdms)
    np.save(savefile,lls)
    np.save("DMGtests/pzdmsarray.npy",pzdmsarray)
    np.save("DMGtests/lllists.npy",lllists)
    np.save("DMGtests/ICS_H0list.npy",H0s)

def plot_results(gamma):
    
    lls=np.load("DMGtests/all_ICS_DMG_lls_H0.npy")
    ngrid,nH0=lls.shape
    savefile="DMGtests/ICS_DMG_lls_plot.pdf"
    
    H0s=np.load("DMGtests/ICS_H0list.npy")
    
    plt.figure()
    plt.xlabel('$\\Delta H_0$ [km/s/Mpc]')
    plt.ylabel('$\\Delta \\log_{10} \\ell(H_0) \\, \\left( \\frac{50}{{\\rm N}_{\\rm FRB}} \\right)$')
    plt.ylim(-3,0)
    plt.xlim(-2,2)
    #plt.xlim(65,70)
    labels=["0","100","200","500","0 200","0 500","500 as 0","200 as 0","100 as 0"]
    styles=['-','--',':','-.','-','--',':','-.','-','--',':','-.']
    
    # shows 200 and 0 vs 200_0
    merged=lls[0,:]+lls[2,:]
    peak=np.max(merged)
    ipeak=np.where(merged==peak)[0]
    maxH0=H0s[ipeak]
    plt.plot(H0s[:]-maxH0,merged-peak,label="$\ell({\\rm DM}_{\\rm ISM}=0) + \ell({\\rm DM}_{\\rm ISM}=200)$",linestyle=styles[0],linewidth=3)
    #peak=np.max(lls3[0,:])
    peak2=np.max(lls[4,:])
    diff=peak-peak2
    diff *= 50./2000.
    plt.plot(H0s[:]-maxH0,lls[4,:]-peak2-diff,label="$\ell({\\rm DM}_{\\rm ISM}=0+ {\\rm DM}_{\\rm ISM}=200)$",linestyle=styles[1],linewidth=3,color=plt.gca().lines[-1].get_color())
    
    
    # shows 500 and 0 vs 500_0
    merged=lls[0,:]+lls[3,:]
    peak=np.max(merged)
    ipeak=np.where(merged==peak)[0]
    maxH0=H0s[ipeak]
    plt.plot(H0s[:]-maxH0,merged-peak,label="$\ell({\\rm DM}_{\\rm ISM}=0) + \ell({\\rm DM}_{\\rm ISM}=500)$",linestyle=styles[2],linewidth=3)
    peak2=np.max(lls[5,:])
    diff=peak-peak2
    diff *= 50./2000.
    plt.plot(H0s[:]-maxH0,lls[5,:]-peak2-diff,label="$\ell({\\rm DM}_{\\rm ISM}=0+ {\\rm DM}_{\\rm ISM}=500)$",linestyle=styles[3],linewidth=3,color=plt.gca().lines[-1].get_color())
    
    #trueH0=67.66
    #plt.plot([trueH0,trueH0],[-50,0],linestyle='--',color='grey',label='True $H_0$')
    plt.legend(loc=[0.19,0.02])
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()
    
    #############now does polyfits to the data...###########
    merged=lls[0,:]+lls[2,:]
    peak=np.max(merged)
    ipeak=int(np.where(merged==peak)[0][0])
    tofity=merged[ipeak-2:ipeak+3]
    tofitx=H0s[ipeak-2:ipeak+3]
    coeffs=np.polyfit(tofitx,tofity,2) #quadratic fit
    themax=-0.5*coeffs[1]/coeffs[0]
    print("max at :",themax)
    
    
    peak=np.max(lls[4,:])
    ipeak=int(np.where(lls[4,:]==peak)[0][0])
    tofity=lls[4,ipeak-2:ipeak+3]
    tofitx=H0s[ipeak-2:ipeak+3]
    coeffs=np.polyfit(tofitx,tofity,2) #quadratic fit
    themax2=-0.5*coeffs[1]/coeffs[0]
    print(coeffs)
    print("max at :",themax2)
    print("induced shift for 0--200 is ",themax2-themax)
    
    ############ gets shift for 500 #############
    merged=lls[0,:]+lls[3,:]
    peak=np.max(merged)
    ipeak=int(np.where(merged==peak)[0][0])
    tofity=merged[ipeak-2:ipeak+3]
    tofitx=H0s[ipeak-2:ipeak+3]
    coeffs=np.polyfit(tofitx,tofity,2) #quadratic fit
    themax=-0.5*coeffs[1]/coeffs[0]
    print("max at :",themax)
    
    
    peak=np.max(lls[5,:])
    ipeak=int(np.where(lls[5,:]==peak)[0][0])
    tofity=lls[5,ipeak-2:ipeak+3]
    tofitx=H0s[ipeak-2:ipeak+3]
    coeffs=np.polyfit(tofitx,tofity,2) #quadratic fit
    themax2=-0.5*coeffs[1]/coeffs[0]
    print(coeffs)
    print("max at :",themax2)
    print("induced shift for 0-500 is ",themax2-themax)
    

def plot_zdm_results():
    
    all_lls=np.load("DMGtests/all_ICS_DMG_lls_H0.npy")
    lllists=np.load("DMGtests/lllists.npy")
    lls=np.load("DMGtests/pzdmsarray.npy")
    H0s=np.load("DMGtests/ICS_H0list.npy")
    
    names=["llpzgdm","llpdm","llpdmgz","llpz"]
    for which in np.arange(4):
        plot_components(H0s,all_lls,lllists,lls,which)

def plot_components(H0s,all_lls,lllists,lls,i):
    which=1 # defined it is GMG 0
    
    plt.figure()
    #for i in np.arange(4):
        #plt.plot(H0s,lls[0,:,i],label=names[i])
    
    plt.plot(H0s,lls[i,:,0]-np.max(lls[i,:,0]),label="p(z|DM)")
    plt.plot(H0s,lls[i,:,1]-np.max(lls[i,:,1]),label="p(DM)")
    plt.plot(H0s,lls[i,:,2]-np.max(lls[i,:,2]),label="p(DM|z)")
    plt.plot(H0s,lls[i,:,3]-np.max(lls[i,:,3]),label="p(z)")
    plt.plot(H0s,lllists[i,:,0]-np.max(lllists[i,:,0]),label="p(z,DM)")
    plt.plot(H0s,lllists[i,:,2]-np.max(lllists[i,:,2]),label="p(s)")
    plt.plot(H0s,all_lls[i,:]-np.max(all_lls[i,:]),label="p(z,DM,s)")
    plt.xlabel('$H_0$ [km/s/Mpc]')
    plt.ylabel('$\ell-\ell_{\\rm max}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("DMGtests/zdm_components_"+str(i)+".pdf")
    plt.ylim(-1,0)
    plt.savefig("DMGtests/zoomed_zdm_components_"+str(i)+".pdf")
    plt.close()


def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            state_dict=None,
               alpha_method=1, NFRB:int=100, lum_func:int=0,sdir=None,DMG=None,subDM=None):
    """ Load up a survey and grid for a CRACO mock dataset
    
    Special version to subtract DMs according to my fiddling of the datasets,
    and force Galactic DMs to particular values
    
    Args:
        cosmo (str, optional): astropy cosmology. Defaults to 'Planck15'.
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        lum_func (int, optional): Flag for the luminosity function. 
            0=power-law, 1=gamma.  Defaults to 0.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters
        subDM subtracts this much DM from the DM values, since Galactic
        DM has been over-estimated.
    Raises:
        IOError: [description]

    Returns:
        tuple: Survey, Grid objects
    """
    # Init state
    from zdm.craco import loading
    state = loading.set_state(alpha_method=alpha_method)

    # Addiitonal updates
    if state_dict is None:
        state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
        state.energy.luminosity_function = lum_func
    state.update_param_dict(state_dict)
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'))
    
    ############## Initialise surveys ##############
    if sdir is not None:
        print("Searching for survey in directory ",sdir)
    else:
        sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, Nbeams=5)
    
    if subDM is not None:
        isurvey.DMs -= subDM
        isurvey.DMEGs -= subDM
        
    # reset survey Galactic DM values
    if DMG is not None:
        pwidths,pprobs=survey.make_widths(isurvey, 
                                    state.width.Wlogmean,
                                    state.width.Wlogsigma,
                                    state.beam.Wbins,
                                    scale=state.beam.Wscale)
        isurvey.DMGs[:]=DMG
        isurvey.init_DMEG(state.MW.DMhalo)
        _ = isurvey.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]

#use 0 for power law, 1 for gamma function
# have not implemented Xavier's version for gamma=0 yet...
gamma=1
make_grids(gamma)
plot_results(gamma)
plot_zdm_results()
