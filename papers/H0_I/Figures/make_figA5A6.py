
"""
This generates Figure A5 and A6 from the H0 with zdm paper.

Firstly, it generats a Monte Carlo set of FRBs, and writes them
to survey files.

It then uses these to create some fake survey files for test purposes.

The third step is evaluating the likelihoods on these survey
files as a function of H0.

Finally, it generated figures A5 and A6 based on this.

All outputs are kept in the directory FigureA5A6

"""
from pkg_resources import resource_filename
import os
import copy
import pickle
import sys
import scipy as sp
import numpy as np
import time
import argparse

from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import iteration as it
from zdm import beams
from zdm import misc_functions

from IPython import embed

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib

from zdm import io

from matplotlib.ticker import NullFormatter

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    #generates MC files
    run_monte_carlo()
    
    # modifies them with the bash script
    command="./modify_figA5A6_data.sh FigureA5A6"
    os.system(command)
    
    # evaluates likelihoods
    evaluate_likelihoods()
    
    # generate plots
    plot_figure_A5()
    plot_figure_A6()


def run_monte_carlo(plots=False):
    "created N*survey.NFRBs mock FRBs for all surveys"
    "Input N : creates N*survey.NFRBs"
    "Output : Sample(list) and Surveys(list)"
    
    opdir="FigureA5A6/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    ############## Load up ##############
    input_dict=io.process_jfile('../Analysis/CRACO/Cubes/craco_H0_Emax_cube.json')

    # Deconstruct the input_dict
    state, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    state['energy']['luminosity_function'] = 2
    ######## choose which surveys to do an MC for #######
    
    # choose which survey to create an MC for
    name = 'CRAFT/ICS'
    # select which to perform an MC for
    which=1 #CRAFT_ICS
    N=1000
    for DMG in [0,100,200,500]:
        opfile=opdir+"CRAFT_ICS_DMG_"+str(DMG)+".dat"
        do_mc(state,name,N,which,DMG,opfile)
    
def do_mc(state_dict, survey_name, Nsamples, which_survey,DMG,opfile):
    
    # check if the output already exists - this is a long routine
    if os.path.exists(opfile):
        print("Output file already exists, skipping")
        return
    
    fout = open(opfile,'w')
    
    fout.write("BW    336 #MHz\n")
    fout.write("FRES  1 #MHz\n")
    fout.write("DIAM 12\n")
    fout.write("NBEAMS 36\n")
    fout.write("BEAM lat50_log #prefix of beam file\n")
    fout.write("THRESH 4.4 #Jy ms to a 1 ms burst, very basic. Likely higher\n")
    fout.write("SNRTHRESH 9 # signal-to-noise threshold: scales jy ms to snr\n")
    
    ############## Initialise survey and grid ##############
    s,g = survey_and_grid(
        state_dict=state_dict,
        survey_name=survey_name, NFRB=9,DMG=DMG,
        sdir="Surveys/")
          
    ############## Initialise surveys ##############
    #saves everything in this directory
    outdir='DMGtests/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    samples = []
    
    name = str(which_survey)
    savefile=outdir+'mc_sample_'+name+'_alpha_'+str(g.state.FRBdemo.alpha_method)+str(Nsamples)+'_DMG'+str(DMG)+'.npy'
    
    try:
        sample=np.load(savefile)
        print("Loading ",sample.shape[0]," samples from file ",savefile)
    except:
        print("Generating ",Nsamples," samples from survey/grid ",which_survey)
        sample=g.GenMCSample(Nsamples)
        sample=np.array(sample)
        np.save(savefile,sample)
    samples.append(sample)
    
    print("########### Set 1: complete ########")
    fout.write("KEY ID 	DM    	DMG     DMEG	Z	SNR	WIDTH\n")
    for j in np.arange(Nsamples):
        DMEG=sample[j,1]
        DMtot=DMEG+DMG+g.state.MW.DMhalo
        SNRTHRESH=9.5
        SNR=SNRTHRESH*sample[j,4]
        z=sample[j,0]
        w=sample[j,3]
        
        string="FRB "+str(j)+'  {:6.1f}  {:6.1f}   {:6.1f}  {:5.3f}   {:5.1f}  {:5.1f}\n'.format(DMtot,DMG,DMEG,z,SNR,w)
        fout.write(string)
    fout.close()
    
    
def evaluate_likelihoods(luminosity_function=2,verbose=False):
    
    opdir="FigureA5A6/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    savefile=opdir+"all_ICS_DMG_lls_H0.npy"
    
    
    # checks if output exists, skips if necessary
    sf2=opdir+"pzdmsarray.npy"
    sf3=opdir+"lllists.npy"
    sf4=opdir+"ICS_H0list.npy"
    
    if (os.path.exists(savefile) and os.path.exists(sf2)
        and os.path.exists(sf3) and os.path.exists(sf4)):
        print("All output already generated, skipping make grids...")
        return None
    
    if not luminosity_function==2:
        raise ValueError("WARNING: luminosity_function!=2 is not currently implemented")
    
    ############## Load up ##############
    input_dict=io.process_jfile('../Analysis/CRACO/Cubes/craco_H0_Emax_cube.json')

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    state_dict['energy']['luminosity_function'] = luminosity_function
    
    # step 1: generate FRBs from fake surveys with 0, 100, 200, and 500 Galactic DM
    
    surveys = []
    # the first four files are simply different sets created at different Galactic DMs
    # the last three are created at one Galactic DM, but relabelled as 0
    names = ['CRAFT_ICS_DMG_0','CRAFT_ICS_DMG_100',
        'CRAFT_ICS_DMG_200','CRAFT_ICS_DMG_500',
        'CRAFT_ICS_DMG_0_200','CRAFT_ICS_DMG_0_500',
        'CRAFT_ICS_DMG_500_as_0','CRAFT_ICS_DMG_200_as_0','CRAFT_ICS_DMG_100_as_0']
    
    # sets up parametrs for simulation
    sdm=[0.,0.,0.,0.,0.,0.,500.,200.,100.] #galactic DM contributions
    nfrb=[1000,1000,1000,1000,2000,2000,1000,1000,1000] #number of FRBs
    
    ############## Initialise survey and grids ##############
    #NOTE: grid will be identical for all three, only bother to update one!
    surveys=[]
    grids=[]
    sdir='FigureA5A6/'
    for i,name in enumerate(names):
        s,g = survey_and_grid(
            state_dict=state_dict,
            survey_name=name,NFRB=nfrb[i],subDM=sdm[i],
            sdir=sdir)
        surveys.append(s)
        grids.append(g)
    
    # number of trial H0 values
    nH0=76
    nsurveys=len(names)
    lls=np.zeros([nsurveys,nH0])
    pzdmsarray=np.zeros([nsurveys,nH0,4])
    lllists=np.zeros([nsurveys,nH0,3])
    H0s=np.linspace(60,75,nH0)
    
    trueH0 = grids[0].state.cosmo.H0
    
    # Let's update H0 (barely) and find the constant for fun too as part of the test
    for ih,H0 in enumerate(H0s):
        print("Evaluating likelihoods for H0=",H0)
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
            if verbose:
                print("H0",ih," Survey ",i," we have :")
                print(llsum)
                print(lllist)
                print(pzdms)
    np.save(savefile,lls)
    np.save(sf2,pzdmsarray)
    np.save(sf3,lllists)
    np.save(sf4,H0s)

def plot_figure_A5(verbose=False):
    """
    Plots figure A5. Note this uses different MC data so results may be different!
    """
    
    lls=np.load("FigureA5A6/all_ICS_DMG_lls_H0.npy")
    ngrid,nH0=lls.shape
    savefile="FigureA5A6/FigureA5.pdf"
    
    H0s=np.load("FigureA5A6/ICS_H0list.npy")
    
    plt.figure()
    plt.xlabel('$\\Delta H_0$ [km/s/Mpc]')
    plt.ylabel('$\\log_{10} \\ell(H_0) - \\log_{10} \\ell_{\\rm max}$')
    plt.ylim(-5,0)
    plt.xlim(-5,5)
    
    
    # do 0, and 6,7,8
    
    # shows DMG=200 and 0 vs 200_0
    # merges the likelihoods for the samples with DMG=200 and DMG=0
    
    styles=['-','--',':','-.']
    labels=['${\\rm DM}^{\\rm true}_{\\rm ISM}={\\rm DM}^{\\rm eval}_{\\rm ISM}=0$',
        '${\\rm DM}^{\\rm true}_{\\rm ISM}=100,{\\rm DM}^{\\rm eval}_{\\rm ISM}=0$',
        '${\\rm DM}^{\\rm true}_{\\rm ISM}=200,{\\rm DM}^{\\rm eval}_{\\rm ISM}=0$',
        '${\\rm DM}^{\\rm true}_{\\rm ISM}=500,{\\rm DM}^{\\rm eval}_{\\rm ISM}=0$']
    
    for i,index in enumerate([0,8,7,6]):
        data=lls[index]
        if i==0:
            ipeak=np.argmax(data)
            maxH0=H0s[ipeak]
        peak=np.max(data)   
        plt.plot(H0s[:]-maxH0,data-peak,label=labels[i],linestyle=styles[i],linewidth=3)
    
    plt.legend(loc=[0.19,0.02])
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

def plot_figure_A6(verbose=False):
    """
    Plots results of above likelihood evaluation
    """
    
    lls=np.load("FigureA5A6/all_ICS_DMG_lls_H0.npy")
    ngrid,nH0=lls.shape
    savefile="FigureA5A6/FigureA6.pdf"
    
    H0s=np.load("FigureA5A6/ICS_H0list.npy")
    
    plt.figure()
    plt.xlabel('$\\Delta H_0$ [km/s/Mpc]')
    plt.ylabel('$\\Delta \\log_{10} \\ell(H_0) \\, \\left( \\frac{50}{{\\rm N}_{\\rm FRB}} \\right)$')
    plt.ylim(-3,0)
    plt.xlim(-2,2)
    #plt.xlim(65,70)
    labels=["0","100","200","500","0 200","0 500","500 as 0","200 as 0","100 as 0"]
    styles=['-','--',':','-.','-','--',':','-.','-','--',':','-.']
    
    # shows DMG=200 and 0 vs 200_0
    # merges the likelihoods for the samples with DMG=200 and DMG=0
    merged=lls[0,:]+lls[2,:]
    peak=np.max(merged)
    ipeak=np.where(merged==peak)[0]
    maxH0=H0s[ipeak]
    plt.plot(H0s[:]-maxH0,merged-peak,label="$\ell({\\rm DM}_{\\rm ISM}=0) + \ell({\\rm DM}_{\\rm ISM}=200)$",linestyle=styles[0],linewidth=3)
    #peak=np.max(lls3[0,:])
    peak2=np.max(lls[4,:])
    diff=peak-peak2
    diff *= 50./2000.
    # plots likelihood for sample which is half 0, half 200
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
    #print("max at :",themax)
    
    
    peak=np.max(lls[4,:])
    ipeak=int(np.where(lls[4,:]==peak)[0][0])
    tofity=lls[4,ipeak-2:ipeak+3]
    tofitx=H0s[ipeak-2:ipeak+3]
    coeffs=np.polyfit(tofitx,tofity,2) #quadratic fit
    themax2=-0.5*coeffs[1]/coeffs[0]
    #print(coeffs)
    #print("max at :",themax2)
    print("induced shift in peak probability for 0--200 is ",themax2-themax)
    
    ############ gets shift for 500 #############
    merged=lls[0,:]+lls[3,:]
    peak=np.max(merged)
    ipeak=int(np.where(merged==peak)[0][0])
    tofity=merged[ipeak-2:ipeak+3]
    tofitx=H0s[ipeak-2:ipeak+3]
    coeffs=np.polyfit(tofitx,tofity,2) #quadratic fit
    themax=-0.5*coeffs[1]/coeffs[0]
    #print("max at :",themax)
    
    
    peak=np.max(lls[5,:])
    ipeak=int(np.where(lls[5,:]==peak)[0][0])
    tofity=lls[5,ipeak-2:ipeak+3]
    tofitx=H0s[ipeak-2:ipeak+3]
    coeffs=np.polyfit(tofitx,tofity,2) #quadratic fit
    themax2=-0.5*coeffs[1]/coeffs[0]
    #print(coeffs)
    #print("max at :",themax2)
    print("induced shift in peak probability for 0-500 is ",themax2-themax)
    

def not_plot_figure_A5(verbose=False):
    
    all_lls=np.load("FigureA5A6/all_ICS_DMG_lls_H0.npy")
    lllists=np.load("FigureA5A6/lllists.npy")
    lls=np.load("FigureA5A6/pzdmsarray.npy")
    H0s=np.load("FigureA5A6/ICS_H0list.npy")
    
    names=["llpzgdm","llpdm","llpdmgz","llpz"]
    for which in np.arange(4):
        plot_components(H0s,all_lls,lllists,lls,which)

def plot_components(H0s,all_lls,lllists,lls,i,opdir="FigureA5A6/"):
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
    plt.savefig(opdir+"zdm_components_"+str(i)+".pdf")
    plt.ylim(-1,0)
    plt.savefig(opdir+"zoomed_zdm_components_"+str(i)+".pdf")
    plt.close()

def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            state_dict=None,
               alpha_method=1, NFRB:int=100, lum_func:int=0,sdir=None,DMG=None,subDM=None):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        cosmo (str, optional): astropy cosmology. Defaults to 'Planck15'.
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        lum_func (int, optional): Flag for the luminosity function. 
            0=power-law, 1=gamma.  Defaults to 0.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters

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
    state_dict['energy']['luminosity_function'] = 2
    
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
        pwidths,pprobs=survey.make_widths(isurvey, state)
                                    #state.width.Wlogmean,
                                    #state.width.Wlogsigma,
                                    #state.width.Wbins,
                                    #scale=state.width.Wscale)
        isurvey.DMGs[:]=DMG
        isurvey.init_DMEG(state.MW.DMhalo)
        _ = isurvey.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]

main()
