#import pytest

#from pkg_resources import resource_filename
import os
#import copy
#import pickle

#from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io

#from IPython import embed

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

def main():
    
    
    
    mu1=1
    mu2=2
    s1=1
    s2=1
    #survey.geometric_lognormals(mu1,s1,mu2,s2,plot=True)
    
    ############## Load up old model ##############
    input_dict=io.process_jfile('scat_test_old.json')
    
    # Deconstruct the input_dict
    state_dict1, cube_dict, vparam_dict1 = it.parse_input_dict(input_dict)
    
    # Initialise survey and grid 
    # For this purporse, we only need two different surveys
    name = 'CRAFT/ICS892'
    s1,g1 = loading.survey_and_grid(
        state_dict=vparam_dict1,
        survey_name=name,NFRB=None) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    ############## Load up new model ##############
    input_dict=io.process_jfile('scat_test_new.json')

    # Deconstruct the input_dict
    state_dict2, cube_dict, vparam_dict2 = it.parse_input_dict(input_dict)
    
    # Initialise survey and grid
    # For this purporse, we only need two different surveys
    s2,g2 = loading.survey_and_grid(
        state_dict=vparam_dict2,
        survey_name=name,NFRB=None) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    ############# do 2D plots ##########
    misc_functions.plot_grid_2(g1.rates,g1.zvals,g1.dmvals,
        name='CRAFT_ICS892_old_scat.pdf',norm=0,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
        project=True,FRBDM=s1.DMEGs,FRBZ=s1.frbs["Z"],Aconts=[0.01,0.1,0.5])
    
    misc_functions.plot_grid_2(g2.rates,g2.zvals,g2.dmvals,
        name='CRAFT_ICS892_new_scat.pdf',norm=0,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
        project=True,FRBDM=s2.DMEGs,FRBZ=s2.frbs["Z"],Aconts=[0.01,0.1,0.5])
    
    # second rates are higher. Why?
    misc_functions.plot_grid_2((g2.rates-g1.rates)/g1.rates,g2.zvals,g2.dmvals,
        name='CRAFT_ICS892_diff_scat.pdf',norm=0,log=False,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
        project=True,FRBDM=s2.DMEGs,FRBZ=s2.frbs["Z"])
    
    ############## examine width explicitly ##########
    plt.figure()
    plt.xlabel('width [ms]')
    plt.ylabel('p(width) dlogw')
    
    plt.scatter(s1.wlist,s1.wplist/np.max(s1.wplist),label='Old width model')
    NX=4001
    x1=np.linspace(0.01,250,NX)
    y1=pcosmic.loglognormal_dlog(np.log(x1),g1.state.width.Wlogmean,g1.state.width.Wlogsigma,(2.*np.pi)**-0.5/g1.state.width.Wlogsigma)
    
    plt.plot(x1,y1/np.max(y1),label='Fit to ASKAP/Parkes data')
    plt.xscale('log')
    
    #### new model ####
    
    plt.scatter(s2.wlist,s2.wplist/np.max(s2.wplist),label='Scattering (new) model')
    
    scale_mean=(1200/600.)**-4
    scale_mean=1
    h,b=survey.geometric_lognormals(g2.state.width.Wlogmean,g2.state.width.Wlogsigma,g2.state.scat.Slogmean*scale_mean,g2.state.scat.Slogsigma,bins=x1,Nrand=100000)
    
    
    plotb=(b[0:-1]+b[1:])/2.
    temph=h*plotb
    plt.plot(plotb,temph/np.max(temph),label='CHIME scatter + width model')
    
    plt.legend(loc='upper left')
    ############ cumulative ##########
    
    ax1=plt.gca()
    ax2=ax1.twinx()
    
    # cumulative distributions
    cy1=np.cumsum(y1/x1)
    cy1/=cy1[-1]
    
    cs1w=np.cumsum(s1.wplist)
    cs1w/=cs1w[-1]
    plt.scatter(s1.wlist,cs1w,marker='+')
    plt.plot(x1,cy1,linestyle='--')
    plt.xlim(1e-2,1e3)
    
    ####### second (CHIME) method 
    
    # cumulative plots 
    cs2=np.cumsum(s2.wplist)
    cs2/=cs2[-1]
    plt.scatter(s2.wlist,cs2,label='Scattering (new) model',marker='+')
    
    #NOTE: do not divide by x, this is a histogram already, not a probability in dlog space
    ch=np.cumsum(h)
    ch/=ch[-1]
    plt.plot(plotb,ch,label='cumulative CHIME scatter + width model',linestyle='--')
    
    
    plt.ylabel('cumulative probabiility')
    
    plt.xlim(1e-2,1e3)
    plt.tight_layout()
    plt.savefig('model_comparison.pdf')
    plt.close()
    
main()
