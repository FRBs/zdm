""" 
This script plots the modelled CHIME efficiency as a function of DM

It creates chime_bias_fit.pdf

"""
import os
from pkg_resources import resource_filename
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io
from zdm import repeat_grid as rep

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp

import matplotlib
import time

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    # plots the polynomial coefficients
    fit_dm_bias()
    
def fit_dm_bias():
    """
    Contains the DM, width, scattering bias models from CHIME
    """
    
    
    ######## PART 1: FITTING TO CHIME DATA ############
    path =  os.path.join(resource_filename('zdm', 'data'), 'Misc')
    data = np.loadtxt(path+"/chime_dm_bias.dat")
    dms = data[:,0]
    bias = data[:,1]
    ldms = np.log(dms)
    fit = sp.interpolate.interp1d(ldms,bias,kind='cubic')
    
    dmvals = np.linspace(dms[0],dms[-1],100)
    ldmvals = np.log(dmvals)
    bias_vals = fit(ldmvals)
    
    
    #does polyfit
    pdmvals = np.linspace(100,10000,201)
    pldmvals = np.log(pdmvals)
    coeffs = np.polyfit(ldms,bias,4)
    print("coeffs are ",coeffs)
    bias_vals2 = np.polyval(coeffs,pldmvals)
    norm = np.max(bias_vals2)
    print("Normalising to unity gives ",norm)
    
    ######## PART 2: NAIVE EFFICIENCY ESTIMATES ############
    
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    load=True
    ldir = 'DMBiasFit/'
    
    if load==False:
        
        state=set_state(chime_response = True)
        t0=time.time()
        name = "CHIME_decbin0"
        sc,gc = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        ec = sc.efficiencies[0,:]
        
        state=set_state(chime_response = False)
        t0=time.time()
        name = "CHIME_decbin0"
        sn,gn = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        en = sn.efficiencies
        
        np.save(ldir + 'ec.npy',ec)
        np.save(ldir + 'en.npy',en)
        np.save(ldir + 'dm.npy',gc.dmvals)
        np.save(ldir + 'wplist.npy',sn.wplist)
        dm = gc.dmvals
        wplist = sn.wplist
        
    else:
        en = np.load(ldir + 'en.npy')
        ec = np.load(ldir + 'ec.npy')
        dm = np.load(ldir + 'dm.npy')
        wplist = np.load(ldir + 'wplist.npy')
    
    NW,NDM = en.shape
    
    mean_eff = np.zeros([NDM])
    for i in np.arange(NW):
        mean_eff += en[i,:] * wplist[i]
        
    
    connect = 1500
    ireweight = np.where(dm > connect)[0]
    reweight = ec[ireweight[0]]/mean_eff[ireweight[0]]
    print("Reweight factor is ",reweight)
    
    modec = np.copy(ec)
    modec[ireweight] = mean_eff[ireweight]*reweight
    
    # approximate fit
    #guess = np.exp(-dm/10000)*1.1
    
    #import scipy
    #r1 = np.where(dm > 2000)[0]
    #r2 = np.where(dm < 3000)[0]
    #r3 = np.intersect1d(r1,r2)
    #p0=[1.1,10000]
    #params, cv = scipy.optimize.curve_fit(MyExp, dm[r3], ec[r3], p0)
    #print('Fitted params are ',params)
    #fit = MyExp(dm,params[0],params[1])
    
    
    
    
    
    
    
    ######## PART 3: PLOTTING ############
    
    plt.figure()
    
    plt.xlabel('DM [pc/cm3]')
    plt.ylabel('Bias [arb. units]')
    plt.xscale('log')
    plt.xlim(1e2,1e4)
    plt.ylim(0.2,1.2)
    
    plt.scatter(dms,bias,label='CHIME data',marker='x',s=15,color='red',zorder=10)
    plt.plot(dmvals,bias_vals,label='cubic spline fit',linewidth=1.5)
    plt.plot(pdmvals,bias_vals2,label='$4^{\\rm th}$ order polyfit',linewidth=1.5)
    plt.plot(pdmvals,bias_vals2/norm,linestyle='--',label='(renormalised)',color=plt.gca().lines[-1].get_color(),linewidth=1.5)
    plt.plot(pdmvals,(bias_vals2/norm)**(2./3.),linestyle=':',label='SNR bias',linewidth=3)
    plt.plot(dm,mean_eff*reweight,label='z-DM',linestyle="-.",color='purple',linewidth=1.5)
    
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(ldir+'chime_bias_fit.pdf')
    plt.close()
    
    # NOTE: the bias is in terms of the number of detected events
    # We must interpret it in terms of S/N bias
    # To do so, we assume an N~SNR^-1.5 relationship
    # However, this is approximate, since the slope of source-counts
    # is DM dependent.




main()
