""" 
This script uses CRAFT surveys to illustrate how likelihoods combine
using PATH and zDM simultaneously

We walk through the calculation, generating step-by-step plots,
to illustrate the method.

Output goes in "illustrations/"

These plots appear in Section 2 of the paper
"""
import os
import time

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from zdm import optical as opt
from zdm import optical_params as op
from zdm import states

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import importlib.resources as resources

from astropath import chance

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main(opdir,iFRB,survey):
    """
    Creates a bunch of p(z,DM) and PATH plots for a given FRB
    
    Args:
        opdir [string]: output directory for plots
        iFRB [int]: nth FRB in survey (0,1,2 etc) to analyse
        survey [string]: survey file name
    """
    
    # set op directory
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    
    # set this to true to calculate p(twau,w). However, this takes quite some time to evaluate - it's slow!
    # That's because of internal arrays that must be built up by the survey
    
    # load states from Hoffman et al 2025
    state = states.load_state("HoffmannHalo25",scat="orig",rep=None)
    state2 = states.load_state("HoffmannHalo25",scat="updated",rep=None)
    
    
    # add estimation from ptauw
    state2.scat.Sbackproject=True
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('data/Surveys')
    #names=['CRAFT_ICS_892']#,'CRAFT_ICS_1300','CRAFT_ICS_1632']
    names=[survey]
    
    # setting ptauw to True takes much longer!
    survey_dict = {}
    survey_dict["WMETHOD"] = 3
    
    ndm=1400
    nz=500
    
    # simple method - quick
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state,sdir=sdir,survey_dict = None,
                                    ndm=ndm, nz=nz) 
    ifrb=4
    
    g=gs[0]
    s=ss[0]
    
    
    surveys = ["VLT/FORS2","DESI LIS","PanSTARRS"]
    means = [26.2,24,21.8]
    stds = [0.34,0.55,0.54]
    
    if True:
        test_field(opdir)
    
    # makes p(z) plot
    if True:
        # longer method - takes a *long* time!
        ss2,gs2 = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state2,sdir=sdir,survey_dict = survey_dict,
                                    ndm=ndm, nz=nz) 
        
        glist = [gs[0],gs2[0]]
        slist = [ss[0],ss2[0]]
        
        make_pz_plot(glist,slist,ifrb,opdir)
    
    if True:
        make_pm_plot(g,s,ifrb,opdir)
    
    if True:
        make_pzgmr_plot(g,s,ifrb,opdir,Field=True)
    
    if True:
        make_pzgu_plot(g,s,ifrb,opdir,surveys,means,stds)
    
    
        
    if True:
        model = opt.marnoch_model()
        w = opt.model_wrapper(model,g.zvals)
        lltot,results = it.get_joint_path_zdm_likelihoods(g,s,w,pwb=True,return_all=True)

    
def test_field(opdir):
    """
    Makes a plot of p(m|z)
    """
    field = opt.Field()
    
    
    # test calculation of how the total volume of the Universe,
    # dV/dz, increases with redshift.
    plt.figure()
    plt.plot(field.zvals,field.volumes)
    plt.xlabel("z")
    plt.ylabel("V(z)")
    plt.tight_layout()
    #plt.yscale("log")
    plt.savefig(opdir+"volume.png")
    plt.close()
    
    # internally, "field" assumes that the gaalxy count stays constant,
    # hence Driver et al is somply  \int dz volume(z) * p(m|z) dz
    # and we can then use this to get p(z|m)
    
    # plots pm vs driver
    rmags = field.rmags
    
    # this is Mpc^3 per steradian per dz
    fpm = field.pm
    
    #this is units per square arcsec per magnitude
    dsigma = chance.differential_driver_sigma(rmags)
    
    
    
    plt.figure()
    plt.plot(rmags,fpm/3e13,label="$\\int_0^2 P(m_r|z) V(z) dz$")
    plt.plot(rmags,dsigma,label="Driver et al. 2016")
    #field.extrapolate_p_mr_field(200)
    #plt.plot(rmags,field.pm,label="Loudas25 $\\int_0^4 P(m_r|z) V(z) dz$")
    #field.extrapolate_p_mr_field(400)
    #plt.plot(rmags,field.pm,label="Loudas25 $\\int_0^{8} P(m_r|z) V(z) dz$")
    
    #plt.plot(rmags,field.pmrz[-1,:]*100,label="Redshift 2.0")
    #plt.plot(rmags,field.pmrz[-100,:]*100,label="Redshift 1.9")
    plt.xlabel("$m_r$")
    plt.ylabel("$P_F(m_r)$ [mag$^{-1}$ arcsec$^{-2}$]")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(opdir+"pm.png")
    plt.close()
    
    mvals=[15,18,21,24]
    for i,m in enumerate(mvals):
        pzf = field.get_pzgm(m)
        pzf /= np.sum(pzf)* field.dz
        plt.plot(field.zvals,pzf,label="$P_F(z|m_r = "+str(m)+")$")
    plt.xlabel("$z$")
    plt.ylabel("$P(z)$")
    plt.ylim(0,7)
    plt.xlim(0,2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pzgm_field.png")
    plt.close()

def make_pzgu_plot(g,s,ifrb,opdir,surveys,means,widths,Field=False):
    """
    makes a plot of p(z) given mr
    
    Args:
        g: grid object
        s: survey object corresponding to the grid
        ifrb: which FRB in the survey to use
        surveys: list of strigns giving optical names
        means: array of means of P(O|m) of optical images
        widths: array of widths of P(O|m) of optical images
        
    """
    plt.figure()
    plt.xlim(0,1)
    plt.ylim(0,4)
    styles = ["-","--",":","-."]
    for i,label in enumerate(surveys):
        opstate = op.OpticalState()
        opstate.id.pU_mean = means[i]
        opstate.id.pU_width = widths[i]
        model = opt.marnoch_model(opstate)
        wrapper = opt.model_wrapper(model,g.zvals)
        DMEG = s.DMEGs[ifrb]
        wrapper.init_path_raw_prior_Oi(DMEG,g)
        #mr =  #  from most likely host galaxy of FRB[3]
        pzgu = wrapper.get_pz_g_U()
        print("For image ",label," p(U) is ",wrapper.estimate_unseen_prior())
        
        pzgu /= (g.zvals[1]-g.zvals[0])
        plt.plot(g.zvals, pzgu, label=label,linestyle = styles[i%4])
        
    plt.xlabel("$z$")
    plt.ylabel("$P(z|U)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pzgu.png")
    plt.close()

def make_pzgmr_plot(g,s,ifrb,opdir,Field=False):
    """
    makes a plot of p(z) given mr
    
    Args:
        g: grid object
        s: survey object corresponding to the grid
        ifrb: which FRB in the survey to use
        
    """
    mr = 19.63 # hard-coded, taken from previous PATH work
    model = opt.marnoch_model()
    wrapper = opt.model_wrapper(model,g.zvals)
    DMEG = s.DMEGs[ifrb]
    wrapper.init_path_raw_prior_Oi(DMEG,g)
    #mr =  #  from most likely host galaxy of FRB[3]
    pz = wrapper.get_pz_g_mr(mr)
    
    plt.figure()
    plt.xlim(0,1)
    plt.ylim(0,3)
    pz /= (g.zvals[1]-g.zvals[0])
    plt.plot(g.zvals, pz, label="$P(z|{\\bf{x_{\\bf rad}}}, m_r = "+str(mr)+")$")
    
    
    
    if Field:
        field = opt.Field()
        pzf = field.get_pzgm(mr)
        plt.plot(field.zvals,pzf,label="$P_F(z, m_r = "+str(mr)+")$")
    
    plt.xlabel("$z$")
    plt.ylabel("$P(z)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pzgmr.png")
    plt.close()
    
def make_pm_plot(g,s,ifrb,opdir):
    """
    creates a model and calculates to p(m|z) distribution - which gets integrated over
    """
    model = opt.marnoch_model()
    
    wrapper = opt.model_wrapper(model,g.zvals)
    DMEG = s.DMEGs[ifrb]
    print("DMEG is ",DMEG)
    wrapper.init_path_raw_prior_Oi(DMEG,g)
    
    plt.figure()
    dam = wrapper.AppMags[1]-wrapper.AppMags[0]
    
    toplot = wrapper.raw_priors / dam
    plt.plot(wrapper.AppMags,toplot,label="$P(m_r|{\\rm DM_{\\rm EG}}="+str(DMEG)+")$")
    toplot = wrapper.priors / dam
    plt.plot(wrapper.AppMags,toplot,label="$P(O|{\\rm DM_{\\rm EG}}="+str(DMEG)+")$",linestyle="--")
    plt.xlabel("$m_r$")
    plt.ylabel("$P(m_r)$")
    plt.ylim(0,0.2)
    plt.xlim(10,30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pmr.png")
    plt.close()
    
    return wrapper
    
def make_pz_plot(glist,slist,ifrb,opdir):
    """
    makes an example p(z) distribution for a given frb
    """
    
    plt.figure()
    
    for ig,g in enumerate(glist):
        s=slist[ig]
    
        # get z-dependent probabilities
        PATH_OP = it.calc_likelihoods_1D(g, s, norm=True, psnr=True, dolist=0, Pn=True, ptauw=False, pwb=True,PATH=True)
        
        # remind ourselves of the info we have
        #print(PATH_OP.keys())
        # should read dict_keys(['pdm', 'pzgdm', 'pn', 'psnrbwgzdm'])
    
        # perform some calculation.
        # What we want is p(z|snr,b,w,DM), p(snr,b,w|DM), and p(DM)
        # What we have is p(snr,b,w|z,DM), p(z|DM), and p(DM)
        # We begin noting that p(z,snr,b,w|DM) = p(z|snr,b,w,DM) * p(snr,b,w|DM) ... (1)
        # We calculate p(z,snr,b,w|DM) = p(snr,b,w|z,DM) * p(z|DM) ... (2)
        #pzsnrbwgdm = PATH_OP["psnrbwgzdm"] * PATH_OP["pzgdm"] #dimensions: NZ x NFRB
        # and also p(snr,b,w|dm) = \int p(z,snr,b,w|DM) dz ....(3)
        #psnrbwgdm = np.sum(pzsnrbwgdm,axis=0) # sums over z-axis. #dimensions: NFRB
        
        # hence, we find from (1) that p(z|snr,b,w,DM) = p(z,snr,b,w|DM) / p(snr,b,w|DM) ...(4)
        #pzgsnrbwdm = pzsnrbwgdm/psnrbwgdm
        
        # all the above now in calc_likelihoods 1D
        pzgsnrbwdm = PATH_OP["pzgsnrbwdm"]
        
        # note that this is simply a normalisation factor off the initial product
        
        
        DMEG = int(s.DMEGs[ifrb])
        
        dz = g.zvals[1]-g.zvals[0]
        
        toplot = PATH_OP["pzgdm"][:,ifrb] / np.sum(PATH_OP["pzgdm"][:,ifrb])/dz
        if ig==0:
            l1,=plt.plot(g.zvals,toplot,label="$P(z|{\\rm DM_{\\rm EG}} = $"+str(DMEG)+"$)$",linestyle="--",linewidth=2-ig)
        else:
            plt.plot(g.zvals,toplot,color=l1.get_color(),linestyle="--",linewidth=2-ig)
            
        toplot = PATH_OP["psnrbwgzdm"][:,ifrb] / np.sum(PATH_OP["psnrbwgzdm"][:,ifrb])/dz
        if ig==0:
            l2,=plt.plot(g.zvals,toplot,label="$P(s,B,w|z,{\\rm DM_{\\rm EG}} = $"+str(DMEG)+"$)$",linestyle="-.",linewidth=2-ig)
        else:
            plt.plot(g.zvals,toplot,color=l2.get_color(),linestyle="-.",linewidth=2-ig)
        #combined = PATH_OP["pzgdm"][:,i] * PATH_OP["psnrbwgzdm"][:,i]
        #toplot = combined / np.sum(combined)/dz
        #plt.plot(g.zvals,toplot,label="$P(s,B,w,z|{\\rm DM_{\\rm EG}} = $"+str(DMEG)+"$)$",linestyle=":")
        
        
        toplot = pzgsnrbwdm[:,ifrb] / np.sum(pzgsnrbwdm[:,ifrb])/dz
        if ig==0:
            l3,=plt.plot(g.zvals,toplot,label="$P(z|s,B,w,{\\rm DM_{\\rm EG}} = $"+str(DMEG)+"$)$",linestyle="-",linewidth=2-ig)
        else:
            l3,=plt.plot(g.zvals,toplot,color=l3.get_color(),linestyle="-",linewidth=2-ig)
    plt.xlabel("z")
    plt.ylabel("P(z)")
    plt.xlim(0,1.5)
    plt.ylim(0,2.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz.png")
    plt.close()
    
    
# inputs used to create plots for Section 2 of paper
opdir="Illustrations/"
ifrb=3
survey="CRAFT_ICS_892"
main(opdir,ifrb,survey)
