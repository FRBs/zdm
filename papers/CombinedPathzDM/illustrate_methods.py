""" 
This script uses CRAFT surveys to illustrate how likelihoods combine
using PATH and zDM simultaneously

We walk through the calculation, generating step-by-step plots,
to illustrate the method.

Output goes in "illustrations/"
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
from zdm import states

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import importlib.resources as resources


import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    # set op directory
    opdir="Illustrations/"
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
    names=['CRAFT_ICS_892']#,'CRAFT_ICS_1300','CRAFT_ICS_1632']
    
    
    # setting ptauw to True takes much longer!
    survey_dict = {}
    survey_dict["WMETHOD"] = 3
    
    ndm=1400
    nz=500
    
    # simple method - quick
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state,sdir=sdir,survey_dict = None,
                                    ndm=ndm, nz=nz) 
    ifrb=3
    
    # makes p(z) plot
    if False:
        # longer method - takes a *long* time!
        ss2,gs2 = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state2,sdir=sdir,survey_dict = survey_dict,
                                    ndm=ndm, nz=nz) 
        
        glist = [gs[0],gs2[0]]
        slist = [ss[0],ss2[0]]
        
        make_pz_plot(glist,slist,ifrb)
    
    # now creates a model and calculates to p(m|z) distribution
    model = opt.marnoch_model()
    g=gs[0]
    s=ss[0]
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
    plt.savefig("pmr.png")
    plt.close()
    
def make_pz_plot(glist,slist,ifrb):
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
        pzsnrbwgdm = PATH_OP["psnrbwgzdm"] * PATH_OP["pzgdm"] #dimensions: NZ x NFRB
        # and also p(snr,b,w|dm) = \int p(z,snr,b,w|DM) dz ....(3)
        psnrbwgdm = np.sum(pzsnrbwgdm,axis=0) # sums over z-axis. #dimensions: NFRB
        
        # hence, we find from (1) that p(z|snr,b,w,DM) = p(z,snr,b,w|DM) / p(snr,b,w|DM) ...(4)
        pzgsnrbwdm = pzsnrbwgdm/psnrbwgdm
        
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
    
        
    
main()
