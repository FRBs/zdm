""" 
This script uses CRAFT surveys to illustrate how the analysis works
for 190611

Output goes in "190611/"

Most routines are very similar to those from illustrate_methods
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
    
    
    # makes p(z) plot
    if False:
        # longer method - takes a *long* time!
        ss2,gs2 = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state2,sdir=sdir,survey_dict = survey_dict,
                                    ndm=ndm, nz=nz) 
        
        g = gs2[0]
        s = ss2[0]
    else:
        g = gs[0]
        s = ss[0]
    
    maglist = [23.03,26.4]
    make_pz_plot(g,s,ifrb,opdir,maglist)
    exit()
    if False:
        make_pm_plot(g,s,ifrb,opdir)
    
    if False:
        make_pzgmr_plot(g,s,ifrb,opdir,Field=True)
    
    if False:
        make_pzgu_plot(g,s,ifrb,opdir,surveys,means,stds)
    
    if True:
        model = opt.marnoch_model()
        w = opt.model_wrapper(model,g.zvals)
        lltot,results = it.get_joint_path_zdm_likelihoods(g,s,w,pwb=True,return_all=True)

def make_pz_plot(g,s,ifrb,opdir,maglist):
    """
    makes an example p(z) distribution for a given frb
    """
    model = opt.marnoch_model()
    w = opt.model_wrapper(model,g.zvals)
    plt.figure()
    
    #PATH_OP = it.calc_likelihoods_1D(g, s, norm=True, psnr=True, dolist=0, Pn=True, ptauw=False, pwb=True,PATH=True)
    ll,OP = it.get_joint_path_zdm_likelihoods(g, s, w, norm=True, psnr=True, Pn=False,
                                    pdmz=True, pNreps=False, ptauw=False, pwb=True,
                                    return_all=True)
    
    
    ## Plots of P(z|xrad) ##
    zdm_op = OP["zdm_s"]
    path_op = OP["path_s"]
    keys_list = list(path_op)
    
    # get the jth PATH_OP where galaxy ifrb was found
    jpath = np.where(path_op["frblist"] == "20190611B")[0][0]
    
    pzgxrad = zdm_op["pzgxrad"][:,ifrb]
    ix = np.argmax(pzgxrad)
    print("Peak p(z) based on radio properties is ",g.zvals[ix])
    
    print("Probabilities of host vs field for lead galaxy are ",path_op["pz"][jpath],path_op["pf"][jpath])
    print("List of candidate p(x|O) are ",path_op["PxO"][jpath])
    print("List of p(O) are ",path_op["PO"][jpath])
    print("List of Pm are ",path_op["Pm"][jpath])
    print("List of rhom are ",path_op["rhom"][jpath])
    print("Implied P(O) is ",path_op["PO"][jpath]*path_op["rhom"][jpath])
    print("Mag list is m is ",path_op["ObsMags"][jpath])
    print("ll of hosts is ",path_op["ll_hosts"][jpath])
    norm = np.sum(path_op["ll_hosts"][jpath]) + path_op["PU"][jpath]
    print("ll of hosts is ",path_op["PU"][jpath]/norm,path_op["ll_hosts"][jpath]/norm)
    
    from astropath import chance
    
    # separates the priors into driver_sigma and p(O)
    mags = path_op["ObsMags"][jpath]
    rhos = chance.differential_driver_sigma(mags)
    print("Driver sigma prior is ",rhos)
    rhos[:] = 1.
    POs = w.path_raw_prior_Oi(mags,None,rhos)
    print("Pure optical model prior is ",POs)
    
    dz = g.zvals[1]-g.zvals[0]
    pzgxrad /= dz
    plt.figure()
    plt.plot(g.zvals,pzgxrad,label="$P(z|{\\bf x}_{\\rm rad})$",linestyle="-",linewidth=3)
    
    # 
    DMEG = s.DMEGs[ifrb]
    w.init_path_raw_prior_Oi(DMEG,pz=pzgxrad)
    
    # construct field galaxy estimatior of p_f(z|m)
    field = opt.Field()
    
    for im,m in enumerate(maglist):
        pz = w.get_pz_g_mr(m)
        pz /= dz
        plt.plot(g.zvals,pz,label="$P(z|{\\bf x}_{\\rm rad},m="+str(m)[:4]+")$",linewidth = 2-im)
        
        if im == 0:
            ix = np.where(g.zvals < 0.378)[0][-1]
            yval = pz[ix]
            plt.plot([0.378,0.378],[0,yval],color=plt.gca().lines[-1].get_color(),linestyle=":",linewidth = 2-im)
        
        # field galaxies
        pf = field.get_pzgm(m,g.zvals)
        # normalises to sum to unity over the array
        pf /= np.sum(pf)
        # normalises such that \int p(f) dz = 1
        pf /= dz
        
        plt.plot(g.zvals,pf,label="$P_F(z|,m="+str(m)[:4]+")$",color=plt.gca().lines[-1].get_color(),linestyle="--",linewidth = 2-im)
        
    
            
    #### plot of p(z|mag) ####
    print("Likelihoods for each galaxy are ",)
    
    plt.xlabel("z")
    plt.ylabel("P(z)")
    plt.xlim(0,0.5)
    plt.ylim(0,10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_190611.png")
    plt.close()
    
    plt.figure()
    for im,m in enumerate(maglist):
        pf = field.get_pzgm(m,g.zvals)
        pf /= np.sum(pf)
        plt.plot(g.zvals,np.cumsum(pf),label="$P_F(z|,m="+str(m)[:4]+")$")
    plt.xlabel("z")
    plt.ylabel("P(z)")
    plt.xlim(0,5)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_field_cumsum.png")
    plt.close()
    
opdir="190611/"
ifrb=4
survey="CRAFT_ICS_1300"
main(opdir,ifrb,survey)
