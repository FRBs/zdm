"""
Script showing how to use zDM as priors for CRAFT
host galaxy magnitudes.

It requirses the FRB and astropath modules to be installed.

This does NOT include optimisation of any parameters
"""

#stabndard Python imports
import pandas
from importlib import resources
import os
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord

from matplotlib import pyplot as plt

# imports from the "FRB" series

from frb.frb import FRB
from zdm import optical as opt
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading
from astropath.priors import load_std_priors
from astropath.path  import PATH
import astropath.priors as pathpriors


def main():
    """
    Loops over all ICS FRBs
    """
    
    ######### List of all ICS FRBs for which we can run PATH #######
    # hard-coded list of FRBs with PATH data in ice paper
    frblist=['FRB20180924B','FRB20181112A','FRB20190102C','FRB20190608B',
        'FRB20190611B','FRB20190711A','FRB20190714A','FRB20191001A',
        'FRB20191228A','FRB20200430A','FRB20200906A','FRB20210117A',
        'FRB20210320C','FRB20210807D','FRB20211127I','FRB20211203C',
        'FRB20211212A','FRB20220105A','FRB20220501C',
        'FRB20220610A','FRB20220725A','FRB20220918A',
        'FRB20221106A','FRB20230526A','FRB20230708A', 
        'FRB20230731A','FRB20230902A','FRB20231226A','FRB20240201A',
        'FRB20240210A','FRB20240304A','FRB20240310A']
    
    NFRB = len(frblist)
    
    # here is where I should initialise a zDM grid
    state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    model = opt.host_model()
    name='CRAFT_ICS_1300'
    ss,gs = loading.surveys_and_grids(survey_names=[name])
    g = gs[0]
    s = ss[0]
    # must be done once for any fixed zvals
    model.init_zmapping(g.zvals)
    
    # do this only for a particular FRB
    # it gives a prior on apparent magnitude and pz
    #AppMagPriors,pz = model.get_posterior(g,DMlist)
    
    # do this once per "model" objects
    pathpriors.USR_raw_prior_Oi = model.path_raw_prior_Oi
    
    allmags = None
    allPOx = None
    
    for frb in frblist:
        # interates over the FRBs. "Do FRB"
        # P_O is the prior for each galaxy
        # P_Ox is the posterior
        # P_Ux is the posterior for it being unobserved
        # mags is the list of galaxy magnitudes
        
        # determines if this FRB was seen by the survey, and
        # if so, what its DMEG is
        imatch = matchFRB(frb,s)
        if imatch is None:
            print("Could not find ",frb," in survey")
            continue
        
        DMEG = s.DMEGs[imatch]
        
        # original calculation
        P_O1,P_Ox1,P_Ux1,mags1 = do_frb(frb,model,usemodel=False,PU=0.1)
        
        model.init_path_raw_prior_Oi(DMEG,g)
        PU = model.estimate_unseen_prior(mag_limit=26) # might not be correct
        P_O2,P_Ox2,P_Ux2,mags2 = do_frb(frb,model,usemodel=True,PU = PU)
        
        if False:
            # compares outcomes
            print("FRB ",frb)
            print(" m_r               P_O: old               new               P_Ox: old               new")
            for i,P_O in enumerate(P_O1):
                print(i,mags1[i],P_O1[i],P_O2[i],P_Ox1[i],P_Ox2[i])
            print("\n")
        
        # keep cumulative histogram of posterior magnitude distributions
        #allmags.append(mags2)
        #allPOx.append(P_Ox2)
        mags2 = np.array(mags2)
        
        if allmags is None:
            allmags = mags2
            allPOx = P_Ox2
        else:
            allmags = np.append(allmags,mags2)
            allPOx = np.append(allPOx,P_Ox2)
    
    Nbins = int(model.Appmax - model.Appmin)+1
    bins = np.linspace(model.Appmin,model.Appmax,Nbins)
    plt.figure()
    plt.hist(allmags,weights = allPOx, bins = bins,label="Posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig("posterior_pOx.png")
    plt.close()
    
def do_frb(name,model,PU=0.1,usemodel = False, sort = False):
    """
    evaluates PATH on an FRB
    
    absolute [bool]: if True, treats rel_error as an absolute value
        in arcseconds
    """
    ######### Loads FRB, and modifes properties #########
    my_frb = FRB.by_name(name)
    
    # do we even still need this? I guess not, but will keep it here just in case
    my_frb.set_ee(my_frb.sig_a,my_frb.sig_b,my_frb.eellipse['theta'],
                my_frb.eellipse['cl'],True)
    
    # reads in galaxy info
    ppath = os.path.join(resources.files('frb'), 'data', 'Galaxies', 'PATH')
    pfile = os.path.join(ppath, f'{my_frb.frb_name}_PATH.csv')
    ptbl = pandas.read_csv(pfile)
    
    # Load prior
    priors = load_std_priors()
    prior = priors['adopted'] # Default
    
    theta_new = dict(method='exp', 
                    max=priors['adopted']['theta']['max'], 
                    scale=0.5)
    prior['theta'] = theta_new
    
    # change this to something depending on the FRB DM
    prior['U']=PU
    
    candidates = ptbl[['ang_size', 'mag', 'ra', 'dec', 'separation']]
    
    this_path = PATH()
    this_path.init_candidates(candidates.ra.values,
                         candidates.dec.values,
                         candidates.ang_size.values,
                         mag=candidates.mag.values)
    this_path.frb = my_frb
    
    frb_eellipse = dict(a=my_frb.sig_a,
                    b=my_frb.sig_b,
                    theta=my_frb.eellipse['theta'])
    
    this_path.init_localization('eellipse', 
                            center_coord=this_path.frb.coord,
                            eellipse=frb_eellipse)
    
    # this results in a prior which is uniform in log space
    # when summed over all galaxies with the same magnitude
    if usemodel:
        this_path.init_cand_prior('user', P_U=prior['U'])
    else:
        this_path.init_cand_prior('inverse', P_U=prior['U'])
        
    # this is for the offset
    this_path.init_theta_prior(prior['theta']['method'], 
                            prior['theta']['max'],
                            prior['theta']['scale'])
    
    P_O=this_path.calc_priors() 
    
    # Calculate p(O_i|x)
    debug = True
    P_Ox,P_U = this_path.calc_posteriors('fixed', 
                         box_hwidth=10., 
                         max_radius=10., 
                         debug=debug)
    mags = candidates['mag']
    
    if sort:
        indices = np.argsort(P_Ox)
        P_O = P_O[indices]
        P_Ox = P_Ox[indices]
        mags = mags[indices]
    
    return P_O,P_Ox,P_U,mags

 
def simplify_name(TNSname):
    """
    Simplifies an FRB name to basics
    """
    # reduces all FRBs to six integers
    
    if TNSname[0:3] == "FRB":
        TNSname = TNSname[3:]
    
    if len(TNSname) == 9:
        name = TNSname[2:-1]
    elif len(TNSname) == 8:
        name = TNSname[2:]
    elif len(TNSname) == 7:
        name = TNSname[:-1]
    elif len(TNSname) == 6:
        name = TNSname
    else:
        print("Do not know how to process ",TNSname)
    return name
       
def matchFRB(TNSname,survey):
    """
    Gets the FRB id from the survey list
    Returns None if not in the survey
    """
    
    name = simplify_name(TNSname)
    match = None
    for i,frb in enumerate(survey.frbs["TNS"]):
        if name == simplify_name(frb):
            match = i
            break
    return match


def plot_frb(name,ralist,declist,plist,opfile):
    """
    does an frb
    
    absolute [bool]: if True, treats rel_error as an absolute value
        in arcseconds
        
    clist: list of astropy coordinates
    plist: list of p(O|x) for candidates hosts
    """
    ######### Loads FRB, and modifes properties #########
    my_frb = FRB.by_name(name)
    
    ppath = os.path.join(resources.files('frb'), 'data', 'Galaxies', 'PATH')
    pfile = os.path.join(ppath, f'{my_frb.frb_name}_PATH.csv')
    ptbl = pandas.read_csv(pfile)
    
    candidates = ptbl[['ang_size', 'mag', 'ra', 'dec', 'separation']]
    
    #raoff=199. + 2910/3600 # -139./3600
    #decoff=-18.8 -139./3600 #+2910/3600
    
    raoff = my_frb.coord.ra.deg
    decoff = my_frb.coord.dec.deg
    
    cosfactor = np.cos(my_frb.coord.dec.rad)
    
    plt.figure()
    plt.xlabel('ra [arcsec] - relative')
    plt.ylabel('dec  [arcsec]  - relative')
    
    
    plt.scatter([(ralist-raoff)*3600*cosfactor],[(declist-decoff)*3600],marker='+',
        c=plist, vmin=0.,vmax=1.,label="Deviated FRB")
    
    
    plt.scatter((candidates['ra']-raoff)*3600*cosfactor,(candidates['dec']-decoff)*3600,
            s=candidates['ang_size']*300, facecolors='none', edgecolors='r',
            label="Candidate host galaxies")
    
    
    # orig scatter plot command
    sc = plt.scatter([(my_frb.coord.ra.deg-raoff)*3600*cosfactor],[(my_frb.coord.dec.deg-decoff)*3600],
        marker='x',label="True FRB")
    plt.colorbar(sc)
    
    for i, ra in enumerate(candidates['ra']):
        ra=(ra-raoff)*3600*cosfactor
        dec=(candidates['dec'][i]-decoff)*3600
        plt.text(ra,dec,str(candidates['ang_size'][i])[0:4])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opfile)
    plt.tight_layout()
    plt.close()

    
main()
    
    
