"""
Script showing how to use zDM as priors for CRAFT
host galaxy magnitudes.

It requirses the FRB and astropath modules to be installed.
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
from zdm import path_interface as pi
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading
from astropath.priors import load_std_priors
from astropath.path  import PATH



def main():
    """
    Top-level functionality to loop over all ice FRBs
    and all permutations of uncertainty
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
    model = pi.host_model()
    name='CRAFT_ICS_1300'
    ss,gs = loading.surveys_and_grids(survey_names=[name])
    g = gs[0]
    model.init_zmapping(g.zvals)
    
    # do this only for a particular FRB
    # it gives a prior on apparent magnitude and pz
    #AppMagPriors,pz = model.get_posterior(g,DMlist)
    
    for frb in frblist:
        # interates over the FRBs. "Do FRB"
        # P_O is the prior for each galaxy
        # P_Ox is the posterior
        # P_Ux is the posterior for it being unobserved
        # mags is the list of galaxy magnitudes
        P_O,P_Ox,P_Ux,mags = do_frb(frb)
        print(P_O,P_Ox,P_Ux,mags)
        exit()
        
        
def do_frb(name,PU=0.1):
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
    
    indices = np.argsort(P_Ox)
    return P_O[indices],P_Ox[indices],P_U,mags[indices]
    
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
    
    
