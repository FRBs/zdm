"""
This file is intended to load a grid, and generate a single
slice of likelihoods through H0

"""
from zdm import iteration as it
from zdm.craco import loading
import numpy as np
import os

def main():
    # create new grids directly from updates state parameters in grid   
    
    # frb_names = ["181112"]
    frb_names = ["181112", "190611", "190711", "191228", "200430", "210117", "210320", "210407", "210912", "220501", "220725", "230526", "230708"]
    # frb_names = ["230526", "230708"]
    # frb_names = ["190711"]
    edir='/fred/oz002/jhoffmann/FRB_library/zdm/zdm/data/Efficiencies/'
    sdir='/fred/oz002/jhoffmann/FRB_library/zdm/zdm/data/Surveys/Hoffmann2023_CRAFT/'
    outdir='../test/'

    print("outdir: " + outdir)

    # Initialise H0 array and save
    H0s = np.linspace(65,75,500)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    np.save(os.path.join(outdir,"H0s.npy"), H0s)

    #===============================================================================
    # # Do single survey
    # s,g = loading.survey_and_grid(survey_name="CRAFT_ICS",sdir=sdir,NFRB=None) 

    # # varies H0
    # llsum=[]
    # for H0 in H0s:
    #     # updates grid to new parameter values
    #     vparams = {}
    #     vparams['H0'] = H0
    #     g.update(vparams)
    #     # returns log-likelihood sum for this survey and grid
    #     llsum.append(get_likelihood(s,g))

    # np.save(os.path.join(outdir,"CRAFT_ICS.npy"), np.array(llsum))

    #===============================================================================
    # Each FRB as individual survey
    for name in frb_names:
        print(name, flush=True)
        # Normal calculation
        s,g = loading.survey_and_grid(survey_name=name,sdir=sdir,NFRB=None) 

        # varies H0
        llsum=[]
        for H0 in H0s:
            # updates grid to new parameter values
            vparams = {}
            vparams['H0'] = H0
            g.update(vparams)
            # returns log-likelihood sum for this survey and grid
            llsum.append(get_likelihood(s,g))

        np.save(os.path.join(outdir,name) + ".npy", np.array(llsum))

        #===============================================================================
        # Calculation with efficiencies
        s_exact,g_exact = loading.survey_and_grid(survey_name=name,sdir=sdir,NFRB=None,edir=edir)

        # varies H0
        llsum_exact=[]

        for H0 in H0s:
            # updates grid to new parameter values
            vparams = {}
            vparams['H0'] = H0
        
            g_exact.update(vparams)
            # returns log-likelihood sum for this survey and grid
            llsum_exact.append(get_likelihood(s_exact,g_exact))
        
        np.save(os.path.join(outdir,name) + "_exact.npy", np.array(llsum_exact))

#===============================================================================
def get_likelihood(s,g,norm=True,Pn=False,psnr=True,dolist=0):
    """
    Returns total ikelihood for a single survey s and grid g
    
    I am turning Pn off now becuse for a single survey it's useless info
    """
    # we only return the total log-likelihood, not separated into components
    
    
    if s.nD==1:
        llsum = it.calc_likelihoods_1D(
            g,s,norm=norm,psnr=psnr,dolist=dolist,Pn=Pn)
    elif s.nD==2:
        llsum = it.calc_likelihoods_2D(
            g,s,norm=norm,psnr=psnr,dolist=dolist,Pn=Pn)
    elif s.nD==3:
        # mixture of 1 and 2D samples. NEVER calculate Pn twice!
        llsum = it.calc_likelihoods_1D(
            g,s,norm=norm,psnr=psnr,dolist=dolist,Pn=Pn)
        # must always have Pn being false for one of these two
        llsum += it.calc_likelihoods_2D(
            g,s,norm=norm,psnr=psnr,dolist=dolist,Pn=False)
    return llsum


main() 
