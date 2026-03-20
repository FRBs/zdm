""" 
This script uses CRAFT surveys to show how all the components
of the likelihood calculation are unpacked.

It uses calls to calc_likelihoods[1/2]D with 
"""
import os

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

def main():
    
    # in case you wish to switch to another output directory
    name="ASKAP"
    opdir=name+"/"
    
    # load states from Hoffman et al 2025
    state = states.load_state("HoffmannEmin25",scat="updated",rep=None)
    
    # add estimation from ptauw
    state.scat.Sbackproject=True
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('data/Surveys')
    names=['CRAFT_ICS_892']#,'CRAFT_ICS_1300','CRAFT_ICS_1632']
    
    # set this to true to calculate p(twau,w). However, this takes quite some time to evaluate - it's slow!
    # That's because of internal arrays that must be built up by the survey
    ptauw=False
    # setting ptauw to True takes much longer!
    if ptauw:
        # ensure we have z-dependent scattering covered
        survey_dict = {}
        survey_dict["WMETHOD"] = 3
        # make this smaller to reduce time
        ndm=100
        nz=100
    else:
        survey_dict=None
        ndm=1400
        nz=500
    
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state,sdir=sdir,survey_dict = survey_dict,
                                    ndm=ndm, nz=nz) 
    
    g=gs[0]
    s=ss[0]
    
    ############# dolist = 0 ##########
    print("####### DOLIST = 0: Total likelihoods ##########")
    # simple illustration of "dolist=0" - returns only the total log likelihoods
    llsum2 = it.calc_likelihoods_2D(g, s, norm=True, psnr=True, dolist=0, Pn=True, ptauw=ptauw, pwb=True)
    llsum1 = it.calc_likelihoods_1D(g, s, norm=True, psnr=True, dolist=0, Pn=False, ptauw=ptauw, pwb=True)
    
    lltotal = llsum2+llsum1
    print("1D and 2D likelihoods ",llsum1,llsum2," sum to ",lltotal,"\n\n\n\n")
    
    ############# dolist = 1 ##########
    # calculate 2D likelihoods from this survey
    # illustrates three different values for dolist
    #llsum = it.calc_likelihoods_2D(g, s, norm=True, psnr=True, dolist=0, Pn=True, ptauw=ptauw, pwb=True)
    
    
    print("\n######## DOLIST = 1: component likelihoods #########")
    
    print("\n#### checking 2D likelihoods ####")
    
    # generates list of likelihoods for all components
    llsum,lllist = it.calc_likelihoods_2D(g, s, norm=True, psnr=True, dolist=1, Pn=True, ptauw=ptauw, pwb=True)
    
    # performs some checks
    print("Check p(DM,z), ",lllist["pzDM"]["pzDM"]," = p(z|DM)p(DM) ",lllist["pzDM"]["pdmgz"]+lllist["pzDM"]["pz"])
    print("Check p(DM,z), ",lllist["pzDM"]["pzDM"]," = p(DM|z)p(z) ",lllist["pzDM"]["pzgdm"]+lllist["pzDM"]["pdm"])
    
    # checks regarding psnr, beam, and width
    print("Check p(SNR,b,w) = p(SNR|bw)*p(bw)",lllist["pbw"]["psnrbw"],lllist["pbw"]["psnr_gbw"]+lllist["pbw"]["pbw"])
    print("Check p(b,w) = p(b|w)*p(w) = p(w|b)p(b)",lllist["pbw"]["pbw"],lllist["pbw"]["pwgb"]+lllist["pbw"]["pb"],
        lllist["pbw"]["pbgw"]+lllist["pbw"]["pw"])
    
    if ptauw:
        print("Check the llsum = p(tau|w) p(snr|b,w,z,DM) p(b|w,z,DM) p(w|z,DM) p(z|DM) p(DM) p(N)")
        print(llsum,lllist["pzDM"]["pzgdm"]+lllist["pzDM"]["pdm"]+lllist["pbw"]["psnr_gbw"]+lllist["pbw"]["pwgb"]+lllist["pbw"]["pb"]
                +lllist["ptauw"]["pbar"])
    else:
        print("Check the llsum = p(snr|b,w,z,DM) p(b|w,z,DM) p(w|z,DM) p(z|DM) p(DM) p(N))")
        print(llsum,lllist["pzDM"]["pzgdm"]+lllist["pzDM"]["pdm"]+lllist["pbw"]["psnr_gbw"]+lllist["pbw"]["pwgb"]+lllist["pbw"]["pb"]
                +lllist["pN"])
    
    print("\n\n#### checking 1D likelihoods ####")
    # generates list of likelihoods for all components
    llsum,lllist = it.calc_likelihoods_1D(g, s, norm=True, psnr=True, dolist=1, Pn=True, ptauw=ptauw, pwb=True)
    
    # performs some checks
    # checks regarding psnr, beam, and width
    print("Check p(SNR,b,w) = p(SNR|bw)*p(bw)",lllist["pbw"]["psnrbw"],lllist["pbw"]["psnr_gbw"]+lllist["pbw"]["pbw"])
    print("Check p(b,w) = p(b|w)*p(w) = p(w|b)p(b)",lllist["pbw"]["pbw"],lllist["pbw"]["pwgb"]+lllist["pbw"]["pb"],
        lllist["pbw"]["pbgw"]+lllist["pbw"]["pw"])
    
    if ptauw:
        print("Check the llsum = p(tau|w) p(snr|b,w,DM) p(b|w,DM) p(w|DM) p(DM) p(N)")
        print(llsum,lllist["pzDM"]["pdm"]+lllist["pbw"]["psnr_gbw"]+lllist["pbw"]["pwgb"]+lllist["pbw"]["pb"]
                +lllist["ptauw"]+lllist["pbar"])
    else:
        print("Check the llsum = p(snr|b,w,DM) p(b|w,DM) p(w|DM) p(DM) p(N))")
        print(llsum,lllist["pzDM"]["pdm"]+lllist["pbw"]["psnr_gbw"]+lllist["pbw"]["pwgb"]+lllist["pbw"]["pb"]
                +lllist["pN"])
    
    
    
    
    ############# dolist = 1 ##########
    # calculate 2D likelihoods from this survey
    # illustrates three different values for dolist
    #llsum = it.calc_likelihoods_2D(g, s, norm=True, psnr=True, dolist=0, Pn=True, ptauw=ptauw, pwb=True)
    
    
    print("\n######## DOLIST = 2: individual FRB likelihoods #########")
    
    print("\n#### checking 2D likelihoods ####")
    
    # generates list of likelihoods for all components
    llsum,lllist,longlist = it.calc_likelihoods_2D(g, s, norm=True, psnr=True, dolist=2, Pn=True, ptauw=ptauw, pwb=True)
    
    print("\nCheck p(DM,z) = p(DM|z)p(z) ")
    for i,pzdm in enumerate(longlist["pzDM"]["pzDM"]):
        print(i,pzdm,longlist["pzDM"]["pdmgz"][i]+longlist["pzDM"]["pz"][i])
    
    
    print("\n\nCheck p(DM,z) = p(z|DM)p(DM) ")
    for i,pzdm in enumerate(longlist["pzDM"]["pzDM"]):
        print(i,pzdm,longlist["pzDM"]["pzgdm"][i]+longlist["pzDM"]["pdm"][i])
    
    
    
    # checks regarding psnr, beam, and width
    print("\n\nCheck p(SNR,b,w) = p(SNR|bw)*p(bw)")
    print("These will be slightly out, because all three are calculated by weighting over b,w,")
    print("    and \sum_bw psnrbw *w(b,w) != (\sum_bw psnrgbw*w(b,w))*(\sum_bw p(bw) w(b,w,)")
    for i,psnrbw in enumerate(longlist["pbw"]["psnrbw"]):
        print(longlist["pbw"]["psnrbw"][i],longlist["pbw"]["psnr_gbw"][i]+longlist["pbw"]["pbw"][i])
        
    
main()
