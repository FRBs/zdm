"""
Runs PATH only, obtaining posteriors for the simulated sample of FRBs

Creates inputs for comparison to this combined analysis, to be read in
using test_Bridget_likelihood.py

Will also write pseudo survey file for latter zdm analysis
"""

import os
import pandas as pd
import importlib.resources as resources
from zdm import optical_numerics as on
import numpy as np
from matplotlib import pyplot as plt

import matplotlib
defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main(outfile,woutfile,N,frbfile,hostfile,CUT=False,prefix="",surveyfile=None,POxcut=0.95):
    """
    Runs original PATH algorithm on FRBs, and outputs FRBs passing POxcut, and weighted by POx
    
    
    args:
        outfile (string):
            name of csv to output FRBs passing POxcut
        
        woutfile (string):
            name of csv to output FRBs weighted by POx
        
        N (int):
            max number of FRBs to run this on
        
        
        frbfile (string):
            csv file output by gen_mc_frbs_w_hosts.py,
            containing fake FRBs
        
        hostfile (string):
            csv file containing assigned hosts from run_assign_host.py
        
        CUT (bool, optional):
            if TRUE, pass FRBs from bright galaxies, assuming they are detected
        
        prefix (string):
            prefix of galaxy/candidate file directory, telling the program for which
            set of outputs from "write_frb_and_cand_files.py" to work with
        
        surveyfile (string, optional):
            if not None, write a fake survey file with this name
        
        POxcut (float):
            value of POx for determing "confidant" hosts
    
    """
    
    
    ### sets system variables to point towards fake FRB data
    galdir = resources.files('zdm').joinpath("../papers/CombinedPathzDM/BridgetMC/"+prefix+"CandidateFiles")
    frbdir = resources.files('zdm').joinpath("../papers/CombinedPathzDM/BridgetMC/"+prefix+"FRBFiles")
    os.environ["ZDM_PATH_FRBDIR"] = str(frbdir)
    os.environ["ZDM_PATH_GALDIR"] = str(galdir)
    
    # generates a list of N FRBs to analyse
    frblist = gen_frb_list(N)
    
    results = on.calc_path_priors(frblist,P_U=0.1,failOK=False,
                                scale=0.5,usemodel=False)
    
    # creates lists summarising the FRBs with highly likely PATH results
    
    truth = pd.read_csv(hostfile)
    zdm = pd.read_csv(frbfile)
    
    allmag=[]
    allpox=[]
    allz=[]
    alli = []
    allseps = []
    allsizes = []
    noti=[]
    rawposts = np.array([])
    rawmags = np.array([])
    rawseps = np.array([])
    rawsizes = np.array([])
    
    for i in np.arange(N):
        # will always find an FRB file by construction
        if results["Ncand"][i] == 0:
            continue
        
        # appends all path posteriors
        if i == 0:
            rawmags = np.array(results["ObsMags"][i])
            rawposts =  np.array(results["POx"][i])
            rawseps =  np.array(results["seps"][i])
            rawsizes =  np.array(results["sizes"][i])
        else:
            rawmags = np.concatenate((rawmags,np.array(results["ObsMags"][i])))
            rawposts = np.concatenate((rawposts,results["POx"][i]))
            rawseps = np.concatenate((rawseps,results["seps"][i]))
            rawsizes = np.concatenate((rawsizes,results["sizes"][i]))
        
        likely = np.where(results["POx"][i] > POxcut)[0]
        FRBfile = str(galdir)+"/FRB"+str(i)+"_PATH.csv"
        
        # tests to see if the true host is even in the FRBfile
        mtrue = truth["mag"][i]
        
        
        # has been excluded as being too bright - just assume host is identified
        if CUT and mtrue < 14.:
            allpox.append(1.)
            allmag.append(mtrue)
            allz.append(zdm["z"][i])
            alli.append(i)
            allseps.append(truth["gal_off"])
            allsizes.append(truth["half_light"])
        elif len(likely) == 1:
            cands = pd.read_csv(FRBfile)
            j = likely[0]
            allpox.append(results["POx"][i][j])
            allmag.append(results["ObsMags"][i][j])
            allz.append(cands["z"][j])
            alli.append(i)
            allseps.append(results["seps"][i][j])
            allsizes.append(results["sizes"][i][j])
        else:
            noti.append(i)
    
    weighted = pd.DataFrame()
    weighted["mags"] = rawmags
    weighted["POx"] = rawposts
    weighted["seps"] = rawseps
    weighted["sizes"] = rawsizes
    weighted.to_csv(woutfile,index=False)
    
    confident = pd.DataFrame()
    confident["z"] = allz
    confident["ifrb"] = alli
    confident["mags"] = allmag
    confident["POx"] = allpox
    confident["seps"] = allseps
    confident["sizes"] = allsizes
    confident.to_csv(outfile,index=False)
    
    
    if surveyfile is not None:
        from write_fake_survey_file import main as wfsf
        wfsf(N,surveyfile,frbfile,hostfile,IOK=alli)
    
    
    plot_confidant_frbs(alli,noti)
    
def plot_confidant_frbs(OK,NOTOK,opdir="MC_Generation_Plots/"):
    """
    Makes zdm plots similar to that of gen_mc_frbs_w_hosts, but showing the difference between
    confidant and all galaxies
    """
    # FRB list
    frbs = pd.read_csv("craco_900_mc_sample.csv")
    
    plt.figure()
    #plt.scatter(frbs["z"][OK],frbs["m_r"][OK],s=1,c=frbs["DMeg"][OK], cmap='cubehelix',marker='o')
    #plt.scatter(frbs["z"][NOTOK],frbs["m_r"][NOTOK],s=1,c=frbs["DMeg"][NOTOK], cmap='cubehelix',marker='x')
    plt.scatter(frbs["z"][OK],frbs["m_r"][OK],s=1,c="red",marker='o',label="$P(O|x) > 0.95$")
    plt.scatter(frbs["z"][NOTOK],frbs["m_r"][NOTOK],c="black", cmap='cubehelix',marker='x',label="$P(O|x) < 0.95$")
    #cbar = plt.colorbar()
    #cbar.set_label("DM$_{\\rm EG}$")
    plt.xlabel("Redshift, $z$")
    plt.ylabel("Apparent magnitude, $m_r$")
    plt.xlim(0,2.5)
    plt.ylim(10,32)
    plt.tight_layout()
    plt.savefig(opdir+"confidant_mr_z_plot.png")
    plt.close()
    
    plt.figure()
    #plt.scatter(frbs["z"][OK],frbs["DMeg"][OK],s=3,c=frbs["m_r"][OK], cmap='cubehelix',marker='o')
    #plt.scatter(frbs["z"][NOTOK],frbs["DMeg"][NOTOK],s=3,c=frbs["m_r"][NOTOK], cmap='cubehelix',marker='x')
    plt.scatter(frbs["z"][OK],frbs["DMeg"][OK],s=3, marker='o',label="$P(O|x) > 0.95$")
    plt.scatter(frbs["z"][NOTOK],frbs["DMeg"][NOTOK],s=5,marker='x',label="$P(O|x) < 0.95$")
    #cbar = plt.colorbar()
    plt.xlim(0,2.5)
    plt.ylim(0,2000)
    plt.xlabel("Redshift, $z$")
    #cbar.set_label("Host $m_r$")
    plt.ylabel("Extragalactic DM, DM$_{\\rm EG}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"confidant_dmeg_z_plot.png")
    plt.close()


def gen_frb_list(N):
    """
    generates a fake list of FRB names
    
    args:
        N (int): number of FRBs to use.
    """
    
    frblist = []
    for i in np.arange(N):
        name = "FRB"+str(i)
        frblist.append(name)
    return frblist


hostfile = "craco_assigned_galaxies.csv"
frbfile = "craco_900_mc_sample.csv"
    
# runs this four times, generating DSA-like hosts, and when including the 14th magnitude cut

# standard run,
#main("m14cut_hosts_1000.csv","w_m14cut_hosts_1000.csv",1000,frbfile,hostfile,
#        CUT=False,prefix="",surveyfile="confidant_short_fake_CRACO_900")

#main("m14cut_hosts_10000.csv","w_m14cut_hosts_10000.csv",10000,frbfile,hostfile,False,prefix="")

main("DSA_like_hosts_1000.csv","w_DSA_like_hosts_1000.csv",1000,frbfile,hostfile,True,prefix="",surveyfile="confidant_short_fake_DSAlike")
#main("DSA_like_hosts_10000.csv","w_DSA_like_hosts_10000.csv",10000,frbfile,hostfile,True,prefix="")
exit()
# repeats this for the sample assumed to have 30" localisations
hostfile = "loc30_craco_assigned_galaxies.csv"
frbfile = "craco_900_mc_sample.csv"
main("loc30_m14cut_hosts_1000.csv","loc30_w_m14cut_hosts_1000.csv",1000,frbfile,hostfile,False,prefix="loc30")
