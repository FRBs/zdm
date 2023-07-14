"""
This is a script to produce diagnostic plots to assess correlations
between different parameters.

It was used to aid in the understanding of Figures 9 in the paper.

Most plots produced show p(parameter|H0=X)
"""

import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt

def main(verbose=False):
    
    # output directory
    opdir="Diagnostic2DPlots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    CubeFile='/Users/cjames/CRAFT/Paper/H0/FullReal/real_down_full_cube.npz'
    if os.path.exists(CubeFile):
        data=np.load(CubeFile)
    else:
        print("Could not file cube output file ",CubeFile)
        print("Please obtain it from [repository]")
        exit()
    
    if verbose:
        print("Data file contains the following items")
        for thing in data:
            print(thing)
    
    lst = data.files
    lldata=data["ll"]
    params=data["params"]
    
    param_vals=ac.get_param_values(data,params)
    
    #reconstructs total pdmz given all the pieces, including unlocalised contributions
    pDMz=data["P_zDM0"]+data["P_zDM1"]+data["P_zDM2"]+data["P_zDM3"]+data["P_zDM4"]
    
    #DM only contribution - however it ignores unlocalised DMs from surveys 1-3
    pDMonly=data["pDM"]+data["P_zDM0"]+data["P_zDM4"]
    
    #do this over all surveys
    P_s=data["P_s0"]+data["P_s1"]+data["P_s2"]+data["P_s3"]+data["P_s4"]
    P_n=data["P_n0"]+data["P_n1"]+data["P_n2"]+data["P_n3"]+data["P_n4"]
    
    #labels=['p(N,s,DM,z)','P_n','P(s|DM,z)','p(DM,z)all','p(DM)all','p(z|DM)','p(DM)','p(DM|z)','p(z)']
    #for datatype in [data["ll"],P_n,P_s,pDMz,pDMonly,data["pzDM"],data["pDM"],data["pDMz"],data["pz"]]:
    
    # builds uvals list
    uvals=[]
    latexnames=[]
    for ip,param in enumerate(data["params"]):
        # switches for alpha
        if param=="alpha":
            uvals.append(data[param]*-1.)
        else:
            uvals.append(data[param])
        if param=="alpha":
            latexnames.append('$\\alpha$')
            ialpha=ip
        elif param=="lEmax":
            latexnames.append('$\\log_{10} E_{\\rm max}$')
        elif param=="H0":
            latexnames.append('$H_0$')
        elif param=="gamma":
            latexnames.append('$\\gamma$')
        elif param=="sfr_n":
            latexnames.append('$n_{\\rm sfr}$')
        elif param=="lmean":
            latexnames.append('$\\mu_{\\rm host}$')
        elif param=="lsigma":
            latexnames.append('$\\sigma_{\\rm host}$')
    
    #latexnames=['$\\log_{10} E_{\\rm max}$','$H_0$','$\\alpha$','$\\gamma$','$n_{\\rm sfr}$','$\\mu_{\\rm host}$','$\\sigma_{\\rm host}$']
    
    list2=[]
    vals2=[]
    # gets Bayesian posteriors
    deprecated,uw_vectors,wvectors=ac.get_bayesian_data(data["ll"])
    for i,vec in enumerate(uw_vectors):
        n=np.argmax(vec)
        val=uvals[i][n]
        if params[i] != "H0":
            list2.append(params[i])
            vals2.append(val)
        else:
            iH0=i
    
    ###### NOTATION #####
    # uw: unweighted
    # wH0: weighted according to H0 knowledged
    # f: fixed other parameters
    # B: best-fit
    
    ############## 2D plots at best-fit valuess ##########
    
    ##### repeats for testing ####
    for wanted in ["pzDM","pz","pDMz","pDM", "P_n0","P_n1","P_n2","P_n3",
            "P_n4", "P_s0","P_s1","P_s2","P_s3","P_s4",
            "P_zDM0","P_zDM1","P_zDM2","P_zDM3","P_zDM4"]:
        # gets the slice corresponding to the best-fit values of all other parameters
        # this is 1D, so is our limit on H0 keeping all others fixed
        for i,item in enumerate(list2):
            
            list3=np.concatenate((list2[0:i],list2[i+1:]))
            vals3=np.concatenate((vals2[0:i],vals2[i+1:]))
            array=ac.get_slice_from_parameters(data,list3,vals3,wanted=wanted)
            
            # log to lin space
            array -= np.max(array)
            array = 10**array
            array /= np.sum(array)
            
            # now have array for slice covering best-fit values
            if i < iH0:
                modi=i
            else:
                modi=i+1
                #array=array.T
                array=array.swapaxes(0,1)
            if isinstance(wanted, list):
                basedir=opdir+wanted[0]+"_bf_normed2D/"
            else:
                basedir=opdir+wanted+"_bf_normed2D/"
            savename=basedir+"lls_"+params[iH0]+"_"+params[modi]+".png"
            os.makedirs(basedir, exist_ok=True)
            
            if params[modi]=="alpha":
                #switches order of array in alpha dimension
                array=np.flip(array,axis=0)
                ac.make_2d_plot(array,latexnames[modi],latexnames[iH0],
                    -param_vals[modi],param_vals[iH0],
                    savename=savename,norm=1)
            else:
                ac.make_2d_plot(array,latexnames[modi],latexnames[iH0],
                    param_vals[modi],param_vals[iH0],
                    savename=savename,norm=1)
    
    ############## 2D plots for total likelihood ###########
    uvals,ijs,arrays,warrays=ac.get_2D_bayesian_data(data["ll"])
    for which,array in enumerate(arrays):
        i=ijs[which][0]
        j=ijs[which][1]
        
        basedir=opdir+"2D/"
        os.makedirs(basedir, exist_ok=True)
        
        savename=opdir+"2D/lls_"+params[i]+"_"+params[j]+".png"
        ac.make_2d_plot(array,latexnames[i],latexnames[j],
            param_vals[i],param_vals[j],
            savename=savename)
        
        basedir=opdir+"normed2D/"
        os.makedirs(basedir, exist_ok=True)
        if params[i]=="H0":
            
            savename=basedir+"lls_"+params[j]+"_"+params[i]+".png"
            
            if params[j]=="alpha":
                ac.make_2d_plot(array.T,latexnames[j],latexnames[i],
                    param_vals[j]*-1.,param_vals[i],
                    savename=savename,norm=1)
            else:
                ac.make_2d_plot(array.T,latexnames[j],latexnames[i],
                    param_vals[j],param_vals[i],
                    savename=savename,norm=1)
        if params[j]=="H0":
            savename=basedir+"lls_"+params[i]+"_"+params[j]+".png"
            ac.make_2d_plot(array,latexnames[i],latexnames[j],
                param_vals[i],param_vals[j],
                savename=savename,norm=1)
    
    

main()
