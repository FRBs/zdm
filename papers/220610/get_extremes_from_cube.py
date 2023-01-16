"""
This is a script to produce parameter sets which are at X% of the accepted values

Notes: I have removed "linear=True" from "get_slice_from_parameters"

"""

import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt

def main(p=0.9,verbose=False):
    
    
    ##### sets p-values at which we extract limits ####
    limvals = np.array([0.68,0.9,0.95,0.997,0.03])
    lim=0.9
    
    ######### sets the values of H0 for priors #####
    Planck_H0 = 67.4
    Planck_sigma = 0.5
    Reiss_H0 = 74.03
    Reiss_sigma = 1.42
    
    ##### loads cube data #####
    cube='real_down_full_cube-001.npz'
    if not os.path.exists(cube):
        print("Could not find cube file ",cube,". Exiting...")
        exit()
    
    data=np.load(cube)
    if verbose:
        for thing in data:
            print(thing)
        print(data["params"])
    
    # gets values of cube parameters
    #param_vals=get_param_values(data,verbose)
    
    # gets latex names
    uvals,latexnames = get_names_values(data)
    
    ######################### Part 1: Gets limits for four different priors on H0 #################
    
    ######### prior 1: flat prior (~no prior) #########
    deprecated,uw_vectors,wvectors=ac.get_bayesian_data(data["ll"])
    lims1=ac.get_limits(uvals,uw_vectors,limvals,None,logspline=True)
    
    ########### priors 2 and 3: H0 fixed to CMB / Planck ###########
    # extracts best-fit values
    list23=[]
    vals2=[]
    vals3=[]
    notH0=[]
    params_notH0 = []
    for i,vec in enumerate(uw_vectors):
        n=np.argmax(vec) # selects the most likely value
        val=uvals[i][n]
        if data["params"][i] == "H0":
            # enables us to select a slice corresponding to particular H0 values
            list23.append(data["params"][i])
            vals2.append(Reiss_H0)
            vals3.append(Planck_H0)
            iH0=i # setting index for Hubble
        else:
            # lists parameters that are not H0
            notH0.append(uvals[i])
            params_notH0.append(data["params"][i])
    
    # gets the slice corresponding to specific values of H0
    Reiss_H0_selection=ac.get_slice_from_parameters(data,list23,vals2,verbose=False)
    Planck_H0_selection=ac.get_slice_from_parameters(data,list23,vals3,verbose=False)
    
    # will have Bayesian limits on all parameters over everything but H0
    deprecated,ReissH0_vectors,deprecated=ac.get_bayesian_data(Reiss_H0_selection)
    lims2=ac.get_limits(notH0,ReissH0_vectors,limvals,None,logspline=True)
    
    deprecated,PlanckH0_vectors,deprecated=ac.get_bayesian_data(Planck_H0_selection)
    lims3=ac.get_limits(notH0,PlanckH0_vectors,limvals,None,logspline=True)
    
    
    ####### prior 4: standard prior on H0 ########
    # generates plots for our standard prior on H0 only
    # applies a prior on H0, which is flat between systematic differences, then falls off as a Gaussian either side
    H0_dim=np.where(data["params"]=="H0")[0][0]
    wlls = ac.apply_H0_prior(data["ll"],H0_dim,data["H0"],Planck_H0,
        Planck_sigma, Reiss_H0, Reiss_sigma)
    deprecated,wH0_vectors,wvectors=ac.get_bayesian_data(wlls)
    lims4=ac.get_limits(uvals,wH0_vectors,limvals,None,logspline=True)
    
    ############# for each set of limits, get parameter sets at these extremes ###########
    lim=0.9
    ilim=np.where(limvals==lim)[0][0]
    lims1=lims1[:,ilim*2:(ilim+1)*2]
    lims2=lims2[:,ilim*2:(ilim+1)*2]
    lims3=lims3[:,ilim*2:(ilim+1)*2]
    lims4=lims4[:,ilim*2:(ilim+1)*2]
    
    ########### now extract parameter sets corresponding to those limits ###########
    
    print("####### Extremes of parameters, assuming no priors on H0 ######")
    
    nparams=len(data["params"])
    # for unweighted data
    for i,p in enumerate(data["params"]):
        print("GETTING extremes for parameter",p,lims1[i,0],lims1[i,1])
        continue
        for il in 0,1:
            # gets slice corresponding to lower limit
            tempslice=ac.get_slice_from_parameters(data,[data["params"][i]],[lims1[i,il]]
                ,verbose=False)
            am=np.argmax(tempslice)
            indices=np.unravel_index(am,tempslice.shape)
            # now prints out parameter values corresponding to the max probabilities
            
            # other parameters before i
            for j in np.arange(i):
                pval=data[data["params"][j]][indices[j]]
                print(j,data["params"][j],indices[j],pval)
            
            print(i,p,lims1[i,il])
            
            # other parameters after i: note indices has length nparams-1
            for j in np.arange(i+1,nparams):
                pval=data[data["params"][j]][indices[j-1]]
                print(j,data["params"][j],indices[j-1],pval)
    
    print("####### Extremes of parameters, assuming planck H0 ######")
    # note lims2 and lims3 do not have H0 as a parameter in them
    # should remove this from the param list
    
    # for unweighted data
    for i,p in enumerate(params_notH0):
        print("GETTING extremes for parameter",p,lims3[i,0],lims3[i,1])
        
        # does this for minimum and maximum values
        for il in 0,1:
            # gets slice corresponding to lower limit
            tempslice=ac.get_slice_from_parameters(data,["H0",params_notH0[i]],[Planck_H0,lims3[i,il]],
                verbose=False)
            const_slice=ac.get_slice_from_parameters(data,["H0",params_notH0[i]],[Planck_H0,lims3[i,il]],
                verbose=False,wanted='lC')
            
            am=np.argmax(tempslice)
            best_constant=const_slice.flatten()[am]
            indices=np.unravel_index(am,tempslice.shape)
            # now prints out parameter values corresponding to the max probabilities
            
            # other parameters before i
            for j in np.arange(i):
                pval=data[params_notH0[j]][indices[j]]
                print(params_notH0[j],pval)
            
            print(p,lims3[i,il])
            
            # other parameters after i: note indices has length nparams-1
            for j in np.arange(i+1,nparams-1):
                pval=data[params_notH0[j]][indices[j-1]]
                print(params_notH0[j],pval)
            print("  lC ",best_constant)
    
    exit()
    
    print("\n\n\n\n####### Extremes of parameters, assuming std prior on H0 ######")
    
    print("WARNING: replacing data values with weighted versions")
    data["ll"]=wlls
    
    nparams=len(data["params"])
    # for unweighted data
    for i,p in enumerate(data["params"]):
        print("GETTING extremes for parameter",p,lims4[i,0],lims4[i,1])
        
        for il in 0,1:
            # gets slice corresponding to lower limit
            tempslice=ac.get_slice_from_parameters(data,[data["params"][i]],[lims4[i,il]]
                ,verbose=False,linear=True)
            am=np.argmax(tempslice)
            indices=np.unravel_index(am,tempslice.shape)
            # now prints out parameter values corresponding to the max probabilities
            
            # other parameters before i
            for j in np.arange(i):
                pval=data[data["params"][j]][indices[j]]
                print(j,data["params"][j],indices[j],pval)
            
            print(i,p,lims4[i,il])
            
            # other parameters after i: note indices has length nparams-1
            for j in np.arange(i+1,nparams):
                pval=data[data["params"][j]][indices[j-1]]
                print(j,data["params"][j],indices[j-1],pval)
     
     

def get_names_values(data):
    """
    Gets a list of latex names and corrected parameter values
    """
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
        elif param=="F":
            latexnames.append('$F$')
    return uvals,latexnames
    
def get_param_values(data,verbose=False):
    """
    Returns the unique cube values for each parameter in the cube
    
    Input:
        data cube (tuple from reading the .npz)
    
    Output:
        list of numpy arrays for each parameter giving their values
    """
    # gets unique values for each axis
    param_vals=[]
    
    #for col in param_list:
    for col in data["params"]: 
          
        #unique=np.unique(col)
        unique=np.unique(data[col])
        param_vals.append(unique)
        if verbose:
            print("For parameter ",col," cube values are ",unique)
    return param_vals


main()
