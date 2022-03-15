"""
This is a script to produce 'useful' outputs for a cube.

It was originally designed for the real mini cube. However,
    it should serve as a template for other cube analyses.

Currently, the script has to "know" how many surveys
    are present, and also know which are localised
    and which not. This possibly could be changed
    in the future.
"""

import numpy as np
import zdm
from zdm import analyze_cube as ac
from matplotlib import pyplot as plt

def main():
    
    data=np.load('real_mini_cube.npz')
    
    lst = data.files
    lldata=data["ll"]
    params=data["params"]
    
    if False:
        combined1=data["pzDM"]+data["pDM"]
        combined2=data["pDMz"]+data["pz"]
        diff=(combined1-combined2)/(combined1+combined2)
    
    #reconstructs total pdmz given all the pieces, including unlocalised contributions
    pDMz=data["P_zDM0"]+data["P_zDM1"]+data["P_zDM2"]+data["P_zDM3"]+data["P_zDM4"]
    
    #DM only contribution - however it ignores unlocalised DMs
    pDMonly=data["pDM"]+data["P_zDM0"]+data["P_zDM4"]
    
    #do this over all surveys
    P_s=data["P_s0"]+data["P_s1"]+data["P_s2"]+data["P_s3"]+data["P_s4"]
    P_n=data["P_n0"]+data["P_n1"]+data["P_n2"]+data["P_n3"]+data["P_n4"]
    
    #labels=['p(N,s,DM,z)','P_n','P(s|DM,z)','p(DM,z)all','p(DM)all','p(z|DM)','p(DM)','p(DM|z)','p(z)']
    #for datatype in [data["ll"],P_n,P_s,pDMz,pDMonly,data["pzDM"],data["pDM"],data["pDMz"],data["pz"]]:
    
    # 1D plots by statistical contribution
    contributions=[data["ll"],P_n,P_s,pDMz]
    labels=['p(N,s,DM,z)','P_n','P(s|DM,z)','p(DM,z)']
    make_1d_plots_by_contribution(data,contributions,labels,prefix="by_contribution")
    
    # 1D plots by surveys
    contributions=[data["P_s0"],data["P_s1"],data["P_s2"],data["P_s3"],data["P_s4"]]
    labels=["CRAFT/FE","CRAFT/ICS (low)","CRAFT/ICS (mid)","CRAFT/ICS (high)","Parkes/Mb"]
    make_1d_plots_by_contribution(data,contributions,labels,prefix="by_survey_")
    
    
    param_vals=get_param_values(data)
    
    ############## 2D plots for total likelihood - parkes ps only ###########
    #uvals,ijs,arrays,warrays=ac.get_2D_bayesian_data(lldata)
    uvals,ijs,arrays,warrays=ac.get_2D_bayesian_data(data["P_s4"])
    for which,array in enumerate(arrays):
        savename="parkes_s_"+params[ijs[which][0]]+"_"+params[ijs[which][1]]+".pdf"
        make_2d_plot(array,params[ijs[which][0]],params[ijs[which][1]],param_vals[ijs[which][0]],param_vals[ijs[which][1]],savename=savename)
        
    
    ############## 2D plots for total likelihood ###########
    uvals,ijs,arrays,warrays=ac.get_2D_bayesian_data(data["ll"])
    for which,array in enumerate(arrays):
        savename="lls_"+params[ijs[which][0]]+"_"+params[ijs[which][1]]+".pdf"
        make_2d_plot(array,params[ijs[which][0]],params[ijs[which][1]],param_vals[ijs[which][0]],param_vals[ijs[which][1]],savename=savename)
      
def get_param_values(data):
    # gets unique values for each axis
    param_vals=[]
    param_list=[data["lEmax"],data["H0"],data["alpha"],data["gamma"],data["sfr_n"],data["lmean"],data["lsigma"]]
    for col in param_list:
        unique=np.unique(col)
        param_vals.append(unique)  
    return param_vals
    
def make_1d_plots_by_contribution(data,contributions,labels,prefix=""):
    """
    contributions: list of vectors giving various likelihood terms
    Labels: lists labels stating what these are
    """
    ######################### 1D plots, split by terms ################
    all_uvals=[]
    all_vectors=[]
    all_wvectors=[]
    
    combined=data["pzDM"]+data["pDM"]
    
    # gets 1D Bayesian curves for each contribution
    for datatype in contributions:
        uvals,vectors,wvectors=ac.get_bayesian_data(datatype)
        all_uvals.append(uvals)
        all_vectors.append(vectors)
        all_wvectors.append(wvectors)
    
    params=data["params"]
    
    # gets unique values for each axis
    param_vals=[]
    param_list=[data["lEmax"],data["H0"],data["alpha"],data["gamma"],data["sfr_n"],data["lmean"],data["lsigma"]]
    for col in [data["lEmax"],data["H0"],data["alpha"],data["gamma"],data["sfr_n"],data["lmean"],data["lsigma"]]:
        unique=np.unique(col)
        param_vals.append(unique)
    
    # assigns different plotting styles to help distinguish curves
    linestyles=['-','--','-.',':','-','--','-.',':','-','--','-.',':']
    for which in np.arange(len(param_list)):
        plt.figure()
        plt.xlabel(params[which])
        plt.ylabel('p('+params[which]+')')
        xvals=param_vals[which]
        #print("Doing this for parameter ",params[which])
        
        for idata,vectors in enumerate(all_vectors):
            vector=vectors[which]
            #print(labels[idata]," has values ",vector)
            plt.plot(xvals,vector,label=labels[idata],linestyle=linestyles[idata])
            #if idata==4:
            #    break
            if idata==1:
                prod = vector
            elif idata>1:
                prod = prod*vector
        prod /= np.sum(prod)
        #plt.plot(xvals,prod,label="dumb product")
        plt.legend()
        plt.savefig(prefix+params[which]+".pdf")
        plt.close()
    
      
def make_2d_plot(array,xlabel,ylabel,xvals,yvals,savename=None):
    """
    Makes 2D plot given an array of probabilities
    
    array: 2D grid of data to plot
    
    xlabel: string to label on x axis
    ylabel: string to label y axis
    
    xvals: np array, values of x data
    yvals: np array, values of y data
    
    savename (optional): string to save data under
    """
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # calculates increment in values
    dx=xvals[-1]-xvals[0]
    dy=yvals[-1]-yvals[0]
    
    nx=xvals.size
    ny=yvals.size
    
    aspect=dx/dy
    
    # the dx/2 etc is so that parameter values line up with the centres of each pixel
    extent=[np.min(xvals)-dx/2.,np.max(xvals)+dx/2.,np.min(yvals)-dy/2.,np.max(yvals)+dy/2.]
    
    plt.imshow(array.T,origin='lower',extent=extent,aspect=aspect)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    cbar=plt.colorbar()
    cbar.set_label('$p('+xlabel+','+ylabel+')$')
    if savename is None:
        savename=xlabel+"_"+ylabel+".pdf"
    plt.savefig(savename)
    plt.close()
    

main()
