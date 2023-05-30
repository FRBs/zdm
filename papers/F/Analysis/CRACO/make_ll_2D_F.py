import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt
from IPython import embed

def main(verbose=False):
    
    # output directory
    opdir="2d_figs/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    CubeFile='Cubes/craco_full_cube.npz'
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
    
    def get_param_values(data,params):
        """
        Gets the unique values of the data from a cube output
        Currently the parameter order is hard-coded

        """
        param_vals=[]
        for param in params:
            col=data[param]
            unique=np.unique(col)
            param_vals.append(unique)  
        return param_vals
    
    param_vals=get_param_values(data, params)
    
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
        elif param=="logF":
            latexnames.append('$\\log_{10} F$')
    
    #latexnames=['$\\log_{10} E_{\\rm max}$','$H_0$','$\\alpha$','$\\gamma$','$n_{\\rm sfr}$','$\\mu_{\\rm host}$','$\\sigma_{\\rm host}$']
    
    list2=[]
    vals2=[]
    # gets Bayesian posteriors
    deprecated,uw_vectors,wvectors=ac.get_bayesian_data(data["ll"])
    for i,vec in enumerate(uw_vectors):
        n=np.argmax(vec)
        val=uvals[i][n]
        if params[i] != "logF":
            list2.append(params[i])
            vals2.append(val)
        else:
            iF=i
    
    ###### NOTATION #####
    # uw: unweighted
    # wH0: weighted according to H0 knowledged
    # f: fixed other parameters
    # B: best-fit
    
    ############## 2D plots at best-fit valuess ##########
    
    # gets the slice corresponding to the best-fit values of all other parameters
    # this is 1D, so is our limit on H0 keeping all others fixed
    for i,item in enumerate(list2):
        
        list3=np.concatenate((list2[0:i],list2[i+1:]))
        vals3=np.concatenate((vals2[0:i],vals2[i+1:]))
        array=ac.get_slice_from_parameters(data,list3,vals3)
        
        # log to lin space
        array[np.isnan(array)] = -1e99
        array -= np.nanmax(array)
        array = 10**array
        array /= np.sum(array)
        
        # now have array for slice covering best-fit values
        if i < iF:
            modi=i
        else:
            modi=i+1
            #array=array.T
            array=array.swapaxes(0,1)
        savename=opdir+"/lls_"+params[iF]+"_"+params[modi]+".png"
        
#         if (latexnames[modi] == '$\\gamma$'):
#             embed(header="gamma")
        
#         if (latexnames[modi] == '$H_0$'):
#             embed(header="H0")
                
        if params[modi]=="alpha":
            #switches order of array in alpha dimension
            array=np.flip(array,axis=0)
            ac.make_2d_plot(array,latexnames[modi],latexnames[iF],
                -param_vals[modi],param_vals[iF],
                savename=savename,norm=1)
        else:
            ac.make_2d_plot(array,latexnames[modi],latexnames[iF],
                param_vals[modi],param_vals[iF],
                savename=savename,norm=1)
    
main()