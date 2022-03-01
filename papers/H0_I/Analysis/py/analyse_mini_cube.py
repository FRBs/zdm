import numpy as np

import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt

def main():
    
    data=np.load('craco_mini_cube.npz')
    for item in data:
        print(item)
    
    lst = data.files
    lldata=data["ll"]
    print("Dimensions are ",lldata.shape)
    params=data["params"]
    
    print("Max and min likelihoods are ",np.nanmax(lldata), np.nanmin(lldata))
    all_uvals=[]
    all_vectors=[]
    all_wvectors=[]
    
    combined=data["pzDM"]+data["pDM"]
    labels=['p(s,DM,z)','p(DM,z)','p(z|DM)','p(DM)','p(DM|z)','p(z)']
    for datatype in [data["ll"],combined,data["pzDM"],data["pDM"],data["pDMz"],data["pz"]]:
        continue
        uvals,vectors,wvectors=ac.get_bayesian_data(datatype)
        all_uvals.append(uvals)
        all_vectors.append(vectors)
        all_wvectors.append(wvectors)
    
    
    param_vals=[]
    for col in [data["lEmax"],data["H0"],data["alpha"],data["gamma"],data["sfr_n"],data["lmean"],data["lsigma"]]:
        unique=np.unique(col)
        param_vals.append(unique)
    
    for which in np.arange(7):
        continue
        plt.figure()
        plt.xlabel(params[which])
        plt.ylabel('p('+params[which]+')')
        xvals=param_vals[which]
        
        for idata,vectors in enumerate(all_vectors):
            vector=vectors[which]
            plt.plot(xvals,vector,label=labels[idata])
        
        plt.legend()
        plt.savefig(params[which]+".pdf")
        plt.close()
    
    
    uvals,ijs,arrays,warrays=ac.get_2D_bayesian_data(lldata)
    
    for which,array in enumerate(arrays):
        plt.figure()
        plt.xlabel(params[ijs[which][0]])
        plt.ylabel(params[ijs[which][1]])
        
        xvals=param_vals[ijs[which][0]]
        yvals=param_vals[ijs[which][1]]
        
        dx=xvals[-1]-xvals[0]
        dy=yvals[-1]-yvals[0]
        aspect=dx/dy
        
        extent=[np.min(xvals),np.max(xvals),np.min(yvals),np.max(yvals)]
        
        plt.imshow(arrays[which].T,origin='lower',extent=extent,aspect=aspect)
        plt.xlabel(params[ijs[which][0]])
        plt.ylabel(params[ijs[which][1]])
        plt.xticks(rotation=90)
        cbar=plt.colorbar()
        cbar.set_label('$p('+params[ijs[which][0]]+','+params[ijs[which][1]]+')$')
        plt.savefig(params[ijs[which][0]]+"_"+params[ijs[which][1]]+".pdf")
        plt.close()
    

main()
