"""
Makes figure 6 by fitting H0 outside the analysed range

"""

import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import quad

def main():
    
    # saves values for H0 and marginalised p(H0) posteriors
    if not (os.path.isfile("H0.npy") and os.path.isfile("pH0.npy")
        and os.path.isfile("ph0_others_all_fixed.npy")):
        get_H0_values()
    
    orig_H0=np.load("H0.npy")
    orig_pH0=np.load("pH0.npy")
    
    # sets the range for fitting the tail of H0
    minH0=80
    maxH0=130
    OK=np.where(orig_H0>=minH0)[0]
    H0=orig_H0[OK]
    pH0=orig_pH0[OK]
    
    # lognormal fit
    p0=[1.,1.,1.]
    popt,pcov=curve_fit(ln,H0,pH0,p0=p0)
    x=np.linspace(minH0,maxH0)
    lnx=ln(x,*popt)
    lnchisqr=np.sum((ln(H0,*popt)-pH0)**2)
    
    #spline interpolation
    spl=interp1d(orig_H0,orig_pH0,kind='cubic')
    longx=np.linspace(orig_H0[0],orig_H0[-1],100)
    
    # performs integration
    p1=quad(spl,orig_H0[0],orig_H0[-1])
    #print("Integral over original range of using a spline comes to ",p1)
    
    p2=quad(ln,orig_H0[-1],maxH0+50.,args=(popt[0],popt[1],popt[2]))
    #print("Integrating from ",orig_H0[-1]," to ",maxH0," gives an additional ",p2)
    
    # this figures shows a check where the lognormal fit is overplotted on the data
    os.makedirs("Figure6")
    
    plt.figure()
    plt.scatter(orig_H0,orig_pH0,label="Data")
    plt.scatter(H0,pH0, label="fitted data")
    plt.plot(longx,spl(longx),label='Spline interpolation')
    plt.plot(x,lnx,label="Lognormal extension",linestyle="--")
    
    plt.xlabel('$H_0$ [km/s/Mpc]')
    plt.ylabel('$p(H_0)$')
    plt.savefig("Figure6/check_fit.png")
    plt.close()
    
    # renormalises data - p1 was original sum, p2 is new sum
    norm=p1[0]+p2[0]
    p1 = p1[0]/norm
    p2 = p2[0]/norm
    #print("Now ratios are ",p1,p2)
    
    # constructs single vector
    nH=1000 #number of H0 points to sample at
    xtotal=np.linspace(orig_H0[0],130.,nH)
    lower=np.where(xtotal <= orig_H0[-1])[0]
    upper=np.where(xtotal > orig_H0[-1])[0]
    ytotal=np.zeros([nH])
    ytotal[lower]=spl(xtotal[lower])/norm
    ytotal[upper]=ln(xtotal[upper],*popt)/norm
    
    # makes cumulative distribution
    cy=np.cumsum(ytotal)
    #print("Approx norm is ",cy[-1]*(xtotal[1]-xtotal[0]))
    cy /= cy[-1]
    
    #orders data
    asyt=np.argsort(ytotal)
    syt=np.sort(ytotal)
    csyt=np.cumsum(ytotal)
    csyt /= csyt[-1]
    
    # values at which to calculate confidence levels
    # (1-99.7)/2, (1-95)/2, (1-90)/2, (1-68)/2 (but to greater accuracy)
    levels=np.array([0.00135,0.0228,0.05,0.15866])
    
    labels=['99.7%','95%','90%','68%']
    linestyles=["--",":","-.","-"]
    extrax=np.linspace(orig_H0[-1],maxH0,100)
    
    plt.figure()
    plt.scatter(orig_H0,orig_pH0/norm,label="Data")
    #plt.scatter(H0,pH0/norm, label="fitted data")
    plt.plot(longx,spl(longx)/norm,label='Spline interpolation')
    plt.plot(extrax,ln(extrax,*popt)/norm,label="Lognormal extension",linestyle="--")
    
    # gets pH0 when all other values fixed
    other_H0=np.load('ph0_others_all_fixed.npy')
    other_H0=other_H0[0]
    spl=interp1d(orig_H0,other_H0,kind='cubic')
    othery=spl(longx)
    plt.plot(longx,othery,label="Fixed parameters",linestyle=":",color="black")
    
    plt.xlabel('$H_0$ [km/s/Mpc]')
    plt.ylabel('$p(H_0)$')
    
    for i,l in enumerate(levels):
        v1,v2,i1,i2=ac.extract_limits(xtotal,ytotal,l)
        plt.plot([xtotal[i1],xtotal[i1]],[0.,ytotal[i1]],color="red",linestyle=linestyles[i])
        plt.plot([xtotal[i2],xtotal[i2]],[0.,ytotal[i2]],color="red",linestyle=linestyles[i])
        plt.text(xtotal[i1]-2,ytotal[i1]+1e-3,labels[i],rotation=90)
        plt.text(xtotal[i2],ytotal[i2]+1e-3,labels[i],rotation=90)
        
        print("limits ",l,v1,v2)
        
    plt.gca().set_ylim(bottom=0)  
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure6/H0_fig6.png")
    plt.close()
    
def ln(x,*params):
    a=params[0]
    b=params[1]
    c=params[2]
    lnx=np.log(x)
    vals=a*np.exp(-0.5*(lnx-b)**2/c)/x
    return vals

def exp(x,*params):
    a=params[0]
    b=params[1]
    c=params[2]
    vals = a*np.exp(-(x-c)/b)
    return vals

def get_H0_fixed_vales():
    
    deprecated,uw_vectors,wvectors=ac.get_bayesian_data(data["ll"])
    
    
 
def get_H0_values():
    CubeFile='real_down_full_cube.npz'
    if os.path.exists(CubeFile):
        data=np.load(CubeFile)
    else:
        print("Could not file cube output file ",CubeFile)
        print("Please obtain it from [repository]")
        exit()
    
    lst = data.files
    params=data["params"]
    
    param_vals=ac.get_param_values(data,params)
    
    iH0=np.where(data["params"] == "H0")
    ################ gets 1D H0 values ############
    deprecated,uw_vectors,wvectors=ac.get_bayesian_data(data["ll"])
    
    print("H0: ",param_vals[1])
    print("Probs: ",uw_vectors[1])
    np.save("H0.npy",param_vals[1])
    np.save("pH0.npy",uw_vectors[1])
    
    # builds uvals list, i.e. of unique values of parameters
    uvals=[]
    for ip,param in enumerate(data["params"]):
        # switches for alpha
        if param=="alpha":
            uvals.append(data[param]*-1.)
        else:
            uvals.append(data[param])
    
    # extract the best-fit parameter values
    list2=[]
    vals2=[]
    for i,vec in enumerate(uw_vectors):
        n=np.argmax(vec)
        val=uvals[i][n]
        if params[i] != "H0":
            list2.append(params[i])
            vals2.append(val)
        else:
            iH0=i
    
    # gets the slice corresponding to the best-fit values of all other parameters
    # this is 1D, so is our limit on H0 keeping all others fixed
    pH0_fixed=ac.get_slice_from_parameters(data,list2,vals2)
    
    pH0_fixed -= np.max(pH0_fixed)
    pH0_fixed = 10**pH0_fixed
    pH0_fixed /= np.sum(pH0_fixed)
    pH0_fixed /= (uvals[iH0][1]-uvals[iH0][0])
    
    # saves this for generating special H0 plot
    np.save("ph0_others_all_fixed.npy",[pH0_fixed])

main()
