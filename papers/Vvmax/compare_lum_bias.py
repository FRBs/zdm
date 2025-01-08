"""
This script creates a plot comparing biased vs unbiased
localised energy function estimates only, producing 
"bias_comparison.pdf"

"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy import stats
from RLFs import *

global E0
# normalisation energy - no physical meaning, just makes units more sensible
E0=1e22


import matplotlib
#matplotlib.rcParams['image.interpolation'] = None
defaultsize=16
ds=4
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(loc=True,plot=True,Truncate=True):
    
    opdir = "v2LumData/v2"
    opdir2 = "MinzOutput/"
    
    # extracts the mean V/Vmax value for these two surveys
    vvmax_data="UnbiasedLocalisedOutput/localised_vvmax_data_NSFR_0.0_alpha_0.0.dat"
    vvmaxbar = get_vvmax(vvmax_data)
    print("Unbiased localised vvmax bar is ",vvmaxbar)
    
    ### infiles
    savefiles = [
        "LocLumData/localised_vvmax_data_NSFR_0.0_alpha_0.npz",
        "UnbiasedLocalisedLumData/localised_vvmax_data_NSFR_0.0_alpha_0.npz"
        ]
    
    labels = [
        "$z_{\\rm loc}$ (biased)",
        "$z_{\\rm loc}$ (unbiased)"
        ]
    
        
    linestyles = ["--","-"]
    markers = ['+','s','o','x','^']
    
    plt.figure()
    
    for i,savefile in enumerate(savefiles):
        data=np.load(savefile)
        bcs = data['bin_centres']
        h = data['histogram']
        MCbounds = data['MCerror']
        fit = data['Schechter']
        fiterr = data['Serr']
        chi2 = data['Schi2']
        Nfrb = data['Nfrb']
        PLfit = data['PowerLaw']
        PLerr = data['PLerr']
        PLchi2 = data['plchi2']
        ndata = data['ndata']
        
        
        Ntot = np.sum(Nfrb)
        # select bin to normalise to
        ibin = np.where(bcs > 1e23)[0][0]
        norm = h[ibin]
        
        if plot:
            if i==0:
                l1=plt.errorbar(bcs,h/norm,yerr = MCbounds/norm,linestyle="",marker=markers[i],label=labels[i])
                l11,=plt.plot(bcs,integrateSchechter(bcs,*fit)/norm,
                    color=plt.gca().lines[-1].get_color(),linestyle="-")
            elif i==1:
                l2=plt.errorbar(bcs,h/norm,yerr = MCbounds/norm,linestyle="",marker=markers[i],label=labels[i])
                l22,=plt.plot(bcs,integrateSchechter(bcs,*fit)/norm,
                    color=plt.gca().lines[-1].get_color(),linestyle="-.")
            
    
    
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles.append(handles[0])
    #handles=handles[1:]
    #labels.append(labels[0])
    #labels=handles[1:]
    
    if plot:
        plt.xlabel('$E_{\\nu}$ [J Hz$^{-1}$]')
        plt.ylabel('$E_{\\nu}\\,  {\\rm RLF}(E_{\\nu})$ [arb. units]')
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(1e-9,0.002)
        
        
        #plt.xlim(1e22,1e26)
        #plt.ylim(1e-2,100)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("bias_comparison.pdf")
        
    

def get_vvmax(datafile):
    """
    Loads data and gets mean vvmax value
    """
    data  = np.loadtxt(datafile)
    
    vvmax = data[:,6]
    vvmaxbar = np.sum(vvmax)/vvmax.size
    return vvmaxbar
    
def plot_lum_func(bcs,h,MCbounds,count,outfile):
    """
    count: number of FRBs per bin
    h: luminosity function
    bcs: bin centres in log space
    MCbounds: lower and upper errors
    """
    
    OK = np.where(h>0)[0]
    MCbounds = MCbounds[:,OK]
    bcs = bcs[OK]
    h = h[OK]
    count = count[OK]
    plt.figure()
    plt.errorbar(bcs,h,yerr = MCbounds,linestyle="",marker='^')
    
    for i,c in enumerate(count):
        plt.text(bcs[i],0.1*h[i],str(c),
            color=plt.gca().lines[-1].get_color(),fontsize=10)
    
    error = np.log10((MCbounds[1]/MCbounds[0])**0.5)
    JamesFit,RyderFit,ShinFit,LuoFit = best_fits(JHz = True)
    RyderGuess = make_first_guess(bcs,h,RyderFit,2)
    JamesGuess = make_first_guess(bcs,h,JamesFit,2)
    ShinGuess = make_first_guess(bcs,h,ShinFit,2)
    LuoGuess = make_first_guess(bcs,h,LuoFit,2)
    
    results = curve_fit(logIntegrateSchechter,np.log10(bcs),np.log10(h),p0=RyderGuess,sigma=error)
    plt.plot(bcs,integrateSchechter(bcs,*results[0]),
        color=plt.gca().lines[-1].get_color(),linestyle="-")
    
    
    plt.xlabel('$E_{\\nu}$ [J Hz$^{-1}$]')
    plt.ylabel('$E_{\\nu}\\,  {\\rm RLF}(E_{\\nu})$ [arb. units]')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-10,1)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()       
    return results[0],[results[1][0,0],results[1][1,1],results[1][2,2]]
    

main()
