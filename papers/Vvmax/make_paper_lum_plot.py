"""
This project takes in V/Vmax data from Wayne, creates a histogram,
and generates errorbars on the data.

It does this by reading in host galaxy FRB data ("loc_data.dat")
and DM-z relation data ("macquart_data.dat")

"""


import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from RLFs import *

import matplotlib
#matplotlib.rcParams['image.interpolation'] = None
defaultsize=16
ds=4
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(loc=True,plot=True):
    
    ### infiles
    savefiles = [
    
        "LocLumData/localised_vvmax_data_NSFR_0.0_alpha_0.npz",
        "zMacquartLumData/v2macquart_vvmax_data_NSFR_0.0_alpha_0.npz"
        ]
    labels = [
        "$z_{\\rm loc}$",
        "$z_{\\rm DM}$"
        ]
    
    if not plot: #this is just for printing outputs for a table
        savefiles = [
            "zMacquartLumData/v2macquart_vvmax_data_NSFR_0.0_alpha_0.npz",
            "LocLumData/localised_vvmax_data_NSFR_0.0_alpha_0.npz"]
        
        labels = ["",""]
        
    linestyles = ["--",":","-."]
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
        
        if True:
            print("\n\n#### i is ",i,"   #####")
            Sndf = ndata-3
            PLndf = ndata-2
            Dndf = 1
            pvalue = 1.-stats.chi2.cdf(chi2, Sndf)
            PLpvalue = 1.-stats.chi2.cdf(PLchi2, PLndf)
            
            # formula as (SS1-SS2)/(ndf2-ndf1) / (SS2/ndf2)
            Fstat = ((PLchi2-chi2)/Dndf)/(chi2/Sndf)
            Fpvalue = 1.-stats.f.cdf(Fstat,1.,Sndf)
            
            print("Power-law best fits:")
            string = "& {0:1.2f} $\pm$ {1:1.2f} & N/A & {2:1.2f} & {3:1.2f}"\
                .format(PLfit[0],PLerr[0],PLchi2/(PLndf),PLpvalue)
            print(string)
            
            print("Schechter function best fits:")
            string = "& {0:1.2f} $\pm$ {1:1.2f} & {2:1.2e} $\pm$ {3:1.2e} & {4:1.2e} & {5:1.2e}"\
                .format(fit[1],fiterr[1],fit[2],fiterr[2],chi2/Sndf,Fpvalue)
            print(string)
            
        
        if plot:
            plt.errorbar(bcs,h/norm,yerr = MCbounds/norm,linestyle="",marker=markers[i],label=labels[i])
            plt.plot(bcs,integrateSchechter(bcs,*fit)/norm,
                color=plt.gca().lines[-1].get_color(),linestyle="-")
    
    
    JamesFit,RyderFit,ShinFit,LuoFit = best_fits(JHz = True)
    RyderGuess = make_first_guess(bcs,h,RyderFit,2)
    JamesGuess = make_first_guess(bcs,h,JamesFit,2)
    ShinGuess = make_first_guess(bcs,h,ShinFit,2)
    LuoGuess = make_first_guess(bcs,h,LuoFit,2)
    
    
    bcs = np.logspace(22.5,26.5,5)
    glabels = ["Ryder et al. (2023)", "Shin et al. (2023)", "Luo et al. (2020)"]
    for i,guess in enumerate([RyderGuess,ShinGuess,LuoGuess]):
        ibin = np.where(bcs > 1e23)[0][0]
        toplot = integrateSchechter(bcs,*guess)
        norm = toplot[ibin]
        if plot:
            plt.plot(bcs,toplot/norm,linestyle=linestyles[i],label=glabels[i])
    
    if plot:
        plt.xlabel('$E_{\\nu}$ [J Hz$^{-1}$]')
        plt.ylabel('$E_{\\nu}\\,  {\\rm RLF}(E_{\\nu})$ [arb. units]')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-5,300)
        #plt.ylim(1e-9,0.002)
        plt.legend(loc="lower left",fontsize=12,frameon=False)
        plt.tight_layout()
        plt.savefig("paper_luminosity.pdf")
        plt.close()
    
    
            
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
    plt.ylabel('$E_{\\nu}\\,  {\\rm RLF}(E_{\\nu})$ [a.u.]')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-10,1)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()       
    return results[0],[results[1][0,0],results[1][1,1],results[1][2,2]]
    


main()
