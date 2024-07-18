"""
This project takes in V/Vmax data from Wayne, creates a histogram,
and generates errorbars on the data.

It does this by reading in host galaxy FRB data ("loc_data.dat")
and DM-z relation data ("macquart_data.dat")

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
    vvmax_data="UnbiasedLocalisedOutput/localised_vvmax_data_NSFR_0.0_alpha_0.dat"
    vvmaxbar = get_vvmax(vvmax_data)
    print("Unbiased localised vvmax bar is ",vvmaxbar)
    
    ### infiles
    savefiles = [
        "UnbiasedLocalisedLumData/localised_vvmax_data_NSFR_0.0_alpha_0.npz"
        ]
    for i,zfrac in enumerate(np.linspace(0,1,11)):
        savefiles.append("MinzLumData/Minzmacquart_vvmax_data_NSFR_0_alpha_0_" \
            +str(zfrac)[0:3]+"_concatenated.npz")
        
        vvmax_data="MinzOutput/Minzmacquart_vvmax_data_NSFR_0_alpha_0_" \
            +str(zfrac)[0:3]+"_concatenated.dat"
        vvmaxbar = get_vvmax(vvmax_data)
        print("i = ",i," vvmaxbar is ",vvmaxbar)
        
    if Truncate:
        # ensures only two lines are plotted
        savefiles = savefiles[0:2]
        
        
    labels = [
        #"$z_{\\rm DM}$",
        "$z_{\\rm loc}; z_{\\rm lim}=0.7$"
        ]
    
    if not plot: #this is just for printing outputs for a table
        savefiles = [
            opdir+"localised_vvmax_data_NSFR_0.0_alpha_0.npz",
            #opdir+"macquart_vvmax_data_NSFR_0.0_alpha_0.npz"
            ]
        
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
        
        #print(fit[1],fiterr[1],fit[2],fiterr[2])
        
        if i==0 or i==1:#or i==1 or i==2 or i==11:
            print("\n\n#### i is ",i,"   #####")
            Sndf = ndata-3
            PLndf = ndata-2
            Dndf = 1
            pvalue = 1.-stats.chi2.cdf(chi2, Sndf)
            PLpvalue = 1.-stats.chi2.cdf(PLchi2, PLndf)
            
            
            # formula as (SS1-SS2)/(ndf2-ndf1) / (SS2/ndf2)
            Fstat = ((PLchi2-chi2)/Dndf)/(chi2/Sndf)
            Fpvalue = 1.-stats.f.cdf(Fstat,1.,Sndf)
            #print("p-value of F-test is ",Fpvalue)
            
            
            print("Power-law best fits:")
            string = "& {0:1.2f} $\pm$ {1:1.2f} & N/A & {2:1.2f} & {3:1.2f}"\
                .format(PLfit[0],PLerr[0],PLchi2/PLndf,PLpvalue)
            print(string)
            #print("chi2 for power-law fit is ",PLchi2," (pvalue ",PLpvalue,")")
            #print("slope of power-law is ",PLfit[0]," +- ",PLerr[0])
            
            print("Schechter function best fits:")
            string = "& {0:1.2f} $\pm$ {1:1.2f} & {2:1.2e} $\pm$ {3:1.2e} & {4:1.2e} & {5:1.2e}"\
                .format(fit[1],fiterr[1],fit[2],fiterr[2],chi2/Sndf,Fpvalue)
            print(string)
            
        if plot:
            if i==0:
                l1=plt.errorbar(bcs,h/norm,yerr = MCbounds/norm,linestyle="",marker=markers[i],label=labels[i])
                l11,=plt.plot(bcs,integrateSchechter(bcs,*fit)/norm,
                    color=plt.gca().lines[-1].get_color(),linestyle="-",label=labels[i]+"(fit)")
            elif i==1:
                l2=plt.errorbar(bcs,h/norm,yerr = MCbounds/norm,linestyle="",marker=markers[1],label="$z_{\\rm DM},z_{\\rm min}=2\\,$Mpc")
                l22,=plt.plot(bcs,integrateSchechter(bcs,*fit)/norm,
                    color=plt.gca().lines[-1].get_color(),linestyle="-.",label="$z_{\\rm DM},z_{\\rm min}$ (fit)")
            elif i==2:
                l3=plt.errorbar(bcs,h/norm,yerr = MCbounds/norm,linestyle="",marker=markers[2],label="$z_{\\rm DM},0.1 z_{\\rm max}$")
                l33,=plt.plot(bcs,integrateSchechter(bcs,*fit)/norm,
                    color=plt.gca().lines[-1].get_color(),linestyle="--",label="$z_{\\rm DM},0.1 z_{\\rm max}$ (fit)")
            elif i==11:
                l4=plt.errorbar(bcs,h/norm,yerr = MCbounds/norm,linestyle="",marker=markers[3],label="$z_{\\rm DM},z_{\\rm max}$")
                l44,=plt.plot(bcs,integrateSchechter(bcs,*fit)/norm,
                    color=plt.gca().lines[-1].get_color(),linestyle=":",label="$z_{\\rm DM},z_{\\rm max}$ (fit)")
    
    
    
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
        
        
        plt.xlim(1e23,1e27)
        plt.ylim(1e-4,10)
        
        # original version with 4 lines
        if Truncate:
            plt.legend(loc="upper right",frameon=False,handles=[l11,l22])
        else:
            plt.legend(loc="upper right",frameon=False,handles=[l11,l22,l33,l44])
        
        plt.tight_layout()
        plt.savefig("zoom_paper_luminosity_unbiased.pdf")
        
        ax=plt.gca()
        ax.lines.remove(l11)
        ax.lines.remove(l22)
        if not Truncate:
            ax.lines.remove(l33)
            ax.lines.remove(l44)
        
        
        if Truncate:
            l5,=plt.plot([3e19,3e23],[1e6*1.5,1*1.5],linestyle=":",label="$E_\\nu {\\rm RLF}(E_\\nu) \\propto E_\\nu^{-1.5}$")
        else:
            l5,=plt.plot([1e19*0.3,1e23*0.3],[1e6*2,1*2],linestyle=":",label="$E_\\nu {\\rm RLF}(E_\\nu) \\propto E_\\nu^{-1.5}$")
        
        if Truncate:
            plt.legend(loc="upper right",frameon=False,handles=[l1,l2,l5])
        else:
            plt.legend(loc="upper right",frameon=False,handles=[l1,l2,l3,l4,l5])
        plt.xlim(1e18,1e27)
        plt.ylim(1e-4,1e7)
        plt.tight_layout()
        plt.savefig("paper_luminosity_unbiased.pdf")
        plt.close()
    

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
