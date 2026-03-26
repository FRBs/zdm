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

def main(loc=True,plot=True):
    
    opdir = "LumData/"
    
    ### infiles
    savefiles = [
        opdir+"localised_vvmax_data_NSFR_0.0_alpha_0.npz",
        opdir+"macquart_vvmax_data_NSFR_0.0_alpha_0.npz",
        opdir+"localised_vvmax_data_NSFR_2.0_alpha_0.npz",
        opdir+"localised_vvmax_data_NSFR_0.0_alpha_-1.5.npz",
        opdir+"localised_vvmax_data_NSFR_2.0_alpha_-1.5.npz"
        ]
    labels = [
        "$z_{\\rm DM}, n_{\\rm SFR}=0, \\alpha=0$",
        "$z_{\\rm loc}, n_{\\rm SFR}=0, \\alpha=0$",
        "$z_{\\rm loc}, n_{\\rm SFR}=2, \\alpha=0$",
        "$z_{\\rm loc}, n_{\\rm SFR}=0, \\alpha=-1.5$",
        "$z_{\\rm loc}, n_{\\rm SFR}=2, \\alpha=-1.5$"
        ]
    
    linestyles = ["-","-","-.","--",":"]
    
    if not plot: #this is just for printing outputs for a table
        savefiles = [
            opdir+"localised_vvmax_data_NSFR_0.0_alpha_0.npz",
            opdir+"localised_vvmax_data_NSFR_0.0_alpha_-1.5.npz",
            opdir+"localised_vvmax_data_NSFR_1.0_alpha_0.npz",
            opdir+"localised_vvmax_data_NSFR_1.0_alpha_-1.5.npz",
            opdir+"localised_vvmax_data_NSFR_2.0_alpha_0.npz",
            opdir+"localised_vvmax_data_NSFR_2.0_alpha_-1.5.npz"
            ]
        
        labels = ["","","","","",""]
        
    #linestyles = ["--",":","-."]
    markers = ['+','s','o','x','^']
    
    lines=[]
    plt.figure()
    
    for i,savefile in enumerate(savefiles):
        data=np.load(savefile)
        bcs = data['bin_centres']
        h = data['histogram']
        MCbounds = data['MCerror']
        fit = data['Schechter']
        fiterr = data['Serr']
        Nfrb = data['Nfrb']
        Ntot = np.sum(Nfrb)
        
        # select bin to normalise to
        ibin = np.where(bcs > 1e23)[0][0]
        norm = h[ibin]
        
        #h /= Ntot
        #fit[0] /= Ntot
        #MCbounds /= Ntot
        
        string = "& {0:1.2f} $\pm$ {1:1.2f} & {2:1.2e} $\pm$ {3:1.2e}".format(fit[1],fiterr[1]**0.5,fit[2],fiterr[2]**0.5)
        #print(fit[1],fiterr[1]**0.5,fit[2],fiterr[2]**0.5)
        print(string)
        if plot:
            plt.errorbar(bcs,h/norm,yerr = MCbounds/norm,linestyle="",marker=markers[i],label=labels[i])
            line,=plt.plot(bcs,integrateSchechter(bcs,*fit)/norm,
                color=plt.gca().lines[-1].get_color(),linestyle=linestyles[i])
            lines.append(line)
    
    handles=[lines[1],lines[0],lines[2],lines[3],lines[4]]
    JamesFit,RyderFit,ShinFit,LuoFit = best_fits(JHz = True)
    RyderGuess = make_first_guess(bcs,h,RyderFit,2)
    JamesGuess = make_first_guess(bcs,h,JamesFit,2)
    ShinGuess = make_first_guess(bcs,h,ShinFit,2)
    LuoGuess = make_first_guess(bcs,h,LuoFit,2)
    
    glabels = ["Ryder et al.", "Shin et al.", "Luo et al."]
    for i,guess in enumerate([RyderGuess,ShinGuess,LuoGuess]):
        ibin = np.where(bcs > 1e23)[0][0]
        toplot = integrateSchechter(bcs,*guess)
        norm = toplot[ibin]
        #if plot:
        #    plt.plot(bcs,toplot/norm,linestyle=linestyles[i],label=glabels[i])
    
    if plot:
        plt.xlabel('$E_{\\nu}$ [J Hz$^{-1}$]')
        plt.ylabel('$E_{\\nu}\\,  {\\rm RLF}(E_{\\nu})$ [arb. units]')
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(1e-9,0.002)
        plt.legend(loc="lower left",fontsize=12,frameon=False,handles=handles,labels=labels)
        plt.tight_layout()
        plt.savefig("paper_luminosity_sfr_alpha.pdf")
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
    plt.ylabel('$E_{\\nu}\\,  {\\rm RLF}(E_{\\nu})$ [arb. units]')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-10,1)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()       
    return results[0],[results[1][0,0],results[1][1,1],results[1][2,2]]
    


       
def get_lum_function(fname, Nreps = 1000, DoFactor = False):
    """
    This function generates a luminosity function from data within
    file 'fname'.
    
    It then generates a binned histogram, using weights of 1/Vmax.
    
    Nreps is the number of iterations to be used for the Monte Carlo
    """
    ####### reads in the data from Wayne's table #######3
    FRBs,Enu,V,Vmax,VVmax = read_data(fname)
    weights = 1./Vmax
    
    # prints sorted values
    # shows how some bins have massive values! ####
    if False:
        iEnu = np.argsort(Enu)
        for i,index in enumerate(iEnu):
            print(Enu[index],weights[index])
    
    
    ######### HISTOGRAMMING ############
    
    # sets the number of histogram bins - this is arbitrary
    Nbins = 5
    bins = np.logspace(22,27,Nbins+1)
    bcs = np.logspace(22.5,26.5,Nbins)
    h,bins = np.histogram(Enu,bins = bins,weights = weights)
    count,bins = np.histogram(Enu,bins = bins)
    
    # we now determine errors in each box with different methods
    # rms is the naive method
    
    rms = np.zeros([Nbins]) # from bootstrap method
    analytic = np.zeros([Nbins]) # std dev of weighted mean method
    rlowers = np.zeros([Nbins]) # lower limit through proper method
    ruppers = np.zeros([Nbins]) # higher limit through proper method
    inverse = np.zeros([2,Nbins])
    
    # prints values of V/Vmax in each
    for ibc,bc in enumerate(bcs):
        # bc is bin centres
        lbc = np.log10(bc)
        lEnu = np.log10(Enu)
        
        # selects FRBs in each bin
        OK = np.where(np.abs(lbc - lEnu)< 0.5)[0]
        
        # generates bootstrap plot
        Nobs = len(OK)
        
        ###### calculates standard bootstrap method #####
        # generates Nreps pseudo-observations
        samp = np.random.poisson(lam=1.0,size = Nreps*Nobs)
        samp = samp.reshape([Nobs,Nreps])
        sums = np.matmul(weights[OK],samp)
        # calculates the rms over all pseudo-samples
        rms[ibc] = (np.sum((sums-h[ibc])**2)/Nreps)**0.5
        
        ##### analytic method ######
        analytic[ibc] = (np.sum(weights[OK]**2))**0.5
        #print(h[ibc],rms[ibc],analytic_error)
        
        ######## Modified bootstrap - sensible! ######
        # generates upper and lower limits
        
        # dfactor is the "resolution" - i.e. how much the estimate changes per iteration
        dfactor = 1.05
        rate = dfactor
        # margin is the max factor between lower and upper bounds for errors that we search
        MARGIN = 2.
        
        # this tries to find an upper limit - how many can there be before we would have 
        # seen more?
        bound1=0.
        bound2=0.
        while True:
            if h[ibc] == 0.:
                rate = 0.
                break
            # generate a pseudosample, and count number which are higher
            samp = np.random.poisson(lam=rate,size = Nreps*Nobs)
            samp = samp.reshape([Nobs,Nreps])
            sums = np.matmul(weights[OK],samp)
            bigger = np.where(sums > h[ibc])[0]
            
            if len(bigger) > Nreps * 0.84135:
                # sets the lower bound: first one that fails
                if bound1 == 0.:
                    bound1 = rate
                else:
                    # check if we are too high:
                    if rate > bound1 * MARGIN:
                        break
            else:
                # track the highest that fails
                bound2 = rate
                
            rate *= dfactor
        
        # we now set the limit to be half-way between the first that fails and
        # the last one that succeeds
        limit = (bound1 + bound2)/2.
        ruppers[ibc] = limit
        
        # resets for lower limit, and repeats the process
        rate = 1./dfactor
        bound1 = 0.
        bound2 = 0.
        while True:
            if h[ibc] == 0.:
                rate = 0.
                break
            samp = np.random.poisson(lam=rate,size = Nreps*Nobs)
            samp = samp.reshape([Nobs,Nreps])
            sums = np.matmul(weights[OK],samp)
            smaller = np.where(sums < h[ibc])[0]
            if len(smaller) > Nreps * 0.84135:
                # sets the lower bound: first one that fails
                if bound1 == 0.:
                    bound1 = rate
                else:
                    # check if we are too high:
                    if rate < bound1 / MARGIN:
                        break
            else:
                # track the lowest that fails
                bound2 = rate
            rate /= dfactor
            
        limit = (bound1 + bound2)/2.
        rlowers[ibc] = limit
        #print("(relative) Rate limits are ",rlowers[ibc],h[ibc],ruppers[ibc])
    
    # this "factor" seems to be a random scaling factor of the histogram
    # it is there *only* to make plots comparable with Wayne's plots
    
    if DoFactor:
        factor = 1e-2 / h[0]
        h *= factor
        rms *= factor
        analytic = analytic * factor
    
    # converts the relative rate limits to actual limits
    # since it multiplies by h, factor included automatically
    inverse[0,:] = (1.-rlowers)*h
    inverse[1,:] = (ruppers-1.)*h
    
    #print(inverse,"\n\n")
    
    return bcs,h,rms,count,analytic,inverse

def make_first_guess(E,L, params,method):
    """
    Makes a first guess by setting the amplitude
    to that of the first bin
    
    Params are always gamma,Emax
    """
    if method==0:
        mag = Schechter(E[0],1.,params[0],params[1])
        guess = [L[0]/mag,params[0],params[1]]
    elif method==1:
        mag = integrateSchechter(E[0],1.,params[0],params[1])
        guess = [L[0]/mag,params[0],params[1]]
    elif method==2:
        # returns the log10 magnitude expected with an amplitude of unity
        mag = logIntegrateSchechter(np.log10(E[0]),1.,params[0],params[1])
        guess = [L[0]/10.**mag,params[0],params[1]]   
    else:
        mag = power_law(E[0],1.,params[0],params[1])
        guess = [L[0]/mag,params[0],params[1]]
    
    
    return guess
    
    
def best_fits(JHz = True):
    """
    Returns best-fit parameters of the Schechter function according to
    different papers.
    
    Only gives gamma and Emax
    
    JHz means divide by 1e7 (erg) and 1e9 (GHz bandwidth)
    """
    JamesFit = [-1.95,41.26]
    RyderFit = [-1.95,41.7]
    ShinFit = [-2.3,41.38]
    LuoFit = [-1.79,41.46]
    for item in [JamesFit,RyderFit,ShinFit,LuoFit]:
        item[1] = 10.**item[1]
        if JHz:
        # convert from erg to J Hz
            item[1] = item[1] / 1e16
    return JamesFit,RyderFit,ShinFit,LuoFit
    
def logIntegrateSchechter(log10E,A,gamma,Emax):
    """
    Returns the logarithms of the IntegrateSchechter function
    Requires log10 E to be sent
    """
    res = integrateSchechter(10.**log10E,A,gamma,Emax)
    res = np.log10(res)
    return res
    
def integrateSchechter(E,A,gamma,Emax):
    """
    Schechter function
    Gamma is differential
    Relative to E0, which is arbitrary
    
    An extra E/E0 due to log binning of histogram
    """
    bw = 10**0.5
    if isinstance(E,np.ndarray):
        y=np.zeros(E.size)
        for i,bc in enumerate(E):
            res = quad(Schechter,bc/bw,bc*bw,args=(A,gamma,Emax))
            y[i]=res[0]
    else:
        res = quad(Schechter,E/bw,E*bw,args=(A,gamma,Emax))
        y = res[0]
    return y

def Schechter(E,A,gamma,Emax):
    """
    Schechter function
    Gamma is differential
    Relative to E0, which is arbitrary
    
    An extra E/E0 due to log binning of histogram
    """
    global E0
    return A*(E/E0)**gamma * np.exp(-E/Emax)


def read_data(infile):
    """
    Reads in output files generated by calc_vvmax.py
    Returns only the V / Vmax values - all that's relevant
    """
    data = np.loadtxt(infile)
    FRBs = data[:,0]
    JHz = data[:,1]
    zmaxB = data[:,2]
    zmaxC = data[:,3]
    V = data[:,4]
    Vmax = data[:,5]
    VVmax = data[:,6]
    
    return FRBs,JHz,V,Vmax,VVmax 



main()
