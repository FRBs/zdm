"""

### WARNING####
This file may be redundant, Histogramming has been supplanted by calc_lf_errors.py
#########

This file reads in V/Vmax values, and creates simple matplotlib
histograms of them.

"""


import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from RLFs import *



import matplotlib
#matplotlib.rcParams['image.interpolation'] = None
defaultsize=16
ds=4
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(SFRs,alphas,prefix="",postfix="",zfrac=None,NREPS=1000):
    
    indir = prefix+"Output/"
    opdir = prefix+"LumData/"
    plotdir = prefix+"LumPlots/"
    
    if not os.path.isdir(opdir):
        os.mkdir(opdir)
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)
    
    # main outer loop
    for i,alpha in enumerate(alphas):
        for j,Nsfr in enumerate(SFRs):
            
            if zfrac is not None:
                zstring="_"+str(zfrac)[0:3]+"_concatenated"
            else:
                zstring=""
            
            infile = indir+postfix+"vvmax_data_NSFR_"+str(Nsfr)[0:3]+"_alpha_"+str(alpha)+zstring+".dat"
            bcs,h,rms,count,analytic,MCbounds = get_lum_function(
                infile,Nreps = NREPS, DoFactor=True)
            
            plotname = plotdir+postfix+"vvmax_data_NSFR_"+str(Nsfr)[0:3]+"_alpha_"+str(alpha)+zstring+".pdf"
            sfit,err,schi2,plfit,plerr,plchi2 = plot_lum_func(bcs,h,MCbounds,count,plotname)
            
            savename = opdir+postfix+"vvmax_data_NSFR_"+str(Nsfr)[0:3]+"_alpha_"+str(alpha)+zstring+".npz"
            
            np.savez(savename,bin_centres=bcs,histogram=h,bootstrap_error = rms,Nfrb=count,
                analytic_error = analytic,MCerror = MCbounds, Schechter = sfit, Serr = err,
                Schi2 = schi2, PowerLaw = plfit,PLerr = plerr, plchi2 = plchi2)
            
            
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
    
    results = curve_fit(logIntegrateSchechter,np.log10(bcs),np.log10(h),p0=RyderGuess,sigma=error,maxfev=1000)
    fit = integrateSchechter(bcs,*results[0])
    # transform erros to logspace
    logerr = 0.5*(1.-MCbounds[0,:]/h) + 0.5*(MCbounds[1,:]/h-1.)
    Schi2 = np.sum((np.log10(h)-np.log10(fit))/logerr)**2
    
    results2 = np.polyfit(np.log10(bcs),np.log10(h),1,cov=True,w=1./error)
    
    #resultsv2 = curve_fit(logIntegratePowerLaw,np.log10(bcs),np.log10(h),p0=RyderGuess[0:2],sigma=error,maxfev=1000)
    #print("Results v2 are ",resultsv2)
    
    # calculates chi2
    fit2 = np.polyval(results2[0],np.log10(bcs))
    chi2 = np.sum(((fit2-np.log10(h))/error)**2 / (h.size-2.))**0.5
    
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
    return results[0],[results[1][0,0],results[1][1,1],results[1][2,2]],Schi2,results2[0],[results2[1][0,0],results2[1][1,1]],chi2
    


       
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
    #Nbins = 5
    #bins = np.logspace(22,27,Nbins+1)
    #bcs = np.logspace(22.5,26.5,Nbins)
    
    # determines number of bins first
    minbin = int(np.floor(np.min(np.log10(Enu))))
    maxbin = int(np.ceil(np.max(np.log10(Enu))))
    Nbins = maxbin-minbin
    bins = np.logspace(minbin,maxbin,Nbins+1)
    bcs = np.logspace(minbin+0.5,maxbin-0.5,Nbins)
    
    h,bins = np.histogram(Enu,bins = bins,weights = weights)
    count,bins = np.histogram(Enu,bins = bins)
    
    OK = np.where(h>0)[0]
    h = h[OK]
    bcs = bcs[OK]
    count = count[OK]
    Nbins = len(OK)
    
    
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
    
    for FRB in [20170428,20210407,20210912,20220610]:
        OK = np.where(FRBs != FRB)
        FRBs = FRBs[OK]
        JHz = JHz[OK]
        zmaxB = zmaxB[OK]
        zmaxC = zmaxC[OK]
        V = V[OK]
        Vmax = Vmax[OK]
        VVmax = VVmax[OK]
    
    return FRBs,JHz,V,Vmax,VVmax 


# standard calculation for localised sample
if True:
    prefix="Loc"
    postfix="localised_"
    NSFR=21
    SFRs = np.linspace(0,2,NSFR)
    alphas=[0,-1.5]
    main(SFRs,alphas,prefix,postfix)
    
# standard calculation for localised sample
if True:
    prefix="zMacquart"
    postfix="v2macquart_"
    NSFR=21
    SFRs = np.linspace(0,2,NSFR)
    alphas=[0,-1.5]
    main(SFRs,alphas,prefix,postfix)

exit()
# we do not produce histograms for the following two cases 
# does the bias-corrected localised plots (limiting zmax to 0.7)
if True:
    prefix="UnbiasedLocalised"
    postfix="localised_"
    NSFR=21
    SFRs = np.linspace(0,2,NSFR)
    alphas=[0,-1.5]    
    main(SFRs,alphas,prefix,postfix)
    #main(loc=False,v2=True)

# calculates errors for the version of the zDM plot accounting for hosts with -ve redhsifts
if True:
    prefix="Minz"
    postfix="Minzmacquart_"
    NSFR=21
    SFRs = [0]
    alphas=[0]
    zfracs=np.linspace(0,1,11)
    for zfrac in zfracs:
        main(SFRs,alphas,prefix,postfix,zfrac=zfrac)
    
