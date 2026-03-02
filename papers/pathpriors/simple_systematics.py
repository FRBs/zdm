"""
This file fits the simple (naive) model to CRAFT ICDS observations.
It varies the simple model parameterisation, to determine systematic effects
"""


#standard Python imports
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# imports from the "FRB" series
from zdm import optical as opt
from zdm import optical_params as op
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading
from zdm import optical_numerics as on
from zdm import states

# other FRB library imports
import astropath.priors as pathpriors

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)




def main():
    """
    Main routine
    Loops over parameterisations, and plots best fits
    """
    
    ######### List of all ICS FRBs for which we can run PATH #######
    # hard-coded list of FRBs with PATH data in ice paper
    frblist=opt.frblist
    
    # Initlisation of zDM grid
    # Eventually, this should be part of the loop, i.e. host IDs should
    # be re-fed into FRB surveys. However, it will be difficult to do this
    # with very limited redshift estimates. That might require posterior
    # estimates of redshift given the observed galaxies. Maybe.
    state = states.load_state("HoffmannHalo25",scat=None,rep=None)
    
    # initialise cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names,init_state=state)
    
    
    NBinList = [6,11,21]
    AbsminList = np.linspace(-26,-25,11)
    AbsmaxList = AbsminList + 10. # increases max, but not in same way
    
    plt.figure()
    plt.xlabel("Absolute magnitude, $M_r$")
    plt.ylabel("$p(M_r)$")
    
    count = 0
    
    opdir = "simple_systematics/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    load = False
    
    colours = ["grey","orange","blue"]
    markers = ['o','x','s']
    xlist = []
    llstats=[]
    ksstats=[]
    kvals=[]
    for i,NModelBins in enumerate(NBinList):
        llstats.append([])
        ksstats.append([])
        kvals.append([])
        for j,Absmin in enumerate(AbsminList):
            Absmax = AbsmaxList[j]
            fname1 = opdir + "bins_"+str(count)+".npy"
            fname2 = opdir + "allx_"+str(count)+".npy"
            
            
            print("Doing model with ",Absmin,Absmax,NModelBins)
            
            opstate = op.OpticalState()
            # sets optical state to use simple linear interpolation
            opstate.simple.AbsModelID = 1
            opstate.simple.AppModelID = 1 # k-correction
            opstate.simple.Absmin = Absmin
            opstate.simple.Absmax = Absmax
            opstate.simple.NModelBins = NModelBins
            
            if load:
                bins = np.load(fname1)
                allx = np.load(fname2)
                model = opt.simple_host_model(opstate)
            else:
                
                #AbsMags = np.linspace(Ansmin,Absmax,NAbsBins)
                allx,model = get_best_fit(ss,gs,frblist,opstate)
                
                bins = model.ModelBins
                np.save(fname1,bins)
                np.save(fname2,allx)
            
            
            model.init_args(allx)
            wrappers = on.make_wrappers(model,gs)
            NFRB,AppMags,AppMagPriors,ObsMags,ObsPriors,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
            llstat = on.calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior)
            ksstat = on.calculate_ks_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,
                                    sumPUprior)
            
            llstats[i].append(llstat)
            ksstats[i].append(ksstat)
            kvals[i].append(allx[0])
            x = allx[1:]
            xlist.append(x)
            if j==0:
                plt.plot(bins,x/np.sum(x)*NModelBins,color=colours[i],label=str(NModelBins)+" bins",marker=markers[i])
                #plt.plot(bins,x/np.sum(x),marker="o",linestyle="",color=colours[i])
            else:
                plt.plot(bins,x/np.sum(x)*NModelBins,color=colours[i],marker=markers[i])
                #plt.plot(bins,x/np.sum(x),marker="o",linestyle="",color=colours[i])
            
            
            
            count += 1
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"model_systematics.png")
    plt.close()
    
    plt.figure()
    print("kvals")
    for i,NModelBins in enumerate(NBinList):
        plt.plot(kvals[i],label="Nbins = "+str(NModelBins),linestyle="",marker='o')
        print(i,NModelBins,np.mean(kvals[i]),np.std(kvals[i]))
    plt.savefig(opdir+"kvals.png")
    plt.close()
    
    plt.figure()
    print("ks stats")
    for i,NModelBins in enumerate(NBinList):
        plt.plot(ksstats[i],label="Nbins = "+str(NModelBins),linestyle="",marker='o')
        print(i,NModelBins,np.mean(ksstats[i]),np.std(ksstats[i]))
    plt.savefig(opdir+"ksstats.png")
    plt.close()
    
    plt.figure()
    print("llstats")
    for i,NModelBins in enumerate(NBinList):
        plt.plot(llstats[i],label="Nbins = "+str(NModelBins),linestyle="",marker='o')
        print(i,NModelBins,np.mean(llstats[i]),np.std(llstats[i]))
    plt.savefig(opdir+"llstats.png")
    plt.close()
    
    for j,Absmin in enumerate(AbsminList):
        # three tests : 20 vs 10, 10 vs 5, 20 vs 5
        dl02 = -2. * (llstats[0][j]-llstats[2][j])
        dl01 = -2. * (llstats[0][j]-llstats[1][j])
        dl12 = -2. * (llstats[1][j]-llstats[2][j])
        
        ddf02 = 15
        ddf01 = 5
        ddf12 = 10
        
        p02 = 1.-chi2.cdf(dl02,ddf02)
        p12 = 1.-chi2.cdf(dl12,ddf12)
        p01 = 1.-chi2.cdf(dl01,ddf01)
        print(j, "th offset: p-values for 5-10, 10-20, and 5-20 are ",p01,p12,p02, " with stats ",dl01,dl12,dl02)
    
def get_best_fit(ss,gs,frblist,opstate):
    """
    Fits simple model parameterised by:
    
    Args:
        ss: list of survey objects
        gs: list of grid objects
        frblist: list of FRBs to process
        opstate: optical state to be used in modelling
    
    Returns:
        Best-fit parameters
    
    """
    
    
    ######## Determnine which statistic to use in optimisation ########
    # setting istat=0 means using a ks statistic to fit p(m_r)
    # setting istat=1 means using a maximum likelihood estimator
    istat=1
    # dok=True means use the k-correction
    dok = True
    # we are using the simple model
    modelname = "simple"
    # set to e.g. 0.9 to reject FRBs with lower posteriors when doing model comparisons
    POxcut = None
    
    
    # Case of simple host model
    # Initialisation of model
    # simple host model
    
    # sets up initial bounds on variables
    if dok:
        opstate.simple.AppModelID = 1 # k-correction
        Nparams = opstate.simple.NModelBins+1
        opstate.simple.AppModelID = 1 # sets to include k-correction
        opstate.simple.k = 0.5 # for some reason, this just doesn't make much difference to results
        bounds = [(-25,25)]+[(0,1)]*(Nparams-1)
    else:
        Nparams = opstate.simple.NModelBins
        # bins now give log-space values, hence -5,2 is range of 10^7
        if opstate.simple.AbsModelID == 3:
            base=(-5,2) # log space
        else:
            base=(0,1) # linear space
        bounds = [base]*(Nparams)
        opstate.simple.AppModelID = 0 # no k-correction
        
    model = opt.simple_host_model(OpticalState = opstate)
    
    # retrieve default initial arguments in vector form
    x0 = model.get_args()
    # initialise aguments to minimisation function
    args=[frblist,ss,gs,model,POxcut,istat]
    
    # set to false to just use hard-coded best fit parameters
    minimise=True
    result = minimize(on.function,x0 = x0,args=args,bounds = bounds)
    x = result.x
    return x,model

main()

