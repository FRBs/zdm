### this file includes various routines to iterate /maximise / minimise
# values on a zdm grid
#import matplotlib.pyplot as plt
import numpy as np
from zdm import pcosmic
from scipy.optimize import minimize
# to hold one of these parameters constant, just remove it from the arg set here
from zdm import cosmology as cos
from zdm import zdm
from zdm import misc_functions
import time
import os
from scipy.stats import poisson
# internal counter
NCF=0

#### this verson currently works with 5 params.
# soon to be upgraded with two values of local DM contribution

def get_likelihood(pset,grid,survey,norm=True,psnr=True):
    """ Returns log-likelihood for parameter set
    """
    #changed this so that calc_likelihood doList=True, helps in debugging while checking likelihoods for different param values 
    if isinstance(grid,list):
        if not isinstance(survey,list):
            raise ValueError("Grid is a list, survey is not...")
        ng=len(grid)
    else:
        ng=1
        ns=1
    if ng==1:
        update_grid(grid,pset,survey)
        if survey.nD==1:
            #loglik=calc_likelihoods_1D(grid,survey,pset,norm=norm,psnr=psnr)
            llsum,lllist,expected=calc_likelihoods_1D(grid,survey,pset,norm=norm,psnr=True,dolist=1)
        elif survey.nD==2:
            #loglik=calc_likelihoods_2D(grid,survey,pset,norm=norm,psnr=psnr)
            llsum,lllist,expected=calc_likelihoods_2D(grid,survey,pset,norm=norm,psnr=True,dolist=1)
        return llsum,lllist,expected
        #negative loglikelihood is NOT returned, positive is.	
    else:
        loglik=0
        for i,g in enumerate(grid):
            s=survey[i]
            update_grid(g,pset,s)
            if s.nD==1:
                #ll=calc_likelihoods_1D(g,s,pset,norm=norm,psnr=psnr)
                llsum,lllist,expected=calc_likelihoods_1D(g,s,pset,norm=norm,psnr=True,dolist=1)
            elif s.nD==2:
                #ll=calc_likelihoods_2D(g,s,pset,norm=norm,psnr=psnr)
                llsum,lllist,expected=calc_likelihoods_2D(g,s,pset,norm=norm,psnr=True,dolist=1)
            loglik += llsum
        return loglik,lllist,expected
        #negative loglikelihood is NOT returned, positive is.	
    

def scan_likelihoods_1D(grid,pset,survey,which,vals,norm=True):
    """ Iterates to best-fit parameter set for a survey with only
    DM information.
    """
    #changed so -loglik as well as list is returned for more efficient debugging
    #set_defaults(grid)
    lscan=np.zeros([vals.size])
    lllist1=[]
    expected1=[]
    for i,val in enumerate(vals):
        pset[which]=val
        #lscan[i]=get_likelihood(pset,grid,survey)
        llsum,lllist,expected=get_likelihood(pset,grid,survey)
        lscan[i]=-llsum
        lllist1.append(lllist)
        expected1.append(expected)
    return lscan,lllist1,expected1
    #return lscan

def get_lnames(which=None):
    ''' Get names of parameters in latex formatting '''
    names=['$E_{\\rm min}$','$E_{\\rm max}$','$\\alpha$','$\\gamma$','$n$','$\\mu_x$','$\\sigma_x$','$\\Phi_0$']
    if which is not None:
        names=names[which]
    return names

def get_names(which=None):
    ''' Returns names of parameter set values '''
    names=['Emin','Emax','alpha','gamma','sfr_n','mu_x','sigma_x','C']
    if which is not None:
        names=names[which]
    return names

def set_defaults(grid):
    # we should make sure each grid does this, do not use "old"!
#    global oldlEmin
#    global oldlEmax
#    global oldalpha
#    global oldgamma
#    global oldsfrn
#    global olddmparams
    
    oldlEmin=np.log10(float(grid.Emin))
    oldlEmax=np.log10(float(grid.Emax))
    oldalpha=grid.alpha
    oldgamma=grid.gamma
    oldsfrn=grid.sfr_n
    old_smean=grid.smear_mean
    old_ssigma=grid.smear_sigma
    C=4.19 # 1 this is NOT a sensible default! #use 4.19 for best fit values to get sensible values
    H0value=grid.H0value
    pset=[oldlEmin,oldlEmax,oldalpha,oldgamma,oldsfrn,old_smean,old_ssigma,C,H0value]
    return pset
    #added H0 as a param

def print_pset(pset):
    """ pset defined as:
    [0]:    log10 Emin
    [1]:    log10 Emax
    [2]:    alpha (nu^-alpha)
    [3]:    gamma (E^gamma)
    [4]:    sfr n
    [5]:    dmx log mean
    [6]:    dmx log sigma #eventually make this a function with various parameters
    [7]:    constant \Phi) ('C') of Nfrb
    """
    print("Log_10 (Emin) : ",pset[0])
    print("Log_10 (Emax) : ",pset[1])
    print("alpha:        : ",pset[2])
    print("gamma:        : ",pset[3])
    print("sfr scaling n : ",pset[4])
    print("DMx params    : ",pset[5:7])
    print("Log_10 (C)    : ",pset[7])
    if pset[8] is not None:
        print("H0            : ",pset[8])
    #print("DMx sigma     : ",pset[6])
    #added H0 as a param

def maximise_likelihood(grid,survey):
    # specifies which set of parameters to pass to the dmx function
    
    if isinstance(grid,list):
        if not isinstance(survey,list):
            raise ValueError("Grid is a list, survey is not...")
        ng=len(grid)
        ns=len(survey)
        if ng != ns:
            raise ValueError("Number of grids and surveys not equal.")
        pset=set_defaults(grid[0]) # just chooses the first one
    else:
        ng=1
        ns=1
        pset=set_defaults(grid)
    
    # fixed alpha=1.6 (Fnu ~ nu**-alpha), Emin 10^30 erg, sfr_n > 0
    eq_cons = {'type': 'eq',
        'fun': lambda x: np.array([x[2]-1.6,x[0]-30]),
        'jac': lambda x: np.array([[0,0,1.0,0,0,0,0,0],[1,0,0,0,0,0,0,0]])
        }
    
    # holds sfr_n >0
    # also holds Emax > 1e40
    ineq_cons = {'type': 'ineq',
        'fun': lambda x: np.array([x[4],x[1]-40]),
        'jac': lambda x: np.array([[0,0,0,0,1,0,0,0],[0,1,0,0,0,0,0,0]])
        }
    
    bounds=((None,None),(39,44),(0,5),(-3,-0.1),(0,3),(0,3),(0,2),(None,None))
    
    # these 'arguments' get sent to the likelihood function
    #results=minimize(get_likelihood,pset,args=(grid,survey),constraints=[eq_cons,ineq_cons],method='SLSQP',tol=1e-10,options={'eps': 1e-4},bounds=bounds)
    results=minimize(get_likelihood,pset,args=(grid,survey),method='L-BFGS-B',tol=1e-10,options={'eps': 1e-4},bounds=bounds)
    
    #print("Results from minimisation are ",results)
    #print("Best-fit values: ",results["x"])
    return results


def update_grid(grid,pset,survey):
    #added routine to update H0 values within grid
    """gets the likelihood
    
    Hierarchy:
    Each indent corresponds to one 'level'.
    This is used in the program control below
    to dictate how far each tree should proceed
    in calculation.
    Direct variable inputs are always listed first
    We see that sfr evolution and dm smearing
    lie just before the pdv step
    Hence, we deal with these first, and
    calc rates as a final step regardless
    of what else has changed.
    
    calc_rates:
        calc_pdv
            Emin
            Emax
            gamma
            calc_thresholds
                F0
                alpha
                bandwidth
        set_evolution
            sfr_n
        
        smear_grid:
            grid
            mask
                dmx_params
    
    Note that the grid is independent of the constant C (trivial dependence)
    """
    
    lEmin=pset[0]
    lEmax=pset[1]
    Emin=10.**lEmin # may cause problems when going forwards and backwards in Emin, Emax... store log values?
    Emax=10.**lEmax
    alpha=pset[2]
    gamma=pset[3]
    sfrn=pset[4]
    H0value=pset[8]

    # assumes all parameters following this are passed to DM distribution
    smear_mean=pset[5]
    smear_sigma=pset[6]

    #taking all the old grid parameters
    oldmask=grid.smear
    oldsmeargrid=grid.smear_grid
    
    oldlEmin=np.log10(float(grid.Emin))
    oldlEmax=np.log10(float(grid.Emax))
    oldalpha=grid.alpha
    oldgamma=grid.gamma
    oldsfrn=grid.sfr_n
    oldsmean=grid.smear_mean
    oldssigma=grid.smear_sigma
    oldH0value=grid.H0
    new_sfr_smear=False #do not need to calculate new product of sfr and smearing
    
    ##check if H0 value changed
    
    if oldH0value != pset[8]:
        cos.set_cosmology(H0=H0value)
        zDMgrid, zvals,dmvals,H0=misc_functions.get_zdm_grid(H0=H0value,new=True,plot=False,method='analytic')
        grid.pass_grid(zDMgrid,zvals,dmvals,H0)
        grid.smear_mean=oldsmean
        grid.smear_sigma=oldssigma
        grid.smear_dm(oldmask,oldsmean,oldssigma)
        grid.calc_thresholds(survey.meta['THRESH'],survey.efficiencies,alpha=oldalpha, weights=survey.wplist)
        grid.calc_dV()
        grid.calc_pdv(Emin,Emax,gamma,survey.beam_b,survey.beam_o)
        grid.set_evolution(oldsfrn) 
        grid.calc_rates()

    ##### checks to see if the dmx parameters have changed ######
    if oldsmean != pset[5] or oldssigma != pset[6]:
        changed_smear=True
        mask=pcosmic.get_dm_mask(grid.dmvals,[smear_mean,smear_sigma],grid.zvals)
        grid.smear_dm(mask,smear_mean,smear_sigma)
        # now have to perform the calc_rates step again
        # must get correct pdv and sfr first
        new_sfr_smear=True
        oldssigma=smear_sigma
        oldsmean=smear_mean
    
    ###### checks to see if the sfr parameters have changed ######
    if grid.alpha_method==0:
        if sfrn != oldsfrn:
            grid.set_evolution(sfrn)
            new_sfr_smear=True
            oldsfrn=sfrn
    elif grid.alpha_method==1:
        if sfrn != oldsfrn or alpha != oldalpha:
            grid.set_evolution(sfrn,alpha=alpha)
            new_sfr_smear=True
            oldsfrn=sfrn
            oldalpha=alpha
    
    ##### examines the 'pdv tree' affecting sensitivity #####
    # begin with alpha
    # alpha does not change thresholds under rate scaling, only spec index
    if grid.alpha_method==0 and alpha != oldalpha:
        grid.calc_thresholds(grid.F0,grid.eff_table,alpha,grid.bandwidth,grid.eff_weights)
        changed_alpha=True
        oldalpha=alpha
    else:
        changed_alpha=False
    
    # handles for potential future use, currently any change requires this
    if lEmin != oldlEmin:
        clEmin=True
    else:
        clEmin=False
    
    if lEmax != oldlEmax:
        clEmax=True
    else:
        clEmax=False
    
    if gamma != oldgamma:
        clgamma=True
    else:
        clgamma=False
    
    # checks if we need to calculate new pdv: this depends on thresholds, i.e. alpha
    # OK now we need to be careful, and distinguish between the different changes
    # note that changed_alpha is always 0 if alpha_method=1
    if changed_alpha or clEmin or clEmax or clgamma:
        grid.calc_pdv(Emin,Emax,gamma)#,newbg=changed_alpha,newEmin=clEmin,newEmax=clEmax,newGamma=clgamma)
        oldlEmin=lEmin
        oldlEmax=lEmax
        oldgamma=gamma
        new_pdv_smear=True
    else:
        new_pdv_smear=False
    
    if new_sfr_smear:
        grid.calc_rates() #includes sfr smearing factors and pdv mult
    elif new_pdv_smear:
        grid.rates=grid.pdv*grid.sfr_smear #does pdv mult only, 'by hand'
    

def calc_likelihoods_1D(grid,survey,pset,doplot=False,norm=True,psnr=False,Pn=True,dolist=0):
    """ Calculates 1D likelihoods using only observedDM values
    Here, Zfrbs is a dummy variable allowing it to be treated like a 2D function
    for purposes of calling.
    
    Norm simply means to normalise likelihoods so that the total comes to unity.
        - Note that the *sum* comes to unity, since each bin in rates is already
            normalised by the volume in the dz bin
    
    Pn: Calculate the probability of observing N bursts (Poisson)
    """
    rates=grid.rates
    dmvals=grid.dmvals
    zvals=grid.zvals
    DMobs=survey.DMEGs
    
    # start by collapsing over z
    pdm=np.sum(rates,axis=0)
    
    ddm=dmvals[1]-dmvals[0]
    kdms=DMobs/ddm
    idms1=kdms.astype('int')
    idms2=idms1+1
    dkdms=kdms-idms1
    pvals=pdm[idms1]*(1.-dkdms) + pdm[idms2]*dkdms
    #print(idms1)
    #print(dkdms)
    if norm:
        global_norm=np.sum(pdm)
        log_global_norm=np.log10(global_norm)
        #pdm /= global_norm
    else:
        log_global_norm=0
    
    # holds individual FRB data
    longlist=np.log10(pvals)-log_global_norm
    
    # sums over all FRBs for total likelihood
    llsum=np.sum(np.log10(pvals))-log_global_norm*DMobs.size
    lllist=[llsum]
    
    ### Assesses total number of FRBs ###
    if Pn and (survey.TOBS is not None):
        expected=CalculateIntegral(grid,survey)
        expected *= 10**pset[7]
        observed=survey.NORM_FRB
        Pn=Poisson_p(observed,expected)
        Nll=np.log10(Pn)
        lllist.append(Nll)
        llsum += Nll
    else:
        lllist.append(0)
        expected=0
    #print("Calculated Pn as ",Pn,expected,survey.TOBS)
    # this is updated version, and probably should overwrite the previous calculations
    if psnr:
        # NOTE: to break this into a p(SNR|b) p(b) term, we first take
        # the relative likelihood of the threshold b value compared
        # to the entire lot, and then we calculate the local
        # psnr for that beam only. But this requires a much more
        # refined view of 'b', rather than the crude standatd 
        # parameterisation
        
        # calculate vector of grid thresholds
        Emax=grid.Emax
        Emin=grid.Emin
        gamma=grid.gamma
        psnr=np.zeros([survey.Ss.size])
        
        # get vector of thresholds as function of z and threshold/weight list
        # note that the dimensions are, nthresh (weights), z, DM
        Eths = grid.thresholds[:,:,idms1]*(1.-dkdms)+ grid.thresholds[:,:,idms2]*dkdms
        
        ##### IGNORE THIS, PVALS NOW CONTAINS CORRECT NORMALISATION ######
        # we have previously calculated p(DM), normalised by the global sum over all DM (i.e. given 1 FRB detection)
        # what we need to do now is calculate this normalised by p(DM),
        # i.e. psnr is the probability of snr given DM, and hence the total is
        # p(snr,DM)/p(DM) * p(DM)/b(burst)
        # get a vector of rates as a function of z
        #rs = rates[:,idms1[j]]*(1.-dkdms[j])+ rates[:,idms2[j]]*dkdms[j]
        rs = rates[:,idms1]*(1.-dkdms)+ rates[:,idms2]*dkdms	
        #norms=np.sum(rs,axis=0)/global_norm
        norms=pvals
        
        zpsnr=np.zeros(Eths.shape[1:])
        beam_norm=np.sum(survey.beam_o)
        #in theory, we might want to normalise by the sum of the omeba_b weights, although it does not matter here
        
        
        for i,b in enumerate(survey.beam_b):
            #iterate over the grid of weights
            bEths=Eths/b #this is the only bit that depends on j, but OK also!
            #now wbEths is the same 2D grid
            #wbEths=bEths #this is the only bit that depends on j, but OK also!
            bEobs=bEths*survey.Ss #should correctky multiply the last dimensions
            for j,w in enumerate(grid.eff_weights):
                temp=(grid.array_diff_lf(bEobs[j,:,:],Emin,Emax,gamma).T*grid.FtoE).T
                zpsnr += temp*survey.beam_o[i]*w #weights this be beam solid angle and efficiency
        
        
        # we have now effectively calculated the local probabilities in the source-counts histogram for a given DM
        # we have to weight this by the sfr_smear factors, and the volumetric probabilities
        # this are the grid smearing factors incorporating pcosmic and the host contributions
        sg = grid.sfr_smear[:,idms1]*(1.-dkdms)+ grid.sfr_smear[:,idms2]*dkdms
        sgV = (sg.T*grid.dV.T).T
        wzpsnr = zpsnr * sgV
        #THIS HAS NOT YET BEEN NORMALISED!!!!!!!!
        # at this point, wzpsnr should look exactly like the grid.rates, albeit
        # A: differential, and 
        # B: slightly modified according to observed and not threshold fluence
        
        # normalises for total probability of DM occurring in the first place.
        # We need to do this. This effectively cancels however the Emin-Emax factor.
        # sums down the z-axis
        psnr=np.sum(wzpsnr,axis=0)
        psnr /= norms #normalises according to the per-DM probability
        
        # keeps individual FRB values
        longlist += np.log10(psnr)
        
        # checks to ensure all frbs have a chance of being detected
        bad=np.array(np.where(psnr == 0.))
        if bad.size > 0:
            snrll = float('NaN') # none of this is possible! [somehow...]
        else:
            snrll = np.sum(np.log10(psnr))
        lllist.append(snrll)
        llsum += snrll
        
        if doplot:
            fig1=plt.figure()
            plt.xlabel('z')
            plt.ylabel('p(SNR | DM,z)p(z)')
            #plt.xlim(0,1)
            tm=0.
            plt.yscale('log')
            
            fig4=plt.figure()
            plt.xlabel('z')
            plt.ylabel('p(DM,z)p(z)')
            #plt.xlim(0,1)
            plt.yscale('log')
            
            fig2=plt.figure()
            plt.xlabel('z')
            plt.ylabel('p(SNR | DM,z)')
            #plt.xlim(0,1)
            plt.yscale('log')
            
            fig3=plt.figure()
            plt.xlabel('z')
            plt.ylabel('p(z)')
            #plt.xlim(0,1)
            
            tm1=np.max(wzpsnr)
            tm2=np.max(rs)
            for j in np.arange(survey.Ss.size):
                linestyle='-'
                if j>=survey.Ss.size/4:
                    linestyle=':'
                if j>=survey.Ss.size/2:
                    linestyle='--'
                if j>=3*survey.Ss.size/4:
                    linestyle='-.'
                
                plt.figure(fig1.number)
                plt.plot(zvals,wzpsnr[:,j],label=str(int(DMobs[j])),linestyle=linestyle,linewidth=1)
                
                plt.figure(fig4.number)
                plt.plot(zvals,rs[:,j],label=str(int(DMobs[j])),linestyle=linestyle,linewidth=2)
                
                plt.figure(fig2.number)
                plt.plot(zvals,zpsnr[:,j],label=str(int(DMobs[j])),linestyle=linestyle)
                
                plt.figure(fig3.number)
                plt.plot(zvals,sgV[:,j],label=str(int(DMobs[j])),linestyle=linestyle)
                
            fig1.legend(ncol=2,loc='upper right',fontsize=8)
            fig1.tight_layout()
            plt.figure(fig1.number)
            plt.ylim(tm1/1e5,tm1)
            fig1.savefig('TEMP_p_wzpsnr.pdf')
            plt.close(fig1.number)
            
            fig4.legend(ncol=2,loc='upper right',fontsize=8)
            fig4.tight_layout()
            plt.figure(fig4.number)
            plt.ylim(tm2/1e5,tm2)
            fig4.savefig('TEMP_p_z.pdf')
            plt.close(fig4.number)
            
            fig2.legend(ncol=2,loc='upper right',fontsize=8)
            fig2.tight_layout()
            fig2.savefig('TEMP_p_zpsnr.pdf')
            plt.close(fig2.number)
            
            fig3.legend(ncol=2,loc='upper right',fontsize=8)
            fig3.tight_layout()
            fig3.savefig('TEMP_p_sgV.pdf')
            plt.close(fig3.number)
            
            print("Total snr probabilities")
            for i,p in enumerate(psnr):
                print(i,survey.Ss[i],p)
        
        
    else:
        lllist.append(0)
    if doplot:
        plt.figure()
        plt.plot(dmvals,pdm,color='blue')
        plt.plot(DMobs,pvals,'ro')
        plt.xlabel('DM')
        plt.ylabel('p(DM)')
        plt.tight_layout()
        plt.savefig('Plots/1d_dm_fit.pdf')
        plt.close()
    
    if dolist==0:
        return llsum
    elif dolist==1:
        return llsum,lllist,expected
    elif dolist==2:
        return llsum,lllist,expected,longlist
    

def calc_likelihoods_2D(grid,survey,pset,doplot=False,norm=True,psnr=False,printit=False,Pn=True,dolist=0):
    """ Calculates 2D likelihoods using observed DM,z values """
    
    ######## Calculates p(DM,z | FRB) ########
    # i.e. the probability of a given z,DM assuming
    # an FRB has been observed. The normalisation
    # below is proportional to the total rate (ish)
    
    rates=grid.rates
    zvals=grid.zvals
    dmvals=grid.dmvals
    
    DMobs=survey.DMEGs
    Zobs=survey.Zs
    #if survey.meta["TOBS"] is not None:
    #	TotalRate=np.sum(rates)*survey.meta["TOBS"]
        # this is in units of number per MPc^3 at Emin
    
    # normalise to total probability of 1
    if norm:
        #norm = np.log10(np.sum(rates))
        #norm *= Zobs.size # one for each and every measurement - same normalisation value
        norm=np.sum(rates) # gets multiplied by event size later
    else:
        norm=1.
    
    # get indices in dm space
    ddm=dmvals[1]-dmvals[0]
    kdms=DMobs/ddm
    idms1=kdms.astype('int')
    idms2=idms1+1
    dkdms=kdms-idms1
    
    # get indices in z space
    dz=zvals[1]-zvals[0]
    kzs=Zobs/dz
    izs1=kzs.astype('int')
    izs2=izs1+1
    dkzs=kzs-izs1
    
    pvals = rates[izs1,idms1]*(1.-dkdms)*(1-dkzs)
    pvals += rates[izs2,idms1]*(1.-dkdms)*dkzs
    pvals += rates[izs1,idms2]*dkdms*(1-dkzs)
    pvals += rates[izs2,idms2]*dkdms*dkzs
    
    bad=np.array(np.where(pvals <= 0.))
    if bad.size > 0:
        pvals[bad]=1e-20 # hopefully small but not infinitely so
    
    
    # holds individual FRB data
    longlist=np.log10(pvals)-np.log10(norm)
    
    llsum=np.sum(np.log10(pvals))#-norm
    llsum -= np.log10(norm)*Zobs.size # once per event
    lllist=[llsum]
    
    
    if Pn and (survey.TOBS is not None):
        expected=CalculateIntegral(grid,survey)
        expected *= 10**pset[7]
        observed=survey.NORM_FRB
        Pn=Poisson_p(observed,expected)
        Pll=np.log10(Pn)
        lllist.append(Pll)
        llsum += Pll
    else:
        expected=0
        lllist.append(0)
    
    # plots figures as appropriate
    if doplot:
        plt.figure()
        plt.plot(dmvals,pdm,color='blue')
        plt.plot(DMobs,pvals,'ro')
        plt.xlabel('DM')
        plt.ylabel('p(DM)')
        plt.tight_layout()
        plt.savefig('Plots/1d_dm_fit.pdf')
        plt.close()
    
    ###### Calculates p(E | z,DM) ########
    # i.e. the probability of observing an FRB
    # with energy E given redshift and DM
    # this calculation ignores beam values
    # this is the derivative of the cumulative distribution
    # function from Eth to Emax
    # this does NOT account for the probability of
    # observing something at a relative sensitivty of b, i.e. assumes you do NOT know localisation in your beam...
    # to do that, one would calculate this for the exact value of b for that event. The detection
    # probability has already been integrated over the full beam pattern, so it would be trivial to
    # calculate this in one go. Or in other words, one could simple add in survey.Bs, representing
    # the local sensitivity to the event [keeping in mind that Eths has already been calculated
    # taking into account the burst width and DM, albeit for a mean FRB]
    # Note this would be even simpler than the procedure described here - we just
    # use b! Huzzah! (for the beam)
    # IF:
    # - we want to make FRB width analogous to beam, THEN
    # - we need an analogous 'beam' (i.e. width) distribution to integrate over,
    #     which gives the normalisation
    if psnr:
        # NOTE: to break this into a p(SNR|b) p(b) term, we first take
        # the relative likelihood of the threshold b value compare
        # to the entire lot, and then we calculate the local
        # psnr for that beam only. But this requires a much more
        # refined view of 'b', rather than the crude standatd 
        # parameterisation
        
        # calculate vector of grid thresholds
        Emax=grid.Emax
        Emin=grid.Emin
        gamma=grid.gamma
        #Eths has dimensions of width likelihoods and nobs
        # i.e. later, the loop over j,w uses the first index
        Eths = grid.thresholds[:,izs1,idms1]*(1.-dkdms)*(1-dkzs)
        Eths += grid.thresholds[:,izs2,idms1]*(1.-dkdms)*dkzs
        Eths += grid.thresholds[:,izs1,idms2]*dkdms*(1-dkzs)
        Eths += grid.thresholds[:,izs2,idms2]*dkdms*dkzs
        
        FtoE = grid.FtoE[izs1]*(1.-dkzs)
        FtoE += grid.FtoE[izs2]*dkzs
        
        beam_norm=np.sum(survey.beam_o)
        
        # now do this in one go
        # We integrate p(snr|b,w) p(b,w) db dw. I have no idea how this could be multidimensional
        psnr=np.zeros(Eths.shape[1:])
        for i,b in enumerate(survey.beam_b):
            bEths=Eths/b # array of shape NFRB, 1/b
            bEobs=bEths*survey.Ss
            for j,w in enumerate(grid.eff_weights):
                temp=grid.array_diff_lf(bEobs[j,:],Emin,Emax,gamma) * FtoE #one dim in beamshape, one dim in FRB
                
                psnr += temp.T*survey.beam_o[i]*w #multiplies by beam factors and weight
                
        # at this stage, we have the amplitude from diff power law summed over beam and weight
        
        # we only alculate the following sg and V factors to get units to be
        # comparable to the 1D case - otherwise it is superfluous
        sg = grid.sfr_smear[izs1,idms1]*(1.-dkdms)*(1-dkzs)
        sg += grid.sfr_smear[izs2,idms1]*(1.-dkdms)*dkzs
        sg += grid.sfr_smear[izs1,idms2]*dkdms*(1-dkzs)
        sg += grid.sfr_smear[izs2,idms2]*dkdms*dkzs
        dV = grid.dV[izs1]*(1-dkzs) +  grid.dV[izs2]*dkzs
        # at this stage, sg and dV account for the DM distribution and SFR;
        # dV is the volume elements
        # we just multiply these together
        sgV = sg*dV
        wzpsnr = psnr.T*sgV
        
        # this step weights psnr by the volumetric values
        
        ######## NORMALISATION DISCUSSION ######
        # we want to calculate p(snr) dpsnr
        # this must be \int p(snr | w,b) p(w,b) dw,b
        # \int p(snr | detection) p(det|w,b) p(w,b) dw,b
        # to make it an indpeendent factor, and not double-count it, means calculating
        # \int p(snr | detection) dsnr p(det|w,b) p(w,b) dw,b / \int p(det|w,b) p(w,b) dw,b
        # array_diff_power_law simply calculates p(snr), which is the probability amplitude
        # -(gamma*Eth**(gamma-1)) / (Emin**gamma-Emax**gamma )
        # this includes the probability; hence need to account for this
        
        # it is essential that this normalisation occurs for a normalised pvals
        # this normalisation essentially undoes the absolute calculation of the rate, i.e. we are normalising by the total distribution
        # hence we *really* ought to be adding the normalisation to this...
        # the idea here is that p(snr,det)/p(det) * p(det)/pnorm. Hence pvals - which contains
        # the normalisation - should be the un-normalised values.
        
        wzpsnr /= pvals
        
        
        # keeps individual FRB values
        longlist += np.log10(wzpsnr)
        
        # checks to ensure all frbs have a chance of being detected
        bad=np.array(np.where(wzpsnr == 0.))
        if bad.size > 0:
            snrll = float('NaN') # none of this is possible! [somehow...]
        else:
            snrll = np.sum(np.log10(wzpsnr))
        
        lllist.append(snrll)
        llsum += snrll
        if printit:
            for i,snr in enumerate(survey.Ss):
                print(i,snr,psnr[i])
    else:
        lllist.append(0)
    if dolist==0:
        return llsum
    elif dolist==1:
        return llsum,lllist,expected
    elif dolist==2:
        return llsum,lllist,expected,longlist

def check_cube_opfile(run,howmany,opfile):
    """
    Checks to see if this cube has already been attempted, and if so, where to begin recalculating
    """
    
    if os.path.exists(opfile):
        with open(opfile) as f:
            j=0
            for i,l in enumerate(f):
                j+=1
                pass
            starti=j+1
    else:
        starti=0
    
    return starti
    
    
    

def missing_cube_likelihoods(grids,surveys,todo,outfile,norm=True,psnr=True,starti=0):
    """
    grids: list of grids
    surveys: list of surveys corresponding to the grids
    psetmins, maxes: bounds of the pset array
    npoints: number of points in the cube
    run, howmany: calculate from (run-1)*howmany+1 to run*howmany
    outfile: where to place the output
    
    norm is to ensure the probability for each FRB is set to 1
    psnr means adding in the probability of the measured SNR
    
    Because of the way the grids are calculated, the longest step is the "update" step.
    By far the longest part of that is iteration over each 
    
    This has to be redone every time a new value of Emin, Emax, alpha, or gamma is used
    
    We therefore want these parameters to change the slowest, i.e. we should first cycle
    through n, and the smearing parameters.
    
    Minimising C each time using a special routine works well too.
    
    AIM: every run, cycle through the entire range of n,C, and smearing.
    """
    # set up arrays to holds current values
    howmany,NPARAMS=todo.shape
    nsets=len(grids)
    lls=np.zeros([nsets])
    
    
    f=open(outfile,'a+')
    
    t0=time.process_time()
    print("Starting at time ",t0)
    # iterates
    for i in np.arange(howmany):
        #
        
        string=str(i)
        t1=time.process_time()
        
        # set initial values of parameters - although we could easily do this in  the end loop
        # provided that current is updated properly this does not care about re-ordering
        pset=todo[i,:]
        
        
        ### calculates actual values at each point ###
        
        C,llC,lltot=minimise_const_only(pset,grids,surveys)
        pset[7]=C
        
        # for the minimised parameters, set the values
        for j,n in enumerate(pset):
            string +=' {:8.2f}'.format(pset[j])
        
        
        # in theory we could save the following step if we have already minimised by oh well. Too annoying!
        ll=0.
        for j,s in enumerate(surveys):
            update_grid(grids[j],pset,s)
            if s.nD==1:
                func=calc_likelihoods_1D
            else:
                func=calc_likelihoods_2D
            lls[j]=func(grids[j],s,pset,norm=norm,psnr=psnr)
            string += ' {:8.2f}'.format(lls[j]) # for now, we are recording the individual log likelihoods, but not the components
        ll=np.sum(lls)
        string += '{:8.2f}'.format(ll)
        string += '\n'
        t2=time.process_time()
        print("Iteration ",i," took ",t2-t1," seconds")
        # write output: parameters, likelihoods (and components)
        f.write(string)
        
        t1=time.process_time()
        #print("Loop: ",i+run*howmany," took ", t1-t0," seconds")
        t0=t1
        
    f.close()


def cube_likelihoods(grids,surveys,psetmins,psetmaxes,npoints,run,howmany,outfile,norm=True,psnr=True,starti=0,clone=None):
    """
    grids: list of grids
    surveys: list of surveys corresponding to the grids
    psetmins, maxes: bounds of the pset array
    npoints: number of points in the cube
    run, howmany: calculate from (run-1)*howmany+1 to run*howmany
    outfile: where to place the output
    
    norm is to ensure the probability for each FRB is set to 1
    psnr means adding in the probability of the measured SNR
    
    Because of the way the grids are calculated, the longest step is the "update" step.
    By far the longest part of that is iteration over each 
    
    This has to be redone every time a new value of Emin, Emax, alpha, or gamma is used
    
    We therefore want these parameters to change the slowest, i.e. we should first cycle
    through n, and the smearing parameters.
    
    Minimising C each time using a special routine works well too.
    
    AIM: every run, cycle through the entire range of n,C, and smearing.
    
    # the output for each one is:
    # total likelihood
    # likelihood for DM/z
    # likelihood for N events
    # likelihood for snr
    # expected number for best-fit constant
    
    """
    
    # check feasible range of job number
    if (howmany <= 0) or (run <= 0) or np.prod(np.abs(npoints)) < (run-1)*howmany+1:
        print("Invalid range of run=",run," and howmany=",howmany," for ",np.prod(npoints)," points")
        exit()
    
    NPARAMS=8
    nsets=len(grids)
    lls=np.zeros([nsets])
    
    f=open(outfile,'a+')
    
    ### calculates actual values at each point ###
    psetvals=[]
    
    active=np.zeros([NPARAMS])
    disable=[]
    for ip,n in enumerate(npoints):
        if n==-1: #code to maximise it at each point
            active[ip]=1
            n=1
            npoints[ip]=1
        else:
            disable.append(ip)
        
        vals=np.linspace(psetmins[ip],psetmaxes[ip],n)
        psetvals.append(vals)
    if np.sum(active)>0:
        minimise=True
    else:
        minimise=False
    
    # check to see if we can minimise the constant only
    if active[-1]==1 and np.sum(active)==1:
        minimise=False
        const_only=True
    else:
        const_only=False
    
    ntotal=np.prod(npoints)
    start=(run-1)*howmany
    
    ####### counters for each dimensions ######
    
    # this is the order of fastest to slowest, i.e. we update index
    # order[0] first, then order [1], etc
    # the order can depend on the methods being used
    # note: takes this from the first grid, it should be identical for all grids
    if grids[0].alpha_method==0:
        order=[7,4,5,6,0,1,2,3]
    elif grids[0].alpha_method==0:
        order=[7,4,2,5,6,0,1,3]
    else:
        raise ValueError("Unknown value of alpha method!",grids[0].alpha_method)
            
    r_npoints=npoints[order]
    ndims=npoints.size
    iorder=np.zeros([ndims],dtype='int')
    for i,n in enumerate(order): #creates an inverse to map back, so we know to change the nth parameter 1st
        iorder[n]=i
    
    # gets cumulative factors
    r_cp=np.zeros([ndims])
    for i in np.arange(ndims):
        if i==0:
            r_cp[i]=1
        else:
            r_cp[i] =r_cp[i-1]*r_npoints[i-1]
    
    # sets counters
    this=start
    # current holids the integer index of which set of values we are 'currently' analysing
    # does this in the re-ordered domain
    r_current=np.zeros([ndims],dtype='int')
    for i in np.arange(ndims):
        r_current[ndims-i-1]=int(this/r_cp[ndims-i-1])
        this -= r_current[ndims-i-1]*r_cp[ndims-i-1]
    
    current=r_current[iorder] # sets initial values
    
    t0=time.process_time()
    print("Starting at time ",t0)
    # iterates
    pset=np.zeros([ndims]) #current parameter set
    for i in np.arange(howmany):
        
        print("Testing ",i," of ",howmany," begin at ",starti)
        if i>=starti:
            
            nth=i+(run-1)*howmany
            string=str(nth)
            t1=time.process_time()
            # set initial values of parameters - although we could easily do this in  the end loop
            # provided that current is updated properly this does not care about re-ordering
            for j,n in enumerate(current):
                if active[j]==0 or i==0: #only update if we are not minimising or it's the first time
                    pset[j]=psetvals[j][n]
            
            ### minimise if appropriate ### 
            if minimise:
                C_ll,C_p=my_minimise(pset,grids,surveys,disable=disable,psnr=True,PenParams=None,Verbose=False)
                toset=np.where(active==1)
                for s in toset:
                    pset[s]=C_p[s]
            
            if const_only:
                C,llC,lltot=minimise_const_only(pset,grids,surveys)
                pset[7]=C
            
            
            
            # for the minimised parameters, set the values
            for j,n in enumerate(current):
                string +=' {:8.2f}'.format(pset[j])
            
            
            # in theory we could save the following step if we have already minimised by oh well. Too annoying!
            ll=0.
            
            for j,s in enumerate(surveys):
                if clone is not None and clone[j] > 0:
                    grids[j].copy(grids[clone[j]])
                else:
                    update_grid(grids[j],pset,s)
                if s.nD==1:
                    func=calc_likelihoods_1D
                else:
                    func=calc_likelihoods_2D
                lls[j],alist,expected=func(grids[j],s,pset,norm=norm,psnr=psnr,dolist=1)
                # these are slow operations but negligible in the grand scheme of things
                
                string += ' {:8.2f}'.format(lls[j]) # for now, we are recording the individual log likelihoods, but not the components
                string += ' {:8.2f}'.format(alist[0]) # DM / z value likelihood
                string += ' {:8.2f}'.format(alist[1]) # N events likelihood
                string += ' {:8.2f}'.format(alist[2]) # SNR likelihoods
                string += ' {:8.2f}'.format(expected) # expected number of events
                
            ll=np.sum(lls)
            string += '{:8.2f}'.format(ll)
            string += '\n'
            t2=time.process_time()
            print("Iteration ",nth," took ",t2-t1," seconds")
            # write output: parameters, likelihoods (and components)
            f.write(string)
            
        
        # update current values. Uses re-ordered version. Has to do this even if skipping
        thisdim=0
        r_current[thisdim]+=1
        while(r_current[thisdim]==r_npoints[thisdim]):
            r_current[thisdim]=0
            thisdim+=1
            r_current[thisdim]+=1
        # maps back to original order of things
        current=r_current[iorder]
        
        t1=time.process_time()
        #print("Loop: ",i+run*howmany," took ", t1-t0," seconds")
        t0=t1
        
    f.close()
    
def my_minimise(pset,grids,surveys,steps=None,disable=None,Verbose=False,MaxIter=200,psnr=False,PenTypes=None,PenParams=None):
    """
    Steps, if given, must be a length NPARAMS array, giving initial step size
    """
    NPARAMS=8
    
    ### set up initial step size ###
    steps0=np.full([NPARAMS],0.1) #initial step size in parameters (0.5 is HUGE!)
    
    if len(pset) != NPARAMS:
        raise ValueError("my minimise needs ",NPARAMS," parameters, not ",params)
    
    if steps is not None:
        if len(steps) != NPARAMS:
            raise ValueError("Step size must be None of ",NPARAMS," in length")
    else:
        steps=steps0
    
    
    if not isinstance(grids,list):
        grids=[grids]
    if not isinstance(surveys,list):
        surveys=[surveys]
    
    
    ### check if any parameters should be disabled ###
    active=np.full([NPARAMS],True,dtype='bool')
    if disable is not None:
        for dis in disable:
            active[dis]=False
    
    minstep=np.full([NPARAMS],1e-3) # minimum step size; completes when achieved
    lastsign=np.zeros([NPARAMS],dtype='int') # holds last values of derivs for sign check
    
    
    #print(pset,grid,survey,func,zeds,steps,active,lastsign,minstep)
    niter=0
    ll=0
    careful=False
    if Verbose:
        print("### Begginning iteration ####")
        print(pset)
    while True:
        niter += 1
        ll2,pset2,steps,active,lastsign,careful=step_log_likelihood2(pset,grids,surveys,steps,active,lastsign,minstep,delta=1e-2,dstep=3.,careful=careful,psnr=psnr,PenTypes=PenTypes,PenParams=PenParams,Verbose=Verbose)
        
        if np.isnan(ll):
            print("Got nan on likelihood with parameters ",pset2)
            break #returns old value
        
        # update parameters
        ll=ll2
        pset=pset2
        
        if Verbose:
            string="Iteration "+str(niter)
            string += ' {:8.3}'.format(ll)
            for p in pset:
                string += ' {:8.3}'.format(p)
            for a in active:
                string += ' {:2d}'.format(int(a))
            print(string)
            print("Step sizes ",steps)
            print("Active parameters ",active)
        if niter > MaxIter:
            print("Maximum number of iterations reached,",MaxIter,", reached, returning.")
            break
        if np.all(active==False):
            break
    
    return ll,pset
    
def step_log_likelihood(pset,grids,surveys,step,active,lastsign,minstep,delta=1e-2,dstep=2.,psnr=True,norm=True,careful=False):
    """
    #### currently only set of a signle survey...#####
    pset is the current parameter ste
    grid is the standard grid
    survey is the survey of interest
    
    func is the minimisation function (1 or 2 D)
    
    step the current step size
    
    delta is the size over which to measure derivatives
        this is sufficient accuracy for anything we wish to do...
    dstep is the factor by which the step size is reduced in
        each iteration
    
    norm determines whether or not to normalise the probability (rates) to a value of 1
    This should always be the case, since the total Poisson probability ought to be calculated in
    a separate step.
    """
    NPARAMS=7 # working off 7 parameters here
    # we can easily turn some off to begin with
    
    
    # save initial parameters
    temp_pset=np.copy(pset)
    temp_step=np.copy(step)
    temp_active=np.copy(active)
    temp_lastsign=np.copy(lastsign)
    
    
    # calculate initial likelihood
    loglik=0
    derivs=np.zeros([NPARAMS])
    
    check_careful=False
    NC=np.zeros([NPARAMS])
    
    for j,grid in enumerate(grids):
        # calculates derivatives in all the variables
        s=surveys[j]
        update_grid(grid,pset,s)
        
        # determine correct function: 1 or  D
        if s.nD==1:
            func=calc_likelihoods_1D
        else:
            func=calc_likelihoods_2D
        
        loglik+=func(grid,s,pset,norm=norm,psnr=psnr)
            
        for i in np.arange(NPARAMS):
            if not active[i]: # tests for inactive dimension
                continue
            temp=pset[i]
            pset[i] += delta
            update_grid(grid,pset,s)
            derivs[i]+=func(grid,s,pset,norm=norm,psnr=psnr)
            
            if np.isnan(derivs[i]) or np.isinf(derivs[i]):
                careful=True
                derivs[i]=0. # same as if inactive - temporarily
                check_careful=True
                NC[i] = 1
            pset[i]=temp
    
    # checks in case we can get nowhere
    if np.sum(NC)==np.sum(active):
        print("Hitting infinities in all active parameters")
        return float('NaN'),pset,step,active,lastsign,careful
    
    # if we are being careful, only step in the best direction
    if careful:
        i=np.argmax(np.abs(derivs)) # looks for the direction of biggest improvement
        si=np.sign(derivs[i])
        if si==lastsign[i]:
            pset[i] += step[i]*si
        else:
            lastsign[i]=si
            step[i]/=dstep
            if step[i] < minstep[i]:
                active[i]=0
    else:
        for i in np.arange(NPARAMS):
            if not active[i]:
                continue
            derivs[i] -= loglik
            if derivs[i]==0: # if getting no change, turn this off
                active[i]=0
            si=np.sign(derivs[i])
            if si==lastsign[i]:
                pset[i] += step[i]*si
            else:
                lastsign[i]=si
                step[i]/=dstep
                if step[i] < minstep[i]:
                    active[i]=0
    
    # checks to see if there was any danger...
    careful=check_careful
    
    # checks final log-likelihoods
    loglik=0.
    for j,grid in enumerate(grids):
        # calculates derivatives in all the variables
        s=surveys[j]
        update_grid(grid,pset,s)
        
        # determine correct function: 1 or  D
        if s.nD==1:
            func=calc_likelihoods_1D
        else:
            func=calc_likelihoods_2D
        
        loglik+=func(grid,s,pset,norm=norm,psnr=psnr)
        if np.isnan(loglik):
            # reset parameters, and be careful!
            pset=temp_pset
            step=temp_step
            active=temp_active
            lastsign=temp_lastsign
            careful=True
            loglik=1e-99
            global NCF
            NCF += 1
            if NCF > 5:
                raise ValueError("We are being too careful, and appear to be stuck!")
        else:
            NCF=0
    
    temp_lastsign=np.copy(lastsign)
    
    
    
    # return current array values
    return loglik,pset,step,active,lastsign,careful
    
def step_log_likelihood2(pset,grids,surveys,step,active,lastsign,minstep,delta=1e-2,dstep=2.,psnr=True,norm=True,careful=False,PenTypes=None,PenParams=None,Verbose=False):
    """
    #### currently only set of a signle survey...#####
    pset is the current parameter ste
    grid is the standard grid
    survey is the survey of interest
    
    func is the minimisation function (1 or 2 D)
    
    step the current step size
    
    delta is the size over which to measure derivatives
        this is sufficient accuracy for anything we wish to do...
    dstep is the factor by which the step size is reduced in
        each iteration
        
    This version only ever takes a step is the largest direction
    
    PenTypes lists the penalties, PenParams are the parameters to the function
    """
    NPARAMS=8 # working off 8 parameters here
    # we can easily turn some off to begin with
    
    
    # calculate initial likelihood
    loglik=0
    derivs=np.zeros([NPARAMS,2])
    ngrids=len(grids)
    tlls=np.zeros([NPARAMS,ngrids,2])
    tll=np.zeros([ngrids])
    if Verbose:
        print("Calculating initial likelihoods...")
    # search initial steps
    for j,grid in enumerate(grids):
        # calculates derivatives in all the variables
        s=surveys[j]
        update_grid(grid,pset,s)
        
        # determine correct function: 1 or  D
        if s.nD==1:
            func=calc_likelihoods_1D
        else:
            func=calc_likelihoods_2D
        tll[j]=func(grid,s,pset,norm=norm,psnr=psnr)
        loglik+=tll[j]
        if Verbose:
            print("Likelihood ",j,tll[j])
        for i in np.arange(NPARAMS):
            if not active[i]: # tests for inactive dimension
                derivs[i,0]=-1e99
                derivs[i,1]=-1e99
                tlls[i,j,0]=-1e99
                tlls[i,j,1]=-1e99
                continue
            temp=pset[i]
            pset[i] -= step[i]
            update_grid(grid,pset,s)
            #derivs[i,0] += func(grid,s,pset,norm=norm,psnr=psnr)
            tlls[i,j,0] = func(grid,s,pset,norm=norm,psnr=psnr)
            derivs[i,0] += tlls[i,j,0] 
            if np.isnan(derivs[i,0]) or np.isinf(derivs[i,0]):
                derivs[i,0]=-1e99
            
            pset[i] += 2.*step[i]
            update_grid(grid,pset,s)
            #derivs[i,1] += func(grid,s,pset,norm=norm,psnr=psnr)
            tlls[i,j,1] = func(grid,s,pset,norm=norm,psnr=psnr)
            derivs[i,1] += tlls[i,j,1]
            if np.isnan(derivs[i,1]) or np.isinf(derivs[i,1]):
                derivs[i,1]=-1e99
            pset[i]=temp
        if np.isnan(loglik) or np.isinf(loglik):
                loglik=-1e99
        if Verbose:
            print(tlls[:,j,0]-tll[j])
            print(tlls[:,j,1]-tll[j])
            
    # Handles the penalties on parameters
    loglik += HandlePenalties(pset,PenTypes,PenParams)
    for i in np.arange(NPARAMS):
        if not active[i]: # tests for inactive dimension
            continue
        temp=pset[i]
        pset[i] -= step[i]
        derivs[i,0] += HandlePenalties(pset,PenTypes,PenParams)
        
        pset[i] += 2.*step[i]
        derivs[i,1] += HandlePenalties(pset,PenTypes,PenParams)
        pset[i]=temp
    
    # looks for the best option
    llmax1=np.max(derivs[:,0])
    llmax2=np.max(derivs[:,1])
    if llmax1 > llmax2:
        sign=-1
        dim=np.argmax(derivs[:,0])
        llmax=llmax1
    else:
        sign=1
        dim=np.argmax(derivs[:,1])
        llmax=llmax2
    loc=np.where(derivs==llmax)[0]
    
    # if this is the case, we need to reduce the step size
    if llmax <= loglik:
        for i in np.arange(NPARAMS):
            if active[i]:
                step[i] /= dstep
        if Verbose:
            print("Reducing all step sizes")
    else: # keep running while the running's good
        count=0
        while(1):
            pset[dim] += step[dim]*sign
            if Verbose:
                print("Incrementing dim ",dim," by ",step[dim]*sign)
            #print("Running! ",loglik,pset)
            ll=0
            for j,grid in enumerate(grids):
                # calculates derivatives in all the variables
                s=surveys[j]
                update_grid(grid,pset,s)
                
                # determine correct function: 1 or  D
                if s.nD==1:
                    func=calc_likelihoods_1D
                else:
                    func=calc_likelihoods_2D
                ll+=func(grid,s,pset,norm=norm,psnr=psnr)
                if np.isnan(ll) or np.isinf(ll):
                    pset[dim] -= step[dim]*sign
                    step[dim] /= dstep
                    break
            if ll < loglik:
                pset[dim] -= step[dim]*sign
                step[dim] /= dstep
                break
            elif ll-loglik < 0.001:
                step[dim] /= dstep
                break
            if Verbose:
                print("Old ll was ",loglik," new is ",ll)
            loglik=ll
            
    # deactivates regions where the step size is now too small
    for i in np.arange(NPARAMS):
        if step[i] < minstep[i]:
            active[i]=False
    
    # return current array values
    return loglik,pset,step,active,lastsign,careful

def HandlePenalties(pset,Ptypes,Pparams):
    logpenalty=0
    if not (Ptypes is not None): # checks to see if we are using penalties
        return 0
    for i,param in enumerate(pset):
        if Ptypes[i]==0: #no restriction
            penalty=0 
        elif Ptypes==1: # Gaussian penalty
            penalty=GaussianPenalty(param,Pparams[i])
        logpenalty += penalty # increment penalty
        
def GaussianPenalty(var,params):
    """ Adds a Gaussian log penalty factor to a log-likelihood variable """
    penalty=-0.5*((var-params[0])/params[1])**2
    return penalty


def CalculateMeaningfulConstant(pset,grid,survey,newC=False):
    """ Gets the flux constant, and quotes it above some energy minimum Emin """
    
    # Units: IF TOBS were in yr, it would be smaller, and raw const greater.
    # also converts per Mpcs into per Gpc3
    units=1e9*365.25
    if newC:
        rawconst=CalculateConstant(grid,survey) #required to convert the grid norm to Nobs
    else:
        rawconst=10**pset[-1]
    const = rawconst*units # to cubic Gpc and days to year
    Eref=1e40 #erg per Hz
    Emin=10**pset[0]
    gamma=pset[3]
    factor=(Eref/Emin)**gamma
    const *= factor
    return const

def ConvertToMeaningfulConstant(pset):
    """ Gets the flux constant, and quotes it above some energy minimum Emin """
    
    # Units: IF TOBS were in yr, it would be smaller, and raw const greater.
    # also converts per Mpcs into per Gpc3
    units=1e9*365.25
    
    const = (10**pset[-1])*units # to cubic Gpc and days to year
    Eref=1e40 #erg per Hz
    Emin=10**pset[0]
    Emax=10**pset[1]
    gamma=pset[3]
    factor=(Eref/Emin)**gamma - (Emax/Emin)**gamma
    const *= factor
    return const

def Poisson_p(observed, expected):
    """ returns the Poisson likelihood """
    p=poisson.pmf(observed,expected)
    return p

def CalculateConstant(grid,survey):
    """ Calculates the best-fitting constant for the total
    number of FRBs. Units are:
        - grid volume units of 'per Mpc^3',
        - survey TOBS of 'days',
        - beam units of 'steradians'
        - flux for FRBs with E > Emin
    Hence the constant is 'Rate (FRB > Emin) Mpc^-3 day^-1 sr^-1'
    This should be scaled to be above some sensible value of Emin
    or otherwise made relevant.
    """
    
    expected=CalculateIntegral(grid,survey)
    observed=survey.NORM_FRB
    constant=observed/expected
    return constant

def CalculateIntegral(grid,survey):
    """ Calculates the total expected number of FRBs for that grid and survey """
    
    # check that the survey has a defined observation time
    if survey.TOBS is not None:
        TOBS=survey.TOBS
    else:
        return 0
    
    total=np.sum(grid.rates)
    return total*TOBS
    
def GetFirstConstantEstimate(grids,surveys,pset):
    ''' simple 1D minimisation of the constant '''
    # ensure the grids are uo-to-date
    for i,g in enumerate(grids):
        update_grid(g,pset,surveys[i])
    
    NPARAMS=8
    # use my minimise in a single parameter
    disable=np.arange(NPARAMS-1)
    C_ll,C_p=my_minimise(pset,grids,surveys,disable=disable,psnr=False,PenTypes=None,PenParams=None)
    newC=C_p[-1]
    print("Calculating C_ll as ",C_ll,C_p)
    return newC


def minus_poisson_ps(log10C,data):
    rs=data[0,:]
    os=data[1,:]
    rsp = rs*10**log10C
    lp=0
    for i,r in enumerate(rs):
        Pn=Poisson_p(os[i],r)
        lp += np.log10(Pn)
    return -lp
    

def minimise_const_only(pset,grids,surveys):
    '''
    Only minimises for the constant, but returns the full likelihood
    It treats the rest as constants
    the grids must be initialised at the currect values for pset already
    '''
    
    # specifies which set of parameters to pass to the dmx function
    
    if isinstance(grids,list):
        if not isinstance(surveys,list):
            raise ValueError("Grid is a list, survey is not...")
        ng=len(grids)
        ns=len(surveys)
        if ng != ns:
            raise ValueError("Number of grids and surveys not equal.")
    else:
        ng=1
        ns=1
    
    # calculates likelihoods while ignoring the constant term
    rs=[] #expected
    os=[] #observed
    lls=np.zeros([ng])
    for j,s in enumerate(surveys):
        update_grid(grids[j],pset,s)
        if s.nD==1:
            func=calc_likelihoods_1D
        else:
            func=calc_likelihoods_2D
        lls[j]=func(grids[j],s,pset,norm=True,psnr=True,Pn=False) #excludes Pn term
        
        ### Assesses total number of FRBs ###
        if s.TOBS is not None:
            r=np.sum(grids[j].rates)*s.TOBS
            r*=10**pset[7]
            o=s.NORM_FRB
            rs.append(r)
            os.append(o)
    print("Total sum of ll w/o const is ",np.sum(lls[j]))
    data=np.array([rs,os])
    ratios=np.log10(data[1,:]/data[0,:])
    bounds=(np.min(ratios),np.max(ratios))
    startlog10C=(bounds[0]+bounds[1])/2.
    bounds=[bounds]
    t0=time.process_time()
    result=minimize(minus_poisson_ps,startlog10C,args=data,bounds=bounds)
    t1=time.process_time()
    dC=result.x
    newC=pset[7]+dC
    llC=-minus_poisson_ps(dC,data)
    lltot=llC+np.sum(lls)
    return newC,llC,lltot
    