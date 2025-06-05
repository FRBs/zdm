### this file includes various routines to iterate /maximise / minimise
# values on a zdm grid
import os
import time
from IPython.terminal.embed import embed
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.optimize import minimize
# to hold one of these parameters constant, just remove it from the arg set here
from zdm import cosmology as cos
from scipy.stats import poisson
import scipy.stats as st
from zdm import repeat_grid as zdm_repeat_grid

# internal counter
NCF=0

def get_likelihood(pset,grid,survey,norm=True,psnr=True):
    """ Returns log-likelihood for parameter set
    norm:normalizatiom
    psnr: probability of snr (S/R)
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
            llsum,lllist,expected=calc_likelihoods_1D(grid,survey,norm=norm,psnr=True,dolist=1)
        elif survey.nD==2:
            llsum,lllist,expected=calc_likelihoods_2D(grid,survey,norm=norm,psnr=True,dolist=1)
        elif survey.nD==3:
            # mixture of 1 and 2D samples. NEVER calculate Pn twice!
            llsum1,lllist1,expected1=calc_likelihoods_1D(grid,survey,norm=norm,psnr=True,dolist=1)
            llsum2,lllist2,expected2=calc_likelihoods_2D(grid,survey,norm=norm,psnr=True,dolist=1,Pn=False)
            llsum = llsum1+llsum2
            # adds log-likelihoods for psnrs, pzdm, pn
            # however, one of these Pn *must* be zero by setting Pn=False
            lllist = [lllist1[0]+lllist2[0], lllist1[1]+lllist2[1], lllist1[2]+lllist2[2]] #messy!
            expected = expected1 #expected number of FRBs ignores how many are localsied
        else:
            raise ValueError("Unknown code ",survey.nD," for dimensions of survey")
        return llsum,lllist,expected
        #negative loglikelihood is NOT returned, positive is.	
    else:
        loglik=0
        for i,g in enumerate(grid):
            s=survey[i]
            update_grid(g,pset,s)
            if s.nD==1:
                llsum,lllist,expected=calc_likelihoods_1D(g,s,norm=norm,psnr=True,dolist=1)
            elif s.nD==2:
                llsum,lllist,expected=calc_likelihoods_2D(g,s,norm=norm,psnr=True,dolist=1)
            elif s.nD==3:
                # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                llsum1,lllist1,expected1=calc_likelihoods_1D(g,s,norm=norm,psnr=True,dolist=1)
                llsum2,lllist2,expected2=calc_likelihoods_2D(g,s,norm=norm,psnr=True,dolist=1,Pn=False)
                llsum = llsum1+llsum2
                # adds log-likelihoods for psnrs, pzdm, pn
                # however, one of these Pn *must* be zero by setting Pn=False
                lllist = [lllist1[0]+lllist2[0], lllist1[1]+lllist2[1], lllist1[2]+lllist2[2]]
                expected = expected1 #expected number of FRBs ignores how many are localsied
            else:
                raise ValueError("Unknown code ",s.nD," for dimensions of survey")
            loglik += llsum
        return loglik,lllist,expected
        #negative loglikelihood is NOT returned, positive is.	
    

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


def get_log_likelihood(grid, s, norm=True, psnr=True, Pn=False, pNreps=True):
    """
    Returns the likelihood for the grid given the survey.

    Inputs:
        grid    =   Grid used
        s       =   Survey to compare with the grid
        norm    =   Normalise
        psnr    =   Include psnr in likelihood
        Pn      =   Include Pn in likelihood
    
    Outputs:
        llsum   =   Total loglikelihood for the grid
    """

    if isinstance(grid, zdm_repeat_grid.repeat_Grid):
        # Repeaters
        if s.nDr==1:
            llsum1, lllist, expected = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr, dolist=1, grid_type=1, Pn=Pn, pNreps=pNreps)
            llsum = llsum1
            # print(s.name, "repeaters:", lllist)
        elif s.nDr==2:
            llsum1, lllist, expected = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr, dolist=1, grid_type=1, Pn=Pn, pNreps=pNreps)
            llsum = llsum1
        elif s.nDr==3:
            llsum1, lllist1, expected1 = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr, dolist=1, grid_type=1, Pn=Pn, pNreps=pNreps)
            llsum2, lllist2, expected2 = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr, dolist=1, grid_type=1, Pn=False, pNreps=False)
            llsum = llsum1 + llsum2
        else:
            print("Implementation is only completed for nD 1-3.")
            exit()

        # Singles
        if s.nDs==1:
            llsum1, lllist, expected = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr, dolist=1, grid_type=2, Pn=Pn)
            llsum += llsum1
            # print(s.name, "singles:", lllist)
        elif s.nDs==2:
            llsum1, lllist, expected = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr, dolist=1, grid_type=2, Pn=Pn)
            llsum += llsum1
        elif s.nDs==3:
            llsum1, lllist1, expected1 = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr, dolist=1, grid_type=2, Pn=Pn)
            llsum2, lllist2, expected2 = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr, dolist=1, grid_type=2, Pn=False)
            llsum = llsum + llsum1 + llsum2
        else:
            print("Implementation is only completed for nD 1-3.")
            exit()
    else:
        if s.nD==1:
            llsum1, lllist, expected = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr, dolist=1, Pn=Pn)
            llsum = llsum1
        elif s.nD==2:
            llsum1, lllist, expected = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr, dolist=1, Pn=Pn)
            llsum = llsum1
        elif s.nD==3:
            llsum1, lllist1, expected1 = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr, dolist=1, Pn=Pn)
            llsum2, lllist2, expected2 = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr, dolist=1, Pn=False)
            llsum = llsum1 + llsum2
        else:
            print("Implementation is only completed for nD 1-3.")
            exit()

    return llsum

def calc_likelihoods_1D(grid,survey,doplot=False,norm=True,psnr=True,
                    Pn=False,pNreps=True,ptauw=False,dolist=0,grid_type=0):
    """ Calculates 1D likelihoods using only observedDM values
    Here, Zfrbs is a dummy variable allowing it to be treated like a 2D function
    for purposes of calling.
    
    grid: the grid object calculated from survey
    
    survey: survey object containing the observed z,DM values
    
    doplot: will generate a plot of z,DM values
    
    psnr:
        True: calculate probability of observing each FRB at the observed SNR
        False: do not calculate this

    Pn:
        True: calculate probability of observing N FRBs
        False: do not calculate this
    
    pNreps:
        True: calculate probability of the number of repetitions for each repeater
        False: do not calculate this
    
    ptauw:
        True: calculate probability of intrinsic width and scattering *given* total width
        False: do not calculate this
    
    dolist
        2: llsum,lllist [Pzdm,Pn,Ps],expected,longlist
            longlist holds the LL for each FRB
        5: llsum,lllist,expected,[0.,0.,0.,0.]

    grid_type:
        0: normal zdm grid
        1: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for repeaters
        2: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for single bursts

    """
    
    if ptauw:
        if not survey.backproject:
            print("WARNING: cannot calculate ptauw for this survey, please initialised backproject")
    
    # Determine which array to perform operations on and initialise
    if grid_type == 1: 
        rates = grid.exact_reps 
        if survey.nozreps is not None:
            DMobs=survey.DMEGs[survey.nozreps]
            nozlist=survey.nozreps
        else:
            raise ValueError("No non-localised singles in this survey, cannot calculate 1D likelihoods")
    elif grid_type == 2: 
        rates = grid.exact_singles 
        if survey.nozsingles is not None:
            DMobs=survey.DMEGs[survey.nozsingles]
            nozlist=survey.nozsingles
        else:
            raise ValueError("No non-localised repeaters in this survey, cannot calculate 1D likelihoods")
    else: 
        rates=grid.rates 
        if survey.nozlist is not None:
            DMobs=survey.DMEGs[survey.nozlist]
            nozlist=survey.nozlist
        else:
            raise ValueError("No non-localised FRBs in this survey, cannot calculate 1D likelihoods")
    
    dmvals=grid.dmvals
    zvals=grid.zvals

    # start by collapsing over z
    # TODO: this is slow - should collapse only used columns
    pdm=np.sum(rates,axis=0)
    
    if np.sum(pdm) == 0:
        if dolist==0:
            return -np.inf
        elif dolist==1:
            return -np.inf, None, None
        elif dolist==2:
            return -np.inf, None, None, None
        elif dolist==5: #for compatibility with 2D likelihood calculation
            return -np.inf, None, None,[0.,0.,0.,0.]
    
    if norm:
        global_norm=np.sum(pdm)
        log_global_norm=np.log10(global_norm)
        #pdm /= global_norm
    else:
        log_global_norm=0
    
    idms1,idms2,dkdms1,dkdms2 = grid.get_dm_coeffs(DMobs)
    
    ################ Calculation of p(DM) #################
    if grid.state.MW.sigmaDMG == 0.0 and grid.state.MW.sigmaHalo == 0.0:
        if np.any(DMobs < 0):
            raise ValueError("Negative DMobs with no uncertainty")

        # Linear interpolation between DMs
        pvals=pdm[idms1]*dkdms1 + pdm[idms2]*dkdms2
    else:
        dm_weights, iweights = calc_DMG_weights(DMobs, survey.DMhalos[nozlist], survey.DMGs[nozlist], dmvals, grid.state.MW.sigmaDMG, 
                                                 grid.state.MW.sigmaHalo, grid.state.MW.logu)
        pvals = np.zeros(len(idms1))
        # For each FRB
        for i in range(len(idms1)):
            pvals[i]=np.sum(pdm[iweights[i]]*dm_weights[i])
    
    # holds individual FRB data
    if dolist == 2:
        longlist=np.log10(pvals)-log_global_norm
    
    # sums over all FRBs for total likelihood
    llsum=np.sum(np.log10(pvals))-log_global_norm*DMobs.size
    lllist=[llsum]
    
    ########### Calculation of p((Tau,w)) ##############
    if ptauw:
        # checks which have OK tau values - in general, this is a subset
        # ALSO: note that this only checks p(tau,iw | w)! It does NOT
        # evaluate p(w)!!! Which is a pretty key thing...
        noztaulist = []
        inoztaulist = []
        for i,iz in enumerate(nozlist):
            if iz in survey.OKTAU:
                noztaulist.append(iz) # for direct indexing of survey
                inoztaulist.append(i) # for getting a subset of zlist
        Wobs = survey.WIDTHs[noztaulist]
        Tauobs = survey.TAUs[noztaulist]
        Iwobs = survey.IWIDTHs[noztaulist]
        ztDMobs=survey.DMEGs[noztaulist]
    
        # This could all be precalculated within the survey.
        iws1,iws2,dkws1,dkws2 = survey.get_w_coeffs(Wobs) # total width in survey width bins
        itaus1,itaus2,dktaus1,dktaus2 = survey.get_internal_coeffs(Tauobs) # scattering time tau
        iis1,iis2,dkis1,dkis2 = survey.get_internal_coeffs(Iwobs) # intrinsic width
        
        # ensures a normalised p(z) distribution for each FRB (shape: nz,nDM)
        if grid.state.MW.sigmaDMG == 0.0 and grid.state.MW.sigmaHalo == 0.0:
            # here, each FRB only has two DM weightings (linear interolation)
            ztidms1,ztidms2,ztdkdms1,ztdkdms2 = grid.get_dm_coeffs(ztDMobs)
            tomult = rates[:,ztidms1]*ztdkdms1 + rates[:,ztidms2]*ztdkdms2
            # normalise to a p(z) distribution for each FRB
            tomult = (tomult.T/np.sum(tomult,axis=0)).T
        else:
            dm_weights, iweights = calc_DMG_weights(DMobs, survey.DMhalos[noztaulist],
                                            survey.DMGs[noztaulist], dmvals, grid.state.MW.sigmaDMG, 
                                             grid.state.MW.sigmaHalo, grid.state.MW.logu)
            # here, each FRB has many DM weightings
            tomult = np.zeros([grid.zvals.size,len(iweights)])
            # construct a p(z) distribution.
            for iFRB,indices in enumerate(iweights):
                # we construct a p(z) vector for each FRB
                indices = indices[0]
                tomult[:,iFRB] = np.sum(rates[:,indices] * dm_weights[iFRB],axis=1)
            # normalise to a p(z) distribution for each FRB
            tomult /= np.sum(tomult,axis=0)
        
        # vectors below are [nz,NFRB] in length
        ptaus = survey.ptaus[:,itaus1,iws1]*dktaus1*dkws1\
            + survey.ptaus[:,itaus1,iws2]*dktaus1*dkws2 \
            + survey.ptaus[:,itaus2,iws1]*dktaus1*dkws1 \
            + survey.ptaus[:,itaus2,iws2]*dktaus1*dkws2
        
        piws = survey.pws[:,iis1,iws1]*dkis1*dkws1 \
            + survey.pws[:,iis1,iws2]*dkis1*dkws2 \
            + survey.pws[:,iis2,iws1]*dkis1*dkws1 \
            + survey.pws[:,iis2,iws2]*dkis1*dkws2
        
        
        # we now multiply by the z-dependencies
        ptaus *= tomult
        piws *= tomult
        
        # sum down the redshift axis to get sum p(tau,w|z)*p(z)
        ptaus = np.sum(ptaus,axis=0)
        piws = np.sum(piws,axis=0)
        
        llptw = np.sum(np.log10(ptaus))
        llpiw = np.sum(np.log10(piws))
        llsum += llptw
        llsum += llpiw
        lllist.append(llptw)
        lllist.append(llpiw)
        
    ############# Assesses total number of FRBs, P(N) #########
    # TODO: make the grid tell you the correct nromalisation
    if Pn and (survey.TOBS is not None):
        if grid_type==1:
            observed=survey.NORM_REPS
            C = grid.Rc
            reps=True
        elif grid_type==2:
            observed=survey.NORM_SINGLES
            C = grid.Rc
            reps=True
        else:
            observed=survey.NORM_FRB
            C = 10**grid.state.FRBdemo.lC
            reps=False
        expected=CalculateIntegral(rates,survey,reps)
        expected *= C

        Pn=Poisson_p(observed,expected)
        
        if Pn==0:
            Nll=-1e10
            if dolist==0:
                return Nll
        else:
            Nll=np.log10(Pn)
        lllist.append(Nll) 
        llsum += Nll
    else:
        lllist.append(0)
        expected=0
    
    # this is updated version, and probably should overwrite the previous calculations
    if psnr:
        # We now evaluate p(snr) at every point in b,w,and z space
        # This is p(snr) = p(Eobs) dE / \int_Eth^inf p(E) dE
        # We then sum p(snr) over the three above dimensions,
        # normalising in each case.
        
        # calculate vector of grid thresholds
        Emax=10**grid.state.energy.lEmax
        Emin=10**grid.state.energy.lEmin
        gamma=grid.state.energy.gamma
        psnr=np.zeros([DMobs.size]) # has already been cut to non-localised number
        
        # Evaluate thresholds at the exact DMobs
        DMEGmeans = survey.DMs[nozlist] - np.median(survey.DMGs + survey.DMhalos)
        idmobs1,idmobs2,dkdmobs1,dkdmobs2 = grid.get_dm_coeffs(DMEGmeans)
        
        # Linear interpolation
        Eths = grid.thresholds[:,:,idmobs1]*dkdmobs1 + grid.thresholds[:,:,idmobs2]*dkdmobs2
        
        # get the correct p(z) distributions to weight the likelihoods by
        if grid.state.MW.sigmaDMG == 0.0:
            # Linear interpolation
            rs = rates[:,idms1]*dkdms1+ rates[:,idms2]*dkdms2
        else:
            rs = np.zeros([grid.zvals.size, len(idms1)])
            # For each FRB
            for i in range(len(idms1)):
                # For each redshift
                for j in range(grid.zvals.size):
                    rs[j,i] = np.sum(grid.rates[j,iweights[i]] * dm_weights[i]) / np.sum(dm_weights[i])
                    
        # this has shape nz,nFRB - FRBs could come from any z-value
        nw,nz,nfrb = Eths.shape
        zpsnr=np.zeros([nz,nfrb])
        # numpy flattens this to the order of [z0frb0,z0f1,z0f2,...,z1f0,...]
        # zpsnr = zpsnr.flatten()
        
        if grid.eff_weights.ndim ==2:
            zwidths = True
        else:
            zwidths = False
        
        # this variable keeps the normalisation of sums over p(b,w) as a function of z
        pbw_norm = 0
        
        if doplot:
            # this will produce a plot of the p(SNR) given a particular width bin
            # for a range of beam values as a function of z for the zeroeth FRB
            plt.figure()
            ax1 = plt.gca()
            plt.xlabel("$z$")
            plt.ylabel("p(snr | b,w,z)")
            
            # this will produce a plot of p(z) for all beam values at an
            # arbitrary w-bin
            plt.figure()
            ax2 = plt.gca()
            plt.xlabel("$z$")
            plt.ylabel("p(b,w|z)")
        if ptauw:
            # hold array representing p(w)
            dpbws = np.zeros([nw,nz,nfrb])
            
        for i,b in enumerate(survey.beam_b):
            #iterate over the grid of weights
            bEths=Eths/b #this is the only bit that depends on j, but OK also!
            #now wbEths is the same 2D grid
            # bEobs has dimensions Nwidths * Nz * NFRB
            bEobs=bEths*survey.Ss[nozlist] #should correctly multiply the last dimensions
            for j,w in enumerate(grid.eff_weights):
                # p(SNR | b,w,DM,z) is given by differential/cumulative
                # however, p(b,w|DM,z) is given by cumulative*w*Omegab / \sum_w,b cumulative*w*Omegab
                # hence, the factor of cumulative cancels when calculating p(SNR,w,b), which is what we do here
                differential = grid.array_diff_lf(bEobs[j,:,:],Emin,Emax,gamma) * bEths[j,:,:]
                cumulative=grid.array_cum_lf(bEobs[j,:,:],Emin,Emax,gamma)
                
                if zwidths:
                    usew = np.repeat(w,nfrb).reshape([nz,nfrb]) # need to reshape this
                else:
                    usew = w
                
                # this keeps track of the \sum_w,b cumulative*w*Omegab
                dpbw = survey.beam_o[i]*usew*cumulative
                pbw_norm += dpbw
                zpsnr += differential*survey.beam_o[i]*usew
                
                if ptauw:
                    # record probability of this w summed over all beams for each FRB
                    dpbws[j,:,:] += dpbw
                
                if doplot and j==5:
                    #arbitrrily plots for FRB iFRB
                    iFRB=0
                    # chooses an arbitrary width to plot at for j=5
                    ax1.plot(grid.zvals,(differential/cumulative)[:,iFRB],label="b_"+str(b)[0:4]+"_w_"+str(j))
                    ax2.plot(grid.zvals,dpbw[:,iFRB],label="b_"+str(b)[0:4]+"_w_"+str(j))
        
        if doplot:
            plt.sca(ax1)
            plt.yscale('log')
            plt.tight_layout()
            plt.legend(fontsize=6)
            plt.savefig("FRB1_psnr_given_bw.png")
            plt.close()
            
            plt.sca(ax2)
            plt.tight_layout()
            plt.legend(fontsize=6)
            plt.savefig("FRB1_pbw_given_z.png")
            plt.close()
        
        # calculate p(w)
        if ptauw:
            # we would like to calculate \int p(w|z) p(z) dz
            # we begin by calculating p(w|z), below, by normalising for each z
            # normalise over all w values for each z
            dpbws /= np.sum(dpbws,axis=0)
            temp = dpbws[iws1,:,inoztaulist]
            temp *= tomult.T
            pws = np.sum(temp,axis=1)
            bad = np.where(pws == 0.)[0]
            pws[bad] = 1.e-10 # prevents nans, but 
            llpws = np.sum(np.log10(pws))
            llsum += llpws
            lllist.append(llpws)
        
        
        # normalise by the beam and FRB width values
        #This ensures that regions with zero probability don't produce nans due to 0/0
        OK = np.where(pbw_norm.flatten() > 0.)
        zpsnr = zpsnr.flatten()
        zpsnr[OK] /= pbw_norm.flatten()[OK]
        zpsnr = zpsnr.reshape([nz,nfrb])
        
        # perform the weighting over the redshift axis, i.e. to multiply by p(z|DM) and normalise \int p(z|DM) dz = 1
        rnorms = np.sum(rs,axis=0)
        zpsnr *= rs
        psnr = np.sum(zpsnr,axis=0) / rnorms
        # normalises for total probability of DM occurring in the first place.
        # We need to do this. This effectively cancels however the Emin-Emax factor.
        # sums down the z-axis
        
        # keeps individual FRB values
        if dolist==2:
            longlist += np.log10(psnr)
        
        # checks to ensure all frbs have a chance of being detected
        bad=np.array(np.where(psnr == 0.))
        if bad.size > 0:
            snrll = -1e10 # none of this is possible! [somehow...]
        else:
            snrll = np.sum(np.log10(psnr))
        
        lllist.append(snrll)
        llsum += snrll
        
        if doplot:
            
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
            
            tm4=np.max(zpsnr)
            tm2=np.max(rs)
            for j in np.arange(survey.Ss.size):
                linestyle='-'
                if j>=survey.Ss.size/4:
                    linestyle=':'
                if j>=survey.Ss.size/2:
                    linestyle='--'
                if j>=3*survey.Ss.size/4:
                    linestyle='-.'
                
                plt.figure(fig4.number)
                plt.plot(zvals,rs[:,j],label=str(int(DMobs[j])),linestyle=linestyle,linewidth=2)
                
                plt.figure(fig2.number)
                plt.plot(zvals,zpsnr[:,j],label=str(int(DMobs[j])),linestyle=linestyle)
                
            fig4.legend(ncol=2,loc='upper right',fontsize=8)
            fig4.tight_layout()
            plt.figure(fig4.number)
            plt.ylim(tm2/1e5,tm2)
            fig4.savefig('TEMP_p_z.pdf')
            plt.close(fig4.number)
            
            fig2.legend(ncol=2,loc='upper right',fontsize=8)
            fig2.tight_layout()
            plt.ylim(tm4/1e5,tm4)
            fig2.savefig('TEMP_p_zpsnr.pdf')
            plt.close(fig2.number)
            
            print("Total snr probabilities")
            for i,p in enumerate(psnr):
                print(i,survey.Ss[i],p)
        
    else:
        lllist.append(0)
    
    if grid_type==1 and pNreps:
        repll = 0
        if len(survey.replist) != 0:
            for irep in survey.replist:
                pReps = grid.calc_exact_repeater_probability(Nreps=survey.frbs["NREP"][irep],DM=survey.DMs[irep],z=None)
                if pReps == 0:
                    repll += -1e10
                else:
                    repll += np.log10(float(pReps))
        lllist.append(repll)
        llsum += repll
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
    elif dolist==5: #for compatibility with 2D likelihood calculation
        return llsum,lllist,expected,[0.,0.,0.,0.]
    

def calc_likelihoods_2D(grid,survey,doplot=False,norm=True,psnr=True,printit=False,
                Pn=False,pNreps=True,ptauw=False,dolist=0,verbose=False,grid_type=0):
    """ Calculates 2D likelihoods using observed DM,z values
    
    grid: the grid object calculated from survey
    
    survey: survey object containing the observed z,DM values
    
    doplot: will generate a plot of z,DM values
    
    psnr:
        True: calculate probability of observing each FRB at the observed SNR
        False: do not calculate this

    Pn:
        True: calculate probability of observing N FRBs
        False: do not calculate this

    pNreps:
        True: calculate probability that each repeater detects the given number of bursts
        False: do not calculate this
    
    ptauw:
        True: calculate probability of intrinsic width and scattering *given* total width
        False: do not calculate this
    
    dolist:
        0: returns total log10 likelihood llsum only [float]
        1: returns llsum, log10([Pzdm,Pn,Ps]), <Nfrbs>
        2: as above, plus a 'long list' giving log10(likelihood)
            for each FRB individually
        3: return (llsum, -np.log10(norm)*Zobs.size, 
                np.sum(np.log10(pvals)), np.sum(np.log10(psnr)))
        4: return (llsum, -np.log10(norm)*Zobs.size, 
                np.sum(np.log10(pvals)), 
                pvals.copy(), psnr.copy())
        5: returns llsum, log10([Pzdm,Pn,Ps]), <Nfrbs>, 
            np.log10([p(z|DM), p(DM), p(DM|z), p(z)])
        else: returns nothing (actually quite useful behaviour!)
    
    norm:
        True: calculates p(z,DM | FRB detected)
        False: calculates p(detecting an FRB with z,DM). Meaningless unless
            some sensible normalisation has already been applied to the grid.
    
    grid_type:
        0: normal zdm grid
        1: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for repeaters
        2: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for single bursts
    """

    ######## Calculates p(DM,z | FRB) ########
    # i.e. the probability of a given z,DM assuming
    # an FRB has been observed. The normalisation
    # below is proportional to the total rate (ish)
    
    if ptauw:
        if not survey.backproject:
            print("WARNING: cannot calculate ptauw for this survey, please initialised backproject")
    
    # Determine which array to perform operations on and initialise
    if grid_type == 1: 
        rates = grid.exact_reps 
        if survey.zreps is not None:
            DMobs=survey.DMEGs[survey.zreps]
            Zobs=survey.Zs[survey.zreps]
            zlist=survey.zreps
        else:
            raise ValueError("No localised singles in this survey, cannot calculate 1D likelihoods")
    elif grid_type == 2: 
        rates = grid.exact_singles 
        if survey.zsingles is not None:
            DMobs=survey.DMEGs[survey.zsingles]
            Zobs=survey.Zs[survey.zsingles]
            zlist=survey.zsingles
        else:
            raise ValueError("No localised repeaters in this survey, cannot calculate 1D likelihoods")
    else: 
        rates=grid.rates 
        if survey.zlist is not None:
            DMobs=survey.DMEGs[survey.zlist]
            Zobs=survey.Zs[survey.zlist]
            zlist=survey.zlist
        else:
            raise ValueError("No nlocalised FRBs in this survey, cannot calculate 1D likelihoods")
    zvals=grid.zvals
    dmvals=grid.dmvals
    
    # normalise to total probability of 1
    if norm:
        norm=np.sum(rates) # gets multiplied by event size later
    else:
        norm=1.
    
    # in the grid, each z and dm value represents the centre of a bin, with p(z,DM)
    # giving the probability of finding the FRB in the range z +- dz/2, dm +- dm/2.
    # threshold for when we shift from lower to upper is if z < zcentral,
    # weight slowly shifts from lower to upper bin
    
    idms1,idms2,dkdms1,dkdms2 = grid.get_dm_coeffs(DMobs)
    izs1,izs2,dkzs1,dkzs2 = grid.get_z_coeffs(Zobs)
    
    ############## Calculate probability p(z,DM) ################
    if grid.state.MW.sigmaDMG == 0.0 and grid.state.MW.sigmaHalo == 0.0:
        if np.any(DMobs < 0):
            raise ValueError("Negative DMobs with no uncertainty")

        # Linear interpolation
        pvals = rates[izs1,idms1]*dkdms1*dkzs1
        pvals += rates[izs2,idms1]*dkdms1*dkzs2
        pvals += rates[izs1,idms2]*dkdms2*dkzs1
        pvals += rates[izs2,idms2]*dkdms2*dkzs2
    else:
        dm_weights, iweights = calc_DMG_weights(DMobs, survey.DMhalos[zlist], survey.DMGs[zlist], dmvals, grid.state.MW.sigmaDMG, 
                                                grid.state.MW.sigmaHalo, grid.state.MW.logu)
        pvals = np.zeros(len(izs1))
        for i in range(len(izs1)):
            pvals[i] = np.sum(rates[izs1[i],iweights[i]] * dm_weights[i] * dkzs1[i] 
                              + rates[izs2[i],iweights[i]] * dm_weights[i] * dkzs2[i])
    
    bad= pvals <= 0.
    flg_bad = False
    if np.any(bad):
        # This avoids a divide by 0 but we are in a NAN regime
        pvals[bad]=1e-50 # hopefully small but not infinitely so
        flg_bad = True
    
    # holds individual FRB data
    if dolist==2:
        longlist=np.log10(pvals)-np.log10(norm)
    
    llsum=np.sum(np.log10(pvals))
    if flg_bad:
        llsum = -1e10
    # 
    llsum -= np.log10(norm)*Zobs.size # once per event
    lllist=[llsum]
    
    #### calculates zdm components p(DM),p(z|DM),p(z),p(DM|z)
    # does this by using previous results for p(z,DM) and
    # calculating p(DM) and p(z)
    if dolist==5:
        # calculates p(dm)
        pdmvals = np.sum(rates[:,idms1],axis=0)*dkdms1
        pdmvals += np.sum(rates[:,idms2],axis=0)*dkdms2
        
        # implicit calculation of p(z|DM) from p(z,DM)/p(DM)
        #neither on the RHS is normalised so this is OK!
        pzgdmvals = pvals/pdmvals
        
        #calculates p(z)
        pzvals = np.sum(rates[izs1,:],axis=1)*dkzs1
        pzvals += np.sum(rates[izs2,:],axis=1)*dkzs2
        
        # implicit calculation of p(z|DM) from p(z,DM)/p(DM)
        pdmgzvals = pvals/pzvals
        
        for array in pdmvals,pzgdmvals,pzvals,pdmgzvals:
            bad=np.array(np.where(array <= 0.))
            if bad.size > 0:
                array[bad]=1e-20 # hopefully small but not infinitely so
        
        # logspace and normalisation
        llpzgdm = np.sum(np.log10(pzgdmvals))
        llpdmgz = np.sum(np.log10(pdmgzvals))
        llpdm = np.sum(np.log10(pdmvals)) - np.log10(norm)*Zobs.size
        llpz = np.sum(np.log10(pzvals)) - np.log10(norm)*Zobs.size
        dolist5_return = [llpzgdm,llpdm,llpdmgz,llpz]
    
    
    ############### Calculate p(N) ###############3
    if Pn and (survey.TOBS is not None):
        if grid_type == 1:
            observed=survey.NORM_REPS
            C = grid.Rc
            reps=True
        elif grid_type == 2:
            observed=survey.NORM_SINGLES
            C = grid.Rc
            reps=True
        else:
            observed=survey.NORM_FRB
            C = 10**grid.state.FRBdemo.lC
            reps=False
        expected=CalculateIntegral(rates,survey,reps)
        expected *= C
        
        Pn=Poisson_p(observed,expected)
        if Pn==0:
            Pll=-1e10
            if dolist==0:
                return Pll
        else:
            Pll=np.log10(Pn)
        lllist.append(Pll)
        if verbose:
            print(f'Pll term = {Pll}')
        llsum += Pll
    else:
        expected=0
        lllist.append(0)

    # plots figures as appropriate
    if doplot:
        plt.figure()
        #plt.plot(dmvals,pdm,color='blue')
        plt.plot(DMobs,pvals,'ro')
        plt.xlabel('DM')
        plt.ylabel('p(DM)')
        plt.tight_layout()
        plt.savefig('1d_dm_fit.pdf')
        plt.close()
    
    ################ Calculates p(tau,w| total width) ###############
    if ptauw:
        # checks which have OK tau values - in general, this is a subset
        # ALSO: note that this only checks p(tau,iw | w)! It does NOT
        # evaluate p(w)!!! Which is a pretty key thing...
        ztaulist = []
        iztaulist = []
        for i,iz in enumerate(zlist):
            if iz in survey.OKTAU:
                ztaulist.append(iz) # for direct indexing of survey
                iztaulist.append(i) # for getting a subset of zlist
        Wobs = survey.WIDTHs[ztaulist]
        Tauobs = survey.TAUs[ztaulist]
        Iwobs = survey.IWIDTHs[ztaulist]
        ztDMobs=survey.DMEGs[ztaulist]
        ztZobs=survey.Zs[ztaulist]
        
        # This could all be precalculated within the survey.
        iws1,iws2,dkws1,dkws2 = survey.get_w_coeffs(Wobs) # total width in survey width bins
        itaus1,itaus2,dktaus1,dktaus2 = survey.get_internal_coeffs(Tauobs) # scattering time tau
        iis1,iis2,dkis1,dkis2 = survey.get_internal_coeffs(Iwobs) # intrinsic width
        
        #ztidms1,ztidms2,ztdkdms1,ztdkdms2 = grid.get_dm_coeffs(ztDMobs)
        ztizs1,ztizs2,ztdkzs1,ztdkzs2 = grid.get_z_coeffs(ztZobs)
        
    
        ptaus = survey.ptaus[ztizs1,itaus1,iws1]*ztdkzs1*dktaus1*dkws1 \
            + survey.ptaus[ztizs1,itaus1,iws2]*ztdkzs1*dktaus1*dkws2 \
            + survey.ptaus[ztizs1,itaus2,iws1]*ztdkzs1*dktaus1*dkws1 \
            + survey.ptaus[ztizs1,itaus2,iws2]*ztdkzs1*dktaus1*dkws2 \
            + survey.ptaus[ztizs2,itaus1,iws1]*ztdkzs2*dktaus1*dkws1 \
            + survey.ptaus[ztizs2,itaus1,iws2]*ztdkzs2*dktaus1*dkws2 \
            + survey.ptaus[ztizs2,itaus2,iws1]*ztdkzs2*dktaus2*dkws1 \
            + survey.ptaus[ztizs2,itaus2,iws2]*ztdkzs2*dktaus2*dkws2
        
        piws = survey.pws[ztizs1,iis1,iws1]*ztdkzs1*dkis1*dkws1 \
            + survey.pws[ztizs1,iis1,iws2]*ztdkzs1*dkis1*dkws2 \
            + survey.pws[ztizs1,iis2,iws1]*ztdkzs1*dkis1*dkws1 \
            + survey.pws[ztizs1,iis2,iws2]*ztdkzs1*dkis1*dkws2 \
            + survey.pws[ztizs2,iis1,iws1]*ztdkzs2*dkis1*dkws1 \
            + survey.pws[ztizs2,iis1,iws2]*ztdkzs2*dkis1*dkws2 \
            + survey.pws[ztizs2,iis2,iws1]*ztdkzs2*dkis2*dkws1 \
            + survey.pws[ztizs2,iis2,iws2]*ztdkzs2*dkis2*dkws2
        
        llptw = np.sum(np.log10(ptaus))
        llpiw = np.sum(np.log10(piws))
        llsum += llptw
        llsum += llpiw
        lllist.append(llptw)
        lllist.append(llpiw)
        
    
    ############ Calculates p(s | z,DM) #############
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
        # refined view of 'b', rather than the crude standard 
        # parameterisation

        # calculate vector of grid thresholds
        Emax=10**grid.state.energy.lEmax
        Emin=10**grid.state.energy.lEmin
        gamma=grid.state.energy.gamma

        # Evaluate thresholds at the exact DMobs
        # The thresholds have already been calculated at mean values
        # of the below quantities. Hence, we use the DM relative to
        # those means, not the actual DMEG for that FRB
        DMEGmeans = survey.DMs[zlist] - np.median(survey.DMGs + survey.DMhalos)
        idmobs1,idmobs2,dkdmobs1,dkdmobs2 = grid.get_dm_coeffs(DMEGmeans)
        
        # Linear interpolation
        Eths = grid.thresholds[:,izs1,idmobs1]*dkdmobs1*dkzs1
        Eths += grid.thresholds[:,izs2,idmobs1]*dkdmobs1*dkzs2
        Eths += grid.thresholds[:,izs1,idmobs2]*dkdmobs2*dkzs1
        Eths += grid.thresholds[:,izs2,idmobs2]*dkdmobs2*dkzs2
        
        FtoE = grid.FtoE[izs1]*dkzs1
        FtoE += grid.FtoE[izs2]*dkzs2
        
        # now do this in one go
        # We integrate p(snr|b,w) p(b,w) db dw.
        # Eths.shape[i] is the number of FRBs: length of izs1
        # this has shape nz,nFRB - FRBs could come from any z-value
        nw,nfrb = Eths.shape
        psnr=np.zeros([nfrb])
        
        if grid.eff_weights.ndim ==2:
            zwidths = True
            usews = np.zeros([nfrb])
        else:
            zwidths = False
        
        # initialised to hold w-b normalisations
        pbw_norm = 0.
        
        if ptauw:
            # hold array representing p(w)
            dpbws = np.zeros([nw,nfrb])
            
        for i,b in enumerate(survey.beam_b):
            bEths=Eths/b # array of shape NFRB, 1/b
            bEobs=bEths*survey.Ss[zlist]
            
            for j,w in enumerate(grid.eff_weights):
                temp=grid.array_diff_lf(bEobs[j,:],Emin,Emax,gamma) # * FtoE #one dim in beamshape, one dim in FRB
                differential = temp.T*bEths[j,:] #multiplies by beam factors and weight
                
                temp2=grid.array_cum_lf(bEths[j,:],Emin,Emax,gamma) # * FtoE #one dim in beamshape, one dim in FRB
                cumulative = temp2.T #*bEths[j,:] #multiplies by beam factors and weight
                
                
                if zwidths:
                    # a function of redshift
                    usew = w[izs1]*dkzs1 + w[izs2]*dkzs2
                    usews += usew
                    usew = usew
                else:
                    usew = w # just a scalar quantity
                
                # the product here is p(SNR|DM,z) = p(SNR|b,w,DM,z) * p(b,w|DM,z)
                # p(SNR|b,w,DM,z) = differential/cumulative
                # p(b,w|DM,z) = survey.beam_o[i]*usew * cumulative / sum(survey.beam_o[i]*usew * cumulative)
                # hence, the "cumulative" part cancels
                
                dpbw = survey.beam_o[i]*usew*cumulative
                
                if ptauw:
                    # record probability of this w summed over all beams for each FRB
                    dpbws[j,:] += dpbw
                
                pbw_norm += dpbw
                
                psnr += differential*survey.beam_o[i]*usew
        
        # calculate p(w)
        if ptauw:
            # normalise over all w values
            dpbws /= np.sum(dpbws,axis=0)
            # calculate pws
            pws = dpbws[iws1,iztaulist]*dkws1 + dpbws[iws2,iztaulist]*dkws2
            bad = np.where(pws == 0.)[0]
            pws[bad] = 1.e-10 # prevents nans, but 
            llpws = np.sum(np.log10(pws))
            llsum += llpws
            lllist.append(llpws)
            
        OK = np.where(pbw_norm > 0.)[0]
        psnr[OK] /= pbw_norm[OK]
        
        if doplot:
            plt.figure()
            plt.xlabel("$s ( \\equiv {\\rm SNR}/{\\rm SNR_{\\rm th}})$")
            plt.ylabel("p(s)")
            plt.scatter(survey.Ss[zlist],psnr,c=Zobs)
            cbar = plt.colorbar()
            cbar.set_label('z')
            plt.tight_layout()
            plt.savefig("2D_ps.png")
            plt.close()
            
        
        # keeps individual FRB values
        if dolist==2:
            longlist += np.log10(psnr)
        
        # checks to ensure all frbs have a chance of being detected
        bad=np.array(np.where(psnr == 0.))
        if bad.size > 0:
            snrll = -1e10 # none of this is possible! [somehow...]
        else:
            snrll = np.sum(np.log10(psnr))
        
        lllist.append(snrll)
        llsum += snrll
        if printit:
            for i,snr in enumerate(survey.Ss):
                print(i,snr,psnr[i])
    else:
        lllist.append(0)

    if grid_type==1 and pNreps:
        repll = 0
        if len(survey.replist) != 0:
            for irep in survey.replist:
                pReps = grid.calc_exact_repeater_probability(Nreps=survey.frbs["NREP"][irep],DM=survey.DMs[irep],z=survey.Zs[irep])
                repll += np.log10(float(pReps))
        lllist.append(repll)
        llsum += repll
    else:
        lllist.append(0)

    if verbose:
        print(f"rates={np.sum(rates):0.5f}," \
            f"nterm={-np.log10(norm)*Zobs.size:0.2f}," \
            f"pvterm={np.sum(np.log10(pvals)):0.2f}," \
            f"wzterm={np.sum(np.log10(psnr)):0.2f}," \
            f"comb={np.sum(np.log10(psnr*pvals)):0.2f}")
    
    
    if dolist==0:
        return llsum
    elif dolist==1:
        return llsum,lllist,expected
    elif dolist==2:
        return llsum,lllist,expected,longlist
    elif dolist==3:
        return (llsum, -np.log10(norm)*Zobs.size, 
                np.sum(np.log10(pvals)), np.sum(np.log10(psnr)))
    elif dolist==4:
        return (llsum, -np.log10(norm)*Zobs.size, 
                np.sum(np.log10(pvals)), 
                pvals.copy(), psnr.copy())
    elif dolist==5:
        return llsum,lllist,expected,dolist5_return

def calc_DMG_weights(DMEGs, DMhalos, DM_ISMs, dmvals, sigma_ISM=0.5, sigma_halo_abs=15.0, log=False):
    """
    Given an uncertainty on the DMG value, calculate the weights of DM values to integrate over

    Inputs:
        DMEGs       =   Extragalactic DMs
        DMhalo      =   Assumed constant (average) DMhalo
        DM_ISMs     =   Array of each DM_ISM value
        dmvals      =   Vector of DM values used
        sigma_ISM   =   Fractional uncertainty in DMG values
        sigma_halo  =   Uncertainty in DMhalo value (in pc/cm3)

    Returns:
        weights     =   Relative weights for each of the DM grid points
        iweights    =   Indices of the corresponding weights
    """
    weights = []
    iweights = []

    # Loop through the DMG of each FRB in the survey and determine the weights
    for i,DM_ISM in enumerate(DM_ISMs):
        # Determine lower and upper DM values used
        # From 0 to DM_total
        DM_total = DMEGs[i] + DM_ISM + DMhalos[i]

        idxs = np.where(dmvals < DM_total)

        # Get weights
        DMGvals = DM_total - dmvals[idxs] # Descending order because dmvals are ascending order
        ddm = dmvals[1] - dmvals[0]

        # Get absolute uncertainty in DM_ISM
        sigma_ISM_abs = DM_ISM * sigma_ISM

        # pISM
        if sigma_ISM_abs == 0.0:
            pISM = None
        elif log:
            pISM = st.lognorm.pdf(DMGvals, scale=DM_ISM, s=sigma_ISM) * ddm
        else:
            pISM = st.norm.pdf(DMGvals, loc=DM_ISM, scale=sigma_ISM_abs) * ddm
    
        # pHalo
        if sigma_halo_abs == 0.0:
            pDMG = None
        elif log:
            sigma_halo = sigma_halo_abs / DMhalos[i]
            pHalo = st.lognorm.pdf(DMGvals, scale=DMhalos[i], s=sigma_halo) * ddm
        else:
            pHalo = st.norm.pdf(DMGvals, loc=DMhalos[i], scale=sigma_halo_abs) * ddm
        
        if pISM is None:
            pDMG = pHalo 
        elif pHalo is None:
            pDMG = pISM
        else:
            pDMG = np.convolve(pISM, pHalo, mode='full')

        # Set upper limit of DMG = DM_total 
        # Reversed because DMGvals are descending order which corresponds to DMEGvals (dmvals) in ascending order
        pDMG = pDMG[-len(DMGvals):] 

        weights.append(pDMG)
        iweights.append(idxs)

    return weights, iweights
 
def CalculateMeaningfulConstant(pset,grid,survey,newC=False):
    """ Gets the flux constant, and quotes it above some energy minimum Emin """
    
    # Units: IF TOBS were in yr, it would be smaller, and raw const greater.
    # also converts per Mpcs into per Gpc3
    units=1e9*365.25
    if newC:
        rawconst=CalculateConstant(grid,survey) #required to convert the grid norm to Nobs
    else:
        rawconst=10**pset[7]
    const = rawconst*units # to cubic Gpc and days to year
    Eref=1e40 #erg per Hz
    Emin=10**pset[0]
    gamma=pset[3]
    factor=(Eref/Emin)**gamma
    const *= factor
    return const

def ConvertToMeaningfulConstant(state,Eref=1e39):
    """ Gets the flux constant, and quotes it above some energy minimum Emin """
    
    # Units: IF TOBS were in yr, it would be smaller, and raw const greater.
    # also converts per Mpcs into per Gpc3
    units=1e9*365.25
    
    const = (10**state.FRBdemo.lC)*units # to cubic Gpc and days to year
    #Eref=1e39 #erg per Hz
    Emin=10**state.energy.lEmin
    Emax=10**state.energy.lEmax
    gamma=state.energy.gamma
    if state.energy.luminosity_function == 0:
        factor=(Eref/Emin)**gamma - (Emax/Emin)**gamma
    else:
        from zdm import energetics
        factor = energetics.vector_cum_gamma(np.array([Eref]),Emin,Emax,gamma)
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
    
    expected=CalculateIntegral(grid.rates,survey,reps=False)
    observed=survey.NORM_FRB
    constant=observed/expected
    return constant

def CalculateIntegral(rates,survey,reps=False):
    """
    Calculates the total expected number of FRBs for that rate array and survey
    
    This does NOT include the aboslute number of FRBs (through the log-constant)
    """
    
    # check that the survey has a defined observation time
    if survey.TOBS is not None:
        if reps:
            TOBS=1 # already taken into account
        else:
            TOBS=survey.TOBS
    else:
        return 0
    
    if survey.max_dm is not None:
        idxs = np.where(survey.dmvals < survey.max_dm)
    else:
        idxs = None

    total=np.sum(rates[:,idxs])
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
    for i,r in enumerate(rsp):
        Pn=Poisson_p(os[i],r)
        if (Pn == 0):
            lp = -1e10
        else:
            lp += np.log10(Pn)
    return -lp
    

def minimise_const_only(vparams:dict,grids:list,surveys:list,
                        Verbose=False, use_prev_grid:bool=True, update=False):
    """
    Only minimises for the constant, but returns the full likelihood
    It treats the rest as constants
    the grids must be initialised at the currect values for pset already

    Args:
        vparams (dict): Parameter dict. Can be None if nothing has varied.
        grids (list): List of grids
        surveys (list): List of surveys
            A bit superfluous as these are in the grids..
        Verbose (bool, optional): [description]. Defaults to True.
        use_prev_grid (bool, optional): 
            If True, make use of the previous grid when 
            looping over them. Defaults to True.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        tuple: newC,llC,lltot
    """

    '''
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
    dC=0
    for j,s in enumerate(surveys):
        # Update - but only if there is something to update!
        if vparams is not None:
            grids[j].update(vparams, 
                        prev_grid=grids[j-1] if (
                            j > 0 and use_prev_grid) else None)
        ### Assesses total number of FRBs ###
        if s.TOBS is not None:
            # If we include repeaters, then total number of FRB progenitors = number of repeater progenitors + number of single burst progenitors
            if isinstance(grids[j], zdm_repeat_grid.repeat_Grid):
                r1= CalculateIntegral(grids[j].exact_singles, s,reps=True)
                r2= CalculateIntegral(grids[j].exact_reps, s,reps=True)
                r= r1 + r2
                r*=grids[j].Rc
            # If we do not include repeaters, then we just integrate rates
            else:
                r=CalculateIntegral(grids[j].rates, s, reps=False)
                r*=10**grids[j].state.FRBdemo.lC #vparams['lC']

            o=s.NORM_FRB
            rs.append(r)
            os.append(o)

    # Check it is not an empty survey. We allow empty surveys as a 
    # non-detection still gives information on the FRB event rate.
    if len(rs) != 0:
        data=np.array([rs,os])
        ratios=np.log10(data[1,:]/data[0,:])
        bounds=(np.min(ratios),np.max(ratios))
        startlog10C=(bounds[0]+bounds[1])/2.
        bounds=[bounds]
        t0=time.process_time()
        # If only 1 survey, the answer is trivial
        if len(surveys) == 1:
            dC = startlog10C
        else:
            result=minimize(minus_poisson_ps,startlog10C,
                        args=data,bounds=bounds)
            dC=result.x
        t1=time.process_time()
        
        # constant needs to include the starting value of .lC
        newC = grids[j].state.FRBdemo.lC + float(dC)
        # likelihood is calculated  *relative* to the starting value
        llC=-minus_poisson_ps(dC,data)
    else:
        newC = grids[j].state.FRBdemo.lC
        llC = 0.0

    if update:
        for g in grids:
            g.state.FRBdemo.lC = newC

            if isinstance(g, zdm_repeat_grid.repeat_Grid):
                g.state.rep.RC *= 10**float(dC)
                g.Rc = g.state.rep.RC

    return newC,llC

def parse_input_dict(input_dict:dict):
    """ Method to parse the input dict for generating a cube
    It is split up into its various pieces

    Args:
        input_dict (dict): [description]

    Returns:
        tuple: dicts (can be empty):  state, cube, input
        
    This is almost deprecated, but not quite!
    """
    state_dict, cube_dict = {}, {}
    # 
    if 'state' in input_dict.keys():
        state_dict = input_dict.pop('state')
    if 'cube' in input_dict.keys():
        cube_dict = input_dict.pop('cube')
    # Return 
    return state_dict, cube_dict, input_dict
