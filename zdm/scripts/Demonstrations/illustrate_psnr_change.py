""" 
This script shows the difference in p(SNR) calculations from the old method to the new

Essentially, the old calculated  (\int dz \int db \int dw p(Eobs)dE) / (\int dz \int db \int dw \int_Eth^inf p(E) dE)
The new calculates \int dz \int db \int dw [  p(Eobs)dE)/\int_Eth^inf p(E) dE ]
i.e., p(snr|w,b,z) = [  p(Eobs)dE)/\int_Eth^inf p(E) dE ]
so the new method is \int p(snr) dw db dz 
"""
import os
from pkg_resources import resource_filename
from zdm import misc_functions
from zdm import parameters
from zdm import loading as loading
from zdm import iteration as it
import numpy as np
import math
from matplotlib import pyplot as plt

def main():
    
    
    ###### begins with a test of normal CRAFT stuff ######
    name = 'CRAFT_ICS_1300'
    
    ss,gs = loading.surveys_and_grids(survey_names=[name],repeaters=False)
    s=ss[0]
    g=gs[0]
    
    result = calc_likelihoods_1D(g, s, norm=True, psnr=True, dolist=1, repeaters=False, Pn=True)
    

def calc_likelihoods_1D(grid,survey,doplot=False,norm=True,psnr=True,Pn=True,dolist=0,repeaters=False,singles=False):
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

    dolist
        2: llsum,lllist [Pzdm,Pn,Ps],expected,longlist
            longlist holds the LL for each FRB
        5: llsum,lllist,expected,[0.,0.,0.,0.]

    repeaters:
        True: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for repeaters
        False: assumes no repeaters considered (or singles = True)
    singles:
        True: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for single bursts
        False: assumes no repeaters considered (or repeaters = True)
    NOTE: repeaters and singles should probably be combined into a single variable...
    """
    
    # Determine which array to perform operations on and initialise
    if repeaters and singles: 
        raise ValueError("Specify the likelihood for repeaters or singles, not both") 
    elif repeaters: 
        rates = grid.exact_reps 
        if survey.nozreps is not None:
            DMobs=survey.DMEGs[survey.nozreps]
            nozlist=survey.nozreps
        else:
            raise ValueError("No non-localised singles in this survey, cannot calculate 1D likelihoods")
    elif singles: 
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
    
    #ddm=dmvals[1]-dmvals[0]
    #kdms=DMobs/ddm
    #idms1=kdms.astype('int')
    #idms2=idms1+1
    #dkdms=kdms-idms1
    
    idms1,idms2,dkdms1,dkdms2 = grid.get_dm_coeffs(DMobs)

    if grid.state.MW.sigmaDMG == 0.0 and grid.state.MW.sigmaHalo == 0.0:
        if np.any(DMobs < 0):
            raise ValueError("Negative DMobs with no uncertainty")

        # Linear interpolation
        pvals=pdm[idms1]*dkdms1 + pdm[idms2]*dkdms2
    else:
        dm_weights, iweights = it.calc_DMG_weights(DMobs, survey.DMhalos[nozlist], survey.DMGs[nozlist], dmvals, grid.state.MW.sigmaDMG, 
                                                 grid.state.MW.sigmaHalo, grid.state.MW.logu)
        pvals = np.zeros(len(idms1))
        # For each FRB
        for i in range(len(idms1)):
            pvals[i]=np.sum(pdm[iweights[i]]*dm_weights[i])

    if norm:
        global_norm=np.sum(pdm)
        log_global_norm=np.log10(global_norm)
        #pdm /= global_norm
    else:
        log_global_norm=0
    
    # holds individual FRB data
    if dolist == 2:
        longlist=np.log10(pvals)-log_global_norm
    
    # sums over all FRBs for total likelihood
    llsum=np.sum(np.log10(pvals))-log_global_norm*DMobs.size
    lllist=[llsum]
    
    ### Assesses total number of FRBs ###
    if Pn and (survey.TOBS is not None):
        if repeaters:
            observed=survey.NORM_REPS
            C = grid.Rc
            reps=True
        elif singles:
            observed=survey.NORM_SINGLES
            C = grid.Rc
            reps=True
        else:
            observed=survey.NORM_FRB
            C = 10**grid.state.FRBdemo.lC
            reps=False
        expected=it.CalculateIntegral(rates,survey,reps)
        expected *= C

        Pn=it.Poisson_p(observed,expected)
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
        # NOTE: to break this into a p(SNR|b) p(b) term, we first take
        # the relative likelihood of the threshold b value compared
        # to the entire lot, and then we calculate the local
        # psnr for that beam only. But this requires a much more
        # refined view of 'b', rather than the crude standatd 
        # parameterisation
        
        # calculate vector of grid thresholds
        Emax=10**grid.state.energy.lEmax
        Emin=10**grid.state.energy.lEmin
        gamma=grid.state.energy.gamma
        psnr=np.zeros([DMobs.size]) # has already been cut to non-localised number
        
        # Evaluate thresholds at the exact DMobs
        #kdmobs=(survey.DMs - np.median(survey.DMGs + survey.DMhalos))/ddm
        #kdmobs=kdmobs[nozlist]
        #kdmobs[kdmobs<0] = 0
        #idmobs1=kdmobs.astype('int')
        #idmobs2=idmobs1+1
        #dkdmobs=kdmobs-idmobs1 # applies to idms2
        DMEGmeans = survey.DMs[nozlist] - np.median(survey.DMGs + survey.DMhalos)
        idmobs1,idmobs2,dkdmobs1,dkdmobs2 = grid.get_dm_coeffs(DMEGmeans)
        
        # Linear interpolation
        Eths = grid.thresholds[:,:,idmobs1]*dkdmobs1 + grid.thresholds[:,:,idmobs2]*dkdmobs2

        ##### IGNORE THIS, PVALS NOW CONTAINS CORRECT NORMALISATION ######
        # we have previously calculated p(DM), normalised by the global sum over all DM (i.e. given 1 FRB detection)
        # what we need to do now is calculate this normalised by p(DM),
        # i.e. psnr is the probability of snr given DM, and hence the total is
        # p(snr,DM)/p(DM) * p(DM)/b(burst)
        # get a vector of rates as a function of z
        #rs = rates[:,idms1[j]]*(1.-dkdms[j])+ rates[:,idms2[j]]*dkdms[j]
        
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
                    
        #norms=np.sum(rs,axis=0)/global_norm
        norms=pvals
        
        # this has shape nz,nFRB - FRBs could come from any z-value
        zpsnr=np.zeros(Eths.shape[1:])
        zpsnr2=np.zeros(Eths.shape[1:])
        zpsnr3=np.zeros(Eths.shape[1:])
        zpsnr2 = zpsnr2.flatten()
        beam_norm=np.sum(survey.beam_o)
        #in theory, we might want to normalise by the sum of the omeba_b weights, although it does not matter here
        
        nplots = norms.size
        sqrt = nplots**0.5
        nrow = math.floor(sqrt)
        ncol = math.ceil(sqrt)
        if nrow*ncol < nplots:
            ncol += 1
        fig,axs = plt.subplots(nrow,ncol,sharex=True,sharey=False)
        axs = axs.reshape(nrow*ncol)
        
        for i,b in enumerate(survey.beam_b):
            #iterate over the grid of weights
            bEths=Eths/b #this is the only bit that depends on j, but OK also!
            #now wbEths is the same 2D grid
            #wbEths=bEths #this is the only bit that depends on j, but OK also!
            bEobs=bEths*survey.Ss[nozlist] #should correctly multiply the last dimensions
            for j,w in enumerate(grid.eff_weights):
                temp=grid.array_diff_lf(bEobs[j,:,:],Emin,Emax,gamma)
                zpsnr += temp * bEths[j,:,:] *survey.beam_o[i]*w
                
                differential = temp * bEths[j,:,:]
                temp2=grid.array_cum_lf(bEobs[j,:,:],Emin,Emax,gamma)
                
                zpsnr3 += temp2 *survey.beam_o[i]*w
                
                cumulative = temp2
                
                OK = np.where(cumulative.flatten()>0)[0]
                zpsnr2[OK] += (differential.flatten()[OK]/cumulative.flatten()[OK])*survey.beam_o[i]*w
        
        NORM = np.sum(survey.beam_o) * np.sum(grid.eff_weights)
        zpsnr2 /= NORM
        zpsnr2=zpsnr2.reshape(Eths.shape[1:])
        rnorms = np.sum(rs,axis=0)
        zpsnr2 *= rs
        
        for iplot in np.arange(nplots):
            axs[iplot].plot(grid.zvals,zpsnr2[:,iplot]/rnorms[iplot],label="exact")
        zpsnr2 = np.sum(zpsnr2,axis=0) / rnorms
        
        
        zpsnr3 = zpsnr/zpsnr3
        zpsnr3 *= rs
        for iplot in np.arange(nplots):
            axs[iplot].plot(grid.zvals,zpsnr3[:,iplot]/rnorms[iplot],label="average b,w")
        zpsnr3 = np.sum(zpsnr3,axis=0) / rnorms
        
        # we have now effectively calculated the local probabilities in the source-counts histogram for a given DM
        # we have to weight this by the sfr_smear factors, and the volumetric probabilities - i.e., everything
        # but the integral of the luminosity function, since the above has just taken the differential
        # these are the grid smearing factors incorporating pcosmic and the host contributions
        if grid.state.MW.sigmaDMG == 0.0:
            # Linear interpolation
            sg = grid.sfr_smear[:,idms1]*dkdms1+ grid.sfr_smear[:,idms2]*dkdms2
        else:
            sg = np.zeros([grid.sfr_smear.shape[0], len(idms1)])
            # For each FRB
            for i in range(len(idms1)):
                # For each redshift
                for j in range(sg.shape[0]):
                    sg[j,i] = np.sum(grid.sfr_smear[j,iweights[i]] * dm_weights[i]) / np.sum(dm_weights[i])
        sgV = (sg.T*grid.dV.T).T
        wzpsnr = zpsnr * sgV
        #THIS HAS NOT YET BEEN NORMALISED!!!!!!!!
        # at this point, wzpsnr should look exactly like the grid.rates, albeit
        # A: differential, and 
        # B: slightly modified according to observed and not threshold fluence
        
        # normalises for total probability of DM occurring in the first place.
        # We need to do this. This effectively cancels however the Emin-Emax factor.
        # sums down the z-axis
        for iplot in np.arange(nplots):
            axs[iplot].plot(grid.zvals,wzpsnr[:,iplot]/norms[iplot],label="orig (avg b,w,z)")
            axs[iplot].set_xlim(0,2)
        axs[4].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("prob_comparison.png")
        plt.close()
        psnr=np.sum(wzpsnr,axis=0)
        psnr /= norms #normalises according to the per-DM probability
        
        print("Original calculation: ",psnr)
        print("No z averaging: ",zpsnr3)
        print("New, correct method: ",zpsnr2)
        
        

main()
