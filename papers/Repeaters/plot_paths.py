"""
This script extracts useful information from the output of
the apparent beam values seen at different declinations.

The output (decs_u.py etc) has no official time units,
However, the ra binning is uniform, and hence corresponds
to time. This should be converted into days.
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main(Nbounds=30):
    
    indir='TiedBeamSimulation/'
    # hard-coded. Whoops!
    Nra=300
    Ndec=1000
    
    
    #indir='Formed/'
    #Nra=80
    #Ndec=1000
    #Ndec2=192
    
    # load the data
    decs_u=np.load(indir+"decs_u.npy")
    decs_l=np.load(indir+"decs_l.npy")
    ras_u=np.load(indir+"ras_u.npy")
    ras_l=np.load(indir+"ras_l.npy")
    results_u=np.load(indir+"result_u.npy")
    results_l=np.load(indir+"result_l.npy")
    
    results_u = results_u.reshape([Ndec,Nra])
    Ndec2 = int(results_l.size/Nra)
    results_l = results_l.reshape([Ndec2,Nra])
    
    bounds,solids,sbounds=make_beamfiles(decs_u,ras_u,results_u,decs_l,ras_l,results_l,Nbounds=6)
    sort_chime_frbs(bounds,solids)
    exit()
    bounds,solids,sbounds=make_beamfiles(decs_u,ras_u,results_u,decs_l,ras_l,results_l,Nbounds=30)
    sort_chime_frbs(bounds,solids)
    # generates CHIME survey files based on these bounds
    
    exit()
    plot_fig1(decs_u,ras_u,results_u,decs_l,ras_l,results_l,plotdec=30)
    exit()
    #decs and ras have shape Ndec,Nra
    dec_vec_u = decs_u[:,0]
    dec_vec_l = decs_l[:,0]
    
    # gets parameters from the shapes of the arrays
    Nfreq,Ndec,Nra=results_u.shape
    Nfreq,Ndec2,Nra=results_l.shape
    
    #plot the data
    plt.figure()
    ifreq=0
    
    for i,dec in enumerate(np.arange(Ndec2)):
        #if i % 5 != 4:
        #    continue
        plt.plot(ras_l[i,:],results_l[ifreq,i,:],label=str(i))
        #if i > 20:
        
        
    #plt.xlim(-90,90)
    plt.legend()
    plt.yscale('log')
    plt.savefig('bvals.pdf')
    plt.close()
    
    # we now calculate b-values with appropriate weighting
    # this is just assuming rate ~ b^-1
    # not quite correct! Will scale differently. Argh!
    plt.figure()
    bmeans_u = np.sum(results_u,axis=2) # sums over ra
    for i in np.arange(Nfreq):
        plt.plot(dec_vec_u,bmeans_u[i,:])
    plt.xlabel('Declination')
    plt.ylabel('b bar')
    plt.savefig('bbar.pdf')
    plt.close()

def make_beamfiles(decs_u,ras_u,results_u,decs_l,ras_l,results_l,smooth=2,Nbounds=6):
    """
    Makes beam histograms for analysis
    Does this over entire declination range
    
    Inputs:
        
        
        decs_u,ras_u,results_u
        decs_l,ras_l,results_l
            These contain beam values (results) for the ras and decs
            listed. These are loaded from tied_beam_sim.py
        
        Smooth: method of smoothing the exposure, to make it look nicer.
        
        Nfiles: Number of files over which to divide CHIME exposure. Defaults to 6. 
    
    """
    
    
    ### loads beamfiles from tied beams only ####
    indir='FormedBeamSimulation/'
    Nra=80
    Ndec=1000
    
    outdir='Nbounds'+str(Nbounds)+'/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # load the data
    fresults_u=np.load(indir+"result_u.npy")
    fresults_l=np.load(indir+"result_l.npy")
    
    fresults_u = fresults_u.reshape([Ndec,Nra])
    Ndec2 = int(fresults_l.size/Nra)
    fresults_l = fresults_l.reshape([Ndec2,Nra])
    
    # already averaged over frequency
    fr_u = fresults_u
    fr_l = fresults_l
    
    print("Maximum f results are ",np.max(fr_u),np.max(fr_l))
    
    ### gets ra and dec dists ###
    prefix = "CHIME_"
    dec_vec_u = decs_u[:,0]
    dec_vec_l = decs_l[:,0]
    ddec = dec_vec_l[1]-dec_vec_l[0]
    il = np.zeros([dec_vec_l.size])
    
    Ndec,Nra=results_u.shape
    Ndec2,Nra=results_l.shape
    
    # already averaged over frequency
    r_u = results_u
    r_l = results_l
    
    # info for individual bins
    Nbins=15
    bins = np.logspace(-3,0,16)
    bbar = np.logspace(-2.9,-0.1,15)
    mean = np.zeros([15])
    
    # we make beam files by averaging the exposure in certain ranges
    
    
    # gets spline fits to CHIME exposure
    ut_spl,lt_spl=get_chime_splines()
    chime_exp = np.zeros([dec_vec_u.size])
    my_exp = np.zeros([dec_vec_u.size])
    fexp = np.zeros([dec_vec_u.size])
    hists = np.zeros([Ndec,Nbins])
    
    
    ##### checks tied beam plot
    tied_dec_max = np.max(fresults_u,axis=1)
    plt.figure()
    plt.xlabel('dec [deg]')
    plt.ylabel('Peak beam at that dec')
    plt.plot(dec_vec_u,tied_dec_max)
    plt.savefig('tied_max_check.pdf')
    plt.close()
    
    #ndays = 200
    
    # gets cosine weighting for averaging purposes
    cos_weights = np.cos(dec_vec_u * np.pi/180.)
    
    # holds histogram results
    hlist=[]
    
    # iterates over decs not in lower transit
    for i,dec in enumerate(dec_vec_u):
        
        # calculate time interval
        drau = ras_u[i,1]-ras_u[i,0]
        
        # determins if this dec is also in the lower transit
        ilower = np.where(dec_vec_l == dec)[0]
        if len(ilower) == 1:
            temp=np.concatenate((r_u[i,:],r_l[ilower[0],:]))
            ftemp=np.concatenate((fr_u[i,:],fr_l[ilower[0],:]))
            dral = ras_l[ilower,1]-ras_l[ilower,0]
            dra = (dral + drau) /2.
        else:
            temp = r_u[i,:]
            ftemp = fr_u[i,:]
            dra = drau
        
        tdays = dra /360. # converts from degrees to fraction of a day
        # this therefore represents time
        
        fh,b=np.histogram(ftemp,bins=bins)
        fh = fh*tdays # was *ndays
        
        h,b=np.histogram(temp,bins=bins)
        h = h*tdays # was *ndays
        hists[i,:]=h
        
        hlist.append(h)
        
        sdec = str(dec)[0:4]
        #np.save(outdir+prefix+sdec+'_hist.npy',h)
        #np.save(outdir+prefix+sdec+'_bins.npy',h)
        
        ####### exposure calculations #########
        
        chime_exp[i] = spl(dec,ut_spl,lt_spl)/24.
        # the below two are equivalent to within coarseness
        # of the histogram bins
        my_exp[i] = np.sum(temp**1.5)*tdays #was *ndays
        #my_exp[i] = np.sum(h*bbar**1.5)
        
        # counts number of bins that are greater than half max in tied beam
        # this therefore equates to what we expect
        gt_fwhm = np.where(ftemp > 0.5)[0]
        fexp[i] = len(gt_fwhm)*tdays # timespent above fwhm
    
    
    
    # does a global normalisation to estimate ndays
    my_total_exposure = np.sum(my_exp*cos_weights)
    chime_total_exposure = np.sum(chime_exp*cos_weights)
    ndays = chime_total_exposure/my_total_exposure
    
    
    
    ntrials=200
    start=200
    diffs=np.zeros([ntrials])
    daytrials=np.arange(start,start+ntrials)
    #smear fexp by typical uncertainty of FRB
    for i,ndays in enumerate(daytrials):
        diffs[i] = np.sum((chime_exp[10:-10] - fexp[10:-10]*ndays)**2)
    plt.figure()
    
    plt.plot(daytrials,diffs)
    plt.xlabel('Ndays')
    plt.ylabel('Exposure difference')
    plt.savefig('exposure_fit.pdf')
    plt.close()
    idays = np.argmin(diffs)
    ndays=daytrials[idays]
    print("Found ndays to be ",ndays)
    
    # NOTE: the above assumes 360 deg of ra is one day. Of course, it's
    # not quite a calendar day. ndays technically is the number of
    # sidereal days, so number of calendar days will be slightly less by ~0.0028%
    print("Total number of days now ",ndays*(24.*60/(24.*60.-4.)))
    
    fexp = np.convolve(fexp, np.full([10],0.1), mode='same')
    # fits this to CHIME exposure
    
    
    # multiplies everything by this relative amount
    my_exp *= ndays
    hists *= ndays
    fexp *= ndays
    
    ############### Now do binning!!! #############
    
    #Step 1: initialise arrrays
    bounds = np.zeros([Nbounds+1]) # declination bounds
    mean_hists = np.zeros([Nbounds,Nbins])
    nsums = np.zeros([Nbounds]) # this is the count of decs in this range
    eff_mean_hists = np.zeros([Nbounds]) # eff
    chime_means = np.zeros([Nbounds]) # eff
    
    #Step 2: fit my exposure to 7th-order polynomial to determine ranges
    coeffs=np.polyfit(dec_vec_u[:],np.log10(my_exp[:]),6)
    f=np.poly1d(coeffs)
    fitvals = 10.**f(dec_vec_u)
    
    #Step 3: calculate range of exposures and declination bounds
    exposure_bounds = np.logspace(np.log10(fitvals[0]),np.log10(fitvals[-1]),Nbounds+1)
    print("The exposure bounds are ",exposure_bounds)
    bounds[0] = dec_vec_u[0]
    bounds[-1] = 90
    
    for ib,exposure in enumerate(exposure_bounds[1:-1]):
        imin = np.where(fitvals < exposure)[0][-1]
        bounds[ib+1] = dec_vec_u[imin] # assumes we have less bins than exposures!
        print(ib,"th bound is ",bounds[ib+1])
    mean_bounds = (bounds[:-1]+bounds[1:])/2.
    
    # makes a string version of bounds
    sbounds=[]
    for i in np.arange(Nbounds+1):
        sbounds.append('{0:4.1f}'.format(bounds[i]))
    
    
    #Step 4: fill information within the bounds
    # iterates over decs not in lower transit
    for i,dec in enumerate(dec_vec_u):
        #list of bounds that dec is less than
        ibound = np.where(dec < bounds)[0]
        # we extract the last bound that dec is less than
        # i.e. if bound is 45, we find that 2+ bounds are less than 45
        # hence this gives us bin 2 -1 =1 (2nd bin)
        ibound = ibound[0]-1
        mean_hists[ibound,:] = mean_hists[ibound,:]+hlist[i]*cos_weights[i]
        nsums[ibound] += 1
    
    # renormalise mean_hists
    mean_hists *= ndays
    
    # we should NOT be renormalising here. Not relevant.
    # this counts the effective CHIME exposure for the histograms of interest
    for ibound in np.arange(Nbounds):
        # we now calculate a normalisation to the histogrammed exposure based on
        # integrating and averaging the CHIME exposure in the interval
        OK1 = np.where(dec_vec_u > bounds[ibound])
        OK2 = np.where(dec_vec_u < bounds[ibound+1])
        OK = np.intersect1d(OK1,OK2,assume_unique=True)
        
        mean_hists[ibound,:] /= np.sum(cos_weights[OK]) #nsums[ibound] 
        eff_mean_hists[ibound] = np.sum(mean_hists[ibound,:]*(bbar**1.5))
        chime_means[ibound] = np.sum(chime_exp[OK] * cos_weights[OK])/np.sum(cos_weights[OK])
    
    ################# Performs smoothing and plotting ##############
    
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    smoothed=np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=hists)
    #data_convolved = np.convolve(hists, kernel, mode='same')
    #corrects for boundary effects
    smoothed[0,:]=hists[0,:]
    smoothed[-1,:]=hists[-1,:]
    # sums over beam axis
    effs = np.sum(smoothed*bbar**1.5,axis=1)
    
    ############## replacethe following with 'Formed' data
    # estimates maximum exposure as function of declination
    FWHM=0.33 #deg
    FWHM4 = 4.*FWHM
    fraction =  FWHM4 / (360.*np.cos(dec_vec_u*np.pi/180.)) # angle on sky at this dec
    toohigh = np.where(fraction > 1.)[0]
    fraction[toohigh] = 1.
    estimated_total = ndays * fraction
    double = np.where(dec_vec_u > 70.)[0]
    estimated_total[double] = estimated_total[double]*2.
    
    plt.figure()
    plt.xlabel('$\\delta$ [deg]')
    plt.ylabel('Exposure [days]')
    plt.plot(dec_vec_u, my_exp, label = 'Simulated exposure',zorder=0)
    plt.hist(mean_bounds,bins=bounds,weights = eff_mean_hists,alpha=1.0,fill=False,
        label='Binned simulation',linestyle='-.',zorder=20,linewidth=2)
    plt.plot(dec_vec_u,chime_exp,label = 'CHIME exposure',zorder=10,linestyle='--',linewidth=3)
    
    # no point plotting this, irrelevant
    #plt.hist(mean_bounds,bins=bounds,weights = chime_means,alpha=1.0, fill=False,
    #    ec='orange',label='Binned CHIME',zorder=30,linestyle=':',linewidth=2)
    
    plt.plot(dec_vec_u, fexp, label = 'Tied beam only',zorder=80)
    #plt.plot(dec_vec_u, estimated_total*0.65, label = 'Analytic guess',zorder=100)
    
    #plt.hist(mean_bounds,bins=bounds,weights = eff_mean_hists,alpha=1.0,fill=False,color='black')
    #plt.plot(dec_vec_u, effs, label = 'smoothed exposure')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(outdir+'exposure_comparison.pdf')
    
    plt.plot(dec_vec_u, fitvals, label = 'polyfit',zorder=0,color='yellow',linewidth=3)
    plt.tight_layout()
    plt.savefig(outdir+'polyfit_exposure_comparison.pdf')
    
    plt.close()
    
    ###### plots the beamshapes at six declintions ######
    
    linestyles=["-","--","-.",":"]
    
    plt.figure()
    plt.xlim(1e-3,1)
    plt.ylim(1e-2,1e3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$B$')
    plt.ylabel('$T(B) \\, d\\log_{10}B$ [days]')
    # we have 5 bins per log10 spacing
    for ibound in np.arange(Nbounds):
        istyle = ibound %4
        label=str(bounds[ibound])[0:5]+'$^{\\circ} < \\delta < $' + str(bounds[ibound+1])[0:5]+'$^{\\circ}$'
        plt.plot(bbar,mean_hists[ibound,:]*5,label=label,linestyle=linestyles[istyle])
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(outdir+'chime_mean_hists.pdf')
    plt.close()
    
    ###### saves the beamshapes #####
    # also writes out the solid angles covered by each
    
    torad = np.pi/180.
    solids=np.zeros([Nbounds])
    for ibound in np.arange(Nbounds):
        lower = bounds[ibound]*torad
        upper = bounds[ibound+1]*torad
        solid = 2.*np.pi * (np.sin(upper)-np.sin(lower))
        solids[ibound]=solid
        print("For bound ",ibound," solid angle is ",solid)
        hfile = outdir+'chime_bound_'+str(ibound)+'_of_'+str(Nbounds)+'_hist.npy'
        bfile = outdir+'chime_bound_'+str(ibound)+'_of_'+str(Nbounds)+'_bins.npy'
        np.save(hfile,mean_hists[ibound,:])
        np.save(bfile,bbar)
    
    np.save(outdir+'bounds.npy',bounds)
    np.save(outdir+'solids.npy',solids)
    np.save(outdir+'sbounds.npy',sbounds)
    
    # returns bounds (degrees) and solid angles for the simulated exposures
    return bounds,solids,sbounds
    
def plot_fig1(decs_u,ras_u,results_u,decs_l,ras_l,results_l,plotdec=30):
    """
    Plots first figures, giving b(ra) at different frequencies
    
    Also creates beamshape files for three scenarios
    """
    
    
    global CHIMEfreqs
    
    # determine which dec matches the wanted plotdec, and extracts that data
    dec_vec_u = decs_u[0,:]
    imin = np.argmin(np.abs(dec_vec_u-plotdec))
    
    res_u = results_l[:,imin,:]
    r_u = ras_u[imin,:]
    d_u = decs_u[imin,:]
    mean_beam = np.mean(res_u,axis=0)
    
    dr = r_u[1]-r_u[0] # dr is in units of degrees of ra
    
    # dt is now in units of days per year, i.e. it is a unit of time
    dt = dr*365.25/360.# *24*60. # dt is now in units of minutes. Should be in days.
    
    
    plt.figure()
    plt.xlabel('LST [deg]')
    plt.ylabel('$B$(LST)')
    styles=['-','--','-',':']
    for i,f in enumerate(CHIMEfreqs):
        plt.plot(r_u,res_u[i,:],label=str(f)+' MHz',linestyle = styles[i])
        
    plt.plot(r_u,mean_beam,label='$\\overline{B}$',linestyle='-',linewidth=2,color='gray',alpha=0.5)
    
    plt.ylim(0,0.7)
    plt.xlim(-6,6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('PaperFigs/ra_of_b.pdf')
    plt.ylim(1e-3,1)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('PaperFigs/ra_of_b_log.pdf')
    plt.close()
    
    # now we make histograms of that data
    
    plt.figure()
    #plt.xlabel('$B^{\\prime} = \\frac{B}{B_{\\rm max}}$')
    #plt.ylabel('ra($B^\\prime$) [min]')# $\\frac{\\nu}{\\rm GHz}$ [min]')
    plt.xlabel('B')
    plt.ylabel('t(B) [min]')
    styles=['-','--','-',':']
    bins = np.logspace(-3,0,16)
    bbar = np.logspace(-2.9,-0.1,15)
    mean = np.zeros([15])
    
    bins2 = np.logspace(-3,0,151)
    bbar2 = np.logspace(-2.99,-0.01,150)
    mean2 = np.zeros([150])
    
    norm=False
    for i,f in enumerate(CHIMEfreqs):
        
        if norm:
            temp = res_u[i,:]/np.max(res_u[i,:])
        else:
            temp = res_u[i,:]
        h,b=np.histogram(temp,bins = bins)
        #h = h*f/1000.
        h = h*dt
        OK = np.where(h > 0.)
        plt.plot(bbar[OK],h[OK],label=str(f)+' MHz',linestyle = styles[i])
        
        #h,b=np.histogram(res_u[i,:],bins = bins)
        mean += h
    # averages over four frequencies
    #mean /= 4.
    #plt.plot(bbar,mean,label='$\\overline{\\rm ra}(B)$',linestyle = '-',color='black')
        
    #averages beamshape first, then calcs histogram
    beam_mean = np.mean(res_u[:,:],axis=0)
    if norm:
        temp = beam_mean/np.max(beam_mean)
    else:
        temp = beam_mean
    hbar,b=np.histogram(temp,bins = bins)
    hbar2,b2=np.histogram(temp,bins = bins2)
    hbar = hbar*dt # hbar is calculated from beam_mean, i.e. over all freqs
    hbar2 = hbar2*dt
    OK = np.where(h > 0.)
    plt.plot(bbar[OK],hbar[OK],label='t($\\overline{B}$)',linestyle='-',linewidth=2,color='gray',alpha=0.5)
    
    plt.legend(loc='upper right')
    plt.xlim(1e-2,1)
    #plt.ylim
    #plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('PaperFigs/b_of_ra_hist.pdf')
    plt.close()
    
    ################### now we do beam calculations ##########
    
    ######## get beam data for simplest approximation #######
    # We calculate a mean value of B using
    # Bbar = \int B^gamma dt
    gamma=-0.95
    gamma=-1.5
    #B0=0.5
    all_average = np.sum(res_u[:,:]**-gamma)#/B0**-gamma # Write this down as beam at B0
    B0 = np.sum(res_u[:,:]**(-gamma+1))/all_average
    print("Calculated B0 to be ",B0)
    all_average /= B0**-gamma
    all_average *= dt/4.
    tempdata = np.array([B0,all_average])
    np.savetxt('Beams/single_point.dat',tempdata)
    np.save('single_point_bins.npy',np.array(B0))
    np.save('single_point_hist.npy',np.array(all_average))
    
    ###### histogram of frequency-averaged beam: medium ######
    tempdata=np.zeros([bbar[OK].size,2])
    tempdata[:,0]=bbar[OK]
    tempdata[:,1]=hbar[OK]
    np.savetxt('Beams/mean_f_hist_beam.dat',tempdata)
    
    np.save('mean_f_bins.npy',bbar[OK])
    np.save('mean_f_hist.npy',hbar[OK])
    
    np.save('mean_f2_bins.npy',bbar2)
    np.save('mean_f2_hist.npy',hbar2)
    
    
    print("Medium ",bbar[OK]," with days ",hbar[OK]," total solid is ",np.sum(hbar))
    print("Simple beam has effective value ",B0," with days ",all_average)
    
    ##### histogram for most detailed beam #####
    
    Nf,Nr=res_u.shape
    all_freq_beam = np.zeros([Nf*Nr])
    for i,f in enumerate(CHIMEfreqs):
        all_freq_beam[i*Nr:(i+1)*Nr]=res_u[i,:]
    all_freq_beam = np.sort(all_freq_beam)
    tempdata = np.zeros([Nf*Nr,2])
    tempdata[:,0] = all_freq_beam
    tempdata[:,1] = dt
    np.savetxt('Beams/all_freq_beam.dat',tempdata)
    
    # makes a histogram with 100 points
    
    bins=np.logspace(-3,0,101)
    h,bins=np.histogram(tempdata[:,0],bins=bins)
    hb = np.linspace(-3,0,101)
    hb = hb[:-1]
    hb = hb + (hb[1] - hb[0])/2.
    hb = 10.**hb
    
    h = h*dt/4. # frequency average
    
    np.save('all_freq_beam_bins.npy',hb)
    np.save('all_freq_beam_hist.npy',h)
    print("### full beam ###")
    print("Sum is ",np.sum(h))
    #for i,b in enumerate(hb):
    #    print(b,h[i])
    
    #plt.figure()
    #plt.scatter([B0],[all_average],marker='+')
    #cs1 = np.cumsum
    
    # does same histogram again with 10 points
    bins=np.logspace(-3,0,21)
    h,bins=np.histogram(tempdata[:,0],bins=bins)
    hb = np.linspace(-3,0,21)
    hb = hb[:-1]
    hb = hb + (hb[1] - hb[0])/2.
    hb = 10.**hb
    
    h = h*dt/4. # frequency average
    
    np.save('all_freq_beam_10_bins.npy',hb)
    np.save('all_freq_beam_10_hist.npy',h)
    print("### full beam ###")
    print("Sum is ",np.sum(h))


def get_chime_splines(plot=False):
    """
    Manually extract data points from CHIME Catalog 1 - Figure 5
    
    # what motivated this comment?
    NOTE: There is a (very) slight and brief drop in CHIME exposure time
    near 20 degrees, but fitting a spline requires the exposure
    time does not decrease with declination. We keep the exposure time as
    ~ 15 hours around this period. This overestimates'
    P_S, but the effect will be very minimal.
    """
    
    global ut_spl, lt_spl
    
    # hard-coded data extracted from CHIME catalogue
    ut_decs = [-6, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 67.5, 70, 72.5,
        75, 77.5, 80, 82.5, 85, 87, 89] # upper transit declinations (degrees)
    ut_exp = [12.5, 13, 13, 13, 14, 15, 15, 15, 15, 16, 17, 19, 20, 23, 27, 30, 31, 40,
        46, 50, 63, 80, 140, 200, 1000] # upper transit exposure times (hours)
    lt_decs = [72.5, 75, 77.5, 80, 82.5, 85, 87, 89] # lower transit declinations (degrees)
    lt_exp = [30, 38, 50, 70, 90, 120, 200, 900] # lower transit exposure times (hours)

    #NOTE: Lower transit data is only relevant for dec > 70 degrees

    #Fit two separate splines to the data points
    ut_spl = UnivariateSpline(ut_decs, ut_exp) # upper tranist
    lt_spl = UnivariateSpline(lt_decs, lt_exp) # lower transit
    ut_spl.set_smoothing_factor(0.5)
    lt_spl.set_smoothing_factor(0.5)
    if plot:
        ############################
        #    PLOT SPLINE FITS     #
        ############################
        
        fig, ax = plt.subplots(figsize=(15,7))
        plt.scatter(ut_decs, np.log10(ut_exp), c = 'darkblue', s = 50, label = 'Upper Transit: CHIME (2021)')
        plt.scatter(lt_decs, np.log10(lt_exp), s = 50, c = 'red', label = 'Lower Transit: CHIME (2021)', marker = 's')
        plt.plot(ut_decs, np.log10(ut_spl(ut_decs)), c= 'skyblue', lw=4, linestyle = '--', label = 'Upper Transit: Fit')
        plt.plot(lt_decs, np.log10(lt_spl(lt_decs)), c='darkorange', lw=4, linestyle = '--', label = 'Lower Transit: Fit')
        plt.axvline(21.7, c = 'black', label = 'FRB 20190425A') # declination of FRB 20190425A
        plt.xlabel('DEC (degrees)', fontsize=18)
        plt.ylabel('$log_{10}(\mathrm{Exposure\ Time})$ (hours)', fontsize=18)
        #plt.title('CHIME exposure vs dec', fontsize=25)
        plt.legend(fontsize=18)
        plt.savefig('Spline.pdf')
        plt.close()
    
    # sets the normalisation constant C
    #normalise_splines()
    
    # checks everything is in order
    #global dec_min,dec_max
    # Checking to see if normalisation works appropriately:
    #solution = integrate.quad(normalised_E, dec_min, dec_max)
    #print("Normalised E(delta) SHOULD be unity = 1. It is:", solution[0])
    return ut_spl,lt_spl
    
def spl(dec,ut_spl,lt_spl):
    """
    Determines if lower, upper, or both transits should be included
    """
    if dec < 70:
        return ut_spl(dec)
    else:
        return ut_spl(dec) + lt_spl(dec)  

def sort_chime_frbs(bounds,solids):
    
    import utilities as ute
    
    
    
    # defines set of bounds to read in
    Nbounds=len(bounds)-1
    bdir = 'Nbounds'+str(Nbounds)+'/'
    
    
    ####### loads CHIME FRBs ######
    DMhalo=50
    names,decs,dms,dmegs,snrs,reps,ireps,widths,nreps = ute.get_chime_data(DMhalo=DMhalo)
    dmgs = dms - dmegs - DMhalo
    
    
    # now breaks this up into declination bins
    #The below is hard-coded and copied from "plot
    #bounds=np.array([-11.,5,20,65,80,85,90])
    lowers=bounds[:-1]
    uppers=bounds[1:]
    for i,lb in enumerate(lowers):
        OK1=np.where(decs > lb)[0]
        OK2=np.where(decs < uppers[i])[0]
        OK=np.intersect1d(OK1,OK2)
        OK3 = np.where(reps==0)
        nOK = np.intersect1d(OK,OK3)
        rOK = np.intersect1d(OK,ireps)
        opdir='TEMP/'
        if not os.path.exists(opdir):
            os.mkdir(opdir)
        opfile = opdir+'CHIME_decbin_'+str(i)+"_of_"+str(Nbounds)+".dat"
        f = open(opfile,"w")
        #print("Found ",len(rOK),len(nOK)," FRBs which do (not) repeat in dec range ",lb,uppers[i])
        
        f.write("BW 400 #MHz\n")
        f.write("NFRB "+str(len(nOK))+" #Number of FRBs\n")
        f.write("BEAM chime_bound_"+str(i)+"_of_"+str(Nbounds)+" #prefix of beam file\n")
        f.write("TOBS "+str(solids[i]) +" # Actually the solid angle of observation!\n")
        f.write("THRESH 5 #Jy ms to a 1 ms burst, very basic. 95% comleteness. Likely lower\n")
        f.write("SNRTHRESH 10 # signal-to-noise threshold: scales jy ms to snr.\n")
        f.write("TRES 0.983 #ms\n")
        f.write("FRES 0.0244 #Mhz\n")
        f.write("FBAR 600 # mean frequency (MHz)\n")
        
        
        f.write("KEY  Xname         DM     DMG     SNR     WIDTH  NREP\n")
        for j in nOK:
            string='FRB  {0:} {1:6.1f}  {2:5.1f}  {3:5.1f} {4:8.3f}  1 \n'.\
                format(names[j],dms[j],dmgs[j],snrs[j],widths[j])
            f.write(string)
        #### searches for repeaters ####
        
        for j in rOK:
            rindex = np.where(ireps == j)[0][0]
            string='FRB  {0:} {1:6.1f}  {2:5.1f}  {3:5.1f} {4:8.3f}  {5:} \n'.\
                format(names[j],dms[j],dmgs[j],snrs[j],widths[j],nreps[rindex])
            f.write(string)
        f.close()

# do this twice, with two different values of Nbounds
main(Nbounds=6)
main(Nbounds=30)
