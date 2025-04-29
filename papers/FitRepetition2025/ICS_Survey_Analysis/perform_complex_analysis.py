"""
Program to analyse observation time data

Reads in CRAFT FRB info from ...?

Load observation data from ICS.txt.
    The is a cat of all observations generated from 
    the script 
    
Time is "converted" (convert_times) from Jan 1st 2018 is t=0
"""


import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import matplotlib
""" Files to analyse total observation time information """
from matplotlib import cm

defaultsize=14
ds=4
font = {'family' : 'helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

#### some hard-coded info
# relative detection rate per hour for FE, ICS1Ghz, ICS 900MHz, and Pks is
# 2 20 43 16
#observed is
# FE: 20 in 1274.6 days (1.57 FRBs/100days)
# ICS: 
# ICS 900: 
# Pks: 12 in 164.4 days (7.3 FRBs/100 days) 

# debugged FREDDA: 
# full implementation April XXXX
# initial bug introduced YYYY


# From Ryan, valid Wednesday January 9th
#AS039 (Ryan driving ASKAP):  2453.92 hours (owner=1)
#AS106 (Filler CRAFT): 4121.06 hours (owner=2)

#Total time when Ryan driving is  314.40986592
#Total filler CRAFT time  3255.742076566204

global tmin
tmin=200.

def main(alpha=-1.5):
    global tmin
    ###########
    #IMPORTANT PARAMETERS
    # tmin: minimum time of the survey
    # entered in days relative to 10th 2019, which is t=0
    # to select only data in FREDDAv3, use tmin=865
    
    ############# sets for modelling mode ##########
    tmin=convert_times(20200406000000) # v3 ran from April 6th 2020 onwards
    remove_start=True
    
    #maxDM: removes FRBs with DM too high to view from 850 MHz
    maxDM=True
    
    ######## loads data from Vanessa's ASKAP log ##########
    # it does *not* include coordinates however...
    vdate,vtime,vowner,vfmid,verr,vsbid,vfootprint,vpitch,vsbids=load_vanessa()
    vtsamp = np.full([vdate.size],1.728) # ASSUMPTIONS!!!
    lowt = np.where(vdate < 20190400000000)[0] # this time is a guess between FRBs 3 and 4
    vtsamp[lowt]=0.864 # first three FRBs
    vdays=convert_times(vdate)
    vtime /= 3600. #seconds to hours
    # do we weight by an assumed integration time? Could be simply 1/sqrt{tint}?
    print("Loading total of ",np.sum(vtime)," hours from ASKAP logs")
    
    # veff is relative to 1.3 GHz closepack36 at 0.9 degree pitch
    veff=make_fp_vector(vfmid,vfootprint,vpitch)
    
    #cannot renormalise veff according to Nantennas - don't know this info!!!
    
    ######## loads IS data ###########
    data=np.loadtxt("ICS_complex.txt")
    
    scans=data[:,0].astype(int)
    days=convert_times(scans)
    Nants=-data[:,1]
    time=data[:,2]
    freq=data[:,3]
    
    nchan=data[:,4]
    dchan=data[:,5]
    tint=data[:,6]
    ra=data[:,7]
    dec=data[:,8]
    owner=data[:,9]
    footprint=data[:,10] # 1 is code for closepack36, 2 is code for square_6x6
    pitch=data[:,11]
    sbids=data[:,12]
    tsamp=data[:,13]*1000. # convert from s to ms
    
    fmid=freq+nchan*dchan/2.
    print("Total number of observations: ",data.shape[0], " over time of ",np.sum(time)," hr")
    
    # makes a plot of footprint factors
    if False:
        fp_plot(footprint,pitch,fmid,time)
    
    # calculates efficiency from beamsize, system temperature, and Nantennas
    # eff is relative to 1.3 GHz closepack36 at 0.9 degree pitch
    eff=make_fp_vector(fmid,footprint,pitch)
    eff *= Nants**0.5/5. # 5 is just a normalisation constant, relative to 25 antennas
    
    #### loads FRB data ####
    frbscans,frbdays,frbfmids,frb_snrs,frb_bs,frbra,frbdec=load_frb_data(maxDM=maxDM,mint=tmin)
    keep = check_frbs(frbscans,scans)
    if False:
        OK=np.where(frb_snrs >= 14.)[0]
        frb_snrs = frb_snrs[OK]
        frbfmids = frbfmids[OK]
        frbscans = frbscans[OK]
        order = np.argsort(frbfmids)
        frbfmids = frbfmids[order]
        frbscans = frbscans[order]
        for i,fmid in enumerate(frbfmids):
            print(i,fmid,frbscans[i])
        exit()
    
    if remove_start:
        OK=np.where(days>tmin)[0]
        scans=scans[OK]
        days=days[OK]
        Nants=Nants[OK]
        time=time[OK]
        freq=freq[OK]
        nchan=nchan[OK]
        dchan=dchan[OK]
        tint=tint[OK]
        ra=ra[OK]
        dec=dec[OK]
        owner=owner[OK]
        fmid=fmid[OK]
        eff=eff[OK]
        tsamp=tsamp[OK]
        
        vOK=np.where(vdays>tmin)[0]
        #vdate,vtime,vowner,vfmid,verr,vsbid
        vdays=vdays[vOK]
        #vtint=vtint[vOK]
        vtime = vtime[vOK]
        vdate = vdate[vOK]
        vowner=vowner[vOK]
        vfmid=vfmid[vOK]
        verr=verr[vOK]
        vsbid=vsbid[vOK]
        vtsamp=vtsamp[vOK]
    #else:
        
        #tmin=200 # days from... when?
        #frbscans,frbdays,frbfmids,frb_snrs,frb_bs,frb_ra,frb_dec=load_frb_data(mint=tmin)
    
    
    ######### We now do plots relying on data for which we have full logging information ######
    
    if False:
        # done!
        make_b_plot(ra,dec,time,frb_bs[keep],fmid)
    
    if False:
        # old and outdated, takes a LONG time
        make_pointing_plots(ra,dec,time,frb_bs,fmid) # old version, before healpix
    
    if False:
        # done! Except healpix sucks
        make_healpix_plots(ra,dec,time,frbra,frbdec,fmid)
    
    # makes fake vNants vector
    mean_ants = np.sum(Nants)/Nants.size
    vNants = np.full([vtime.size],mean_ants)
    
    # combines Vanessa's data and this one
    # Here, "c:" means "combined"
    # this includes footprint and pitch info, number of antennas, etc
    # ceff here is 
    cdays,ctime,cfmid,cowner,ceff,ctsamp,cfootprint,cpitch,cNants=make_combined(
                                owner,time,vowner,vtime,days,
                                vdays,fmid,vfmid,eff,veff,sbids,vsbids,tsamp,vtsamp,
                                footprint,vfootprint,pitch,vpitch,Nants,vNants)
    
    
    OK = np.where(cdays > tmin)[0]
    
    LOW=np.where(cfmid < 1000.)[0]
    HIGH=np.where(cfmid > 1400)[0]
    MID=np.where(np.abs(cfmid-1200) < 200)[0]
    labels=["low","mid","high"]
    if True:
        # one per frequency bin
        for i,SEL in enumerate([LOW,MID,HIGH]):
            label = labels[i]
            make_histograms(ctime[SEL],cfmid[SEL],cfootprint[SEL],cpitch[SEL],cNants[SEL],label=label)
        
        # average histogram for all observations
        make_histograms(ctime[OK],cfmid[OK],cfootprint[OK],cpitch[OK],cNants[OK],label="average")
    
    frblists=make_frb_y(frbdays,frbfmids)
    #print("Total Vanessa time was ", np.sum(ctime)-np.sum(time),np.sum(time))
    
    print("Total observation time is ",np.sum(ctime))
    
    # accounts for the DM smearing, scattering, inrinsic width
    weffs = get_width_efficiencies(ctsamp,cfmid,maxDM=maxDM)
    print("Mean value of weffs is ",np.sum(weffs)/weffs.size)
    ceff *= weffs
    
    ceff *= (cfmid/1272)**alpha
    
    # calculates mean properties of the survey
    fbar = np.sum(cfmid[OK] * ctime[OK] * ceff[OK]) / np.sum(ceff[OK]*ctime[OK])
    
    print("Mean survey frequency is ",fbar)
    print("Total time is ",np.sum(ctime[OK]))
    print("Total weighted time is ",np.sum(ctime[OK] * ceff[OK]))
    
    
    ##### generates a single effective sensitivity curve #####
    # weights by time and efficiency here
    if True:
        mean_curve = get_dm_efficiencies(ctsamp,cfmid[OK],ceff[OK]*ctime[OK],outfile="mean_ICS_efficiency.npy")
    
    exit()
    
    ############ cumulative rate plots vs efficiencies ##########
    
    # we now compare effective efficiencies to the Fly's Eye rate
    flyes_eye_eff = make_fp_vector(np.array([1297.5]),np.array([1]),np.array([0.9]),sfile="sens_data.dat")
    flyes_eye_w = get_width_efficiencies(np.array([1.26]),np.array([1297.5]),maxDM=maxDM)
    fe_eff = flyes_eye_eff * flyes_eye_w 
    
    # normalises efficiencies by the average
    mean_ceff = np.sum(ceff*ctime)/np.sum(ctime)
    ceff /= mean_ceff
    
    fe_eff /= mean_ceff
    print("Fly's eye efficiency was ",fe_eff)
    
    if False:
        make_frequency_hist2(cfmid,ctime)
    print("\n\n\n########### ALPHA is ",alpha," ########")
    if True:
        # combined data
        do_ccumulative_plots(cfmid,ctime,cdays,frblists,cowner,tmin=tmin,xeffs = ceff)
    print("#######################################\n\n\n")
    return
    month_histograms(days,time,owner,vdays,vtime,vowner)
    
    
    #returns 23,8,13,2 FRBs in total, low,med,high
    
    #vanessa's data only, not expected to be accurate
    #do_vcumulative_plots(vfmid,vtime,vdays,frblists,owner,tmin=tmin)
    
    
    
    # our logs... unlikely to be accurate?
    #do_cumulative_plots(fmid,tint,Nants,time,days,frblists,owner,tmin=tmin)
    
    
    make_nants_plot(Nants,time)

def make_fp_vector(ICSfmids,ICSfootprints,ICSpitches,sfile="sens_data.dat"):
    """
    
    sfile contains precalculated data on the product of Omega B^1.5,
        which should be approximately proportional to the FRB rate.
        This is contained in the 4th (last) column of the file.
    
    This routine simply efficiently looks up the values of this product
        from that file, and creates a vector of them for the entire data set
    
    
    
    """
    data = np.loadtxt(sfile)
    fps = np.unique(data[:,0])
    freqs = np.unique(data[:,1])
    
    pitches = np.unique(data[:,2])
    
    # creates array for effective efficiency
    effs = np.zeros([ICSfmids.size])
    
    # 9083 1 1272.5 0.9
    
    # now creates an array of relevant data
    for ifp,fp in enumerate(fps):
        OK1 = np.where(ICSfootprints==fp)[0]
        if len(OK1)==0:
            continue
        
        for ifreq,freq in enumerate(freqs):
            OK2 = np.where(ICSfmids[OK1].astype(int)==int(freq*1e6))[0]
            if len(OK2)==0:
                continue
            
            for ipitch,pitch in enumerate(pitches):
                OK3 = np.where(ICSpitches[OK1][OK2] == pitch)[0]
                
                if len(OK3)==0:
                    continue
                
                # slightly faster, but honestly not worthwhile the risk of getting it wrong
                if False:
                    index = ifp + ipitch*fps.size +  ifreq*fps.size*pitches.size
                    eff = data[index,4]/0.8053456715457191
                    # 0.805 is hard-coded efficiency of closepack36 0.9 degree pitch at 1.272 GHz
                else:
                    sOK1 = np.where(data[:,0] == fp)[0]
                    sOK2 = np.where(data[:,1][sOK1] == freq)[0]
                    sOK3 = np.where(data[:,2][sOK1][sOK2] == pitch)[0]
                    eff = data[:,4][sOK1][sOK2][sOK3]
                    if len(eff) > 1:
                        print("Found more than one possible efficiency! Data doubled...")
                        print("Check entries ",sOK1[sOK2][sOK3]," in ",sfile)
                        exit()
                # fills the array
                effs[OK1[OK2[OK3]]] = eff
                
    
    zeros = np.where(effs==0)[0]
    print("Could not find efficiencies for ",len(zeros)," observations")
    
    #for index in zeros:
    #    print(index,ICSfootprints[index],ICSfmids[index],ICSpitches[index],
    #        ICSfootprints[index] in fps,ICSfmids[index] in freqs*1e6+0.5,
    #s        ICSpitches[index] in pitches)
    
    return effs   

def make_histograms(ICStimes,ICSfmids,ICSfootprints,ICSpitches,Nants,label="",sfile="sens_data.dat"):
    """
    Routine that makes weighted beam histograms for low, mid, high etc observations
    
    It does this by loading in histograms for each and every pointing configuration
    
    It then weights these according to the observation time and number of antennas,
    which essentially states that solid angle ~ time ~ Nants^0.75
    
    It then normalises by total time and 25^0.75.
    
    That is, when total time is used in the ASKAP survey file, and sensitivity is
    corresponds to 25 antennas, that survey file gives a good description of the average
    conditions for that frequency when using these beam histogram files.
    
    
    """
    data = np.loadtxt(sfile)
    fps = np.unique(data[:,0])
    freqs = np.unique(data[:,1])
    pitches = np.unique(data[:,2])
    
    # creates array for effective efficiency
    effs = np.zeros([ICSfmids.size])
    
    # 9083 1 1272.5 0.9
    totalhist=0.
    totaltime=0.
    # now creates an array of relevant data
    for ifp,fp in enumerate(fps):
        OK1 = np.where(ICSfootprints==fp)[0]
        if len(OK1)==0:
            continue
        
        for ifreq,freq in enumerate(freqs):
            OK2 = np.where(ICSfmids[OK1].astype(int)==int(freq*1e6))[0]
            if len(OK2)==0:
                continue
            
            for ipitch,pitch in enumerate(pitches):
                OK3 = np.where(ICSpitches[OK1][OK2] == pitch)[0]
                
                if len(OK3)==0:
                    continue
                
                # loads histogram
                datadir = "/Users/cjames/CRAFT/Git/zdm/zdm/beam_generator/ASKAP/"
                
                if fp==1:
                    fpstring="closepack"
                else:
                    fpstring="square"
                fname = datadir+fpstring+"_"+str(pitch)+"_"+str(freq*1e6)+".npy"
                hist = np.load(fname)
                
                thesetimes = ICStimes[OK1[OK2[OK3]]]
                theseants = Nants[OK1[OK2[OK3]]]
                
                weights = thesetimes * (theseants/25.)**0.75
                weight = np.sum(weights)
                totaltime += np.sum(thesetimes)
                totalhist = totalhist + hist*weight
    
    # re-divides by totaltime so that we have, in the end, zero time-weighting
    totalhist /= totaltime
    np.save("ASKAP_"+label+"_hist.npy",totalhist)
    print(label," has total time of ",totaltime," (compare with ",np.sum(ICStimes),")")
    
    return

def get_dm_efficiencies(tsamps,flists,weights,outfile,plot=True):
    """
    Returns a total efficiency due to sampling
    time and frequency
    
    DMs - list of DMs
    tsamps: list of samplin times
    weights: list of weights
    """
    DMs = np.linspace(0.,7000.,701)
    
    tlist = np.unique(tsamps)
    flist = np.unique(flists)
    
    curves = get_sensitivity_curves(DMs,tlist,flist)
    effs = np.zeros([tsamps.size])
    
    csum = 0.
    
    for i,t in enumerate(tlist):
        OK1 = np.where(tsamps == t)[0]
        for j,f in enumerate(flist):
            OK2 = np.where(flists[OK1]==f)[0]
            if len(OK2)>0:
                total_weight = np.sum(weights[OK1[OK2]])
                csum += total_weight * curves[i][j]
    
    
    csum /= np.sum(weights)
    
    if plot:
        plt.figure()
        plt.plot(DMs,csum)
        plt.xlim(0,7000)
        plt.xlabel('DM [pc cm$^{-3}$]')
        plt.ylabel("$\\overline{\\epsilon} (DM)$")
        plt.tight_layout()
        plt.savefig("average_dm_efficiency.png")
        plt.close()
    
    tosave = np.array([DMs,csum])
    print("shape is ",tosave.shape)
    np.save(outfile,tosave)
    
    return csum



def get_sensitivity_curves(DMs,tlist,flist,nu_res = 1,iDM = 4096):
    """
    Generates sensitivity curves for each observation based on the survey.py routine
    
    Inputs:
        DMs: array of DMs for calculating sensitivity
    
    Factors we expect to change:
        w: FRB intrinsic width
        fbar: observation frequency
        tres: time resolution
    
    Factors we expect to stay constant:
        nu_res: 1 MHz
        Nchan: 336 MHz
    
    """
    ########## initiualise lognormal width distribution #######3
    from zdm import parameters
    state = parameters.State()
    import os
    from pkg_resources import resource_filename
    from zdm import survey
    
    # sets path to ASKAP surveys
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/')
    
    # standard 1.4 GHz CRAFT data
    survey_name = 'CRAFT_ICS_1300'
    
    # loads survey
    s = survey.load_survey(survey_name,state,DMs,sdir=sdir)
    
    ######### iterate ######
    DMFRB=0. #(only used if dsmear is true and we need to subtract this from the width)
    df = 336.
    
    curves = []
    for i,tres in enumerate(tlist):
        curves.append([])
        for j,fbar in enumerate(flist):
            s.meta['FBAR'] = fbar
            s.meta['TRES'] = tres
            widths,weights = survey.make_widths(s,state)
            rel_s_bar = 0.
            for k,w in enumerate(widths):
                
                # calculates this array for all FRBs
                rel_s = survey.calc_relative_sensitivity(DMFRB,DMs,w,fbar,tres,nu_res,dsmear=False,model='Quadrature')
                rel_s_bar += rel_s * weights[k]
            
            # sets the curves to zero after a maximum DM
            # solves 4096 * tres = Delta t = 4.196e-3 * DM * (numax^-2 - numin^-2)
            DM_max = iDM * tres / ((fbar/1e3 - df/2e3)**-2 - (fbar/1e3+df/2e3)**-2) / 4.149
            bad = np.where(DMs > DM_max)[0]
            rel_s_bar[bad]=0.
            
            curves[i].append(rel_s_bar)
    
    return curves
                
def get_width_efficiencies(tsamps,flists,maxDM=False,normF=1272,normT=1.182):
    """
    Returns a total efficiency due to sampling
    time and frequency
    """
    tlist = np.unique(tsamps)
    flist = np.unique(flists)
    
    tf_factors = calc_tf_factors(tlist,flist,maxDM=maxDM)
    effs = np.zeros([tsamps.size])
    
    for i,t in enumerate(tlist):
        OK1 = np.where(tsamps == t)[0]
        for j,f in enumerate(flist):
            OK2 = np.where(flists[OK1]==f)[0]
            if len(OK2)>0:
                effs[OK1[OK2]] = tf_factors[i,j]
    
    norm_factor = calc_tf_factors(np.array([normT]),np.array([normF]),maxDM=maxDM)
    print("Calculated norm factor as ",norm_factor)
    effs /= norm_factor[0,0]
    
    return effs
            
def calc_tf_factors(tlist,flist,infile="Tyson", maxDM=False):
    """
    Loads in FRB width and scattering data
    
    Uses FRBs detected by CRAFT as the basis
    
    Flist is frequencies in MHz
    
    """
    
    if infile == "Danica":
        data=np.loadtxt("danica_data.dat")
        dm=data[:,0]
        scat1ghz=data[:,1]
        width=data[:,2]
    else:
        # data from Tyson
        data=np.loadtxt("tyson_data.dat")
        dm=data[:,2]
        scat=data[:,4]
        apparentwidth=data[:,3]
        fbar = data[:,1]
        
        scat1ghz = scat * (fbar/1e3)**4 # correction
        
        width = apparentwidth**2 -scat**2
        bad = np.where(width <= 0.)[0]
        width[bad] = 1e-4
        width = width**0.5
        
    if maxDM:
        OK = np.where(dm < 950)
        dm = dm[OK]
        scat = scat[OK]
        width = width[OK]
    
    NFRB=dm.size
    
    rates = np.zeros([tlist.size,flist.size])
    
    for i,t in enumerate(tlist):
        for j,f in enumerate(flist):
            for k in np.arange(NFRB):
                s=scat1ghz[k]*(f/1e3)**-4
                # single-channel smearing
                dms = 4.18 * 464 * ((f - 0.0005)**-2 - (f + 0.0005)**-2)
                w=width[k]
                eff = (s**2 + dms**2 + w**2 + t**2)**0.5
                # effectvie width
                # sensitivity goes as width^0.5
                # rate goes as sensitivity ^1.5
                # hence: 0.75
                rates[i,j] += eff**-0.75
    rates /= NFRB
    return rates
    
def fp_plot(footprints,pitchs,fmids,time):
    """
    Generates statistics on which footprints are used at what frequencies
    """
    
    
    # gets unique lists of footprints
    fps=np.array([1,2])
    
    # gets unique lists of frequencies
    allfreqs = np.unique(fmids)
    nfreq = allfreqs.size
    
    # gets uniqe list of pitchs
    allpitches = np.unique(pitchs)
    npitch = allpitches.size
    
    sums = np.zeros([2,nfreq,npitch])
    
    for ifp,footprint in enumerate(fps):
        OK1 = np.where(footprints == footprint)[0]
        if len(OK1 == 0):
            continue
        
        for ifreq,freq in enumerate(allfreqs):
            OK2 = np.where(fmids[OK1] == freq)[0]
            if len(OK2 == 0):
                continue
                    
            for ipitch,pitch in enumerate(allpitches):
                OK3 = np.where(pitchs[OK1][OK2]==pitch)[0]
                if len(OK3 == 0):
                    continue
                
                total = np.sum(time[OK1][OK2][OK3])
                
                sums[ifp,ifreq,ipitch] = total
    
    
        
    
def check_frbs(frbids,allids):
    """
    Scans through FRB sbids to check if they are listed in
    the allids observations. Returns indices of those
    which are.
    """
    
    keep = []
    for i,scan in enumerate(frbids):
        if scan in allids:
            keep.append(i)
    return keep
   
def make_combined(owner,time,vowner,vtime,days,vdays,fmid,vfmid,eff,veff,sbids,
    vsbids,tsamp,vtsamp,footprint,vfootprint,pitch,vpitch,Nants,vNants):
    """
    Combines logs of ASKAP data, since CRAFT filler time accidentally was not logged by
    CRAFT prior to mid 2021.
    
    This takes ASKAP logs, and combines those with existing CRAFT logs, to find missing obs
    
    Owners here has the scale:
    1: AS039
    2: AS106
    3: other
    """
    
    O1=np.where(owner==1)[0]
    O2=np.where(owner==2)[0]
    O3=np.where(owner==3)[0]
    
    t1=np.sum(time[O1])
    t2=np.sum(time[O2])
    t3=np.sum(time[O3])
    
    vO1=np.where(vowner==1)[0]
    vO2=np.where(vowner==2)[0]
    vO3=np.where(vowner==3)[0]
    
    vt1=np.sum(vtime[vO1])
    vt2=np.sum(vtime[vO2])
    vt3=np.sum(vtime[vO3])
    print("ASKAP logs: time by owner is ",vt1,vt2,vt3)
    
    # we now analyse the sbids: in short, we are finding all SBIDs in vO1 which are
    # not within O1, i.e. CRAFT-owned
    newvO1=[]
    for index in vO1:
        if not vsbids[index] in sbids[O1]:
            newvO1.append(index)
    print("Rejected ",len(vO1)-len(newvO1)," overlapping datasets")
    
    
    newvO2=[]
    for index in vO2:
        if not vsbids[index] in sbids[O2]:
            newvO2.append(index)
    print("Found ",np.sum(vtime[newvO2])," times in vo2")
    
    # keeps datasets in CRAFT logs which are NOT in ASKAP
    # doing it this way means overlaps are kept from CRAFT
    #newO1=[]
    #for index in O1:
    #    if not sbids[index] in vsbids[newvO1]:
    #        newO1.append(index)
    #print("Keeping ",len(newO1)," CRAFT datasets in CRAFT logs, total time ",
    #    np.sum(time[newO1])+np.sum(time[O2])+np.sum(time[O3]))
    # we always keep all CRAFT logs
    newO1 = O1
    
    # calculates scaling factor by comparing overlapping times
    sblist=[]
    csum=0.
    vsum=0.
    
    # checks scaling factor between ASKAP logs and here
    # relies on overlapping schedblock ids
    for i,sb in enumerate(vsbids[vO2]):
        # checks we have not done this before
        if sb in sblist:
            continue
        else:
            sblist.append(sb)
        indices = np.where(sbids[O2]==sb)[0]
        vindices = np.where(vsbids[vO2]==sb)[0]
        csum += np.sum(time[O2[indices]])
        vsum += np.sum(vtime[vO2[vindices]])
        #print(sb," matched! Adding ",np.sum(time[O2[indices]]),np.sum(vtime[vO2[vindices]]))
    
    scale=csum/vsum # a scale to slightly reduce the total obs time
    #print("Must scale down Vanessa's time by ",scale)
    #print("Scaling ASKAP logs down by ",scale)
    # turns off this functionality
    scale=1.
    
    cdays=np.concatenate((vdays[newvO1],vdays[newvO2],days[newO1],days[O2],days[O3]))
    ctime=np.concatenate((vtime[newvO1]*scale,vtime[newvO2]*scale,time[newO1],time[O2],time[O3]))
    cfmid=np.concatenate((vfmid[newvO1],vfmid[newvO2],fmid[newO1],fmid[O2],fmid[O3]))
    cowner=np.concatenate((vowner[newvO1],vowner[newvO2],owner[newO1],owner[O2],owner[O3]))
    ceff=np.concatenate((veff[newvO1],veff[newvO2],eff[newO1],eff[O2],eff[O3]))
    ctsamp=np.concatenate((vtsamp[newvO1],vtsamp[newvO2],tsamp[newO1],tsamp[O2],tsamp[O3]))
    cfootprint=np.concatenate((vfootprint[newvO1],vfootprint[newvO2],footprint[newO1],footprint[O2],footprint[O3]))
    cpitch=np.concatenate((vpitch[newvO1],vpitch[newvO2],pitch[newO1],pitch[O2],pitch[O3]))
    cNants=np.concatenate((vNants[newvO1],vNants[newvO2],Nants[newO1],Nants[O2],Nants[O3]))
    print("We have added ",np.sum(ctime)-np.sum(time)," hr from ASKAP's logs")
    
    if False:
        plt.figure()
        
        cv = np.cumsum(vtime[vO1])
        order = np.argsort(days[O1])
        cr = np.cumsum(time[O1][order])
        
        plt.plot(vdays[vO1],cv)
        plt.plot(days[O1][order],cr)
        plt.show()
    
    return cdays,ctime,cfmid,cowner,ceff,ctsamp,cfootprint,cpitch,cNants
    
def month_histograms(t1,dt1,o1,t2,dt2,o2):
    
    # time in units of months
    m1=t1/30.
    m2=t2/30.
    
    ######AS039######
    cut1=np.where(o1==1.)
    cut2=np.where(o2==1.)
    bins=np.linspace(0,60,41)
    
    plt.figure()
    plt.xlabel('Approx month (t/30 days)')
    plt.ylabel('Time per month')
    plt.hist(m1[cut1],weights=dt1[cut1],bins=bins,alpha=0.5,label='AS039: CRAFT logs')
    plt.hist(m2[cut2],weights=dt2[cut2],bins=bins,alpha=0.5,label='ASKAP logs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_per_month_hist_AS039.pdf')
    plt.close()
    
    ######AS106######
    cut1=np.where(o1==2.)
    cut2=np.where(o2==2.)
    
    plt.figure()
    plt.xlabel('Approx month (t/30 days)')
    plt.ylabel('Time per month')
    plt.hist(m1[cut1],weights=dt1[cut1],bins=bins,alpha=0.5,label='AS106: CRAFT logs')
    plt.hist(m2[cut2],weights=dt2[cut2],bins=bins,alpha=0.5,label='ASKAP logs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_per_month_hist_AS106.pdf')
    plt.close()
    
    
    
    
    
    
def make_frb_y(frbdays,frbfmids):
    # assumes already sorted by time
    
    lists=[]
    LOW=np.where(frbfmids < 1000.)[0]
    HIGH=np.where(frbfmids > 1400)[0]
    MID=np.where(np.abs(frbfmids-1112.) < 50)[0]
    for i,days in enumerate([frbdays,frbdays[LOW],frbdays[MID],frbdays[HIGH]]):
        x,y=make_cumulative_frb_list(days)
        lists.append([x,y])
    return lists
        
def make_cumulative_frb_list(frbdays): 
    NFRB=frbdays.size
    frby=np.zeros([NFRB])
    frby[:]=1.
    frby=np.cumsum(frby)
    # makes step plot
    frbx=np.zeros([2*NFRB+2])
    frby=np.zeros([2*NFRB+2])
    for i,day in enumerate(frbdays):
        frbx[2*i+1]=day
        frbx[2*i+2]=day
        frby[2*i+1]=i
        frby[2*i+2]=i+1
    frby[-1]=frby[-2]
    frbx[-1]=frbx[-2]+1000 # adds large increment to represent big time increase
    return frbx,frby


def make_frequency_hist2(fmid,time):
    ###### makes a histogram of frequency #######
    plt.figure()
    plt.xlabel('Central frequency [MHz]')
    plt.ylabel('Observations time [hr]')
    bins=np.linspace(700,1800,46)
    plt.hist(fmid,weights=time,bins=bins)
    plt.xlim(800,1800)
    plt.tight_layout()
    plt.savefig("tobs_by_frequency.pdf")
    plt.close()
    
def make_frequency_hist(freq,nchan,dchan,time):
    ###### makes a histogram of frequency #######
    plt.figure()
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Observations time [hr]')
    bins=np.linspace(700,1800,46)
    plt.hist(freq+nchan*dchan/2.,weights=time,bins=bins)
    plt.tight_layout()
    plt.savefig("tobs_by_frequency.pdf")
    plt.close()


def make_healpix_plots(ras,decs,times,frbra,frbdec,fmid,NSIDE = 200,outfile = 'healpix.fits'):
    """
    makes a healpix plot: from Keith Bannister
    """
    import healpy as hp
    fov_radius = np.radians(2.75) # radians. Should be value-by-value
    resol= hp.nside2resol(NSIDE, arcmin=True)
    npix = hp.nside2npix(NSIDE)
    m = np.zeros(npix, dtype=float)
    for i,ra in enumerate(ras):
        vec = hp.ang2vec(ra, decs[i], lonlat=True)
        ipix_disk = hp.query_disc(nside=NSIDE, vec=vec, radius=fov_radius)
        m[ipix_disk] += times[i]
    # make a colormap with white as zero, but ecerything else normal
    #from colormaps.utils import concat
    #concat2 = concat([cmaps.ice, cmaps.BkBlAqGrYeOrReViWh200], ratios=[0.25,0.75])
    
    hp.write_map(outfile, m, overwrite=True, nest=None, coord='C', fits_IDL=False)
    plt.figure()
    #cmap = plt.get_cmap("cubehelix")
    cmap = plt.get_cmap("nipy_spectral")
    
    #frbdays,frbfmids,frb_snrs,frb_bs,frbra,frbdec=load_frb_data(mint=tmin)
    
    x=np.arange(frbra.size).astype('str')
    
    # prints exposures for some key fields
    fields = ([340,-32],[350,-32],[293.5,-63.8],[330,-80],[58,-25],[220,-5])
    
    
    if False:
        # gets theta,phi coordinates of pixels
        theta,phi=hp.pixelfunc.pix2ang(NSIDE,np.arange(npix))
        theta = -np.degrees(theta-pi/2.)
        phi = np.degrees(pi*2.-phi)
        
        
        
    elif False:
        hp.mollview(m, unit='Exposure [hr]', coord='C', 
            cmap=cmap.reversed(),title="",notext=False)
        
        hp.graticule(dpar=30,dmer=30)
        #plt.scatter(x,y,marker='o',s=100)
        hp.projscatter(frbra,frbdec,marker='x',s=30,lonlat=True,coord=["C"],
            linewidths=2,color='white')
        x,y = hammer_xy(frbra,frbdec)
    
    else:
        from healpy.newvisufunc import projview
        fontsize={"cbar_label":14,"xlabel":14,"ylabel":14}
        
        projview(m,coord=["C"],
            graticule=True, graticule_labels=True, graticule_color="black",
            unit=r"Exposure [hr]", xlabel="RA", ylabel="DEC",
            cb_orientation="vertical", min=0, max=1200,cbar_ticks=[0,200,400,600,800,1000,1200],
            latitude_grid_spacing=30, projection_type="hammer",
            title="",
            cmap=cmap.reversed(),fontsize=fontsize)
            ##custom_xtick_labels=["A", "B", "C", "D", "E"],
            ##custom_ytick_labels=["F", "G", "H", "I", "J"],
            #);
            #
        
        theta = np.radians(-frbdec)+np.pi/2.
        phi = np.radians(frbra)
        mod = np.where(phi > np.pi)
        phi[mod] -= 2.*np.pi
        hp.newvisufunc.newprojplot(theta,phi, fmt=None, color="white",marker='x',linestyle="",
            linewidth=2)
        
    ax=plt.gca()
    
    
    #for field in fields:
    #    vec = hp.ang2vec(field[0], field[1], lonlat=True)
    #    ipix_disk = hp.query_disc(nside=NSIDE, vec=vec, radius=0.004)
    #    print("Field ",field," has exposure ",m[ipix_disk])
    #hp.projscatter(frbra,frbdec,marker='+',s=30,lonlat=True,coord=["C"],
    #    linewidths=4,color='purple')
    
    
    plt.tight_layout()
    plt.savefig("exposure_map.pdf")
    plt.close()
    

def hammer_xy(lon,lat):
    """
    Hammer projection
    """
    
    x = 2**1.5 * np.cos(lat)*np.sin(lon/2.) / (1 + np.cos(lat) * np.cos(lon/2.))**0.5
    y = 2**0.5 * np.sin(lat) / (1 + np.cos(lat) * np.cos(lon/2.))**0.5
    return x,y


def make_b_plot(ra,dec,time,frb_bs,freq): 
    
    # get FRb data, add these to plot
    global tmin
    
    c=SkyCoord(ra=ra*u.degree,dec=dec*u.degree,frame='icrs')
    b=c.galactic.b
    l=c.galactic.l
    l=np.array(l)
    b=np.array(b)
    
    ###### makes a plot of cumulative time as a function of b #######
    
    order=np.argsort(np.abs(b))
    newb=np.abs(b[order])
    btime=time[order]
    cbtime=np.cumsum(btime)
    
    frb_bs=np.sort(np.copy(np.abs(frb_bs)))
    cx=np.zeros([2*frb_bs.size+2])
    cy=np.zeros([2*frb_bs.size+2])
    cx[0]=0.
    cy[0]=0.
    for i in np.arange(frb_bs.size):
        cx[2*i+1]=frb_bs[i]
        cx[2*i+2]=frb_bs[i]
        cy[2*i+1]=i
        cy[2*i+2]=i+1
    cx[-1]=90
    cy[-1]=frb_bs.size
    
    
    plt.figure()
    l1,=plt.plot(newb,cbtime,label='Exposure [hr]',color='blue')
    
    plt.ylabel('cumulative exposure [hr]')
    plt.xlabel('Galactic latitude, b [deg]')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    l2,=plt.plot(cx,cy,label='N FRBs',color='orange',linestyle='--')
    ax1.set_ylim(0,np.max(cbtime))
    ax1.set_xlim(0,90)
    ax2.set_ylim(0,frb_bs.size)
    plt.ylabel('cumulative FRBs')
    plt.legend(loc="upper left",handles=[l1,l2],labels=['Exposure','FRBs'])
    plt.tight_layout()
    plt.savefig('cumulative_b.pdf')
    plt.close()
    
def make_pointing_plots(ra,dec,time,frb_bs,freq):  
    """
     plots pointing directions in ra,dec
    """
    global tmin
    scans,frbdays,frbfmids,frb_snrs,frb_bs,frbra,frbdec=load_frb_data(mint=tmin)
    
    
    ########## Exposure Plots ##########
    
    
    torad=np.pi/180.
    sdec=np.sin(dec*torad)
    bbins=np.linspace(-1,1,21)
    lbins=np.linspace(0,360,37)
    cH,xedge,yedge=np.histogram2d(ra,sdec,bins=[lbins,bbins],weights=time)
    #cH is histogram of total time per field
    
    plt.figure()
    plt.imshow(cH.T,origin='lower',extent=[0,360,-1,1.],cmap=plt.get_cmap("plasma"),aspect=180)
    cbar=plt.colorbar()
    plt.xlabel("$\\alpha$ [deg]")
    plt.ylabel("$\\sin \\delta$")
    cbar.set_label("Exposure [hr]")
    
    plt.scatter(frbra*np.pi/180.,np.sin(frbdec*np.pi/180.),marker='x')
    plt.tight_layout()
    plt.savefig("exposure_celestial_coordinates.pdf")
    plt.close()
    
    # makes a much more detailed grid
    bbins2=np.linspace(-1,1,201)
    lbins2=np.linspace(0,360,361)
    cH2,xedge2,yedge2=np.histogram2d(ra,sdec,bins=[lbins2,bbins2],weights=time)
    cH2 = np.array(cH2)
    
    # NOTE: actually ra,dec not l,b!!!
    bcentres = (bbins2[1:] + bbins2[0:-1])/2.
    lcentres = (lbins2[1:] + lbins2[0:-1])/2.
    plt.figure()
    
    maxx,maxy=np.where(cH2 == np.max(cH2))
    bestra = (lbins2[maxx]+lbins2[maxx+1])/2
    bestdec = np.arcsin((bbins2[maxy]+bbins2[maxy+1])/2.)*180./np.pi
    print("Peak exposure at ",bestra,bestdec)
    plt.imshow(cH2.T,origin='lower',extent=[0,360,-1,1.],cmap=plt.get_cmap("plasma"),aspect=180)
    cbar=plt.colorbar()
    plt.xlabel("$\\alpha$ [deg]")
    plt.ylabel("$\\sin \\delta$")
    cbar.set_label("Exposure [hr]")
    plt.tight_layout()
    plt.savefig("exposure_celestial_coordinates_fine.pdf")
    plt.close()
    print("Hours per FRB are ",np.sum(cH2)/frbra.size)
    
    ######### creates a detailed grid for plotting sky coordinates #####
    
    new_arr = smear_pointings(cH2,lcentres,bcentres,2.5)
    
    plt.figure()
    
    maxx,maxy=np.where(new_arr == np.max(new_arr))
    bestra = (lbins2[maxx]+lbins2[maxx+1])/2
    bestdec = np.arcsin((bbins2[maxy]+bbins2[maxy+1])/2.)*180./np.pi
    print("Peak exposure at ",bestra,bestdec)
    plt.imshow(new_arr.T,origin='lower',extent=[0,360,-1,1.],cmap=plt.get_cmap("plasma"),aspect=180)
    cbar=plt.colorbar()
    plt.xlabel("$\\alpha$ [deg]")
    plt.ylabel("$\\sin \\delta$")
    cbar.set_label("Exposure [hr]")
    
    plt.scatter(frbra,np.sin(frbdec*np.pi/180.),marker='x')
    plt.tight_layout()
    plt.savefig("smeared_exposure_celestial_coordinates.pdf")
    plt.close()
    
    ###### plots pointing directions in Galactic coordinates #####
    
    c=SkyCoord(ra=ra*u.degree,dec=dec*u.degree,frame='icrs')
    b=c.galactic.b
    l=c.galactic.l
    l=np.array(l)
    b=np.array(b)
    torad=np.pi/180.
    sb=np.sin(b*torad)
    #bbins=np.linspace(-90,90,18)
    #bbins=np.linspace(-1,1,21)
    #lbins=np.linspace(0,360,37)
    H,xedge,yedge=np.histogram2d(l,sb,bins=[lbins,bbins],weights=time)
    
    
    plt.figure()
    plt.imshow(H.T,origin='lower',extent=[0,360,-1,1.],cmap=plt.get_cmap("plasma"),aspect=180)
    cbar=plt.colorbar()
    plt.xlabel("$\\ell$ [deg]")
    plt.ylabel("$\\sin b$")
    cbar.set_label("Exposure [hr]")
    plt.tight_layout()
    plt.savefig("exposure_galactic_coordinates.pdf")
    plt.close()
    
    
    
    ###### makes a histogram of the time per field #######
    fH = cH.flatten()
    fH /= 24. # total time per field in days
    bins = np.linspace(0,100.,101) # bins in units of days
    plt.figure()
    plt.hist(fH,bins=bins)
    plt.xlabel('$T_{\\rm field}$ [days]')
    plt.yscale('log')
    plt.ylabel('$N_{\\rm fields}$') # sort-of... only if H has the same number of bins as pointings in the sky
    plt.xlim(0,45)
    plt.tight_layout()
    plt.savefig("points_with_days.pdf")
    plt.close()
    
    
    ######### splits this by frequency #########
    LOW=np.where(freq < 1000.)[0]
    HIGH=np.where(freq > 1400)[0]
    MID=np.where(np.abs(freq-1200) <= 200)[0]
    print("Lengths are ",len(LOW),len(MID),len(HIGH),freq.size)
    
    # histogram the values of ra and dec into bins, weighted by the time per field
    # the result will be a time per pointing 2D hist in units of ra and dec
    LOWH,xedge,yedge=np.histogram2d(ra[LOW],sdec[LOW],bins=[lbins,bbins],weights=time[LOW])
    MIDH,xedge,yedge=np.histogram2d(ra[MID],sdec[MID],bins=[lbins,bbins],weights=time[MID])
    HIGHH,xedge,yedge=np.histogram2d(ra[HIGH],sdec[HIGH],bins=[lbins,bbins],weights=time[HIGH])
    # the below are fractions of time in each field
    fLOWH = LOWH.flatten()/24./fH # /24 converts from hours to days. fH is total. This is fractional
    fMIDH = MIDH.flatten()/24./fH
    fHIGHH = HIGHH.flatten()/24./fH
    bad = np.where(fH==0.)[0]
    fLOWH[bad]=0.
    fMIDH[bad]=0.
    fHIGHH[bad]=0.
    
    # the below histograms thefull time per field, but gives fractions according the the low/mid/high weighting
    plt.figure()
    plt.hist(fH,bins=bins,weights=fLOWH+fMIDH+fHIGHH,label='HIGH')
    plt.hist(fH,bins=bins,weights=fLOWH+fMIDH,label='MID')
    plt.hist(fH,bins=bins,weights=fLOWH,label='LOW')
    plt.xlabel('Time per field [days]')
    
    plt.ylabel('Nfields') # sort-of... only if H has the same number of bins as pointings in the sky
    plt.legend()
    plt.tight_layout()
    plt.ylim(0,40)
    plt.savefig("linear_by_frequency_points_with_days.pdf")
    
    plt.ylim(0.2,1000)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("by_frequency_points_with_days.pdf")
    plt.close()
    
    #################################################
    # calculates some statistics for time per field #
    #################################################
    
    TotalTime = np.sum(fH)
    print("TotalTime is ",TotalTime*24.)
    print("Times for low/mid/high are ",np.sum(time[LOW]),np.sum(time[MID]),np.sum(time[HIGH]))
    print("This sums to ",np.sum(time[LOW])+np.sum(time[MID])+np.sum(time[HIGH]))
    # times OK up to here
    
    # Now does per frequency info
    Tlow = np.sum(LOWH)
    Tmid = np.sum(MIDH)
    Thigh = np.sum(HIGHH)
    TTots = np.array([Tlow,Tmid,Thigh])
    
    # gets number of fields
    NFlow = np.where(LOWH.flatten() > 0.)[0]
    NFmid = np.where(MIDH.flatten() > 0.)[0]
    NFhigh = np.where(HIGHH.flatten() > 0.)[0]
    NFlow = len(NFlow)
    NFmid = len(NFmid)
    NFhigh = len(NFhigh)
    NF = np.array([NFlow,NFmid,NFhigh])
    
    # mean time per field
    TBarLow = Tlow/NFlow
    TBarMid = Tmid/NFmid
    TBarHigh = Thigh/NFhigh
    Tbars = np.array([TBarLow,TBarMid,TBarHigh])
    
    # time-weighted time per field
    TTlow = (np.sum(LOWH**2)/NFlow)**0.5
    TTmid = (np.sum(MIDH**2)/NFmid)**0.5
    TThigh = (np.sum(HIGHH**2)/NFhigh)**0.5
    TTbars = np.array([TTlow,TTmid,TThigh])
    
    NFTlow = Tlow / TTlow
    NFTmid = Tmid/TTmid
    NFThigh = Thigh/TThigh
    NFT = np.array([NFTlow,NFTmid,NFThigh])
    
    print("Calculated time per low/mid/high as ",Tlow,Tmid,Thigh)
    print("Nfields are ",NF)
    print ("Hence, time per field is ",Tbars)
    print("Weighted by time, it is ",NFT)
    print(" and for TTbars ",TTbars)
    
    # saves information allowing construction of repeat grid
    outdir="Repetition/"
    
    ###### full field info #######
    # just in case it's needed (if it is, may God have mercy on our souls)
    nz = np.where(fLOWH.flatten() > 0.)[0]
    fLOWH = fLOWH.flatten()[nz]
    nz = np.where(fMIDH.flatten() > 0.)[0]
    fMIDH = fMIDH.flatten()[nz]
    nz = np.where(fHIGHH.flatten() > 0.)[0]
    fHIGHH = fHIGHH.flatten()[nz]
    
    np.save(outdir+"LOWH.npy",fLOWH)
    np.save(outdir+"MIDH.npy",fMIDH)
    np.save(outdir+"HIGHH.npy",fHIGHH)
    
    ####### Histograms ######
    # all time, regardless of frequency
    nz = np.where(fH > 0.)[0]
    TimePerField,bins = np.histogram(fH[nz],bins=bins)
    np.save(outdir+"allfieldshist.npy",TimePerField)
    
    TotalTime = np.sum(fH)
    nnz = len(np.where(fH > 0.)[0])
    TField = TotalTime/nnz
    TTfield = np.sum(fH**2/nnz)**0.5
    TNfields = TotalTime/TTfield
    
    AllTimeInfo = np.array([TField,nnz,TTfield,TNfields])
    np.save(outdir+"alltimeinfo.npy",AllTimeInfo)
    
    # gets time per field in each bin. This is because
    # we can't just use bin centres
    tbins = np.zeros([bins.size-1])
    tbins_total = 0.
    for i,b in enumerate(bins[:-1]):
        blow = b
        bhigh = bins[i+1]
        OK1 = np.where(fH > blow)[0]
        OK2 = np.where(fH < bhigh)[0]
        OK = np.intersect1d(OK1,OK2)
        nfields = len(OK)
        Tfields = np.sum(fH[OK])
        if nfields>0:
            tbins[i]=Tfields/nfields
            tbins_total += tbins[i] * nfields
            print("Nfields ",nfields,TimePerField[i])
    print("Total calculated time from field bins is ",tbins_total)
    
    LowHist,bins = np.histogram(fLOWH,bins=bins)
    np.save(outdir+"LowHist.npy",LowHist)
    
    MidHist,bins = np.histogram(fMIDH,bins=bins)
    np.save(outdir+"MidHist.npy",MidHist)
    
    HighHist,bins = np.histogram(fHIGHH,bins=bins)
    np.save(outdir+"HighHist.npy",HighHist)
    np.save(outdir+"bins.npy",bins)
    np.save(outdir+"tbins.npy",tbins)
    
    ### Total time info ####
    np.save(outdir+"Tbars.npy",Tbars)
    np.save(outdir+"TTbars.npy",TTbars)
    np.save(outdir+"NF.npy",NF)
    np.save(outdir+"NFT.npy",NFT)
    np.save(outdir+"Ttot.npy",TTots)
    
    np.save(outdir+"total_time.npy",TotalTime)

def smear_pointings(array,ras,sindecs,smear):
    """
    Smears pointings in array by 'smear' in degrees
    """
    
    xs = np.cos(ras)
    nra,ndec = array.shape
    newarr = np.zeros(array.shape)
    xs = np.zeros(array.shape)
    ys = np.zeros(array.shape)
    zs = np.zeros(array.shape)
    
    cosdecs = (1.-sindecs**2.)**0.5
    sinras = np.sin(ras * np.pi/180.)
    cosras = np.cos(ras * np.pi/180.)
    xs = np.outer(cosras,cosdecs)
    ys = np.outer(sinras,cosdecs)
    zs = np.repeat([sindecs],nra,axis=0)
    #print(xs.shape,zs.shape,array.shape,ras.shape,sindecs.shape)
    
    
    
    goal = np.cos(smear * np.pi/180.)
    print("The target goal is ",goal,smear)
    
    for i in np.arange(nra):
        print("Smearing ",i," of ",nra)
        for j in np.arange(ndec):
            dists = xs*xs[i,j] + ys*ys[i,j] + zs*zs[i,j]
            OKi,OKj = np.where(dists > goal)
            #print("For ",i,j," we have ",len(OKi)," matchs")
            newarr[i,j] += np.sum(array[OKi,OKj])
    return newarr

def make_nants_plot(Nants,time):   
    ######## number of operating antennas #####
    
    plt.figure()
    bins=np.linspace(0.5,36.5,36)
    plt.hist(Nants,bins=bins,weights=time,alpha=0.5,color='red')
    plt.xlabel('$N_{\\rm nats}$')
    plt.ylabel('observing time [hr]')
    plt.xticks([6,12,18,24,30,36])
    plt.tight_layout()
    plt.savefig("Nant.pdf")
    plt.close()

def calc_weights3(tint,time,fmid,Nants):   
    ######### EXPOSURES ########
    # DM 300 FRB, where smearing is 1.4ms at 1.4 GHz
    DM=300
    tsmear=1.3*(DM/300.)*(fmid/1400.)**-3
    weff=(tint**2+tsmear**2)**0.5 # effective width of burst
    teffect=weff**-0.75 # senstivity down as width up with sqrt(w); rate as w^1.5
    
    Neffect=Nants**0.75
    
    # effect of frequency on rate
    feffect=(fmid/1000.)**-1.5
    
    weight=time*Neffect*teffect*feffect
    return weight


def load_vanessa():
    file1 = open('craftdump_inf.csv', 'r')
    Lines = file1.readlines()
    dts=[]
    freqs=[]
    errs=[]
    dates=[]
    owners=[]
    terr = 0.
    tmiss = 0.
    sbids=[]
    footprint=[]
    pitch=[]
    sbids=[]
    
    for i,line in enumerate(Lines):
        if i==0:
            continue
        words=line.split(',')
        project=words[4]
        # these projects are: 
        if project=='AS106':
            owner=2
        elif project=='AS039':
            owner=1
        else:
            owner=3 
        
        duration=words[7]
        err=words[2]
        if err=='OBSERVED' or err=='COMPLETED' or err=='PROCESSING':
            errcode=0
        elif err=='ERRORED':
            errcode=1
        elif err=='RETIRED': # we observe FRBs in retired mode
            errcode=2
        else:
            print("Unknown error code ",err)
        if errcode==1:
            terr += float(duration)
            continue
        
        freq=words[11]
        if freq=='-':
            tmiss += float(duration)
            continue
            #freq = 1272.5
            
        if words[9] == "" or words[9] == "-":
            #tmiss += float(duration)
            #continue
            words[9]="0.9"
            
        start=words[5]
        sbid=words[0]
        start=start.split(' ')
        date=start[0].split('-')
        time=start[1].split(':')
        
        # removes separators from string to reproduce long format
        datestring = date[0]+date[1]+date[2]
        datestring = datestring +time[0]+time[1]+time[2][0:2]
        
        
        if words[8] == "closepack36":
            footprint.append(1)
        elif words[8] == "square_6x6":
            footprint.append(2)
        else:
            footprint.append(0)
            
        
        
        pitch.append(float(words[9]))
        
        freqs.append(float(freq))
        dates.append(int(datestring))
        dts.append(float(duration))
        owners.append(int(owner))
        sbids.append(int(words[0]))
        errs.append(errcode)
        sbids.append(sbid)
    print("Total error time ignored is ",terr/3600.)
    print("Total missing data is ",tmiss/3600.)
    
    dts=np.array(dts)
    dates=np.array(dates)
    freqs=np.array(freqs)
    owners=np.array(owners)
    errs=np.array(errs)
    sbids=np.array(sbids,dtype='int')
    footprint = np.array(footprint)
    pitch=np.array(pitch)
    sbids = np.array(sbids)
    
    return dates,dts,owners,freqs,errs,sbids,footprint,pitch,sbids

def calc_weights3(tint,time,fmid,Nants):   
    ######### EXPOSURES ########
    # assumes tres dominated
    # purely scaled assuming DM smearing is dominant
    sens = (Nants/25.)**0.5 #incoherent sum
    sens *= (tint/1250)**-0.5 #sens as width^0.5 tint dominated
    sens *= (fmid/1200.)**-0.4 # frequency "spectral index" (this value for rate approximation
    weight = time*(sens**1.5)
    return weight

def calc_weights4(tint,time,fmid,Nants):   
    ######### EXPOSURES ########
    # assumes scattering dominated
    # purely scaled assuming DM smearing is dominant
    sens = (Nants/25.)**0.5 #incoherent sum
    sens *= (fmid/1200.)**2 #width as f/1200^-4, sens as width^0.5 scattering dominated
    sens *= (fmid/1200.)**-1.5 # frequency "spectral index" (this value for rate approximation
    weight = time*(sens**1.5)
    return weight

def calc_weights(tint,time,fmid,Nants):   
    ######### EXPOSURES ########
    # purely scaled assuming DM smearing is dominant
    sens = (Nants/25.)**0.5 #incoherent sum
    sens *= (fmid/1200.)**1.5 #width as f/1200^-3, sens as width^0.5 DM smearing dominated
    sens *= (fmid/1200.)**-1.8 # frequency "spectral index" (this value for rate approximation
    weight = time*(sens**1.5)
    return weight
   
def load_frb_data(mint=None,maxt=None, maxDM=False):
    data=np.loadtxt('FRBlist.dat',dtype='str')
    
    SNRs=data[:,1].astype('float')
    fupper=data[:,2].astype('float')
    bs=data[:,3].astype('float')
    ra=get_deg(data[:,4])*15.
    dec=get_deg(data[:,5])
    fmid=fupper-336./2.
    scans = data[:,0].astype('int')
    times=np.copy(data[:,0]).astype('float')
    days=convert_times(times)
    DMs=data[:,6].astype('float')
    
    if mint is not None:
        OK=np.where(days > mint)[0]
        days=days[OK]
        fmid=fmid[OK]
        SNRs=SNRs[OK]
        bs=bs[OK]
        ra=ra[OK]
        dec=dec[OK]
        scans=scans[OK]
        DMs=DMs[OK]
    
    if maxDM:
        # removes FRBs with DMs above the max DM for the min observation frequency
        # min integration time is 1.182
        min_time = 1.182 * 4096
        min_freq = 831.5
        bw=336
        DMcut = min_time / (4.15e6 * ((850-bw/2.)**-2 - (850 + bw/2.)**-2))
        print("DM cutoff is ",DMcut)
        
        OK = np.where(DMs < DMcut)[0]
        print("Removing ",DMs.size-len(OK)," FRBs for having too high a DM")
        days=days[OK]
        fmid=fmid[OK]
        SNRs=SNRs[OK]
        bs=bs[OK]
        ra=ra[OK]
        dec=dec[OK]
        scans=scans[OK]
        DMs=DMs[OK]
        print("Mean DM is ",np.sum(DMs)/DMs.size)
        
    return scans,days,fmid,SNRs,bs,ra,dec
    
def get_deg(strings):
    degs = np.zeros(len(strings))
    for i,string in enumerate(strings):
        bits = string.split(':')
        degs[i] = float(bits[0])+float(bits[1])/60.+float(bits[2])/3600.
    return degs
    
    
def do_vcumulative_plots(xfmid,xtime,xdays,frblists,xowner,tmin=None):
    ##### crude model of sensitivity ######
    # does this without certain information - using data sent by Vanessa
    
    ##### QUESTION - WHY BOTH TIME, TINT, AND DAYS????? ######
    
    # first orders data
    args=np.argsort(xdays)
    days=xdays[args]
    fmid=xfmid[args]
    time=xtime[args]
    days=xdays[args]
    owner=xowner[args]
    
    # calculates different weighting schemes
    
    outfiles=["vanessa_hour_exposure.pdf"]
    ylabels=["Exposure [hr]"]
    norms=[False,False]
    expected=[False,False]
    print("tmin is ",tmin)
    for i,weight in enumerate([time]):
        plot_with_weight(days,weight,fmid,outfiles[i],frblists,
            ylabels[i],norm=norms[i],expected=expected[i],tmin=tmin)
    
    o1=np.where(owner==1)[0]
    o2=np.where(owner==2)[0]
    o3=np.where(owner==3)[0]
    # OWNER code
    #if owner == 'AS039':
    #    ocode=1
    #elif owner=='AS106':
    #    ocode=2
    #else:
    #    ocode=3
    labels=["AS039","AS106","Other"]
    plt.figure()
    plt.xlabel('Days since Jan 1st 2018')
    
    plt.ylabel('Cumulative exposure [hr]')
    plt.ylim(0,4000)
    ctime=np.cumsum(time)
    for i,selection in enumerate([o1,o2,o3]):
        plt.plot(days[selection],np.cumsum(time[selection]),label=labels[i])
    plt.legend()
    
    date_labels()
    plt.xlim(300,1450)
    plt.tight_layout()
    plt.savefig("vanessa_by_owner.pdf")
    plt.close()


def do_ccumulative_plots(xfmid,xtime,xdays,frblists,xowner,tmin=0,xeffs=None):
    ##### crude model of sensitivity ######
    # does this without certain information - using data sent by Vanessa
    
    ##### QUESTION - WHY BOTH TIME, TINT, AND DAYS????? ######
    
    # first orders data
    args=np.argsort(xdays)
    days=xdays[args]
    fmid=xfmid[args]
    time=xtime[args]
    days=xdays[args]
    owner=xowner[args]
    effs=xeffs[args]
    
    # calculates different weighting schemes
    
    outfiles=["combined_hour_exposure.pdf"]
    ylabels=["Exposure [hr]"]
    weights=[time]
    if effs is not None:
        outfiles.append("weighted_combined_hour_exposure.pdf")
        
        weights.append(time*effs)
        
    norms=[False,False]
    expected=[False,False]
    
    print("tmin is ",tmin)
    for i,weight in enumerate(weights):
        if i==0:
            continue
        plot_with_weight(days,weight,fmid,outfiles[i],frblists,
            ylabels[0],norm=norms[i],expected=expected[i],tmin=tmin,
            April=True)
    
    
    o1=np.where(owner==1)[0]
    o2=np.where(owner==2)[0]
    o3=np.where(owner==3)[0]
    # OWNER code
    #if owner == 'AS039':
    #    ocode=1
    #elif owner=='AS106':
    #    ocode=2
    #else:
    #    ocode=3
    labels=["AS039","AS106","Other"]
    plt.figure()
    plt.xlabel('Days since Jan 1st 2018')
    
    plt.ylabel('Cumulative exposure [hr]')
    plt.ylim(0,4000)
    ctime=np.cumsum(time)
    for i,selection in enumerate([o1,o2,o3]):
        plt.plot(days[selection],np.cumsum(time[selection]),label=labels[i])
    plt.legend()
    
    date_labels()
    plt.xlim(tmin,1450)
    plt.tight_layout()
    plt.savefig("combined_by_owner.pdf")
    plt.close()

def do_cumulative_plots(xfmid,xtint,xNants,xtime,xdays,frblists,xowner,tmin=None):
    ##### crude model of sensitivity ######
    
    # first orders data
    args=np.argsort(xdays)
    days=xdays[args]
    fmid=xfmid[args]
    tint=xtint[args]
    Nants=xNants[args]
    time=xtime[args]
    days=xdays[args]
    owner=xowner[args]
    
    # calculates different weighting schemes
    w1=calc_weights(tint,time,fmid,Nants)
    outfiles=["weighted_exposure.pdf","hour_exposure.pdf"]
    ylabels=["weighted exposure [hr]","Exposure [hr]"]
    norms=[False,False]
    expected=[False,False]
    print("tmin is ",tmin)
    for i,weight in enumerate([w1,time]):
        plot_with_weight(days,weight,fmid,outfiles[i],frblists,
            ylabels[i],norm=norms[i],expected=expected[i],tmin=tmin)
    
    o1=np.where(owner==1)[0]
    o2=np.where(owner==2)[0]
    o3=np.where(owner==3)[0]
    # OWNER code
    #if owner == 'AS039':
    #    ocode=1
    #elif owner=='AS106':
    #    ocode=2
    #else:
    #    ocode=3
    labels=["AS039","AS106","Other"]
    plt.figure()
    plt.xlabel('Days since Jan 1st 2018')
    
    plt.ylabel('Cumulative exposure [hr]')
    plt.ylim(0,4000)
    ctime=np.cumsum(time)
    for i,selection in enumerate([o1,o2,o3]):
        plt.plot(days[selection],np.cumsum(time[selection]),label=labels[i])
    plt.legend()
    
    date_labels()
    plt.xlim(300,1450)
    plt.tight_layout()
    plt.savefig("by_owner.pdf")
    plt.close()

def date_labels():
    yeardays=[0,365,730,1095,1460,1825]
    ylabels=["2018 Jan","2019 Jan","2020 Jan","2021 Jan","2022 Jan", "2023 Jan"]
    mlabels=["Feb","Mar","Apr","May","June","July","Aug","Sep","Oct","Nov","Dec"]
    cmonth=np.array([0,31,59,90,120,151,181,212,243,273,304,334])
    values=[]
    labels=[]
    for i,y in enumerate(yeardays):
        values.append(y)
        labels.append(ylabels[i])
        for j,label in enumerate(mlabels):
            if j % 2 == 0:
                continue
            values.append(y+cmonth[j+1])
            labels.append(label)
    plt.xticks(values,labels)
    plt.xticks(rotation=90,fontsize=8)
  
def plot_with_weight(days,weight,fmid,outfile,frblists,ylabel,norm=False,expected=False,tmin=None,
    April=False):
    """
    weight is simply an hour-equivalent time which weights actual hours by some function
    
    FRB data
        frblists gives frb times of observation
    
    ASKAP data
        fmid is the frequency of observationweight is effective observation time [hr]
    
    Norm: if true, weight integrates to unity
    """
    
    # process weights into cumulative
    cweight=np.cumsum(weight)
    
    # normalises to unity
    if norm:
        norm=cweight[-1]
        cweight /= norm
    
    ######### separates frequency ranges #######
    # does this for frequency ranges
    LOW=np.where(fmid < 1000.)
    low_cweight = np.cumsum(weight[LOW])
    if norm:
        low_cweight /= norm
    
    # does this for frequency ranges
    MID1=np.where(fmid > 1000.)[0]
    MID2=np.where(fmid < 1500.)[0]
    MID=np.intersect1d(MID1,MID2)
    
    HIGH=np.where(fmid > 1500.)[0]
    
    mid_cweight = np.cumsum(weight[MID])
    if norm:
        mid_cweight /= norm
    
    high_cweight = np.cumsum(weight[HIGH])
    
    
    print("Total hours to ",outfile," are ",low_cweight[-1],mid_cweight[-1],high_cweight[-1])
    
    # unpack FRB list data
    [frbx,frby],[lfrbx,lfrby],[mfrbx,mfrby],[hfrbx,hfrby]=frblists
    print("Total FRBs are ",lfrby[-1],mfrby[-1],hfrby[-1])
    ######## plot #######
    plt.figure()
    
    ####### Fredda Date Change #######
    # Fixed: Tuesday April 7th 2020, SBID is 
    #FREDDA3=np.where(times==20200408061039)[0]
    #plt.plot([FREDDA3,FREDDA3],[0,1],linestyle='--',color='black')
    #plt.text(FREDDA3+200,0.05,'FREDDA fix',rotation=90,fontsize=12)
    #plt.xlim(150,days[-1])
    
    plt.plot(days,cweight,label='Total',linestyle=":")
    ax1=plt.gca()
    ax2=ax1.twinx()
    
    ax2.plot(frbx,frby,color=ax1.lines[-1].get_color(),linestyle="-")
    
    ##### we now select only frb data in a certain time interval #####
    #   The y-axis limits are normalised such that the total number of FRBs,
    #    and cumulative time, are equal in this interval
    tmax = 365*6
    OK=np.where(frbx<tmax)[0][-1]
    plt.ylim(0,frby[OK])
    plt.ylabel("$N_{\\rm FRB}$")
    
    #plots low frequencies
    ax1.plot(days[LOW],low_cweight,linestyle=":",label='900 MHz')
    ax2.plot(lfrbx,lfrby,color=ax1.lines[-1].get_color(),linestyle="-")
    #lfrby expected rate is 43/2 times that of Fly's Eye, which was 1.57 FRBs/100 days
    # hence this should be 34 FRBs/100 days = 34 FRBs/2400 hr, slope = 2400 hr/34 FRB = 70.6~70 hr/FRB
    if expected: 
        ax2.plot(days[LOW],low_cweight/70.*low_cweight[-1]/lfrby[-1]/2.,linestyle=":",color=plt.gca().lines[-1].get_color())
        #plt.plot(lfrbx,lfrby*70.,color=plt.gca().lines[-1].get_color())
    
    # plots mid frequencies
    ax1.plot(days[MID],mid_cweight,linestyle=":",label='1.3 GHz')
    ax2.plot(mfrbx,mfrby,color=ax1.lines[-1].get_color(),linestyle="-")
    # rate of mid is 20/43 compared to low, hence slope is 70*43/20 ~ 150 hr/FRB
    if expected: 
        #plt.plot(mfrbx,mfrby*150.,color=plt.gca().lines[-1].get_color())
        ax2.plot(days[MID],mid_cweight/150.*mid_cweight[-1]/mfrby[-1]/2.,linestyle=":",color=plt.gca().lines[-1].get_color())
    
    
    # plots high frequencies
    ax1.plot(days[HIGH],high_cweight,linestyle=":",label='1.6 GHz')
    
    ax2.plot(hfrbx,hfrby,color=ax1.lines[-1].get_color(),linestyle="-")
    # rate of mid is 20/43 compared to low, hence slope is 70*43/20 ~ 150 hr/FRB
    if expected: 
        #plt.plot(mfrbx,mfrby*150.,color=plt.gca().lines[-1].get_color())
        ax2.plot(days[HIGH],high_cweight/150.*high_cweight[-1]/hfrby[-1]/2.,linestyle=":",color=plt.gca().lines[-1].get_color())
    
    
    plt.sca(ax1)
    plt.ylabel(ylabel)
    
    date_labels()
    
    plt.xlim(tmin,tmax)
    OK=np.where(days < tmax)[0][-1]
    
    #plt.ylim(0,int(cweight[OK]/1000)*1000+1000)
    plt.ylim(0.,cweight[OK])
    # adding text for standard plot
    ICSpaper=True
    if ICSpaper and tmin <= 300:
        plt.text(1800,14300,"Total")
        plt.text(1650,7200,"1.3 GHz")
        plt.text(1750,3400,"900 MHz")
        plt.text(1800,1600,"1.6 GHz")
    
        April6_2020 = 365*2+31+28+31+6
        plt.plot([April6_2020,April6_2020],[0,cweight[OK]],color='black',linestyle='--')
        #plt.text(April6_2020+30,cweight[OK]/3.,"Stable
        
        August30_2019 = 365*2 - 30*2-31*2-1
        plt.plot([August30_2019,August30_2019],[0,cweight[OK]],color='black',linestyle='--')
        
        x1 = (tmin+August30_2019)/3.
        x2 = (August30_2019 + April6_2020)/2.2
        x3 = April6_2020 * 1.3
        y = cweight[OK] * 0.8
        plt.text(x1,y,'v1')
        plt.text(x2,y,'v2')
        plt.text(x3,y,'v3')
        plt
    elif ICSpaper:
        # setting tmin = 865
        plt.text(1900,10000,"Total")
        plt.text(1900,5300,"900 MHz")
        plt.text(2000,2200,"1.3 GHz")
        plt.text(1850,600,"1.6 GHz")
        plt.ylabel("Weighted exposure [hr]")
        plt.sca(ax2)
        plt.ylabel("$N_{\\rm FRB}~[{\\rm DM} < 980\\,{\\rm pc \\, cm^{-3}})$")
    else:  
        plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    
    #print("livetimes (hr) are ",cweight[np.where(days < tmax)[0][-1]],
    #    low_cweight[np.where(days[LOW] < tmax)[0][-1]],mid_cweight[np.where(days[MID] < tmax)[0][-1]])
    ##### prints out total info ######
    #print("Livetimes (hr) are ",cweight[-1]," with low/high being ",low_cweight[-1],mid_cweight[-1])
    #print("Number of FRBs are ",frby[np.where(frbx < tmax)[0][-1]],
    #    lfrby[np.where(lfrbx < tmax)[0][-1]],mfrby[np.where(mfrbx < tmax)[0][-1]])

def convert_times(stimes):
    """ converts sbid times to simple months since begin 2018 """
    cmonth=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
    year=np.array(stimes/10000000000,dtype='int')
    month=np.array(stimes/100000000,dtype='int')-year*100
    day=np.array(stimes/1000000,dtype='int')-year*10000-month*100
    #returns days since 2018
    total_days=(year-2018)*365 + cmonth[month] + day
    return total_days

#for alpha in np.linspace(1.5,0.6,10):
#    main(alpha=alpha)
main()
