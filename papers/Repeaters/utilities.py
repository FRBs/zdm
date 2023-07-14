"""
Contains utility functions for reading CHIME data,
and other useful things
"""


import numpy as np
import os
from pkg_resources import resource_filename
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import survey
from zdm import grid

def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            init_state=None,
            state_dict=None, iFRB:int=0,
               alpha_method=1, NFRB:int=100, 
               lum_func:int=2,sdir=None):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        init_state (State, optional):
            Initial state
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        iFRB (int, optional): Starting index for the FRBs.  Defaults to 0
        lum_func (int, optional): Flag for the luminosity function. 
            0=power-law, 1=gamma.  Defaults to 0.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters

    Raises:
        IOError: [description]

    Returns:
        tuple: Survey, Grid objects
    """
    
    # Init state
    if init_state is None:
        state = loading.set_state(alpha_method=alpha_method)
        # Addiitonal updates
        if state_dict is None:
            state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
            state.energy.luminosity_function = lum_func
        state.update_param_dict(state_dict)
    else:
        state = init_state
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'),
        zlog=False,nz=500)

    ############## Initialise surveys ##############
    if sdir is not None:
        print("Searching for survey in directory ",sdir)
    else:
        sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    
    
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, nbins=5,
                                 iFRB=iFRB, original=True)
    
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]


def make_bayes(arr,givenorm=False):
    """
    Makes a logarithmic array into a p-distribution
    assuming uniform Bayesian priors
    """
    themax = np.max(arr)
    arr -= themax
    arr = 10**arr
    thesum = np.sum(arr)
    arr /= thesum
    if givenorm:
        return arr,themax,thesum    
    else:
        return arr

def get_contour_level(arr,levels):
    """
    determines the level corresponding to a given contour in
    an array
    """
    
    # gets a vector of sorted cumulative distributions
    shape = arr.shape
    vec = arr.flatten()
    svec = np.sort(vec)
    
    # normalised cumulative distribution
    csvec = np.cumsum(svec)
    csvec /= csvec[-1]
    
    # determines threshold
    if isinstance(levels,float):
        levels = [levels]
    cuts=[]
    for i,l in enumerate(levels):
        j = np.where(csvec < 1.-l)[0][-1]
        cut = svec[j] # j is the last one lower than it
        cuts.append(cut) # i.e. this is highest exclusion
    return cuts

def cdf(x,dm,cs):
    """
    Function to return a cdf given dm and cs via linear interpolation
    """
    
    nx = np.array(x)
    #y=np.zeros(nx.size)
    #y[x <= dm[0]]=0.
    #y[x >= dm[-1])=1.
    
    ddm = dm[1]-dm[0]
    ix1 = (x/ddm).astype('int')
    ix2 = ix1+1
    
    kx2 = x/ddm-ix1
    kx1 = 1.-kx2
    c = cs[ix1]*kx1 + cs[ix2]*kx2
    return c


def nu_cdf(xs,dm,cs):
    """
    Function to return a cdf given dm and cs via linear interpolation
    Non-uniform dm data
    """
    nx = np.array(xs)
    #y=np.zeros(nx.size)
    #y[x <= dm[0]]=0.
    #y[x >= dm[-1])=1.
    vals = np.zeros([nx.size])
    for i,x in enumerate(xs):
        ix1=np.where(dm < x)[0][-1]
        ix2 = ix1+1
    #ix1 = (x/ddm).astype('int')
    #ix2 = ix1+1
        ddm = dm[ix2]-dm[ix1]
        kx2 = (x-dm[ix1])/ddm
        kx1 = 1.-kx2
        vals[i] = cs[ix1]*kx1 + cs[ix2]*kx2
    return vals

def get_extra_chime_reps(DMhalo=50.):
    """
    
    """
    infile='CHIME_FRBs/new_repeaters.dat'
    NFRB=25
    decs = np.zeros([NFRB])
    dmegs = np.zeros([NFRB])
    with open(infile) as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            words=line.split()
            dec=float(words[0])
            DM=float(words[1])
            DMG=float(words[2])
            DMEG=DM-DMG-DMhalo
            dmegs[i]=DMEG
            decs[i]=dec
            if i==24:
                break
    return decs,dmegs

def get_chime_data(DMhalo=50,snrcut=None):
    """
    Imports data from CHIME catalog 1
    
    Returns select info from the catalog, for further processing
    
    DMhalo is the assumed DM halo value
    snrcut, if not None, removes everything with snr less than that value.
    
    """
    chimedir = 'CHIME_FRBs/'
    infile = chimedir+'chimefrbcat1.csv'
    
    idec=5
    idm=29
    idmeg=9
    iname=0
    irep=2
    iwidth=32
    isnr=10
    
    # hard-coded number of FRBs
    NFRB=536
    decs=np.zeros([NFRB])
    dms=np.zeros([NFRB])
    dmegs=np.zeros([NFRB])
    dmgs=np.zeros([NFRB])
    snrs=np.zeros([NFRB])
    widths=np.zeros([NFRB])
    names=[]
    reps=np.zeros([NFRB])
    
    # holds repeater info
    rnames=[]
    ireps=[]
    nreps=[]
    badcount=0
    
    with open(infile) as f:
        lines = f.readlines()
        count=0
        for i,line in enumerate(lines):
            if i==0:
                continue
            words=line.split(',')
            # seems to indicate new bursts have been added
            #if words[5][:2]=="RA":
            #    badcount += 1
                #print("BAD : ",badcount)
                #continue
            snr=float(words[isnr])
            if snrcut is not None:
                if snr < snrcut:
                    continue
            decs[count]=float(words[idec])
            dms[count]=float(words[idm])
            dmegs[count]=float(words[idmeg])
            names.append(words[iname])
            snrs[count]=float(words[isnr])
            # guards against upper limits
            if words[iwidth][0]=='<':
                widths[count]=0.
            else:
                widths[count]=float(words[iwidth])*1e3 #in ms
            dmgs[count] = dms[count]-dmegs[count]
            rep=words[irep]
            
            if rep=='-9999' or rep=='':
                # indicates it is not a repeat burst
                reps[count]=0
            else:
                # it is a repeat burst
                reps[count]=1
                # is it the first repeat burst?
                if rep in rnames:
                    ir = rnames.index(rep)
                    nreps[ir] += 1
                else:
                    rnames.append(rep)
                    ireps.append(count)
                    nreps.append(1)
            count += 1
    # excises the two "repeating" FRBs with nreps = 1 from the list
    real_ireps=[]
    real_nreps=[]
    for i,ir in enumerate(ireps):
        if nreps[i] > 1:
            real_ireps.append(ir)
            real_nreps.append(nreps[i])
        else:
            #print("Setting ",names[ireps[i]]," to zero")
            reps[ireps[i]]=0.
    
    dmegs -= DMhalo
    return names,decs,dms,dmegs,snrs,reps,real_ireps,widths,real_nreps

def get_chime_dec_dm_data(DMhalo=50,newdata=False, sort=False, donreps=False):
    """
    return a list of signle and repeat CHIME FRB dms and decs
    """
    names,decs,dms,dmegs,snrs,reps,ireps,widths,nreps = get_chime_data(DMhalo=DMhalo)
    
    # sorts by declination
    singles = np.where(reps==0)
    sdecs = decs[singles]
    rdecs = decs[ireps]
    sdmegs = dmegs[singles]
    rdmegs = dmegs[ireps]
    
    if newdata == 'only':
        rdecs,rdmegs = get_extra_chime_reps(DMhalo=DMhalo)
    elif newdata:
        Xdec,Xdmeg = get_extra_chime_reps(DMhalo=DMhalo)
        rdecs = np.concatenate((rdecs,Xdec))
        rdmegs = np.concatenate((rdmegs,Xdmeg))
    
    if sort:
        sdmegs = np.sort(sdmegs)
        rdmegs = np.sort(rdmegs)
        sdecs = np.sort(sdecs)
        rdecs = np.sort(rdecs)
    
    if donreps:
        nreps = nreps[ireps]
        return sdmegs,rdmegs,sdecs,rdecs,nreps
    else:
        return sdmegs,rdmegs,sdecs,rdecs
    
def get_chime_rs_dec_histograms(DMhalo=50,newdata=False):
    """
    Returns normalsied cumulative histograms of CHIME singles and repeaters
    """    
    
    names,decs,dms,dmegs,snrs,reps,ireps,widths,nreps = get_chime_data(DMhalo=DMhalo)
    
    # sorts by declination
    singles = np.where(reps==0)[0]
    sdecs = decs[singles]
    rdecs = decs[ireps]
    
    if newdata == 'only':
        rdecs,Xdmeg = get_extra_chime_reps(DMhalo=DMhalo)
        #rdecs = np.concatenate((rdecs,Xdec))
    elif newdata:
        Xdec,Xdmeg = get_extra_chime_reps(DMhalo=DMhalo)
        rdecs = np.concatenate((rdecs,Xdec))
    
    sdecs = np.sort(sdecs)
    rdecs = np.sort(rdecs)
    
    ns = sdecs.size
    nr = rdecs.size
    
    #creates cumulative hist
    sxvals = np.zeros([ns*2+2])
    rxvals = np.zeros([nr*2+2])
    syvals = np.zeros([ns*2+2])
    ryvals = np.zeros([nr*2+2])
    for i,dec in enumerate(sdecs):
        sxvals[i*2+1]=dec
        sxvals[i*2+2]=dec
        syvals[i*2+1]=i/ns
        syvals[i*2+2]=(i+1)/ns
    syvals[-1]=1.
    sxvals[-1]=90
    
    for i,dec in enumerate(rdecs):
        rxvals[i*2+1]=dec
        rxvals[i*2+2]=dec
        ryvals[i*2+1]=i/nr
        ryvals[i*2+2]=(i+1)/nr
    ryvals[-1]=1.
    rxvals[-1]=90
    
    
    return sxvals,syvals,rxvals,ryvals

def get_chime_rs_dm_histograms(DMhalo=50,newdata=False):
    """
    Returns normalsied cumulative histograms of CHIME singles and repeaters
    """    
    
    names,decs,dms,dmegs,snrs,reps,ireps,widths,nreps = get_chime_data(DMhalo=DMhalo)
    
    # sorts by declination
    singles = np.where(reps==0)
    sdms = dmegs[singles]
    #reps = np.where(reps>0)
    rdms = dmegs[ireps]
    
    if newdata == 'only':
        Xdec,rdms = get_extra_chime_reps(DMhalo=DMhalo)
    elif newdata:
        Xdec,Xdmeg = get_extra_chime_reps(DMhalo=DMhalo)
        rdms = np.concatenate((rdms,Xdmeg))
    
    sdms = np.sort(sdms)
    rdms = np.sort(rdms)
    
    ns = sdms.size
    nr = rdms.size
    
    #creates cumulative hist
    sxvals = np.zeros([ns*2+2])
    rxvals = np.zeros([nr*2+2])
    syvals = np.zeros([ns*2+2])
    ryvals = np.zeros([nr*2+2])
    for i,dm in enumerate(sdms):
        sxvals[i*2+1]=dm
        sxvals[i*2+2]=dm
        syvals[i*2+1]=i/ns
        syvals[i*2+2]=(i+1)/ns
    syvals[-1]=1.
    sxvals[-1]=5000
    
    for i,dm in enumerate(rdms):
        rxvals[i*2+1]=dm
        rxvals[i*2+2]=dm
        ryvals[i*2+1]=i/nr
        ryvals[i*2+2]=(i+1)/nr
    ryvals[-1]=1.
    rxvals[-1]=5000
    
    return sxvals,syvals,rxvals,ryvals




def plot_2darr(arr,labels,savename,ranges,rlabels,clabel=None,crange=None,\
    conts=None,Nconts=None,RMlim=None,scatter=None,Allowed=False):
    """
    does 2D plot
    
    array is the 2D array to plot
    labels are the x and y axis labels [ylabel,xlabel]
    Here, savename is the output file
    Ranges are the [xvals,yvals]
    Rlabels are [xtics,ytics]
    
    """
    from matplotlib import pyplot as plt
    
    ratio=np.abs((ranges[0][1]-ranges[0][0])/(ranges[0][2]-ranges[0][1]))
    if ratio > 1.01 or ratio < 0.99:
        log0=True
    else:
        log0=False
    
    ratio=np.abs((ranges[1][1]-ranges[1][0])/(ranges[1][2]-ranges[1][1]))
    if ratio > 1.01 or ratio < 0.99:
        log1=True
    else:
        log1=False
    
    dr1 = ranges[1][1]-ranges[1][0]
    dr0 = ranges[0][1]-ranges[0][0]
    
    aspect = (ranges[0].size/ranges[1].size)
    
    extent = [ranges[1][0]-dr1/2., ranges[1][-1]+dr1/2.,\
            ranges[0][0]-dr0/2.,ranges[0][-1]+dr0/2.]
    
    im = plt.imshow(arr,origin='lower',aspect=aspect,extent=extent)
    ax=plt.gca()
    
    # sets x and y ticks to bin centres
    ticks = rlabels[1].astype('str')
    for i,tic in enumerate(ticks):
        ticks[i]=tic[:5]
    ax.set_xticks(ranges[1][1::2])
    ax.set_xticklabels(ticks[1::2])
    plt.xticks(rotation = 90) 
    ticks = rlabels[0].astype('str')
    for i,tic in enumerate(ticks):
        ticks[i]=str(rlabels[0][i])[0:4]
    ax.set_yticks(ranges[0][::4])
    ax.set_yticklabels(ticks[::4])
    
    plt.xlabel(labels[1])
    plt.ylabel(labels[0])
    
    #cax = fig.add_axes([ax.get_position().x1+0.03,ax.get_position().y0,0.02,ax.get_position().height])
    #cbar = plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    cbar = plt.colorbar(shrink=0.55)
    if clabel is not None:
        cbar.set_label(clabel)
    if crange is not None:
        if len(crange) == 2:
            plt.clim(crange[0],crange[1])
        else:
            themax=np.nanmax(arr)
            plt.clim(crange+themax,themax)
    
    
    if conts is not None:
        if len(conts) == 2:
            
            ax = plt.gca()
            cs=ax.contour(conts[0],levels=[conts[1]],origin='lower',colors="black",\
                linestyles=[':'],linewidths=[3],extent=extent)
        else:
            colors=["red","white","black"]
            styles=[':','-.','--','-']
            for k,cont in enumerate(conts[0]):
                print("Doing multiple conts")
                cs=ax.contour(cont[0],levels=[cont[1]],origin='lower',colors=colors[k],\
                    linestyles=styles[k],linewidths=[3],extent=extent)
    if Nconts is not None:
        ax = plt.gca()
        cs=ax.contour(Nconts[0],levels=[Nconts[1]],origin='lower',colors="orange",\
            linestyles=['-.'],linewidths=[3],extent=extent)
    
    if Allowed:
        plt.text(1,-2.5,'Allowed')
    
    if RMlim is not None:
        plt.plot([RMlim,RMlim],[extent[2],extent[3]],linestyle='--',color='white',linewidth=3)
    
    if scatter is not None:
        sx=scatter[0]
        sy=scatter[1]
        sm=scatter[2]
        for i, m in enumerate(sm):
            #ax.plot((i+1)*[i,i+1],marker=m,lw=0)
            #plt.plot(sx[i],sy[i],marker=m,color='red',linestyle="",markersize=12)
            plt.text(sx[i],sy[i],m,color='red',fontsize=16,ha='left',va='center',
            fontweight='extra bold')
    
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()
