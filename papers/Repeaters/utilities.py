"""
Contains utility functions for reading CHIME data,
and other useful things
"""


import numpy as np

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

def get_chime_data(DMhalo=50):
    """
    Imports data from CHIME catalog 1
    
    Returns select info from the catalog, for further processing
    
    """
    chimedir = 'CHIME_FRBs/'
    infile = chimedir+'chimefrbcat1.csv'
    
    idec=6
    idm=18
    idmeg=26
    iname=0
    irep=2
    iwidth=42
    isnr=17
    
    NFRB=600
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
        count=-1
        for i,line in enumerate(lines):
            if count==-1:
                columns=line.split(',')
                #for ic,w in enumerate(columns):
                #    print(ic,w)
                count += 1
                continue
            words=line.split(',')
            # seems to indicate new bursts have been added
            #if words[5][:2]=="RA":
            #    badcount += 1
                #print("BAD : ",badcount)
                #continue
            decs[i-1]=float(words[idec])
            dms[i-1]=float(words[idm])
            dmegs[i-1]=float(words[idmeg])
            names.append(words[iname])
            snrs[i-1]=float(words[isnr])
            # guards against upper limits
            if words[iwidth][0]=='<':
                widths[i-1]=0.
            else:
                widths[i-1]=float(words[iwidth])*1e3 #in ms
            dmgs[i-1] = dms[i-1]-dmegs[i-1]
            rep=words[irep]
            
            
            if rep=='-9999':
                reps[i-1]=0
            else:
                reps[i-1]=1
                if rep in rnames:
                    ir = rnames.index(rep)
                    nreps[ir] += 1
                else:
                    rnames.append(rep)
                    ireps.append(i-1)
                    nreps.append(1)
            count += 1
    #print("Total of ",len(rnames)," repeating FRBs found")
    #print("Total of ",len(np.where(reps==0)[0])," once-off FRBs")
    dmegs -= DMhalo
    #for irep in ireps:
    #    print(irep,names[irep],dms[irep],dmegs[irep])
    print("Seen three bursts from 121102!!! Not in cat 1 though. Why?")
    return names,decs,dms,dmegs,snrs,reps,ireps

def get_chime_dec_dm_data(DMhalo=50,newdata=False, sort=False):
    """
    return a list of signle and repeat CHIME FRB dms and decs
    """
    names,decs,dms,dmegs,snrs,reps,ireps = get_chime_data(DMhalo=DMhalo)
    
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
    
    return sdmegs,rdmegs,sdecs,rdecs
    
def get_chime_rs_dec_histograms(DMhalo=50,newdata=False):
    """
    Returns normalsied cumulative histograms of CHIME singles and repeaters
    """    
    
    names,decs,dms,dmegs,snrs,reps,ireps = get_chime_data(DMhalo=DMhalo)
    
    
    # sorts by declination
    singles = np.where(reps==0)
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
    
    names,decs,dms,dmegs,snrs,reps,ireps = get_chime_data(DMhalo=DMhalo)
    
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
