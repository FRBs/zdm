"""
This program simulates a single Gaussian beam, and
assumes N formed beams from M identical antennas are formed
"""

import numpy as np

def main():
    """
    main program
    """
    
    Freq,FWHM,Nants,axs,ays,Nbeams,bzs,bas,pz,pa = read_simple_beamfile('DSA110_beamfile.dat')
    
    Nbeams = 1
    bzs = bzs[0:1]
    bas = bas[0:1]
    
    gridz,grida,weights = create_grid(pz,pa,300,6.)
    
    envelope = sim_formed_beams(axs,ays,Freq,bzs,bas,gridz,grida)
    
    primary = apply_primary_beamshape(envelope,pz,pa,FWHM,gridz,grida)
    
    combined = primary * envelope
    
    combined /= np.max(combined)
    
    bins = np.logspace(-4,0,401)
    h,b=np.histogram(combined.flatten(),bins,weights=weights.flatten())
    
    np.save('DSA_log_hist.npy',h)
    np.save('DSA_log_bins.npy',b)
    
    plot=True
    if plot:
        
        from matplotlib import pyplot as plt
        # note that b will have one more element
        bcs = b[:-1] + (b[1]-b[0])/2.
        plt.figure()
        plt.plot(bcs,h)
        plt.xlabel('B')
        plt.ylabel('$\\Omega(B)$')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('dsa_omega_b.pdf')
        plt.close()
        
        plt.figure()
        plt.imshow(envelope,extent=(grida[0,0],grida[0,-1],gridz[0,0],gridz[-1,0]),origin='lower')
        plt.xlabel('azimuth angle [deg]')
        plt.ylabel('zenith [deg]')
        plt.tight_layout()
        plt.savefig('dsa_tied_beam_pattern.pdf')
        plt.close()
    
        
        plt.figure()
        plt.imshow(primary,extent=(grida[0,0],grida[0,-1],gridz[0,0],gridz[-1,0]),origin='lower')
        plt.xlabel('azimuth angle [deg]')
        plt.ylabel('zenith [deg]')
        plt.tight_layout()
        plt.savefig('dsa_primary_beam_pattern.pdf')
        plt.close()
        
        plt.figure()
        plt.imshow(combined,extent=(grida[0,0],grida[0,-1],gridz[0,0],gridz[-1,0]),origin='lower')
        plt.xlabel('azimuth angle [deg]')
        plt.ylabel('zenith [deg]')
        plt.tight_layout()
        plt.savefig('dsa_beam_pattern.pdf')
        plt.close()

def apply_primary_beamshape(envelope,pz,pa,FWHM,gridz,grida):
    """
    Applies primary beamshape correction to formed beams
    """
    
    # calculates distances from grid points to primary
    # beam centre
    px,py,pz = get_xyz(pz,pa)
    
    gx,gy,gz = get_xyz(gridz,grida)
    
    cosines = px*gx + py*gy + pz*gz
    angles = np.arccos(cosines)
    
    sigma = (np.pi/180. * FWHM/2.)*(2*np.log(2))**-0.5
    
    primary = Gauss(angles,sigma)
    return primary

def Gauss(r,sigma):
    """
    Simple Gaussian function, normalised to a peak
    amplitude of unity
    
    Inputs:
        r [float] is a radial offset from centre
        sigma [float] is a std deviation in the same units
    
    Return:
        returns value of Gaussian at r points
    """
    
    return np.exp(-0.5 * (r/sigma)**2)

  
def get_xyz(z,a):
    """
    returns x,y,z coordinates of given zenith, azimuth position
    """
    zr = z * np.pi/180.
    ar = a * np.pi/180.
    
    cz = np.cos(zr)
    sz = np.sin(zr)
    ca = np.cos(ar)
    sa = np.sin(ar)
    
    x = sz * sa
    y = sz * ca
    z = cz
    return x,y,z

def sim_formed_beams(axs,ays,nu,bz,ba,gz,ga):
    """
    This program simulated the formed beamshape
    given a grid of antenna positions, a frequency,
    and a pointing direction (zenith, azimuth)
    
    Inputs:
        axs: numpy array of antenna x positions (East, m)
        ays: numpy array of antenna y positions (North, m)
        nu: central frequency (Hz)
        zenith: pointing zenith angle (deg)
        azimuth: pointing azimuth, East from North (deg)
        bz: beam phase centre zenith (deg)
        ba: beam phase centre azimuth (deg)
        gz: grid of zenith angles (deg)
        ga: grid of azimuth angles (deg)
    """
    
    # get number of beams. Allows for there to be a single beam or multiple
    bz = np.array(bz)
    ba = np.array(ba)
    Nbeams = bz.size
    
    # gets shape of sky grid, gz.
    # NOTE: shape of 'gz' and 'ga' arrays does NOT need to be
    # in simple deltaz, deltaa format - but it helps to label them as such
    # could just be in x-y format
    Nz,Na = gz.shape
    
    wavelength = 2.99792458e8/nu
    
    # apparent baselines in units of lambda
    #xbaseline = axs * xcosine / wavelength
    #ybaseline = ays * ycosine / wavelength
    
    # first: calculate direction cosines in x and y directions for
    # each grid point
    
    gxcosines,gycosines = get_cosines(gz,ga)
    
    # these are the distances incurred for each m in the x and y directions respectively
    # now multiply by x and y distances in units of phase factors
    phase_factor = 2. * np.pi / wavelength
    
    #calculate delay corresponding to phase centre
    bxcosines,bycosines = get_cosines(bz,ba)
    
    # set up grid for all beams
    formed_beams = np.zeros([Nbeams,Nz,Na],dtype='complex')
    
    # loop over antennas, calculating delay for each
    for iant,ax in enumerate(axs):
        dax = ax * phase_factor
        day = ays[iant] * phase_factor
        
        # gives delays to each position for this antenna
        # units are phase
        delays = dax * gxcosines + day * gycosines
        
        
        # gives base delay for each beam
        # units also phase
        delay0s = dax * bxcosines + day * bycosines
        
        # loops over beam
        for ibeam,delay0 in enumerate(delay0s):
            # delays to each position for this beam
            bdelays = delays - delay0
            formed_beams[ibeam,:,:] += np.exp(bdelays * 1j)
            
    
    mags = np.abs(formed_beams)
    envelope = np.max(mags,axis=0)
    
    return envelope
    
    
 
def create_grid(zen,az,Ngrid,extent):
    """
    creates a grid of sky positions,
    along with steradians of each cell
    
    The grid is centred on position zen,az
    and is equally spaced in zenith and azimuth coordinates
    
    extent is the total degree extent in each direction about the centre
    
    Returns the grid zenith, grid azimuth,
    and solid angle values of the grid
    """ 
    dgrid = extent / (Ngrid-1.)
    
    zmin = zen - (Ngrid-1.)/2.*dgrid
    zmax = zmin + (Ngrid-1.)*dgrid
    zvals = np.linspace(zmin,zmax,Ngrid)
    
    # could scale these by zenith angle perhaps?
    amin = az - (Ngrid-1.)/2.*dgrid
    amax = amin + (Ngrid-1.)*dgrid
    avals = np.linspace(amin,amax,Ngrid)
    
    apoints = np.repeat(avals,Ngrid)
    apoints = apoints.reshape([Ngrid,Ngrid])
    apoints = apoints.T
    
    zpoints = np.repeat(zvals,Ngrid)
    zpoints = zpoints.reshape([Ngrid,Ngrid])
    
    # zenith angle increment
    dz = dgrid * np.pi/180.
    # aziumuth angle increment
    da = dgrid * np.pi/180. * np.sin(zvals*np.pi/180.)
    wgt = da * dz
    
    weights = np.repeat(wgt,Ngrid)
    weights = weights.reshape([Ngrid,Ngrid])
    
    return zpoints,apoints,weights
    
def read_simple_beamfile(inputfile):
    """
    Reads a file containing information on the telescope
    """
    with open(inputfile) as infile:
        lines = infile.readlines()
        
        Freq=float(lines[0].split()[0])
        FWHM=float(lines[1].split()[0])
        pz=float(lines[2].split()[0])
        pa=float(lines[2].split()[1])
        
        Nants = int(lines[3].split()[0])
        ax0 = float(lines[4].split()[0])
        ay0 = float(lines[4].split()[1])
        ax1 = float(lines[5].split()[0])
        ay1 = float(lines[5].split()[1])
        
        Nbeams = int(lines[6].split()[0])
        ba0 = float(lines[7].split()[0])
        ba1 = float(lines[8].split()[0])
    
    # initialises beams
    bas = np.linspace(ba0,ba1,Nbeams)
    bzs = np.full([Nbeams],pz)
    
    # initialises antenna locations
    axs = np.linspace(ax0,ax1,Nants)
    ays = np.linspace(ay0,ay1,Nants)
    
    return Freq,FWHM,Nants,axs,ays,Nbeams,bzs,bas,pz,pa

def read_complex_beamfile(inputfile):
    """
    Reads a file containing information on the telescope
    """
    with open(inputfile) as infile:
        lines = infile.readlines()
        for i,line in enumerate(lines):
            words = line.split()
            if i==0:
                # metadata
                Nants = int(words[0])
                Nbeams = int(words[1])
                Freq = float(words[2])
                
                axs = np.zeros([Nants])
                ays = np.zeros([Nants])
                
                bzs = np.zeros([Nbeams])
                bas = np.zeros([Nbeams])
                
            elif i==1:
                # prmary beam data
                pz = float(words[0]) # primary beam zenith angle [deg]
                pa = float(words[1]) # primary beam azimuth angle [deg]
                pw = float(words[2]) # primary beam width [deg]
                pm = int(words[3]) # method for specifying primary beam
                
                # converts to Gaussian sigma
                psigma = process_beam_width(pw,pm)
            elif i < Nants +2:
                # this is antenna data
                ax = float(words[0]) # primary beam zenith angle [deg]
                ay = float(words[1])
                axs[i-2] = ax
                ays[i-2] = ay
                
            else:
                # this is formed beam data
                bz = float(words[0])
                ba = float(words[1])
                bzs[i-2-Nants] = bz
                bas[i-2-Nants] = ba
                
def process_beam_width(value,method):
    """
    converts different measures of beam width
    to a standard Gaussian sigma
    
    Input:
        value [float]: value of beamwidth
        method [int]: definition of value
    
    Return:
        sigma [float]: std dev in degrees of Gaussian beam     
    """
    
    
    if method == 0:
        # actual sigma of beam
        sigma = value
    elif method == 1:
        # sigma is FWHM (aka HPBW) of beam:
        sigma = (value/2.)*(2*np.log(2))**-0.5
    else:
        raise ValueError("Invalid method for beam width method found")
    
    return sigma

    
def get_cosines(z,a):
    """
    Get direction cosines corresponding to the zentih angle z and azimuth angle a
    """       
    # grid zenith and azimuth angles in radians
    zr = z * np.pi/180.
    ar = a * np.pi/180.
    
    
    
    # cosines and sines of those angles
    cz = np.cos(zr)
    ca = np.cos(ar)
    sa = np.sin(ar)
    
    
    
    # direction cosines: azimuth is East from North,
    # zenith is from zenith, x is East and y is North
    xcosines = cz * sa
    ycosines = cz * ca
    
    return xcosines,ycosines


main()
