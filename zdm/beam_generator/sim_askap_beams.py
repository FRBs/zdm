"""
This program simulates 36 Gaussian beams on the sky,
nominally from ASKAP, and produces their beam pattern.

It assumes that all beams are perfect Gaussians, 
and includes a tapering of beam performance away from
boresight according to Hotan et al (an alternative is
provided in "askap_beam_amps", but not used)

It writes out beamfile information for use by the zdm code
"""


import numpy as np
from matplotlib import pyplot as plt


def main(plot=True,pattern="square",pitch=0.9,freq=1400,gsize=800,spacing=0.02):
    """
    
    Pattern:
        Square: square array
        Closepack: triangular grid
    
    Sep:
        Beam separation in degrees
    
    Frequency:
        frequency in MHz
        
    gsize [int]: number of grid elements on which to calculate beam pattern
    
    spacing [float]: spacing in degrees between calculation points. Total
        area is (gsize * spacing)^2
        
    All files produced have pattern_sep_freq notation
    """
    
    
    # defines grid in x,y - this will be waaaay too small!
    gsize = gsize
    xgrid,ygrid = make_grid(gsize,spacing)
    
    ###### defines antenna pattern ########
    beamx, beamy, beamw, beama = gen_askap_beams(pattern,pitch,freq*1e6)
    
    Nbeams = beamx.size
    
    # loops through beams, filling the grid
    beams = np.zeros([Nbeams,gsize,gsize])
    for i in np.arange(Nbeams):
        beams[i,:,:] = beama[i] * grid_beam(xgrid,ygrid,beamx[i],beamy[i],beamw[i])
    
    envelope = np.max(beams,axis=0)
    
    if plot:
        plot_envelope(envelope,xgrid[0,:])
    
    
    # makes a beam histogram
    bins = np.logspace(-4,0,401)
    
    weights = calculate_weights(xgrid,ygrid)
    
    # calculates single value
    contributions = weights * envelope**1.5
    total = np.sum(contributions) * 100
    
    tsys = get_tsys(freq)
    
    adjusted = total * (80./tsys)**1.5
    
    print(pattern,freq/1e6,pitch,total,adjusted)
    
    
    ########### Makes a histogram of the beamshape ###########
    tsys0 = get_tsys(1272.5) # normalising frequency
    envelope *= tsys0/tsys # adjusts so B=1 is same sensitivity at 1272.5 GHz
    # makes a beam histogram
    bins = np.logspace(-4,0,401)
    h,b=np.histogram(envelope.flatten(),bins,weights=weights.flatten())
    
    np.save('ASKAP/'+pattern+'_'+str(pitch)+'_'+str(freq)+'.npy',h)
    np.save('ASKAP/log_bins.npy',b)
    
    if False:
        db = bins[1]/bins[0]
        bcs = b[:-1]*db**0.5
        plt.figure()
        plt.plot(bcs,h)
        lat50b = np.load('../data/BeamData/lat50_log_bins.npy')
        lat50h = np.load('../data/BeamData/lat50_log_hist.npy')
        plt.plot(10**lat50b,lat50h*1000/400*35/40)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    #h,b=np.histogram(envelope.flatten(),bins,weights=weights.flatten())
    
    #if plot:
    #    plot_histogram(h,b)

def get_tsys(freq,plot=False):
    """
    reads in interpolated data from Hotan et al
    
    Freq: frequency in MHz
    """   
    
    data = np.loadtxt("askap_tsys.dat")
    x = data[:,0]
    y = data[:,1]
    deg = 8
    coeffs = np.polyfit(x,y,deg)
    
    
    if plot:
        xvals = np.linspace(700,1800,201)
        yvals = np.polyval(coeffs,xvals)
        plt.figure()
        plt.plot(xvals,yvals,color='grey')
        plt.plot(x,y,linestyle="",color='red',marker='o')
        plt.ylim(0,130)
        plt.savefig("askap_tsys.pdf")
        plt.close()
    
    Tsys = np.polyval(coeffs,freq)
    return Tsys
    
def gen_askap_beams(pattern,sep,freq):
    """
    Generates beam centres
    
    pattern [string]: square or closepack
    
    sep [float, deg]: beam separation
    
    freq [float]: frequency in Hz
    """
    
    if pattern == "square":
        # square grid
        x=np.linspace(0,5.*sep,6)
        x=np.repeat(x,6)
        x = x.reshape([6,6])
        y=np.copy(x)
        y=y.T
        x -= sep*2.5
        y -= sep*2.5
    elif pattern == "closepack":
        # triangular grid
        x = np.zeros([6,6])
        y = np.zeros([6,6])
        for iy in np.arange(6):
            if iy % 2 == 0:
                x0 = 0.
            else:
                x0 = 0.5 * sep
            x[iy,:] = np.arange(6) * sep + x0
            y[iy,:] = iy*sep*3**0.5/2.
        x -= np.sum(x)/36.
        y -= np.sum(y)/36.
    
    x = x.flatten()
    y = y.flatten()
    
    # implements function of distance
    dists = (x**2 + y**2)**0.5
    
    # from Hotan et al
    avals = get_avals(dists)
    
    
    # function of frequency only
    # Gaussian sigma
    diam = 12.
    width = get_beamwidth(freq)
    beamw = np.full([36],width)
    
    return x, y, beamw, avals
        
def get_beamwidth(freq):
    """
    Gets beamwidth with frequency in Hz
    returns Gaussian sigma in degrees
    
    freq [float]: frequency in HHz
    """
    D = 12.
    c_light = 299792458.
    HPBW=1.09*(c_light/(freq))/D # from RACS
    sigma=(HPBW/2.)*(2*np.log(2))**-0.5
    deg_sigma = sigma * 180./np.pi
    return deg_sigma

def askap_beam_amps(theta):
    """
    Equation 7 from lat50 paper of James et al.
    
    """
    
    theta_off = 0.8
    sigma_theta = 3.47
    
    # gets locations where amplitude is unity
    shape = theta.shape
    theta = theta.flatten()
    small = np.where(theta < theta_off)
    
    # sets entire array to Gaussian
    amps = np.exp(-0.5 * ((theta-theta_off)/sigma_theta)**2)
    # resets small values to unity
    amps[small] = 1.
    
    amps = amps.reshape(shape)
    return amps

def calculate_weights(xvals,yvals):
    """
    calculates histogram weights, accounting for sine
    """
    # converts to steradians
    xvals *= np.pi/180.
    yvals *= np.pi/180.
    
    # calculates distances from the origin
    dists = (xvals**2 + yvals**2)**0.5
    
    cosfactor = np.cos(dists) #deformation factor due to curvature of sky
    base = (xvals[0,1]-xvals[0,0])**2 # pixel size
    weights = cosfactor * base
    return weights

def plot_histogram(h,b):
    """
    makes a simple plot of Omega(b)
    """
    
    from matplotlib import pyplot as plt
    # note that b will have one more element
    bcs = b[:-1] + (b[1]-b[0])/2.
    plt.figure()
    plt.plot(bcs,h)
    plt.xlabel('B')
    plt.ylabel('$\\Omega(B)$')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('omega_b.pdf')
    plt.close()
    

def plot_envelope(env,coords):
    """
    Makes simple plot of beamshape
    """
    
    from matplotlib import pyplot as plt
    dc = coords[1]-coords[0]
    themin = coords[0]-dc/2.
    themax = coords[-1] + dc/2.
    
    plt.figure()
    plt.imshow(env,extent=(themin,themax,themin,themax),origin='lower')
    plt.xlabel('x [deg]')
    plt.ylabel('y [deg]')
    plt.tight_layout()
    plt.savefig('askap_beam_pattern.pdf')
    plt.close()
   
def make_grid(Npoints,ddeg):
    """
    Generates a grid in which to store beam values.
    
    Inputs:
        Npoints [int]: number of points in each dimension.
            Final array will be Npoints X Npoints
        ddeg [float]: Degree increment between points
    """
    # generates xpoints 
    extent = (Npoints-1.)*ddeg/2.
    degrees = np.linspace(extent,-extent,Npoints)
    ypoints = np.repeat(degrees,Npoints)
    ypoints = ypoints.reshape([Npoints,Npoints])
    
    degrees = np.linspace(-extent,extent,Npoints)
    xpoints = np.repeat(degrees,Npoints)
    xpoints = xpoints.reshape([Npoints,Npoints])
    xpoints = xpoints.T
    
    return xpoints,ypoints

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


def get_avals(thetas):
    """
    Values extrated from Hotan et al.
    """
    
    data = np.loadtxt("avals.dat")
    x = data[:,0]
    y = data[:,1]
    avals = np.interp(thetas,x,y)
    return avals
    
def grid_beam(xvals,yvals,beamx,beamy,beamw):
    """
    Fills grid with sensitivities of a beam
    
    Inputs:
        xvals [np 2d array]: x coordinates (degrees)
            of grid centres (not bin edges)
        yvals [np 2d array]: y coordinates (degrees)
            of grid centres (not bin edges)
        beamx: coordinates of beam in x-direction
        beamy: coordinates of beam in y-direction
        beamw: beamwidth in degrees
    
    Returns:
        grid (np 2d array), filled with beam values
    """
    
    
    offsets = ((xvals - beamx)**2 + (yvals - beamy)**2 )**0.5
    values = Gauss(offsets,beamw)
    return values


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

# original
# this is a hard-coded list of *every* combination of frequency and pitch angle
# used by ASKAP to date
flist = [ 819.,  832.,  864.,  920.,  936., 950.,  970., 980., 990., 1032., 1100.,
            1112., 1266., 1272., 1297., 1320., 1407., 1418., 1632., 1641., 1656.]
plist = [0.72, 0.75, 0.84, 0.9,  1.05, 1.1 ]
array = ["closepack","square"]
for freq in flist:
    for p in plist:
        for config in array:
            main(freq=freq,pattern = config, pitch=p)
