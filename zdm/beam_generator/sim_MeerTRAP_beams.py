"""
This program simulates a MeerTRAP Gaussian beams on the sky,
and superimposes formed beams with a fixed FWHM on top
of this

It assumes that all beams are perfect Gaussians.

It writes out beamfile information for use by the zdm code
"""


import numpy as np
from matplotlib import pyplot as plt


def main(plot=True,freq=1284,gsize=800,spacing=0.001):
    """
    
    Frequency:
        frequency in MHz
        
    gsize [int]: number of grid elements on which to calculate beam pattern
    
    spacing [float]: spacing in degrees between calculation points. Total
        area is (gsize * spacing)^2
        
    All files produced have pattern_sep_freq notation
    """
    
    ############# Make the primary beam ##########
    # defines grid in x,y - this will be waaaay too small!
    gsize = gsize
    xgrid,ygrid = make_grid(gsize,spacing)
    
    # loops through beams, filling the grid
    Diam = 13.5
    beamw = get_beamwidth(freq,Diam,const=1.09)
    primary_beam = grid_beam(xgrid,ygrid,0,0,beamw)
    
    #plt.figure()
    #plt.imshow(beams,origin='lower')
    #plt.show()
    
    
    ############# Get tied beam specifications ############
    
    # MeerTRAPforms 768 beams that span a triangular grid.
    # I assume that the overlap at the 25% power point corresponds
    # to the length of each triangular edge of the grid.
    # The total quoted area of the survey in
    # https://doi.org/10.1093/mnras/stad2041
    # Table 3 of 1448 deg^2 hr over 317.5 days implies an
    # instantaneous area of 1448/24/317.5 = 0.19 square degrees
    # This then corresponds to 0.19 / 768 = 0.0002474 deg2 per beam
    # In a triangular grid, the total area governed by a vertex is a 
    # trapezoid with interiorangles of 60 and 120 degrees.
    # The area is related to the side length as A = l^2 sin 60
    # We then solve for l a l = (A/sin 60 )^0.5 = (0.0002474 * 2/3^0.5)^0.5
    # = 0.0169 separation. Hence, this is the full width corresponding to
    # 25% separation.
    # When does a Gausian get to 25% separation?
    x=np.linspace(0.,2.,2001)
    y=np.exp(-0.5*x**2)
    i25 = np.where(y < 0.25)[0][0]
    x25 = x[i25]
    # this corresponds to the value of sigma at 25% power
    formed_sigma = 0.0169 / x25
    #print("The sigma for a MeerTRAP tied beam is ",formed_sigma," in degrees")
    
    formed_x,formed_y = gen_triangular_grid(Nx=32,Ny=24,sep=0.0169*2.)
    
    # begins by setting envelope on tied beam to zero
    tied_envelope = np.zeros([gsize,gsize]).flatten()
    
    for i in np.arange(formed_x.size):
        beams = grid_beam(xgrid,ygrid,formed_x[i],formed_y[i],formed_sigma).flatten()
        bigger = np.where(beams > tied_envelope)
        tied_envelope[bigger] = beams[bigger]
    
    envelope = tied_envelope.reshape([gsize,gsize]) * primary_beam
    #plt.figure()
    #plt.imshow(envelope,origin='lower')
    #plt.savefig("MeerTRAP_beam.pdf")
    #plt.close()
    plot_envelope(envelope,xgrid[0,:],outfile="MeerTRAP_beam.pdf")
    
    # makes a beam histogram
    bins = np.logspace(-4,0,401)
    
    weights = calculate_weights(xgrid,ygrid)
    
    ########### Makes a histogram of the beamshape ###########
    
    bins = np.logspace(-4,0,401)
    h,b=np.histogram(envelope.flatten(),bins,weights=weights.flatten())
    
    plot_b = np.logspace(-3.995,-0.005,400)
    
    np.save('MeerTRAP_coherent_log_hist.npy',h)
    np.save('MeerTRAP_coherent_log_bins.npy',b)
    plt.figure()
    plt.plot(plot_b,h)
    plt.xlabel("$B$")
    plt.ylabel("$\\Omega(B)$")
    plt.savefig("MeerTRAP_OmegaB.pdf")
    plt.close()

def gen_triangular_grid(Nx=32,Ny=24,sep=0.01):
    """
    Generates beam centres
    
    pattern [string]: square or closepack
    
    sep [float, deg]: beam separation
    
    """
    # triangular grid
    x = np.zeros([Nx,Ny])
    y = np.zeros([Nx,Ny])
    for iy in np.arange(Ny):
        if iy % 2 == 0:
            x0 = 0.
        else:
            x0 = 0.5 * sep
        x[iy,:] = np.arange(Ny) * sep + x0
        y[iy,:] = iy*sep*3**0.5/2.
    # re-centres the array
    x -= np.sum(x)/(Nx*Ny)
    y -= np.sum(y)/(Nx*Ny)
    
    x = x.flatten()
    y = y.flatten()
    
    return x,y
        
def get_beamwidth(freq,D,const=1.09):
    """
    Gets beamwidth with frequency in Hz
    returns Gaussian sigma in degrees
    
    freq [float]: frequency in HHz
    """
    c_light = 299792458.
    HPBW=const*(c_light/(freq))/D # from RACS
    sigma=(HPBW/2.)*(2*np.log(2))**-0.5
    deg_sigma = sigma * 180./np.pi
    return deg_sigma

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
    

def plot_envelope(env,coords,outfile):
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
    plt.savefig(outfile)
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


main()
