"""
This program simulates N Gaussian beams on the sky,
takes their envelope, then converts them into
beamdata files for use with surveys in the zdm code

It is best utilised for multibeam beam patterns

E.g. use 'FAST.dat' for input
"""


import numpy as np


def main(plot=True):
    
    gsize = 700
    xgrid,ygrid = make_grid(gsize,0.001)
    
    beamx, beamy, beamw, beama = read_beamfile('FAST.dat')
    
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
    h,b=np.histogram(envelope.flatten(),bins,weights=weights.flatten())
    
    
    if plot:
        plot_histogram(h,b)
    
    np.save('FAST_log_hist.npy',h)
    np.save('FAST_log_bins.npy',b)

def calculate_weights(xvals,yvals):
    """
    calculates histogram weights, accounting for sine
    """
    # converts to steradians
    xvals *= np.pi/180.
    yvals *= np.pi/180.
    
    # calculates distances from the origin
    dists = (xvals**2 + yvals**2)**0.5
    
    sinefactor = np.sin(dists) #deformation factor due to curvature of sky
    base = (xvals[0,1]-xvals[0,0])**2 # pixel size
    weights = sinefactor * base
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
    plt.savefig('beam_pattern.pdf')
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
    
def read_beamfile(beamfile):
    """
    Reads an input beam file.
    The file must be in the correct format.
    """
    beamx = np.empty([0])
    beamy = np.empty([0])
    beamw = np.empty([0])
    beama = np.empty([0])
    
    with open(beamfile) as infile:
        
        for i,line in enumerate(infile):
            words = line.split()
            
            if len(line) <= 1:
                break
            
            if i==0:
                method = int(words[0]) # method of describing beams
                if method != 1 and method != 2:
                    print("invalid method")
                width = float(words[1])
                # If width == -1: read in from file
                
                # FWHM or sigma
                wmethod = int(words[2])
                
                wscale = float(words[3])
                # scale widths by value according to frequency
                
                unit = float(words[4]) # units - divide by this value
                
            else:
                if line[0] == '#':
                    continue
                if method == 1:
                    bx,by,bw,ba = gen_circular_beams(words,width)
                elif method == 2:
                    bx,by,bw,ba = read_beam_coordinates(words,width)
                
                beamx = np.append(beamx,bx)
                beamy = np.append(beamy,by)
                beamw = np.append(beamw,bw)
                beama = np.append(beama,ba)
    
    beamx = beamx / unit
    beamy = beamy / unit
    beamw = beamw * wscale / unit
    beamw = process_beam_width(beamw,wmethod)
    beama = beama / np.max(beama)
    
    return beamx, beamy, beamw, beama
    
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
    
    
    
def gen_circular_beams(words):
    """
    Generates beam centres for a circle of N
    beams a distance r from the overall centre
    and offset by some fractional spacing
    
    Inputs:
        words (line describing inputs)
    """
    Nbeams = int(words[0])
    r = float(words[1])
    offset = float(words[2])
    angles = (np.arange(Nbeams) + offset)* 2. * np.pi/ Nbeams
    
    beamx = r * np.cos(angles)
    beamy = r * np.sin(angles)
    return beamx,beamy

def read_beam_coordinates(words,width):
    """
    Just reads in the beam centre coordinates
    """
    
    beamx = float(words[0])
    beamy = float(words[1])
    
    if width == -1:
        if len(words) < 3:
            raise ValueError("Cannot find beam width")
        else:
            bw = float(words[2])
    else:
        bw = width
    
    if len(words) == 4:
        amp = float(words[3])
    else:
        amp == 1.
    
    return beamx,beamy,bw,amp


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
