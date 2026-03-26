"""
Set of functions used to fit scattering/width data
"""

import numpy as np


def function_wrapper(ifunc,args,xvals=None,logxmin=-6,logxmax=6,N=1000,cspline=None):
    """
    Wrapper for fitting functions, so they can be called by an ID, rather
    than by name.
    
    Args:
        ifunc [int]: integer ID of function
        args [list]: aguments to the function
        xvals [np.ndarray or None]: if provided, x values at which to
            evaluate the function. If not, function evaluated between
            logxmin, logxmax
        logxmin [float]: min value at which to evaluate the function in log10 space
        logxmax [flot]: max value at which to evaluate the function in log10 space
        N [int]: number of points at which to evaluate the function
        cspline [scipy spline object]: Spline object giving completeness
    
    Returns:
        xs: x values at which function is evaluated [only if xvals not provided]
        ys: value of function at xvals/xs 
    """
    
    # modifies the likelihood function by the completeness, and normalises
    xs = np.logspace(logxmin, logxmax, N)
    logxs = np.linspace(logxmin, logxmax, N)
    dlx = logxs[1]-logxs[0]
    
    a1 = args[0]
    if ifunc != 3:
        a2 = args[1]
    # get expected distribution and hence cdf
    A=1.
    if ifunc==0:
        ftaus = lognormal(xs,A,a1,a2)
    elif ifunc==1:
        ftaus = halflognormal(xs,A,a1,a2)
    elif ifunc == 2:
        ftaus = boxcar(xs,A,a1,a2)
    elif ifunc == 3:
        ftaus = logconstant(xs,A,a1)
    elif ifunc == 4:
        a3 = args[2]
        ftaus = smoothboxcar(xs,A,a1,a2,a3)
    elif ifunc == 5:
        a3 = args[2]
        ftaus = uppersmoothboxcar(xs,A,a1,a2,a3)
    elif ifunc == 6:
        a3 = args[2]
        ftaus = lowersmoothboxcar(xs,A,a1,a2,a3)
    elif ifunc==7:
        ftaus = halflognormal2(xs,A,a1,a2)
        
    if cspline is not None:
        ctaus = cspline(logxs)
        ftaus *= ctaus
    
    if xvals is not None:
        if ifunc==0:
            ptaus = lognormal(xvals,A,a1,a2)
        elif ifunc==1:
            ptaus = halflognormal(xvals,A,a1,a2)
        elif ifunc == 2:
            ptaus = boxcar(xvals,A,a1,a2)
        elif ifunc == 3:
            ptaus = logconstant(xvals,A,a1)
        elif ifunc == 4:
            a3 = args[2]
            ptaus = smoothboxcar(xvals,A,a1,a2,a3)
        elif ifunc == 5:
            a3 = args[2]
            ptaus = uppersmoothboxcar(xvals,A,a1,a2,a3)
        elif ifunc == 6:
            a3 = args[2]
            ptaus = lowersmoothboxcar(xvals,A,a1,a2,a3)
        elif ifunc==47:
            ptaus = halflognormal2(xvals,A,a1,a2)
    
    
    # checks normalisation in log-space.
    # This is such that \sum f(x) dlogx = 1
    norm = np.sum(ftaus)*dlx
    if xvals is None:
        ys = ftaus/norm
        return xs,ys
    else:
        ys = ptaus/norm
        return ys
    



def boxcar(x,*args):
    """
    Constant above some min value
    """
    logx = np.log10(x)
    A = args[0]
    xmin = args[1]
    xmax = args[2]
    y = np.zeros([logx.size])
    OK = np.where(logx > xmin)
    y[OK] = A
    notOK = np.where(logx > xmax)
    y[notOK] = 0.
    return y


def smoothboxcar(x,*args):
    """
    Boxcar, with smooth Gaussian edges on both sides
    """
    logx = np.log10(x)
    A = args[0]
    xmin = args[1]
    xmax = args[2]
    sigma = args[3]
    y = np.full([logx.size],A)
    large = np.where(logx > xmax)
    y[large] = lognormal(x[large],A,xmax,sigma)
    small = np.where(logx < xmin)
    y[small] = lognormal(x[small],A,xmin,sigma)
    
    #y[notOK] = 0.
    return y

def uppersmoothboxcar(x,*args):
    """
    Boxcar, with smooth Gaussian edge on upper side only
    """
    logx = np.log10(x)
    A = args[0]
    xmin = args[1]
    xmax = args[2]
    sigma = args[3]
    y = np.full([logx.size],A)
    large = np.where(logx > xmax)
    y[large] = lognormal(x[large],A,xmax,sigma)
    small = np.where(logx < xmin)
    y[small] = 0.
    
    #y[notOK] = 0.
    return y
    
def lowersmoothboxcar(x,*args):
    """
    Boxcar, with smooth Gaussian edge on upper side only
    """
    logx = np.log10(x)
    A = args[0]
    xmin = args[1]
    xmax = args[2]
    sigma = args[3]
    y = np.full([logx.size],A)
    large = np.where(logx > xmax)
    y[large] = 0.
    small = np.where(logx < xmin)
    y[small] = lognormal(x[small],A,xmin,sigma)
    
    #y[notOK] = 0.
    return y

def logconstant(x,*args):
    """
    Constant above some min value
    """
    logx = np.log10(x)
    A = args[0]
    xmin = args[1]
    y = np.zeros([logx.size])
    OK = np.where(logx > xmin)
    y[OK] = A
    return y

def halflognormal(x,*args):
    """
    Just like a lognormal, but only lower half
    """
    logx = np.log10(x)
    A=args[0]
    mu = args[1]
    sigma=args[2]
    
    y = A * np.exp(-0.5 * (logx-mu)**2/sigma**2)
    OK = np.where(logx > mu)
    y[OK] = A
    
    return y

def halflognormal2(x,*args):
    """
    Just like a lognormal, but only lower half
    """
    logx = np.log10(x)
    A=args[0]
    mu = args[1]
    sigma=args[2]
    
    y = A * np.exp(-0.5 * (logx-mu)**2/sigma**2)
    OK = np.where(logx < mu)
    y[OK] = A
    
    return y

def lognormal(x,*args):
    """
    lognormal, but just log10x
    """
    logx = np.log10(x)
    A=args[0]
    mu = args[1]
    sigma=args[2]
    y = A * np.exp(-0.5 * (logx-mu)**2/sigma**2)
    return y
    
