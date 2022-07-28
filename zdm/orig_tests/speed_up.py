import numpy as np
#import scipy as sp
from scipy import integrate
import time

from IPython import embed

import os, ctypes
from scipy import integrate, LowLevelCallable

lib = ctypes.CDLL(os.path.abspath('testlib.so'))
lib.f.restype = ctypes.c_double
lib.f.argtypes = (ctypes.c_int, 
                  ctypes.POINTER(ctypes.c_double))

func = LowLevelCallable(lib.f)

def loglognormal_dlog(logDM,*args):
    '''x values, mean and sigma are already in logspace
    returns p dlogx
    '''
    logmean=args[0]
    logsigma=args[1]
    norm=args[2]
    #norm=(2.*np.pi)**-0.5/logsigma
    return norm*np.exp(-0.5*((logDM-logmean)/logsigma)**2)

def integrate_pdm(ddm,ndm,logmean,logsigma,csumcut=0.999):
    # do this for the z=0 case
    mask=np.zeros([ndm])
    norm=(2.*np.pi)**-0.5/logsigma
    args=(logmean,logsigma,norm)
    pdm,err=integrate.quad(loglognormal_dlog,np.log(ddm*0.5)-logsigma*10,
                           np.log(ddm*0.5),args=args)
    mask[0]=pdm
    #csum=pdm
    #imax=ndm
    for i in np.arange(1,ndm):
        #if csum > CSUMCUT:
        #    imax=i
        #    break
        dmmin=(i-0.5)*ddm
        dmmax=dmmin+ddm
        pdm,err=integrate.quad(loglognormal_dlog,np.log(dmmin),np.log(dmmax),
                               args=args)
        #csum += pdm
        mask[i]=pdm
        
    #mask=mask[0:imax]
    return mask


def integrate_pdm(ddm,ndm,logmean,logsigma,csumcut=0.999,
                  use_C=False):
    # do this for the z=0 case
    mask=np.zeros([ndm])
    norm=(2.*np.pi)**-0.5/logsigma
    args=(logmean,logsigma,norm)
    if use_C:
        pdm,err=integrate.quad(func,np.log(ddm*0.5)-logsigma*10,
                           np.log(ddm*0.5),args=args)
    else:                        
        pdm,err=integrate.quad(loglognormal_dlog,np.log(ddm*0.5)-logsigma*10,
                           np.log(ddm*0.5),args=args)
    mask[0]=pdm
    #csum=pdm
    #imax=ndm
    for i in np.arange(1,ndm):
        #if csum > CSUMCUT:
        #    imax=i
        #    break
        dmmin=(i-0.5)*ddm
        dmmax=dmmin+ddm
        if use_C:
            pdm,err=integrate.quad(func,np.log(dmmin),np.log(dmmax),
                               args=args)
        else:                            
            pdm,err=integrate.quad(loglognormal_dlog,np.log(dmmin),np.log(dmmax),
                               args=args)
        #csum += pdm
        mask[i]=pdm
        
    #mask=mask[0:imax]
    return mask



def get_dm_mask(use_C=False):
    # Read data
    data = np.load('dm_file.npz')
    params = data['params']
    dmvals = data['dmvals']
    zvals = data['zvals']

    if len(params) != 2:
        raise ValueError("Incorrect number of DM parameters!",params," (expected log10mean, log10sigma)")
        exit()
    #expect the params to be log10 of actual values for simplicity
    # this converts to natural log
    logmean=params[0]/0.4342944619
    logsigma=params[1]/0.4342944619
    
    ddm=dmvals[1]-dmvals[0]
    
    ##### first generates a mask from the lognormal distribution #####
    # in theory allows a mask up to length of the DN values, but will
    # get truncated
    # the first value has half weight (0 to 0.5)
    # the rest have width of 1
    mask=np.zeros([dmvals.size])

    ndm=dmvals.size
    nz=zvals.size
    mask=np.zeros([nz,ndm])
    for j,z in enumerate(zvals):
        # with each redshift, we reduce the effects of a 'host' contribution by (1+z)
        # this means that we divide the value of logmean by by 1/(1+z)
        # or equivalently, we multiply the ddm by this factor
        # here we choose the former, but it is the same
        mask[j,:]=integrate_pdm(ddm*(1.+z),ndm,logmean,logsigma,
                                use_C=use_C)
        mask[j,:] /= np.sum(mask[j,:])

    return mask

def tst_func():
    # Read data
    data = np.load('dm_file.npz')
    params = data['params']
    dmvals = data['dmvals']
    zvals = data['zvals']

    if len(params) != 2:
        raise ValueError("Incorrect number of DM parameters!",params," (expected log10mean, log10sigma)")
        exit()
    #expect the params to be log10 of actual values for simplicity
    # this converts to natural log
    logmean=params[0]/0.4342944619
    logsigma=params[1]/0.4342944619
    norm=(2.*np.pi)**-0.5/logsigma
    
    ddm=dmvals[1]-dmvals[0]

    args=(logmean,logsigma,norm)
    pdm,err=integrate.quad(
        func,np.log(ddm*0.5)-logsigma*10,
                           np.log(ddm*0.5),args=args)
    pdm2,err=integrate.quad(
        loglognormal_dlog,np.log(ddm*0.5)-logsigma*10,
                           np.log(ddm*0.5),args=args)
    print(f"C={pdm}, python={pdm2}")

t0=time.process_time()
print("Starting at time ",t0)
mask_python = get_dm_mask()
t1=time.process_time()
print("Took ", t1-t0," seconds")

# Test me
#tst_func()
t0=time.process_time()
print("C: Starting at time ",t0)
mask_C = get_dm_mask(use_C=True)
t1=time.process_time()
print("C: Took ", t1-t0," seconds")

assert np.isclose(np.max(np.abs(mask_python-mask_C)), 0.)
print("Accuracy test passed!")
