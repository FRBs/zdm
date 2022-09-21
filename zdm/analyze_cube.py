""" Codes to analyze cube outputs """
import os
import numpy as np
import pickle
import glob

import math
import scipy

import pandas

from matplotlib import pyplot as plt

from zdm import io
from zdm import iteration

from IPython import embed

def slurp_cube(input_file:str, prefix:str, outfile:str, 
               nsurveys, debug:bool=False,
               suffix:str='.csv'):
    """ Slurp the cube ASCII output files and write 
    lC and ll into a numpy savez file

    Args:
        input_file (str): parameter file used to generate the cube
        prefix (str): prefix on the output files
        outfile (str): output file name.  Should have .npz extension
        nsurveys (int): Number of surveys in the analysis. 
        debug (int, optional): Debug?
        suffix (str, optional): File ending.  Allows for old .out files
    """
    # Grab em.  The order doesn't matter
    files = glob.glob(prefix+'*'+suffix)

    # Init
    input_dict=io.process_jfile(input_file)
    _, cube_dict, vparam_dict = iteration.parse_input_dict(input_dict)
    PARAMS = list(vparam_dict.keys())

    # Prep
    order, iorder = iteration.set_orders(cube_dict['parameter_order'], PARAMS)
    cube_shape = iteration.set_cube_shape(vparam_dict, order)

    param_shape = np.array([0]+cube_shape)[iorder].tolist()[:-1]

    # Outputs
    ll_cube = np.zeros(param_shape, dtype=np.float32)
    ll_cube[:] = -9e9
    lC_cube = np.zeros(param_shape, dtype=np.float32)

    pzDM_cube = np.zeros(param_shape, dtype=np.float32)
    pDM_cube = np.zeros(param_shape, dtype=np.float32)
    pDMz_cube = np.zeros(param_shape, dtype=np.float32)
    pz_cube = np.zeros(param_shape, dtype=np.float32)

    survey_items = ['lls', 'P_zDM', 'P_n', 'P_s', 'N']
    names = ['icube'] + PARAMS
    for ss in range (nsurveys):
        names += [item+f'_{ss}' for item in survey_items]
    names += ['ll']
    
    # Loop on cube output files
    survey_arrays = {}
    nsurvey = 0
    for ss, dfile in enumerate(files):

        print(f"Loading: {dfile}")
        df = pandas.read_csv(dfile)

        # Generate survey arrays
        if ss == 0:
            # Count the surveys
            for key in df.keys():
                if 'P_zDM' in key and len(key) > len('P_zDM'): 
                    # Generate them
                    for item in survey_items:
                        survey_arrays[item+key[-1]] = np.zeros(param_shape)
                    nsurvey += 1

        # Get indices
        indices = []
        ns = df.n

        for n in ns:
            r_current = np.array([0]+list(np.unravel_index(
                        int(n), cube_shape, order='F')))
            current = r_current[iorder][:-1] # Truncate lC
            # Ravel me back
            idx = np.ravel_multi_index(current, ll_cube.shape)
            indices.append(idx)

        # Set
        ll_cube.flat[indices] = df.lls
        lC_cube.flat[indices] = df.lC
        # New ones
        pzDM_cube.flat[indices] = df.p_zgDM
        pDM_cube.flat[indices] = df.p_DM
        pDMz_cube.flat[indices] = df.p_DMgz
        pz_cube.flat[indices] = df.p_z

        # Survey items
        for key in survey_arrays.keys():
            survey_arrays[key].flat[indices] = getattr(df, key)

        # Check
        if debug:
            embed(header='69 of analyze')
    
    # Grids
    out_dict = dict(ll=ll_cube, 
                    lC=lC_cube, 
                    params=PARAMS[:-1],
                    pzDM=pzDM_cube,
                    pDM=pDM_cube,
                    pDMz=pDMz_cube,
                    pz=pz_cube)
    # Save the parameter values too
    for name in PARAMS[:-1]:
        out_dict[name] = np.linspace(vparam_dict[name]['min'], 
                               vparam_dict[name]['max'],
                               vparam_dict[name]['n'])
    # Survey items
    for key in survey_arrays.keys():
        out_dict[key] = survey_arrays[key]

    # Write
    np.savez(outfile, **out_dict) #ll=ll_cube, lC=lC_cube, params=PARAMS[-1])
    print(f"Wrote: {outfile}")


def apply_gaussian_prior(lls:np.ndarray,
                    iparam:int,
                    values:np.ndarray,
                    mean:float,
                    sigma:float):
    """
    Applies a prior to parameter iparam
    with mean mean and deviation sigma.
    
    Returns a vector of length lls modified
    by that prior.
    """
    NDIMS= len(lls.shape)
    if iparam < 0 or iparam >= NDIMS:
        raise ValueError("Data only has ",NDIMS," dimensions.",
            "Please select iparam between 0 and ",NDIMS-1," not ",iparam)
    
    wlls = np.copy(lls)
    
    for iv,val in enumerate(values):
        # select ivth value from iparam dimension
        big_slice = [slice(None,None,None)]*NDIMS
        big_slice[iparam] = iv
        
        #calculate weights. Yes I know this is silly.
        weight = np.exp(-0.5*((val-mean)/sigma)**2)
        weight = np.log10(weight)
        wlls[tuple(big_slice)] += weight
    return wlls

def apply_H0_prior(lls:np.ndarray,
                    H0dim:int,
                    H0values:np.ndarray,
                    cmbH0:float,
                    cmb_sigma:float,
                    sn1aH0:float,
                    sn1a_sigma:float):
    """
    Applies a prior as a function of H0
    
    This is flat between two systematic values
    
    Args:
        lls (np.ndarray): values of likelihoods
        
        H0dim (int): dimension of H0 in the data
        
        H0values (float): vector specifying values of H0
        
        cmbH0 (float): value of H0 from the CMB
    
        cmb_sigma (float): 1 sigma uncertainty on H0 from CMB
    
        sn1aH0 (float): value of H0 from SN1a (distance ladder) measurements
    
        sn1a_sigma (float): 1 sigma uncertainty on H0 from SN1A
    
    
    Returns a vector of length lls modified
    by that prior.
    """
    
    # orders limits appropriately
    if cmbH0 < sn1aH0:
        lowH0=cmbH0
        lowsigma=cmb_sigma
        highH0=sn1aH0
        highsigma=sn1a_sigma
    else:
        lowH0=sn1aH0
        lowsigma=sn1a_sigma
        highH0=cmbH0
        highsigma=cmb_sigma
    
    NDIMS= len(lls.shape)
    if H0dim < 0 or H0dim >= NDIMS:
        raise ValueError("Data only has ",NDIMS," dimensions.",
            "Please select H0dim between 0 and ",NDIMS-1," not ",H0dim)
    
    wlls = np.copy(lls)
    
    for iv,val in enumerate(H0values):
        # select ivth value from iparam dimension
        big_slice = [slice(None,None,None)]*NDIMS
        big_slice[H0dim] = iv
        
        #calculate weights.
        if val < lowH0:
            weight = -0.5*((val-lowH0)/lowsigma)**2 * np.log10(np.exp(1))
        elif val > highH0:
            weight = -0.5*((val-highH0)/highsigma)**2 * np.log10(np.exp(1))
        else:
            weight=0.
        
        wlls[tuple(big_slice)] += weight
    
    return wlls

def get_slice_from_parameters(data,plist,mcvals,verbose=False,wanted="ll"):
    """
    Selects from data according to parameters which are
    as close to those in the mcvals as possible.
    
    Args:
        data: .npz structure generated by slurp_cube
        plist (list of strings): is a list of strings giving parameters
        mcvals (list of floats): is a float list of values one wants to approximate
        verbose (bool): toggle output level
        wanted: which column from the data to extract
    
    Returns:
        Array of NDIM=dim(data)-dim(plist) likelihood values
    """
    NDIMS=len(data["params"])
    if verbose:
        print("We have ",NDIMS," dimensions to this cube")
    
    # sets up array of which values to keep
    which=np.full([NDIMS],-1)
    
    # finds the order of these parameters in the multidimensional cube
    iplist=[]
    for param in plist:
        iplist.append(np.where(data["params"] == param)[0][0])
    
    # identifies the closest parameter values
    iclosest=[]
    for i,param in enumerate(plist):
        vals=data[param]
        vals=np.array(vals)
        diffs=(mcvals[i]-vals)**2
        imindiff=np.argmin(diffs)
        iclosest.append(imindiff)
        if verbose:
            print('For ',param,' MC truth ',mcvals[i]," closest cube value is the ",
                imindiff,"th, with value ",vals[imindiff])
        which[iplist[i]]=imindiff
    
    # selects the slice while keeping all MC parameters constant
    big_slice = [slice(None,None,None)]*NDIMS
    for i in np.arange(NDIMS):
        if which[i] >= 0: #insert code of -1 for "leave free"
            big_slice[i] = which[i]
    if isinstance(wanted, list):
        for i,item in enumerate(wanted):
            if i==0:
                selection=data[item][tuple(big_slice)]
            else:
                selection += data[item][tuple(big_slice)]
    else:
        selection=data[wanted][tuple(big_slice)]
    selection = np.array(selection)
    return selection

def get_bayesian_data(lls:np.ndarray, 
                      plls:np.ndarray=None, 
                      pklfile=None):
    """ Method to perform simple Bayesian analysis
    on the Log-likelihood cube

    Args:
        lls (np.ndarray): Log-likelood cube
        plls (np.ndarray, optional): Log-likelihood cube corrected for priors (e.g. alpha). Defaults to None.
        pklfile (str, optional): If given, write
            the output to this pickle file. Defaults to None.

    Returns:
        tuple: uvals,vectors,wvectors
            lists of np.ndarray's of LL analysis
            One item per parameter in the cube
    """
    NDIMS= len(lls.shape)
    
    # multiplies all log-likelihoods by the maximum value
    # ensures no floating point problems
    global_max = np.nanmax(lls)
    lls -= global_max
    
    #eventually remove this line
    if plls is None:
        plls = lls
    
    origlls=lls
    
    if plls is not None:
        w_global_max = np.nanmax(plls)
        plls = plls - w_global_max
    
    uvals=[]
    
    for i in np.arange(NDIMS):
        unique = np.arange(lls.shape[i])
        uvals.append(unique)

    # we now have a list of unique values for each dimension
    vectors=[] # this will contain the best values for 1d plots
    wvectors=[] # holds same as above, but including spectral penalty factor from ASKAP obs
    
    # loop over the DIMS
    for i in np.arange(NDIMS):
        
        # does 1D values
        vector=np.zeros([len(uvals[i])])
        wvector=np.zeros([len(uvals[i])])

        # selects for lls a subset corresponding only to that particular value of a variables
        for iv, ivv in enumerate(uvals[i]):
            big_slice = [slice(None,None,None)]*NDIMS
            # Construct the slice
            big_slice[i] = ivv
            #set1=np.where(data[:,idim]==ivv) #selects for a set of values
            #lls=data[set1,llindex]
            lls=origlls[tuple(big_slice)].flatten()
            
            # ignores all values of 0, which is what missing data is
            ignore=np.where(lls == 0.)[0]
            lls[ignore]=-99999
            
            # selects all fits that are close to the peak (i.e. percentage within 0.1%)
            try:
                themax=np.nanmax(lls)
            except:
                # all nans, probability =0. Easy!
                vector[iv]=0.
                wvector[iv]=0.
                continue
            
            OKlls=np.isfinite(lls) & (lls > themax-3)
            vector[iv]=np.sum(10**lls[OKlls])
            
            if plls is not None:
                wlls=plls[tuple(big_slice)].flatten()
                wthemax=np.nanmax(wlls)
                OKwlls=np.isfinite(wlls) & (wlls > wthemax-3)
                wvector[iv]=np.sum(10**wlls[OKwlls])
            
            #import pdb; pdb.set_trace()
        # Check
        vector *= 1./np.sum(vector)
        vector *= 1./np.abs(uvals[i][1]-uvals[i][0])
        vectors.append(vector)
        if plls is not None:
            wvector *= 1./np.sum(wvector)
            wvector *= 1./np.abs(uvals[i][1]-uvals[i][0])
            wvectors.append(wvector)	
        
    
    # now makes correction
    lls += global_max
    if plls is not None:
        plls += w_global_max
    
    # Pickle?
    if pklfile is not None:
        with open(pklfile, 'wb') as output:
            pickle.dump(uvals, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vectors, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(wvectors, output, pickle.HIGHEST_PROTOCOL)
        
    # result is just the total probability, normalised to unity, when summed over the parameter space
    # technically needs to be divided by the x-increment in bins.
    return uvals,vectors,wvectors


def get_2D_bayesian_data(lls:np.ndarray, 
                      plls:np.ndarray=None, 
                      pklfile=None):
    """ Method to perform simple Bayesian analysis
    on the Log-likelihood cube

    Args:
        lls (np.ndarray): Log-likelood cube
        plls (np.ndarray, optional): Log-likelihood cube corrected for priors (e.g. alpha). Defaults to None.
        pklfile (str, optional): If given, write
            the output to this pickle file. Defaults to None.

    Returns:
        tuple: uvals,ijs,arrays,warrays,
            uvals: values of each array, in form
            ijs: order of which parameters are combined in the arrays (e.g. [2,4])
            arrays: lists of 2D np.ndarray's of LL analysis
            warrays: weighted with prior on alpha
            ijs, arrays, and warrays have Nitems = Nparams*(Nparams-1)/2
            
    """
    NDIMS= len(lls.shape)
    
    # multiplies all log-likelihoods by the maximum value
    global_max = np.nanmax(lls)
    lls -= global_max
    
    #eventually remove this line
    if plls is None:
        plls = lls
    
    if plls is not None:
        w_global_max = np.nanmax(plls)
        plls = plls - w_global_max
    
    origlls=lls
    uvals=[]
    
    for i in np.arange(NDIMS):
        unique = np.arange(lls.shape[i])
        uvals.append(unique)

    # we now have a list of unique values for each dimension
    arrays=[] # this will contain the best values for 1d plots
    warrays=[] # holds same as above, but including spectral penalty factor from ASKAP obs
    ijs=[]
    
    # loop over the first dimensional combination
    for i in np.arange(NDIMS):
        
        # loops over the second dimension
        for j in (np.arange(NDIMS-i-1)+i+1):
                
            # does 1D values
            array=np.zeros([len(uvals[i]),len(uvals[j])])
            warray=np.zeros([len(uvals[i]),len(uvals[j])])
            
            # selects for lls a subset corresponding only to that particular value of a variables
            for iv, ivv in enumerate(uvals[i]):
                big_slice = [slice(None,None,None)]*NDIMS
                # Construct the slice
                big_slice[i] = ivv
                
                for jv, jvv in enumerate(uvals[j]):
                    # Construct the slice
                    big_slice[j] = jvv
                
                
                    #lls=data[set1,llindex]
                    lls=origlls[tuple(big_slice)].flatten()
                    
                    # ignores all values of 0, which is what missing data is
                    ignore=np.where(lls == 0.)[0]
                    lls[ignore]=-99999
                    
                    try:
                        themax=np.nanmax(lls)
                    except:
                        # all nans, probability =0. Easy!
                        arrays[iv,jv]=0.
                        warrays[iv,jv]=0.
                        continue
                    
                    OKlls=np.isfinite(lls) & (lls > themax-3)
                    array[iv,jv]=np.sum(10**lls[OKlls])
                    
                    if plls is not None:
                        wlls=plls[tuple(big_slice)].flatten()
                        wthemax=np.nanmax(wlls)
                        OKwlls=np.isfinite(wlls) & (wlls > wthemax-3)
                        warray[iv,jv]=np.sum(10**wlls[OKwlls])
                    
                    
            #normalisation over the parameter space to unity
            array *= 1./np.sum(array)
            arrays.append(array)
            if plls is not None:
                warray *= 1./np.sum(warray)
                warrays.append(warray)
            
            ijs.append([i,j])
    
    lls += global_max
    if plls is not None:
        plls += w_global_max
    
    # Pickle?
    if pklfile is not None:
        with open(pklfile, 'wb') as output:
            pickle.dump(uvals, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vectors, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(wvectors, output, pickle.HIGHEST_PROTOCOL)
       
    # result is just the total probability, normalised to unity, when summed over the parameter space
    # technically needs to be divided by the x-increment in bins.
    return uvals,ijs,arrays,warrays

def get_maxl_data(lls:np.ndarray, 
                      plls:np.ndarray=None, 
                      pklfile=None):
    """ Method to perform simple Bayesian analysis
    on the Log-likelihood cube

    Args:
        lls (np.ndarray): Log-likelood cube
        plls (np.ndarray, optional): Log-likelihood cube corrected for priors (e.g. alpha). Defaults to None.
        pklfile (str, optional): If given, write
            the output to this pickle file. Defaults to None.

    Returns:
        tuple: uvals,vectors,wvectors
            lists of np.ndarray's of LL analysis
            One item per parameter in the cube
    """
    NDIMS= len(lls.shape)
    
    # multiplies all log-likelihoods by the maximum value
    # ensures no floating point problems
    global_max = np.nanmax(lls)
    lls -= global_max
    
    #eventually remove this line
    if plls is None:
        plls = lls
    
    origlls=lls
    
    if plls is not None:
        w_global_max = np.nanmax(plls)
        plls = plls - w_global_max
    
           
    
    
    uvals=[]
    
    for i in np.arange(NDIMS):
        unique = np.arange(lls.shape[i])
        uvals.append(unique)

    # we now have a list of unique values for each dimension
    vectors=[] # this will contain the best values for 1d plots
    wvectors=[] # holds same as above, but including spectral penalty factor from ASKAP obs
    
    # loop over the DIMS
    for i in np.arange(NDIMS):
        
        # does 1D values
        vector=np.zeros([len(uvals[i])])
        wvector=np.zeros([len(uvals[i])])

        # selects for lls a subset corresponding only to that particular value of a variables
        for iv, ivv in enumerate(uvals[i]):
            big_slice = [slice(None,None,None)]*NDIMS
            # Construct the slice
            big_slice[i] = ivv
            #set1=np.where(data[:,idim]==ivv) #selects for a set of values
            #lls=data[set1,llindex]
            lls=origlls[tuple(big_slice)].flatten()
            
            # ignores all values of 0, which is what missing data is
            ignore=np.where(lls == 0.)[0]
            lls[ignore]=-99999
            
            # selects all fits that are close to the peak (i.e. percentage within 0.1%)
            try:
                themax=np.nanmax(lls)
            except:
                # all nans, probability =0. Easy!
                vector[iv]=0.
                wvector[iv]=0.
                continue
            
            vector[iv]=themax
            
            if plls is not None:
                wlls=plls[tuple(big_slice)].flatten()
                wthemax=np.nanmax(wlls)
                wvector[iv]=wthemax
            
            #import pdb; pdb.set_trace()
        # Check
        vectors.append(vector)
        if plls is not None:
            wvectors.append(wvector)	
        
    
    # now makes correction
    lls += global_max
    if plls is not None:
        plls += w_global_max
    
    # Pickle?
    if pklfile is not None:
        with open(pklfile, 'wb') as output:
            pickle.dump(uvals, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vectors, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(wvectors, output, pickle.HIGHEST_PROTOCOL)
        
    # result is just the total probability, normalised to unit, when summed over the parameter space
    # technically needs to be divided by the x-increment in bins.
    return uvals,vectors,wvectors


def get_2D_maxl_data(lls:np.ndarray, 
                      plls:np.ndarray=None, 
                      pklfile=None):
    """ Method to perform simple Bayesian analysis
    on the Log-likelihood cube

    Args:
        lls (np.ndarray): Log-likelood cube
        plls (np.ndarray, optional): Log-likelihood cube corrected for priors (e.g. alpha). Defaults to None.
        pklfile (str, optional): If given, write
            the output to this pickle file. Defaults to None.

    Returns:
        tuple: uvals,ijs,arrays,warrays,
            uvals: values of each array, in form
            ijs: order of which parameters are combined in the arrays (e.g. [2,4])
            arrays: lists of 2D np.ndarray's of LL analysis
            warrays: weighted with prior on alpha
            ijs, arrays, and warrays have Nitems = Nparams*(Nparams-1)/2
            
    """
    NDIMS= len(lls.shape)
    
    # multiplies all log-likelihoods by the maximum value
    global_max = np.nanmax(lls)
    lls -= global_max
    
    #eventually remove this line
    if plls is None:
        plls = lls
    
    if plls is not None:
        w_global_max = np.nanmax(plls)
        plls = plls - w_global_max
    
    origlls=lls
    uvals=[]
    
    for i in np.arange(NDIMS):
        unique = np.arange(lls.shape[i])
        uvals.append(unique)

    # we now have a list of unique values for each dimension
    arrays=[] # this will contain the best values for 1d plots
    warrays=[] # holds same as above, but including spectral penalty factor from ASKAP obs
    ijs=[]
    
    # loop over the first dimensional combination
    for i in np.arange(NDIMS):
        
        # loops over the second dimension
        for j in (np.arange(NDIMS-i-1)+i+1):
                
            # does 1D values
            array=np.zeros([len(uvals[i]),len(uvals[j])])
            warray=np.zeros([len(uvals[i]),len(uvals[j])])
            
            # selects for lls a subset corresponding only to that particular value of a variables
            for iv, ivv in enumerate(uvals[i]):
                big_slice = [slice(None,None,None)]*NDIMS
                # Construct the slice
                big_slice[i] = ivv
                
                for jv, jvv in enumerate(uvals[j]):
                    # Construct the slice
                    big_slice[j] = jvv
                    lls=origlls[tuple(big_slice)].flatten()
                    
                    # ignores all values of 0, which is what missing data is
                    ignore=np.where(lls == 0.)[0]
                    lls[ignore]=-99999
                    
                    try:
                        themax=np.nanmax(lls)
                    except:
                        # all nans, probability =0. Easy!
                        arrays[iv,jv]=0.
                        warrays[iv,jv]=0.
                        continue
                    
                    array[iv,jv]=themax
                    
                    if plls is not None:
                        wlls=plls[tuple(big_slice)].flatten()
                        wthemax=np.nanmax(wlls)
                        warray[iv,jv]=wthemax
                    
                    
            #normalisation over the parameter space to unity
            arrays.append(array)
            if plls is not None:
                warray *= 1./np.sum(warray)
                warrays.append(warray)
            
            ijs.append([i,j])
    
    lls += global_max
    if plls is not None:
        plls += w_global_max
    
    # Pickle?
    if pklfile is not None:
        with open(pklfile, 'wb') as output:
            pickle.dump(uvals, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vectors, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(wvectors, output, pickle.HIGHEST_PROTOCOL)
       
    # result is just the total probability, normalised to unity, when summed over the parameter space
    # technically needs to be divided by the x-increment in bins.
    return uvals,ijs,arrays,warrays


def interpolate_points(oldx,oldy,logspline=False, kind='cubic'):
    """
    performs simle spline interpolation of the data
    args:
        oldx: np array of original data on x axis
        oldy: np array of original data on y axis
        logspline (bool): perform interpolation in log space
        kind: type of spline to fit
    returns:
        x,y: interpolated points
    """
    
    #### does unweighted plotting ####
    x=np.linspace(oldx[0],oldx[-1],400)
    
    if logspline:
        # does interpolation in log-space
        f=scipy.interpolate.interp1d(oldx,np.log(oldy), kind=kind)
        y=np.exp(f(x))
    else:
        f=scipy.interpolate.interp1d(oldx,oldy, kind=kind)
        y=f(x)
        # check to ensure the splines have not done anything too dumb
        
        f2=scipy.interpolate.interp1d(oldx,np.log(oldy), kind=kind)
        y2=np.exp(f2(x))
        
        # replaces linear interpolation in small-value regime where splines can be dumb
        if np.min(y2) < 1.e-2:
            very_small = np.where(y2 < 1e-2)[0]
            y[very_small]=y2[very_small]
    
    y[np.where(y < 0.)]=0.
    return x,y

def extract_limits(x,y,p,method=1):
    """
    args:
        x (np.ndarray): xvalues of data (independent variable on which we place limits)
        y (np.ndarray): yvalues of data (probability distribution)
        p: probability of confidence interval (e.g. 0.68 for ~1 sigma)
        method: 1: includes most-likely points in C.I. first
                2: sets equal probability to either side
    returns:
        v0,v1: lower and upper bounds of range
    """
    
    # sets intervals according to highest likelihood
    if method==1: 
        # this sorts from lowest to highest
        sy=np.sort(y)
        # highest to lowest
        sy=sy[::-1]
        # now 0 to 1
        csy=np.cumsum(sy)
        csy /= csy[-1]
        
        # this is the likelihood we cut on
        cut=np.where(csy < 1.-2.*p)[0] # allowed values in interval
        cut=cut[-1] # last allowed value
        cut=sy[cut]
        OK=np.where(y > cut)[0]
        ik1=OK[0]
        ik2=OK[-1]
        v0=x[ik1]
        v1=x[ik2]
    elif method==2:
        cy=np.cumsum(y)
        cy /= cy[-1] # ignores normalisation in dx direction
        
        # gets lower value
        inside=np.where(cy > p)[0]
        ik1=inside[0]
        v0=x[ik1]
        # gets upper value
        inside=np.where(cy > 1.-p)[0]
        ik2=inside[0]
        v1=x[ik2]
    return v0,v1,ik1,ik2

def do_single_plots(uvals,vectors,wvectors,names,tag=None, fig_exten='.png',
                    dolevels=False,log=True,outdir='SingleFigs/',
                    vparams_dict=None, prefix='',truth=None,latexnames=None,
                    units=None,logspline=True, others=None,kind=None):
    """ Generate a series of 1D plots of the cube parameters

    Args:
        uvals (list): List, each element containing a
            np.ndarray giving the parameter values
            for each parameter. Total length nparams.
        vectors (list): [For each parameter, contains
            an unweighted vector giving
            1D probabilities for that value of the parameter]
        wvectors ([list]): [For each parameter, contains
            a weighted (with prior) vector giving
            1D probabilities for that value of the parameter]
        names ([type]): [description]
        tag ([type], optional): [description]. Defaults to None.
        fig_exten (str, optional): [description]. Defaults to '.png'.
        dolevels (bool, optional): [description]. Defaults to False.
        log (bool, optional): [description]. Defaults to True.
        outdir (str, optional): [description]. Defaults to 'SingleFigs/'.
        vparams_dict (dict, optional): parameter dict -- used to set x-values. Defaults to None.
        prefix (str, optional): [description]. Defaults to ''.
        latexnames: latex of axis names
        units: latex of units to attach to x-axis only
        logspline(bool): do spline fitting in logspace?
        others(list of arrays): list of other plots to add to data

    """
    
    if tag is not None:
        outdir=tag+outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    
    if log:
        logfile=outdir+'limits.dat'
        logfile=open(logfile,'w')
    
    if dolevels:
        results=np.zeros([len(uvals),9]) # holds mean and error info for each parameter
        prior_results=np.zeros([len(uvals),9]) # does the same with alpha priors
    
    for i,vals in enumerate(uvals):
        
        
        if len(vals) == 1:
            continue
        if len(vals) < 4:
            print("Length of the values is too small for spline interpolation, skipping...")
            continue
            kind = 'linear'
        else:
            if kind is None:
                kind = 'cubic'
        
        # does the for alpha
        plt.figure()
        lw=3
        

        # Convert vals?
        if vparams_dict is not None:
            # Check
            assert vparams_dict[names[i]]['n'] == len(vals)
            vals = np.linspace(vparams_dict[names[i]]['min'], 
                               vparams_dict[names[i]]['max'],
                               len(vals))
        
        # get raw ylimits
        # removes zeroes, could lead to strange behaviour in theory
        ymax=np.max(vectors[i])
        temp=np.where((vectors[i] > 0.) & (np.isfinite(vectors[i])) )
        
        # set to integers and get range
        ymax=math.ceil(ymax)
        ymin=0.
        
        x,y=interpolate_points(vals[temp],vectors[i][temp],logspline,kind)
        
        norm=np.sum(y)*(x[1]-x[0]) # integral y dx ~ sum y delta x
        norm=np.abs(norm)
        y /= norm
        vectors[i][temp] /= norm
        plt.plot(x,y,label='Uniform',color='blue',linewidth=lw,linestyle='-')
        plt.plot(vals[temp],vectors[i][temp],color='blue',linestyle='',marker='s')
        
        
        # weighted plotting
        if wvectors is not None:
            wx,wy=interpolate_points(vals[temp],wvectors[i][temp],logspline)
            wnorm=np.sum(wy)*(x[1]-x[0])
            wnorm = np.abs(wnorm)
        
            wvectors[i][temp] /= wnorm
            wy /= wnorm
            plt.plot(x,wy,label='Gauss',color='orange',linewidth=lw,linestyle='--')
        
        ax=plt.gca()
        ax.xaxis.set_ticks_position('both')
        #ax.Xaxis.set_ticks_position('both')
        if wvectors is not None:
            ymax=np.max([np.max(wy),np.max(y)])
        else:
            ymax=np.max(y)
        
        #ymax=(np.ceil(ymax*5.))/5.
        
        
        if dolevels==True:# and i != 1:
            limvals=np.array([0.00135,0.0228,0.05,0.15866])
            labels=['99.7%','95%','90%','68%']
            styles=['--',':','-.','-']
            upper=np.max(vectors[i])
            
            besty=np.max(y)
            imax=np.argmax(y)
            xmax=x[imax]
            results[i,0]=xmax
            string=names[i]+" & {0:4.2f}".format(xmax)
            for iav,av in enumerate(limvals):
                # need to integrate from min to some point
                # gets cumulative distribution
                # sets intervals according to highest likelihood
                
                v0,v1,ik1,ik2=extract_limits(x,y,av,method=1)
                
                string += " & $_{"
                string += "{0:4.2f}".format(v0-xmax)
                string += "}^{+"
                string += "{0:4.2f}".format(v1-xmax)
                string += "}$ "
                results[i,2*iav+1]=v0-xmax
                results[i,2*iav+2]=v1-xmax
                
                hl=0.03
                doff=(x[-1]-x[0])/100.
                ybar=(av+ymax)/2.
                xbar=(v0+v1)/2.
                
                # need to separate the plots
                if wvectors is not None:
                    if ik1 != 0:
                        #if iav==3 and i==4:
                        #    ybar -= 0.8
                        plt.plot([x[ik1],x[ik1]],[ymax,y[ik1]],color='blue',linestyle=styles[iav],alpha=0.5)
                        if i==1:
                            t=plt.text(x[ik1]+doff*0.5,(ymax)+(-3.6+iav)*0.2*ymax,labels[iav],rotation=90,fontsize=12)
                            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                    if ik2 != wy.size-1:
                        plt.plot([x[ik2],x[ik2]],[ymax,y[ik2]],color='blue',linestyle=styles[iav],alpha=0.5)
                        if i != 1:
                            t=plt.text(x[ik2]-doff*3,(ymax)+(-3.6+iav)*0.2*ymax,labels[iav],rotation=90,fontsize=12)
                            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                else:
                    plt.plot([x[ik1],x[ik1]],[0,y[ik1]],color='red',linestyle=styles[iav])
                    plt.plot([x[ik2],x[ik2]],[0,y[ik2]],color='red',linestyle=styles[iav])
                    Dx=x[-1]-x[0]
                    if Dx < 0.:
                        plt.text(x[ik1],y[ik1]+ymax*0.03,labels[iav],color='red',rotation=90)
                        plt.text(x[ik2]+0.02*Dx,y[ik2]+ymax*0.03,labels[iav],color='red',rotation=90)
                    else:
                        plt.text(x[ik1]-0.02*Dx,y[ik1]+ymax*0.03,labels[iav],color='red',rotation=90)
                        plt.text(x[ik2],y[ik2]+ymax*0.03,labels[iav],color='red',rotation=90)
                    #print("For parameter ",i," CI ",iav, " is ",x[ik1]," to ",x[ik2])
            string += " & "
            
        #could just ignore the weightings   
        if wvectors is not None:
            plt.plot(vals[temp],wvectors[i][temp],color='orange',linestyle='',marker='o')
            if dolevels==True:
                limvals=np.array([0.0015,0.025,0.05,0.16])
                labels=['99.7%','95%','90%','68%']
                styles=['--',':','-.','-']
                upper=np.max(wvectors[i])
                
                besty=np.max(wy)
                imax=np.argmax(wy)
                xmax=x[imax]
                prior_results[i,0]=xmax
                string+=" {0:4.2f}".format(xmax)
                for iav,av in enumerate(limvals):
                    
                    # sets intervals according to highest likelihood
                    v0,v1,ik1,ik2=extract_limits(x,wy,av,method=1)
                    
                    string += " & $_{"
                    string += "{0:4.2f}".format(v0-xmax)
                    string += "}^{+"
                    string += "{0:4.2f}".format(v1-xmax)
                    string += "}$ "
                    prior_results[i,2*iav+1]=v0-xmax
                    prior_results[i,2*iav+2]=v1-xmax
                    
                    # version 2
                    hl=0.03
                    
                    doff=(x[-1]-x[0])/100.
                    if i==1:
                        doff=0.001
                    ybar=(av+ymin)/2.
                    xbar=(v0+v1)/2.
                    if ik1 != 0:
                        plt.plot([x[ik1],x[ik1]],[ymin,wy[ik1]],color='orange',linestyle=styles[iav])
                        if i ==1:
                            t=plt.text(x[ik1]+doff*0.5,wy[ik1]/2.2,labels[iav],rotation=90,fontsize=12)
                            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                        
                    if ik2 != wy.size-1:
                        
                        plt.plot([x[ik2],x[ik2]],[ymin,wy[ik2]],color='orange',linestyle=styles[iav])
                        if i != 1:
                            t=plt.text(x[ik2]-doff*3,wy[ik2]/2.2,labels[iav],rotation=90,fontsize=12)
                            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
        other_styles=[":","--","-."]
        # plot any other plots
        if others is not None:
            if others[i] is not None:
                for io,data in enumerate(others[i]):
                    x,y=interpolate_points(vals,data,logspline)
                    norm=np.sum(y)*(x[1]-x[0]) # integral y dx ~ sum y delta x
                    norm=np.abs(norm)
                    y /= norm
                    plt.plot(x,y,color='grey',linewidth=1,linestyle=other_styles[io % 3])
        if dolevels:
            string += "\\\\"
            if log:
                logfile.write(string+'\n')
            else:
                print(string)
        #plt.ylim(0.,ymax)
        plt.gca().set_ylim(bottom=0)
        if truth is not None:
            plt.plot([truth[i],truth[i]],plt.gca().get_ylim(),color='black',linestyle=':')
            Dx=x[-1]-x[0]
            plt.text(truth[i]+0.01*Dx,ymax*0.4,'simulated truth',rotation=90)
        
        if latexnames is not None:
            if units is not None:
                plt.xlabel(latexnames[i]+" "+units[i])
            else:
                plt.xlabel(latexnames[i])
            plt.ylabel('$p($'+latexnames[i]+'$)$')
        else:
            plt.xlabel(names[i])
            plt.ylabel('p('+names[i]+')')
        if i==4 and wvectors is not None:
            plt.legend(loc='upper left',title='Prior on $\\alpha$')
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, prefix+names[i]+fig_exten), dpi=200)
        plt.close()
    if log:
        logfile.close()
    if dolevels:
        return results,prior_results
    else:
        return

def make_1d_plots_by_contribution(data,contributions,labels,prefix="",
    fig_exten='.png',log=False,splines=True,latexnames=None,units=None):
    """
    contributions: list of vectors giving various likelihood terms
    args:
        splines (bool): draw cubic splines
        Labels: lists labels stating what these are
        latexnames: latex for x and p(X)
        units: appends units to x axis but not p(X)
    """
    ######################### 1D plots, split by terms ################
    all_uvals=[]
    all_vectors=[]
    all_wvectors=[]
    combined=data["pzDM"]+data["pDM"]
    
    # gets 1D Bayesian curves for each contribution
    for datatype in contributions:
        uvals,vectors,wvectors=get_bayesian_data(datatype)
        all_uvals.append(uvals)
        all_vectors.append(vectors)
        all_wvectors.append(wvectors)
    params=data["params"]
    
    # gets unique values for each axis
    param_vals=[]
    param_list=[data["lEmax"],data["H0"],data["alpha"],data["gamma"],data["sfr_n"],data["lmean"],data["lsigma"]]
    xlatexnames=['\\log_{10} E_{\\rm max} {\\rm [erg]}','H_0 {\\rm [km\,s^{-1}\,Mpc^{-1}]}','\\alpha','\\gamma','n_{\\rm sfr}','\\mu_{\\rm host} {\\rm [pc\,cm^{-3}]}','\\sigma_{\\rm host}']
    ylatexnames=['\\log_{10} E_{\\rm max}','H_0','\\alpha','\\gamma','n_{\\rm sfr}','\\mu_{\\rm host}','\\sigma_{\\rm host}']
    
    for col in param_list:
        unique=np.unique(col)
        param_vals.append(unique)
    # assigns different plotting styles to help distinguish curves
    linestyles=['-','--','-.',':','-','--','-.',':','-','--','-.',':']
    for which in np.arange(len(param_list)):
        plt.figure()
        plt.xlabel('$'+xlatexnames[which]+'$')
        plt.ylabel('$p('+ylatexnames[which]+')$')
        xvals=param_vals[which]
        
        for idata,vectors in enumerate(all_vectors):
            if splines:
                xdata=np.linspace(xvals[0],xvals[-1],100)
                f=scipy.interpolate.interp1d(xvals,np.log(vectors[which]),
                                      kind='cubic')
                ydata=np.exp(f(xdata))
                plt.plot(xdata,ydata,label=labels[idata],linestyle=linestyles[idata])
                plt.scatter(xvals,vectors[which],color=plt.gca().lines[-1].get_color())
            else:
                ydata=vectors[which]
                xdata=xvals
                #print(labels[idata]," has values ",vector)
                plt.plot(xdata,ydata,label=labels[idata],linestyle=linestyles[idata])
        
        if log:
            plt.yscale('log')
            #plt.ylim(np.max(vector)*1e-3,np.max(vector)) #improve this
        plt.legend()
        plt.savefig(prefix+params[which]+fig_exten)
        plt.close()
    
def gen_vparams(indices:tuple, vparam_dict:dict):
    new_dict = {}
    for ss, key in enumerate(vparam_dict.keys()):
        if vparam_dict[key]['n'] <= 0:
            continue
        # 
        vals = np.linspace(vparam_dict[key]['min'], 
                           vparam_dict[key]['max'],
                           vparam_dict[key]['n'])
        # 
        new_dict[key] = vals[indices[ss]]
    # Return
    return new_dict



def make_2d_plot(array,xlabel,ylabel,xvals,yvals,savename=None,norm=None):
    """
    Makes 2D plot given an array of probabilities
    
    array: 2D grid of data to plot
    
    xlabel: string to label on x axis
    ylabel: string to label y axis
    
    xvals: np array, values of x data
    yvals: np array, values of y data
    
    savename (optional): string to save data under
    """
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # calculates increment in values
    Dx=xvals[-1]-xvals[0]
    Dy=yvals[-1]-yvals[0]
    
    nx=xvals.size
    ny=yvals.size
    
    dx=np.abs(Dx/nx)
    dy=np.abs(Dy/ny)
    
    aspect=np.abs(dx/dy * nx/ny)
    
    # the dx/2 etc is so that parameter values line up with the centres of each pixel
    extent=[np.min(xvals)-dx/2.,np.max(xvals)+dx/2.,np.min(yvals)-dy/2.,np.max(yvals)+dy/2.]
    
    if norm==0:
        array=array/np.max(array,axis=0)
        clabel='$p($'+ylabel+'|'+xlabel+'$)$'
    elif norm==1:
        array=array.T/np.max(array,axis=1)
        array=array.T
        clabel='$p($'+xlabel+'|'+ylabel+'$)$'
    else:
        clabel='$p($'+xlabel+','+ylabel+'$)$'
    
    plt.imshow(array.T,origin='lower',extent=extent,aspect=aspect)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    cbar=plt.colorbar()
    cbar.set_label(clabel)
    if savename is None:
        savename=xlabel+"_"+ylabel+".pdf"
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

