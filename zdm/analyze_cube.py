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
               nsurveys, debug:bool=False):
    """ Slurp the cube ASCII output files and write 
    lC and ll into a numpy savez file

    Args:
        input_file (str): parameter file used to generate the cube
        prefix (str): prefix on the output files
        outfile (str): output file name.  Should have .npz extension
        nsurveys (int): Number of surveys in the analysis. 
        debug (int, optional): Debug?
    """
    # Grab em.  The order doesn't matter
    files = glob.glob(prefix+'*.out') 

    # Init
    input_dict=io.process_jfile(input_file)
    _, cube_dict, vparam_dict = iteration.parse_input_dict(input_dict)
    PARAMS = list(vparam_dict.keys())

    # Prep
    order, iorder = iteration.set_orders(cube_dict['parameter_order'], PARAMS)
    cube_shape = iteration.set_cube_shape(vparam_dict, order)

    param_shape = np.array([0]+cube_shape)[iorder].tolist()[:-1]
    ll_cube = np.zeros(param_shape)
    lC_cube = np.zeros(param_shape)
    ll_cube[:] = -9e9

    survey_items = ['lls', 'DM_z', 'N', 'SNR', 'Nex']
    names = ['icube'] + PARAMS
    for ss in range (nsurveys):
        names += [item+f'_{ss}' for item in survey_items]
    names += ['ll']
    
    # Loop on cube output files
    for dfile in files:
        print(f"Loading: {dfile}")
        df = pandas.read_csv(dfile, header=None, delimiter=r"\s+", names=names)

        for index, row in df.iterrows():
            # Unravel
            r_current = np.array([0]+list(np.unravel_index(
                        int(row.icube), cube_shape, order='F')))
            current = r_current[iorder][:-1] # Truncate lC
            # Ravel me back
            idx = np.ravel_multi_index(current, ll_cube.shape)
            # Set
            ll_cube.flat[idx] = row.ll
            lC_cube.flat[idx] = row.lC
        # Check
        if debug:
            embed(header='69 of analyze')
    
    # Write
    np.savez(outfile, ll=ll_cube, lC=lC_cube)
    print(f"Wrote: {outfile}")


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
            
    origlls=lls
    if plls is None:
        plls = lls
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
            wlls=plls[tuple(big_slice)].flatten()
            
            # selects all fits that are close to the peak (i.e. percentage within 0.1%)
            try:
                themax=np.nanmax(lls)
            except:
                # all nans, probability =0. Easy!
                vector[iv]=0.
                wvector[iv]=0.
                continue
            
            wthemax=np.nanmax(wlls)
            OKlls=np.isfinite(lls) & (lls > themax-3)
            OKwlls=np.isfinite(lls) & (wlls > wthemax-3)
            
            vector[iv]=np.sum(10**lls[OKlls])
            wvector[iv]=np.sum(10**wlls[OKwlls])
        vector *= 1./np.sum(vector)
        wvector *= 1./np.sum(wvector)	
        vectors.append(vector)
        wvectors.append(wvector)
    
    # Pickle?
    if pklfile is not None:
        with open(pklfile, 'wb') as output:
            pickle.dump(uvals, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vectors, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(wvectors, output, pickle.HIGHEST_PROTOCOL)
        
    # result is just the total probability, normalised to unit, when summed over the parameter space
    # technically needs to be divided by the x-increment in bins.
    return uvals,vectors,wvectors


def do_single_plots(uvals,vectors,wvectors,names,tag=None, fig_exten='.png',
                    dolevels=False,log=True,outdir='SingleFigs/',
                    vparams_dict=None, prefix=''):
    """ Generate a series of 1D plots of the cube parameters

    Args:
        uvals (np.ndarray): [description]
        vectors (): [description]
        wvectors ([type]): [description]
        names ([type]): [description]
        tag ([type], optional): [description]. Defaults to None.
        fig_exten (str, optional): [description]. Defaults to '.png'.
        dolevels (bool, optional): [description]. Defaults to False.
        log (bool, optional): [description]. Defaults to True.
        outdir (str, optional): [description]. Defaults to 'SingleFigs/'.
        vparams_dict (dict, optional): parameter dict -- used to set x-values. Defaults to None.
        prefix (str, optional): [description]. Defaults to ''.

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
            kind = 'linear'
        else:
            kind = 'cubic'
        # does the for alpha
        plt.figure()
        #ok=np.where(dodgiesv[i]==0)[0]
        #bad=np.where(dodgiesv[i]==1)[0]
        #ok=np.array(ok)
        #bad=np.array(bad)
        lw=3
        

        # Convert vals?
        if vparams_dict is not None:
            # Check
            assert vparams_dict[names[i]]['n'] == len(vals)
            vals = np.linspace(vparams_dict[names[i]]['min'], 
                               vparams_dict[names[i]]['max'],
                               len(vals))
        
        # get raw ylimits
        ymax=np.max(vectors[i])
        temp=np.where(wvectors[i] > -900)
        
        # set to integers and get range
        ymax=math.ceil(ymax)
        
        ymin=0.
        
        ####### annoying special by-hand crap #######
        #if i==1:
        #    plt.xlim(-2.5,1)
        #if i==3:
        #    plt.xlim(0,4)
        
        # set limits and ticks
        #plt.ylim(ymin,ymax)
        #plt.yticks(yvals)
        
        #### does unweighted plotting ####
        x=np.linspace(vals[temp][0],vals[temp][-1],400)
        f=scipy.interpolate.interp1d(vals[temp],vectors[i][temp], kind=kind)
        y=f(x)
        y[np.where(y < 0.)]=0.
        
        norm=np.sum(y)*(x[1]-x[0]) # integral y dx ~ sum y delta x
        norm=np.abs(norm)
        y /= norm
        vectors[i][temp] /= norm
        plt.plot(x,y,label='Uniform',color='blue',linewidth=lw,linestyle='-')
        plt.plot(vals[temp],vectors[i][temp],color='blue',linestyle='',marker='s')
        
        # weighted plotting
        wf=scipy.interpolate.interp1d(vals[temp],wvectors[i][temp],
                                      kind=kind)

        wy=wf(x)
        wy[np.where(wy < 0.)]=0.
        wnorm=np.sum(wy)*(x[1]-x[0])
        wnorm = np.abs(wnorm)
        wvectors[i][temp] /= wnorm
        wy /= wnorm
        plt.plot(x,wy,label='Gauss',color='orange',linewidth=lw,linestyle='--')
        
        ax=plt.gca()
        ax.xaxis.set_ticks_position('both')
        #ax.Xaxis.set_ticks_position('both')
        ymax=np.max([np.max(wy),np.max(y)])
        
        ymax=(np.ceil(ymax*5.))/5.
        
        
        if dolevels==True:# and i != 1:
            limvals=np.array([0.015,0.025,0.05,0.16])
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
                if True:
                    
                    # this sorts from lowest to highest
                    sy=np.sort(y)
                    # highest to lowest
                    sy=sy[::-1]
                    # now 0 to 1
                    csy=np.cumsum(sy)
                    csy /= csy[-1]
                    
                    # this is the likelihood we cut on
                    cut=np.where(csy < 1.-2.*av)[0] # allowed values in interval
                    
                    cut=cut[-1] # last allowed value
                    cut=sy[cut]
                    OK=np.where(y > cut)[0]
                    ik1=OK[0]
                    ik2=OK[-1]
                    
                    v0=x[ik1]
                    v1=x[ik2]
                if False:
                    cy=np.cumsum(y)
                    cy /= cy[-1] # ignores normalisation in dx direction
                    
                    # gets lower value
                    inside=np.where(cy > av)[0]
                    ik1=inside[0]
                    v0=x[ik1]
                    
                    # gets upper value
                    inside=np.where(cy > 1.-av)[0]
                    ik2=inside[0]
                    v1=x[ik2]
                
                string += " & $_{"
                string += "{0:4.2f}".format(v0-xmax)
                string += "}^{+"
                string += "{0:4.2f}".format(v1-xmax)
                string += "}$ "
                results[i,2*iav+1]=v0-xmax
                results[i,2*iav+2]=v1-xmax
                # version 1
                #plt.plot([x[ik1],x[ik2]],[av,av],color='orange',linestyle='-')
                #if iav==0:
                #	xbar=(x[ik1]+x[ik2])/2.
                #plt.text(xbar,av+0.1,labels[iav])
                #if i==1: #do not plot
                #	continue
                # version 2
                hl=0.03
                doff=(x[-1]-x[0])/100.
                ybar=(av+ymax)/2.
                xbar=(x[ik1]+x[ik2])/2.
                if ik1 != 0:
                    if iav==3 and i==4:
                        ybar -= 0.8
                    #plt.arrow(x[ik1]+hl,ybar,x[ik2]-x[ik1]-2*hl,0.,color='orange',head_width=0.05,head_length=hl)
                    plt.plot([x[ik1],x[ik1]],[ymax,y[ik1]],color='blue',linestyle=styles[iav],alpha=0.5)
                    if i==1:
                        t=plt.text(x[ik1]+doff*0.5,(ymax)+(-3.6+iav)*0.2*ymax,labels[iav],rotation=90,fontsize=12)
                        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                    
                if ik2 != wy.size-1:
                    plt.plot([x[ik2],x[ik2]],[ymax,y[ik2]],color='blue',linestyle=styles[iav],alpha=0.5)
                    if i != 1:
                        t=plt.text(x[ik2]-doff*3,(ymax)+(-3.6+iav)*0.2*ymax,labels[iav],rotation=90,fontsize=12)
                        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
            string += " & "
            
            
        
        plt.plot(vals[temp],wvectors[i][temp],color='orange',linestyle='',marker='o')
        if dolevels==True:
            limvals=np.array([0.015,0.025,0.05,0.16])
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
                if True:
                    
                    # this sorts from lowest to highest
                    sy=np.sort(wy)
                    # highest to lowest
                    sy=sy[::-1]
                    # now 0 to 1
                    csy=np.cumsum(sy)
                    csy /= csy[-1]
                    
                    # this is the likelihood we cut on
                    cut=np.where(csy < 1.-2.*av)[0] # allowed values in interval
                    
                    cut=cut[-1] # last allowed value
                    cut=sy[cut]
                    OK=np.where(wy > cut)[0]
                    ik1=OK[0]
                    ik2=OK[-1]
                    
                    v0=x[ik1]
                    v1=x[ik2]
                if False:
                    cy=np.cumsum(wy)
                    cy /= cy[-1] # ignores normalisation in dx direction
                    
                    # gets lower value
                    inside=np.where(cy > av)[0]
                    ik1=inside[0]
                    v0=x[ik1]
                    
                    # gets upper value
                    inside=np.where(cy > 1.-av)[0]
                    ik2=inside[0]
                    v1=x[ik2]
                
                
                string += " & $_{"
                string += "{0:4.2f}".format(v0-xmax)
                string += "}^{+"
                string += "{0:4.2f}".format(v1-xmax)
                string += "}$ "
                prior_results[i,2*iav+1]=v0-xmax
                prior_results[i,2*iav+2]=v1-xmax
                #0:4.2f}^{1:4.2f}$ ".format(v0-xmax,v1-xmax)
                #print("Parameter ",i," level ",labels[iav]," interval is ",v0,v1)
                # version 1
                #plt.plot([x[ik1],x[ik2]],[av,av],color='orange',linestyle='-')
                #if iav==0:
                #	xbar=(x[ik1]+x[ik2])/2.
                #plt.text(xbar,av+0.1,labels[iav])
                
                # version 2
                hl=0.03
                
                doff=(x[-1]-x[0])/100.
                if i==1:
                    doff=0.001
                ybar=(av+ymin)/2.
                xbar=(x[ik1]+x[ik2])/2.
                if ik1 != 0:
                    #plt.arrow(x[ik1]+hl,ybar,x[ik2]-x[ik1]-2*hl,0.,color='orange',head_width=0.05,head_length=hl)
                    plt.plot([x[ik1],x[ik1]],[ymin,wy[ik1]],color='orange',linestyle=styles[iav])
                    if i ==1:
                        t=plt.text(x[ik1]+doff*0.5,wy[ik1]/2.2,labels[iav],rotation=90,fontsize=12)
                        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                    
                if ik2 != wy.size-1:
                    
                    plt.plot([x[ik2],x[ik2]],[ymin,wy[ik2]],color='orange',linestyle=styles[iav])
                    if i != 1:
                        t=plt.text(x[ik2]-doff*3,wy[ik2]/2.2,labels[iav],rotation=90,fontsize=12)
                        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                    
                    #plt.arrow(x[ik2]-hl,ybar,-1*(x[ik2]-x[ik1]-2*hl),0.,color='orange',head_width=0.05,head_length=hl)
                #else:
                #	plt.plot([x[ik1],x[ik1]],[ymin,av],color='orange',linestyle=styles[iav])
                #	plt.plot([x[ik2],x[ik2]],[ymin,av],color='orange',linestyle=styles[iav])
                #	plt.text(x[ik1]+0.01,ybar-0.05,labels[iav],rotation=90)
                #	plt.text(x[ik2]-0.04,ybar-0.05,labels[iav],rotation=90)
                    #plt.text([x[ik2]-0.05,ybar-0.05,labels[iav],rotation=90)
                    
                    #plt.arrow(xbar,ybar,(x[ik2]-x[ik1]-2*hl)/2,0.,color='orange',head_width=0.05,head_length=hl)
                    #plt.arrow(xbar,ybar,(x[ik2]-x[ik1]-2*hl)/-2,0.,color='orange',head_width=0.05,head_length=hl)
                
                
                #plt.text(xbar-0.05,ybar+0.05,labels[iav])
            string += "\\\\"
            if log:
                logfile.write(string+'\n')
            else:
                print(string)
        plt.ylim(0.,ymax)
        plt.xlabel(names[i])
        plt.ylabel('p('+names[i]+')')
        if i==4:
            plt.legend(loc='upper left',title='Prior on $\\alpha$')
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, prefix+names[i]+fig_exten))
        plt.close()
    if log:
        logfile.close()
    if dolevels:
        return results,prior_results
    else:
        return