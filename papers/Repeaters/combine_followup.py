"""
This script combines the results from C.W.James et al, ApJ  895 L22 (2020)
with those from this work to create a joint limit.


NOTE: the numpy files can be found at /Users/cjames/CRAFT/Paper/FollowUpRepitition/LimitPrecalculations
"""


import numpy as np
import matplotlib.pyplot as plt
import utilities as ute
import os

import matplotlib
matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    Poisson = False
    
    ###### old data ######
    indir = 'FollowUpData/'
    if Poisson:
        llf = indir+"k_1.0_g_-0.9_fits_llgrid.npy"
    else:
        llf = indir+"k_0.34_g_-0.9_fits_llgrid.npy"
    
    ll_grid = np.load(llf)
    
    rmaxesf = indir+"k_0.34_g_-0.9_fits_Rmaxes.npy"
    rmaxes = np.load(rmaxesf)
    
    rminsf = indir+"k_0.34_g_-0.9_fits_Rmins.npy"
    rmins = np.load(rminsf)
    
    indicesf = indir+"k_0.34_g_-0.9_fits_indices.npy"
    indices = np.load(indicesf)
    
    if Poisson:
        indir=indir+"Poisson_"
    
    # checks this is correct - makes some plots!
    # takes best value in Rmax dimension
    ll_gridx = np.nanmax(ll_grid,axis=2)
    
    # checks this is correct - makes some plots!
    # takes best value in Rmax dimension
    ll_gridz = np.nanmax(ll_grid,axis=0)
    
    
    if True:
        # plots a test version of Figure 1 from the ApJ paper
        name=indir + 'test_rmin_indices.pdf'
        labels = ['$R_{\\rm min}$','$\\gamma_r$']
        ute.plot_2darr(ll_gridx,labels,name,[np.log10(rmins),indices],[rmins,indices],\
                    clabel="$p_{\\rm followup}$",crange=[-15,-5.5])
        
        
        name=indir + 'test_rmax_indices.pdf'
        labels = ['$R_{\\rm max}$','$\\gamma_r$']
        ute.plot_2darr(ll_gridz.T,labels,name,[np.log10(rmaxes),indices],[rmaxes,indices],\
                    clabel="$p_{\\rm followup}$",crange=None)
        
        
        # generates plot of best Rmaxes for old data
        ni,nj,nk = ll_grid.shape
        best_rmaxes = np.zeros([ni,nj])
        for i in np.arange(ni):
            for j in np.arange(nj):
                irmax = np.argmax(ll_grid[i,j,:])
                best_rmaxes[i,j] = rmaxes[irmax]
                
        
        name=indir + 'best_rmaxes_ApJ.pdf'
        ute.plot_2darr(np.log10(best_rmaxes),labels,name,[np.log10(rmins),indices],[rmins,indices],\
                    clabel="$R_{\\rm max}$",crange=None)
        
     
    ####### now imports latest data ######
    
    infile = 'Rfitting39_1.0/mc_FC391.0converge_set_0_output.npz'
    
    cdata=np.load(infile)
        
    lpNs=cdata['arr_3']
    Rmins=cdata['arr_5']
    Rmaxes=cdata['arr_6']
    Rgammas=cdata['arr_7']
    lrkps=cdata['arr_9']
    ltdm_krs=cdata['arr_11']
    MCrank = cdata['arr_18']
    
    
    if True:
        for irm,rmax in enumerate(rmaxes):
            ll_gridhrm = ll_grid[:,:,irm]
            name=indir + 'high_rmax_'+str(irm)+"_"+str(rmax)[0:6]+'.pdf'
            labels = ['$R_{\\rm min}$','$\\gamma_r$']
            
            iclose_rmax = np.where(Rmaxes > rmax)[0][0]
            close_rmax = Rmaxes[iclose_rmax]
            #if close_rmax/rmax < 3. and rmax / close_rmax < 3.:
                
            print(rmax,"close rmax ",close_rmax)
            # for this value of rmax, get rgamma, rmin data, show it as a line
            thesermins = np.zeros([Rgammas.size])
            markers=['+']*Rgammas.size
            for j,rg in enumerate(Rgammas):
                thesermins[j] = Rmins[j,iclose_rmax]
                
            scatter = [Rgammas,np.log10(thesermins),markers]
            
            ute.plot_2darr(ll_gridhrm,labels,name,[np.log10(rmins),indices],[np.log10(rmins),indices],\
                    clabel="$p_{\\rm followup}$",crange=[-7,-4.4],scatter=scatter)
            break
    
        
    
    # constructs dec, Nburst, dm, p(N) product
    lps = lrkps + MCrank + ltdm_krs + lpNs
    
    
    ###### scale rmins and rmaxes #######
    gamma=-0.9
    alpha = -1.9
    scale = (600./1300.)**alpha + (1e39/1e38)**gamma
    
    rmins *= scale
    rmaxes *= scale
    
    savefile = indir + 'interpolated_values.npy'
    savefilenbrm = indir + 'brm_values.npy'
    savefilebrm = indir + 'best_rmins.npy'
    load = False
    if os.path.exists(savefile) and os.path.exists(savefilenbrm) \
        and os.path.exists(savefilebrm) and load:
        newll = np.load(savefile)
        newllbrm = np.load(savefilenbrm)
        brms = np.load(savefilebrm)
    else:
        newll = np.zeros([Rgammas.size,Rmaxes.size])
        newllbrm = np.zeros([Rgammas.size,Rmaxes.size])
        brms = np.zeros([Rgammas.size,Rmaxes.size])
        for i,Rgamma in enumerate(Rgammas):
            for j,Rmax in enumerate(Rmaxes):
                Rmin = Rmins[i,j]
                val = interpolate(ll_grid,Rmin,Rgamma,Rmax,rmins,indices,rmaxes)
                newll[i,j] = val
                valbrm,brm = interpolate_bestRmin(ll_grid,Rmin,Rgamma,Rmax,rmins,indices,rmaxes)
                newllbrm[i,j] = valbrm
                brms[i,j]=brm
        np.save(savefile,newll)
        np.save(savefilenbrm,newllbrm)
        np.save(savefilebrm,brms)
    
    print("Max/min of newll is ",np.max(newll),np.min(newll))
    print("Max/min of ll_grid is ",np.nanmax(ll_grid),np.nanmin(ll_grid))
    
    # we now have an array of interpolated log-likelihoods, newll, 
    # from the ApJ Lett results to the same grid as our results
    # now we plot them individually, and combine them
    
    
    name=indir + 'ApJ_only.pdf'
    labels = ['$\\gamma_r$','$R_{\\rm max}$']
    print("Relative maxima are ",np.max(newll),np.nanmax(ll_grid),np.max(newllbrm))
    ute.plot_2darr(newll,labels,name,[Rgammas,np.log10(Rmaxes)],[Rgammas,Rmaxes],\
                    clabel="$p_{\\rm followup}$",crange=[-12,-10])
    
    
    #### Bayesian analysis #####
    newll = ute.make_bayes(newll)
    
    ###### gets contur values ######
    # gets 68% contour interval for ptot
    temp = newll.flatten()
    temp = np.sort(temp)
    temp2 = np.cumsum(temp)
    temp2 /= temp2[-1]
    # cumsum begins summing smallest to largest. Find 1.-0.68
    ilimit = np.where(temp2 < 0.32)[0][-1]
    cl68 = temp[ilimit]
    ilimit = np.where(temp2 < 0.05)[0][-1]
    cl95 = temp[ilimit]
    
    conts=[[[newll,cl68],[newll,cl95]]]
    
    ####### generates plots of old data only #########
    name=indir + 'ApJ_only_bayes.pdf'
    ute.plot_2darr(newll,labels,name,[Rgammas,np.log10(Rmaxes)],[Rgammas,Rmaxes],\
                    clabel="$p_{\\rm ASKAP}$",crange=[0,0.05],conts=conts)
    
    ####### values when picking the best Rmin #######
    name=indir + 'brm_ApJ_only.pdf'
    ute.plot_2darr(newllbrm,labels,name,[Rgammas,np.log10(Rmaxes)],[Rgammas,Rmaxes],\
                    clabel="$p_{\\rm ASKAP}$",crange=None)
    
    newllbrm = ute.make_bayes(newllbrm)
    name=indir + 'brm_ApJ_only_bayes.pdf'
    ute.plot_2darr(newllbrm,labels,name,[Rgammas,np.log10(Rmaxes)],[Rgammas,Rmaxes],\
                    clabel="$p_{\\rm ASKAP}$",crange=[0,0.05])
    
    
    name=indir + 'brms.pdf'
    ute.plot_2darr(np.log10(brms),labels,name,[Rgammas,np.log10(Rmaxes)],[Rgammas,Rmaxes],\
                    clabel="Best Rmin values",crange=[-8,-1.2])
    
    
    
    
    
def interpolate_bestRmin(array,Rmin,Rgamma,Rmax,Rmins,Rgammas,Rmaxs):
    """
    2D linear interpolate of point Rmin, Rmax, Rgamma
    In the Rmin direction, it just finds the most likely value
    """

    i1,i2,x1,x2=get_indices(Rmin,Rmins)
    
    
    j1,j2,y1,y2=get_indices(Rgamma,Rgammas)
    k1,k2,z1,z2=get_indices(Rmax,Rmaxs)
    
    best11 = np.argmax(array[:,j1,k1])
    best12 = np.argmax(array[:,j1,k2])
    best21 = np.argmax(array[:,j2,k1])
    best22 = np.argmax(array[:,j2,k2])
    
    brm = np.log(Rmins[best11]) + np.log(Rmins[best12]) \
         + np.log(Rmins[best21]) + np.log(Rmins[best22])
    brm = np.exp(brm/4.)
    
    val = array[best11,j1,k1]*y1*z1 \
        + array[best21,j2,k1]*y2*z1 \
        + array[best12,j1,k2]*y1*z2 \
        + array[best22,j2,k2]*y2*z2
    return val,brm

    
def interpolate(array,Rmin,Rgamma,Rmax,Rmins,Rgammas,Rmaxs):
    """
    3D linear interpolate of point Rmin, Rmax, Rgamma
    """

    i1,i2,x1,x2=get_indices(Rmin,Rmins)
    j1,j2,y1,y2=get_indices(Rgamma,Rgammas)
    k1,k2,z1,z2=get_indices(Rmax,Rmaxs)
    
    val = array[i1,j1,k1]*x1*y1*z1 + array[i2,j1,k1]*x2*y1*z1 \
        + array[i1,j2,k1]*x1*y2*z1 + array[i2,j2,k1]*x2*y2*z1 \
        + array[i1,j1,k2]*x1*y1*z2 + array[i2,j1,k2]*x2*y1*z2 \
        + array[i1,j2,k2]*x1*y2*z2 + array[i2,j2,k2]*x2*y2*z2
    return val
    
def get_indices(val,array):
    """
    Returns simple linear interpolation values for the array
    Array must be increasing
    
    """
    if array[1] > array[0]:
        if val <= array[0]:
            i1=0
            i2=1
            k1=1.
            k2=0.
        elif val >= array[-1]:
            i1=array.size-2
            i2=array.size-1
            k1=0.
            k2=1.
        else:
            i1 = np.where(array < val)[0][-1]
            i2 = i1+1
            dval = array[i2]-array[i1]
            k2 = (val - array[i1])/dval
            k1 = 1.-k2
    else: #array in decreasing order
        if val >= array[0]:
            i1=0
            i2=1
            k1=1.
            k2=0.
        elif val <= array[-1]:
            i1=array.size-2
            i2=array.size-1
            k1=0.
            k2=1.
        else:
            i1 = np.where(array > val)[0][-1]
            i2 = i1+1
            darr = array[i1]-array[i2]
            k1 = (val - array[i2])/darr
            k2 = 1.-k1
    return i1,i2,k1,k2
    
    
        


main()



