"""
This plots the output of fit_repeating_distributions.py

It takes the output npz files from each set, and produces projected 2D distributions for each.

It can do this either for the full 3D parameter scan over Rgamma, Rmin, Rmax space,
or else the 2D scan over Rgamma, Rmax that converges for Rmin


"""

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

import matplotlib
#matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)




def main():
    # simple switching between cases
    converge = True
    Nsets=5
    if converge:
        make_converge_plots(Nsets=Nsets)
    else:
        make_3D_plots(Nsets=Nsets)


def make_converge_plots(Nsets=5):
    """
    Plots output from fit_repeating_distributions "converge_state" routine
    """
    
    dims=['$\\gamma_{r}$','$R_{\\rm max}$']
    pdims=['$p(R_{\\rm min})$','$p(R_{\\rm max})$','$p(\\gamma_{r})$']
    names=['Rmin','Rmax','Rgamma']
    string=['Ptot','Pn','Pdm']
    pstrings=['$P_{\\rm tot}$','$P_{N}$','$P_{\\rm DM}$']
    
    # 'fn' stands for 'fixed norm', i.e. same normalisation between different parameter sets
    opdir = 'Posteriors/'
    
    setnames = ['$\\epsilon_C: {\\rm min}\\, \\alpha$','$\\epsilon_z:$ Best fit','${\\rm min}\\, E_{\\rm max}$',
        '${\\rm max} \\,\\gamma$','${\\rm min} \\, \\sigma_{\\rm host}$']
    
    
    # keeps normalisations between sets
    norm1ds=[[None,None,None],[None,None,None],[None,None,None]]
    norm2ds=[[None,None,None],[None,None,None],[None,None,None]]
    
    for i in np.arange(Nsets):
        if i==1:
            exit()
        # rgs: repeat grids, to be re-used
        # all_bin_lp: log probability, best guess of some combination
        # all_bin_lpn: summed log10 Poisson probability of number of reps in each bin
        # all_bin_lpdm: summed log10 p(DM) summed over each bin
        # lPN: log10 Poisson probability of summed reps over all bins
        # scaled_tr: expected number of repeaters, after scaling
        # lskp, lrkp: log10 ks test results for singles and repeater distributions over declination
        # ltdm_ks,ltdm_kr ks test values over entire DM range
        # lprod_bin_dm_kr,lprod_bin_dm_ks ks test values over individual bins, product thereof
        # skp, rkp: ks stats for the declination distribution
        
        infile='Rfitting/mc_converge_set_'+str(i)+'_'+'_output.npz'
        data=np.load(infile)
        
        
        lps=data['arr_0']
        lns=data['arr_1']
        ldms=data['arr_2']
        
        lpNs=data['arr_3']
        NRs=data['arr_4']
        
        Rmins=data['arr_5']
        Rmaxes=data['arr_6']
        Rgammas=data['arr_7']
        
        
        lskps=data['arr_8']
        lrkps=data['arr_9']
        
        ltdm_kss=data['arr_10']
        ltdm_krs=data['arr_11']
        lprod_bin_dm_krs=data['arr_12']
        lprod_bin_dm_kss=data['arr_13']
        
        Rstar = data['arr_14']
        
        # mC data
        mcrs = data['arr_15']
        MChs = data['arr_16']
        shortlPs = data['arr_17']
        medlPs = data['arr_18']
        longlPs = data['arr_19']
        
        alabels = ["$P_{\\rm tot}$",\
            "$\\sum_{\\delta} \\log_{10} p(N_{\\delta})$",\
            "$\\sum_{\\delta} \\log_{10} p({\\rm DM}_{\\delta})$",\
            "$\\log_{10} p(N_{\\rm tot})$",\
            "$N_{\\rm tot}$",\
            "$p(\\delta_s)$",\
            "$p_{\\delta_r}$",\
            "$p_{\\rm ks} ({\\rm DM}_s)$",\
            "$p_{\\rm DM}$",\
            "$\\sum_{\\delta} \\log_{10} p_{\\rm ks} ({\\rm DM}_s)$",\
            "$p_{\\rm ks} ({\\rm DM}_r)$",\
            "$R_{\\rm min}$",\
            "$p_{\\rm bursts}$",\
            "$p_{\\rm bursts}$",\
            "$p_{\\rm bursts}$",\
            "$P_{\\rm tot}$"]
            #"$\\log_{10} {\\ell} ({\\rm N}_{\\rm reps})$",\
            #"$\\log_{10} {\\ell} ({\\rm N}_{\\rm reps})$",\
            #"$\\log_{10} {\\ell} ({\\rm N}_{\\rm reps})$"\
            #]
        
        alabels[2] = "$\\log_{10} \\ell_{\\rm DM}$"
        alabels[6] = "${\\rm KS}_{\\delta}$"
        #alabels[0] = alabels[2] + " + " +alabels[6]
        
        anames = ['Ptot','sumpN','sumpDM','pN','N','ksdec','krdec','ksdm','krdm','sumksdm',\
            'sumkrdm','Rmin','MCshort','MCmed','MClong','anyN_Ptot']
        Rmins = np.log10(Rmins)
        #NRs = np.log10(NRs)
        
        #transformations prior to plotting
        prod_bin_dm_krs = 10**lprod_bin_dm_krs
        rawp=prod_bin_dm_krs.flatten()
        truep = correct_sixprod(rawp,N=6)
        true_prod_bin_dm_krs = truep.reshape(lprod_bin_dm_krs.shape)
        
        
        #copyldms = np.copy(true_prod_bin_dm_krs)
        
        # constructs dec, Nburst, dm product
        lps = lrkps + longlPs + ltdm_krs
        
        
        ##### we get rid of the kdm penalty for low DMs, in case this is the problem #####
        #for iRmax,Rmax in enumerate(Rmaxes):
        #    ibest = np.argmax(ltdm_krs[:,iRmax])
        #    ltdm_krs[ibest:,iRmax] = ltdm_krs[ibest,iRmax]
        copylps = lrkps + longlPs + ltdm_krs
        
        theshape = lps.shape
        bad = np.where(NRs.flatten() > 18)[0]
        lps = lps.flatten()
        lps[bad]=-90
        lps = lps.reshape(theshape)
        
        copylps = copylps.flatten()
        copylps[bad]=-90
        copylps = copylps.reshape(theshape)
        
        lps = make_bayes(lps)
        copylps = make_bayes(copylps)
        bshortlPs = make_bayes(shortlPs)
        bmedlPs = make_bayes(medlPs)
        blonglPs = make_bayes(longlPs)
        
        
        # gets 68% contour interval for ptot
        temp = lps.flatten()
        temp = np.sort(temp)
        temp2 = np.cumsum(temp)
        
        temp2 /= temp2[-1]
        # cumsum begins summing smallest to largest. Find 1.-0.68
        ilimit = np.where(temp2 < 0.32)[0][-1]
        l68 = temp[ilimit]
        ilimit = np.where(temp2 < 0.05)[0][-1]
        l95 = temp[ilimit]
        
        # gets 68% contour interval for ptot
        temp = copylps.flatten()
        temp = np.sort(temp)
        temp2 = np.cumsum(temp)
        
        temp2 /= temp2[-1]
        # cumsum begins summing smallest to largest. Find 1.-0.68
        ilimit = np.where(temp2 < 0.32)[0][-1]
        cl68 = temp[ilimit]
        ilimit = np.where(temp2 < 0.05)[0][-1]
        cl95 = temp[ilimit]
        
        
        alist = [lps,lns,ldms,lpNs,NRs,lskps,10**lrkps,ltdm_kss,10**ltdm_krs,\
            lprod_bin_dm_kss,true_prod_bin_dm_krs,Rmins,shortlPs,medlPs,\
            longlPs,copylps]
        
        cranges = [None]*len(alist)
        #cranges[0] = [-2]
        cranges[2] = [-2]
        cranges[3] = [-2,0]
        cranges[4] = [16,20]
        cranges[5] = [-2]
        cranges[6] = [0,1]
        cranges[7] = [-10]
        cranges[8] = [0,1]
        cranges[9] = [-2]
        cranges[10] = [0,1]
        cranges[11] = [-8,-1]
        cranges[12] = [-2]
        cranges[13] = [-2]
        cranges[14] = [-2]
        
        Rgammas = Rgammas - 0.001
        
        sx=[-0.25,0.25,3,3]
        sy=[-1.3,-3.0,-3.0,-2.0]
        sm=['$a$','$b$','$c$','$d$']
        scatter=[sx,sy,sm]
        
        for j,arr in enumerate(alist):
            print(j,"th plot is ",anames[j])
            labels = ['$\\gamma_r$','$R_{\\rm max}$']
            name=opdir+'set_'+str(i)+'_'+anames[j]+'.pdf'
            if anames[j]=='Rmin':
                plot_2darr(arr,labels,name,[Rgammas,np.log10(Rmaxes)],[Rgammas,Rmaxes],\
                    clabel=alabels[j],crange=cranges[j],conts=[Rmins,np.log10(0.05)],\
                    Nconts=[NRs,21],RMlim=np.log10(0.5),scatter=scatter,Allowed=True)
            elif anames[j] == 'Ptot' or anames[j] == 'anyN_Ptot':
                plot_2darr(arr,labels,name,[Rgammas,np.log10(Rmaxes)],[Rgammas,Rmaxes],\
                    clabel=alabels[j],crange=cranges[j],\
                    conts=[[[arr,l68],[lps,l95]]])
                    
                    #RMlim=np.log10(0.5)) #Nconts=[NRs,21],[Rmins,np.log10(0.05)],
            else:
                plot_2darr(arr,labels,name,[Rgammas,np.log10(Rmaxes)],[Rgammas,Rmaxes],\
                    clabel=alabels[j],crange=cranges[j],scatter=scatter)
            #conts=[Rmins,-2],RMlim=0.)
        
        # generates some 1D slices
        plt.figure()
        plt.plot(Rgammas,Rmins[:,1],label='$R_{\\rm max} = 0.316$')
        plt.plot(Rgammas,Rmins[:,2],label='$R_{\\rm max} = 1$')
        plt.plot(Rgammas,Rmins[:,3],label='$R_{\\rm max} = 3.16$')
        plt.xlabel("$\\gamma_r$")
        plt.ylabel("$R_{\\rm min}$")
        plt.ylim(-3,-1)
        plt.legend()
        plt.tight_layout()
        plt.savefig('Posteriors/set_'+str(i)+'_example_rmin.pdf')
        plt.close()
        
        
        # now does posterior for lps
        
        
        
        
        # projects onto 1D
        #lps -= np.nanmax(lps)
        #lps = 10**lps
        pgamma = np.sum(lps,axis=1)
        prmax = np.sum(lps,axis=0)
        
        
        
        from zdm import analyze_cube as ac
        pglims = ac.extract_limits(Rgammas,pgamma,0.68,dumb=True,interp=True)
        rmlims = ac.extract_limits(Rmaxes,prmax,0.68,dumb=True,interp=True)
        
        ipeak = np.argmax(pgamma)
        gpeak = Rgammas[ipeak]
        ipeak = np.argmax(prmax)
        mpeak = Rmaxes[ipeak]
        
        
        #plt.figure()
        #cpgamma = np.cumsum(pgamma)
        #cpgamma /= cpgamma[-1]
        #pgamma /= np.max(pgamma)
        #plt.plot(Rgammas,pgamma)
        #plt.plot(Rgammas,cpgamma)
        #plt.tight_layout()
        #plt.savefig('rgamma_limit.pdf')
        #plt.close()
        
        #plt.figure()
        #plt.xlabel('Rmax')
        #plt.ylabel('Prmax')
        #plt.xscale('log')
        #plt.plot(Rmaxes,prmax)
        #plt.show()
        #plt.close()
        print("Set ",i," Found gamma limits ",pglims[0:2]," peak ",gpeak)
        print("Set ",i," Found rmax limits ",rmlims[0:2]," peak ",mpeak)
        

def make_bayes(arr):
    arr -= np.max(arr)
    arr = 10**arr
    arr /= np.sum(arr)
    return arr

def plot_2darr(arr,labels,savename,ranges,rlabels,clabel=None,crange=None,\
    conts=None,Nconts=None,RMlim=None,scatter=None,Allowed=False):
    """
    does 2D plot
    
    array is the 2D array to plot
    labels are the x and y axis labels [ylabel,xlabel]
    Here, savename is the output file
    Ranges are the [xvals,yvals]
    Rlabels are [xtics,ytics]
    
    """
    ratio=np.abs((ranges[0][1]-ranges[0][0])/(ranges[0][2]-ranges[0][1]))
    if ratio > 1.01 or ratio < 0.99:
        log0=True
    else:
        log0=False
    
    ratio=np.abs((ranges[1][1]-ranges[1][0])/(ranges[1][2]-ranges[1][1]))
    if ratio > 1.01 or ratio < 0.99:
        log1=True
    else:
        log1=False
    
    dr1 = ranges[1][1]-ranges[1][0]
    dr0 = ranges[0][1]-ranges[0][0]
    
    aspect = (ranges[0].size/ranges[1].size)
    
    extent = [ranges[1][0]-dr1/2., ranges[1][-1]+dr1/2.,\
            ranges[0][0]-dr0/2.,ranges[0][-1]+dr0/2.]
    
    im = plt.imshow(arr,origin='lower',aspect=aspect,extent=extent)
    ax=plt.gca()
    
    # sets x and y ticks to bin centres
    ticks = rlabels[1].astype('str')
    for i,tic in enumerate(ticks):
        ticks[i]=tic[:5]
    ax.set_xticks(ranges[1][1::2])
    ax.set_xticklabels(ticks[1::2])
    plt.xticks(rotation = 90) 
    
    ticks = rlabels[0].astype('str')
    for i,tic in enumerate(ticks):
        ticks[i]=str(rlabels[0][i])[0:4]
    ax.set_yticks(ranges[0][::4])
    ax.set_yticklabels(ticks[::4])
    
    plt.xlabel(labels[1])
    plt.ylabel(labels[0])
    
    #cax = fig.add_axes([ax.get_position().x1+0.03,ax.get_position().y0,0.02,ax.get_position().height])
    #cbar = plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    cbar = plt.colorbar(shrink=0.55)
    if clabel is not None:
        cbar.set_label(clabel)
    if crange is not None:
        if len(crange) == 2:
            plt.clim(crange[0],crange[1])
        else:
            themax=np.nanmax(arr)
            plt.clim(crange+themax,themax)
    
    
    if conts is not None:
        if len(conts) == 2:
            
            ax = plt.gca()
            cs=ax.contour(conts[0],levels=[conts[1]],origin='lower',colors="black",\
                linestyles=[':'],linewidths=[3],extent=extent)
        else:
            colors=["red","white","black"]
            styles=[':','-.','--','-']
            for k,cont in enumerate(conts[0]):
                print("Doing multiple conts")
                cs=ax.contour(cont[0],levels=[cont[1]],origin='lower',colors=colors[k],\
                    linestyles=styles[k],linewidths=[3],extent=extent)
    if Nconts is not None:
        ax = plt.gca()
        cs=ax.contour(Nconts[0],levels=[Nconts[1]],origin='lower',colors="orange",\
            linestyles=['-.'],linewidths=[3],extent=extent)
    
    if Allowed:
        plt.text(1,-2.5,'Allowed')
    
    if RMlim is not None:
        plt.plot([RMlim,RMlim],[extent[2],extent[3]],linestyle='--',color='white',linewidth=3)
    
    if scatter is not None:
        sx=scatter[0]
        sy=scatter[1]
        sm=scatter[2]
        for i, m in enumerate(sm):
            #ax.plot((i+1)*[i,i+1],marker=m,lw=0)
            plt.plot(sx[i],sy[i],marker=m,color='red',linestyle="")
    
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


def make_3D_plots(Nsets=5):
    dims=['$R_{\\rm min}$','$R_{\\rm max}$','$\\gamma_{r}$']
    pdims=['$p(R_{\\rm min})$','$p(R_{\\rm max})$','$p(\\gamma_{r})$']
    names=['Rmin','Rmax','Rgamma']
    string=['Ptot','Pn','Pdm']
    pstrings=['$P_{\\rm tot}$','$P_{N}$','$P_{\\rm DM}$']
    
    # 'fn' stands for 'fixed norm', i.e. same normalisation between different parameter sets
    opdir = 'Posteriors/fn_'
    
    setnames = ['$\\epsilon_C: {\\rm min}\\, \\alpha$','$\\epsilon_z:$ Best fit','${\\rm min}\\, E_{\\rm max}$',
        '${\\rm max} \\,\\gamma$','${\\rm min} \\, \\sigma_{\\rm host}$']
    
    picklefile='Posteriors/fixednorm_all_results.pkl'
    
    if os.path.exists(picklefile):
        with open(picklefile, 'rb') as infile:
            lists=pickle.load(infile)
            mlists=pickle.load(infile)
        load=True
    else:
        load=False
    
    # keeps normalisations between sets
    norm1ds=[[None,None,None],[None,None,None],[None,None,None]]
    norm2ds=[[None,None,None],[None,None,None],[None,None,None]]
    
    for i in np.arange(Nsets):
        if load:
            continue
        # this is the savez command
        #np.savez(outfile,lps,lns,ldms,lpNs,Nrs,Rmins,Rmaxes,Rgammas,lskps,lrkps,ltdm_kss,\
        #            ltdm_krs,lprod_bin_dm_krs,lprod_bin_dm_kss)
        infile='Rfitting/set_'+str(i)+'_iteration_output.npz'
        data=np.load(infile)
        
        lps=data['arr_0']
        lns=data['arr_1']
        ldms=data['arr_2']
        
        lnNs=data['arr_3']
        NRs=data['arr_4']
        
        Rmins=data['arr_5']
        Rmaxes=data['arr_6']
        Rgammas=data['arr_7']
        
        
        lskps=data['arr_8']
        lrkps=data['arr_9']
        
        ltdm_kss=data['arr_10']
        ltdm_krs=data['arr_11']
        lprod_bin_dm_krs=data['arr_12']
        lprod_bin_dm_kss=data['arr_13']
        
        params=[Rmins,Rmaxes,Rgammas]
        pvals=[np.log10(Rmins),np.log10(Rmaxes),Rgammas]
        
        
        if i==0:
            list1=np.zeros([3,Nsets,Rmins.size])
            list2=np.zeros([3,Nsets,Rmaxes.size])
            list3=np.zeros([3,Nsets,Rgammas.size])
            mlist1=np.zeros([3,Nsets,Rmins.size])
            mlist2=np.zeros([3,Nsets,Rmaxes.size])
            mlist3=np.zeros([3,Nsets,Rgammas.size])
            
            lists=[list1,list2,list3]
            mlists=[mlist1,mlist2,mlist3]
        
        ###### forces data to conform to limits from strong FRBs ######
        force = True
        if force:
            
            # produce a copy with limits
            modlps = np.copy(lps)
            modldms = np.copy(ldms)
            modlns = np.copy(lns)
            
            # define a "small" value
            small = np.min(modlps) - 10. # this should be "sufficiently small"
            
            Rminmax = 1.1e-2 # 171020
            setzero = np.where(Rmins > Rminmax)[0]
            modlps[setzero,:,:] = small
            modlns[setzero,:,:] = small
            modldms[setzero,:,:] = small
            
            Rmaxmin = 4. # 121102
            setzero = np.where(Rmaxes < Rmaxmin)[0]
            modlps[:,setzero,:] = small
            modlns[:,setzero,:] = small
            modldms[:,setzero,:] = small
            
            modified_data=[modlps,modlns,modldms]
        
        # this loops over the three variables
        for j in np.arange(3):
            
            labels = dims[:j] + dims[j+1:]
            ranges = pvals[:j] + pvals[j+1:]
            rlabels = params[:j] + params[j+1:]
            
            # loops over the the three probabilities
            for k,data in enumerate([lps,lns,ldms]):
                
                ######## plots for unlimited data ##########
                arr,norm2d=compress_2d(data,j,norm2ds[j][k])
                norm2ds[j][k]=norm2d
                
                name=opdir+'set_'+str(i)+'_integrated_'+string[k]+'_'+names[j]+'.pdf'
                plot_2darr(arr,labels,name,ranges,rlabels)
                
                # passing the norms results in keeping the same normalisation as a global over all Nsets
                arr1,norm1d=compress_1d(data,j,norm1ds[j][k])
                norm1ds[j][k]=norm1d
                
                if force:
                    ######## plots for limited data ##########
                    # gets the correctt data
                    moddata = modified_data[k]
                    
                    ######## plots for unlimited data ##########
                    modarr,norm2d=compress_2d(moddata,j,norm2ds[j][k])
                    
                    name=opdir+'set_'+str(i)+'_mod_integrated_'+string[k]+'_'+names[j]+'.pdf'
                    plot_2darr(modarr,labels,name,ranges,rlabels)
                    
                    modarr1,tempnorm=compress_1d(moddata,j,norm1ds[j][k])
                
                
                ####### plots both sets at once, comparing them ######
                
                name=opdir+'set_'+str(i)+'_1D_'+string[k]+'_'+names[j]+'.pdf'
                plot_1darr(arr1,dims[j],name,pvals[j],rlabels,other=modarr1)
                
                lists[j][k,i,:]=arr1
                if force:
                    mlists[j][k,i,:]=modarr1
        
    if not load:
        with open(picklefile, 'wb') as output:
            pickle.dump(lists, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(mlists, output, pickle.HIGHEST_PROTOCOL)
    
    # now plot summaries over all parameter sets
    for k,pstring in enumerate(pstrings):
        
        # loops over the three variables
        for j in np.arange(3):
            # pvals should be inherited
            outfile=opdir+'all_'+names[j]+'_'+string[k]+'.pdf'
            plot_1darrs(pvals[j],lists[j][k,:,:],mlists[j][k,:,:],outfile,dims[j],pdims[j],setnames)
            
            if force:
                outfile=opdir+'all_mo_'+names[j]+'_'+string[k]+'.pdf'
                plot_1darrs(pvals[j],mlists[j][k,:,:],None,outfile,dims[j],pdims[j],setnames)
            
            outfile=opdir+'all_um_'+names[j]+'_'+string[k]+'.pdf'
            plot_1darrs(pvals[j],lists[j][k,:,:],None,outfile,dims[j],pdims[j],setnames)
            
            for iset in np.arange(5):
                lists[j][k,iset,:] /= np.sum(lists[j][k,iset,:])
                if force:
                    mlists[j][k,iset,:] /= np.sum(mlists[j][k,iset,:])
            
            # repeats, but normalises the data
            outfile=opdir+'nall_'+names[j]+'_'+string[k]+'.pdf'
            plot_1darrs(pvals[j],lists[j][k,:,:],mlists[j][k,:,:],outfile,dims[j],pdims[j],setnames)
            
            if force:
                outfile=opdir+'nall_mo_'+names[j]+'_'+string[k]+'.pdf'
                plot_1darrs(pvals[j],mlists[j][k,:,:],None,outfile,dims[j],pdims[j],setnames)
                
            outfile=opdir+'nall_um_'+names[j]+'_'+string[k]+'.pdf'
            plot_1darrs(pvals[j],lists[j][k,:,:],None,outfile,dims[j],pdims[j],setnames)
            
def plot_1darrs(xvals,yvals,myvals,outfile,xlabel,ylabel,labels):
    """
    Plots all the results for some parameter
    """           
    nres,nx=np.shape(yvals)
    
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in np.arange(nres):
        plt.plot(xvals,yvals[i,:],linestyle='-',label=labels[i])
        if myvals is not None:
            plt.plot(xvals,myvals[i,:],linestyle='--',color=plt.gca().lines[-1].get_color())
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    
    
    
def plot_1darr(arr,label,savename,xvals,rlabels,other=None):
    """
    does 1D plot
    """
    ratio=np.abs((xvals[1]-xvals[0])/(xvals[2]-xvals[1]))
    if ratio > 1.01 or ratio < 0.99:
        log0=True
    else:
        log0=False
    
    dx = xvals[1]-xvals[0]
    
    plt.figure()
    plt.plot(xvals,arr)
    if other is not None:
        plt.plot(xvals,other,linestyle='--')
    ax=plt.gca()
    if log0:
        plt.xscale('log')
    # sets x and y ticks to bin centres
    #ticks = rlabels[1].astype('str')
    #for i,tic in enumerate(ticks):
    #    ticks[i]=tic[:5]
    #ax.set_xticks(ranges[1])
    #ax.set_xticklabels(ticks)
    #plt.xticks(rotation = 90) 
    
    #ticks = rlabels[0].astype('str')
    #for i,tic in enumerate(ticks):
    #    if rlabels[0][i] < 0.01:
    #        ticks[i]='{:1.3e}'.format(rlabels[0][i])
    #    else:
    #        ticks[i]=str(rlabels[0][i])[0:5]
    #ax.set_yticks(ranges[0])
    #ax.set_yticklabels(ticks)
    
    plt.xlabel(label)
    plt.ylabel('p('+label+')')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def compress_1d(lps,idim,norm=None):
    """
    Integrates prob dist over all but idim
    """
    themax = np.max(lps)
    temp = lps - themax
    temp = 10**temp
    
    n1,n2,n3 = lps.shape
    if idim == 0:
        temp = np.sum(temp,axis=2)
        temp = np.sum(temp,axis=1)
    elif idim == 1:
        temp = np.sum(temp,axis=2)
        temp = np.sum(temp,axis=0)
    elif idim == 2:
        temp = np.sum(temp,axis=1)
        temp = np.sum(temp,axis=0)
    else:
        print("Invalid value of idim")
    if norm is not None:
        temp /= norm
    else:
        norm=np.sum(temp)
        temp /= norm
    
    return temp,norm

def old_plot_2darr(arr,labels,savename,ranges,rlabels,clabel=None,crange=None):
    """
    does 2D plot
    
    array is the 2D array to plot
    labels are the x and y axis labels [ylabel,xlabel]
    Here, savename is the output file
    Ranges are the [xvals,yvals]
    Rlabels are [xtics,ytics]
    
    """
    ratio=np.abs((ranges[0][1]-ranges[0][0])/(ranges[0][2]-ranges[0][1]))
    if ratio > 1.01 or ratio < 0.99:
        log0=True
    else:
        log0=False
    
    ratio=np.abs((ranges[1][1]-ranges[1][0])/(ranges[1][2]-ranges[1][1]))
    if ratio > 1.01 or ratio < 0.99:
        log1=True
    else:
        log1=False
    
    dr1 = ranges[1][1]-ranges[1][0]
    dr0 = ranges[0][1]-ranges[0][0]
    
    aspect = (ranges[0].size/ranges[1].size)
    plt.figure()
    plt.imshow(arr,origin='lower',aspect=aspect,extent=[ranges[1][0]-dr1/2.,
        ranges[1][-1]+dr1/2.,ranges[0][0]-dr0/2.,ranges[0][-1]+dr0/2.])
    ax=plt.gca()
    
    # sets x and y ticks to bin centres
    ticks = rlabels[1].astype('str')
    for i,tic in enumerate(ticks):
        ticks[i]=tic[:5]
    ax.set_xticks(ranges[1][::4])
    ax.set_xticklabels(ticks[::4])
    plt.xticks(rotation = 90) 
    
    ticks = rlabels[0].astype('str')
    for i,tic in enumerate(ticks):
        if rlabels[0][i] < 0.01:
            ticks[i]='{:1.3e}'.format(rlabels[0][i])
        else:
            ticks[i]=str(rlabels[0][i])[0:5]
    ax.set_yticks(ranges[0][::4])
    ax.set_yticklabels(ticks[::4])
    
    plt.xlabel(labels[1])
    plt.ylabel(labels[0])
    cbar = plt.colorbar()
    if clabel is not None:
        print(clabel)
        cbar.set_label(clabel)
    if crange is not None:
        themax=np.nanmax(arr)
        plt.clim(crange+themax,themax)
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

 
def compress_2d(lps,idim,norm=None):
    """
    This routine creates a 2D probability distribution such that
    the sum over all dimensions is unity
    """
    
    n1,n2,n3=lps.shape
    
    
    # scales up the probabilities to something workable
    themax = np.max(lps)
    temp = lps - themax
    #for i in np.arange(n3):
    #    print("MEOW MEOW MEOW MEOW")
    #    print(lps[:,:,i])
    #exit()
    
    ps = 10**temp
    new_arr = np.sum(ps,axis=idim)
    
    if norm is not None:
        new_arr /= norm
    else:
        norm = np.sum(new_arr)
        new_arr /= norm
    return new_arr,norm

def correct_sixprod(pvals,N=6):
    """
    Corrects a product of N pvalues to a single true pvalue
    """
    from scipy.integrate import quad
    truep = np.zeros(pvals.shape)
    for i,p in enumerate(pvals):
        truep[i] = 1.-quad(fn,p,1.,args=(N))[0]
        
    return truep
def fn(z,n):
    """
    Probability density function for product of n
        uniform distributions
    z: between 0 and 1
    n: number of functions
    """
    fn = (-1)**(n-1)  * np.log(z)**(n-1) / np.math.factorial(n-1)
    return fn

def test_correction(N=6,M=1000,outfile='test_p_correction.pdf'):

    nums = np.random.uniform(size=[N,M])
    
    rawp = np.prod(nums,axis=0)

    truep=correct_sixprod(rawp,N=N)
    plt.figure()
    plt.hist(truep,bins=np.linspace(0,1,11))
    plt.xlabel('corrected p')
    plt.ylabel('N(p)')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
  
    
main()
