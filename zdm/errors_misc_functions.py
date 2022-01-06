import numpy as np
import matplotlib.pyplot as plt
import zdm

import pickle

def get_sc_grid(grid,nsnr:int,snrs:np.ndarray, calc_psz:bool=False):
    """Generate an s,DM grid

    Args:
        grid ([type]): [description]
        nsnr (int):  Length of the snrs array.  Seems superfluous..
        snrs (np.ndarray): [description]
        calc_psz(bool, optional):
            If True, calcualte p(s,z) instead!

    Returns:
        tuple: Two np.ndarray's.  
            One is p(s) and the other is p(s,DM) or p(s,z)
    """
    

    # holds cumulative and differential source counts
    cpsnrs=np.zeros([nsnr])
    psnrs=np.zeros([nsnr-1])
    
    if not calc_psz:
        nother=grid.dmvals.size
    else:
        # holds DM-dependent source counts
        nother=grid.zvals.size

    # Generate the grids
    cpgrid=np.zeros([nsnr,nother])
    pgrid=np.zeros([nsnr-1,nother])
    
    backup1=np.copy(grid.thresholds)
    
    # modifies grid to simplify beamshape
    grid.beam_b=np.array([grid.beam_b[-1]])
    grid.beam_o=np.array([grid.beam_o[-1]])
    grid.b_fractions=None
    
    for i,s in enumerate(snrs):
        
        grid.thresholds=backup1*s
        grid.calc_pdv()
        grid.calc_rates()
        rates=grid.rates
        if calc_psz:
            cpgrid[i,:]=np.sum(rates,axis=1)
        else:
            cpgrid[i,:]=np.sum(rates,axis=0)
        cpsnrs[i]=np.sum(cpgrid[i,:])
    
    # the last one contains cumulative values
    for i,s in enumerate(snrs):
        if i==0:
            continue
        psnrs[i-1]=cpsnrs[i-1]-cpsnrs[i]
        pgrid[i-1,:]=cpgrid[i-1,:]-cpgrid[i,:]
    return psnrs, pgrid
    
def error_get_source_counts(grid,errorsets,Emin,plot=None,Slabel=None,load=False,tag=None):
    """
    Calculates the source-counts function for a given grid
    It does this in terms of p(SNR)dSNR
    """
    # this is closely related to the likelihood for observing a given psnr!
    
    
    
    nsnr=71
    snrmin=0.001
    snrmax=1000.
    
    snrs=np.logspace(0,2,nsnr) # histogram array of values for s=SNR/SNR_th
    
    if load:
        with open('Pickle/error_snrs'+tag+'.pkl', 'rb') as infile:
            psnrs=pickle.load(infile)
            Epsnrs=pickle.load(infile)
            err_psnrs=pickle.load(infile)
        
    else:
        psnrs,dmpsnrs=get_sc_grid(grid,nsnr,snrs)
        Epsnrs,Edmpsnrs=get_sc_grid(Emin,nsnr,snrs)
        
        err_psnrs=[]
        err_dmpsnrs=[]
        for eset in errorsets:
            err_psnr,err_dmpsnr=get_sc_grid(eset,nsnr,snrs)
            err_psnrs.append(err_psnr)
            err_dmpsnrs.append(err_dmpsnr)
        with open('Pickle/error_snrs'+tag+'.pkl', 'wb') as output:
            pickle.dump(psnrs, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(Epsnrs, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(err_psnrs, output, pickle.HIGHEST_PROTOCOL)
    
    mod=1.5
    snrs=snrs[:-1]
    imid=int((nsnr+1)/2)
    xmid=snrs[imid]
    ymid=psnrs[imid]
    slopes=np.linspace(1.3,1.7,5)
    ys=[]
    for i,s in enumerate(slopes):
        ys.append(ymid*xmid**s*snrs**-s)
    
    # scales other values to pass through xmin,ymid
    Epsnrs *= ymid/Epsnrs[imid]
    
    for i,err in enumerate(err_psnrs):
        err *= ymid/err[imid]
    
    
    if plot is not None:
        # use the following point to scale standard plots to a left-hand min of 1,1
        fixpoint=ys[0][0]*snrs[0]**mod
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1,3)
        plt.xlim(0.5,100)
        
        plt.xlabel('$s=\\frac{\\rm SNR}{\\rm SNR_{\\rm th}}$')
        plt.ylabel('$p_s(s) s^{1.5} d\\,\\log(s)$ [a.u.]')
        
        
        plt.plot(snrs,psnrs*snrs**mod/fixpoint,label='Best fit ('+Slabel+')',color='orange',linewidth=3,zorder=3) # this is in relative units
        plt.plot(snrs,Epsnrs*snrs**mod/fixpoint,label='Emin',color='red',linestyle=':',linewidth=2,zorder=5) # this is in relative units
        
        for i,err in enumerate(err_psnrs):
            #err *= ymid/err[imid]
            if i==0:
                plt.plot(snrs,err*snrs**mod/fixpoint,label='90% C.L.',color='grey',linewidth=1,zorder=2) # this is in relative units
            else:
                plt.plot(snrs,err*snrs**mod/fixpoint,color='grey',linewidth=1,zorder=2)
            
        temp=[1.02,1.3,1.6,2.05,2.5]
        for i,s in enumerate(slopes):
            plt.plot(snrs,ys[i]*snrs**mod/fixpoint,linestyle='--',zorder=0,color='black')#,label='slope='+str(s)[0:3])
            plt.text(0.65,temp[i],'$s^{-'+str(s)[0:3]+'}$',zorder=0)#,rotation=30-i*15)	
        
        
        #for i,g in enumerate(errorsets):
        #	psnrs,dmpsnrs=get_sc_grid(grid,nsnr,snrs)
        
        ax=plt.gca()
        #labels = [item.get_text() for item in ax.get_yticklabels()]
        #print("Labels are ",labels)
        #labels[0] = '1'
        #labels[1] = '2'
        #labels[2] = '3'
        #ax.set_yticklabels(labels)
        ax.set_yticks([1,2,3])
        ax.set_yticklabels(['1','2','3'])
        plt.legend(fontsize=12)#,loc=[6,8])
        plt.tight_layout()
        plt.savefig(plot)
        plt.close()
    #return snrs,psnrs,dmpsnrs
    

def get_p_zgdm(DMs,grid,dmvals,zvals):
    """ Calcuates the probability of redshift given a DM
    We already have our grid of observed DM values.
    Just take slices! """
        
    
    priors=np.zeros([DMs.size,zvals.size])
    for i,dm in enumerate(DMs):
        DM2=np.where(dmvals > dm)[0][0]
        DM1=DM2-1
        kDM=(dm-dmvals[DM1])/(dmvals[DM2]-dmvals[DM1])
        priors[i,:]=kDM*grid[:,DM2]+(1.-kDM)*grid[:,DM1]
        priors[i,:] /= np.sum(priors[i,:])
    return priors

def error_get_zgdm_priors(survey,grid,errorgrids,mingrid,dirname,basename,dmvals,zvals):
    """ Plots priors as a function of redshift for each FRB in the survey
    Likely outdated, should use the likelihoods function.
    Does this for a list of error grids also
    """
    DMs=survey.DMEGs
    priors=get_p_zgdm(DMs,grid,dmvals,zvals)
    epriors=[]
    for i,e in enumerate(errorgrids):
        eprior=get_p_zgdm(DMs,errorgrids[i],dmvals,zvals)
        epriors.append(eprior)
    
    minpriors=get_p_zgdm(DMs,mingrid,dmvals,zvals)
    
    plt.figure()
    plt.xlabel('$z$')
    plt.ylabel('$p(z|{\\rm DM})$')
    for i,dm in enumerate(survey.DMs):
        if i<10:
            style="-"
        else:
            style=":"
        
        plt.plot(zvals,priors[i,:],label=str(dm),linestyle=style)
        
    plt.xlim(0,0.5)
    plt.legend(fontsize=8,ncol=2)
    plt.tight_layout()
    plt.savefig(dirname+basename)
    plt.close()
    
    
    for i,dm in enumerate(survey.DMs):
        plt.figure()
        plt.xlabel('$z$')
        plt.ylabel('$p(z|{\\rm DM})$ [a.u.]')
        
        plt.plot(zvals,priors[i,:],label='Standard ' +str(dm),linestyle='-')
        plt.plot(zvals,minpriors[i,:],label='with Emin included',linestyle=':')
        
        for k,ep in enumerate(epriors):
            if k==0:
                plt.plot(zvals,ep[i,:],linestyle=style,color='grey',label='90\% parameter errors')
            else:
                plt.plot(zvals,ep[i,:],linestyle=style,color='grey')
        xmax=(int(dm/100))/10.-0.05
        plt.ylim(0,0.1)
        plt.xlim(0,xmax)
        plt.legend(fontsize=8,ncol=2)
        plt.tight_layout()
        plt.savefig(dirname+str(int(dm))+'_'+basename)
        plt.close()

def compare_z_fits(surveys,bestrates,errorrates,n0rates,Eminrates,zvals,dmvals,outdir='ErrorPlots',ks=False,ylim=None,obs=True):
    """ compiles a histogram of DM and compares fit expectations to it """
    
    ### compiles a list of DMs ###
    
    # if survey is multiple surveys, add all DMs together
    
    for i,s in enumerate(surveys):
        if s.Zs is not None:
            zs=s.Zs
            NZ=len(zs)
            which=i
    # width of histogram bins in Z
    HIST_WIDTH=0.1
    
    ######## Best ######
    bestrate=bestrates[which]
    
    ### now compiles expectations from parameter sets 
    
    
    #rates=g.rates
    pz=np.sum(bestrate,axis=1)
    
    # normalise to observed values
    norm=NZ/np.sum(pz)
    pz *= norm
        
    # correct for bin width effect
    rel_bin_size=HIST_WIDTH/(zvals[1]-zvals[0])
    pz *= rel_bin_size
    best_pz = pz
    
    ######## Emin ######
    Eminrate=Eminrates[which]
    
    #rates=g.rates
    pz=np.sum(Eminrate,axis=1)
    
    # normalise to observed values
    norm=NZ/np.sum(pz)
    pz *= norm
        
    # correct for bin width effect
    rel_bin_size=HIST_WIDTH/(zvals[1]-zvals[0])
    pz *= rel_bin_size
    Emin_pz = pz
    
    ######## n0 ########
    n0rate=n0rates[which]
    #rates=g.rates
    pz=np.sum(n0rate,axis=1)
    
    # normalise to observed values
    norm=NZ/np.sum(pz)
    pz *= norm
        
    # correct for bin width effect
    rel_bin_size=HIST_WIDTH/(zvals[1]-zvals[0])
    pz *= rel_bin_size
    n0_pz = pz
    
    
    error_pz=[]
    for i,rateset in enumerate(errorrates):
        n0rate=rateset[which]
        #rates=g.rates
        pz=np.sum(n0rate,axis=1)
        
        # normalise to observed values
        norm=NZ/np.sum(pz)
        pz *= norm
            
        # correct for bin width effect
        rel_bin_size=HIST_WIDTH/(zvals[1]-zvals[0])
        pz *= rel_bin_size
        
        error_pz.append(pz)
    
    nbins=21
    bins=np.linspace(0,(nbins-1)*HIST_WIDTH,nbins)
    
    h,b=np.histogram(zs,bins=bins)
    print("About to histogram ",zs,bins,h)
    bcs=bins[:-1]+HIST_WIDTH/2. # because I know the spacing is 100
    
    
    clist=plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure()
    
    plt.xlabel('$z$')
    plt.ylabel('$N_{\\rm FRB}$')
    #plt.xlim(0,2600)
    if obs:
        plt.bar(bcs,h,width=HIST_WIDTH,label='observed',color=clist[0])
    plt.xlim(0,1.5)
    if ylim is not None:
        plt.ylim(0,ylim)
    else:
        plt.ylim(0,4)
    for i,pz in enumerate(error_pz):
        plt.plot(zvals,pz,linewidth=1,linestyle='-.',color='gray')
    plt.plot(zvals,best_pz,linewidth=3,linestyle='-',label='best fit',color=clist[1])
    plt.plot(zvals,n0_pz,linewidth=3,linestyle='--',label='no evolution',color=clist[2])
    plt.plot(zvals,Emin_pz,linewidth=3,linestyle=':',label='$E_{\\rm min}$',color=clist[3])
    plt.plot(-zvals,error_pz[0],linewidth=1,linestyle='-.',color='gray',label='90% C.L.')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(outdir+'/Zerr.pdf')
    
    plt.close()
    
    
    make_zcdf_plot(zvals,[best_pz,n0_pz,Emin_pz],zs,['best fit','no evolution','$E_{\\rm min}$'],sys=error_pz,outdir=outdir)
    if ks:
        NMC=100000
        ks_test(zvals,best_pz,zs,NMC,label=' (best fit)',tag='best_z',outdir=outdir,xmax=0.6)
        ks_test(zvals,n0_pz,zs,NMC,label=' (no evolution)',tag='n0_z',outdir=outdir,xmax=0.6)
    return

def compare_z_fits2(survey,bestrate,errorrates,n0rate,Eminrate,zvals,dmvals,outdir='ErrorPlots',xmax=None,ymax=None):
    """ compiles a histogram of DM and compares fit expectations to it """
    
    ### compiles a list of DMs ###
    
    # if survey is multiple surveys, add all DMs together
    
    survey=survey[0]
    # widht of histogram bins in Z
    HIST_WIDTH=0.1
    NZ=survey.NFRB
    
    ### now compiles expectations from parameter sets 
    
    #rates=g.rates
    pz=np.sum(bestrate[0],axis=1)
    
    # normalise to observed values
    norm=NZ/np.sum(pz)
    pz *= norm
        
    # correct for bin width effect
    rel_bin_size=HIST_WIDTH/(zvals[1]-zvals[0])
    pz *= rel_bin_size
    best_pz = pz
    
    cum=np.cumsum(best_pz)
    cum=cum/cum[-1]
    iz=np.where(zvals < 0.1)[0][-1]
    print(cum[iz]," frbs from z < 0.1 (best)")
    
    ######## Emin ######
    
    
    #rates=g.rates
    pz=np.sum(Eminrate[0],axis=1)
    
    # normalise to observed values
    norm=NZ/np.sum(pz)
    pz *= norm
        
    # correct for bin width effect
    rel_bin_size=HIST_WIDTH/(zvals[1]-zvals[0])
    pz *= rel_bin_size
    Emin_pz = pz
    
    cum=np.cumsum(Emin_pz)
    cum=cum/cum[-1]
    iz=np.where(zvals < 0.1)[0][-1]
    print(cum[iz]," frbs from z < 0.1 (Emin)")
    
    ######## n0 ########
    #rates=g.rates
    pz=np.sum(n0rate[0],axis=1)
    
    # normalise to observed values
    norm=NZ/np.sum(pz)
    pz *= norm
        
    # correct for bin width effect
    rel_bin_size=HIST_WIDTH/(zvals[1]-zvals[0])
    pz *= rel_bin_size
    n0_pz = pz
    
    cum=np.cumsum(n0_pz)
    cum=cum/cum[-1]
    iz=np.where(zvals < 0.1)[0][-1]
    print(cum[iz]," frbs from z < 0.1 (n0)")
    
    error_pz=[]
    for i,rateset in enumerate(errorrates):
        n0rate=rateset[0]
        #rates=g.rates
        pz=np.sum(n0rate,axis=1)
        
        # normalise to observed values
        norm=NZ/np.sum(pz)
        pz *= norm
            
        # correct for bin width effect
        rel_bin_size=HIST_WIDTH/(zvals[1]-zvals[0])
        pz *= rel_bin_size
        
        error_pz.append(pz)
        
        cum=np.cumsum(pz)
        cum=cum/cum[-1]
        iz=np.where(zvals < 0.1)[0][-1]
        print(cum[iz]," frbs from z < 0.1 (error ",i,")")
    
    nbins=21
    bins=np.linspace(0,(nbins-1)*HIST_WIDTH,nbins)
    
    
    
    
    clist=plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure()
    
    plt.xlabel('$z$')
    plt.ylabel('$N_{\\rm FRB}$')
    #plt.xlim(0,2600)
    if survey.Zs is not None:
        zs=survey.Zs
        h,b=np.histogram(zs,bins=bins)
        bcs=bins[:-1]+HIST_WIDTH/2. # because I know the spacing is 100
        plt.bar(bcs,h,width=HIST_WIDTH,label='observed',color=clist[0])
    
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    
    for i,pz in enumerate(error_pz):
        plt.plot(zvals,pz,linewidth=1,linestyle='-.',color='gray')
    plt.plot(zvals,best_pz,linewidth=3,linestyle='-',label='best fit',color=clist[1])
    plt.plot(zvals,n0_pz,linewidth=3,linestyle='--',label='no evolution',color=clist[2])
    plt.plot(zvals,Emin_pz,linewidth=3,linestyle=':',label='$E_{\\rm min}$',color=clist[3])
    plt.plot(-zvals,error_pz[0],linewidth=1,linestyle='-.',color='gray',label='90% C.L.')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(outdir+'/Zerr.pdf')
    
    plt.close()
    
    return

def compare_dm_fits(surveys,bestrates,errorrates,n0rates,Eminrates,zvals,dmvals,outdir='ErrorPlots',ks=False):
    """ compiles a histogram of DM and compares fit expectations to it """
    
    ### compiles a list of DMs ###
    print("Comparing DM fits")
    # if survey is multiple surveys, add all DMs together
    dms=[]
    ns=[]
    for i,s in enumerate(surveys):
        dms.append(s.DMEGs)
        ns.append(len(dms[i]))
    
    dms=np.concatenate(dms)
    
    ### now compiles expectations from parameter sets 
    HIST_WIDTH=100
    for i,rates in enumerate(bestrates):
        #rates=g.rates
        pdm=np.sum(rates,axis=0)
        
        # normalise to observed values
        norm=ns[i]/np.sum(pdm)
        pdm *= norm
        
        # correct for bin width effect
        rel_bin_size=HIST_WIDTH/(dmvals[1]-dmvals[0])
        
        pdm *= rel_bin_size
        if i==0:
            best_pdm = pdm
        else:
            best_pdm += pdm
    
    ###### for n0 #########
    for i,rates in enumerate(n0rates):
        #rates=g.rates
        pdm=np.sum(rates,axis=0)
        
        # normalise to observed values
        norm=ns[i]/np.sum(pdm)
        pdm *= norm
        # correct for bin width effect
        rel_bin_size=HIST_WIDTH/(dmvals[1]-dmvals[0])
        
        pdm *= rel_bin_size
        if i==0:
            n0_pdm = pdm
        else:
            n0_pdm += pdm
    
    ###### for Emin #########
    for i,rates in enumerate(Eminrates):
        #rates=g.rates
        pdm=np.sum(rates,axis=0)
        
        # normalise to observed values
        norm=ns[i]/np.sum(pdm)
        pdm *= norm
        # correct for bin width effect
        rel_bin_size=HIST_WIDTH/(dmvals[1]-dmvals[0])
        
        pdm *= rel_bin_size
        if i==0:
            Emin_pdm = pdm
        else:
            Emin_pdm += pdm
    
    temp=np.zeros(best_pdm.shape)
    error_pdms=[]
    for i,rateset in enumerate(errorrates):
        # create a new version
        gs_apdm=np.copy(temp)
        for j,g in enumerate(rateset):
            pdm=np.sum(g,axis=0)
            # normalise to observed values
            # (grids will be in the same order)
            norm=ns[j]/np.sum(pdm)
            pdm *= norm
            # correct for bin width effect
            rel_bin_size=HIST_WIDTH/(dmvals[1]-dmvals[0])
            
            pdm *= rel_bin_size
            gs_apdm += pdm
        error_pdms.append(gs_apdm)
        
    bins=np.linspace(0,2600,27)
    h,b=np.histogram(dms,bins=bins)
    bcs=bins[:-1]+50 # because I know the spacing is 100
    
    clist=plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure()
    
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N_{\\rm FRB}$')
    plt.xlim(0,2600)
    plt.bar(bcs,h,width=100,color=clist[0])
    
    for i,apdm in enumerate(error_pdms):
        plt.plot(dmvals,apdm,linewidth=1,linestyle='-',color='gray')
    plt.plot(dmvals,best_pdm,linewidth=3,linestyle='-',color=clist[1],label='best fit')
    plt.plot(dmvals,n0_pdm,linewidth=3,linestyle='--',color=clist[2],label='no evolution')
    plt.plot(dmvals,Emin_pdm,linewidth=3,linestyle=':',label='$E_{\\rm min}$',color=clist[3])
    plt.plot(-dmvals,apdm,linewidth=1,linestyle='-',color='gray',label='90% C.L.')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+'/DMerr.pdf')
    
    plt.close()
    
    
    make_cdf_plot(dmvals,[best_pdm,n0_pdm,Emin_pdm],dms,['best fit','no evolution','$E_{\\rm min}$'],outdir=outdir)
    
    if ks:
        NMC=10000 # was 100k for DM. Strange
        ks_test(dmvals,best_pdm,dms,NMC,label=' (best fit)',tag='best_dm',outdir=outdir,xmax=0.4)
        ks_test(dmvals,n0_pdm,dms,NMC,label=' (no evolution)',tag='n0_dm',outdir=outdir,xmax=0.4)
    

def make_cumulative_hist(x,dms):
    # sorts through the DMs to make the cumulative histogram
    #ordered_dms=np.sort(dms)
    
    y=np.zeros([x.size])
    for i,x in enumerate(x):
        temp=np.where(dms < x)[0]
        y[i]=len(temp)
    y /= y[-1]
    return y

def make_zcdf_plot(x,theories,obs,labels,sys=None,outdir='ErrorPlots'):
    # get cumulatvie theory
    cobs=make_cumulative_hist(x,obs)
    
    clist=plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    nx=x.size
    # number of FRBs
    nobs=obs.size
    
    #### plots cumultive hist ###
    plt.figure()
    
    plt.xlabel('z')
    plt.ylabel('CDF')
    plt.plot(x,cobs,label='observed')
    for i,theory in enumerate(theories):
        ctheory=np.cumsum(theory)
        ctheory /= ctheory[-1]
        plt.plot(x,ctheory,label=labels[i])
    
    if sys is not None:
        for i,theory in enumerate(sys):
            ctheory=np.cumsum(theory)
            ctheory /= ctheory[-1]
            if i==0:
                plt.plot(x,ctheory,color='gray', label='90\%')
            else:
                plt.plot(x,ctheory,color='gray')
    
    plt.xlim(0,1.5)
    plt.ylim(0,1)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+'/all_z_cdfs.pdf')
    plt.close()

def make_cdf_plot(x,theories,obs,labels,outdir='ErrorPlots'):
    # get cumulatvie theory
    cobs=make_cumulative_hist(x,obs)
    
    
    nx=x.size
    # number of FRBs
    nobs=obs.size
    
    #### plots cumultive hist ###
    plt.figure()
    
    plt.xlabel('DM [pc cm$^{-3}$]')
    plt.ylabel('CDF')
    plt.plot(x,cobs,label='observed')
    for i,theory in enumerate(theories):
        ctheory=np.cumsum(theory)
        ctheory /= ctheory[-1]
        plt.plot(x,ctheory,label=labels[i])
    
    plt.xlim(0,3000)
    plt.ylim(0,1)
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outdir+'/all_dm_cdfs.pdf')
    plt.close()

def ks_test(x,theory,obs,nMC,label='',tag='',outdir='ErrorPlots',xmax=0.5):
    # get cumulatvie theory
    cobs=make_cumulative_hist(x,obs)
    
    ctheory=np.cumsum(theory)
    ctheory /= ctheory[-1]
    
    nx=x.size
    # number of FRBs
    nobs=obs.size
    
    # get ks stat
    ks_stat=np.max(np.abs(ctheory-cobs))
    print("ks stat is ",ks_stat,", now simulating its significance...")
    
    #### plots cumultive hist ###
    plt.figure()
    
    plt.xlabel('DM [pc cm$^{-3}$]')
    plt.ylabel('CDF')
    plt.plot(x,cobs,label='Observed')
    plt.plot(x,ctheory,label='Predicted')
    #plt.xlim(0,3000)
    #plt.ylim(0,1)
    
    ks_stats=np.zeros(nMC)
    nctheory=ctheory.size
    for i in np.arange(nMC):
        
        rs=np.random.rand(nobs)
        dmlist=[]
        for j,r in enumerate(rs):
            idm0=np.array(np.where(ctheory < r)[0])
            
            if idm0.size==0:
                idm0=nctheory-2
            else:
                idm0=idm0[-1]
                if idm0 > nctheory-2:
                    idm0=nctheory-2
            idm1=idm0+1
            k=(ctheory[idm1]-r)/(ctheory[idm1]-ctheory[idm0])
            dmsim=k*x[idm0] + (1-k)*x[idm1]
            dmlist.append(dmsim)
        
        dmlist=np.sort(dmlist)
        cmc_obs=make_cumulative_hist(x,dmlist)
        ks_stats[i]=np.max(np.abs(ctheory-cmc_obs))
        #if i==0:
        #	plt.plot(x,cmc_obs,label='simulated')
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+'/cdf'+tag+'.pdf')
    plt.close()
    
    ### plots the ks distribution and gets a p-value ###
    mean=np.mean(ks_stats)
    std_dev=(np.sum((ks_stats-mean)**2)/nMC)**0.5
    nsigma=3
    #bins=np.linspace(mean-std_dev*nsigma,mean+std_dev*nsigma,101)
    bins=np.linspace(0,xmax,51)
    h,b=np.histogram(ks_stats,bins=bins)
    db=(bins[1]-bins[0])
    plotb=bins[0:-1]+db/2
    
    nless=np.where(ks_stats < ks_stat)[0]
    p_value=1.-len(nless)/nMC
    print("p-value of this ks test is ",p_value)
    
    p_value=np.round(p_value,decimals=2)
    
    plt.figure()
    plt.xlabel('ks statistic')
    plt.ylabel('p(ks) [a.u.]')
    plt.bar(plotb,h,width=db,label='simulated\n'+label)
    plt.plot([ks_stat,ks_stat],[0,np.max(h)],label='observed\n(p='+str(p_value)+')',color='red')
    plt.legend()
    #plt.tight_layout()
    plt.savefig(outdir+'/pvalue_ks_stat'+tag+'.pdf')
    plt.close()

def err_get_source_counts(grids,plot=None):
    """
    Calculates the source-counts function for a given grid
    It does this in terms of p(SNR)dSNR
    """
    print("Getting cumulative source counts")
    # this is closely related to the likelihood for observing a given psnr!
    # just use one of them for getting Emax etc
    # calculate vector of grid thresholds
    grid=grids[0]
    Emax=grid.Emax
    Emin=grid.Emin
    gamma=grid.gamma
    ndm=grid.dmvals.size
    
    nsnr=51
    #snrmin=0.001
    #snrmax=1000.
    
    snrs=np.logspace(0,2,nsnr) # histogram array of values for s=SNR/SNR_th
    
    
    for i,grid in enumerate(grids):
        # holds cumulative and differential source counts
        cpsnrs=np.zeros([nsnr])
        psnrs=np.zeros([nsnr-1])
        
        # holds DM-dependent source counts
        dmcpsnrs=np.zeros([nsnr,ndm])
        dmpsnrs=np.zeros([nsnr-1,ndm])
        backup1=np.copy(grid.thresholds)
        
        # modifies grid to simplify beamshape. I have no idea why it is doing this...
        # maybe just to make it faster?
        #grid.beam_b=np.array([grid.beam_b[-1]]) # sets value of beam to max b
        #grid.beam_o=np.array([grid.beam_o[-1]]) # sets to corresponding value of above
        #grid.b_fractions=None # just resets this
        
        # it seems we are calculating the cumulative rate as a function of s
        for i,s in enumerate(snrs):
            
            grid.thresholds=backup1*s #increases thresholds by a constant factor of s
            grid.calc_pdv(Emin,Emax,gamma)
            grid.calc_rates()
            rates=grid.rates
            dmcpsnrs[i,:]=np.sum(rates,axis=0) # rate as function of dm
            cpsnrs[i]=np.sum(dmcpsnrs[i,:]) # total rate
        
        # the last one contains cumulative values
        for i,s in enumerate(snrs):
            if i==0:
                continue
            psnrs[i-1]=cpsnrs[i-1]-cpsnrs[i]
            dmpsnrs[i-1,:]=dmcpsnrs[i-1,:]-dmcpsnrs[i,:]
    
    # adjusts values by 1.5
    mod=1.5
    snrs=snrs[:-1]
    imid=int((nsnr+1)/2)
    xmid=snrs[imid]
    ymid=psnrs[imid]
    slopes=np.linspace(1.3,1.7,5)
    ys=[]
    for i,s in enumerate(slopes):
        ys.append(ymid*xmid**s*snrs**-s)
    
    if plot is not None:
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$s=\\frac{\\rm SNR}{\\rm SNR_{\\rm th}}$')
        plt.ylabel('$p(s) s^{1.5} d\\,\\log(s)$')
        plt.plot(snrs,psnrs*snrs**mod,label='Prediction',color='black',linewidth=2) # this is in relative units
        for i,s in enumerate(slopes):
            plt.plot(snrs,ys[i]*snrs**mod,label='slope='+str(s)[0:3])
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot)
        plt.close()
    return snrs,psnrs,dmpsnrs



def error_plot_psnrs(gridsets,labels,surveys,psets,plot=None):
    """ for each gridset in gridets, we have surveys grids """
    
    import os
    if plot is not None:
        basedir=os.path.dirname(plot)
    
    ##### calculates observed SNR histogram for each gridset ####
    
    sobs=[]
    for i,s in enumerate(surveys):
        sobs.append(s.Ss)
    sobs=np.concatenate(sobs)
    Nobs=sobs.size
    sbins=np.linspace(0,2,101)
    lsobs=np.log10(sobs)
    shist,sbins=np.histogram(lsobs,bins=sbins)
    
    dbins=sbins[1]-sbins[0]
    sx=sbins[0:-1]+dbins/2.
    
    
    plt.figure()
    plt.xlabel('$\\log_{10} (s={\\rm SNR}/{\\rm SNR}_{\\rm th})$')
    plt.ylabel('N(s)')
    plt.bar(sx,shist,width=dbins,alpha=0.5,label='observed')
    
    
    xmaxs=[10,10,100]
    ymins=[0.01,0.01,0.001]
    ymaxs=[0.2,0.2,0.2]
    
    #### gets expectations for each gridset####
    NS=401
    smin=0
    smax=4
    NS=51
    smin=0
    smax=2
    slist=np.logspace(0,2,NS)
    lslist=np.log10(slist)
    dlogs=(smax-smin)/NS*np.log(10) # width in natural log space
    
    csets=[]
    for i,gridset in enumerate(gridsets):
        pset=psets[i]
        for j,grid in enumerate(gridset):
            
            title=basedir+'/'+labels[i]+'_'+surveys[j].name+'.pdf'
            print("Doing grid ",j," from gridset ",i,"...")
            if surveys[j].Zs is not None:
                slist,psnrs=calc_psnr_2D(grid,surveys[j],pset,slist=slist,doplot=title,xlim=[1,xmaxs[j]],ylim=[ymins[j],ymaxs[j]])
            else:
                slist,psnrs=calc_psnr_1D(grid,surveys[j],pset,slist=slist,doplot=title,xlim=[1,xmaxs[j]],ylim=[ymins[j],ymaxs[j]])
            print("... done")
            
            
            NFRBs=psnrs.shape[0]
            #normalises each FRB so the sum is equal to 1
            # accounts for log-spacing and d log width
            wpsnrs = psnrs*slist
            norm=np.sum(wpsnrs,axis=1)/dlogs
            
            psnrs = (psnrs.T/norm).T
            
            # adds to cumulative count over all observations
            if j==0:
                cpsnrs=np.sum(psnrs,axis=0)
            else:
                cpsnrs += np.sum(psnrs,axis=0)
            
        
        # we now how have the cumulative sum of psnr over all FRBs in all surveys
        # we can then compare this to the observed histogram to see how we go
        csets.append(cpsnrs)
        
        # we have already normalised each FRB so that the log sum should come to NFRB
        # however, we should still re-normalise to the bin width of the histogram
        # this means multiplyng by slist and dlogs of the original histogram
        
        # now convert to log-space
        # first multiply by the slist itself 
        cpsnrs *= slist*dbins*np.log(10) # to convert the bin width into a natural log width
        
        plt.plot(np.log10(slist),cpsnrs,label=labels[i])
        
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot)
    
def calc_psnr_1D(grid,survey,pset,slist,doplot=None,xlim=[1,100],ylim=[0.01,1]):
    """ Calculates psnr as a function of snr
    """
    rates=grid.rates
    dmvals=grid.dmvals
    zvals=grid.zvals
    DMobs=survey.DMEGs
    #DMobs=np.sort(DMobs)
    DMobs=np.sort(survey.DMEGs)
    idmsort=np.argsort(survey.DMEGs)
    
    # start by collapsing over z
    pdm=np.sum(rates,axis=0)
    
    ddm=dmvals[1]-dmvals[0]
    kdms=DMobs/ddm
    idms1=kdms.astype('int')
    idms2=idms1+1
    dkdms=kdms-idms1
    pvals=pdm[idms1]*(1.-dkdms) + pdm[idms2]*dkdms
    
    global_norm=np.sum(pdm)
    log_global_norm=np.log10(global_norm)
    
    
    llsum=np.sum(np.log10(pvals))-log_global_norm*DMobs.size
    
    NS=slist.size
    psnrs=np.zeros([survey.Ss.size,NS]) # generates plot for each individual FRB
    for k,s in enumerate(slist):
        # NOTE: to break this into a p(SNR|b) p(b) term, we first take
        # the relative likelihood of the threshold b value compared
        # to the entire lot, and then we calculate the local
        # psnr for that beam only. But this requires a much more
        # refined view of 'b', rather than the crude standatd 
        # parameterisation
        
        # calculate vector of grid thresholds
        Emax=grid.Emax
        Emin=grid.Emin
        gamma=grid.gamma
        psnr=np.zeros([survey.Ss.size])
        
        # get vector of thresholds as function of z and threshold/weight list
        # note that the dimensions are, nthresh (weights), z, DM
        Eths = grid.thresholds[:,:,idms1]*(1.-dkdms)+ grid.thresholds[:,:,idms2]*dkdms
        
        ##### IGNORE THIS, PVALS NOW CONTAINS CORRECT NORMALISATION ######
        # we have previously calculated p(DM), normalised by the global sum over all DM (i.e. given 1 FRB detection)
        # what we need to do now is calculate this normalised by p(DM),
        # i.e. psnr is the probability of snr given DM, and hence the total is
        # p(snr,DM)/p(DM) * p(DM)/b(burst)
        # get a vector of rates as a function of z
        #rs = rates[:,idms1[j]]*(1.-dkdms[j])+ rates[:,idms2[j]]*dkdms[j]
        rs = rates[:,idms1]*(1.-dkdms)+ rates[:,idms2]*dkdms	
        #norms=np.sum(rs,axis=0)/global_norm
        norms=pvals
        
        zpsnr=np.zeros(Eths.shape[1:])
        beam_norm=np.sum(survey.beam_o)
        #in theory, we might want to normalise by the sum of the omeba_b weights, although it does not matter here
        
        
        for i,b in enumerate(survey.beam_b):
            #iterate over the grid of weights
            bEths=Eths/b #this is the only bit that depends on j, but OK also!
            #now wbEths is the same 2D grid
            #wbEths=bEths #this is the only bit that depends on j, but OK also!
            
            #bEobs=bEths*survey.Ss #should correctky multiply the last dimensions
            # we simply now replace survey.Ss with the value of s
            bEobs=bEths*s
            
            for j,w in enumerate(grid.eff_weights):
                temp=(grid.array_diff_lf(bEobs[j,:,:],Emin,Emax,gamma).T*grid.FtoE).T
                zpsnr += temp*survey.beam_o[i]*w #weights this be beam solid angle and efficiency
        
        
        # we have now effectively calculated the local probabilities in the source-counts histogram for a given DM
        # we have to weight this by the sfr_smear factors, and the volumetric probabilities
        # this are the grid smearing factors incorporating pcosmic and the host contributions
        sg = grid.sfr_smear[:,idms1]*(1.-dkdms)+ grid.sfr_smear[:,idms2]*dkdms
        sgV = (sg.T*grid.dV.T).T
        wzpsnr = zpsnr * sgV
        
        
        #THIS HAS NOT YET BEEN NORMALISED!!!!!!!!
        # at this point, wzpsnr should look exactly like the grid.rates, albeit
        # A: differential, and 
        # B: slightly modified according to observed and not threshold fluence
        
        # normalises for total probability of DM occurring in the first place.
        # We need to do this. This effectively cancels however the Emin-Emax factor.
        # sums down the z-axis
        psnr=np.sum(wzpsnr,axis=0)
        psnr /= norms #normalises according to the per-DM probability
        
        
        psnrs[:,k]=psnr
        
        
        # checks to ensure all frbs have a chance of being detected
        bad=np.array(np.where(psnr == 0.))
        if bad.size > 0:
            snrll = float('NaN') # none of this is possible! [somehow...]
        else:
            snrll = np.sum(np.log10(psnr))
        
        llsum += snrll
        
    if doplot is not None:
        plt.figure()
        ax=plt.gca()
        ax.set_aspect('auto')
        
        dlogs=np.log(slist[1])-np.log(slist[0])
        wpsnrs = psnrs*(slist)
        norm=np.sum(wpsnrs,axis=1)
        plotpsnrs = (psnrs.T/norm).T #correctly normalised now
        print("For plot ",doplot," norms were ",norm)
        #now multiply by slist to some power
        plotpsnrs*=slist
        
        xmin=xlim[0]
        xmax=xlim[1]
        ymin=ylim[0]
        ymax=ylim[1]
        
        plt.ylim(ymin,ymax)
        plt.xlim(xmin,xmax)
        plt.yscale('log')
        plt.xscale('log')
        
        linestyles=['-',':','--','-.']
        markerstyles=['o','^','x']
        ylist=[]
        DMs=survey.DMs[idmsort]
        Ss=survey.Ss[idmsort]
        for j,DM in enumerate(DMs):
            ls=linestyles[int(j/10)]
            ms=markerstyles[int(j/10)]
            plt.plot(slist,plotpsnrs[j],linestyle=ls,zorder=1)
            # adds in observed plot
            sobs=Ss[j]
            
            i2=np.where(slist>sobs)[0][0]
            i1=i2-1
            k=(sobs-slist[i1])/(slist[i2]-slist[i1])
            y=(k*plotpsnrs[j,i2]+(1.-k)*plotpsnrs[j,i1])
            plt.scatter(Ss[j],y,color=plt.gca().lines[-1].get_color(),s=70,marker=ms,zorder=2)
            plt.plot(slist,-plotpsnrs[j],label=str(int(round(DM,0))),linestyle=ls,zorder=1,marker=ms,color=plt.gca().lines[-1].get_color())
            
        #plt.plot(DMobs,pvals,'ro')
        plt.xlabel('$s$')
        plt.ylabel('$s \\, p_s(s)$')
        
        if xmax == 10:
            from matplotlib.ticker import ScalarFormatter
            ax=plt.gca()
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())
            
            import matplotlib.ticker as ticker
            
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        
        plt.legend(loc='upper right',ncol=4,fontsize=10)
        plt.tight_layout()
        ax.set_aspect('auto')
        plt.savefig(doplot)
        plt.close()
    return slist,psnrs


def calc_psnr_2D(grid,survey,pset,slist,doplot=None,xlim=[1,100],ylim=[0.01,1]):
    """ Calculates 2D likelihoods using observed DM,z values """
    
    ######## Calculates p(DM,z | FRB) ########
    # i.e. the probability of a given z,DM assuming
    # an FRB has been observed. The normalisation
    # below is proportional to the total rate (ish)
    
    rates=grid.rates
    zvals=grid.zvals
    dmvals=grid.dmvals
    
    #DMobs=survey.DMEGs
    DMobs=np.sort(survey.DMEGs)
    idmsort=np.argsort(survey.DMEGs)
    Zobs=survey.Zs[idmsort]
    
    #Zobs=survey.Zs
    
    
    
    #if survey.meta["TOBS"] is not None:
    #	TotalRate=np.sum(rates)*survey.meta["TOBS"]
        # this is in units of number per MPc^3 at Emin
    
    # normalise to total probability of 1
    norm=np.sum(rates) # gets multiplied by event size later
    
    
    # get indices in dm space
    ddm=dmvals[1]-dmvals[0]
    kdms=DMobs/ddm
    idms1=kdms.astype('int')
    idms2=idms1+1
    dkdms=kdms-idms1
    
    # get indices in z space
    dz=zvals[1]-zvals[0]
    kzs=Zobs/dz
    izs1=kzs.astype('int')
    izs2=izs1+1
    dkzs=kzs-izs1
    
    pvals = rates[izs1,idms1]*(1.-dkdms)*(1-dkzs)
    pvals += rates[izs2,idms1]*(1.-dkdms)*dkzs
    pvals += rates[izs1,idms2]*dkdms*(1-dkzs)
    pvals += rates[izs2,idms2]*dkdms*dkzs
    
    bad=np.array(np.where(pvals <= 0.))
    if bad.size > 0:
        pvals[bad]=1e-20 # hopefully small but not infinitely so
    llsum=np.sum(np.log10(pvals))#-norm
    llsum -= np.log10(norm)*Zobs.size # once per event
    
    
    
    ###### Calculates p(E | z,DM) ########
    # i.e. the probability of observing an FRB
    # with energy E given redshift and DM
    # this calculation ignores beam values
    # this is the derivative of the cumulative distribution
    # function from Eth to Emax
    # this does NOT account for the probability of
    # observing something at a relative sensitivty of b, i.e. assumes you do NOT know localisation in your beam...
    # to do that, one would calculate this for the exact value of b for that event. The detection
    # probability has already been integrated over the full beam pattern, so it would be trivial to
    # calculate this in one go. Or in other words, one could simple add in survey.Bs, representing
    # the local sensitivity to the event [keeping in mind that Eths has already been calculated
    # taking into account the burst width and DM, albeit for a mean FRB]
    # Note this would be even simpler than the procedure described here - we just
    # use b! Huzzah! (for the beam)
    # IF:
    # - we want to make FRB width analogous to beam, THEN
    # - we need an analogous 'beam' (i.e. width) distribution to integrate over,
    #     which gives the normalisation
    
    NS=slist.size
    psnrs=np.zeros([survey.Ss.size,NS]) # generates plot for each individual FRB
    for k,s in enumerate(slist):
        # NOTE: to break this into a p(SNR|b) p(b) term, we first take
        # the relative likelihood of the threshold b value compare
        # to the entire lot, and then we calculate the local
        # psnr for that beam only. But this requires a much more
        # refined view of 'b', rather than the crude standatd 
        # parameterisation
        
        # calculate vector of grid thresholds
        Emax=grid.Emax
        Emin=grid.Emin
        gamma=grid.gamma
        #Eths has dimensions of width likelihoods and nobs
        # i.e. later, the loop over j,w uses the first index
        Eths = grid.thresholds[:,izs1,idms1]*(1.-dkdms)*(1-dkzs)
        Eths += grid.thresholds[:,izs2,idms1]*(1.-dkdms)*dkzs
        Eths += grid.thresholds[:,izs1,idms2]*dkdms*(1-dkzs)
        Eths += grid.thresholds[:,izs2,idms2]*dkdms*dkzs
        
        FtoE = grid.FtoE[izs1]*(1.-dkzs)
        FtoE += grid.FtoE[izs2]*dkzs
        
        beam_norm=np.sum(survey.beam_o)
        
        # now do this in one go
        # We integrate p(snr|b,w) p(b,w) db dw. I have no idea how this could be multidimensional
        psnr=np.zeros(Eths.shape[1:])
        for i,b in enumerate(survey.beam_b):
            bEths=Eths/b # array of shape NFRB, 1/b
            #bEobs=bEths*survey.Ss
            bEobs=bEths*s
            for j,w in enumerate(grid.eff_weights):
                temp=grid.array_diff_lf(bEobs[j,:],Emin,Emax,gamma) * FtoE #one dim in beamshape, one dim in FRB
                
                psnr += temp.T*survey.beam_o[i]*w #multiplies by beam factors and weight
                
        # at this stage, we have the amplitude from diff power law summed over beam and weight
        
        # we only alculate the following sg and V factors to get units to be
        # comparable to the 1D case - otherwise it is superfluous
        sg = grid.sfr_smear[izs1,idms1]*(1.-dkdms)*(1-dkzs)
        sg += grid.sfr_smear[izs2,idms1]*(1.-dkdms)*dkzs
        sg += grid.sfr_smear[izs1,idms2]*dkdms*(1-dkzs)
        sg += grid.sfr_smear[izs2,idms2]*dkdms*dkzs
        dV = grid.dV[izs1]*(1-dkzs) +  grid.dV[izs2]*dkzs
        # at this stage, sg and dV account for the DM distribution and SFR;
        # dV is the volume elements
        # we just multiply these together
        sgV = sg*dV
        wzpsnr = psnr.T*sgV
        
        # this step weights psnr by the volumetric values
        
        ######## NORMALISATION DISCUSSION ######
        # we want to calculate p(snr) dpsnr
        # this must be \int p(snr | w,b) p(w,b) dw,b
        # \int p(snr | detection) p(det|w,b) p(w,b) dw,b
        # to make it an indpeendent factor, and not double-count it, means calculating
        # \int p(snr | detection) dsnr p(det|w,b) p(w,b) dw,b / \int p(det|w,b) p(w,b) dw,b
        # array_diff_power_law simply calculates p(snr), which is the probability amplitude
        # -(gamma*Eth**(gamma-1)) / (Emin**gamma-Emax**gamma )
        # this includes the probability; hence need to account for this
        
        # it is essential that this normalisation occurs for a normalised pvals
        # this normalisation essentially undoes the absolute calculation of the rate, i.e. we are normalising by the total distribution
        # hence we *really* ought to be adding the normalisation to this...
        # the idea here is that p(snr,det)/p(det) * p(det)/pnorm. Hence pvals - which contains
        # the normalisation - should be the un-normalised values.
        
        wzpsnr /= pvals
        
        psnrs[:,k]=wzpsnr
        
        # checks to ensure all frbs have a chance of being detected
        bad=np.array(np.where(wzpsnr == 0.))
        if bad.size > 0:
            snrll = float('NaN') # none of this is possible! [somehow...]
        else:
            snrll = np.sum(np.log10(wzpsnr))
        
        llsum += snrll
    if doplot is not None:
        plt.figure()
        ax=plt.gca()
        ax.set_aspect('auto')
        
        dlogs=np.log(slist[1])-np.log(slist[0])
        wpsnrs = psnrs*(slist)
        norm=np.sum(wpsnrs,axis=1)
        plotpsnrs = (psnrs.T/norm).T #correctly normalised now
        print("For plot ",doplot," norms were ",norm)
        #now multiply by slist to some power
        plotpsnrs*=slist
        
        xmin=xlim[0]
        xmax=xlim[1]
        ymin=ylim[0]
        ymax=ylim[1]
        
        plt.ylim(ymin,ymax)
        plt.xlim(xmin,xmax)
        plt.yscale('log')
        plt.xscale('log')
        
        linestyles=['-',':','--','-.']
        markerstyles=['o','^','x']
        ylist=[]
        DMs=survey.DMs[idmsort]
        Ss=survey.Ss[idmsort]
        for j,DM in enumerate(DMs):
            ls=linestyles[int(j/10)]
            ms=markerstyles[int(j/10)]
            plt.plot(slist,plotpsnrs[j],linestyle=ls,zorder=1)
            # adds in observed plot
            sobs=Ss[j]
            i2=np.where(slist>sobs)[0][0]
            i1=i2-1
            k=(sobs-slist[i1])/(slist[i2]-slist[i1])
            y=(k*plotpsnrs[j,i2]+(1.-k)*plotpsnrs[j,i1])
            plt.scatter(Ss[j],y,color=plt.gca().lines[-1].get_color(),s=70,marker=ms,zorder=2)
            plt.plot(slist,-plotpsnrs[j],label=str(int(round(DM,0))),linestyle=ls,zorder=1,marker=ms,color=plt.gca().lines[-1].get_color())
            
        #plt.plot(DMobs,pvals,'ro')
        plt.xlabel('$s$')
        plt.ylabel('$s \\, p_s(s)$')
        from matplotlib.ticker import ScalarFormatter
        ax=plt.gca()
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        import matplotlib.ticker as ticker

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        
        plt.legend(loc='upper right',ncol=4,fontsize=10)
        plt.tight_layout()
        ax.set_aspect('auto')
        plt.savefig(doplot)
        plt.close()
    return slist,psnrs
    
