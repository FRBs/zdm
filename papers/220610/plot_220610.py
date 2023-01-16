""" 
This script creates zdm grids and plots localised FRBs for FRB 20220610A,
using the exact conditions at which 20220610A was observed.

It will produce the following sets of outputs in directory 220610/
- Planck_[param]_[min/max][nohost].pdf where:
    - param runs over the six parameters of the FRB population
        (lEmax, gamma, alpha, sfr_n, lmean host, lsigma host)
    - max/min represents cases wherre the above parameters has been
        set to its 90% min/max confidence limit respectively, and
        other parameters are at values corresponding to this, when
        assuming the Planck value of H0 (~67.4 km/s/Mpc)
    - [nohost] (either present or not): if 'nohost', then
        the FRB host galaxy contribution is set to zero
In each case, a zdm plot is produced for the exact observing conditions
at which 20220610A was observed, using survey file 220610_only.dat

Other plots are:
    - Planck_standard[nohost].pdf: as above, but for best-fit parameters

    - A set of plots giving p(dmhost):
        dmhost.pdf [included in the Science paper] (Fr)
        p_dmhost.pdf (p)
        210117_dmhost.pdf (FAr)
        210117_FAST_p_dmhost.pdf  (FpA)
        FAST_p_dmhost.pdf  (Fp)
        rest_frame_FAST_p_dmhost.pdf  (Frp)
        rest_frame_210117_FAST_p_dmhost.pdf (FpAr)
        
       with the following inclusions:
        includes estimates also for FAST FRB 20190520B 'F'
        includes estimates also for ASKAP FRB 20210117 'A'
        includes posterior estimates for host galaxy contribution 'p'
        plotted in the rest-frame of the host galaxies 'r'
    
    - pzgdm_FRB220610.pdf
        Probability distribution p (z|DM) for FRB220610A given
        the previous best fit parameter estimates, the 90%
        ranges when varying parameters, and the new best-fit Emax
    
    - cumulative_pzgdm_FRB220610.pdf [included in the paper]
        As above, but cumulative distributions
"""
import os

from zdm import analyze_cube as ac
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    from astropy.cosmology import Planck18
    
    doE=False
    if doE:
        convert_energy()
    
    # in case you wish to switch to another output directory
    opdir = "220610/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    load=False
    
    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    names = '220610_only'
    sdir = "../../zdm/data/Surveys/"
    
    labels=["lEmax","alpha","gamma","sfr_n","lmean","lsigma"]
    
    # loads extreme parameter sets. Run get_extremes_from_cube.py
    # these are by default at the 90% C.L.
    sets=read_extremes()
    pzgdms=[]
    for i,pset in enumerate(sets):
        continue
        # breaks if loading data
        if load:
            continue
        nth=int(i/2)
        if i==nth*2:
            ex="_min.pdf"
        else:
            ex="_max.pdf"
        
        opfile=opdir+"Planck_"+labels[nth]+ex
        zvals,pzgdm=plot_expectations(names,sdir,pset,opfile)
        pzgdms.append(pzgdm)
    
    # plots using parameters for Planck H0 with new Emax value, and
    # zero host galaxy DM, to get p(DM>obs)
    if not load:
        vparams = {}
        vparams["H0"] = 67.4
        vparams["lEmax"] = 41.63
        vparams["gamma"] = -0.948
        vparams["alpha"] = 1.03
        vparams["sfr_n"] = 1.15
        vparams["lmean"] = 0.01
        vparams["lsigma"] = 0.57
        
        opfile=opdir+"zero_host_dm.pdf"
        #zvals,new_pzgdm=plot_expectations(names,sdir,vparams,opfile,dmhost=False)
    
    
    # plots using parameters for Planck H0 with new Emax value
    if not load:
        vparams = {}
        vparams["H0"] = 67.4
        vparams["lEmax"] = 41.63
        vparams["gamma"] = -0.948
        vparams["alpha"] = 1.03
        vparams["sfr_n"] = 1.15
        vparams["lmean"] = 2.22
        vparams["lsigma"] = 0.57
        
        opfile=opdir+"Planck_new_Emax.pdf"
        zvals,new_pzgdm=plot_expectations(names,sdir,vparams,opfile,dmhost=False)
    
    # generates plots for Planck H0 best-fit values
    if not load:
        vparams = {}
        vparams["H0"] = 67.4
        vparams["lEmax"] = 41.26
        vparams["gamma"] = -0.948
        vparams["alpha"] = 1.03
        vparams["sfr_n"] = 1.15
        vparams["lmean"] = 2.22
        vparams["lsigma"] = 0.57
        
        opfile=opdir+"Planck_standard.pdf"
        zvals,std_pzgdm=plot_expectations(names,sdir,vparams,opfile,dmhost=True)
        pzgdms.append(std_pzgdm)
        pzgdms.append(new_pzgdm)
        pzgdms=np.array(pzgdms)
        np.save('pzgdms.npy',pzgdms)
        np.save('zvals.npy',zvals)
    
    # loads saved data
    if load:
        pzgdms=np.load('pzgdms.npy')
        std_pzgdm=pzgdms[-2,:]
        new_pzgdm=pzgdms[-1,:]
        pzgdms=pzgdms[:-2,:]
        zvals=np.load('zvals.npy')
    
    # generates plots for p(z|DM) and similar
    
    plt.figure()
    plt.xlabel('$z$')
    plt.ylabel('$p(z|{\\rm DM}_{20220610})$')
    #for i,pzgdm in enumerate(pzgdms):
    for i in np.arange(12):
        pzgdm = pzgdms[i,:]
        if i==0:
            plt.plot(zvals,pzgdm,linewidth=1,color='gray',label='90\%')
        else:
            plt.plot(zvals,pzgdm,linewidth=1,color='gray')
    plt.plot(zvals,std_pzgdm,linewidth=3,color='blue',label='Best fit')
    plt.plot(zvals,new_pzgdm,linewidth=3,color='orange',label='New Emax',linestyle="--")
    plt.xlim(0,2)
    plt.ylim(0,3.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'pzgdm_FRB220610.pdf')
    plt.close()
    
    plt.figure()
    plt.xlabel('$z$')
    plt.ylabel('cdf$(z|20220610A)$')
    iz=np.where(zvals>1.0153)[0][0]
    mins = np.zeros([pzgdms[0].size])
    mins[:]=1e99
    maxs = np.zeros([pzgdms[0].size])
    for i,pzgdm in enumerate(pzgdms):
        pzgdm = np.cumsum(pzgdm)
        pzgdm /= pzgdm[-1]
        print("cum prob for ",i," is ",pzgdm[iz])
        
        higher = np.where(pzgdm > maxs)
        maxs[higher] = pzgdm[higher]
        
        lower = np.where(pzgdm < mins)
        mins[lower] = pzgdm[lower]
        
        #if i==0:
        #    l1,=plt.plot(zvals,pzgdm,linewidth=1,color='gray',label='90% parameter limits',linestyle=":")
        #else:
        #    plt.plot(zvals,pzgdm,linewidth=1,color='gray',linestyle=":")
    for i,min in enumerate(mins):
        print(i,zvals[i],mins[i],maxs[i])
    plt.fill_between(zvals,mins,y2=maxs,alpha=0.3,color='gray',label='90% parameter limits')
    
    std_pzgdm = np.cumsum(std_pzgdm)
    std_pzgdm /= std_pzgdm[-1]
    print("cum prob for Std is ",std_pzgdm[iz])
    l2,=plt.plot(zvals,std_pzgdm,linewidth=3,color='blue',label='Previous best fit')
    
    new_pzgdm = np.cumsum(new_pzgdm)
    new_pzgdm /= new_pzgdm[-1]
    print("cum prob for New Emax is ",new_pzgdm[iz])
    l3,=plt.plot(zvals,new_pzgdm,linewidth=3,color='orange',label='New Emax',linestyle="--")
    
    plt.plot([1.01,1.01],[0,1],linestyle='-', color='black')
    plt.text(1.02,0.3,'$z_{\\rm 20220610A}$',rotation=90)
    
    plt.xlim(0,1.5)
    plt.ylim(0,1)
    plt.legend(loc=(0.12,0.05),handles=[l2,l3])#,l1])
    plt.tight_layout()
    plt.savefig(opdir+'cumulative_pzgdm_FRB220610_v2.pdf')
    plt.close()
    
def plot_expectations(name,sdir,vparams,opfile,dmhost=False):
    '''
    Initialise surveys and grids for specified parameters
    Plots zdm distribution
    Gets p(z|DM) for 20220610A
    
    Args:
        Name (string): survey name to load
        sdir (string): place to load survey from
        vparams (dict): parameters for the grid
        opfile (string): output name for the plot
        dmhost (bool): if True, performs p(z|dm) analysis for this parameter set
            The result is a series of plots giving p(DM_host) data.
    
    Returns:
        zvals (np.ndarray) z values for the below
        p(z|dm) (np.ndarray) Probability of redshift z given the DM of 20220610a
    
    '''
    # hard-coded values for this FRB
    DMEG220610=1458-31-50
    Z220610=1.0153
    
    zvals = []
    dmvals = []
    nozlist = []
    state = set_special_state()
    s, g = loading.survey_and_grid(
        survey_name=name, NFRB=None, sdir=sdir, lum_func=2,
        init_state=state
        )  # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set up new parameters
    g.update(vparams)
    # gets cumulative rate distribution
    
    # does plot of p(DM|z)
    ddm=g.dmvals[1]-g.dmvals[0]
    dz=g.zvals[1]-g.zvals[0]
    idm=int(DMEG220610/ddm)
    iz=int(Z220610/dz)
    
    pdmgz = g.rates[iz,:]
    cpdmgz = np.cumsum(pdmgz)
    cpdmgz /= cpdmgz[-1]
    print("Cumulative probability to observed dm is ",cpdmgz[idm])
    
    pzgdm = g.rates[:,idm]/np.sum(g.rates[:,idm])/dz
    
    logmean=g.state.host.lmean
    
    if True:
        vparams["lmean"] = 0.01
        g.update(vparams)
        ############# do 2D plots ##########
        opfile=opfile[:-4]+"_nohost.pdf"# removes .pdf
        
        misc_functions.plot_grid_2(
            g.rates,
            g.zvals,
            g.dmvals,
            name=opfile,
            norm=3,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]",
            project=False,
            FRBDM=s.DMEGs,
            FRBZ=s.frbs["Z"],
            Aconts=[0.01, 0.1, 0.5],
            zmax=1.5,
            DMmax=2000,
            DMlines=nozlist,
            Macquart=g.state,
            H0=g.state.cosmo.H0
        )
    # dows calculation for host DM
    # does plot of p(DM|z)
    
    
    if dmhost:
        # probability distribution
        dmdist = g.grid[iz,:]
        #dm vals
        dmhost = (DMEG220610-g.dmvals)
        iOK=np.where(dmhost > 0.)[0]
        temp=np.cumsum(dmdist)
        temp /= temp[-1]
        print("Cumulative distribution for DM host up until zero is ",temp)
        print("iOK is ",iOK)
        print("Meow",temp[iOK[-1]])
        exit()
        plt.figure()
        plt.plot(dmhost,temp)
        plt.savefig('temp_pdm_cumulative.pdf')
        plt.close()
        dmhost = dmhost[iOK]
        dmdist = dmdist[iOK]
        dmdist /= np.sum(dmdist) * ddm #normalisation to a probability distribution
        
        
        # prior
        #args=[(g.state.host.lmean-np.log10(1.+Z220610))/0.4342944619,g.state.host.lsigma/0.4342944619]
        #*(1+Z220610)
        priors=pcosmic.linlognormal_dlin(dmhost*(1.+Z220610),2.2/0.4342944619,0.5/0.4342944619)
        norm_priors=np.sum(priors)*ddm
        priors /= norm_priors
        
        convolve = priors * dmdist
        convolve /= np.sum(convolve)*ddm
        
        plt.figure()
        plt.ylim(0,0.0035)
        plt.xlim(0,1000)
        plt.xlabel('Observer frame ${\\rm DM}_{\\rm host} \\, [{\\rm pc\\,cm}^{-3}]$')
        plt.ylabel('$p({\\rm DM}_{\\rm host})$')
        plt.plot(dmhost,dmdist,label="No prior on host")
        plt.plot(dmhost,priors,label="Best fit host prior")
        plt.plot(dmhost,convolve,label="Posterior")
        leg=plt.legend()
        plt.tight_layout()
        plt.savefig('220610/p_dmhost.pdf')
        
        imax = np.argmax(dmdist)
        dmmax = dmhost[imax]
        print("Peak DM host is ",dmmax*(1+Z220610))
        
        v1,v2,k1,k2=ac.extract_limits(dmhost,dmdist,0.16) # 0.16 = (1-68%)/2
        print("220610: 1 sigma bounds are ",v1*(1+Z220610),v2*(1+Z220610))
        v1,v2,k1,k2=ac.extract_limits(dmhost,dmdist,0.48) # 0.49 ~ (1-0)/2
        print("Median is ",v1*(1+Z220610),v2*(1+Z220610))
        
        # compare to the other one!
        # 20190520B
        zFAST = 0.241
        DMFAST = 1204.7-50.-60.
        izFAST = int(zFAST/dz)
        
        # probability distribution
        dmdistFAST = g.grid[izFAST,:]
        #dm vals
        dmhostFAST = (DMFAST-g.dmvals)
        iOK=np.where(dmhostFAST > 0.)[0]
        dmhostFAST = dmhostFAST[iOK]
        dmdistFAST = dmdistFAST[iOK]
        dmdistFAST /= np.sum(dmdistFAST) * ddm #normalisation to a probability distribution
        
        v1,v2,k1,k2=ac.extract_limits(dmhostFAST,dmdistFAST,0.16) # 0.16 = (1-68%)/2
        print("FAST: 1 sigma bounds are ",v1*(1+zFAST),v2*(1+zFAST))
        v1,v2,k1,k2=ac.extract_limits(dmhostFAST,dmdistFAST,0.42) # 0.49 ~ (1-0)/2
        print("FAST: Median is ",v1*(1+zFAST),v2*(1+zFAST))
        
        priorsFAST=pcosmic.linlognormal_dlin(dmhostFAST*(1.+zFAST),2.2/0.4342944619,0.5/0.4342944619)
        norm_priors=np.sum(priorsFAST)*ddm
        priorsFAST /= norm_priors
        
        convolveFAST = priorsFAST * dmdistFAST
        convolveFAST /= np.sum(convolveFAST)*ddm
        
        leg.remove()
        plt.plot(dmhostFAST,dmdistFAST,label="FRB 20190520B")
        plt.plot(dmhostFAST,priorsFAST,label="host prior")
        plt.plot(dmhostFAST,convolveFAST,label="posterior")
        
        leg=plt.legend()
        plt.tight_layout()
        plt.savefig('220610/FAST_p_dmhost.pdf')
        
        z210117=0.214
        DM210117=730-50.-34.4
        iz210117 = int(z210117/dz)
        
        # probability distribution
        dmdist210117 = g.grid[iz210117,:]
        #dm vals
        dmhost210117 = (DM210117-g.dmvals)
        iOK=np.where(dmhost210117 > 0.)[0]
        dmhost210117 = dmhost210117[iOK]
        dmdist210117 = dmdist210117[iOK]
        dmdist210117 /= np.sum(dmdist210117) * ddm #normalisation to a probability distribution
        
        v1,v2,k1,k2=ac.extract_limits(dmhost210117,dmdist210117,0.16) # 0.16 = (1-68%)/2
        print("210117: 1 sigma bounds are ",v1*(1+z210117),v2*(1+z210117))
        v1,v2,k1,k2=ac.extract_limits(dmhost210117,dmdist210117,0.42) # 0.49 ~ (1-0)/2
        print("210117: Median is ",v1*(1+z210117),v2*(1+z210117))
        
        priors210117=pcosmic.linlognormal_dlin(dmhost210117*(1.+z210117),2.2/0.4342944619,0.5/0.4342944619)
        norm_priors=np.sum(priors210117)*ddm
        priors210117 /= norm_priors
        
        convolve210117 = priors210117 * dmdist210117
        convolve210117 /= np.sum(convolve210117)*ddm
        
        leg.remove()
        plt.plot(dmhost210117,dmdist210117,label="FRB 20210117")
        plt.plot(dmhost210117,priors210117,label="host prior")
        plt.plot(dmhost210117,convolve210117,label="posterior")
        
        leg=plt.legend()
        plt.tight_layout()
        plt.savefig('220610/210117_FAST_p_dmhost.pdf')
        plt.close()
        
        # rest-frame DM
        plt.figure()
        plt.ylim(0,0.0035)
        plt.xlim(0,2000)
        plt.xlabel('Rest frame ${\\rm DM}_{\\rm host}~[{\\rm pc\\,cm}^{-3}]$')
        plt.ylabel('$p({\\rm DM}_{\\rm host})$')
        plt.plot(dmhost*(1.+Z220610),dmdist/(1.+Z220610),label="No prior on host")
        plt.plot(dmhost*(1.+Z220610),priors/(1.+Z220610),label="Best fit host prior")
        plt.plot(dmhost*(1.+Z220610),convolve/(1.+Z220610),label="Posterior")
        plt.plot(dmhostFAST*(1.+zFAST),dmdistFAST/(1.+zFAST),label="FRB 20190520B")
        plt.plot(dmhostFAST*(1.+zFAST),convolveFAST/(1.+zFAST),label="Posterior")
        plt.plot(dmhost210117*(1.+z210117),dmdist210117/(1.+z210117),label="FRB 210117")
        plt.plot(dmhost210117*(1.+z210117),convolve210117/(1.+z210117),label="Posterior")
        plt.legend()
        plt.tight_layout()
        plt.savefig('220610/rest_frame_210117_FAST_p_dmhost.pdf')
        plt.close()
        
        # publication version
        plt.figure()
        plt.ylim(0,0.009)
        plt.xlim(0,1750)
        plt.xlabel('Rest frame ${\\rm DM}_{\\rm host} [{\\rm pc\\,cm}^{-3}]$')
        plt.ylabel('$p({\\rm DM}_{\\rm host})$')
        plt.plot(dmhost*(1.+Z220610),dmdist/(1.+Z220610),label="FRB 20220610A")
        plt.plot(dmhostFAST*(1.+zFAST),dmdistFAST/(1.+zFAST),label="FRB 20190520B",linestyle='--')
        plt.plot(dmhost*(1.+Z220610),priors/(1.+Z220610),label="Expectation",linestyle=":")
        leg=plt.legend()
        plt.tight_layout()
        plt.savefig('220610/dmhost.pdf')
        
        leg.remove()
        plt.plot(dmhost210117*(1.+z210117),dmdist210117/(1.+z210117),label="FRB 20210117",linestyle='-.')
        leg=plt.legend()
        plt.tight_layout()
        plt.savefig('220610/210117_dmhost.pdf')
        plt.close()
        
    return g.zvals,pzgdm

Planck_H0 = 67.4
def read_extremes(infile='planck_extremes.dat',H0=Planck_H0):
    """
    reads in extremes of parameters from a get_extremes_from_cube
    """
    f = open(infile)
    
    sets=[]
    
    for pset in np.arange(6):
        # reads the 'getting' line
        line=f.readline()
        
        pdict={}
        # gets parameter values
        for i in np.arange(7):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        sets.append(pdict)
        
        pdict={}
        # gets parameter values
        for i in np.arange(7):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        sets.append(pdict)
    return sets


#from pkg_resources import resource_filename
    
from astropy.cosmology import Planck15, Planck18
    
def set_special_state(alpha_method=1, cosmo=Planck18):
    
    ############## Initialise parameters ##############
    state = parameters.State()

    # Variable parameters
    vparams = {}
    
    vparams['FRBdemo'] = {}
    vparams['FRBdemo']['alpha_method'] = alpha_method
    vparams['FRBdemo']['source_evolution'] = 0
    
    vparams['beam'] = {}
    vparams['beam']['Bthresh'] = 0
    vparams['beam']['Bmethod'] = 5
    
    vparams['width'] = {}
    vparams['width']['Wlogmean'] = 1.70267
    vparams['width']['Wlogsigma'] = 0.899148
    vparams['width']['Wbins'] = 10
    vparams['width']['Wscale'] = 2
    vparams['width']['Wthresh'] = 0.5
    vparams['width']['Wmethod'] = 2
    
    vparams['scat'] = {}
    vparams['scat']['Slogmean'] = 0.7
    vparams['scat']['Slogsigma'] = 1.9
    vparams['scat']['Sfnorm'] = 600
    vparams['scat']['Sfpower'] = -4.
    
     # constants of intrinsic width distribution
    vparams['MW']={}
    vparams['MW']['DMhalo']=50
    
    vparams['host']={}
    vparams['energy'] = {}
    
    if vparams['FRBdemo']['alpha_method'] == 0:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.7
        vparams['energy']['alpha'] = 1.55
        vparams['energy']['gamma'] = -1.09
        vparams['FRBdemo']['sfr_n'] = 1.67
        vparams['FRBdemo']['lC'] = 3.15
        vparams['host']['lmean'] = 2.11
        vparams['host']['lsigma'] = 0.53
    elif  vparams['FRBdemo']['alpha_method'] == 1:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.4
        vparams['energy']['alpha'] = 0.65
        vparams['energy']['gamma'] = -1.01
        vparams['FRBdemo']['sfr_n'] = 0.73
        # NOTE: I have not checked what the best-fit value
        # of lC is for alpha method=1
        vparams['FRBdemo']['lC'] = 1 #not best fit, OK for a once-off
        
        vparams['host']['lmean'] = 2.18
        vparams['host']['lsigma'] = 0.48
    vparams['energy']['luminosity_function'] = 2
    state.update_param_dict(vparams)
    state.set_astropy_cosmo(cosmo)

    # Return
    return state

def beam_value(freq=1271.5,D=12,d_off=0.11):
    '''initialises a Gaussian beam
    D in m, freq in MHz
    Doff in degrees
    '''
    import scipy.constants as constants
    #calculate sigma from standard relation
    # Gauss uses sigma=0.42 lambda/N
    # uses 1.2 lambda on D
    # check with Parkes: 1.38 GHz at 64m is 14 arcmin
    HPBW=1.22*(constants.c/(freq*1e6))/D
    print("Half power beam width found to be ",HPBW)
    
    # calculates Gaussian sigma
    sigma=(HPBW/2.)*(2*np.log(2))**-0.5
    print("Corresponding sigma is ",sigma)
    
    # returns beam value at offset distance
    d_rad = d_off * np.pi/180.
    B = np.exp(-0.5*(d_rad/sigma)**2)
    print("Beam value at detection is ",B)
    return B

def convert_energy():
    state=parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # temp: does fluence-energy conversion
    cos.print_cosmology(state)
    Z220610=1.0153
    print("Luminosity distance is ",cos.dl(Z220610))
    print(Planck18.H0.value)
    Z220610+= 0.0002
    print("Luminosity distance is ",cos.dl(Z220610))
    Z220610-= 0.0004
    print("Luminosity distance is ",cos.dl(Z220610))
    #E=F*4*np.pi*(dl(z))**2/(1.+z)**(2.-alpha)
    #E *= 9.523396e22*bandwidth 
    #dl: luminosity distance (Mpc)
    E=cos.F_to_E(45.,Z220610,alpha=0.,bandwidth=1.) # now spectral scaling
    print("Energy in erg/Hz is ",E)

main()
