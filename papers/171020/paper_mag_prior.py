import numpy as np
from scipy import interpolate


def main():
    
    file1 = open('zpriors_added.csv', 'r')
    Lines = file1.readlines()
    rmags=[]
    ras=[]
    decs=[]
    zs=[]
    for line in Lines:
        words=line.split(',')
        rmag=words[15]
        gal=words[0]
        rmags.append(float(rmag))
        ra=float(words[7])
        dec=float(words[8])
        z=float(words[17])
        ras.append(ra)
        decs.append(dec)
        zs.append(z)
    
    
    decs=np.array(decs)
    ras=np.array(ras)
    rmags=np.array(rmags)
    zs=np.array(zs)
    
    # calculates adjusted rmags based on redshifts
    #dl = (0.01/zs)**2 #luminosity change to redshift of 0.01
    #dmag = np.log10(dl)*2.5
    #magprime=rmags+dmag
    #for i,r in enumerate(rmags):
    #    print(i,r,zs[i],dmag[i],magprime[i])
    
    #rmags=np.array([19.2233, 20.3648, 20.0347, 17.5727, 20.6629, 18.2049, 18.3037, 18.5523, 0, 19.7793, 18.6564, 18.6821, 19.7689, 17.7779, 17.8776, 19.8369, 15.1775, 17.2688, 20.6667, 17.519, 19.7686, 18.4509, 18.8426])
    #rmags[8]=np.max(rmags)
    
    priors=[]
    for rmag in rmags:
        prior=driver_sigma(rmag)
        priors.append(prior)
    priors=np.array(priors)
    priors=1./priors
    norm=np.sum(priors)
    priors = priors/norm
    
    
    
    ### output
    for i,line in enumerate(Lines):
        string=line[:-1]+"," + str(priors[i])
        print(string)

def get_z_priors(redshifts):
    
    ########## gets FRB data ##########
    #Place your redshift values here
    #dummy_redshifts=[0.01,0.06,0.007,0.001]
    NZ=len(redshifts)
    
    
    ############ reads in p(z,snr) prior data ##########
    # first priors for best-fit population
    best_prior=np.load("z_priors_bestfit.npy")
    
    # now priors for systematics
    systematic_priors=[]
    NSYS=12
    for i in np.arange(NSYS):
        prior=np.load("z_priors_systematic_"+str(i)+".npy")
        systematic_priors.append(prior)
    
    # now redshift values at which each prior is calculated
    zvals=np.load("zvalues_for_priors.npy")
    dz=zvals[1]-zvals[0]
    ######## evaluates priors #######
    
    prior_list = np.zeros([NZ])
    sys_prior_list = np.zeros([NZ,NSYS])
    
    for i,z in enumerate(redshifts):
        if z>zvals[-1]:
            raise ValueError("Warning - priors not evaluated beyond ",zvals[-1])
        
        if z<zvals[0]:
            raise ValueError("Warning - priors not evaluated below ",zvals[0]," value ",z," invalid")
        
        # linear interpolation
        iz1=int((z-zvals[0])/dz)
        iz2=iz1+1
        
        # linear weightings
        w2=(z-zvals[iz1])/dz
        w1=1.-w2
        
        # result
        p=w1*best_prior[iz1] + w2*best_prior[iz2]
        prior_list[i]=p
        for j,prior in enumerate(systematic_priors):
            p=w1*prior[iz1] + w2*prior[iz2]
            sys_prior_list[i,j]=p
    
    # normalises priors to sum to unity
    norm=np.sum(prior_list)
    prior_list /= norm
    
    sys_norm = np.sum(sys_prior_list,axis=0)
    for j in np.arange(NSYS):
        sys_prior_list[:,j] /= sys_norm[j]
    return prior_list,sys_prior_list
    # print out results
    #for i in np.arange(NZ):
    #    print("\nGalaxy ",i," at z=",redshifts[i]," prior ",
    #        prior_list[i]," with systematic priors ",
    #        sys_prior_list[i,:])
   
def write_original_format(priors):
    ### to append data to Freya's original file format ###
    file2 = open('R-MAG_CANDIDATES.csv', 'r')
    Lines2 = file2.readlines()
    
    for i,line in enumerate(Lines2):
        if i==0:
            continue
        print(line[:-1],",",str(priors[i-1])[0:6])
        
    
# Spline parameters(globals) are for rmag vs sigma
driver_tck = (np.array([15., 15., 15., 15., 30., 30., 30., 30.]),
    np.array([-6.41580144, -3.53188049, -1.68500105, -0.63090954, 0., 0., 0., 0.]), 3)
driver_spl = interpolate.UnivariateSpline._from_tck(driver_tck)


def deprecated_oldz():
    ######## gets redshift data ##########
    # must have been from old file
    # now corrected
    zs=[]
    zras=[]
    zdecs=[]
    file3 = open("mod_P.csv")
    Lines3 = file3.readlines()
    for line in Lines3:
        words=line.split(',')
        z=words[3]
        zra=words[1]
        zdec=words[2]
        zras.append(float(zra))
        zdecs.append(float(zdec))
        zs.append(float(z))
    zs=np.array(zs)
    zras=np.array(zras)
    zdecs=np.array(zdecs)
    pz,sys_pz=get_z_priors(zs)
    
    # matches coordinates
    for i,zra in enumerate(zras):
        zdec=zdecs[i]
        mindiff=1e9
        jmin=-1
        for j,ra in enumerate(ras):
            dec=decs[j]
            diff=((zdec-dec)**2+(zra-ra)**2)**0.5
            if diff < mindiff:
                mindiff=diff
                jmin=j
        #print("Found min diff of ",diff," for j= ",jmin)
    
    


def driver_sigma(rmag):
    """
    Estimated incidence of galaxies per sq arcsec with r > rmag
    using Driver et al. 2016 number counts.
    Spline parameters (globals) are for rmag vs sigma
    Args:
        rmag (float or np.ndarray): r band magnitude of galaxy
    Returns:
        float or np.ndarray:  Galaxy number density
    """
    return 10**driver_spl(rmag)
   
main()
