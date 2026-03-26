import numpy as np


def main():
    
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
    
    ##### loads galaxy redshifts ######
    infile="Data/second_cut.csv"
    redshifts=[]
    with open(infile) as data:
        for line in data:
            words=line.split(',')
            z=float(words[17])
            redshifts.append(z)
    redshifts=np.array(redshifts)
    NZ=redshifts.size
    
    ######## evaluates priors #######
    
    prior_list = np.zeros([NZ])
    sys_prior_list = np.zeros([NZ,NSYS])
    
    for i,z in enumerate(redshifts):
        
        if z>zvals[-1]:
            raise ValueError("Warning - priors not evaluated beyond ",zvals[-1])
        
        if z<zvals[0]:
            raise ValueError("Warning - priors not evaluated below ",zvals[0])
        
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
    
    ##################################
    # normalises priors to sum to unity
    norm=np.sum(prior_list)
    prior_list /= norm
    
    sys_norm = np.sum(sys_prior_list,axis=0)
    for j in np.arange(NSYS):
        sys_prior_list[:,j] /= sys_norm[j]
    
    print("Please save this to zpriors_added.csv")
    with open(infile) as data:
        i=0
        for line in data:
            string=line[:-1]+", " + str(prior_list[i])
            for j in np.arange(12):
                string += ","+str(sys_prior_list[i,j])
            print(string)
            i += 1
    # print out results
    #for i in np.arange(NZ):
    #    print("\nGalaxy ",i," at z=",dummy_redshifts[i]," prior ",
    #        prior_list[i]," with systematic priors ",
    #        sys_prior_list[i,:])

main()
