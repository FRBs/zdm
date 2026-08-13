import matplotlib.pyplot as plt
import numpy as np
import matplotlib


matplotlib.rcParams['image.interpolation'] = None

defaultsize=16
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    ############ reads in p(z,snr) prior data ##########
    # first priors for best-fit population
    best_prior=np.load("z_priors_bestfit.npy")
    
    # now priors for systematics
    systematic_priors=[]
    NSYS=12
    #for i in np.arange(NSYS):
    #    prior=np.load("z_priors_systematic_"+str(i)+".npy")
    #    systematic_priors.append(prior)
    
    # now redshift values at which each prior is calculated
    zvals=np.load("zvalues_for_priors.npy")
    dz=zvals[1]-zvals[0]
    
    plt.figure()
    plt.plot(zvals,best_prior,linewidth=3,color='blue')
    plt.plot([0.00867,0.00867],[0,50],linestyle="--",linewidth=3,color='red')
    plt.xlabel('$z$')
    plt.ylabel('$p(z|{\\rm DM},{\\rm SNR},w)$')
    plt.xlim(0.,0.08)
    plt.ylim(0,50)
    plt.tight_layout()
    plt.savefig('paper_plot_pz_bestfit_only.pdf')
    plt.savefig('paper_plot_pz_bestfit_only.png')
    plt.close()
    

main()
