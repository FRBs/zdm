""" 
then for DMhost, generate it ecplicitly, then you'd generate a tau from the measurements 
of Bhat et al (there's a slightly update from Cordes et al, I forget the paper), but you'd 
multiply tau by 3 for farfield; then you'd reduce tau according to (1+z)^3, and you'd simulation 
observation bias the prejudices against high tau. Then you have your sample of scattering and 
DM"host" = DMfrb-DMmw-DMcosmic (maybe with uncertainties in DMmw thrown in?), and you se if there's a correlation. 
Or rather, how many FRBs you'd need to see one.

Below, I've generated a grid by setting host to zero, and all intrinsic FRB widths to zero. 
So the width-dependent detection bias is simply due to DM smearing. You might MC sample FRBs 
from this distribution, then:
MC sample a DM host
MC sample a corresponding scattering time from Bhat et al
Determine if that FRB would actually be observed, by using 
SN' = SN_MC *[  (tsamp^2 + dm_smearing^2)/(tsamp^2 + dm_smearing^2) + tau^2). ]^0.25
the using all observed FRBs, you know their z, DM, and tau, so you can calculate 
"DM_host"_estimated via DMfrb - DMcosmic (actually, DM MW can be to first order ignored here, 
though we should follow this up), and plot tau vs that. See what you get!


For DM host - tau 
τ (DM, ν)]mw,psr = 1.90 × 10−7 ms × ν−α DM^1.5 × (1 + 3.55 × 10−5 DM^3) from the refs in Faber et al. 2024 https://arxiv.org/pdf/2405.14182
sigma logtau = 0.76
"""
import os

from astropy.cosmology import Planck18
#from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
#from zdm import survey
#from zdm import pcosmic
#from zdm import iteration as it
from zdm import loading
#from zdm import io
import matplotlib.pyplot as plt

import numpy as np
#from matplotlib import pyplot as plt
from pkg_resources import resource_filename
#import time

from frb.dm import igm


#params 
nmcs = 1000 
nu = 1. # GHz
t_samp = 1.182  # ms
bandwidth = 1.0 # MHz
snrthreshold = 10. # SNR threshold
mean_DM_MW = 80. # pc cm^-3
disp_DM_MW = 50. # pc cm^-3



def creategrid():

    # in case you wish to switch to another output directory
    #opdir = "../ScatSim/"
    #if not os.path.exists(opdir):
    #    os.mkdir(opdir)
    
    # Initialise surveys and grids
    #sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    sdir = '/Users/lmasriba/FRBs/zdm/zdm/data/Surveys'
    
    names = ["CRAFT_ICS_1300"]
    
    # essentially turns off DM host and sets all FRB widths to ~0 (or close enough)
    param_dict = {'lmean': 0.01, 'lsigma': 0.4, 'Wlogmean': -1,'Wbins': 1,
        'Wlogsigma': 0.1, 'Slogmean': -2,'Slogsigma': 0.1}
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)
    
    
    surveys, grids = loading.surveys_and_grids(survey_names = names,
        repeaters=False, sdir=sdir)
    
    # plots it
    #misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
    #        name=os.path.join(opdir,'tau_dm_grid.png'),norm=3,log=True,
    #        label='$\\log_{10} p({\\rm DM}_{\\rm EG}|z)$ [a.u.]',
    #        project=False,
    #        zmax=2.5,DMmax=3000,cmap="Oranges",Aconts=[0.01,0.1,0.5])#,
    #            pdmgz=[0.05,0.5,0.95])

    return grids[0]
    
    

def create_frbs(NMC, snr_thresh):
    # get the grid
    g = creategrid()

    # create DM_EG by sampling the grid
    frbs = g.GenMCSample(NMC)
    frbs = np.array(frbs)
    zs = frbs[:,0]
    DMcos = frbs[:,1]
    snrs = frbs[:,3]
    #bs = frbs[:,2]
    #ws = frbs[:,4]

    # create NMC  DM host and corresponding tau via the MW_PSR relation
    dm_hosts = 10**np.random.normal(loc=2.18, scale=0.42, size=NMC)
    tau_host = 10**np.random.normal(loc=np.log10(1.9e-7 * nu**-4 * dm_hosts**1.5 * (1 + 3.55e-5*dm_hosts**3)), scale=0.76) # ms

    '''
    # Create a log-log plot of dm_host and tau_host
    plt.figure(figsize=(8, 6))
    plt.loglog(dm_hosts, tau_host, 'o', markersize=5, alpha=0.7)
    # Plot the theoretical line for comparison
    dm_line = np.logspace(0, 4, 100)
    tau_line = 1.9e-7 * dm_line**1.5 * (1 + 3.55e-5 * dm_line**3)
    tau_up = 10**(np.log10(1.9e-7 * dm_line**1.5 * (1 + 3.55e-5 * dm_line**3)) + 0.76)
    tau_dw = 10**(np.log10(1.9e-7 * dm_line**1.5 * (1 + 3.55e-5 * dm_line**3)) - 0.76)
    plt.loglog(dm_line, tau_line, '-', color='k')
    plt.loglog(dm_line, tau_up, '--', color='k')
    plt.loglog(dm_line, tau_dw, '--', color='k')
    plt.xlabel('DM Host (pc cm$^{-3}$)', fontsize=12)
    plt.ylabel('$\\tau$ Host (ms)', fontsize=12)
    #plt.title('Log-Log Plot of DM Host vs Tau Host', fontsize=14)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    #plt.legend()
    plt.show()
    #plt.savefig('./dm_host_tau_host.png', dpi=300)'''


    # this was tau intrinsic but what we would observe would be x3 more scattering due to plane waves and 1/(1+z)^3 due to redshift dimming
    tau_host =  tau_host *3 / (1 + zs)**3 # ms

    # DM MW
    dm_MW = np.random.normal(loc=mean_DM_MW,scale=disp_DM_MW, size=NMC)
    dm_MW[np.where(dm_MW < 10.)[0]] = 20. # pc cm^-3

    # DM smearing
    DM_smear = 8.3 * bandwidth * (dm_hosts+DMcos+ dm_MW) * nu**-3.0 *1e-3 # ms

    # true SNR accounting for observational biases   
    true_snr = snrs * ((t_samp**2 + DM_smear**2)/(t_samp**2 + DM_smear**2 + tau_host**2))**0.25
    #print (f"True SNR: {true_snr}")
    #print (f"snrs: {snrs}")


    # select only those that are above the SNR threshold
    observable = np.where(true_snr > snr_thresh)[0]
    print (f"Number of observable FRBs: {len(observable)}")
    # select the corresponding DM host and tau host
    snrs = true_snr[observable]
    DMcos = DMcos[observable]
    zs = zs[observable]
    tau_host = tau_host[observable]
    dm_hosts = dm_hosts[observable]
    dm_MW = dm_MW[observable]

    # total DM 
    dm_frb = DMcos + dm_hosts + dm_MW   

    # DMcosmic from Macquart et al. 2020
    DMcos_macquart = [igm.average_DM(zs[i]).value for i in range(len(zs))] # pc cm^-3

    # estimated DM host
    dm_host_estimated = dm_frb - DMcos_macquart - mean_DM_MW

    # save the results
    np.savez('tau_dm_sims.npz', dm_host_est=dm_host_estimated, tau_host=tau_host, zs=zs, DMcos_macquart=DMcos_macquart, dm_frb=dm_frb, dm_MW=dm_MW,DMcos=DMcos, snrs=snrs, dm_hosts=dm_hosts)




# other function doing other stuff loading npz and stuff

# load the data
def load_data(file_path):
    """
    # Load the data from the npz file"
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    # Load the data
    data = np.load(file_path)
    dm_host = data['dm_host_est']
    tau_host = data['tau_host']
    zs = data['zs']
    DMcos_macquart = data['DMcos_macquart']
    dm_frb = data['dm_frb']
    dm_MW = data['dm_MW']
    DMcos = data['DMcos']
    snrs = data['snrs']
    dm_hosts_sampled = data['dm_hosts']
    return dm_host, tau_host, zs, DMcos_macquart, dm_frb, dm_MW, DMcos, snrs, dm_hosts_sampled


def tau_z_plot(snr_th,nobj=None):
    """
    # Create a plot of tau vs z
    """
    # Load the data
    file_path = 'tau_dm_sims.npz'
    dm_host, tau_host, zs, DMcos_macquart, dm_frb, dm_MW, DMcos, snrs, dm_hosts_sampled = load_data(file_path)

    # Filter the data based on the SNR threshold
    zs = zs[snrs > snr_th]
    tau_host = tau_host[snrs > snr_th]  
    if nobj is not None and nobj < len(zs):
        # Randomly select nobj samples
        indices = np.random.choice(len(zs), size=nobj, replace=False)
        zs = zs[indices]
        tau_host = tau_host[indices]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(zs, tau_host, c='blue', alpha=0.5)
    plt.xlabel('Redshift (z)', fontsize=12)
    plt.ylabel('$\\tau$ Host (ms)', fontsize=12)
    #plt.title('Scatter Plot of Tau Host vs Redshift', fontsize=14)

    # Plot the theoretical line for comparison
    z_line = np.linspace(0, np.max(zs), 100)
    tau_line =  np.max(tau_host) /(1+z_line)**3  
    plt.plot(z_line, tau_line, '-', color='k', label='(1+z)^-3')



    plt.grid(True)
    #plt.xlim(0, 5)
    plt.ylim(1e-7, 1e7)
    plt.xscale('linear')
    plt.yscale('log')
    plt.legend()
    #plt.savefig('./tau_z_plot_'+str(snr_th)+'.png', dpi=300)
    plt.show()

def cum_tau_plot(snr_th, nobj=None):
    """
    # Create a cumulative distribution plot of tau
    """
    # Load the data
    file_path = 'tau_dm_sims.npz'
    dm_host, tau_host, zs, DMcos_macquart, dm_frb, dm_MW, DMcos, snrs, dm_hosts_sampled = load_data(file_path)

    # Filter the data based on the SNR threshold
    tau_host = tau_host[np.where(snrs > snr_th)]  

    # make a subset of tau_host
    if nobj is not None and nobj < len(tau_host):
        # Randomly select nobj samples
        indices = np.random.choice(len(tau_host), size=nobj, replace=False)
        tau_host = tau_host[indices]

    # sort tau_host in ascending order
    #tau_host = np.sort(tau_host)
    # Create the cumulative distribution
    #cum_tau = np.cumsum(tau_host)/ np.sum(tau_host)

    # Create the cumulative distribution plot
    plt.figure(figsize=(8, 6))
    plt.hist(tau_host, bins=np.logspace(np.log10(1e-7), np.log10(1e7), 100), cumulative=True, color='blue', alpha=0.5, density=True)   
    #plt.plot(cum_tau,  color='blue', alpha=0.5)
    plt.xlabel('$\\tau$ Host (ms)', fontsize=12)
    plt.ylabel('Cumulative Count', fontsize=12)
    #plt.title('Cumulative Distribution of Tau Host', fontsize=14)
    plt.grid(True)
    #plt.xlim(1, 1e4)
    #plt.ylim(0, 1)
    plt.xscale('log')
    #plt.yscale('log')

    #plt.savefig('./cum_tau_plot_'+str(snr_th)+'.png', dpi=300)
    plt.show()


def plot_dm_tau(snr_th):
    """
    # Create a log-log plot of dm_host and tau_host"
    """
    # Load the data
    file_path = 'tau_dm_sims.npz'
    dm_host_estimated, tau_host, zs, DMcos_macquart, dm_frb, dm_MW, DMcos, snrs, dm_hosts_sampled = load_data(file_path)
    # Filter the data based on the SNR threshold
    dm_host_estimated = dm_host_estimated[np.where(snrs > snr_th)]
    tau_host = tau_host[np.where(snrs > snr_th)]


    plt.figure(figsize=(8, 6))
    plt.loglog(dm_host_estimated, tau_host, 'o', markersize=5, alpha=0.7)
    # Plot the theoretical line for comparison
    #dm_line = np.logspace(0, 4, 100)
    #tau_line = 1.9e-7 * dm_line**1.5 * (1 + 3.55e-5 * dm_line**3)
    #tau_up = 10**(np.log10(1.9e-7 * dm_line**1.5 * (1 + 3.55e-5 * dm_line**3)) + 0.76)
    #tau_dw = 10**(np.log10(1.9e-7 * dm_line**1.5 * (1 + 3.55e-5 * dm_line**3)) - 0.76)
    #plt.loglog(dm_line, tau_line, '-', color='k')
    #plt.loglog(dm_line, tau_up, '--', color='k')
    #plt.loglog(dm_line, tau_dw, '--', color='k')
    plt.xlabel('DM Host (pc cm$^{-3}$)', fontsize=12)
    plt.ylabel('$\\tau$ Host (ms)', fontsize=12)
    plt.xlim(1, 10000)
    plt.ylim(1e-7, 1e7)
    #plt.title('Log-Log Plot of DM Host vs Tau Host', fontsize=14)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    #plt.legend()
    #plt.savefig('./dm_host_est_tau_host_'+str(snr_th)+'.png' , dpi=300)
    plt.show()





# # Main function to run the simulation and plotting


#create_frbs(10000, 0)

cum_tau_plot(0,500)
#tau_z_plot(10)
#plot_dm_tau(10)