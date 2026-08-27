#!/usr/bin/env python
# coding: utf-8

# # Purpose
# 
# - Used to visualise HDF5 files from MCMC analysis
# - Developed to handle output files from MCMC.py and MCMC2.py
# - Produces plots for walkers
# - Produces corner plot
# - Produces more detailed analysis for the best fit parameters

# In[41]:


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import corner
import emcee
import json

from zdm import survey
from zdm import cosmology as cos
from zdm import loading as loading
from zdm import MCMC_analysis as analysis
import zdm.misc_functions as mf
import zdm.iteration as it
from zdm import parameters
from zdm.MCMC import calc_log_posterior
from astropy.cosmology import Planck18


plt.rcParams['font.size'] = 14


def main(filenames,labels,prefix,burnin=200):
    """
    Main function to process MCMC output
    
    Args:
        Filenames [list of strings]: specifies the MCMC output (.h5 files) from e.g. a slurm job.
        Labels [list of strings]: defined the latex labvels for plotting of MCMC variables
        Prefix [string]: prefix to prepend the plot files with
    """
    
    nparam=len(labels)
    
    # hard-coded truth parameters for the MC simulation
    # use the below to show other lines on this plot. E.g. to show a standard H0 value.
    # Typically used to show "correct", i.e. true, values, against the MCMC estimates
    truth = True
    if truth:
        
        tlabels = [r"$\log_{10} F$",r"$n_{\rm sfr}$", r"$\alpha$", r'$\mu_{\rm host}$', r'$\sigma_{\rm host}$',
             r'$\log_{10} E_{\rm max}$', r'$\gamma$', r"$H_0$",
             r"${\rm DM}_{\rm halo}$"]
        
        
        param_dict={r"$\log_{10} F$": -0.495,
                        r"$n_{\rm sfr}$": 2.88,
                        r"$\alpha$": -1.55,
                        r'$\mu_{\rm host}$': 2.13,
                        r'$\sigma_{\rm host}$': 0.46,
                        r"$f_{\rm sfr}$": 0.5,
                        r'$\log_{10} E_{\rm max}$': 40.9,
                        r'$\log_{10} E_{\rm min}$':38.22,
                        r'$\gamma$': -1.12,
                        r"${\rm DM}_{\rm halo}$": 68,
                        r"$H_0$": 70.63,
                        r"$\theta_0$": 0.5,
                        'min_lat': None}
        truths = [param_dict[param] for param in tlabels]
    else:
        truths = None
    
    
    samples = []

    # Q1: why multiple files? Are these independent MCMC runs that can be added together for a larger data-set?
    for i, filename in enumerate(filenames):
        reader = emcee.backends.HDFBackend(filename + '.h5')
        samples.append(reader.get_chain())


    # # Negate $\alpha$
    # 
    # - In our code we assume $\alpha$ is negative and so $\alpha=2$ here corresponds to a negative spectral index.
    # - So here, we change that to a negative for clarity

    # Make alpha negative
    a=-1
    for i, x in enumerate(labels):
        if x == r"$\alpha$":
            a = i
            break

    if a != -1:
        for sample in samples:
            sample[:,:,a] = -sample[:,:,a]  

    # plot walkers
    analysis.plot_walkers(samples,labels,prefix+"raw_walkers.png",legend=False,truths=truths)


    ######## Define burnin sample ##########
    # here are many ways to do this. But visually tends to be
    # the best. Here, a hard-coded value of 200 is used.
    # Please inspect walker
    good_samples = analysis.std_rej(samples, burnin=burnin)


    analysis.plot_autocorrelations(good_samples,prefix+"autocorrelation_times.png")

    burnin = (np.ones(len(good_samples)) * burnin).astype(int)

    analysis.plot_walkers(good_samples,labels,prefix+"final_walkers.png",burnin=burnin,legend=False,truths=truths)

    # NOTE - there is no current way to plot the final walkers with bad points removed

    # Get the final sample without burnin and without bad walkers
    final_sample = [[] for i in range(samples[0].shape[2])]

    # we now remove the burnin from each
    for j,sample in enumerate(good_samples):
        for i in range(sample.shape[2]):
            final_sample[i].append(sample[burnin[j]:,:,i].flatten())
    final_sample = np.array([np.hstack(final_sample[i]) for i in range(len(final_sample))]).T
    
    ######## Cornerplot #########
    fig = plt.figure(figsize=(nparam+1,nparam+1))
    
    therange = None
    #therange=[]
    #for i in np.arange(nparam):
    #    therange.append(None)
    therange=[[-1,0],[1,4],[-3,0],[1.5,2.5],[0.1,1],[40.5,41.3],[-1.3,-0.8],[40,100],[10,100]]
    
    titles = ['' for i in range(final_sample.shape[1])]
    print("Type of final sample is :",type(final_sample),final_sample.shape)
    
    print(len(labels))
    
    fig = corner.corner(final_sample,labels=labels, show_titles=True, titles=titles, 
                  fig=fig,title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 10}, 
                  quantiles=[0.16,0.5,0.84], truths=truths,range=therange);
    
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=8)
    
    plt.savefig(prefix+"cornerplot.png")
    
    x = np.corrcoef(final_sample.T)
    
    plt.figure()
    plt.imshow(x,origin="lower",cmap="coolwarm")#cmap="Pastel1")
    cbar = plt.colorbar(cmap="Pastel1")
    plt.clim(-1,1)
    plt.xticks(np.linspace(0.,nparam-1,nparam),labels,rotation=90)
    plt.yticks(np.linspace(0.,nparam-1,nparam),labels)
    plt.tight_layout()
    plt.savefig(prefix+"correlation.png")
    plt.close()


##### v1 #####
labels = [r"$\log_{10} F$",r"$n_{\rm sfr}$", r"$\alpha$", r'$\mu_{\rm host}$',
             r'$\sigma_{\rm host}$',r'$\log_{10} E_{\rm max}$',
             r'$\gamma$', r"$H_0$",
             r"${\rm DM}_{\rm halo}$"]

#filenames = ['Output/confidant_mcmc_v3_output_W40']
filenames = ['Output/f1.5_confidant_mcmc_v2_output_W40']
# this name gets added to all produced plots
prefix="Plots/f1.5_confidant_"
burnin=1000
main(filenames,labels,prefix,burnin=burnin)
exit()
