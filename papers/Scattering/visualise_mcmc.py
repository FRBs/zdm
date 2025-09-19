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


plt.rcParams['font.size'] = 16


def main(filenames,labels,prefix):
    """
    Main function to process MCMC output
    
    Args:
        Filenames [list of strings]: specifies the MCMC output (.h5 files) from e.g. a slurm job.
        Labels [list of strings]: defined the latex labvels for plotting of MCMC variables
        Prefix [string]: prefix to prepend the plot files with
    """
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
    analysis.plot_walkers(samples,labels,prefix+"raw_walkers.png",legend=False)


    ######## Define burnin sample ##########
    # here are many ways to do this. But visually tends to be
    # the best. Here, a hard-coded value of 200 is used.
    # Please inspect walker
    good_samples = analysis.std_rej(samples, burnin=200)


    analysis.plot_autocorrelations(good_samples,prefix+"autocorrelation_times.png")


    # # Implement burnin and change priors

    burnin = 250 # define this by inspecting the raw walkers
    # keep burnin the same over all samples - in theory, this could be different though
    # (but probably should not be!)
    burnin = (np.ones(len(good_samples)) * burnin).astype(int)

    # removes the burnin from each sample


    analysis.plot_walkers(good_samples,labels,prefix+"final_walkers.png",burnin=burnin,legend=False)

    xlim=[1000,1200]
    analysis.plot_walkers(good_samples,labels,prefix+"final_walkers_zoom.png",burnin=burnin,legend=False,xlim=xlim)

    # NOTE - there is no current way to plot the final walkers with bad points removed

    # Get the final sample without burnin and without bad walkers
    final_sample = [[] for i in range(samples[0].shape[2])]

    # we now remove the burnin from each
    for j,sample in enumerate(good_samples):
        for i in range(sample.shape[2]):
            final_sample[i].append(sample[burnin[j]:,:,i].flatten())
    final_sample = np.array([np.hstack(final_sample[i]) for i in range(len(final_sample))]).T

    # - Changes prior to discard samples outside the specified prior range
    # - Implements the burnin using either the predefined burnin or a constant specified
    # e.g.:
    # final_sample = analysis.change_priors(final_sample, 5, min=38)
    # final_sample = analysis.change_priors(final_sample, 7, max=110.0)
    # final_sample = analysis.change_priors(final_sample, 9, max=80.0)
    # final_sample = analysis.change_priors(final_sample, 1, max=1.0, min=-3.5)


    ######## Cornerplot #########

    # use the below to show other lines on this plot. E.g. to show a standard H0 value.
    # Typically used to show "correct", i.e. true, values, against the MCMC estimates
    truth = False
    if truth:
        lmean = 2.27
        DMhalo = np.log10(50.0)
        param_dict={'logF': np.log10(0.32), 'sfr_n': 1.13, 'alpha': 1.5, 'lmean': lmean, 'lsigma': 0.55, 
                'lEmax': 41.26, 'lEmin': 39.5, 'gamma': -0.95, 'DMhalo': DMhalo, 'H0': 73,
                'min_lat': None}
        truths = [param_dict[param] for param in labels]
    else:
        truths = None

    fig = plt.figure(figsize=(12,12))

    titles = ['' for i in range(final_sample.shape[1])]
    corner.corner(final_sample,labels=labels, show_titles=True, titles=titles, 
                  fig=fig,title_kwargs={"fontsize": 15},label_kwargs={"fontsize": 15}, 
                  quantiles=[0.16,0.5,0.84], truths=truths);
    print(prefix+"cornerplot.png")
    plt.savefig(prefix+"cornerplot.png")


sfr = "sfr_"
labels = [r"$\mu_w$", r"$\sigma_w$", r"$\mu_{\tau}$", r"$\sigma_\tau$"]
if sfr != "":
    labels.append(r"$n_{\rm sfr}$")

for half in ["","hl_"]:

    filenames = ['MCMC_outputs/mcmc_lognormal_'+half+sfr+'v1',
                'MCMC_outputs/mcmc_lognormal_'+half+sfr+'v2',
                'MCMC_outputs/mcmc_lognormal_'+half+sfr+'v3',
                'MCMC_outputs/mcmc_lognormal_'+half+sfr+'v4']
    # this name gets added to all produced plots
    prefix="MCMC_Plots/"+half+sfr
    main(filenames,labels,prefix)
