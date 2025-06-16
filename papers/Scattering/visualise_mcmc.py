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


# # Load files
# 
# - labels = list of parameters (in order)
# - filenames = list of .h5 files to use (without .h5 extension)

# In[ ]:


# labels = ["sfr_n", "alpha", "lmean", "lsigma", "lEmax", "lEmin", "gamma", "H0", "DMhalo"]
# labels = [r"$n$", r"$\alpha$", r"log$\mu$", r"log$\sigma$", r"log$E_{\mathrm{max}}$", r"log$E_{\mathrm{min}}$", r"$\gamma$", r"$H_0$", "DMhalo"]
labels = [r"$\mu_w$", r"$\sigma_w$", r"$\mu_\tau$", r"$\sigma_\tau$"]

half = False

if half:

    filenames = ['MCMC_outputs/v2_mcmc_halflognormal','MCMC_outputs/v3_mcmc_halflognormal']
    # this name gets added to all produced plots
    prefix="MCMC_Plots/halflognormal_"
else:
    filenames = ['MCMC_outputs/v2_mcmc_lognormal','MCMC_outputs/v3_mcmc_lognormal']
    # this name gets added to all produced plots
    prefix="MCMC_Plots/lognormal_" # 100x100 zDM points
    
    filenames = ['MCMC_outputs/v4_mcmc_lognormal'] # 300 x 300 zdm points
    prefix = "MCMC_Plots/v4lognormal_"
    
    filenames = ['MCMC_outputs/v5_mcmc_lognormal'] # 15 beam values
    prefix = "MCMC_Plots/v5lognormal_"
    
    filenames = ['MCMC_outputs/v6_mcmc_lognormal'] # 15 beam values
    prefix = "MCMC_Plots/v6lognormal_"
    
    filenames = ['MCMC_outputs/v7_mcmc_lognormal'] #turn off p(w)
    prefix = "MCMC_Plots/v7lognormal_"
    
    filenames = ['MCMC_outputs/v8_mcmc_lognormal'] # turn off p(scat|w)
    prefix = "MCMC_Plots/v8lognormal_"

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
analysis.plot_walkers(samples,labels,prefix+"raw_walkers.png")


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


analysis.plot_walkers(good_samples,labels,prefix+"final_walkers.png",burnin=burnin)
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

plt.savefig(prefix+"cornerplot.png")
exit()
#### LEAVE HERE ######

# # Point estimates
# 
# - Use finer histogram binning than the corner plot
# - Obtain point estimates and confidence intervals using median / mode

# In[64]:


nBins = 20
win_len = int(nBins/10)
CL = 0.68

best_fit = {}

for i in range(len(labels)):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    hist, bin_edges, _ = ax.hist(final_sample[:,i], bins=nBins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    bins = -np.diff(bin_edges)/2.0 + bin_edges[1:]

    ax.set_xlabel(labels[i])
    ax.set_ylabel("P("+labels[i]+")")

    # Use mode ordered
    # ordered_idxs = np.argsort(hist)

    # sum = hist[ordered_idxs[0]] * bin_width
    # j = 1
    # while(sum < 1-CL):
    #     sum += hist[ordered_idxs[j]] * bin_width
    #     j = j+1

    # best = bins[ordered_idxs[-1]]
    # lower = bins[np.min(ordered_idxs[j:])]
    # upper = bins[np.max(ordered_idxs[j:])]

    # Use median
    best = np.quantile(final_sample[:,i], 0.5)
    # best = bins[np.argmax(hist)]
    lower = np.quantile(final_sample[:,i], 0.16)
    upper = np.quantile(final_sample[:,i], 0.84)

    best_fit[labels[i]] = best
    u_lower = best - lower
    u_upper = upper - best
    ax.axvline(lower, color='r')
    ax.axvline(best, color='r')
    ax.axvline(upper, color='r')
    # print(labels[i] + ": " + str(best) + " (-" + str(u_lower) + "/+" + str(u_upper) + ")")
    print(rf'{labels[i]}: {best} (-{u_lower}/+{u_upper})')

print(best_fit)



# In[65]:


import scipy.stats as st


# In[66]:


# nsamps = np.linspace(3, np.log10(final_sample.shape[0]/10), 30)
# nsamps = [int(10**x) for x in nsamps]
# print("Number of samps: " + str(nsamps))

# for i in range(len(labels)):
#     # nsamps = []
#     std = []
#     for j in range(len(nsamps)):
#         best = []
#         nruns = int(final_sample.shape[0] / nsamps[j])
#         for k in range(nruns):
#             # best.append(np.quantile(final_sample[nsamps[j]*k:nsamps[j]*(k+1),i], 0.5))
#             step = int(final_sample.shape[0]/nsamps[j])
#             best.append(np.quantile(final_sample[k::step,i], 0.5))
#         std.append(np.std(best))

#     # print(labels[i] + ": " + str(std))

#     line = st.linregress(np.log10(nsamps),np.log10(std))
#     x = np.linspace(nsamps[0], nsamps[-1], 50)
#     y = 1/np.sqrt(x)
#     y = y / y[0] * std[0]
#     y = 10**(line.slope*np.log10(x) + line.intercept)
#     # print(line.slope)
#     print(labels[i] + ": " + str(10**(line.slope*np.log10(final_sample.shape[0]) + line.intercept)))
#     print(str(line.slope))
#     fig = plt.figure(figsize=(6,4))
#     ax = fig.add_subplot(1,1,1)

#     ax.plot(nsamps, std)
#     ax.loglog(x,y)
#     ax.set_xlabel("Number of samples")
#     ax.set_ylabel("Standard deviation")
#     ax.set_title(labels[i])


# # Load surveys and grids
# 
# - Loads the surveys and grids with the best fit parameters from above.
# - Plots P(DM) and DMEG weights for each FRB

# In[67]:


s_names = [
    # "FAST2",
    # "FAST2_old"
    # "DSA",
    "FAST", 
    # "CRAFT_class_I_and_II", 
    # "private_CRAFT_ICS_892_14", 
    # "private_CRAFT_ICS_1300_14", 
    # "private_CRAFT_ICS_1632_14", 
    # "parkes_mb_class_I_and_II"
]
# rs_names = ["CHIME/CHIME_decbin_0_of_6",
#             "CHIME/CHIME_decbin_1_of_6",
#             "CHIME/CHIME_decbin_2_of_6",
#             "CHIME/CHIME_decbin_3_of_6",
#             "CHIME/CHIME_decbin_4_of_6",
#             "CHIME/CHIME_decbin_5_of_6"]
rs_names = []

state = parameters.State()
state.set_astropy_cosmo(Planck18) 
# state.update_params(best_fit)
# state.update_param('luminosity_function', 2)
# state.update_param('alpha_method', 0)
# state.update_param('sfr_n', 1.36)
# state.update_param('alpha', 1.5)
# state.update_param('lmean', 1.97)
# state.update_param('lsigma', 0.92)
# state.update_param('lEmax', 41.3)
# state.update_param('gamma', -0.63)
# state.update_param('H0', 70.0)
# state.update_param('DMhalo', 50.0)

if len(s_names) != 0:
    surveys, grids = loading.surveys_and_grids(survey_names = s_names, init_state=state, repeaters=False, nz=500, ndm=1400)
else:
    surveys = []
    grids = []

if len(rs_names) != 0:
    rep_surveys, rep_grids = loading.surveys_and_grids(survey_names = rs_names, init_state=state, repeaters=True, nz=500, ndm=1400)
    for s,g in zip(rep_surveys, rep_grids):
        surveys.append(s)
        grids.append(g)


# In[68]:


newC, llc = it.minimise_const_only(None, grids, surveys)
llsum = 0
for s,g in zip(surveys, grids):

    g.state.FRBdemo.lC = newC

    # Calc pdm
    rates=g.rates
    dmvals=g.dmvals
    pdm=np.sum(rates,axis=0)

    # # Calc psnr
    # min = s.SNRTHRESHs[0]
    # max = np.max(s.SNRs)
    # snrs = np.linspace(min,max, 50)
    # psnr = get_psnr(snrs, s, g)

    # Plot pdm + snr
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,2,1)
    ax.set_title(s.name)
    ax.set_xlabel("DM")
    ax.set_ylabel("P(DM)")
    ax.set_xlim(xmax=3000)
    ax.plot(dmvals, pdm)
    ax.vlines(s.DMEGs, np.zeros(len(s.DMs)), np.max(pdm)*np.ones(len(s.DMs)), ls='--', colors='r')

    # ax = fig.add_subplot(1,2,2)
    # ax.set_xlabel("log SNR")
    # ax.set_ylabel("log P(SNR)")
    # ax.plot(np.log10(snrs), np.log10(psnr))
    # ax.vlines(np.log10(s.SNRs), np.min(np.log10(psnr))*np.ones(len(s.SNRs)), np.max(np.log10(psnr))*np.ones(len(s.SNRs)), ls='--', colors='r')

    # Get expected and observed
    expected=it.CalculateIntegral(g.rates,s)
    expected *= 10**g.state.FRBdemo.lC
    observed=s.NORM_FRB

    print(s.name + " - expected, observed: " + str(expected) + ", " + str(observed))

    llsum += it.get_log_likelihood(g,s,Pn=True)


# In[ ]:


uDMGs = 0.5
# DMhalo = 100.0

fig = plt.figure(figsize=(6,4*len(s_names)))

for j,(s,g) in enumerate(zip(surveys, grids)):
    ax = fig.add_subplot(len(surveys),1,j+1)
    plt.title(s.name)
    ax.set_xlabel('DM')
    ax.set_ylabel('Weight')

    # s.DMhalo = DMhalo
    # s.init_DMEG(DMhalo)

    dmvals=g.dmvals
    DMobs=s.DMEGs

        # calc_DMG_weights(DMEGs, DMhalos, DM_ISMs, dmvals, sigma_ISM=0.5, sigma_halo=15.0, percent_ISM=True)
    dm_weights, iweights = it.calc_DMG_weights(DMobs, s.DMhalos, s.DMGs, dmvals, uDMGs)

    pdm = np.sum(g.rates, axis=0)
    pdm = pdm / np.max(pdm) * np.max(dm_weights[0])

    for i in range(len(DMobs)):
        ax.plot(dmvals[iweights[i]], dm_weights[i], '.-', label=s.frbs["TNS"][i] + " " + str(s.DMGs[i]))

    # ax.plot(dmvals, pdm) # Upper limit is not correct because grid has not been updated so efficiencies have not been recalc'd
    ax.set_xlim(right=3000)
    # ax.legend()


