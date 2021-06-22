
"""
This script illustrates how to generate MC samples and evaluate them
Note that in this case, only MC samples for:
- one survey (ASKAP fly's eye)
- evaluated at the same parameters as they are generated
- without varying the detected number of bursts
is generated.

In general, one will want to generate the samples with one set of parameters,
and then evaluate them with another set of grids.
Hence the saving of 'mc_sample.npy'

"""

import argparse

import numpy as np
from zdm import zdm
#import pcosmic
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib
from pkg_resources import resource_filename
import os
import sys

import scipy as sp
import time
from matplotlib.ticker import NullFormatter
from zdm import iteration as it

from zdm import survey
from zdm import cosmology as cos
from zdm import beams

import pickle
import copy

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

##### definitions of parameters ########
	# pset defined as:
	# [0]:	log10 Emin
	# [1]:	log10 Emax
	# [2]:	alpha (spectrum: nu^alpha)
	# [3]:	gamma
	# [4]:	sfr n
	# [5}: log10 mean host DM
	# [6]: log10 sigma host DM
	
from zdm import misc_functions# import *
#import pcosmic

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():

	############## Initialise cosmology ##############
	cos.init_dist_measures()
	
	# get the grid of p(DM|z). See function for default values.
	# set new to False once this is already initialised
	zDMgrid, zvals,dmvals,H0=misc_functions.get_zdm_grid(
		new=True,plot=False,method='analytic')
	# NOTE: if this is new, we also need new surveys and grids!
	
	# constants of beam method
	thresh=0
	method=2
	
	
	# sets which kind of source evolution function is being used
	#source_evolution=0 # SFR^n scaling
	source_evolution=1 # (1+z)^(2.7n) scaling
	
	
	# sets the nature of scaling with the 'spectral index' alpha
	alpha_method=0 # spectral index interpretation: includes k-correction. Slower to update
	#alpha_method=1 # rate interpretation: extra factor of (1+z)^alpha in source evolution
	
	############## Initialise surveys ##############
	
	# constants of intrinsic width distribution
	Wlogmean=1.70267
	Wlogsigma=0.899148
	DMhalo=50
	
	#These surveys combine time-normalised and time-unnormalised samples 
	NewSurveys=False
	#sprefix='Full' # more detailed estimates. Takes more space and time
	sprefix='Std' # faster - fine for max likelihood calculations, not as pretty
	
	if sprefix=='Full':
		Wbins=10
		Wscale=2
		Nbeams=[20,20,20]
	elif sprefix=='Std':
		Wbins=5
		Wscale=3.5
		Nbeams=[5,5,10]
	
	# location for survey data
	sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/')
	if NewSurveys:
		
		print("Generating new surveys, set NewSurveys=False to save time later")
		#load the lat50 survey data
		lat50=survey.survey()
		lat50.process_survey_file(sdir+'CRAFT_class_I_and_II.dat')
		lat50.init_DMEG(DMhalo)
		lat50.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
		pwidths,pprobs=survey.make_widths(lat50,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
		efficiencies=lat50.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
		
		
		# load ICS data
		ICS=survey.survey()
		ICS.process_survey_file(sdir+'CRAFT_ICS.dat')
		ICS.init_DMEG(DMhalo)
		ICS.init_beam(nbins=Nbeams[1],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
		pwidths,pprobs=survey.make_widths(ICS,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
		efficiencies=ICS.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
		
		# load Parkes data
		pks=survey.survey()
		pks.process_survey_file(sdir+'parkes_mb_class_I_and_II.dat')
		pks.init_DMEG(DMhalo)
		pks.init_beam(nbins=Nbeams[2],method=2,plot=False,thresh=thresh) # need more bins for Parkes!
		pwidths,pprobs=survey.make_widths(pks,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
		efficiencies=pks.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
		
		
		names=['ASKAP/FE','ASKAP/ICS','Parkes/Mb']
		
		surveys=[lat50,ICS,pks]
		if not os.path.isdir('Pickle'):
			os.mkdir('Pickle')
		with open('Pickle/'+sprefix+'surveys.pkl', 'wb') as output:
			pickle.dump(surveys, output, pickle.HIGHEST_PROTOCOL)
			pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Pickle/'+sprefix+'surveys.pkl', 'rb') as infile:
			surveys=pickle.load(infile)
			names=pickle.load(infile)
			lat50=surveys[0]
			ICS=surveys[1]
			pks=surveys[2]
	print("Initialised surveys ",names)
	
	dirnames=['ASKAP_FE','ASKAP_ICS','Parkes_Mb']
	
	#### these are hard-coded best-fit parameters ####
	# initial parameter values. SHOULD BE LOGSIGMA 0.75! (WAS 0.25!?!?!?)
	# Best-fit parameter values (result from cube iteration)
	lEmin=30. # log10 in erg
	lEmax=41.84 # log10 in erg
	alpha=1.54 # spectral index. WARNING: here F(nu)~nu^-alpha in the code, opposite to the paper!
	gamma=-1.16 # slope of luminosity distribution function
	sfr_n=1.77 #scaling with star-formation rate
	lmean=2.16 # log10 mean of DM host contribution in pc cm^-3
	lsigma=0.51 # log10 sigma of DM host contribution in pc cm^-3
	C=4.19 # log10 constant in number per Gpc^-3 yr^-1 at z=0
	pset=[lEmin,lEmax,alpha,gamma,sfr_n,lmean,lsigma,C]
	
	
	# generates zdm grids for the specified parameter set
	NewGrids=False
	if sprefix=='Full':
		gprefix='best'
	elif sprefix=='Std':
		gprefix='Std_best'
	
	if NewGrids:
		print("Generating new grids, set NewGrids=False to save time later")
		grids=misc_functions.initialise_grids(surveys,zDMgrid, zvals,dmvals,pset,wdist=True,source_evolution=source_evolution,alpha_method=alpha_method)
		with open('Pickle/'+gprefix+'grids.pkl', 'wb') as output:
			pickle.dump(grids, output, pickle.HIGHEST_PROTOCOL)
	else:
		print("Loading grid ",'Pickle/'+gprefix+'grids.pkl')
		with open('Pickle/'+gprefix+'grids.pkl', 'rb') as infile:
			grids=pickle.load(infile)
	glat50=grids[0]
	gICS=grids[1]
	gpks=grids[2]
	print("Initialised grids")
	
	
	#testing_MC!!! Generate pseudo samples from lat50
	which=0
	g=grids[which]
	s=surveys[which]
	
	savefile='mc_sample.npy'
	
	try:
		sample=np.load(savefile)
		print("Loading ",sample.shape[0]," samples from file ",savefile)
	except:
		Nsamples=10000
		print("Generating ",Nsamples," samples from survey/grid ",which)
		sample=g.GenMCSample(Nsamples)
		sample=np.array(sample)
		np.save(savefile,sample)
	
	# plot some sample plots
	#do_basic_sample_plots(sample)
	
	#evaluate_mc_sample_v1(g,s,pset,sample)
	evaluate_mc_sample_v2(g,s,pset,sample)
	
	
def evaluate_mc_sample_v1(grid,survey,pset,sample,opdir='Plots'):
	"""
	Evaluates the likelihoods for an MC sample of events
	Simply replaces individual sets of z, DM, s with MC sets
	Will produce a plot of Nsamples/NFRB pseudo datasets.
	"""
	t0=time.process_time()
	
	nsamples=sample.shape[0]
	
	# get number of FRBs per sample
	Npersurvey=survey.NFRB
	# determines how many false surveys we have stats for
	Nsurveys=int(nsamples/Npersurvey)
	
	print("We can evaluate ",Nsurveys,"MC surveys given a total of ",nsamples," and ",Npersurvey," FRBs in the original data")
	
	# makes a deep copy of the survey
	s=copy.deepcopy(survey)
	
	lls=[]
	#Data order is DM,z,b,w,s
	# we loop through, artificially altering the survey with the composite values.
	for i in np.arange(Nsurveys):
		this_sample=sample[i*Npersurvey:(i+1)*Npersurvey,:]
		s.DMEGs=this_sample[:,0]
		s.Ss=this_sample[:,4]
		if s.nD==1: # DM, snr only
			ll=it.calc_likelihoods_1D(grid,s,pset,psnr=True,Pn=True,dolist=0)
		else:
			s.Zs=this_sample[:,1]
			ll=it.calc_likelihoods_2D(grid,s,pset,psnr=True,Pn=True,dolist=0)
		lls.append(ll)
	t1=time.process_time()
	dt=t1-t0
	print("Finished after ",dt," seconds")
	
	lls=np.array(lls)
	
	plt.figure()
	plt.hist(lls,bins=20)
	plt.xlabel('log likelihoods [log10]')
	plt.ylabel('p(ll)')
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.savefig(opdir+'/ll_histogram.pdf')
	plt.close()


def evaluate_mc_sample_v2(grid,survey,pset,sample,opdir='Plots',Nsubsamp=1000):
	"""
	Evaluates the likelihoods for an MC sample of events
	First, gets likelihoods for entire set of FRBs
	Then re-samples as needed, a total of Nsubsamp times
	"""
	t0=time.process_time()
	
	nsamples=sample.shape[0]
	
	# makes a deep copy of the survey
	s=copy.deepcopy(survey)
	NFRBs=s.NFRB
	
	s.NFRB=nsamples # NOTE: does NOT change the assumed normalised FRB total!
	s.DMEGs=sample[:,0]
	s.Ss=sample[:,4]
	if s.nD==1: # DM, snr only
		llsum,lllist,expected,longlist=it.calc_likelihoods_1D(grid,s,pset,psnr=True,Pn=True,dolist=2)
	else:
		s.Zs=sample[:,1]
		llsum,lllist,expected,longlist=it.calc_likelihoods_2D(grid,s,pset,psnr=True,Pn=True,dolist=2)
	
	# we should preserve the normalisation factor for Tobs from lllist
	Pzdm,Pn,Psnr=lllist
	
	# plots histogram of individual FRB likelihoods including Psnr and Pzdm
	plt.figure()
	plt.hist(longlist,bins=100)
	plt.xlabel('Individual Psnr,Pzdm log likelihoods [log10]')
	plt.ylabel('p(ll)')
	plt.tight_layout()
	plt.savefig(opdir+'/individual_ll_histogram.pdf')
	plt.close()
	
	# generates many sub-samples of the data
	lltots=[]
	for i in np.arange(Nsubsamp):
		thislist=np.random.choice(longlist,NFRBs) # samples with replacement, by default
		lltot=Pn+np.sum(thislist)
		lltots.append(lltot)
	
	plt.figure()
	plt.hist(lltots,bins=20)
	plt.xlabel('log likelihoods [log10]')
	plt.ylabel('p(ll)')
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.savefig(opdir+'/sampled_ll_histogram.pdf')
	plt.close()
	
	t1=time.process_time()
	dt=t1-t0
	print("Finished after ",dt," seconds")
	
	
def do_basic_sample_plots(sample,opdir='Plots'):
	"""
	Data order is DM,z,b,w,s
	
	"""
	if not os.path.exists(opdir):
		os.mkdir(opdir)
	zs=sample[:,0]
	DMs=sample[:,1]
	plt.figure()
	plt.hist(DMs,bins=100)
	plt.xlabel('DM')
	plt.ylabel('Sampled DMs')
	plt.tight_layout()
	plt.savefig(opdir+'/DM_histogram.pdf')
	plt.close()
	
	plt.figure()
	plt.hist(zs,bins=100)
	plt.xlabel('z')
	plt.ylabel('Sampled redshifts')
	plt.tight_layout()
	plt.savefig(opdir+'/z_histogram.pdf')
	plt.close()
	
	bs=sample[:,2]
	plt.figure()
	plt.hist(np.log10(bs),bins=5)
	plt.xlabel('log10 beam value')
	plt.yscale('log')
	plt.ylabel('Sampled beam bin')
	plt.tight_layout()
	plt.savefig(opdir+'/b_histogram.pdf')
	plt.close()
	
	ws=sample[:,3]
	plt.figure()
	plt.hist(ws,bins=5)
	plt.xlabel('width bin (not actual width!)')
	plt.ylabel('Sampled width bin')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(opdir+'/w_histogram.pdf')
	plt.close()
	
	s=sample[:,4]
	plt.figure()
	plt.hist(np.log10(s),bins=100)
	plt.xlabel('$\\log_{10} (s={\\rm SNR}/{\\rm SNR}_{\\rm th})$')
	plt.yscale('log')
	plt.ylabel('Sampled $s$')
	plt.tight_layout()
	plt.savefig(opdir+'/s_histogram.pdf')
	plt.close()
	
main()
