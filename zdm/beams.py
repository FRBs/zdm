# collection of functions to handle telescope beam effects
from pkg_resources import resource_filename
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

# Path to survey data
beams_path = os.path.join(resource_filename('zdm', 'data'), 'BeamData')

def gauss_beam(thresh=1e-3,nbins=10,freq=1.4e9,D=64,sigma=None):
	'''initialises a Gaussian beam
	D in m, freq in Hz
	e.g. Parkes HWHM is 7 arcmin at 1.4 GHz
	#thresh 1e-3 means -6.7 is range in lnb, 3.74 range in sigma
	'''
	dlnb=-np.log(thresh)/nbins #d log bin
	log10min=np.log10(thresh)
	dlog10b=log10min/nbins
	log10b=(np.arange(nbins)+0.5)*dlog10b#+log10min
	b=10**log10b
	if sigma is not None:
		sigma=sigma #keeps sigma in radians
	else:
		#calculate sigma from standard relation
		# Gauss uses sigma=0.42 lambda/N
		# uses 1.2 lambda on D
		# check with Parkes: 1.38 GHz at 64m is 14 arcmin
		HPBW=1.22*(constants.c/(freq*1e6))/D
		sigma=(HPBW/2.)*(2*np.log(2))**-0.5
	# this gives FWHP=0.23 deg = 13.8 arcmin i.e. agrees with Parkes
	#print("basic deg2 over range 1e-3 is ",2*np.pi*sigma**2*(180/np.pi)**2*6.9)
	omega_b=np.full([nbins],2*np.pi*dlnb*sigma**2) #omega_b per dlnb - makes sense
	return b,omega_b

def load_beam(suffix,basedir=beams_path):
	"""
	Retrieves beam data.
	The '_bins' file should contain the (nbins+1) bin edges
	The '_hist' file should contain solid angles within each bin
	Summing the _hist should return the total solid angle over
		which the calculation has been performed.
	
	"""
	logb=np.load(os.path.join(basedir,suffix+'_bins.npy'))
	# standard, gets best beam estimates: no truncation
	omega_b=np.load(os.path.join(basedir,suffix+'_hist.npy'))
	
	# checks if the log-bins are 10^logb or just actual b values
	#in a linear scale the first few may be zero...
	N=np.array(np.where(logb < 0))
	if N.size ==0: #it must be a linear array
		logb=np.log10(logb)
	
	if logb.size == omega_b.size+1:
		# adjust for the bin centre
		db=logb[1]-logb[0]
		logb=logb[:-1]+db/2.
	
	return logb,omega_b


def simplify_beam(logb,omega_b,nbins,thresh=0.,weight=1.5,method=1,savename=None):
	""" Simplifies a beam to smaller histogram
	
	Thresh is the threshold below which we cut out measurements.
	Defaults to including 99% of the rate. Simpler!
	weight tells us how to scale the omega_b to get effective means
	"""
	
	# Calculates relative rate as a function of beam position rate of -1.5
	b=10**logb
	rate=omega_b*b**weight
	crate=np.cumsum(rate)
	crate /= crate[-1]
	
	if method==1:
		# tries to categorise each in increments of 1/nbins
		# i.e. each bin has equal probability of detecting an FRB
		thresholds=np.linspace(0,1.,nbins+1)
		cuts=np.zeros([nbins],dtype='int')
		for i in np.arange(nbins):
			thresh=thresholds[i]
			cuts[i]=np.where(crate>thresh)[0][0] # first bin exceeding value
		
		# new arrays
		new_b=np.zeros([nbins])
		new_o=np.zeros([nbins])
		
		# separating j from i is mild protection against strange corner cases
		j=0
		for i in np.arange(nbins-1):
			start=cuts[i]
			stop=cuts[i+1]
			if start==stop:
				continue
			
			new_b[j]=np.sum(rate[start:stop]*b[start:stop])/np.sum(rate[start:stop])
			new_o[j]=np.sum(omega_b[start:stop])
			j += 1
		
		# last one manually
		start=cuts[nbins-1]
		new_b[j]=np.sum(rate[start:]*b[start:])/np.sum(rate[start:])
		new_o[j]=np.sum(omega_b[start:])
	
		# concatenates to true bins
		new_b=new_b[0:j+1]
		new_o=new_o[0:j+1]
	elif method==2:
		
		# gets the lowest bin where the cumulative rate is above the threshold
		include=np.where(crate > thresh)[0]
		
		# include every 'every' bin
		#every=(int (len(include)/nbins))+1
		
		every=len(include)/float(nbins)
		
		# new arrays
		new_b=np.zeros([nbins])
		new_o=np.zeros([nbins])
		
		#start=b.size-every*nbins
		start=include[0]
		
		for i in np.arange(0,nbins-1):
			stop=include[0]+int((i+1)*every)
			#print(i,start,stop)
			#print('   ',rate[start:stop])
			#print('   ',b[start:stop])
			new_b[i]=np.sum(rate[start:stop]*b[start:stop])/np.sum(rate[start:stop])
			new_o[i]=np.sum(omega_b[start:stop])
			start=stop
			
		
		# last ones
		
		new_b[nbins-1]=np.sum(rate[start:]*b[start:])/np.sum(rate[start:])
		new_o[nbins-1]=np.sum(omega_b[start:])
	elif method==3:
		# returns full beam! INSANE!!!!!!
		# new arrays
		new_b=b
		new_o=omega_b
	elif method==4: # tries very hard to get the first few bins, then becomes sparser from there
		# makes a log-space of binning, starting from the end and working back
		ntot=b.size
		
		# new arrays
		new_b=np.zeros([nbins])
		new_o=np.zeros([nbins])
		
		#if Nbins, places them at about ntot**(i/nbins
		start=ntot-1
		for i in np.arange(nbins):
			stop=start
			start=int(ntot-ntot**((i+1.)/nbins))
			if start>=stop: #always descends at least once
				start =stop-1
			if start < 0:
				start=0
			new_b[i]=np.sum(rate[start:stop]*b[start:stop])/np.sum(rate[start:stop])
			new_o[i]=np.sum(omega_b[start:stop])
	### plots if appropriate
	if savename is not None:
		
		# note that omega_b is just unscaled total solid angle
		plt.figure()
		plt.xlabel('$B$')
		plt.ylabel('$\\Omega(B)$/bin')
		plt.yscale('log')
		plt.xscale('log')
		plt.plot(10**logb,omega_b,label='original_binning')
		plt.plot(new_b,new_o,'ro',label='simplified',linestyle=':')
		plt.plot(10**logb,rate,label='Relative rate')
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig(savename)
		plt.close()
	
	return new_b,new_o
