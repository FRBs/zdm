import numpy as np
import cosmology as cos
import time

class grid:
	"""A class to hold a grid of z-dm plots
	
	Fundamental assumption: each z point represents FRBs created *at* that redshift
	Interpolation is performed under this assumption.
	
	It also assumes a linear uniform grid.
	"""
	
	def __init__(self):
		self.grid=None
		# we need to set these to trivial values to ensure correct future behaviour
		self.beam_b=np.array([1])
		self.beam_o=np.array([1])
		self.b_fractions=None
		
	def pass_grid(self,grid,zvals,dmvals):
		self.grid=grid
		self.zvals=zvals
		self.dmvals=dmvals
		self.check_grid()
		self.calc_dV()
		
		# this contains all the values used to generate grids
		# these parameters begin at None, and get filled when
		# ever something is regenerated. They are semi-hierarchical
		# in that if a low-level value is reset, high-level ones
		# get put to None.
	
	
	
	def load_grid(self,gridfile,zfile,dmfile):
		self.grid=load_data(gridfile)
		self.zvals=load_data(zfile)
		self.dmvals=load_data(dmfile)
		self.check_grid()
		self.volume_grid()
	
	
	def check_grid(self):
		
		self.nz=self.zvals.size
		self.ndm=self.dmvals.size
		self.dz=self.zvals[1]-self.zvals[0]
		self.ddm=self.dmvals[1]-self.dmvals[0]
		shape=self.grid.shape
		if shape[0] != self.nz:
			if shape[0] == self.ndm and shape[1] == self.nz:
				print("Transposing grid, looks like first index is DM")
				self.grid=self.grid.transpose
			else:
				raise ValueError("wrong shape of grid for zvals and dm vals")
		else:
			if shape[1] == self.ndm:
				print("Grid successfully initialised")
			else:
				raise ValueError("wrong shape of grid for zvals and dm vals")
		
		#checks that the grid is approximately linear to high precision
		expectation=self.dz*np.arange(0,self.nz)+self.zvals[0]
		diff=self.zvals-expectation
		maxoff=np.max(diff**2)
		if maxoff > 1e-6*self.dz:
			raise ValueError("Maximum non-linearity in z-grid of ",maxoff**0.5,"detected, aborting")
		
		expectation=self.ddm*np.arange(0,self.ndm)+self.dmvals[0]
		diff=self.dmvals-expectation
		maxoff=np.max(diff**2)
		if maxoff > 1e-6*self.ddm:
			raise ValueError("Maximum non-linearity in dm-grid of ",maxoff**0.5,"detected, aborting")
		
		
		
	def calc_dV(self):
		""" Calculates volume per steradian probed by a survey.
		
		Does this only in the z-dimension (for obvious reasons!)
		"""
		if cos.INIT==False:
			print('WARNING: cosmology not yet initiated, using default parameters.')
			cos.init_dist_measures()
		self.dV=cos.dvdtau(self.zvals)*self.dz
		
	
	def EF(self,alpha=0,bandwidth=1e9):
		"""Calculates the fluence--energy conversion factors as a function of redshift
		"""
		self.FtoE=cos.F_to_E(1,self.zvals,alpha=alpha,bandwidth=bandwidth)
	
	def set_evolution(self,n):
		""" Scales volumetric rate by SFR """
		self.sfr_n=n
		self.sfr=cos.sfr(self.zvals)**n
	
	
	# not used
	#def set_efficiencies(self, eff_func):
	#	""" Sets the efficiency function that
	#	gives the response as a function of DM.
	#	"""
	#	self.eff_func=eff_func
	#	self.efficiencies=eff_func(self.dmvals)
	def calc_pdv(self,Emin,Emax,gamma,beam_b=None,beam_o=None):
		""" Calculates the rate per cell.
		Assumed model: a power-law between Emin and Emax (erg)
		               with slope gamma.
		Efficiencies: list of efficiency response to DM
		So-far: does NOT include time x solid-angle factor
		
		NOW: this includes a solid-angle and beam factor if initialised
		
		This will recalculate beam factors if they are passed, however
		during iteration this is not recalculated
		"""
		
		if beam_b is not None:
			self.beam_b=beam_b
			self.beam_o=beam_o
			try:
				x=beam_o.shape
				x=beam_b.shape
			except:
				raise ValueError("Beam values must be numby arrays! Currently ",beam_o,beam_b)
		self.Emin=Emin
		self.Emax=Emax
		self.gamma=gamma
		# linear weighted sum of probabilities: pdVdOmega now. Could also be used to include time factor
		
		# this implementation allows us to access the b-fractions later on
		if (not (self.b_fractions is not None)) or (beam_b is not None):
			self.b_fractions=np.zeros([self.zvals.size,self.dmvals.size,self.beam_b.size])
		
		# for some arbitrary reason, we treat the beamshape slightly differently... no need to keep an intermediate product!
		for i,b in enumerate(self.beam_b):
			for j,w in enumerate(self.eff_weights):
				
				if j==0:
					self.b_fractions[:,:,i] = self.beam_o[i]*w*array_power_law(self.thresholds[j,:,:]/b,Emin,Emax,gamma)
				else:
					self.b_fractions[:,:,i] += self.beam_o[i]*w*array_power_law(self.thresholds[j,:,:]/b,Emin,Emax,gamma)
				
		# here, b-fractions are unweighted according to the value of b.
		self.fractions=np.sum(self.b_fractions,axis=2) # sums over b-axis [ we could ignore this step?]
		self.pdv=np.multiply(self.fractions.T,self.dV).T
	def calc_rates(self):
		""" multiplies the rate per cell with the appropriate pdm plot """
		
		
		try:
			self.sfr
		except:
			print("WARNING: no evolutionary weight yet applied")
			exit()
		
		try:
			self.smear_grid
		except:
			print("WARNING: no DMx smearing yet applied")
			exit()
		
		try:
			self.pdv
		except:
			print("WARNING: no volumetric probability pdv yet calculated")
			exit()
		
		self.sfr_smear=np.multiply(self.smear_grid.T,self.sfr).T
			# we do not NEED the following, but it records this info 
			# for faster computation later
			#self.sfr_smear_grid=np.multiply(self.smear_grid.T,self.sfr).T
			#self.pdv_sfr=np.multiply(self.pdv.T,self.sfr)
		
		self.rates=self.pdv*self.sfr_smear
		
		#try:
		#	self.smear_grid
		#except:
		#	print("WARNING: DM grid has not yet been smeared for DMx!")
		#	self.pdv_smear=self.pdv*self.grid
		#else:
		#	self.pdv_smear=self.pdv*self.sfr_smear
		#
		#try:
		#	self.sfr
		#except:
		#	print("WARNING: no evolutionary weight yet applied")
		#else:
		#	self.rates=np.multiply(self.pdv_smear.T,self.sfr).T
			# we do not NEED the following, but it records this info 
			# for faster computation later
			#self.sfr_smear_grid=np.multiply(self.smear_grid.T,self.sfr).T
			#self.pdv_sfr=np.multiply(self.pdv.T,self.sfr)
		
	def calc_thresholds(self, F0, eff_table, alpha=0, bandwidth=1e9,weights=None):
		""" Sets the effective survey threshold on the zdm grid
		F0: base survey threshold
		eff: table of efficiencies corresponding to DM-values
		"""
		# keep the inputs for later use
		self.F0=F0
		
		self.alpha=alpha
		self.bandwidth=bandwidth
		if eff_table.ndim==1:
			self.nthresh=1
			self.eff_weights=np.array([1])
			self.eff_table=np.array([eff_table]) # make it an extra dimension
		else:
			self.nthresh=eff_table.shape[0]
			if weights is not None:
				if weights.size != self.nthresh:
					raise ValueError("Dimension of weights must equal first dimension of efficiency table")
			else:
				raise ValueError("For a multidimensional efficiency table, please set relative weights")
			self.eff_weights=weights/np.sum(weights) #normalises this!
			self.eff_table=eff_table
		Eff_thresh=F0/self.eff_table
		
		self.EF(alpha,bandwidth) #sets FtoE values - could have been done *WAY* earlier
		
		self.thresholds=np.zeros([self.nthresh,self.zvals.size,self.dmvals.size])
		for i in np.arange(self.nthresh):
			self.thresholds[i,:,:]=np.outer(self.FtoE,Eff_thresh[i,:])
		
		
	def smear_dm(self,smear,mean,sigma):
		""" Smears DM using the supplied array.
		Example use: DMX contribution
		"""
		# just easier to have short variables for this
		
		ls=smear.size
		lz,ldm=self.grid.shape
		self.smear_mean=mean
		self.smear_sigma=sigma
		self.smear_grid=np.zeros([lz,ldm])
		self.smear=smear
		#for j in np.arange(ls,ldm):
		#	self.smear_grid[:,j]=np.sum(np.multiply(self.grid[:,j-ls:j],smear[::-1]),axis=1)
		#for j in np.arange(ls):
		#	self.smear_grid[:,j]=np.sum(np.multiply(self.grid[:,:j+1],np.flip(smear[:j+1])),axis=1)
		
		# this method is O~7 times faster than the 'brute force' above for large arrays
		for i in np.arange(lz):
			# we need to get the length of mode='same', BUT
			# we do not want it 'centred', hence must make cut on full
			if smear.ndim==1:
				self.smear_grid[i,:] = np.convolve(self.grid[i,:],smear,mode='full')[0:ldm]
			elif smear.ndim==2:
				self.smear_grid[i,:] = np.convolve(self.grid[i,:],smear[i,:],mode='full')[0:ldm]
			else:
				raise ValueError("Wrong number of dimensions for DM smearing ",smear.shape)
			
	def get_p_zgdm(self, DMs):
		""" Calcuates the probability of redshift given a DM
		We already have our grid of observed DM values.
		Just take slices!
		
		"""
		# first gets ids of matching DMs
		priors=np.zeros([DMs.size,self.zvals.size])
		for i,dm in enumerate(DMs):
			DM2=np.where(self.dmvals > dm)[0][0]
			DM1=DM2-1
			kDM=(dm-self.dmvals[DM1])/(self.dmvals[DM2]-self.dmvals[DM1])
			priors[i,:]=kDM*self.rates[:,DM2]+(1.-kDM)*self.rates[:,DM1]
			priors[i,:] /= np.sum(priors[i,:])
		return priors
	
	def copy(self,grid):
		""" copies all values from grid to here
		is OK if this is a shallow copy
		explicit function to open the possibility of making it faster
		"""
		self.grid=grid.grid
		self.beam_b=grid.beam_b
		self.b_fractions=grid.b_fractions
		self.zvals=grid.zvals
		self.dmvals=grid.dmvals
		self.nz=grid.nz
		self.ndm=grid.ndm
		self.dz=grid.dz
		self.ddm=grid.ddm
		self.dV=grid.dV
		self.FtoE=grid.FtoE
		self.sfr_n=grid.sfr_n
		self.sfr=grid.sfr
		self.beam_b=grid.beam_b
		self.beam_o=grid.beam_o
		self.Emin=grid.Emin
		self.Emax=grid.Emax
		self.gamma=grid.gamma
		self.b_fractions=grid.b_fractions
		self.fractions=grid.fractions
		self.pdv=grid.pdv
		self.smear_grid=grid.smear_grid
		self.sfr_smear=grid.sfr_smear
		self.rates=grid.rates
		self.F0=grid.F0
		self.alpha=grid.alpha
		self.bandwidth=grid.bandwidth
		self.nthresh=grid.nthresh
		self.eff_weights=grid.eff_weights
		self.eff_table=grid.eff_table
		self.thresholds=grid.thresholds
		self.smear_mean=grid.smear_mean
		self.smear_sigma=grid.smear_sigma
		self.smear=grid.smear
		self.smear_grid=grid.smear_grid
		
		
		
def array_power_law(Eth,Emin,Emax,gamma):
	""" Calculates the fraction of bursts above a certain power law
	for a given Eth, where Eth is an N-dimensional array
	"""
	dims=Eth.shape
	Eth=Eth.flatten()
	if gamma >= 0: #handles crazy dodgy cases. Or just return 0?
		result=np.zeros([Eth.size])
		result[np.where(Eth < Emax)]=1.
		result=result.reshape(dims)
		Eth=Eth.reshape(dims)
		return result
	result=array_power_law2(Eth,Emin,Emax,gamma)
	result=result.reshape(dims)
	return result


def array_power_law2(Eth,Emin,Emax,gamma):		
	result=(Eth**gamma-Emax**gamma ) / (Emin**gamma-Emax**gamma )
	# should not happen
	low=np.where(Eth < Emin)[0]
	if len(low) > 0:
		result[low]=1.
	high=np.where(Eth > Emax)[0]
	if len(high)>0:
		result[high]=0.
	
	return result

def array_diff_power_law(Eth,Emin,Emax,gamma):
	""" Calculates the differential fraction of bursts for a power law
	at a given Eth, where Eth is an N-dimensional array
	"""
	dims=Eth.shape
	Eth=Eth.flatten()
	if gamma >= 0: #handles crazy dodgy cases. Or just return 0?
		result=np.zeros([Eth.size])
		result[np.where(Eth < Emax)]=1.
		result=result.reshape(dims)
		Eth=Eth.reshape(dims)
		return result
	
	result=array_diff_power_law2(Eth,Emin,Emax,gamma)
	result=result.reshape(dims)
	return result


def array_diff_power_law2(Eth,Emin,Emax,gamma):
	result=-(gamma*Eth**(gamma-1)) / (Emin**gamma-Emax**gamma )
	
	low=np.where(Eth < Emin)[0]
	if len(low) > 0:
		result[low]=0.
	high=np.where(Eth > Emax)[0]
	if len(high) > 0:
		result[high]=0.
	
	return result

def vector_power_law(Eth,Emin,Emax,gamma):
	""" Calculates the fraction of bursts above a certain power law
	for a given Eth.
	"""
	result=(Eth**gamma-Emax**gamma ) / (Emin**gamma-Emax**gamma )
	low=np.where(Eth < Emin)[0]
	if len(low) > 0:
		result[low]=1.
	high=np.where(Eth > Emax)[0]
	if len(high)>0:
		results[high]=0.
	return result

# power-laws here are differential
def power_law_norm(Emin,Emax,gamma):
	""" Calculates the normalisation factor for a power-law """
	return Emin**gamma-Emax**-gamma

def power_law(Eth,Emin,Emax,gamma):
	""" Calculates the fraction of bursts above a certain power law
	for a given Eth.
	"""
	if Eth <= Emin:
		return 1
	elif Eth >= Emax:
		return 0
	else:
		return (Eth**gamma-Emax**gamma ) / (Emin**gamma-Emax**gamma )

def load_data(filename):
	if filename.endswith('.npy'):
		data=np.load(filename)
	elif filename.endswith('.txt') or filename.endswith('.txt'):
		# assume a simple text file with whitespace separator
		data=np.loadtxt(filename)
	else:
		raise ValueError('unrecognised type on z-dm file ',filename,' cannot read data')
	return data


