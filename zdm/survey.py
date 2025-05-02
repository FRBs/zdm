# ###############################################
# This file defines a class to hold an FRB survey
# Essentially, this is relevant when multiple
# FRBs are discovered by the same instrument
# ##############################################

import numpy as np
import os
from pkg_resources import resource_filename
from scipy.integrate import quad
from dataclasses import dataclass, fields

import pandas
from astropy.table import Table
import json

from ne2001 import density

from zdm import beams, parameters
from zdm import pcosmic
from zdm import survey_data
from zdm import galactic_dm_models
from zdm import misc_functions
import matplotlib.pyplot as plt

from IPython import embed
import warnings

from astropy import units as u
from astropy.coordinates import SkyCoord

class OldSurvey:
    """A class to hold an FRB survey

    Attributes:
        frbs (dict): Holds the data for the FRBs

    """
    def __init__(self):
        self.init=False
        self.do_beam=False
        #self.DM=[] #observed
        #self.w=[] #observed widths
        #self.fbar= #mean frequency
        #self.sens_meth="None"
    
    def get_efficiency(self,DMlist,model="Quadrature",dsmear=True,edir=None):
        """ Gets efficiency to FRBs
        Returns a list of relative efficiencies
        as a function of dispersion measure for each FRB.
        """
        efficiencies=np.zeros([self.NFRB,DMlist.size])
        for i in np.arange(self.NFRB):
            efficiencies[i,:]=calc_relative_sensitivity(self.DMs[i],DMlist,self.WIDTHs[i],
                self.FBARs[i],self.TRESs[i],self.FRESs[i],model=model,dsmear=dsmear,edir=edir)
        # keep an internal record of this
        self.efficiencies=efficiencies
        self.DMlist=DMlist
        self.wplist=np.array([1])# weight of 1
        mean_efficiencies=np.mean(efficiencies,axis=0)
        self.mean_efficiencies=mean_efficiencies
        return efficiencies
    
    def get_efficiency_from_wlist(self,DMlist,wlist,plist, 
                                  model="Quadrature", 
                                  addGalacticDM=True,
                                  edir=None):
        """ Gets efficiency to FRBs
        Returns a list of relative efficiencies
        as a function of dispersion measure for each width given in wlist
        
        
        DMlist:
            - list of dispersion measures (pc/cm3) at which to calculate efficiency
        
        wlist:
            list of intrinsic FRB widths
        
        plist:
            list of relative probabilities for FRBs to have widths of wlist
        
        model: method of estimating efficiency as function of width, DM, and time resolution
            Takes values of "Quadrature" or "Sammons" (from Mawson Sammons summer project)
        
        addGalacticDM:
            - True: this routine adds in contributions from the MW Halo and ISM, i.e.
                it acts like DMlist is an extragalactic DM
            - False: just used the supplied DMlist
        
        """
        efficiencies=np.zeros([wlist.size,DMlist.size])
        if addGalacticDM:
            # the following is safe against surveys with zero FRBs
            toAdd = self.DMhalo + self.meta["DMG"]
        else:
            toAdd = 0.
        
        for i,w in enumerate(wlist):
            efficiencies[i,:]=calc_relative_sensitivity(
                None,DMlist+toAdd,w,
                np.median(self.frbs['FBAR']),
                np.median(self.frbs['TRES']),
                np.median(self.frbs['FRES']),
                model=model,
                dsmear=False,
                edir=edir)
        # keep an internal record of this
        self.efficiencies=efficiencies
        self.wplist=plist
        self.wlist=wlist
        self.DMlist=DMlist
        mean_efficiencies=np.mean(efficiencies,axis=0)
        self.mean_efficiencies=mean_efficiencies #be careful here!!! This may not be what we want!
        return efficiencies
    
    def process_survey_file(self,filename:str, NFRB:int=None,
                            iFRB:int=0):
        """ Loads a survey file, then creates 
        dictionaries of the loaded variables 

        Args:
            filename (str): Survey filename
            NFRB (int, optional): Use only a subset of the FRBs in the Survey file.
                Mainly used for Monte Carlo analysis
            iFRB (int, optional): Start grabbing FRBs at this index
                Mainly used for Monte Carlo analysis
                Requires that NFRB be set
        """
        info=[]
        keys=[]
        self.meta={} # dict to contain survey metadata, in dictionary format
        self.frbs={} # dict to contain arrays of data for each FRB proprty
        basename=os.path.basename(filename)
        name=os.path.splitext(basename)[0]
        self.name=name
        self.iFRB = iFRB

        # read in raw data from survey file
        nlines=0
        with open(filename) as infile:
            i=-1
            for line in infile:
                # initial split. Identifies keys and removes blank and comment lines
                i += 1
                line=line.strip()
                if line=="": #remove comment lines
                    continue
                elif line[0]=='#': #remove blank lines
                    continue
                else:
                    nocomments=line.split("#")[0]
                    words=nocomments.split()
                    key=words[0]
                    keys.append(key)
                    rest=words[1:]
                    info.append(rest)
                    nlines+=1
        
        #### Find the number of FRBs in the file ###
        self.info=info
        self.keys=keys

        # NFRB
        if NFRB is None:
            self.NFRB=keys.count('FRB')
        else:
            self.NFRB = min(keys.count('FRB'),NFRB)
        if self.NFRB==0:
            print('No FRBs found in file '+filename) #was error, but actually we want this
        
        self.meta['NFRB']=self.NFRB
        
        #### separates FRB and non-FRB keys
        self.frblist=self.find(keys,'FRB')

        if NFRB is not None:
            # Take the first set - ensures we do not overrun the total number of FRBs
            if self.NFRB < NFRB+iFRB:
                raise ValueError("Cannot return sufficient FRBs, did you mean NFRB=None?")
            themax = min(NFRB+iFRB,self.NFRB)
            self.frblist=self.frblist[iFRB:themax]
        
        ### first check for the key li        self.meta['NBINS']=int(self.meta['NBINS'])et the FRB table
        iKEY=self.do_metakey('KEY')
        self.keylist=info[iKEY]
        
        # the following can only be metadata
        which=1
        self.do_keyword_char('BEAM',which,None) # prefix of beam file
        self.do_keyword('TOBS',which,None) # total observation time, hr
        self.do_keyword('DIAM',which,None) # Telescope diamater (in case of Gauss beam)
        self.do_keyword('NBEAMS',which,1) # Number of beams (multiplies sr)
        self.do_keyword('NORM_FRB',which,self.NFRB) # number of FRBs to norm obs time by
        self.do_keyword('NBINS',which,1) # Number of bins for the analysis
        # Hack to recast as int
        self.meta['NBINS']=int(self.meta['NBINS'])
        
        # Adding beam keywords for backwards compatibility
        # for CHIME: 3,0,0,CHIME
        # Otherwise: 2,0,2,Quadrature
        if False:
            # set for CHIME. In theory, could test for this using
            # the filename.
            self.do_keyword('BMETHOD',which,3) # method to describe beam
            self.do_keyword('BTHRESH',which,0) # mon beam value to include
            self.do_keyword('WMETHOD',which,0) # method to deal with FRB widths
            self.do_keyword('WBIAS',which,"CHIME") # bias effect due to FRB width
        else:
            # set for all other FRBs
            self.do_keyword('BMETHOD',which,2) # method to describe beam
            self.do_keyword('BTHRESH',which,0) # mon beam value to include
            self.do_keyword('WMETHOD',which,2) # method to deal with FRB widths
            self.do_keyword('WBIAS',which,"Quadrature") # bias effect due to FRB width
        
        # the following properties can either be FRB-by-FRB, or metadata
        which=3
        # perhaps we should set "-1" as the default for all of these,
        # to force people to manually enter that data, and not
        # accidentally use ASKAP default values?
        self.do_keyword('THRESH',which)
        self.do_keyword('TRES',which,1.265)
        self.do_keyword('FRES',which,1)
        self.do_keyword('FBAR',which,1196)
        self.do_keyword('BW',which,336)
        self.do_keyword('SNRTHRESH',which,9.5)
        self.do_keyword('DMG',which,35) # Galactic contribution to DM, defaults to 35 if no FRBs present
        self.do_keyword('NREP',which,1) # listed under either, since all FRBs could indeed by once-off
        
        
        # The following properties can only be FRB-by-FRB
        which=2
        self.do_keyword('SNR',which)
        self.do_keyword('DM',which)
        self.do_keyword('WIDTH',which,0.1) # defaults to unresolved width in time
        self.do_keyword_char('TNS',which,None, dtype='str') # obviously we don't need names,!
        self.do_keyword('Gl',which,None) # Galactic longitude
        self.do_keyword('Gb',which,None) # Galactic latitude
        #
        self.do_keyword_char('XRA',which,None, dtype='str') # obviously we don't need names,!
        self.do_keyword_char('XDec',which,None, dtype='str') # obviously we don't need names,!
        
        self.do_keyword('Z',which,None)

        if self.frbs["Z"] is not None:
            
            self.Zs=self.frbs["Z"]
            # checks for any redhsifts identically equal to zero
            #exactly zero can be bad... only happens in MC generation
            # 0.001 is chosen as smallest redshift in original fit
            zeroz = np.where(self.Zs == 0.)[0]
            if len(zeroz) >0:
                self.Zs[zeroz]=0.001
            
            # checks to see if there are any FRBs which are localised
            self.zlist = np.where(self.Zs > 0.)[0]
            if len(self.zlist) < self.NFRB:
                self.nozlist = np.where(self.Zs < 0.)[0]
                self.nD=3 # code for both
            else:
                self.nozlist = None
                self.nD=2
        else:
            self.nD=1
            self.Zs=None
            self.nozlist=np.arange(self.NFRB)
            self.zlist=None
        
        ### processes galactic contributions
        self.process_dmg()
        
        
        ### get pointers to correct results ,for better access
        self.DMs=self.frbs['DM']
        self.DMGs=self.frbs['DMG']
        self.SNRs=self.frbs['SNR']
        self.WIDTHs=self.frbs['WIDTH']
        self.TRESs=self.frbs['TRES']
        self.FRESs=self.frbs['FRES']
        self.FBARs=self.frbs['FBAR']
        self.BWs=self.frbs['BW']
        self.THRESHs=self.meta['THRESH']
        self.SNRTHRESHs=self.meta['SNRTHRESH']
        self.NORM_FRB=self.meta['NORM_FRB']
        
        self.Ss=self.SNRs/self.SNRTHRESHs
        self.TOBS=self.meta['TOBS']
        self.Ss[np.where(self.Ss < 1.)[0]]=1
        
        # sets the 'beam' values to unity by default
        self.beam_b=np.array([1])
        self.beam_o=np.array([1])
        self.NBEAMS=1
        
        self.init=True
        print("FRB survey sucessfully initialised with ",self.NFRB," FRBs starting from", self.iFRB)
        
    def init_DMEG(self,DMhalo):
        """ Calculates extragalactic DMs assuming halo DM """
        self.DMhalo=DMhalo
        self.DMEGs=self.DMs-self.DMGs-DMhalo
    
    def do_metakey(self,key):
        """ This kind of key can *only* be metadata, it cannot be an FRb-by-FRB property """
        n=self.keys.count(key)
        if (n != 1):
            raise ValueError('Key ',key,' should appear once and only once, list is ',self.keys)
            
        elif (n==1):
            i=self.keys.index(key)
            self.meta[key]=self.info[i]
        return i
    
    def do_keyword(self,key,which=3,default=-1):
        """ This kind of key can either be in the metadata, or the table, not both
        IF which ==     1: must be metadata
                2: must be FRB-by-FRB
                3: could be either
        
        If default ==     None: it is OK to not be present
                -1: fail if not there
                Another value: set this as the default
        """
        n=self.keys.count(key)
        if (n > 1): # 
            raise ValueError('Repeat information: key ',key,' appears more than once')
        elif (which !=2) and (n==1): # single piece of info for all FRBs
            ik=self.keys.index(key)
            self.meta[key]=float(self.info[ik][0])
            self.frbs[key]=np.full([self.NFRB],float(self.info[ik][0])) # fills with this value
            
        elif (which != 1) and (self.keylist.count(key)==1) and self.NFRB >0: #info varies according to each FRB
            ik=self.keylist.index(key)
            mean=0.
            values=np.zeros([self.NFRB])
            for i,j in enumerate(self.frblist):
                values[i]=float(self.info[j][ik])
                mean += values[i]
            self.frbs[key] = values
            self.meta[key] = mean/self.NFRB
        else:
            if default==None:
                # do nothing if default is None - does not need to be present
                self.meta[key]=None
                self.frbs[key]=None
            elif default == '-1':
                raise ValueError('No information on ',key,' available')
            else:
                self.meta[key]=default
                self.frbs[key]=np.full([self.NFRB],default)
    
    def do_keyword_char(self,key:str,
                        which:int,default=-1, 
                        dtype='float'):
        """
        Slurp in a set of keywords

        This kind of key can either be in the metadata, 
        or the table, not both

        IF which ==     1: must be metadata
                2: must be FRB-by-FRB
                3: could be either
        
        If default ==     None: it is OK to not be present
                -1: fail if not there
                Another value: set this as the default

        Args:
            key (str): [description]
            which (int): [description]
            default (int, optional): [description]. Defaults to -1.
            dtype (str, optional): Data type for the variable. Defaults to 'float'.

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        n=self.keys.count(key)
        if (n > 1): # 
            raise ValueError('Repeat information: key ',key,' appears more than once')
        elif (which !=2) and (n==1): # single piece of info for all FRBs
            ik=self.keys.index(key)
            self.meta[key]=self.info[ik][0]
            self.frbs[key]=np.full([self.NFRB],self.info[ik][0]) # fills with this value
            
        elif (which != 1) and (self.keylist.count(key)==1): #info varies according to each FRB
            ik=self.keylist.index(key)
            if dtype == 'str':
                values = []
                for i,j in enumerate(self.frblist):
                    values.append(self.info[j][ik])
                values = np.array(values)
            else:
                values=np.zeros([self.NFRB], dtype=dtype)
                for i,j in enumerate(self.frblist):
                    values[i]=self.info[j][ik]
            self.frbs[key] = values
        else:
            if default==None:
                self.meta[key]=None
                self.frbs[key]=None
            elif default == '-1':
                raise ValueError('No information on ',key,' available')
            else:
                self.meta[key]=default
                self.frbs[key]=np.full([self.NFRB],default)

    
    def process_dmg(self):
        """ Estimates galactic DM according to
        Galactic lat and lon only if not otherwise provided
        """
        
        if self.frbs["DMG"] is None:
            if self.frbs["Gl"] is None or self.frbs["Gb"] is None:
                raise ValueError('Can not estimate Galactic contributions.\
                    Please enter Galactic coordinates, or else manually enter \
                    it as DMG')
            print("Calculating DMG from NE2001. Please record this, it takes a while!")
            ne = density.ElectronDensity() #default position is the sun
            DMGs=np.zeros([self.NFRB])
            for i,l in enumerate(self.frbs["Gl"]):
                b=self.frbs["Gb"][i]
                
                ismDM = ne.DM(l, b, 100.)
            
                print(i,l,b,ismDM)
            DMGs=np.array(DMGs)
            self.frbs["DMG"]=DMGs
            self.DMGs=DMGs
            
    def find(self,lst, val):
            return [i for i, x in enumerate(lst) if x==val]
    
    def get_key(self,key):
        if key in self.keys:
            return self.keys
    
    def init_beam(self,plot=False,method=1,thresh=1e-3,Gauss=False):
        """ Initialises the beam """
        if Gauss:
            b,omegab=beams.gauss_beam(thresh=thresh,
                                      nbins=self.meta['NBINS'],
                                      freq=self.meta["FBAR"],D=self.meta["DIAM"])
            self.beam_b=b
            self.beam_o=omegab*self.meta["NBEAMS"]
            self.orig_beam_b=self.beam_b
            self.orig_beam_o=self.beam_o
            
        elif self.meta["BEAM"] is not None:
            
            logb,omegab=beams.load_beam(self.meta["BEAM"])
            self.orig_beam_b=10**logb
            self.orig_beam_o=omegab
            if plot:
                savename='Plots/Beams/'+self.name+'_'+self.meta["BEAM"]+'_'+str(method)+'_'+str(thresh)+'_beam.pdf'
            else:
                savename=None
            b2,o2=beams.simplify_beam(logb,omegab,self.meta['NBINS'],
                                      savename=savename,method=method,thresh=thresh)
            self.beam_b=b2
            self.beam_o=o2
            self.do_beam=True
            # sets the 'beam' values to unity by default
            self.NBEAMS=b2.size
            
        else:
            print("No beam found to initialise...")
            
    def __repr__(self):
        """ Over-ride print representation

        Returns:
            str: Items of the FURBY
        """
        repr = '<{:s}: \n'.format(self.__class__.__name__)
        repr += f'name={self.name}'
        return repr
    

class Survey:
    def __init__(self, state, survey_name:str, 
                 filename:str, 
                 dmvals:np.ndarray,
                 NFRB:int=None, 
                 iFRB:int=0,
                 edir=None,
                 rand_DMG=False):
        """ Init an FRB Survey class

        Args:
            state (_type_): _description_
            survey_name (str): 
                Name of the survey
            filename (str): _description_
            dmvals (np.ndarray): _description_
            NFRB (int, optional): _description_. Defaults to None.
            iFRB (int, optional): _description_. Defaults to 0.
            edir (str, optional): Location of efficiency files
            rand_DMG (bool): If true randomise the DMG values within uncertainty
        """
        # Proceed
        self.name = survey_name
        self.dmvals = dmvals
        # Load up
        self.process_survey_file(filename, NFRB, iFRB, min_lat=state.analysis.min_lat, dmg_cut=state.analysis.DMG_cut)
        # Check if repeaters or not and set relevant parameters
        # Now done in loading
        # self.repeaters=False
        # self.init_repeaters()
        # DM EG
        self.init_halo_coeffs()
        if rand_DMG:
            self.randomise_DMG(state.MW.sigmaDMG)

        self.init_DMEG(state.MW.DMhalo, state.MW.halo_method)
        # Zs
        self.init_zs() # This should be redone every time DMhalo is changed IF we use a flat cutoff on DMEG
        # Allows survey metadata to over-ride parameter defaults if present.
        # This is required when mixing CHIME and non-CHIME FRBs
        beam_method = self.meta['BMETHOD']
        beam_thresh = self.meta['BTHRESH']
        width_bias = self.meta['WBIAS']
        
        self.init_beam(
                       method=beam_method, 
                       plot=False, 
                       thresh=beam_thresh) # tells the survey to use the beam file
        # Efficiency: width_method passed through "self" here
        pwidths,pprobs=make_widths(self, state)
        _ = self.get_efficiency_from_wlist(dmvals,
                                       pwidths,pprobs,model=width_bias, edir=edir) 
        self.calc_max_dm()

    def init_repeaters(self):
        """
        Checks to see if this is a repeater survey and if so ensures all the
        relevant information is present.
        """
        # self.repeaters = self.meta["REPEATERS"]
        self.repeaters = True
        
        # checks to see if NREP info is present. If not, assumes all single bursts
        if not "NREP" in self.frbs:
            print("Warning, no information of repetition provided")
            print("Assuming all FRBs are once-off bursts")
            self.frbs["NREP"] = np.full([self.NFRB],1,dtype='int')
        
        # Set repeater/singles list
        self.replist = np.where(self.frbs["NREP"] > 1)[0]
        self.singleslist = np.where(self.frbs["NREP"] == 1)[0]
        
        #------------------------------------------------------------------
        self.drift_scan = self.meta['DRIFT_SCAN']
        
        if self.drift_scan == 2:
            self.Nfields = 1
            self.Tfield = self.TOBS
        elif self.drift_scan == 1:
            # Check we have the necessary data to construct Nfields and Tfield        
            self.Nfields = self.meta['NFIELDS']
            self.Tfield = self.meta['TFIELD']
            
            if self.Nfields is None:
                if self.Tfield is None or self.TOBS is None:
                    raise ValueError("At least 2 of NFIELDS, TFIELD and TOBS must be set in repeater surveys")
                else:
                    self.Nfields = int(round(self.TOBS / self.Tfield))
                    # To account for rounding errors - TOBS is used anyways
                    self.Tfield = self.TOBS / self.Nfields
            elif self.Tfield is None:
                if self.TOBS is None:
                    raise ValueError("At least 2 of NFIELDS, TFIELD and TOBS must be set in repeater surveys")
                else:
                    self.Tfield = self.TOBS / self.Nfields
            elif self.TOBS is None:
                self.TOBS = self.Nfields * self.Tfield
            else:
                # All three are set
                if abs(self.Tfield - self.TOBS / self.Nfields) > 0.001:
                    raise ValueError("WARNING: Inconsistent values of Tfield, Tobs, Nfields: ",
                            self.Tfield, self.TOBS, self.Nfields)
             
            
        #------------------------------------------------------------------
        # Check we have number of repeaters / singles
        self.NORM_REPS = self.meta['NORM_REPS']
        self.NORM_SINGLES = self.meta['NORM_SINGLES']
        
        if self.NORM_FRB == 0:
            # Case of no detections
            if (self.NORM_REPS is None and self.NORM_SINGLES is None) or (self.NORM_REPS == 0 or self.NORM_SINGLES == 0):
                self.NORM_REPS = 0
                self.NORM_SINGLES = 0
            # Case invalid
            elif self.NORM_REPS is None or self.NORM_SINGLES is None:
                raise ValueError("At least 2 of NORM_FRB, NORM_REPS and NORM_SINGLES must be set in repeater surveys")
            # Case of NORM_FRB not set
            else:
                self.NORM_FRB = self.NORM_REPS + self.NORM_SINGLES
                # print("NORM_FRB set to NORM_REPS + NORM_SINGLES = " + str(self.NORM_FRB))
        elif self.NORM_REPS is None:
            # Case invalid
            if self.NORM_SINGLES is None or self.NORM_SINGLES > self.NORM_FRB:
                raise ValueError("At least 2 of NORM_FRB, NORM_REPS and NORM_SINGLES must be set in repeater surveys and NORM_FRB = NORM_REPS + NORM_SINGLES")
            # Case NORM_REPS not set
            else:
                self.NORM_REPS = self.NORM_FRB - self.NORM_SINGLES
                # print("NORM_REPS set to NORM_FRB - NORM_SINGLES = " + str(self.NORM_REPS))
        elif self.NORM_SINGLES is None:
            # Do not need to consider NORM_REPS == None as that is done in the previous elif
            # Case invalid
            if self.NORM_REPS > self.NORM_FRB:
                raise ValueError("At least 2 of NORM_FRB, NORM_REPS and NORM_SINGLES must be set in repeater surveys and NORM_FRB = NORM_REPS + NORM_SINGLES")
            # Case NORM_SINGLES not set
            else:
                self.NORM_SINGLES = self.NORM_FRB - self.NORM_REPS
                # print("NORM_SINGLES set to NORM_FRB - NORM_REPS = " + str(self.NORM_SINGLES))
        # Case all 3 are set
        else:
            # Common sense check
            if self.NORM_FRB != self.NORM_REPS + self.NORM_SINGLES:
                raise ValueError("NORM_FRB != NORM_REPS + NORM_SINGLES")
        
        # Initialise repeater zs
        self.init_zs_reps()

    def randomise_DMG(self, uDMG=0.5):
        """ Change the DMG_ISM values to a random value within uDMG Gaussian uncertainty """
        
        new_DMGs = np.random.normal(self.DMGs, uDMG*self.DMGs)
        neg = np.where(new_DMGs < 0)[0]
        while len(neg) != 0:
            new_DMGs[neg] = np.random.normal(self.DMGs[neg], uDMG*self.DMGs[neg])
            neg = np.where(new_DMGs < 0)[0]
        
        self.DMGs = new_DMGs

    def init_DMEG(self,DMhalo,halo_method=0):
        """ Calculates extragalactic DMs assuming halo DM """
        self.DMhalo=DMhalo
        self.process_dmhalo(halo_method)
        self.DMEGs=self.DMs-self.DMGs - self.DMhalos
        # self.DMEGs=self.DMs-self.DMGals
        # self.DMEGs_obs=self.DMs-self.DMGs-DMhalo
        # self.DMEGs = np.copy(self.DMEGs_obs)
        # self.DMEGs[self.DMEGs < 0] = 10. # Minimum value of 10. pc/cm^3
    
    def process_dmhalo(self, halo_method):
        """
        Calculates directionally dependent DMhalo from Yamasaki and Totani 2020 
        and rescaling to an average of self.DMhalo
        
        self.c, self.Gls and self.Gbs should be loaded in process_survey_file
        """

        # Constant halo
        if halo_method == 0:
            self.DMhalos = np.ones(self.DMs.shape) * self.DMhalo
            # self.DMGals = self.DMhalos + self.DMGs
    
        # Yamasaki and Totani 2020
        elif halo_method == 1:
            no_coords = np.where(self.Gls == 1.0)[0]
            # if np.any(np.isnan(self.XRA[no_coords])) or np.any(np.isnan(self.XDec[no_coords])):
            #     raise ValueError('Galactic coordinates must be set if using directional dependence')
            
            if len(no_coords) != 0:
                for i in no_coords:
                    coords = SkyCoord(ra=self.XRA[i], dec=self.XDec[i], frame='icrs', unit="deg")

                    self.Gls[i] = coords.galactic.l.value
                    self.Gbs[i] = coords.galactic.b.value

            if np.any(self.Gls == 1.0) or np.any(self.Gbs == 1.0):
                raise ValueError('Galactic coordinates must be set if using directional dependence')

            for i in range(len(self.Gls)):
                if self.Gls[i] > 180.0:
                    self.Gls[i] = 360.0 - self.Gls[i]

            # Convert to rads
            self.Gls = self.Gls * np.pi / 180
            self.Gbs = self.Gbs * np.pi / 180

            # Evaluate each one
            self.DMhalos = np.zeros(self.DMs.shape, dtype='float')
            for i in range(8):
                for j in range(8-i):
                    self.DMhalos = self.DMhalos + self.c[i][j] * np.abs(self.Gls)**i * np.abs(self.Gbs)**j
            
            self.DMhalos = self.DMhalos * self.DMhalo / 43
            # self.DMGals = self.DMhalos + self.DMGs

        # Sanskriti et al. 2020
        elif halo_method == 2:
            no_coords = np.where(self.Gls == 1.0)[0]
            # if np.any(np.isnan(self.XRA[no_coords])) or np.any(np.isnan(self.XDec[no_coords])):
            #     raise ValueError('Galactic coordinates must be set if using directional dependence')
            
            if len(no_coords) != 0:
                for i in no_coords:
                    coords = SkyCoord(ra=self.XRA[i], dec=self.XDec[i], frame='icrs', unit="deg")
                    self.Gls[i] = coords.galactic.l.value
                    self.Gbs[i] = coords.galactic.b.value

            if np.any(self.Gls == 1.0) or np.any(self.Gbs == 1.0):
                raise ValueError('Galactic coordinates must be set if using directional dependence')
        
            self.DMhalos = np.zeros(len(self.frbs))
            self.DMhalo = 0
            # self.DMGals = np.zeros(len(self.frbs))
            self.DMG_el = np.zeros(len(self.frbs))
            self.DMG_eu = np.zeros(len(self.frbs))
            for i, (Gl, Gb) in enumerate(zip(self.Gls, self.Gbs)):
                self.DMGs[i], self.DMG_el[i], self.DMG_eu[i] = galactic_dm_models.dmg_sanskriti2020(Gl, Gb)

        # self.DMGal = np.median(self.DMGals)

    def init_halo_coeffs(self):
        """
        Initialise coefficients for Yamasaki and Totani 2020 implementation of
        directionally dependent DMhalo
        """
        self.c = [
            [250.12, -871.06, 1877.5, -2553.0, 2181.3, -1127.5, 321.72, -38.905],
            [-154.82, 783.43, -1593.9, 1727.6, -1046.5, 332.09, -42.815],
            [-116.72, -76.815, 428.49, -419.00, 174.60, -27.610],
            [216.67, -193.30, 12.234, 32.145, -8.3602],
            [-129.95, 103.80, -22.800, 0.44171],
            [39.652, -21.398, 2.7694],
            [-6.1926, 1.6162,],
            [0.39346]
        ]

    def init_zs(self):
        """
        Gets zlist and nozlist and determines which z values to use
        """
        # Ignore redshifts above MAX_LOC_DMEG
        self.min_noz = self.meta["MAX_LOC_DMEG"]
        # Ignore redshifts above the minimum unlocalised DM if MAX_LOC_DMEG==0
        if self.min_noz == 0:
            nozlist = np.where(self.frbs["Z"] < 0.)[0]
            if len(nozlist != 0):
                self.min_noz = np.min(self.DMEGs[nozlist])

        # Do not get rid of redshifts if MAX_LOC_DMEG==-1
        if self.min_noz > 0:
            high_dm = np.where(self.DMEGs > self.min_noz)[0]
            self.ignored_Zs = self.frbs["Z"].values[high_dm]
            self.ignored_Zlist = high_dm[self.ignored_Zs > 0]
            self.ignored_Zs = self.ignored_Zs[self.ignored_Zs > 0]
            self.frbs["Z"].values[high_dm] = -1.0
            print("Ignoring redshifts with DMEG > " + str(self.min_noz))
        else:
            self.ignored_Zs = []
            self.ignored_Zlist = []

        # Pandas resolves None to Nan
        if len(self.frbs["Z"])>0 and np.isfinite(self.frbs["Z"].values[0]):
            
            self.Zs=self.frbs["Z"].values
            # checks for any redhsifts identically equal to zero
            #exactly zero can be bad... only happens in MC generation
            # 0.001 is chosen as smallest redshift in original fit
            zeroz = np.where(self.Zs == 0.)[0]
            if len(zeroz) >0:
                self.Zs[zeroz]=0.001
            
            # checks to see if there are any FRBs which are localised
            self.zlist = np.where(self.Zs > 0.)[0]
            if len(self.zlist) < self.NFRB:
                self.nozlist = np.where(self.Zs < 0.)[0]
                if len(self.nozlist) == len(self.Zs):
                    self.nD=1 # they all had -1 as their redshift!
                    self.zlist=None
                else:
                    self.nD=3 # code for both
            else:
                self.nozlist = None
                self.nD=2
        else:
            self.nD=1
            self.Zs=None
            self.nozlist=np.arange(self.NFRB)
            self.zlist=None
        
    def init_zs_reps(self):
        """
        Gets zlist and nozlist for repeaters and singles. Basically the same as init_zs but for repeaters.
        """
        # Case of no repeaters detected
        if len(self.replist) == 0:
            self.nDr = 1
            self.zreps = None
            self.nozreps = np.arange(0)

            # This also accounts for the case of no FRBs at all
            self.nDs = self.nD
            self.zsingles = self.zlist
            self.nozsingles = self.nozlist

        # Case of no singles (all repeaters)
        elif len(self.singleslist) == 0:
            self.nDs = 1
            self.zsingles = None
            self.nozsingles = np.arange(0)
            
            self.nDr = self.nD
            self.zreps = self.zlist
            self.nozreps = self.nozlist
        
        # Case of some singles and some repeaters
        else:
            if self.nD == 1:
                self.zreps = None
                self.zsingles = None
                self.nozreps = np.array([i for i in self.replist if i in self.nozlist])
                self.nozsingles = np.array([i for i in self.singleslist if i in self.nozlist])
                self.nDr = 1
                self.nDs = 1
            elif self.nD == 2:
                self.zreps = np.array([i for i in self.replist if i in self.zlist])
                self.zsingles = np.array([i for i in self.singleslist if i in self.zlist])
                self.nozreps = None
                self.nozsingles = None
                self.nDr = 2
                self.nDs = 2
            else:
                self.nozreps = np.array([i for i in self.replist if i in self.nozlist])
                self.zreps = np.array([i for i in self.replist if i in self.zlist])
                self.nozsingles = np.array([i for i in self.singleslist if i in self.nozlist])
                self.zsingles = np.array([i for i in self.singleslist if i in self.zlist])

                # Repeater dimensions
                if len(self.nozreps) == 0:
                    self.nozreps = None
                    self.nDr = 2
                elif len(self.zreps) == 0:
                    self.zreps = None
                    self.nDr = 1
                else:
                    self.nDr = 3
        
                # Singles dimensions
                if len(self.nozsingles) == 0:
                    self.nozsingles = None
                    self.nDs = 2
                elif len(self.zsingles) == 0:
                    self.zsingles = None
                    self.nDs = 1
                else:
                    self.nDs = 3
            

    def process_survey_file(self,filename:str, 
                            NFRB:int=None,
                            iFRB:int=0,
                            min_lat=None,
                            dmg_cut=None): 
        """ Loads a survey file, then creates 
        dictionaries of the loaded variables 

        Args:
            filename (str): Survey filename
            NFRB (int, optional): Use only a subset of the FRBs in the Survey file.
                Mainly used for Monte Carlo analysis
            iFRB (int, optional): Start grabbing FRBs at this index
                Mainly used for Monte Carlo analysis
                Requires that NFRB be set
            original (bool, optional):
        """
        self.iFRB = iFRB
        self.NFRB = NFRB
        self.meta = {}

        # Read
        frb_tbl = Table.read(filename, format='ascii.ecsv')
        # Survey Data
        self.survey_data = survey_data.SurveyData.from_jsonstr(
            frb_tbl.meta['survey_data'])
        # Meta -- for convenience for now;  best to migrate away from this
        for key in self.survey_data.params:
            DC = self.survey_data.params[key]
            self.meta[key] = getattr(self.survey_data[DC],key)
        
        # Get default values from default frb data
        default_frb = survey_data.FRB()
        
        # we now populate missing fields with the default values
        for field in fields(default_frb):
            # checks to see if this is a field in metadata: if so, takes priority
            if field.name in self.meta.keys():
                default_value = self.meta[field.name]
            else:
                default_value = getattr(default_frb, field.name)
            
            # now checks for missing data, fills with the default value
            if field.name in frb_tbl.columns:
                
                # iterate over fields, checking if they are populated
                for i,val in enumerate(frb_tbl[field.name]):
                    if isinstance(val,np.ma.core.MaskedArray):
                        frb_tbl[field.name][i] = default_value
            else:
                default_value = getattr(default_frb, field.name)
                frb_tbl[field.name] = default_value
                print("WARNING: no ",field.name," found in survey",
                    "replcing with default value of ",default_value)
        
        self.frbs = frb_tbl.to_pandas()
        
        # Cut down?
        # NFRB
        if self.NFRB is not None:
            self.NFRB=min(len(self.frbs), NFRB)
            if self.NFRB < NFRB+iFRB:
                raise ValueError("Cannot return sufficient FRBs, did you mean NFRB=None?")
            # Not sure the following linematters given the Error above
            themax = max(NFRB+iFRB,self.NFRB)
            self.frbs=self.frbs[iFRB:themax]
        
        # Min latitude
        if min_lat is not None and min_lat > 0.0:
            excluded = 0
            # blanks = np.where(self.frbs['Gb'].values == None)[0]

            # if len(blanks) > 0:
            #     warnings.warn("Some FRBs have no Gb value, using DMG cut of 50 instead", UserWarning)
            # self.frbs = self.frbs[self.frbs['Gb'].values != None]

            # excluded += len(self.frbs[np.abs(self.frbs['Gb'].values) <= min_lat])
            # self.frbs = self.frbs[np.abs(self.frbs['Gb'].values) > min_lat]

            frbs =  []

            for i, frb in self.frbs.iterrows():
                if np.isnan(frb['Gb']):
                    warnings.warn("FRB " + frb['TNS'] + " has no Gb value, using DMG cut of 50 instead", UserWarning)
                    if frb['DMG'] < 50:
                        frbs.append(frb)
                    else:
                        excluded += 1
                elif np.abs(frb['Gb']) > min_lat:
                    frbs.append(frb)
                else:
                    excluded += 1
            
            print("Using minimum galactic latitude of " + str(min_lat) + ". Excluding " + str(excluded) + " FRBs")
        # Max DM
        if dmg_cut is not None:
            self.frbs = self.frbs[np.abs(self.frbs['DMG'].values) < dmg_cut]
        # Get new number of FRBs
        self.NFRB = len(self.frbs)
        
        # Vet
        vet_frb_table(self.frbs, mandatory=True)
        
        print("Loaded FRB info")
        
        if len(self.frbs) > 0:
            # first, replacing missing values with survey values
            
            # replace default values with observed media values
            # it's unclear if median or mean is the best here
            self.meta['SNRTHRESH'] = np.median(self.frbs['SNRTHRESH'])
            self.meta['THRESH'] = np.median(self.frbs['THRESH'])
            self.meta['BW'] = np.median(self.frbs['BW'])
            self.meta['FBAR'] = np.median(self.frbs['FBAR'])
            self.meta['FRES'] = np.median(self.frbs['FRES'])
            self.meta['TRES'] = np.median(self.frbs['TRES'])
            self.meta['WIDTH'] = np.median(self.frbs['WIDTH'])
            self.meta['DMG'] = np.mean(self.frbs['DMG'])
        
        # fills in missing coordinates is possible
        self.fix_coordinates(verbose=False)
        
        ### processes galactic contributions
        self.process_dmg()
        
        ### get pointers to correct results ,for better access
        self.DMs=self.frbs['DM'].values
        self.DMGs=self.frbs['DMG'].values
        self.SNRs=self.frbs['SNR'].values
        self.WIDTHs=self.frbs['WIDTH'].values
        self.TRESs=self.frbs['TRES'].values
        self.FRESs=self.frbs['FRES'].values
        self.FBARs=self.frbs['FBAR'].values
        self.BWs=self.frbs['BW'].values
        self.THRESHs=self.frbs['THRESH'].values
        self.SNRTHRESHs=self.frbs['SNRTHRESH'].values
        self.Ss=self.SNRs/self.SNRTHRESHs
        self.TOBS=self.meta['TOBS']
        self.NORM_FRB=self.meta['NORM_FRB']

        # sets the 'beam' values to unity by default
        self.beam_b=np.array([1])
        self.beam_o=np.array([1])
        self.NBEAMS=1
        
        # checks for incorrectSNR values
        toolow = np.where(self.Ss < 1.)[0]
        if len(toolow) > 0:
            print("FRBs ",toolow," have SNR < SNRTHRESH!!! Please correct this. Exiting...")
            exit()
        
        print("FRB survey sucessfully initialised with ",self.NFRB," FRBs starting from", self.iFRB)

    def fix_coordinates(self,verbose=False):
        """
        Takes and FRB, and fills out missing coordinate values
        Note that now, RA, DEC, Gl, and Gb will be present
        But their default values are None
        """
        
        
        for i,gl in enumerate(self.frbs['Gl']):
            if gl is None or self.frbs['Gb'][i] is None:
                # test RA
                if self.frbs['RA'][i] is None or self.frbs['DEC'][i] is None:
                    if verbose:
                        print("WARNING: no coordinates calculable for FRB ",i)
                else:
                    Gb,Gl = misc_functions.j2000_to_galactic(self.frbs['RA'][i], self.frbs['DEC'][i])
                    self.frbs['Gb'][i] = Gb
                    self.frbs['Gl'][i] = Gl
            elif self.frbs['RA'][i] is None or self.frbs['DEC'][i] is None:
                RA,Dec = misc_functions.galactic_to_j2000(gl, self.frbs['Gb'][i])
                self.frbs['RA'][i] = RA
                self.frbs['DEC'][i] = Dec
    
    def process_dmg(self):
        """ Estimates galactic DM according to
        Galactic lat and lon only if not otherwise provided
        """
        if len(self.frbs["TNS"].values) != 0 and not np.isfinite(self.frbs["DMG"].values[0]):
            print("Checking Gl and Gb")
            if np.isfinite(self.frbs["Gl"].values[0]) and np.isfinite(self.frbs["Gb"].values[0]):
                raise ValueError('Can not estimate Galactic contributions.\
                    Please enter Galactic coordinates, or else manually enter \
                    it as DMG')
            print("Calculating DMG from NE2001. Please record this, it takes a while!")
            ne = density.ElectronDensity()
            DMGs=np.zeros([self.NFRB])
            for i,l in enumerate(self.frbs["Gl"]):
                b=self.frbs["Gb"][i]
                ismDM = ne.DM(l, b, 100.)
                print(i,l,b,ismDM)
            DMGs=np.array(DMGs)
            self.frbs["DMG"]=DMGs
            self.DMGs=DMGs
        

    def init_beam(self,plot=False,
                  method=1,thresh=1e-3):
        """ Initialises the beam """
        # Gaussian beam if method == 0
        if method==0:
            b,omegab=beams.gauss_beam(thresh=thresh,
                                      nbins=self.meta["NBINS"],
                                      freq=self.meta["FBAR"],D=self.meta["DIAM"])
            self.beam_b=b
            self.beam_o=omegab*self.meta["NBEAMS"]
            self.orig_beam_b=self.beam_b
            self.orig_beam_o=self.beam_o
            
        elif self.meta["BEAM"] is not None:
            
            logb,omegab=beams.load_beam(self.meta["BEAM"])
            self.orig_beam_b=10**logb
            self.orig_beam_o=omegab
            if plot:
                savename='Plots/Beams/'+self.name+'_'+self.meta["BEAM"]+'_'+str(method)+'_'+str(thresh)+'_beam.pdf'
            else:
                savename=None
            b2,o2=beams.simplify_beam(logb,omegab,self.meta["NBINS"],
                                      savename=savename,method=method,thresh=thresh)
            self.beam_b=b2
            self.beam_o=o2
            self.do_beam=True
            # sets the 'beam' values to unity by default
            self.NBEAMS=b2.size
            
        else:
            print("No beam found to initialise...")

    def calc_max_dm(self):
        '''
        Calculates the maximum searched DM.
        
        Calculates bandwidth using 
        '''
        fbar=self.meta['FBAR']
        t_res=self.meta['TRES']
        nu_res=self.meta['FRES']
        max_idt=self.meta['MAX_IDT']
        max_dm=self.meta['MAX_DM']

        if max_dm is None and max_idt is not None:
            k_DM=4.149 #ms GHz^2 pc^-1 cm^3
            #f_low = fbar - (Nchan/2. - 1)*nu_res
            #f_high = fbar + (Nchan/2. - 1)*nu_res
            f_low = fbar - self.meta['BW']/2. # bottom of lowest band
            f_high = fbar + self.meta['BW']/2. # top of highest band
            max_dt = t_res * max_idt
            max_dm = max_dt / (k_DM * ((f_low/1e3)**(-2) - (f_high/1e3)**(-2)))

        self.max_dm = max_dm

    def get_efficiency_from_wlist(self,DMlist,wlist,plist, 
                                  model="Quadrature", 
                                  addGalacticDM=True,
                                  edir=None):
        """ Gets efficiency to FRBs
        Returns a list of relative efficiencies
        as a function of dispersion measure for each width given in wlist
        
        
        DMlist:
            - list of dispersion measures (pc/cm3) at which to calculate efficiency
        
        wlist:
            list of intrinsic FRB widths
        
        plist:
            list of relative probabilities for FRBs to have widths of wlist
        
        model: method of estimating efficiency as function of width, DM, and time resolution
            Takes values of "Quadrature", "Sammons" (from Mawson Sammons summer project),
                            "CHIME" or a file name
        
        addGalacticDM:
            - True: this routine adds in contributions from the MW Halo and ISM, i.e.
                it acts like DMlist is an extragalactic DM
            - False: just used the supplied DMlist
        
        edir:
            - Directory where efficiency files are contained. Only relevant if specific FRB responses are used
        
        """
        efficiencies=np.zeros([wlist.size,DMlist.size])
        
        if addGalacticDM:
            # toAdd = self.DMhalo + self.meta['DMG']
            toAdd = np.median(self.DMhalos + self.DMGs)
            # toAdd = self.DMGal
        else:
            toAdd = 0.
        
        for i,w in enumerate(wlist):
            efficiencies[i,:]=calc_relative_sensitivity(
                None,DMlist+toAdd,w,
                self.meta['FBAR'],
                self.meta['TRES'],
                self.meta['FRES'],
                max_idt=self.meta['MAX_IDT'],
                max_dm=self.meta['MAX_DM'],
                model=model,
                dsmear=False,
                edir=edir)
        # keep an internal record of this
        self.efficiencies=efficiencies
        self.wplist=plist
        self.wlist=wlist
        self.DMlist=DMlist
        mean_efficiencies=np.mean(efficiencies,axis=0)
        self.mean_efficiencies=mean_efficiencies #be careful here!!! This may not be what we want!
        return efficiencies
    
    def __repr__(self):
        """ Over-ride print representation

        Returns:
            str: Items of the FURBY
        """
        repr = '<{:s}: \n'.format(self.__class__.__name__)
        repr += f'name={self.name}'
        return repr
        
# implements something like Mawson's formula for sensitivity
# t_res in ms
def calc_relative_sensitivity(DM_frb,DM,w,fbar,t_res,nu_res,Nchan=336,max_idt=None,max_dm=None,model='Quadrature',dsmear=True,edir=None):
    """ Calculates DM-dependent sensitivity
    
    This function adjusts sensitivity to a given burst as a function of DM.
    
    It includes DM smearing between channels,
    burst intrinsic width,
    the observation frequency
    time- and frequency-resolutions etc.
    
    NOTE: DM_frb *only* used if dsmear = True: combine these to default to None?
    
    Arguments:
        DM_frb [float]: measured DM of a particular FRB. Used only if dsmear=True.
        DMs [np.ndarray] DMs at which to calculate the DM bias effect. pc/cm3
        w [float]: FRB width [ms]
        fbar: mean frequency of the observation [Mhz]
        t_res: time resolution of the observation [ms]
        nu_res: frequency resolution of the observation [Mhz]
        model: Quadrature,Sammons, or CHIME: method to calculate bias. 
                NOTE: Quadrature_s and Sammons_s should be input to this function as
                        just Quadrature and Sammons respectively
        dsmear: subtract DM smearing from measured width to calculate intrinsic
    """
    
    # this model returns the parameterised CHIME DM-dependent sensitivity
    # it is independent of width
    if model=='CHIME':
        # polynomial coefficients for fit to CHIME DM bias data (4th order poly)
        coeffs = np.array([ 7.79309074e-03, -2.09210057e-01,  1.93122752e+00,
            -7.05813760e+00, 8.93355593e+00])
        # this constant normalises the above to a peak efficiency of 100%
        coeffs /= 1.118694423940629
        # fit is to natural log of DM values
        ldm = np.log(DM)
        rate = np.polyval(coeffs,ldm)
        # scale rate by assumed Cartesian logN-logS
        sensitivity = rate**(2./3.)

    # calculates relative sensitivity to bursts as a function of DM
    # Check for Quadrature and Sammons
    elif model == 'Quadrature' or model == 'Sammons':
        # constant of DM
        k_DM=4.149 #ms GHz^2 pc^-1 cm^3
        
        # total smearing factor within a channel
        dm_smearing=2*(nu_res/1.e3)*k_DM*DM/(fbar/1e3)**3 #smearing factor of FRB in the band
        
        # this assumes that what we see are measured widths including all the smearing factors
        # hence we must first adjust for this prior to estimating the DM-dependence
        # for this we use the *true* DM at which the FRB was observed
        if dsmear==True:
            measured_dm_smearing=2*(nu_res/1.e3)*k_DM*DM_frb/(fbar/1e3)**3 #smearing factor of FRB in the band
            uw=w**2-measured_dm_smearing**2-t_res**2 # uses the quadrature model to calculate intrinsic width uw
            if uw < 0:
                uw=0
            else:
                uw=uw**0.5
        else:
            uw=w
        
        if model=='Quadrature':
            sensitivity=(uw**2+dm_smearing**2+t_res**2)**-0.25
        elif model=='Sammons':
            sensitivity=0.75*(0.93*dm_smearing + uw + 0.35*t_res)**-0.5
        # calculates relative sensitivity to bursts as a function of DM
    
    # If model not CHIME, Quadrature or Sammons assume it is a filename
    else:
        if edir is None:
            edir = resource_filename('zdm', 'data/Efficiencies')
        filename = os.path.expanduser(os.path.join(edir, model + ".npy"))
        
        if not os.path.exists(filename):
            raise ValueError("Model is not CHIME, Quadrature or Sammons and hence is expected to be the name of a file containing the efficiencies but " + filename + " does not exist.")

        # Should contain DM in the first row and efficiencies in the second row
        sensitivity_array = np.load(filename)
        sensitivity = np.interp(DM, sensitivity_array[0,:], sensitivity_array[1,:], right=1e-2)

    return sensitivity

def geometric_lognormals(lmu1,ls1,lmu2,ls2,bins=None,
                         Nrand=10000,plot=False,Nbins=101):
    '''
    Numerically evaluates the resulting distribution of y=\sqrt{x1^2+x2^2},
    where logx1~normal and logx2~normal with log-mean lmu and 
    log-sigma ls.
    This is typically used for two log-normals of intrinsic
    FRB width and scattering time
    
    lmu1, ls1 (float, float): log mean and log-sigma of the first distribution
    
    lmu2, ls2 (float, float): log-mean and log-sigma of the second distribution
    
    bins (np.ndarray([NBINS+1],dtype='float')): bin edges for resulting plot.
    
    Returns:
        hist: histogram of probability within bins
        chist: cumulative histogram of probability within bins
        bins: bin edges for histogram
    
    '''
    
    #draw from both distributions
    np.random.seed(1234)
    x1s=np.random.normal(lmu1,ls1,Nrand)
    x2s=np.random.normal(lmu2,ls2,Nrand)
    
    ys=(np.exp(x1s*2)+np.exp(x2s*2))**0.5
    
    if bins is None:
        #bins=np.linspace(0,np.max(ys)/4.,Nbins)
        delta=1e-3
        # ensures the first bin begins at 0
        bins=np.zeros([Nbins+1])
        bins[1:]=np.logspace(np.log10(np.min(ys))-delta,np.log10(np.max(ys))+delta,Nbins)
    hist,bins=np.histogram(ys,bins=bins)
    chist=np.zeros([Nbins+1])
    chist[1:]=np.cumsum(hist)
    chist /= chist[-1]
    
    if plot:
        plt.figure()
        plt.hist(ys,bins=bins)
        plt.xlabel('$y, Y=\\sqrt{X_1^2+X_2^2}$')
        plt.ylabel('$P(Y=y)$')
        plt.tight_layout()
        plt.savefig('adding_lognormals.pdf')
        plt.close()
        
        lbins=np.linspace(-3.,5.,81)
        plt.figure()
        plt.xlabel('$log y, Y=\\sqrt{X_1^2+X_2^2}$')
        plt.ylabel('$P(logY=logy)$')
        plt.hist(np.log(ys),bins=lbins)
        plt.savefig('log_adding_lognormals.pdf')
        plt.close()
        
    
    # renomalises - total will be less than unity, assuming some large
    # values fall off the largest bin
    #hist = hist/Nrand
    return hist,chist,bins
    
def make_widths(s:Survey,state):
    """
    This method takes a distribution of intrinsic FRB widths 
    (lognormal, defined by wlogmean and wlogsigma), and returns 
    a list of w_i, p(w_i), where the w_i are i=1...N values of 
    width, and p(w_i) are statistical weights associated with each. 

    The \sum_i p(w_i) should sum to unity always. Each w_i is used 
    to calculate a separate efficiency table.
    
    Args:
        s (Survey,required): instance of survey class
        state (state class,required): instance of the state class

    Returns:
        list: list of widths
    """
    # variables which can be over-ridden by a survey, but which
    # appear by default in the parameter set
    
    # method to use to calculate FRB width distribution
    if 'WMETHOD' in s.meta and s.meta['WMETHOD'] is not None:
        width_method = s.meta['WMETHOD']
    else:
        width_method = state.width.Wmethod
    
    # just extracting for now to get thrings straight
    nbins=state.width.Wbins
    scale=state.width.Wscale
    thresh=state.width.Wthresh
    #width_method=state.width.Wmethod
    wlogmean=state.width.Wlogmean
    wlogsigma=state.width.Wlogsigma
    
    slogmean=state.scat.Slogmean
    slogsigma=state.scat.Slogsigma
    sfnorm=state.scat.Sfnorm
    sfpower=state.scat.Sfpower
    
    # constant of DM
    k_DM=4.149 #ms GHz^2 pc^-1 cm^3
    
    # Parse
    # OLD
    #    tres=s.meta['TRES']
    #    nu_res=s.meta['FRES']
    #    fbar=s.meta['FBAR']
    tres=s.meta['TRES']
    nu_res=s.meta['FRES']
    fbar=s.meta['FBAR']
    
    ###### calculate a characteristic scaling pulse width ########
    
    # estimates this for a DM of 100
    DM=100
    
    # total smearing factor within a channel
    dm_smearing=2*(nu_res/1.e3)*k_DM*DM/(fbar/1e3)**3 #smearing factor of FRB in the band
    
    # inevitable width due to dm and time resolution
    wequality=(dm_smearing**2 + tres**2)**0.5
    
    # initialise min/max of width bins
    wmax=wequality*thresh
    wmin=wmax*np.exp(-3.*wlogsigma) # three standard deviations below the mean
    # keeps track of numerical normalisation to ensure it ends up at unity
    wsum=0.
    
    ######## generate width distribution ######
    # arrays to hold widths and weights
    weights=[]
    widths=[]
    if width_method==0:
        # do not take a distribution, just use 1ms for everything
        # this is done for tests, for complex surveys such as CHIME,
        # or for estimating the properties of a single FRB
        weights.append(1.)
        widths.append(np.exp(wlogmean))
    elif width_method==1:
        # take intrinsic lognrmal width distribution only
        # normalisation of a log-normal
        norm=(2.*np.pi)**-0.5/wlogsigma
        args=(wlogmean,wlogsigma,norm)
        
        for i in np.arange(nbins):
            weight,err=quad(pcosmic.loglognormal_dlog,np.log(wmin),np.log(wmax),args=args)
            width=(wmin*wmax)**0.5
            widths.append(width)
            weights.append(weight)
            wsum += weight
            wmin = wmax
            wmax *= scale
    elif width_method==2:
        # include scattering distribution
        # scale scattering time according to frequency in logspace
        slogmean = slogmean + sfpower*np.log(fbar/sfnorm)
        
        #gets cumulative hist and bin edges
        dist,cdist,cbins=geometric_lognormals(wlogmean,
                                              wlogsigma,
                                              slogmean,
                                              slogsigma)
        
        # In the below, imin1 and imin2 are the two indices bracketing the minimum
        # bin, while imax1 and imax2 bracket the upper max bin
        imin1=0
        kmin=0.
        imin2=1
        maxbins=cdist.size
        for i in np.arange(nbins):
            if i==nbins-1 or wmax >= cbins[-1]:
                imax2=maxbins-1
            else:
                imax2=np.where(cbins > wmax)[0][0]
                if imax2 >= maxbins:
                    imax2=maxbins-1
            
            imax1=imax2-1
            
            # interpolating max bin. kmax applies to imax2, 1-kmax to imax1
            kmax=(wmax-cbins[imax1])/(cbins[imax2]-cbins[imax1])
            
            #these are cumulative bins
            # the area in the middle is just cmax-cmin
            cmin=kmin*cdist[imin2]+(1-kmin)*cdist[imin1]
            cmax=kmax*cdist[imax2]+(1-kmax)*cdist[imax1]
            if i==0:
                weight=cmax #forces integration from zero
            else:
                weight=cmax-cmin #integrates from bin min to bin max
            # upper bins becomes lower bins
            imin1=imax1
            imin2=imax2
            kmin=kmax
            
            width=(wmin*wmax)**0.5
            widths.append(width)
            weights.append(weight)
            wsum += weight
            
            # updates widths of bin mins and maxes
            wmin = wmax
            wmax *= scale
    
    elif width_method==3:
        # use specific width of FRB. This requires there to be only a single FRB in the survey
        if s.meta['NFRB'] != 1:
            raise ValueError("If width method in make_widths is 3 only one FRB should be specified in the survey but ", str(s.meta['NFRB']), " FRBs were specified")
        else:
            weights.append(1.)
            widths.append(s.frbs['WIDTH'][0])
    else:
        raise ValueError("Width method in make_widths must be 0, 1 or 2, not ",width_method)
    weights[-1] += 1.-wsum #adds defecit here
    weights=np.array(weights)
    widths=np.array(widths)
    # removes unneccesary bins
    keep=np.where(weights>1e-4)[0]
    weights=weights[keep]
    widths=widths[keep]
    
    return widths,weights


def load_survey(survey_name:str, state:parameters.State, 
                dmvals:np.ndarray,
                sdir:str=None, NFRB:int=None, 
                nbins=None, iFRB:int=0, original:bool=False,
                dummy=False,
                edir=None,
                rand_DMG=False,):
    """Load a survey

    Args:
        survey_name (str): Name of the survey
            e.g. CRAFT/FE
        state (parameters.State): Parameters for the state
        dmvals (np.ndarray): DM values
        sdir (str, optional): Path to survey files. Defaults to None.
        nbins (int, optional):  Sets number of bins for Beam analysis
            [was NBeams]
        NFRB (int, optional): Cut the total survey down to a random
            subset [useful for testing]
        iFRB (int, optional): Start grabbing FRBs at this index
            Mainly used for Monte Carlo analysis
            Requires that NFRB be set
        original (bool, optional): 
            Load the original survey file (not recommended)
        dummy (bool,optional)
            Skip many initialisation steps: used only when loading
            survey parameters for conversion to the new survey format

    Raises:
        IOError: [description]

    Returns:
        Survey: instance of the class
    """
    
    print(f"Loading survey: {survey_name}")

    if sdir is None:
        sdir = os.path.join(
            resource_filename('zdm', 'data'), 'Surveys')
        if original:
            sdir = os.path.join(sdir, 'Original')

    # Hard code real surveys
    if survey_name == 'CRAFT/FE':
        dfile = 'CRAFT_class_I_and_II'
    elif survey_name == 'CRAFT/ICS':
        dfile = 'CRAFT_ICS'
    elif survey_name == 'CRAFT/ICS892':
        dfile = 'CRAFT_ICS_892'
    elif survey_name == 'CRAFT/ICS1632':
        dfile = 'CRAFT_ICS_1632'

    elif survey_name == 'PKS/Mb':
        dfile = 'parkes_mb_class_I_and_II'
    elif 'private' in survey_name: 
        dfile = survey_name
    else:
        dfile = survey_name
    
    if original:
        dfile += '.dat'
    else:
        dfile += '.ecsv'

    print(f"Loading survey: {survey_name} from {dfile}")

    # Do it
    if original:
        srvy=OldSurvey()
        srvy.name = survey_name
        srvy.process_survey_file(os.path.join(sdir, dfile), 
                                NFRB=NFRB, iFRB=iFRB)
        
        if not dummy:
            srvy.init_DMEG(state.MW.DMhalo)
            
            beam_method = srvy.meta['BMETHOD']
            beam_thresh = srvy.meta['BTHRESH']
            width_bias = srvy.meta['WBIAS']
            
            srvy.init_beam(method=beam_method, plot=False,
                    thresh=beam_thresh) # tells the survey to use the beam file
            pwidths,pprobs=make_widths(srvy,state)
            _ = srvy.get_efficiency_from_wlist(dmvals,pwidths,pprobs,
                                            model=width_bias) 
    else:                                
        srvy = Survey(state, 
                         survey_name, 
                         os.path.join(sdir, dfile), 
                         dmvals,
                         NFRB=NFRB, iFRB=iFRB,
                         edir=edir, rand_DMG=rand_DMG)
    return srvy

def refactor_old_survey_file(survey_name:str, outfile:str, 
                             clobber:bool=False,sdir=None):
    """Refactor an old survey file to the new format

    Args:
        survey_name (str): Name of the survey
        outfile (str): Name of the output file
        clbover (bool, optional): Clobber the output file. Defaults to False.
        sdir: director to find survey in. Defaults to None, i.e.
            the zdm/data/Survey/Original will be used
    """
    
    state = parameters.State()
    srvy_data = survey_data.SurveyData()
    
    # Load up original

    # temporary workaround for private survey files... sorry X!

    nbins = None
    if 'private' in survey_name:
        nbins = 5 # only works for the CRAFT private samples

    isurvey = load_survey(survey_name, state, 
                         np.linspace(0., 2000., 1000),
                         original=True,sdir=sdir,
                         dummy=True, nbins=nbins)
    
    # FRBs
    frbs = pandas.DataFrame(isurvey.frbs)

    # Fill in fixed survey_data from meta
    # Telescope
    for field in srvy_data.telescope.fields:
        if not (field in isurvey.meta) or isurvey.meta[field] is None:
            isurvey.meta[field]='-1'
            print("Could not find field ",field," setting to -1")
        setattr(srvy_data.telescope,field, srvy_data.telescope.__dataclass_fields__[field].type(
                isurvey.meta[field]))
        
    # Observing
    for field in srvy_data.observing.fields:
        if not (field in isurvey.meta) or isurvey.meta[field] is None:
            isurvey.meta[field]='-1'
            print("Could not find field ",field," setting to -1")
        if field !='NORM_FRB' or 'NORM_FRB' in isurvey.meta:
            setattr(srvy_data.observing,field, srvy_data.observing.__dataclass_fields__[field].type(
                isurvey.meta[field]))
        else:
            srvy_data.observing.NORM_FRB = len(frbs)
            
    # Trim down FRB table
    for key in srvy_data.to_dict().keys():
        for key2 in srvy_data.to_dict()[key]:
            if key2 in frbs.keys():
                frbs.drop(columns=[key2], inplace=True)

    # Rename ID to TNS
    frbs.rename(columns={'ID':'TNS'}, inplace=True)

    # Vet+populate the FRBs
    vet_frb_table(frbs, mandatory=False, fill=True)

    # Add X columns (ancillay)
    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 
                   'G', 'H', 'I', 'J', 'K']:
        # Add it 
        key = f'X{letter}'
        isurvey.do_keyword_char(key,3,None, dtype='str')
        if isurvey.frbs[key] is None:
            continue
        # Add it
        frbs[key] = isurvey.frbs[key]

    # Order the columns
    frbs = frbs.reindex(sorted(frbs.columns), axis=1)

    # Move TNS to the front
    col = frbs.pop("TNS")
    frbs.insert(0, col.name, col)

    # Convert for I/O
    frbs = Table.from_pandas(frbs)

    # Meta
    frbs.meta['survey_data'] = json.dumps(
        srvy_data.to_dict(), sort_keys=True, indent=4, 
        separators=(',', ': '))

    # Write me
    frbs.write(outfile, overwrite=clobber, format='ascii.ecsv')
    
    print(f"Wrote: {outfile}")

def vet_frb_table(frb_tbl:pandas.DataFrame,
                  mandatory:bool=False,
                  fill:bool=False):
    frb_data = survey_data.FRB()
    # Loop on the stadnard fields
    for field in frb_data.__dataclass_fields__.keys():
        if field in frb_tbl.keys():
            not_none = frb_tbl[field].values != None
            if np.any(not_none):
                idx0 = np.where(not_none)[0][0]
                assert isinstance(
                    frb_tbl.iloc[idx0][field], 
                    frb_data.__dataclass_fields__[field].type), \
                        f'Bad data type for {field}'
        elif mandatory:
            raise ValueError(f'{field} is missing in your table!')
        elif fill:
            frb_tbl[field] = None
