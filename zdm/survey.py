# ###############################################
# This file defines a class to hold an FRB survey
# Essentially, this is relevant when multiple
# FRBs are discovered by the same instrument
# ##############################################

import numpy as np
import os
from pkg_resources import resource_filename
from scipy.integrate import quad

import pandas
from astropy.table import Table
import json


from ne2001 import density

from zdm import beams, parameters
from zdm import pcosmic
from zdm import survey_data

import matplotlib.pyplot as plt

from IPython import embed

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
    
    def get_efficiency(self,DMlist,model="Quadrature",dsmear=True):
        """ Gets efficiency to FRBs
        Returns a list of relative efficiencies
        as a function of dispersion measure for each FRB.
        """
        efficiencies=np.zeros([self.NFRB,DMlist.size])
        for i in np.arange(self.NFRB):
            efficiencies[i,:]=calc_relative_sensitivity(self.DMs[i],DMlist,self.WIDTHs[i],
                self.FBARs[i],self.TRESs[i],self.FRESs[i],model=model,dsmear=dsmear)
        # keep an internal record of this
        self.efficiencies=efficiencies
        self.DMlist=DMlist
        self.wplist=np.array([1])# weight of 1
        mean_efficiencies=np.mean(efficiencies,axis=0)
        self.mean_efficiencies=mean_efficiencies
        return efficiencies
    
    def get_efficiency_from_wlist(self,DMlist,wlist,plist, 
                                  model="Quadrature", 
                                  addGalacticDM=True):
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
                dsmear=False)
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
        self.do_keyword_char('ID',which,None, dtype='str') # obviously we don't need names,!
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
        #print(self.keys)
        #print(self.keylist)
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
                 iFRB:int=0):
        """ Init an FRB Survey class

        Args:
            state (_type_): _description_
            survey_name (str): 
                Name of the survey
            filename (str): _description_
            dmvals (np.ndarray): _description_
            NFRB (int, optional): _description_. Defaults to None.
            iFRB (int, optional): _description_. Defaults to 0.
        """
        # Proceed
        self.name = survey_name
        # Load up
        self.process_survey_file(filename, NFRB, iFRB)
        # DM EG
        self.init_DMEG(state.MW.DMhalo)
        # Beam
        self.init_beam(
                       method=state.beam.Bmethod, 
                       plot=False, 
                       thresh=state.beam.Bthresh) # tells the survey to use the beam file
        # Efficiency
        pwidths,pprobs=make_widths(self, state)
        _ = self.get_efficiency_from_wlist(dmvals,
                                       pwidths,pprobs) 

    def init_DMEG(self,DMhalo):
        """ Calculates extragalactic DMs assuming halo DM """
        self.DMhalo=DMhalo
        self.DMEGs=self.DMs-self.DMGs-DMhalo

    def process_survey_file(self,filename:str, 
                            NFRB:int=None,
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
        # FRB data
        self.frbs = frb_tbl.to_pandas()

        # Cut down?
        if self.NFRB is None:
            self.NFRB=len(self.frbs)
        else:
            self.NFRB=min(len(self.frbs), NFRB)
            if self.NFRB < NFRB+iFRB:
                raise ValueError("Cannot return sufficient FRBs, did you mean NFRB=None?")
            # Not sure the following linematters given the Error above
            themax = min(NFRB+iFRB,self.NFRB)
            self.frbs=self.frbs[iFRB:themax]
        # Vet
        vet_frb_table(self.frbs, mandatory=True)

        # Pandas resolves None to Nan
        if np.isfinite(self.frbs["Z"][0]):
            
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
        self.Ss[np.where(self.Ss < 1.)[0]]=1
        
        # sets the 'beam' values to unity by default
        self.beam_b=np.array([1])
        self.beam_o=np.array([1])
        self.NBEAMS=1
        
        print("FRB survey sucessfully initialised with ",self.NFRB," FRBs starting from", self.iFRB)

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
                  method=1,thresh=1e-3,Gauss=False):
        """ Initialises the beam """
        if Gauss:
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

    def get_efficiency_from_wlist(self,DMlist,wlist,plist, 
                                  model="Quadrature", 
                                  addGalacticDM=True):
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
            toAdd = self.DMhalo + np.mean(self.DMGs)
        else:
            toAdd = 0.
        
        for i,w in enumerate(wlist):
            efficiencies[i,:]=calc_relative_sensitivity(
                None,DMlist+toAdd,w,
                np.median(self.frbs['FBAR']),
                np.median(self.frbs['TRES']),
                np.median(self.frbs['FRES']),
                model=model,
                dsmear=False)
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
def calc_relative_sensitivity(DM_frb,DM,w,fbar,t_res,nu_res,model='Quadrature',dsmear=True):
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
        model: Quadrature,Sammons, or CHIME: method to calculate bias
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
        return sensitivity
        
    # constant of DM; this is ~0.1% accurate, which is good enough here.
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
        sensitivity=(uw**2+dm_smearing**2+t_res**2)**-0.5
    elif model=='Sammons':
        sensitivity=0.75*(0.93*dm_smearing + uw + 0.35*t_res)**-0.5
    else:
        raise ValueError(model," is an unknown DM smearing model --- use CHIME, Quadrature, or Sammons")
    # calculates relative sensitivity to bursts as a function of DM
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
    # just extracting for now toget thrings straight
    nbins=state.width.Wbins
    scale=state.width.Wscale
    thresh=state.width.Wthresh
    width_method=state.width.Wmethod
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
    tres=np.median(s.frbs['TRES'])
    nu_res=np.median(s.frbs['FRES'])
    fbar=np.median(s.frbs['FBAR'])
    
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
        widths.append(np.exp(slogmean))
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
                nbins=None, iFRB:int=0, original:bool=False):
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
        if nbins is None:
            raise IOError("You must specify nbins with a private survey file")
    else:
        dfile = survey_name
    
    if original:
        dfile += '.dat'
    else:
        dfile += '.ecsv'

    
    #    if defNbeams is None:
    #        raise IOError("You must specify Nbeams with a private survey file")
    
    
    #### NOTE: the following is deleted as  Nbeams now part of each survey
    #    defNbeams = 5
    ## survey data over-writes the defaults
    ## Note this only applies for some beam methods
    #if Nbeams is None:
    #    Nbeams = defNbeams
    #srvy.init_beam(nbins=Nbeams, method=state.beam.Bmethod, plot=False,
    #            thresh=state.beam.Bthresh) # tells the survey to use the beam file

    # Do it
    if original:
        srvy=OldSurvey()
        srvy.name = survey_name
        srvy.process_survey_file(os.path.join(sdir, dfile), 
                                NFRB=NFRB, iFRB=iFRB)
        #srvy.process_survey_file(os.path.join(sdir, dfile), NFRB=NFRB, iFRB=iFRB)
        srvy.init_DMEG(state.MW.DMhalo)
        srvy.init_beam(method=state.beam.Bmethod, plot=False,
                    thresh=state.beam.Bthresh) # tells the survey to use the beam file
        pwidths,pprobs=make_widths(srvy,state)
        _ = srvy.get_efficiency_from_wlist(dmvals,pwidths,pprobs,
                                            model=state.width.Wbias) 
    else:                                
        srvy = Survey(state, 
                         survey_name, 
                         os.path.join(sdir, dfile), 
                         dmvals,
                         NFRB=NFRB, iFRB=iFRB)
    return srvy

def refactor_old_survey_file(survey_name:str, outfile:str, 
                             clobber:bool=False):
    """Refactor an old survey file to the new format

    Args:
        survey_name (str): Name of the survey
        outfile (str): Name of the output file
        clbover (bool, optional): Clobber the output file. Defaults to False.
    """
    
    state = parameters.State()
    srvy_data = survey_data.SurveyData()
    
    # Load up original
    isurvey = load_survey(survey_name, state, 
                         np.linspace(0., 2000., 1000),
                         original=True)

    # FRBs
    frbs = pandas.DataFrame(isurvey.frbs)

    # Fill in fixed survey_data from meta
    # Telescope
    for field in srvy_data.telescope.fields:
        print(f"Ingesting {field}")
        setattr(srvy_data.telescope,field, srvy_data.telescope.__dataclass_fields__[field].type(
                isurvey.meta[field]))

    # Observing
    for field in srvy_data.observing.fields:
        print(f"Ingesting {field}")
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
    frbs.write(outfile, overwrite=clobber,format='ascii.ecsv')
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
