################ COSMOLOGY.PY ###############

# Author: Clancy W. James
# clancy.w.james@gmail.com


# This file defines a class to hold an FRB survey
# Essentially, this is relevant when multiple
# FRBs are discovered by the same instrument

##############################################


from IPython.terminal.embed import embed
import numpy as np
import os
from pkg_resources import resource_filename
from scipy.integrate import quad

from typing import IO

from zdm import beams, parameters
from zdm import pcosmic
class Survey:
    """A class to hold an FRB survey"""
    
    
    def __init__(self):
        self.init=False
        self.do_beam=False
        #self.DM=[] #observed
        #self.w=[] #observed widths
        #self.fbar= #mean frequency
        #self.sens_meth="None"
    
    # NEXT STEP: go ahead and try to generate a weighted DM-z distribution, see how you go! (fool...)
    #def get_jyms():
    #	for i in np.arange(self.NFRB):
    #		jyms=width**0.5 * SNR/efficiency
    
    def get_efficiency(self,DMlist,model="Quadrature",dsmear=True):
        """ Gets efficiency to FRBs
        Returns a list of relative efficiencies
        as a function of dispersion measure for each FRB.
        """
        efficiencies=np.zeros([self.NFRB,DMlist.size])
        for i in np.arange(self.NFRB):
            efficiencies[i,:]=calc_relative_sensitivity(self.DMs[i],DMlist,self.WIDTHs[i],self.FBARs[i],self.TRESs[i],self.FRESs[i],model=model,dsmear=dsmear)
        # keep an internal record of this
        self.efficiencies=efficiencies
        self.DMlist=DMlist
        self.wplist=np.array([1])# weight of 1
        mean_efficiencies=np.mean(efficiencies,axis=0)
        self.mean_efficiencies=mean_efficiencies
        return efficiencies
    
    def get_efficiency_from_wlist(self,DMlist,wlist,plist,model="Quadrature"):
        """ Gets efficiency to FRBs
        Returns a list of relative efficiencies
        as a function of dispersion measure for each width given in wlist
        """
        efficiencies=np.zeros([wlist.size,DMlist.size])
        for i,w in enumerate(wlist):
            efficiencies[i,:]=calc_relative_sensitivity(None,DMlist,w,self.meta["FBAR"],self.meta["TRES"],self.meta["FRES"],model=model,dsmear=False)
        # keep an internal record of this
        self.efficiencies=efficiencies
        self.wplist=plist
        self.wlist=wlist
        self.DMlist=DMlist
        mean_efficiencies=np.mean(efficiencies,axis=0)
        self.mean_efficiencies=mean_efficiencies #be careful here!!! This may not be what we want!
        return efficiencies
    
    def process_survey_file(self,filename, NFRB=None):
        """ Loads a survey file, then creates dictionaries of the loaded variables """
        info=[]
        keys=[]
        self.meta={} # dict to contain survey metadata, in dictionary format
        self.frbs={} # dict to contain arrays of data for each FRB proprty
        basename=os.path.basename(filename)
        name=os.path.splitext(basename)[0]
        self.name=name
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

        # 
        if NFRB is None:
            self.NFRB=keys.count('FRB')
        else:
            self.NFRB = NFRB
        if self.NFRB==0:
            raise ValueError('No FRBs found in file '+filename) #change this?


        self.meta['NFRB']=self.NFRB
        
        #### separates FRB and non-FRB keys
        self.frblist=self.find(keys,'FRB')

        if NFRB is not None:
            # Take the first set
            self.frblist=self.frblist[0:NFRB]
        
        ### first check for the key list to interpret the FRB table
        iKEY=self.do_metakey('KEY')
        self.keylist=info[iKEY]
        
        # the following can only be metadata
        which=1
        self.do_keyword_char('BEAM',which,None) # prefix of beam file
        self.do_keyword('TOBS',which,None) # total observation time, hr
        self.do_keyword('DIAM',which,None) # Telescope diamater (in case of Gauss beam)
        self.do_keyword('NBEAMS',which,1) # Number of beams (multiplies sr)
        self.do_keyword('NORM_FRB',which,self.NFRB) # number of FRBs to norm obs time by
        
        self.NORM_FRB=self.meta['NORM_FRB']
        # the following properties can either be FRB-by-FRB, or metadata
        which=3
        
        self.do_keyword('THRESH',which)
        self.do_keyword('TRES',which,1.265)
        self.do_keyword('FRES',which,1)
        self.do_keyword('FBAR',which,1196)
        self.do_keyword('BW',which,336)
        self.do_keyword('SNRTHRESH',which,9.5)
        self.do_keyword('DMG',which,None) # Galactic contribution to DM
        
        
        # The following properties can only be FRB-by-FRB
        which=2
        self.do_keyword('SNR',which)
        self.do_keyword('DM',which)
        self.do_keyword('WIDTH',which,0.1) # defaults to unresolved width in time
        self.do_keyword_char('ID',which,None) # obviously we don't need names!
        self.do_keyword('Gl',which,None) # Galactic latitude
        self.do_keyword('Gb',which,None) # Galactic longitude
        
        self.do_keyword('Z',which,None)
        if self.frbs["Z"] is not None:
            self.nD=2
            self.Zs=self.frbs["Z"]
        else:
            self.nD=1
            self.Zs=None
        
        ### processes galactic contributions
        self.process_dmg()
        
        
        ### get pointers to correct results ,for better access
        self.DMs=self.frbs['DM']
        self.DMGs=self.frbs['DMG']
        self.SNRs=self.frbs['SNR']
        self.WIDTHs=self.frbs['WIDTH']
        self.THRESHs=self.frbs['THRESH']
        self.TRESs=self.frbs['TRES']
        self.FRESs=self.frbs['FRES']
        self.FBARs=self.frbs['FBAR']
        self.BWs=self.frbs['BW']
        self.SNRTHRESHs=self.frbs['SNRTHRESH']
        
        self.Ss=self.SNRs/self.SNRTHRESHs
        self.TOBS=self.meta['TOBS']
        self.Ss[np.where(self.Ss < 1.)[0]]=1
        
        # sets the 'beam' values to unity by default
        self.beam_b=np.array([1])
        self.beam_o=np.array([1])
        self.NBEAMS=1
        
        self.init=True
        print("FRB survey succeffully initialised with ",self.NFRB," FRBs")
        
    def init_DMEG(self,DMhalo):
        """ Calculates extragalactic DMs assuming halo DM """
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
        IF which == 	1: must be metadata
                2: must be FRB-by-FRB
                3: could be either
        
        If default == 	None: it is OK to not be present
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
            
        elif (which != 1) and (self.keylist.count(key)==1): #info varies according to each FRB
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
    
    def do_keyword_char(self,key,which,default=-1):
        """ This kind of key can either be in the metadata, or the table, not both
        IF which == 	1: must be metadata
                2: must be FRB-by-FRB
                3: could be either
        
        If default == 	None: it is OK to not be present
                -1: fail if not there
                Another value: set this as the default
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
            values=np.zeros([self.NFRB])
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
            import ne2001
            from ne2001 import density
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
        print(self.keys)
        print(self.keylist)
        exit()
        #if key in self.keys:
        #	return self.keys
    
    def init_beam(self,nbins=10,plot=False,method=1,thresh=1e-3,Gauss=False):
        """ Initialises the beam """
        if Gauss:
            b,omegab=beams.gauss_beam(thresh=thresh,nbins=nbins,freq=self.meta["FBAR"],D=self.meta["DIAM"])
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
            b2,o2=beams.simplify_beam(logb,omegab,nbins,savename=savename,method=method,thresh=thresh)
            self.beam_b=b2
            self.beam_o=o2
            self.do_beam=True
            # sets the 'beam' values to unity by default
            self.NBEAMS=b2.size
            
        else:
            print("No beam found to initialise...")
            
    

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
    """
    
    # constant of DM
    k_DM=4.149 #ms GHz^2 pc^-1 cm^3
    
    # total smearing factor within a channel
    dm_smearing=2*(nu_res/1.e3)*k_DM*DM/(fbar/1e3)**3 #smearing factor of FRB in the band
    
    # this assumes that what we see are measured widths including all the smearing factors
    # hence we must first adjust for this prior to estimating the DM-dependence
    # for this we use the *true* DM at which the FRB was observed
    if dsmear==True:
        measured_dm_smearing=2*(nu_res/1.e3)*k_DM*DM_frb/(fbar/1e3)**3 #smearing factor of FRB in the band
        uw=w**2-measured_dm_smearing**2-t_res**2
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
    
    # calculates relative sensitivity to bursts as a function of DM
    return sensitivity
    
    """ Tries to get intelligent choices for width binning assuming some intrinsic distribution
    Probably should make this a 'self' function.... oh well, for the future!
    """

def make_widths(s:Survey,wlogmean,wlogsigma,nbins,scale=2,thresh=0.5):
    """

    Args:
        s (Survey): [description]
        wlogmean ([type]): [description]
        wlogsigma ([type]): [description]
        nbins ([type]): [description]
        scale (int, optional): [description]. Defaults to 2.
        thresh (float, optional): [description]. Defaults to 0.5.

    Returns:
        [type]: [description]
    """



    
    # constant of DM
    k_DM=4.149 #ms GHz^2 pc^-1 cm^3
    
    tres=s.meta['TRES']
    nu_res=s.meta['FRES']
    fbar=s.meta['FBAR']
    
    # estimates this for a DM of 100
    DM=100
    
    # total smearing factor within a channel
    dm_smearing=2*(nu_res/1.e3)*k_DM*DM/(fbar/1e3)**3 #smearing factor of FRB in the band
    
    wequality=(dm_smearing**2 + tres**2)**0.5
    
    weights=[]
    widths=[]
    norm=(2.*np.pi)**-0.5/wlogsigma
    args=(wlogmean,wlogsigma,norm)
    wmax=wequality*thresh
    wmin=wmax*np.exp(-3.*wlogsigma)
    wsum=0.
    #print("Initialised wmin, wmax at ",wmin,wmax," from ",logmean,logsigma,wequality,thresh)
    for i in np.arange(nbins):
        weight,err=quad(pcosmic.loglognormal_dlog,np.log(wmin),np.log(wmax),args=args)
        width=(wmin*wmax)**0.5
        widths.append(width)
        weights.append(weight)
        wsum += weight
        wmin = wmax
        wmax *= scale
        #print("After iteration ",i," wmin, max are ",wmin,wmax)
    weights[-1] += 1.-wsum #adds defecit here
    weights=np.array(weights)
    widths=np.array(widths)
    return widths,weights


def load_survey(survey_name:str, state:parameters.State, dmvals:np.ndarray,
                sdir:str=None, NFRB:int=None):
    """Load a survey

    Args:
        survey_name (str): Name of the survey
            e.g. CRAFT/FE
        state (parameters.State): Parameters for the state
        dmvals (np.ndarray): DM values
        sdir (str, optional): Path to survey files. Defaults to None.
        NFRB (int, optional): Cut the total survey down to a random
            subset [useful for testing]

    Raises:
        IOError: [description]

    Returns:
        Survey: instance of the class
    """
    print(f"Loading survey: {survey_name}")
    if sdir is None:
        sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    if survey_name == 'CRAFT/FE':
        dfile = 'CRAFT_class_I_and_II.dat'
        Nbeams = 5
    elif survey_name == 'CRAFT/ICS':
        dfile = 'CRAFT_ICS.dat'
        Nbeams = 5
    elif survey_name == 'CRAFT/CRACO':
        dfile = 'CRAFT_CRACO_MC_frbs.dat'
        Nbeams = 5
    elif survey_name == 'CRAFT/CRACO_1':  # alpha_method = 1
        dfile = 'CRAFT_CRACO_MC_frbs_alpha1.dat'
        Nbeams = 5
    elif survey_name == 'CRAFT/CRACO_1_5000':  # alpha_method = 1
        dfile = 'CRAFT_CRACO_MC_frbs_alpha1_5000.dat'
        Nbeams = 5
    elif survey_name == 'CRAFT/ICS892':
        dfile = 'CRAFT_ICS_892.dat'
        Nbeams = 5
    elif survey_name == 'PKS/Mb':
        dfile = 'parkes_mb_class_I_and_II.dat'
        Nbeams = 10
    else:
        raise IOError("Bad survey name!!")
    # Do it
    srvy=Survey()
    srvy.name = survey_name
    srvy.process_survey_file(os.path.join(sdir, dfile), NFRB=NFRB)
    srvy.init_DMEG(state.MW.DMhalo)
    srvy.init_beam(nbins=Nbeams, method=2, plot=False,
                thresh=state.beam.thresh) # tells the survey to use the beam file
    pwidths,pprobs=make_widths(srvy, 
                                    state.width.logmean,
                                    state.width.logsigma,
                                    state.beam.Wbins,
                                    scale=state.beam.Wscale)
    _ = srvy.get_efficiency_from_wlist(dmvals,pwidths,pprobs)

    return srvy