import os
from matplotlib import pyplot as plt
from numpy import random
import numpy as np
from matplotlib import pyplot as plt
import importlib.resources as resources

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from zdm import optical as opt
from zdm import states
from zdm import survey


def create_fake_survey(smearing=False,Survey="CRACO"):
    """
    Creates four fake survey files
    """
    sdir = str(resources.files('zdm').joinpath('data/Surveys'))
    
    if Survey == "CRACO":
        Prefix,Suffix = load_craco_text()
        name=['CRAFT_CRACO_900']
        opdir="./CRACO/" # directory to place fake surveys in. Here!
    elif Survey == "MeerTRAP":
        Prefix,Suffix = load_meertrap_text()
        name=['MeerTRAPcoherent']
        opdir="./MeerTRAP/" # directory to place fake surveys in. Here!
    else:
        raise ValueError("Survey ",Survey," not recognised")
    
    # make output directories
    if not os.path.isdir(opdir):
        os.mkdir(opdir)
    
    #param_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37,
    #            'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -7.61}
    
    # use default state
    state=states.load_state(case="HoffmannHalo25",scat=None,rep=None)

    
    
    ss,gs=loading.surveys_and_grids(survey_names=name,repeaters=False,init_state=state,sdir=sdir)
    gs=gs[0]
    #gs.state.photo.smearing=smearing
    gs.calc_rates()
    samples=gs.GenMCSample(100)
    zvals=np.zeros(len(samples))
    fp=open(opdir+"Spectroscopic.ecsv","w+")
    fp.write(Prefix+Suffix)
    fp.write("TNS        DM             RA      DEC         Z                 SNR                   WIDTH        Gl      Gb     DMG           FBAR     BW\n")
    for i in range(len(samples)):
        fp.write('{0:5}'.format(str(i)))
        fp.write('{0:20}'.format(str(samples[i][1]+35)))
        fp.write('{0:10}'.format("00:00:00"))
        fp.write('{0:10}'.format("00:00:00"))
        fp.write('{0:25}'.format(str(samples[i][0])))
        fp.write('{0:25}'.format(str(samples[i][3]*10)))
        fp.write('{0:8}'.format("-1.0"))
        fp.write('{0:8}'.format("-1.0"))
        fp.write('{0:8}'.format("-1.0"))
        fp.write('{0:8}'.format("35.0"))
        fp.write('{0:8}'.format("888"))
        fp.write('{0:8}'.format("288"))
        fp.write("\n")
    fp.close()
    
    # We now smear the redshift values by the z-error
    
    if smearing is True:
        sigmas=np.array([0.035])
        for sigma in sigmas:
            for i in range(len(samples)):
                zvals[i]=samples[i][0]
            
            fp=open(opdir+"Smeared.ecsv","w+")
            fp.write(Prefix+', "Z_PHOTO": 0.03'+Suffix)
            fp.write("TNS        DM             RA      DEC         Z                 SNR                   WIDTH        Gl      Gb     DMG           FBAR     BW\n")
            smear_error=random.normal(loc=0,scale=sigma,size=100)
            newvals=zvals+smear_error
            for i in range(len(samples)):
                fp.write('{0:5}'.format(str(i)))
                fp.write('{0:20}'.format(str(samples[i][1]+35)))
                fp.write('{0:10}'.format("00:00:00"))
                fp.write('{0:10}'.format("00:00:00"))
                fp.write('{0:25}'.format(str(newvals[i])))
                fp.write('{0:25}'.format(str(samples[i][3]*10)))
                fp.write('{0:8}'.format("-1.0"))
                fp.write('{0:8}'.format("-1.0"))
                fp.write('{0:8}'.format("-1.0"))
                fp.write('{0:8}'.format("35.0"))
                fp.write('{0:8}'.format("888"))
                fp.write('{0:8}'.format("288"))
                fp.write("\n")
            fp.close()
    
    frac_path = str(resources.files('zdm').joinpath('../papers/lsst/Data'))
    
    if Survey == "CRACO":
        fz=np.load(frac_path+"/fz_24.7.npy")[0:500]
    elif Survey == "MeerTRAP":
        fz=np.load(frac_path+"/fz_27.5.npy")[0:500]                                                                 
    
    zs=np.load(frac_path+"/zvals.npy")[0:500]
    
    fp=open(opdir+"zFrac.ecsv","w+")
    fp1=open(opdir+"Smeared_and_zFrac.ecsv","w+")
    
    fp.write(Prefix+', "Z_FRACTION": 24.7'+Suffix)
    fp1.write(Prefix+', "Z_FRACTION": 24.7, "Z_PHOTO": 0.03'+Suffix)
    
    fp.write("TNS        DM             RA      DEC         Z                 SNR                   WIDTH        Gl      Gb     DMG           FBAR     BW\n")
    fp1.write("TNS        DM             RA      DEC         Z                 SNR                   WIDTH        Gl      Gb     DMG           FBAR     BW\n")
    for i in range(len(samples)):
        prob_thresh=random.rand()
        j=np.where(zs>samples[i][0]-0.005)[0][0]
        prob=fz[j]
        if prob>=prob_thresh:
            fp.write('{0:5}'.format(str(i)))
            fp.write('{0:20}'.format(str(samples[i][1]+35)))
            fp.write('{0:10}'.format("00:00:00"))
            fp.write('{0:10}'.format("00:00:00"))
            fp.write('{0:25}'.format(str(samples[i][0])))
            fp.write('{0:25}'.format(str(samples[i][3]*10)))
            fp.write('{0:8}'.format("-1.0"))
            fp.write('{0:8}'.format("-1.0"))
            fp.write('{0:8}'.format("-1.0"))
            fp.write('{0:8}'.format("35.0"))
            fp.write('{0:8}'.format("888"))
            fp.write('{0:8}'.format("288"))
            fp.write("\n")

            fp1.write('{0:5}'.format(str(i)))
            fp1.write('{0:20}'.format(str(samples[i][1]+35)))
            fp1.write('{0:10}'.format("00:00:00"))
            fp1.write('{0:10}'.format("00:00:00"))
            fp1.write('{0:25}'.format(str(samples[i][0]+smear_error[i])))
            fp1.write('{0:25}'.format(str(samples[i][3]*10)))
            fp1.write('{0:8}'.format("-1.0"))
            fp1.write('{0:8}'.format("-1.0"))
            fp1.write('{0:8}'.format("-1.0"))
            fp1.write('{0:8}'.format("35.0"))
            fp1.write('{0:8}'.format("888"))
            fp1.write('{0:8}'.format("288"))
            fp1.write("\n")

    fp.close()
    fp1.close()


def load_meertrap_text():
    """
    returns CRACO prefixes and suffixes
    """
    
    
    Prefix="""# %ECSV 1.0
# ---
# datatype:
# - {name: TNS, datatype: string}
# - {name: DM, datatype: float64}
# - {name: RA, datatype: string}
# - {name: DEC, datatype: string}
# - {name: Z, datatype: float64}
# - {name: SNR, datatype: float64}
# - {name: WIDTH, datatype: float64}
# - {name: Gl, unit: deg, datatype: float64}
# - {name: Gb, unit: deg, datatype: float64}
# - {name: DMG, datatype: float64}
# - {name: FBAR, datatype: float64}
# - {name: BW, datatype: float64}
# meta: !!omap
# - {survey_data: '{"observing": {"NORM_FRB": 1,"TOBS": 317.5"""
# we need to split the obs string into two so we can insert the zfraction as required
    Suffix="""},
#                   "telescope": {"BEAM": "MeerTRAP_coherent_log",
#                        "BTHRESH": 0.25,"NBEAMS": 1,"NBINS": 5,
#                        "FRES":0.836,"THRESH":0.069,
#                        "TRES": 0.30624, "FBAR":1284}}'}\n"""

    return Prefix, Suffix

def load_craco_text():
    """
    returns CRACO prefixes and suffixes
    """
    
    
    Prefix="""# %ECSV 1.0
# ---
# datatype:
# - {name: TNS, datatype: string}
# - {name: DM, datatype: float64}
# - {name: RA, datatype: string}
# - {name: DEC, datatype: string}
# - {name: Z, datatype: float64}
# - {name: SNR, datatype: float64}
# - {name: WIDTH, datatype: float64}
# - {name: Gl, unit: deg, datatype: float64}
# - {name: Gb, unit: deg, datatype: float64}
# - {name: DMG, datatype: float64}
# - {name: FBAR, datatype: float64}
# - {name: BW, datatype: float64}
# meta: !!omap
# - {survey_data: '{"observing": {"NORM_FRB": 17,"TOBS": 64.68,"MAX_IW": 8, "MAXWMETH": 2"""
# we need to split the obs string into two so we can insert the zfraction as required
    Suffix="""},
#                   "telescope": {"BEAM": "CRACO_900", "DMMASK": "craco_900_mask.npy",
#                                 "DIAM": 12.0, "NBEAMS": 1, "NBINS": 5, "FBAR": 906,
#                                 "TRES": 13.8, "FRES": 1.0, "THRESH": 1.01}}'}\n"""

    return Prefix, Suffix
    
create_fake_survey(True,Survey="CRACO")
create_fake_survey(True,Survey="MeerTRAP")
