
from zdm import misc_functions
from zdm import loading
from zdm import states
import numpy as np
import pandas as pd
import os

def main(NMAX,label):
    """
    Creates a fake CRACO_900 survey
    """
    frbs = pd.read_csv("craco_900_mc_sample.csv")
    hosts = pd.read_csv("craco_assigned_galaxies.csv")
    NFRB=len(frbs)
    Nhosts = len(hosts)
    
    
    opdir = "Surveys/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # load correct halo value
    state = states.load_state("HoffmannHalo25")
    
    # load survey file to retrieve some info
    name = "CRAFT_CRACO_900"
    ss, gs = loading.surveys_and_grids(survey_names = [name],repeaters=False,init_state=state)
    g = gs[0]
    s = ss[0]
    
    DMhalo = np.median(s.DMhalos)
    DMG = np.median(s.DMGs)
    
    print("Adding MW contributions of ",DMG," pc/cm3 for the ISM, and ",DMhalo," for the halo")
    
    # extra info to add to FRB file
    Gbs = np.zeros([Nhosts])
    Gls = np.zeros([Nhosts])
    DMGs = np.zeros([Nhosts])
    DMtots = np.zeros([Nhosts])
    
    for i in np.arange(Nhosts):
        host=hosts.loc[i]
        j = host["FRB_ID"]
        frb = frbs.loc[j]
        Gb,Gl = misc_functions.j2000_to_galactic(host['ra'], host['dec'])
        Gbs[i] = Gb
        Gls[i] = Gl
        DMGs[i] = DMG
        DMtots[i] = DMG + DMhalo + frb["DMeg"]
        
    hosts["DM"] = DMtots
    hosts["Gl"] = Gls
    hosts["Gb"] = Gbs
    hosts["DMG"] = DMGs
    
    # we now add the Galactic DMG values
    Prefix, Suffix = load_craco_text()
    
    fp=open(opdir+label+".ecsv","w+")
    fp.write(Prefix+Suffix)
    fp.write("TNS       DM      RA          DEC        Z     SNR    WIDTH   B    Gl         Gb         DMG \n")
    for i in np.arange(Nhosts):
        if i==NMAX:
            break
        host=hosts.loc[i]
        j = host["FRB_ID"]
        frb = frbs.loc[j]
        fp.write('{0:8}'.format("FRB"+str(i)))
        fp.write('  {:.2f}'.format(host["DM"]))
        fp.write('  {:.6f}'.format(host["ra"]))
        fp.write('  {:.6f}'.format(host["dec"]))
        fp.write('  {:.3f}'.format(frb["z"]))
        fp.write('  {:.2f}'.format(frb["s"]*10.))
        fp.write('  {:.3f}'.format(frb["w"]))
        fp.write('  {:.3f}'.format(frb["B"]))
        fp.write('  {:.6f}'.format(host["Gl"]))
        fp.write('  {:.6f}'.format(host["Gl"]))
        fp.write('  {:.1f}'.format(host["DMG"]))
        fp.write("\n")
        
    fp.close()





def load_craco_text():
    """
    returns CRACO prefixes and suffixes. Taken from papers/lsst/Photometric
    Work by Bryce Smith and Clancy James
    """
    
    
    Prefix="""# %ECSV 1.0
# ---
# datatype:
# - {name: TNS, datatype: string}
# - {name: DM, datatype: float64}
# - {name: RA, datatype: float64}
# - {name: DEC, datatype: float64}
# - {name: Z, datatype: float64}
# - {name: SNR, datatype: float64}
# - {name: WIDTH, datatype: float64}
# - {name: B, datatype: float64}
# - {name: Gl, unit: deg, datatype: float64}
# - {name: Gb, unit: deg, datatype: float64}
# - {name: DMG, datatype: float64}
# meta: !!omap
# - {survey_data: '{"observing": {"NORM_FRB": 17,"TOBS": 64.68,"MAX_IW": 8, "MAXWMETH": 2"""
# we need to split the obs string into two so we can insert the zfraction as required
    Suffix="""},
#                   "telescope": {"BEAM": "CRACO_900", "DMMASK": "craco_900_mask.npy",
#                                 "DIAM": 12.0, "NBEAMS": 1, "NBINS": 5, "FBAR": 906.0,
#                                 "TRES": 13.8, "FRES": 1.0, "THRESH": 1.01, "SNRTHRESH": 10.0}}'}\n"""

    return Prefix, Suffix




main(NMAX=10000,label="fake_CRACO_900")
main(NMAX=1000,label="short_fake_CRACO_900")
main(NMAX=100,label="very_short_fake_CRACO_900")
