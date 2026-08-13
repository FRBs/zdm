"""
File containing routines to read in host galaxy data
"""

import numpy as np
import pandas as pd

def read_meerkat():
    """
    returns z and mr data from Pastor-Morales et al
    https://arxiv.org/pdf/2507.05982
    Detection method provided in private communication (Pastor-Morales)
    """
    
    data=np.loadtxt("Data/meerkat_mr.txt",comments='#')
    z=data[:,2]
    mr = data[:,3]
    loc = data[:,4] # 1 is coherent beam, 0 incoherent only
    z = np.abs(z) # -ve is
    w = data[:,5] #PO|x
    
    # removes incoherent sum data
    good = np.where(loc==1)[0]
    z=z[good]
    loc=loc[good]
    mr=mr[good]
    w = w[good]
    
    # removes missing data
    good = np.where(z != 9999)
    z = z[good]
    loc=loc[good]
    mr=mr[good]
    w=w[good]
    
    return z,mr,w

def convert_craft():
    """
    CRAFT ICS data
    """
    
    import pandas as pd
    DF = pd.read_csv("Data/CRAFT_ICS_HTR_Catalogue1.csv")
    
    DF2 = pd.DataFrame(DF["TNS"])
    DF2["z"] = DF["Z"]
    DF2.to_csv("Data/temp_craft_hosts.csv",index=False)
    
def read_craft():
    """
    CRAFT ICS data
    """
    
    DF = pd.read_csv("Data/craft_ics_hosts.csv")
    
    z = np.array(DF["z"])
    mr = np.array(DF["mr"])
    nfrb = len(mr)
    w = np.full([nfrb],1.) # artificial, but all are highy confidence
    return z,mr,w
       
    
    
def read_dsa():
    """
    Reads in DSA data from sharma et al
    """
    DF = pd.read_csv("Data/dsa_hosts.csv")
    
    z = np.array(DF["z"])
    mr = np.array(DF["mr"])
    nfrb = len(mr)
    w = np.array(DF["phost"]) # only gives most likely hosts
    return z,mr,w
