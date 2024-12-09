"""
Calculates primary beam distance for this FRB. Use this for beam values B!
Although likely this is incorrect... (because the primary beam value is
highly uncertain)
"""

import numpy as np

def main():
    #detection beam was beam 00
    bra = 333.216312589
    bdec = -19.989420334
    bx,by,bz = calc_xyz_deg(bra,bdec)
    
    hra = 22*15. +15*15/60. + 24.61*15/3600
    hdec = -19 - 35/60. - 4/3600.
    hx,hy,hz = calc_xyz_deg(hra,hdec)
    
    ctheta = bx*hx + by*hy + bz*hz
    dtheta = np.arccos(ctheta)
    dtheta_deg = dtheta * 180./np.pi
    print(dtheta_deg)

def calc_xyz_deg(ra,dec):
    """
    
    """
    ra *= np.pi/180.
    dec *= np.pi/180.
    
    x = np.cos(ra)*np.cos(dec)
    y = np.sin(ra)*np.cos(dec)
    z = np.sin(dec)
    return x,y,z

main()
