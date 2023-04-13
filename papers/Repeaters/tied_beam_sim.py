"""

Calculates CHIME beams


Key assumptions:
    - Assumes a locally flat sky when calculating a synthesized beam
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
import scipy as sp
from scipy import interpolate
import os
import chime_frb_constants
import cfbm
import time

# CHIME latitude - hard coded
CHIME_lat_deg = 49. + 19/60. + 15/3600.
CHIMEcolours = ['blue','red','green','pink']
CHIMEfreqs = [760,640,540,410] #MHz
# number of frequencies specified by CHIME for their beam
Nfreq = 4

def main():
    """
    main program
    """
    
    global decs_u, decs_l, ras_u, ras_l, result_u, result_l, Ndec2, Nra, Ndec
    global xs_u,ys_u,xs_l,ys_l
    
    mode = 'Standard'
    mode = 'Tied'
    
    
    #### defines beam settings #####
    if mode == 'Standard':
        print("Using standard beam config (tied and primary)")
        opdir='TiedBeamSimulation/'
        Nra=300
        bm = cfbm.CompositeBeamModel(interpolate_bad_freq=True)
        ##### defines frequency settings #####
        freqs = chime_frb_constants.FREQ
        every=1024
        off=int(every/2)
        use_freqs = np.array(freqs[off::every])
    else:
        ####### formed beams only #######
        print("Using tied beam mode only")
        opdir='FormedBeamSimulation/'
        Nra = 80
        bm = cfbm.FFTFormedSincNSBeamModel()
        freqs = chime_frb_constants.FREQ
        use_freqs = np.array([freqs[8192]])
        
    beams0=np.arange(256)
    beams=np.zeros([1024],dtype='int')
    beams[0:256]=beams0
    beams[256:512]=beams0+1000
    beams[512:768]=beams0+2000
    beams[768:1024]=beams0+3000
    
    Ndec = 1000
    
    # the below are ra/dec grids for upper and lower transits
    # decs_u,ras_u,decs_l,ras_l = 
    t0=time.time()
    generate_ra_dec_grid(doplot=False,max_angle=2)
    t1=time.time()
    print("Grid generaton tool ",t1-t0," seconds")
    
    
    ###### define positions ######
    pos_u = np.array([xs_u,ys_u]).T
    pos_l = np.array([xs_l,ys_l]).T
    
    #envelope_u = np.zeros([xs_u.size])
    #envelope_l = np.zeros([xs_l.size])
    
    for i,beam in enumerate(beams):
        t0=time.time()
        #print("inputs :",[beam],pos_u,use_freqs)
        fsens_u = bm.get_sensitivity([beam],pos_u,use_freqs)
        fsens_l = bm.get_sensitivity([beam],pos_l,use_freqs)
        t1=time.time()
        print("Beam ",i,"allf took ",t1-t0,"s")
        
        # averages over frequency
        sens_u=np.average(fsens_u,axis=2)[:,0]
        sens_l=np.average(fsens_l,axis=2)[:,0]
        
        bigger_u = np.where(result_u < sens_u)[0]
        bigger_l = np.where(result_l < sens_l)[0]
        
        result_u[bigger_u] = sens_u[bigger_u]
        result_l[bigger_l] = sens_l[bigger_l]
        
    
    
    plot_points=False
    if plot_points:
        plt.figure()
        plt.scatter(xs_u,ys_u)
        plt.scatter(xs_l,ys_l)
        plt.xlabel('x')
        plt.ylabel('s')
        plt.tight_layout()
        plt.savefig('sampled_beam_points.pdf')
        plt.close()
    
    
    decs_u.reshape(Nra,Ndec)
    
    decs_u = decs_u.reshape([Ndec,Nra])
    decs_l = decs_l.reshape([Ndec2,Nra])
    ras_u = ras_u.reshape([Ndec,Nra])
    ras_l = ras_l.reshape([Ndec2,Nra])
    #result_u = result_u.reshape([Nfreq,Ndec,Nra])
    #result_l = result_l.reshape([Nfreq,Ndec2,Nra])
    np.save(opdir+"decs_u.npy",decs_u)
    np.save(opdir+"decs_l.npy",decs_l)
    np.save(opdir+"ras_u.npy",ras_u)
    np.save(opdir+"ras_l.npy",ras_l)
    np.save(opdir+"result_u.npy",result_u)
    np.save(opdir+"result_l.npy",result_l)


def generate_ra_dec_grid(max_zenith=60.,max_angle=9.,doplot=False):
    """
    generates one giant list of ra,dec coordinates at which we want to evaluate
    Near the equator, we want a grid with width max_angle
    Near the pole, must be wider in ra
    
    inputs:
        max zenith angle to consider
        Nra: number of points in ra direction
        Ndec: number of points in dec direction
        max_angle: max distance from the zenith line
        
    Solving for ra limits:
        - zenith line is on x-z plane
        - y is distance from that plane
        - angular distance larger than y
        - solving for y = max_angle => we are conservative. Good!
        - y coordinate is cos(dec) * sin(ra)
    """
    # this routine will generate the following global variables
    global decs_u, decs_l, ras_u, ras_l, result_u, result_l
    global xs_u,ys_u,xs_l,ys_l
    global Nfreq, Ndec, Ndec2, Nra
    
    # calculates declinations
    min_dec = CHIME_lat_deg - max_zenith
    max_dec = 90.
    ddec = (max_dec-min_dec)/Ndec
    max_dec -= ddec/2.
    min_dec += ddec/2.
    decs_deg = np.linspace(min_dec,max_dec,Ndec)
    decs_rad = decs_deg * np.pi/180.
    
    max_ra = max_angle * np.pi/180.
    max_angle_rad = max_angle * np.pi/180.
    
    # sets up array aboe the pole
    # this array can only go from -90 to +90 in ra.
    # the next one handles the other side
    ras_u = np.zeros([Ndec,Nra])
    for i,dec in enumerate(decs_rad):
        sinra = max_angle_rad/np.cos(dec)
        if sinra > 1.:
            maxra = np.pi/2.
        else:
            maxra = np.arcsin(sinra)
        maxra_deg = maxra * 180./np.pi
        dra = maxra_deg*2./Nra
        ras_u[i,:] = np.linspace(-maxra_deg+dra/2.,maxra_deg-dra/2.,Nra)
    
        
    # now sets up array on the other side of the pole
    # CHIME_lat_deg + max_zenith angle is the angle away from the equator
    # when this gets above 90 degrees, we count down from 90
    # hence it's 90 - (CHIME_lat_deg + max_zenith -90)
    min_dec = 180.-CHIME_lat_deg - max_zenith
    #max_dec = 90.
    #Ndec2 = int((max_dec - min_dec)/ddec)
    #ddec = (max_dec-min_dec)/Ndec2
    #max_dec -= ddec/2.
    #min_dec += ddec/2.
    #decs2_deg = np.linspace(min_dec,max_dec,Ndec2)
    doubled = np.where(decs_deg > min_dec)[0]
    decs2_deg = decs_deg[doubled]
    decs2_rad = decs_rad[doubled]
    Ndec2 = len(doubled)
    
    ras_l = np.zeros([Ndec2,Nra])
    
    
    # sets up array above the pole
    # this array can only go from -90 to +90 in ra.
    # the next one handles the other side
    for i,dec in enumerate(decs2_rad):
        sinra = max_angle_rad/np.cos(dec)
        if sinra > 1.:
            maxra = np.pi/2.
        else:
            maxra = np.arcsin(sinra)
        maxra_deg = maxra * 180./np.pi
        dra = maxra_deg*2./Nra
        ras_l[i,:] = np.linspace(-maxra_deg+dra/2.+180.,maxra_deg-dra/2.+180.,Nra)
    
    decs_u = np.tile(decs_deg,(Nra,1)).T
    decs_l = np.tile(decs2_deg,(Nra,1)).T
    
    if doplot:
        
        plt.figure()
        plt.xlabel('ra')
        plt.ylabel('dec')
        plt.xlim(-90,270)
        plt.ylim(-10,90)
        plt.scatter(ras_u.flatten(),decs_u.flatten(),color='blue',label='Upper transit')
        plt.scatter(ras_l.flatten(),decs_l.flatten(),color='red',label='lower transit')
        plt.legend()
        plt.tight_layout()
        plt.savefig('sampled_points.pdf')
        plt.close()
    
    ### generates grids to hold the results
    # these grids hold the beam values as a function of declination
    
    
    result_u = np.zeros([Ndec*Nra])
    result_l = np.zeros([Ndec2*Nra])
    result_u = result_u.flatten()
    result_l = result_l.flatten()
    
    decs_u = decs_u.flatten()
    ras_u = ras_u.flatten()
    decs_l = decs_l.flatten()
    ras_l = ras_l.flatten()
    
    ## converts to x/y coordinates
    
    # this time approximately equals transit
    time=18.95/360. * 24 * 3600
    coords_l=cfbm.get_position_from_equatorial(ras_l,decs_l,time=time)
    xs_l = coords_l[0]
    ys_l = coords_l[1]
    coords_u=cfbm.get_position_from_equatorial(ras_u,decs_u,time=time)
    xs_u = coords_u[0]
    ys_u = coords_u[1]
    
    #return decs_deg,ras,decs2_deg,ras2

main()
