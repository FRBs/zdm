import numpy as np
from matplotlib import pyplot as plt
from zdm import real_loading as loading
from zdm import pcosmic
from zdm import cosmology as cos
from scipy import interpolate
import matplotlib

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    state = init_cos()
    
    fname="Data/all_loc_frb_data_w171020.dat"
    names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,Fnu,zloc,Nants \
             = load_loc_frb_data(fname)
    
    DMhalo = 50. # halo and host
    DMhost = 50.
    DMeg = DM-DMG - DMhalo
    zDM = get_macquart_z(DMeg,state,DMhost)
    
    plt.figure()
    plt.plot([0,1.6],[0,1.6],linestyle=":",color="black",linewidth=1)
    plt.scatter(zloc,zDM,color="blue",marker='o',s=10)
    
    plt.xlim(0,1.6)
    plt.ylim(-0.1,1.5)
    plt.ylabel("$z_{\\rm DM}$")
    plt.xlabel("$z_{\\rm loc}$")
    plt.gca().set_xticks(np.linspace(0,1.5,4))
    plt.gca().set_yticks(np.linspace(0,1.5,4))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig("macquart_scatter.pdf")
    plt.close()

def get_macquart_z(DMeg,state,DMhost):
    """
    gets z(DM) from the Macquart relation
    """
    
    # gets z from macquart relation
    zvals=np.linspace(1e-3,2,2000)
    
    macquart_relation=pcosmic.get_mean_DM(zvals, state)
    macquart_relation += DMhost/(1.+zvals)
    splines = interpolate.CubicSpline(macquart_relation,zvals)
    zFRBs = splines(DMeg)
    
    return zFRBs

def init_cos():
    """
    
    """
    state = loading.set_state()
    state.cosmo.H0 = 70.0
    state.cosmo.Omega_b = 0.0486
    cos.set_cosmology(state)
    cos.init_dist_measures()
    return state

def load_loc_frb_data(fname):
    """
    Loads table of localised FRBs
    """
    data = np.loadtxt(fname)
    names = data[:,0]
    DM = data[:,2]
    freqMHz = data[:,4]
    SNR = data[:,1]
    #SNRth = data[:,4]
    DMG = data[:,3]
    tsamp = data[:,5]
    width = data[:,6]
    BeamThetaDeg = data[:,7]
    oldFnu = data[:,8]
    zloc = data[:,9]
    Nants = data[:,10]
    newFnu = data[:,11]
    
    SNRth = np.full([newFnu.size],9.5)
    
    return names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,newFnu,zloc,Nants


main()
