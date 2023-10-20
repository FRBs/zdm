import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    data=np.loadtxt('output.dat')
    print(data.shape)
    ntot,ncol = data.shape
    nR = int(ntot/3)
    
    tfracs=[0.01,0.1,1]
    d1 = data[:nR,:]
    d2 = data[nR:2*nR,:]
    d3 = data[2*nR:,:]
    
    markers=['o','+','x']
    styles=['-.','--',':']
    
    plt.figure()
    
    for i,d in enumerate([d1,d2,d3]):
        plt.plot(d[:,0],d[:,3]/1000.,label='$f_{\\rm obs}=$'+ str(tfracs[i]),\
            linestyle=styles[i],marker=markers[i],linewidth=2)
    plt.plot(d[:,0],PoissonM(d[:,0]),label='Poisson',linestyle='-',marker='s',\
        linewidth=2)
    plt.legend()
    plt.xlabel('Expected rate R')
    plt.ylabel('Fraction detected to repeat')
    plt.xscale('log')
    plt.ylim(0,1)
    plt.xlim(0.1,10)
    plt.tight_layout()
    plt.savefig('weibull_sim.pdf')
    plt.close()
    
    plt.figure()
    
    for i,d in enumerate([d1,d2,d3]):
        fraction = d[:,2]/1000. /Poisson1(d[:,0])
        plt.plot(d[:,0],fraction,label='$f_{\\rm obs}=$'+ str(tfracs[i]),\
            linestyle=styles[i],marker=markers[i],linewidth=2)
        
        if (i==0):
            fit = model(d[:,0],1,-0.1)
            plt.plot(d[:,0],fit,label='fit',\
                linestyle=styles[i],marker=markers[i],linewidth=2)
    plt.legend()
    plt.xlabel('Expected rate R')
    plt.ylabel('Fraction of singles relative to Poisson')
    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(0,1)
    plt.xlim(0.1,10)
    plt.tight_layout()
    plt.savefig('fit_fraction.pdf')
    plt.close()
    
    plt.figure()
    
    for i,d in enumerate([d1,d2,d3]):
        plt.plot(d[:,0],d[:,3]/d[:,2],label='$f_{\\rm obs}=$'+ str(tfracs[i]),\
            linestyle=styles[i],marker=markers[i],linewidth=2)
    plt.plot(d[:,0],PoissonM(d[:,0])/Poisson1(d[:,0]),label='Poisson',linestyle='-',marker='s',\
        linewidth=2)
    plt.legend()
    plt.xlabel('Expected rate R')
    plt.ylabel('Fraction detected to repeat')
    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(0,1)
    plt.xlim(0.1,10)
    plt.tight_layout()
    plt.savefig('ratio_weibull_sim.pdf')
    plt.close()
    
    
    weighting = d[:,0]**-1
    NR = np.sum(PoissonM(d[:,0])*weighting)
    NS = np.sum(Poisson1(d[:,0])*weighting)
    print("Poisson",NR,NS,NR/NS)
    rP=NR/NS
    for i,d in enumerate([d1,d2,d3]):
        NR = np.sum(d[:,3]*weighting)
        NS = np.sum(d[:,2]*weighting)
        print(tfracs[i],NR,NS,NR/NS,NR/NS/rP)
    

def model(R,*params):
    """
    model of ratio Rk/Rpoisson as function of true rate R
    """
    R0=params[0]
    power=params[1]
    logR = np.log10(R)
    ratio = ((R/R0)**power)
    #ratio = np.exp(-R/R0)
    return ratio
    

def PoissonM(L):
    """
    Calculates Poisson chance of repetition exactly
    """
    P = 1.- (np.exp(-L) + L*np.exp(-L))
    return P
    
def Poisson1(L):
    """
    Calculates Poisson chance of repetition exactly
    """
    P = L*np.exp(-L)
    return P

main()
