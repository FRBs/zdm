"""
This script shows how to use the host_model class
to estimate priors on host galaxy magnitudes
"""

from zdm import optical as opt
from zdm import loading
import numpy as np
from zdm import cosmology as cos
from zdm import parameters

from matplotlib import pyplot as plt

def main():
    
    state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    
    model = opt.host_model()
    
    name='CRAFT_ICS_1300'
    
    
    ss,gs = loading.surveys_and_grids(survey_names=[name])
    g = gs[0]
    model.init_zmapping(g.zvals)
    
    if False:
        plot_redshift_mapping(model,zvals)
        
    # for a DM 1500 FRB, what is the posterior magnitude distribution?
    DMlist = np.linspace(200,2000,10)
    
    AppMagPriors,pz = model.get_posterior(g,DMlist)
    
    if False:
        plot_redshift_prior(g.zvals,pz,DMlist)
    
    plot_AppMag_prior(model.AppMags,AppMagPriors,DMlist)

def plot_AppMag_prior(AppMags,Priors,DMlist):
    """
    Plots caluclated prior on hosy magnitudes
    """ 
    plt.figure()
    plt.xlabel("Apparent magnitude, $m_r$")
    plt.ylabel("$p(m_r)|{\\rm DM}$")
    
    #plt.ylim(0,None)
    for i,DM in enumerate(DMlist):
        plt.plot(AppMags,Priors[:,i],label=str(DM))
    plt.legend()
    plt.tight_layout()
    plt.savefig("prior_on_apparent_magnitude.png")
    plt.close()
    
def plot_redshift_prior(zvals,pz,DMlist):
    """
    Plots prior on redshift
    """
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0,2)
    #plt.ylim(0,None)
    for i,DM in enumerate(DMlist):
        plt.plot(zvals,pz[:,i],label=str(DM))
    plt.legend()
    plt.tight_layout()
    plt.savefig("prior_on_redshift.png")
    plt.close()
    

def plot_mag_prior():
    """
    Plots caluclated prior on hosy magnitudes
    """ 
    plt.figure()
    plt.xlabel('Apparent magnitude, $m_r$')
    plt.ylabel("Relative probability, $P(m_r)$")
    plt.plot(model.AppMags,pmags)
    plt.tight_layout()
    plt.savefig("posterior_DM_1500.png")
    plt.close()
    

def plot_redshift_mapping(model,zvals):
    # plots mapping of absolute to apparent magnitudes
    # for each redshift
    plt.figure()
    #plt.hist(model.zmap.flatten())
    
    for i,z in enumerate(zvals):
        plt.plot(model.AppMags,model.maghist[:,i],label=str(z))
        break
    plt.xlabel('Apparent Magnitudes')
    plt.ylabel('Relative contribution to redshift')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
main()
