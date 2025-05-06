""" 
Simply plots the Macquart relation


"""


from zdm import pcosmic
from zdm.craco import loading
import numpy as np
def main():
    
    # Initialise surveys and grids
    name='CRAFT_ICS_1300'
    
    # approximate best-fit values from recent analysis
    vparams = {}
    vparams['H0'] = 70 # choose your preferred value!
    
    sdir='../data/Surveys/'
    
    nozlist=[]
    s,g = loading.survey_and_grid(
        survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set up new parameters
    g.update(vparams)
    
    zvals=np.linspace(0.1,5.,50)
    macquart_relation=pcosmic.get_mean_DM(zvals, g.state)
    for i,z in enumerate(zvals):
        print(z,macquart_relation[i])
    #print(macquart_relation)
    
main()
