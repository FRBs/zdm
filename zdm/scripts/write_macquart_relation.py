""" 
Simply writes out the Macquart relation

"""

from zdm import pcosmic
from zdm import loading
from zdm import states
import numpy as np

def main():
    
    # Initialise surveys and grids
    name='CRAFT_ICS_1300'
    
    # approximate best-fit values from recent analysis
    state=states.load_state()
    state.cosmo.H0 = 70.0
    print(state)
    sdir='../data/Surveys/'
    
    #print(state)
    
    zvals=np.linspace(0.01,5.,500)
    macquart_relation=pcosmic.get_mean_DM(zvals, state)
    for i,z in enumerate(zvals):
        print(z,macquart_relation[i])
    
    
main()
