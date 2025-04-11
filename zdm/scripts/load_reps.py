""" 
This script shows how to use repeating FRB grids.

It produces four outputs in the "Repeaters" directory,
showing zDM for:
- 1: All bursts (single bursts, and bursts from repeating sources)
- 2: FRBs expected as single bursts
- 3: Repeating FRBs (each source counts once)
- 4: Bursts from repeaters (each source counts Nburst times)

We expect 1 = 2+4 (if not, it's a bug!)

"""
from zdm import parameters
from zdm import loading as loading
from zdm import iteration as it

import matplotlib.pyplot as plt
import numpy as np

import os

def main(plots=False):
    
    # standard 1.4 GHz CRAFT data
    # name = 'CRAFT_ICS'
        
    names = ['CHIME_decbin_0_of_6', 'CHIME_decbin_1_of_6', 'CHIME_decbin_2_of_6', 'CHIME_decbin_3_of_6', 'CHIME_decbin_4_of_6', 'CHIME_decbin_5_of_6']
    
    state = parameters.State()
    state.update_param('min_lat', None)
    # state.update_param('lRmin', -1.0)
    # state.update_param('lRmax', -0.9)
    # state.update_param('Rgamma', -3.0)
    
    sdir='../data/Surveys/CHIME'
    # use loading.survey_and_grid for proper estimates
    # remove loading for width-based estimates
    # the below is hard-coded for a *very* simplified analysis!
    # using loading. gives 5 beams and widths, ignoring that gives a single beam
    ss,gs = loading.surveys_and_grids(init_state=state, survey_names=names,repeaters=True,sdir=sdir)
    s = ss[0]
    g = gs[0]

    print("Repeater dimension: " + str(s.nDr))
    print("Singles dimension: " + str(s.nDs))
    print("FRB dimension: " + str(s.nD))
    # print(s.nozreps, s.zreps, s.nozsingles, s.zsingles)

    print("Rmin:", g.Rmin)
    print("Rmax:", g.Rmax)
    print("Rgamma:", g.Rgamma)

    it.minimise_const_only(None, gs, ss, update=True)
    for g in gs:
        print("lC, Rc", g.state.FRBdemo.lC, g.Rc)

    # ll_sum = 0
    # for g,s in zip(gs, ss):
    #     ll = it.get_log_likelihood(g,s,pNreps=True,Pn=True)
    #     ll_sum += ll
    #     print(ll)
    
    # print(ll_sum)

    if plots:
        print("Making plots")

        # Create subplots
        fig, axes = plt.subplots(len(ss), 2, figsize=(8, 2*len(ss)))

        # Plot data
        for i in range(len(ss)):
            s = ss[i]
            g = gs[i]

            dm_singles = np.sum(g.exact_singles, axis=0)
            dm_singles = dm_singles / np.sum(dm_singles) / (g.dmvals[1] - g.dmvals[0])
            counts, bins = np.histogram(s.DMEGs[s.zsingles])

            axes[i, 0].plot(g.dmvals, dm_singles)
            axes[i, 0].hist(bins[:-1], bins, weights=counts, density=True)
            axes[i, 0].set_title(s.name)
            axes[i, 0].set_xlabel('DM')
            axes[i, 0].set_ylabel('p(DM)')
            axes[i, 0].set_xlim(0, 2000)

            dm_reps = np.sum(g.exact_reps, axis=0)
            dm_reps = dm_reps / np.sum(dm_reps) / (g.dmvals[1] - g.dmvals[0])
            counts, bins = np.histogram(s.DMEGs[s.zreps])

            axes[i, 1].plot(g.dmvals, dm_reps)
            axes[i, 1].hist(bins[:-1], bins, weights=counts, density=True)
            # axes[i, 1].set_title(s.name)
            axes[i, 1].set_xlabel('DM')
            axes[i, 1].set_ylabel('p(DM)')
            axes[i, 1].set_xlim(0, 2000)

        # Adjust layout
        plt.tight_layout()
        plt.savefig('repeaters.png')
        plt.show()


main(True)
