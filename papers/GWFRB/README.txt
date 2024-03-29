###############################################################
Probability calculation scripts for Moroianu et al.,
    "An assessment of the Association Between a Fast Radio
    Burst and Binary Neutron Star Merger"
    

This branch of the main zDM code contains each of four scripts:

calc_pDM.py
    Used to produce Figure 2 of the paper
    Used to calculate P_DM for FRB 20190425A

calc_PS_all_GW.py
    Used to calculate P_S for all GW events
    using the initial selection criteria, i.e.
    given an FRB occurs uniformly between -2.5
    and +0 hr (or any alternative time range)
    after the GW, what is the chance
    it lands in the 90% localisation region?

process_ps_pt.py
    Used to process the output of calc_PS_all_GW.py,
    by converting the coincidence probabilities
    of that script - which must be copied to
    'summary_2hr.txt' - into an effective number
    of extra trials.

pvalue_calculation.py
    This script calculates the p-value of
    the measurement. It assumes one FRB is
    detected in coincidence with GW190425,
    calculates the chance of it passing
    the selection criteria, and for all
    those that do, generates a distribution
    of p'-values given it has passed the
    criteria. The adjusted p-value is
    calculated by accounting for the
    chance of it passing. This is a complicated
    script.
    
    
    
