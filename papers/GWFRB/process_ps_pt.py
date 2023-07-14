"""
This script processes the output from "calc_PS_all_GW.py"

Copy over the p-values for that script to the right-most column
of "summary_2hr.txt". If you wish to perform calculations for
a different time window, do that too.

The idea is to calculate the effective extra trials from considering
other GW events.
"""

import numpy as np

def main():
    tranges=[2.,26.]
    for i,infile in enumerate(['summary_2hr.txt']):
        
        process(infile,tranges[i]/24.,verbose=False)

def process(infile,trange,verbose=False):
    
    data=np.loadtxt(infile, dtype='str')
    
    pvals=data[:,3].astype('float')
    names = data[:,0]
    
    # this first rate is the rate about GW190425
    # rate = 1.93 # units: days
    # this rate is the time-averaged rate over
    # the 85 day overlap period of CHIME catalog 1
    # and O3.
    rate = 1.62 # 85 days, 138 FRBs
    
    # gets rid of the event in question
    remove = np.where(names == 'GW190425')[0]
    pvals[remove]=0. # set to ignore this
    expected = rate*trange
    
    extra=np.sum(pvals)*expected
    print(infile,"Total expected number is ",extra)
    print("Extra trials are ",extra/0.135+1)
    #calculates p some = 1-pnone for temporal
    p_none_T = np.exp(-rate*trange)
    p_some_T = 1.-p_none_T
    
    # spatial 
    p_none_S = np.exp(-pvals)
    p_some_S = 1.-p_none_S
    
    # p some in total is 1-total chance of nothing
    p_some = p_some_S * p_some_T
    p_none = 1.-p_some
    p_none_total = np.prod(p_none)
    p_some_total = 1.-p_none_total
    
    print(infile,"Chance of detecting another event is ",p_some_total)
    
    if verbose:
        for i,pval in enumerate(pvals):
            print(names[i]," P_S = ",pval," p_none = ",p_none_S[i]," total p some = ",p_some[i])
    
    

main()
