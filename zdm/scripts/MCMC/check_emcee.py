"""
This is a function to test if emcee is running on your computer

It defines a simple probability (log_prob) and 
submits this a a pooled job over your cpus.

Original test code taken from
https://emcee.readthedocs.io/en/stable/tutorials/parallel/
Note a modification to allow emcee to work on mac osx
"""


import time
import numpy as np


def log_prob(theta):
    t = time.time() + np.random.uniform(0.005, 0.008)
    while True:
        if time.time() >= t:
            break
    return -0.5 * np.sum(theta**2)
    
import emcee


np.random.seed(42)
initial = np.random.randn(32, 5)
nwalkers, ndim = initial.shape
nsteps = 20

import os
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing as mp
# this mod is required for running on mac osx
Pool = mp.get_context('fork').Pool
from multiprocessing import cpu_count

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,pool=pool)
    start = time.time()
    sampler.run_mcmc(initial, nsteps, progress=True)
    end = time.time()
    serial_time = end - start
    print("MP took {0:.1f} seconds".format(serial_time))
