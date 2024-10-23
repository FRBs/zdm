#!/bin/bash

# runs a likelihood to constrain Emax for a single slice


./constrain_Emax.py --min=40.5 --max=43.5 --nstep=31 -o constrain.npz
