import argparse
import numpy as np
import os

from zdm import dmg_sanskriti2020

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='b_FRB',type=float,help="Galactic latitude")
    parser.add_argument(dest='l_FRB',type=float,help="Galactic longitude")
    parser.add_argument('--th',dest='sep_th',type=float,default=1,help="Separation threshold (maximum angular separation allowed) in degrees")   
    parser.add_argument('--tol',dest='sep_tol',type=float,default=0.1,help="Separation tolerance (tolerance of minimum angular separation) in degrees")   
    args = parser.parse_args()

    dmg_sanskriti2020.dmg_sanskriti2020(args.l_FRB, args.b_FRB, args.sep_th, args.sep_tol)

main()
