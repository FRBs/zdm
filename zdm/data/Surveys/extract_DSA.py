"""
File: extract_DSA.py
Author: Jordan Hoffmann
Date: 14/11/23
Purpose: 
    Convert CSV files from https://code.deepsynoptic.org/dsa110-archive/ to an ecsv file
"""

import argparse
from astropy.table import Table
from astropy.table import MaskedColumn
from astropy.coordinates import SkyCoord
from astropy import units as u

import json

import numpy as np

from ne2001 import density

import os

#==============================================================================

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='infile', nargs='+', type=str, help="list of CSV files to extract")
    parser.add_argument('-o', dest='outfile', default="DSA.ecsv", type=str, help="Output filename")
    parser.add_argument('-b', dest='BW', default=187.5, type=int, help="Bandwidth (MHz)")
    parser.add_argument('--fbar', dest='FBAR', default=1405, type=int, help="Central observational frequency (MHz)")
    parser.add_argument('--df', dest='FRES', default=0.244141, type=float, help="Frequency resolution (MHz)")
    parser.add_argument('-s', dest='SNRTHRESH', default=8.5, type=float, help="SNR threshold")
    parser.add_argument('--th', dest='THRESH', default=1.187, type=float, help="Fluence threshold (Jy ms)")
    parser.add_argument('-t', dest='TOBS', default=None, type=float, help="Observation time (hours)")
    parser.add_argument('--dt', dest='TRES', default=0.262144, type=float, help="Time resolution (ms)")
    parser.add_argument('-n', dest='NBEAMS', default=256, type=float, help="Number of beams")

    args = parser.parse_args()

    # Read in csvs
    data_list = []
    for infile in args.infile:
        data = np.genfromtxt(infile, skip_header=1, delimiter=',', dtype=str)
        data_list.append(data)

    # Columns in data_array correspond to:
    # Num, Internal Naame, MJD, DM, Width, SNR, RA, DEC, PosErr, Notes, Version, DOI    
    data_array = np.concatenate(data_list)

    t = Table()
    t.meta['survey_data'] = {}

    if args.TOBS != None:
        t.meta['survey_data']['observing'] = {
            'NORM_FRB': data_array.shape[0],
            'TOBS': args.TOBS
        }

    t.meta['survey_data']['telescope'] = {
        'BMETHOD': 0, # Gaussian beam
        'DIAM': 4.65,
        'NBEAMS': args.NBEAMS,
        'NBINS': 10
    }

    t.meta['survey_data'] = json.dumps(t.meta['survey_data'])

    XDec = [float(data_array[i,7][1:-1]) for i in range(data_array.shape[0])]
    XRA = [float(data_array[i,6][1:-1]) for i in range(data_array.shape[0])]

    coords = SkyCoord(ra=XRA, dec=XDec, frame='icrs', unit="deg")
    gcoords = coords.galactic

    Gl = gcoords.l
    Gb = gcoords.b

    ne = density.ElectronDensity() #default position is the sun
    DMGs=np.zeros(len(Gl))*u.pc/u.cm**3
    for i,l in enumerate(Gl):
        b=Gb[i]
        DMGs[i] = ne.DM(l, b, 100.)
        
    t['TNS'] = [data_array[i,9][5:14] for i in range(data_array.shape[0])]
    t['BW'] = MaskedColumn([args.BW for _ in range(data_array.shape[0])], dtype='float64')
    t['DM'] = MaskedColumn([float(data_array[i,3][1:-1]) for i in range(data_array.shape[0])], dtype='float64')
    t['DMG'] = MaskedColumn(DMGs, dtype='float64')
    t['FBAR'] = MaskedColumn([args.FBAR for _ in range(data_array.shape[0])], dtype='float64')
    t['FRES'] = MaskedColumn([args.FRES for _ in range(data_array.shape[0])], dtype='float64')
    t['Gb'] = Gb
    t['Gl'] = Gl
    t['NREP'] = MaskedColumn(np.ones(data_array.shape[0]), dtype='int64')
    t['SNR'] = MaskedColumn([float(data_array[i,5][1:-1]) for i in range(data_array.shape[0])], dtype='float64')
    t['SNRTHRESH'] = MaskedColumn([args.SNRTHRESH for _ in range(data_array.shape[0])], dtype='float64')
    t['THRESH'] = MaskedColumn([args.THRESH for _ in range(data_array.shape[0])], dtype='float64')
    t['TRES'] = MaskedColumn([args.TRES for _ in range(data_array.shape[0])], dtype='float64')
    t['WIDTH'] = MaskedColumn([float(data_array[i,4][1:-1]) for i in range(data_array.shape[0])], dtype='float64')
    t['XDec'] = MaskedColumn(XDec, dtype='float64')
    t['XRA'] = MaskedColumn(XRA, dtype='float64')
    t['Z'] = MaskedColumn([np.ma.masked for _ in range(data_array.shape[0])], dtype='float64')

    # t.write(args.outfile, format='ascii.ecsv')

#     header=f"""# %ECSV 1.0
# # ---
# # datatype:
# # - {{name: TNS, datatype: string}}
# # - {{name: BW, datatype: float64}}
# # - {{name: DM, datatype: float64}}
# # - {{name: DMG, datatype: float64}}
# # - {{name: FBAR, datatype: float64}}
# # - {{name: FRES, datatype: float64}}
# # - {{name: Gb, datatype: float64}}
# # - {{name: Gl, datatype: float64}}
# # - {{name: NREP, datatype: int64}}
# # - {{name: SNR, datatype: float64}}
# # - {{name: SNRTHRESH, datatype: float64}}
# # - {{name: THRESH, datatype: float64}}
# # - {{name: TRES, datatype: float64}}
# # - {{name: WIDTH, datatype: float64}}
# # - {{name: XDec, datatype: string, subtype: 'float64[null]'}}
# # - {{name: XRA, datatype: string, subtype: 'float64[null]'}}
# # - {{name: Z, datatype: string, subtype: 'float64[null]'}}
# # meta: !!omap
# # - {{survey_data: "{{\\n    \\"observing\\": {{\\n        \\"NORM_FRB\\": {data_array.shape[0]},\\n        \\"TOBS\\": {args.TOBS}\\n    }},\\n    \\"telescope\\": {{\\n        \\"\\
# #     BEAM\\": \\"parkes_mb_log\\",\\n        \\"DIAM\\": 4.65,\\n        \\"NBEAMS\\": 256,\\n        \\"NBINS\\": 1\\n    }}\\n}}"}}
# # schema: astropy-2.0
# TNS BW DM DMG FBAR FRES Gb Gl NREP SNR SNRTHRESH THRESH TRES WIDTH XDec XRA Z
# """

#     with open(args.outfile, "w") as f:
#         f.write(header)
#         for i in range(data_array.shape[0]):
#             line = data_array[i,9][5:14] + " " \
#                     + str(args.BW) + " " \
#                     + data_array[i,3][1:-1] + " " \
#                     + "\"\"" + " " \
#                     + str(args.FBAR) + " " \
#                     + str(args.FRES) + " " \
#                     + "\"\"" + " " \
#                     + "\"\"" + " " \
#                     + "1" + " " \
#                     + data_array[i,5][1:-1] + " " \
#                     + str(args.SNRTHRESH) + " " \
#                     + str(args.THRESH) + " " \
#                     + str(args.TRES) + " " \
#                     + data_array[i,4][1:-1] + " " \
#                     + data_array[i,7][1:-1] + " " \
#                     + data_array[i,6][1:-1] + " " \
#                     + "\"\"\n"
            
#             f.write(line)

#==============================================================================

main()