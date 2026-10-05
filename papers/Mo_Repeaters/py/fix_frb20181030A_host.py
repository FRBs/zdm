""" One-off fix of the FRB20181030A host JSON in the FRB repo (Q15/Q23/Q25 of
claude_prompts/frb_host_table.md).  Values checked against Bhardwaj+2021b
(ApJL 919, L24; papers/Bhardwaj_2021_ApJL_919_L24.pdf).

Run in the `astro` env:  python py/fix_frb20181030A_host.py
Then (yourself):  git rm frb/data/Galaxies/FRB20181030A_host.json ; git add frb/data/Galaxies/20181030A/
"""
import os
import json

from astropy.coordinates import SkyCoord

from frb.galaxies import nebular

gal_path = os.path.join(os.getenv('HOME'), 'Astronomy', 'FRB', 'FRB',
                        'frb', 'data', 'Galaxies')
infile = os.path.join(gal_path, 'FRB20181030A_host.json')
outdir = os.path.join(gal_path, '20181030A')
outfile = os.path.join(outdir, 'FRB20181030A_host.json')

d = json.load(open(infile))

# Photometry: rename to defs.valid_filters
#  The fluxes in Bhardwaj+2021b Table 10 are already corrected for Galactic
#  extinction (Schlafly & Finkbeiner 2011) for lambda < 10 um
ren = {'GALEX_FUV': 'GALEX_FUV', 'GALEX_NUV': 'GALEX_NUV',
       'DESI_g': 'DECaL_g', 'DESI_r': 'DECaL_r', 'DESI_z': 'DECaL_z',
       '2MASS_J': '2MASS_j', '2MASS_H': '2MASS_h', '2MASS_Ks': '2MASS_k',
       'WISE_w1': 'WISE_W1', 'WISE_w2': 'WISE_W2',
       'WISE_w3': 'WISE_W3', 'WISE_w4': 'WISE_W4'}
p = d['photom']
new = {}
for old, nw in ren.items():
    new[nw] = p[old]
    new[nw+'_err'] = p[old+'_err']
    new[nw+'_ref'] = 'Bhardwaj2021b'
# Herschel PACS/SPIRE entries are dropped (not in defs.valid_filters)

# EBV: the old 0.42 is the *host* Balmer-decrement E(B-V) (Bhardwaj+2021b,
#  Table 6), not Galactic.  Replace with the Galactic value, as in build_hosts.py
coord = SkyCoord(ra=d['ra'], dec=d['dec'], unit='deg')
new['EBV'] = float(nebular.get_ebv(coord)['meanValue'])
print(f"Galactic E(B-V) = {new['EBV']:.3f}")
d['photom'] = new

# Derived
der = d['derived']
der.pop('Z_spec')            # was a copy of the redshift
der.pop('Mstar_spec')        # mass is from Prospector, not pPXF
der['Mstar'] = 5.8e9         # Bhardwaj+2021b Table 6
der['Mstar_loerr'] = 2.0e9
der['Mstar_uperr'] = 1.6e9
der['Mstar_ref'] = 'Bhardwaj2021b'
der['AV_nebular'] = 1.3      # Balmer decrement; Bhardwaj+2021b Sec 2.4
der['AV_nebular_err'] = 0.2
der['AV_nebular_ref'] = 'Bhardwaj2021b'
# SFR (Q30): the old SFR_nebular = 0.35 is SFR_total = SFR(NUV) + 0.6 SFR(TIR)
#  (log SFR = -0.45 +/- 0.1; Bhardwaj+2021b Table 6)
der['SFR_photom'] = 0.36
der['SFR_photom_err'] = 0.08
der['SFR_photom_ref'] = 'Bhardwaj2021b'
# Halpha SFR from the slit spectrum; only covers part of the galaxy, so a
#  lower limit, flagged by _err = -998 (Q37)
der['SFR_nebular'] = 0.033
der['SFR_nebular_err'] = -998.
der['SFR_nebular_ref'] = 'Bhardwaj2021b'

os.makedirs(outdir, exist_ok=True)
with open(outfile, 'w') as f:
    json.dump(d, f, indent=4, sort_keys=True)
print(f'Wrote {outfile}')
