""" Build CHIME_FRB_hosts.csv: host-galaxy properties of CHIME FRBs
for Clancy's repeater analysis.

See claude_prompts/frb_host_table.md for the decisions (Q1-Q40) behind
every choice made here.

Sources, in order of precedence:
  1. Host JSONs in the FRB repo (frb/data/Galaxies)
  2. FRBs_base.csv (redshift fall-back, repeater flag, P(O|x))
  3. Literature values in LIT_HOSTS below, verified against the PDFs
     in papers/ (only used where the repo has nothing)

Run in the `astro` env from papers/Mo_Repeaters:
    python py/build_host_table.py
"""
import os
import glob
import json
import warnings

import numpy as np
import pandas

from astropy.coordinates import SkyCoord
from astropy import units

from frb.galaxies import nebular
from frb.galaxies import photom as frbphotom
from frb.surveys import survey_utils

FRB_REPO = os.path.join(os.getenv('HOME'), 'Astronomy', 'FRB', 'FRB')
GAL_PATH = os.path.join(FRB_REPO, 'frb', 'data', 'Galaxies')
BASE_FILE = os.path.join(FRB_REPO, 'frb', 'data', 'FRBs', 'FRBs_base.csv')

this_path = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(this_path, '..', 'CHIME_FRB_hosts.csv')
PHOTOM_CACHE = os.path.join(this_path, 'lit_host_photom_cache.json')

# Q33: ignore host values for FRBs with a PATH P(O|x) below this
POX_MIN = 0.9
# Set to True to drop those rows altogether (instead of blanking host values)
DROP_LOW_POX = False

# Q12/Q21: CHIME-detected FRBs localized by another instrument (in the base table)
EXTRA_BASE = ['FRB20121102A', 'FRB20201124A']

# Q32: repo entries stored under a later burst name -> source name
RENAME = {'FRB20231204A': 'FRB20190303A',   # KKO (ApJS 280, 6) Sec 6.1
          'FRB20231128A': 'FRB20191106C'}   # KKO Sec 6.6

# Reference labels
REF_MAP = {'Leung+2025': 'CHIME/FRB+2025(ApJS280,6)'}
LEUNG_L = 'Leung+2025(ApJL991,L25)'

# Mstar_ref values known to be Prospector fits (Q26)
PROSPECTOR_REFS = ['Gordon2023', 'Bhardwaj2021b']

# Mag ranking (Q18): pointed > DECaL/DELVE/DES > Pan-STARRS > SDSS
POINTED = ['WFC3_F606W', 'LRISr_R', 'VLT_FORS2_R', 'GMOS_S_r', 'GMOS_N_r',
           'HSC_r', 'DEIMOS_r', 'NOT_r']
MAG_RANK = POINTED + ['DECaL_r', 'DELVE_r', 'DES_r', 'Pan-STARRS_r', 'SDSS_r']

# ---------------------------------------------------------------------------
# Literature values (verified; see the Missing Repeaters table and Q40)
#   logM: log10 Mstar/Msun; logM_err: symmetrized dex (None = blank, Q34)
#   SFR entries: (value, err, flag, type, ref); flag 0=meas., 1=upper, 2=lower
#   mag: only set when the paper states the value is Galactic-extinction
#        corrected; otherwise taken from Legacy Survey / Pan-STARRS (Q39)
#   mag_fallback: paper value, corrected here, if no catalog detection
LIT_HOSTS = {
    'FRB20180814A': dict(
        coord='04h22m56.01s +73d39m40.7s',   # PanSTARRS-DR1 J042256.01+733940.7
        z=0.06835, z_ref='Michilli+2023',
        logM=10.78, logM_err=(0.12+0.18)/2, Mstar_source='Prospector:Michilli+2023',
        SFR=[(10**-0.5, None, 1, 'SED', 'Michilli+2023')],
        mag_fallback=(17.15, None, 'Pan-STARRS_r', 'Michilli+2023')),
    'FRB20190110C': dict(
        coord='16h37m16.43s +41d26m36.30s',   # Ibik+2024a Table 3
        z=0.12244, z_ref='Ibik+2024a',
        logM=np.log10(2.5e10),
        logM_err=(np.log10(2.6e10/2.5e10) + np.log10(2.5e10/2.33e10))/2,
        Mstar_source='Prospector:Ibik+2024a',
        SFR=[(0.1575, 0.0006, 0, 'nebular', 'Ibik+2024a'),
             (0.54, 0.04, 0, 'SED', 'Ibik+2024a')]),
    'FRB20200223B': dict(
        coord='00h33m04.68s +28d49m52.60s',   # Ibik+2024a Table 3
        z=0.06024, z_ref='Ibik+2024a',
        logM=np.log10(5.6e10),
        logM_err=(np.log10(6.74e10/5.6e10) + np.log10(5.6e10/4.67e10))/2,
        Mstar_source='Prospector:Ibik+2024a',
        SFR=[(0.59, 0.04, 0, 'SED', 'Ibik+2024a')]),  # Halpha N/A (AGN)
    'FRB20190417A': dict(
        coord='19h39m05.892s +59d19m36.99s',  # FRB position (Moroianu+2025)
        z=0.12817, z_ref='Ibik+2024b',
        logM=7.88, logM_err=(0.12+0.14)/2, Mstar_source='Prospector:Moroianu+2025',
        SFR=[(0.19, 0.01, 0, 'nebular', 'Moroianu+2025')],
        mag_fallback=(22.42, 0.06, 'GMOS_N_r', 'Moroianu+2025')),
    'FRB20220912A': dict(
        coord='347.2702 +48.7066',             # PSO J347.2702+48.7066
        z=0.0771, z_ref='Ravi+2023',
        logM=10.0, logM_err=0.1, Mstar_source='Prospector:Ravi+2023',
        SFR=[(0.1, None, 2, 'nebular', 'Ravi+2023')],
        mag_fallback=(19.65, None, 'Pan-STARRS_r', 'Ravi+2023')),
    'FRB20240114A': dict(
        coord='21h27m39.84s +04d19m45.8s',    # Bhardwaj+2025 Table 3
        z=0.130287, z_ref='Bhardwaj+2025',
        logM=8.55, logM_err=(0.12+0.14)/2, Mstar_source='Prospector:Bhardwaj+2025',
        SFR=[(0.061, (0.004+0.003)/2, 0, 'nebular', 'Bhardwaj+2025')],
        mag_fallback=(21.94, 0.11, 'DECaL_r', 'Bhardwaj+2025')),
    'FRB20240209A': dict(
        coord='289.85036 +86.06090',           # Eftekhari+2024 Table 2
        z=0.1384, z_ref='Eftekhari+2024',
        logM=11.34, logM_err=0.01, Mstar_source='Prospector:Eftekhari+2024',
        SFR=[(0.36, None, 1, 'SED', 'Eftekhari+2024')],
        mag=(16.79, 0.02, 'GMOS_N_r', 'Eftekhari+2024')),  # MW-corrected (Tab 1)
    'FRB20190208A': dict(
        coord='18h54m11.27s +46d55m21.67s',   # EVN position (Hewitt+2024)
        z=None, logM=None, SFR=[],
        mag_fallback=(27.32, 0.16, 'GTC_r', 'Hewitt+2024')),
    'FRB20181119A': dict(coord=None, z=None, logM=None, SFR=[]),
}

# Literature values that supplement repo hosts lacking them
#  (keyed on the CSV name, i.e. after RENAME)
LIT_SUPPLEMENT = {
    # Q38: tiers -> Leung CIGALE wins over Chang+2015 (log 10.65)
    'FRB20191106C': dict(logM=9.47, Mstar_source='Photometric:CIGALE:'+LEUNG_L,
                         SFR=[(1.53, None, 0, 'nebular', 'Ibik+2024a')],
                         note='Mstar_alt:10.65(Chang+2015 via Ibik+2024a)'),
    'FRB20190303A': dict(logM=10.75, logM_err=0.03, Mstar_source='Other:SDSS:Michilli+2023',
                         SFR=[(10**0.84, np.log(10)*10**0.84*0.04, 0, 'SDSS', 'Michilli+2023')]),
    'FRB20230926A': dict(logM=10.49, Mstar_source='Photometric:CIGALE:'+LEUNG_L),
    'FRB20231011A': dict(logM=9.59, Mstar_source='Photometric:CIGALE:'+LEUNG_L),
    'FRB20231123A': dict(logM=9.42, Mstar_source='Photometric:CIGALE:'+LEUNG_L),
    'FRB20231201A': dict(logM=9.47, Mstar_source='Photometric:CIGALE:'+LEUNG_L),
    'FRB20231229A': dict(logM=9.87, Mstar_source='Photometric:CIGALE:'+LEUNG_L),
    'FRB20231230A': dict(logM=10.04, Mstar_source='Photometric:CIGALE:'+LEUNG_L),
    'FRB20230222B': dict(logM=10.19, Mstar_source='Other:NED-LVS:'+LEUNG_L),
    'FRB20231223C': dict(logM=10.40, Mstar_source='Other:NED-LVS:'+LEUNG_L),
}

# FRB20181030A: corrections from Bhardwaj+2021b (Q30), applied in memory
#  until py/fix_frb20181030A_host.py has been run on the FRB repo
FIX_181030A = {
    'derived': {'Mstar': 5.8e9, 'Mstar_loerr': 2.0e9, 'Mstar_uperr': 1.6e9,
                'Mstar_ref': 'Bhardwaj2021b',
                'SFR_photom': 0.36, 'SFR_photom_err': 0.08,
                'SFR_photom_ref': 'Bhardwaj2021b',
                'SFR_nebular': 0.033, 'SFR_nebular_err': -998.,
                'SFR_nebular_ref': 'Bhardwaj2021b'},
    'rename_photom': {'DESI_r': 'DECaL_r'},
}


def load_host_json(name):
    """ Return the host JSON dict for an FRB, or None """
    files = glob.glob(os.path.join(GAL_PATH, name[3:], f'{name}_host.json'))
    if len(files) == 1:
        return json.load(open(files[0]))
    if name == 'FRB20181030A':
        # Not yet moved/fixed in the FRB repo
        old = os.path.join(GAL_PATH, 'FRB20181030A_host.json')
        if os.path.isfile(old):
            d = json.load(open(old))
            for o, n in FIX_181030A['rename_photom'].items():
                d['photom'][n] = d['photom'].pop(o)
                d['photom'][n+'_err'] = d['photom'].pop(o+'_err')
            for key in ['Mstar_spec', 'SFR_nebular_err', 'Z_spec']:
                d['derived'].pop(key, None)
            d['derived'].update(FIX_181030A['derived'])
            return d
    return None


def sym_err(der, key, log=False):
    """ Symmetrized error for der[key]; in dex if log """
    val = der[key]
    if key+'_loerr' in der and key+'_uperr' in der:
        lo, up = der[key+'_loerr'], der[key+'_uperr']
    elif key+'_err' in der:
        lo = up = der[key+'_err']
    else:
        return None
    if log:
        return (np.log10(val+up) - np.log10(val-lo))/2
    return (lo+up)/2


def good_mag(phot, band):
    """ A usable measurement (or a 999 upper limit)? """
    if band not in phot or not np.isfinite(phot[band]) or phot[band] <= -999.:
        return False
    err = phot.get(band+'_err')
    return err is None or err == 999. or (np.isfinite(err) and err >= 0.)


def best_mag(phot):
    """ Pick the r-band magnitude per the Q18 ranking (falls back to any band) """
    for band in MAG_RANK:
        if good_mag(phot, band):
            return band
    # Not r-band: take the bluest available optical band from these surveys
    for band in ['Pan-STARRS_i', 'DECaL_z', 'Pan-STARRS_z', 'Pan-STARRS_y']:
        if good_mag(phot, band):
            return band
    return None


def repo_mstar(der):
    """ Mstar from the repo per the Q7/Q16/Q26 tiers -> (logM, err, source) """
    if 'Mstar' in der:
        ref = der.get('Mstar_ref', 'repo')
        if ref in PROSPECTOR_REFS:
            tier = 'Prospector'
        elif ref.startswith('Bhardwaj2021'):  # M81 (FRB20200120E); method not CIGALE
            tier = 'Other'
        else:
            tier = 'Photometric'
        return (np.log10(der['Mstar']), sym_err(der, 'Mstar', log=True),
                f'{tier}:{ref}')
    if 'Mstar_spec' in der:
        return (np.log10(der['Mstar_spec']), sym_err(der, 'Mstar_spec', log=True),
                'Spectroscopic:'+der.get('Mstar_spec_ref', 'repo'))
    return None


def repo_sfrs(der):
    """ List of SFR candidates (value, err, flag, type, ref) in Q8 order """
    out = []
    for key, stype in [('SFR_nebular', 'nebular'), ('SFR_SED', 'SED'),
                       ('SFR_photom', 'photom')]:
        if key not in der:
            continue
        err = sym_err(der, key)
        flag = 0
        if err is not None and err == 999.:
            flag, err = 1, None
        elif err is not None and err == -998.:
            flag, err = 2, None
        elif err is not None and err == -999.:
            err = None
        if stype == 'photom' and der.get(key+'_ref') == 'Bhardwaj2021b':
            stype = 'UV+IR'
        out.append((der[key], err, flag, stype, der.get(key+'_ref', 'repo')))
    return out


def choose_sfr(cands):
    """ Q37: first measured value in priority order; else first limit """
    for c in cands:
        if c[2] == 0:
            return c
    return cands[0] if len(cands) > 0 else None


def catalog_mag(name, coord, cache):
    """ Legacy Survey / Pan-STARRS r-band, MW-corrected (G23; Q39) """
    if name in cache:
        return cache[name]
    result = None
    EBV = float(nebular.get_ebv(coord)['meanValue'])
    for survey, band in [('DECaL', 'DECaL_r'), ('Pan-STARRS', 'Pan-STARRS_r')]:
        try:
            tbl = survey_utils.load_survey_by_name(
                survey, coord, 2*units.arcsec).get_catalog()
        except Exception as e:
            warnings.warn(f'{name}: {survey} query failed: {e}')
            continue
        if tbl is None or len(tbl) == 0 or band not in tbl.keys():
            continue
        # Nearest object
        sep = coord.separation(SkyCoord(ra=tbl['ra'], dec=tbl['dec'], unit='deg'))
        row = tbl[np.argmin(sep)]
        if not np.isfinite(row[band]) or row[band] <= 0.:
            continue
        corr = frbphotom.extinction_correction(band, EBV)
        result = dict(mag=float(row[band]) - 2.5*np.log10(corr),
                      err=float(row[band+'_err']), band=band, EBV=EBV)
        break
    if result is None:
        result = dict(EBV=EBV)
    cache[name] = result
    return result


def main():
    base = pandas.read_csv(BASE_FILE)
    sample = base[(base.telescope == 'CHIME') | base.Name.isin(EXTRA_BASE)]

    cache = json.load(open(PHOTOM_CACHE)) if os.path.isfile(PHOTOM_CACHE) else {}

    rows = []
    # ---- Base-table FRBs
    for _, b in sample.iterrows():
        name = RENAME.get(b.Name, b.Name)
        refs = []
        if name != b.Name:
            refs.append(f'name:repo_as_{b.Name}')
        row = dict(TNS_NAME=name, Repeater=bool(b.repeater) or name in RENAME.values())
        pox = b['P(O|x)']
        if np.isfinite(pox) and pox < POX_MIN:
            if DROP_LOW_POX:
                continue
            refs.append(f'host_ignored:P(O|x)={pox:.3f}<{POX_MIN}')
            row['Refs'] = ';'.join(refs)
            rows.append(row)
            continue
        zref = REF_MAP.get(str(b.refs).split(',')[0], str(b.refs).split(',')[0])
        host = load_host_json(b.Name)
        if host is not None:
            rz = host['redshift']
            if 'z_spec' in rz or 'z' in rz:
                row['Redshift'] = rz.get('z', rz.get('z_spec'))
                if 'z_spec' not in rz and 'z_phot_err' in rz:
                    row['Redshift_err'] = rz['z_phot_err']
                refs.append(f'z:hostJSON({zref})')
            phot = host.get('photom', {})
            band = best_mag(phot)
            if band is not None:
                row['Magnitude'] = phot[band]
                err = phot.get(band+'_err')
                row['Mag_flag'] = 1 if err == 999. else 0
                if err is not None and err != 999.:
                    row['Magnitude_err'] = err
                row['Band'] = band
                refs.append(f"mag:{band}({phot.get(band+'_ref', 'repo')})")
            der = host.get('derived', {})
            ms = repo_mstar(der)
            sfr_c = repo_sfrs(der)
        else:
            ms, sfr_c = None, []
            if np.isfinite(b.z):
                row['Redshift'] = b.z
                refs.append(f'z:base({zref})')
        # Literature supplements (repo wins; Q26)
        sup = LIT_SUPPLEMENT.get(name, {})
        if ms is None and 'logM' in sup:
            ms = (sup['logM'], sup.get('logM_err'), sup['Mstar_source'])
        if len(sfr_c) == 0:
            sfr_c = sup.get('SFR', [])
        if 'note' in sup:
            refs.append(sup['note'])
        fill_mstar_sfr(row, refs, ms, sfr_c)
        row['Refs'] = ';'.join(refs)
        rows.append(row)

    # ---- Literature-only FRBs
    for name, lit in LIT_HOSTS.items():
        refs = []
        row = dict(TNS_NAME=name, Repeater=True)
        if lit['z'] is not None:
            row['Redshift'] = lit['z']
            refs.append(f"z:{lit['z_ref']}")
        # Magnitude (Q39)
        if 'mag' in lit:
            m, e, band, ref = lit['mag']
            row.update(Magnitude=m, Magnitude_err=e, Mag_flag=0, Band=band)
            refs.append(f'mag:{band}({ref};MW-corrected)')
        elif lit['coord'] is not None:
            coord = SkyCoord(lit['coord'], unit=(units.hourangle, units.deg)) \
                if 'h' in lit['coord'] else SkyCoord(lit['coord'], unit='deg')
            cat = catalog_mag(name, coord, cache)
            if 'mag' in cat:
                row.update(Magnitude=cat['mag'], Magnitude_err=cat['err'],
                           Mag_flag=0, Band=cat['band'])
                refs.append(f"mag:{cat['band']}(catalog;G23,EBV={cat['EBV']:.3f})")
            elif 'mag_fallback' in lit:
                m, e, band, ref = lit['mag_fallback']
                # Correct with the closest filter curve in the repo
                filt = {'GTC_r': 'SDSS_r', 'GMOS_N_r': 'GMOS_r'}.get(band, band)
                corr = frbphotom.extinction_correction(filt, cat['EBV'])
                row.update(Magnitude=m - 2.5*np.log10(corr), Magnitude_err=e,
                           Mag_flag=0, Band=band)
                refs.append(f"mag:{band}({ref};G23[{filt}],EBV={cat['EBV']:.3f})")
        ms = None if lit['logM'] is None else (lit['logM'], lit['logM_err'],
                                                lit['Mstar_source'])
        fill_mstar_sfr(row, refs, ms, lit['SFR'])
        row['Refs'] = ';'.join(refs)
        rows.append(row)

    json.dump(cache, open(PHOTOM_CACHE, 'w'), indent=2)

    cols = ['TNS_NAME', 'Repeater', 'Redshift', 'Redshift_err', 'Magnitude',
            'Magnitude_err', 'Mag_flag', 'Band', 'Stellar_Mass', 'Mstar_err',
            'Mstar_source', 'SFR', 'SFR_err', 'SFR_flag', 'SFR_type', 'Refs']
    df = pandas.DataFrame(rows, columns=cols).sort_values('TNS_NAME')
    for col in ['Mag_flag', 'SFR_flag']:
        df[col] = df[col].astype('Int64')
    df.to_csv(OUTFILE, index=False, float_format='%.6g')
    print(f'Wrote {len(df)} rows to {os.path.abspath(OUTFILE)}')


def fill_mstar_sfr(row, refs, ms, sfr_c):
    if ms is not None:
        row['Stellar_Mass'] = ms[0]
        row['Mstar_err'] = ms[1]
        row['Mstar_source'] = ms[2]
        refs.append(f'Mstar:{ms[2]}')
    sfr = choose_sfr(sfr_c)
    if sfr is not None:
        row.update(SFR=sfr[0], SFR_err=sfr[1], SFR_flag=sfr[2], SFR_type=sfr[3])
        refs.append(f'SFR:{sfr[3]}({sfr[4]})')


if __name__ == '__main__':
    main()
