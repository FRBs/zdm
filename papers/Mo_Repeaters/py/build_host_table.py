""" Build CHIME_FRB_hosts.csv: host-galaxy properties of CHIME FRBs
for Clancy's repeater analysis.

See claude_prompts/frb_host_table.md for the decisions (Q1-Q40) behind
every choice made here.

Sources, in order of precedence:
  1. Host JSONs in the FRB repo (frb/data/Galaxies); the literature values
     are now in the repo (frb/data/Galaxies/Literature; add_frbs.md)
  2. FRBs_base.csv (redshift fall-back, repeater flag, P(O|x))
  3. public_hosts.csv P_Ox, where FRBs_base.csv has no P(O|x) (reported, not cut on)

Run in the `astro` env from papers/Mo_Repeaters:
    python py/build_host_table.py
"""
import os
import glob
import json

import numpy as np
import pandas

FRB_REPO = os.path.join(os.getenv('HOME'), 'Astronomy', 'FRB', 'FRB')
GAL_PATH = os.path.join(FRB_REPO, 'frb', 'data', 'Galaxies')
BASE_FILE = os.path.join(FRB_REPO, 'frb', 'data', 'FRBs', 'FRBs_base.csv')
HOSTS_FILE = os.path.join(GAL_PATH, 'public_hosts.csv')

this_path = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(this_path, '..', 'CHIME_FRB_hosts.csv')

# Q41: no P(O|x) cut is applied; the P_Ox column is reported instead
#  so the user can choose the cut

# Q12/Q21: CHIME-detected FRBs localized by another instrument (in the base table)
EXTRA_BASE = ['FRB20121102A', 'FRB20201124A']

# Reference labels
REF_MAP = {'Leung+2025': 'CHIME/FRB+2025(ApJS280,6)'}

# Mstar_ref values known to be Prospector fits (Q26)
PROSPECTOR_REFS = ['Gordon2023', 'Bhardwaj2021b', 'Michilli2023', 'Ibik2024a',
                   'Moroianu2025', 'Ravi2023', 'Bhardwaj2025', 'Eftekhari2024']
# Method suffixes on Mstar_ref (add_frbs.md Q9) -> (tier, method)
REF_SUFFIX = {'_CIGALE': ('Photometric', 'CIGALE'),
              '_NEDLVS': ('Other', 'NED-LVS'),
              '_SDSS': ('Other', 'SDSS')}

# Notes added to Refs (add_frbs.md Q11, prompt 10)
NOTES = {'FRB20191106C': 'Mstar:Leung2025b_Table1_typo_corrected(C.Leung,priv.comm.)'}

# Mag ranking (Q18): pointed > DECaL/DELVE/DES > Pan-STARRS > SDSS
POINTED = ['WFC3_F606W', 'LRISr_R', 'VLT_FORS2_R', 'GMOS_S_r', 'GMOS_N_r',
           'GTC_OSIRIS_r', 'HSC_r', 'DEIMOS_r', 'NOT_r']
MAG_RANK = POINTED + ['DECaL_r', 'DELVE_r', 'DES_r', 'Pan-STARRS_r', 'SDSS_r']


def load_host_json(name):
    """ Return the host JSON dict for an FRB, or None """
    files = glob.glob(os.path.join(GAL_PATH, name[3:], f'{name}_host.json'))
    if len(files) == 1:
        return json.load(open(files[0]))
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
    if lo in (-999., 999.) or up in (-999., 999.):
        return lo if not log else None   # flags (Q24/Q37) are handled by the caller
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
        suffix = [sfx for sfx in REF_SUFFIX if ref.endswith(sfx)]
        if len(suffix) == 1:
            tier, method = REF_SUFFIX[suffix[0]]
            ref = f'{method}:{ref[:-len(suffix[0])]}'
        elif ref in PROSPECTOR_REFS:
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
        elif stype == 'photom' and der.get(key+'_ref', '').endswith('_SDSS'):
            stype = 'SDSS'
        out.append((der[key], err, flag, stype, der.get(key+'_ref', 'repo')))
    return out


def choose_sfr(cands):
    """ Q37: first measured value in priority order; else first limit """
    for c in cands:
        if c[2] == 0:
            return c
    return cands[0] if len(cands) > 0 else None


def main():
    base = pandas.read_csv(BASE_FILE)
    sample = base[(base.telescope == 'CHIME') | base.Name.isin(EXTRA_BASE)]
    hosts = pandas.read_csv(HOSTS_FILE, dtype={'FRB': str})
    host_pox = dict(zip('FRB'+hosts.FRB, hosts.P_Ox))

    rows = []
    # ---- Base-table FRBs
    for _, b in sample.iterrows():
        name = b.Name
        refs = []
        row = dict(TNS_NAME=name, Repeater=bool(b.repeater))
        pox = b['P(O|x)']
        if not np.isfinite(pox):
            pox = host_pox.get(name, np.nan)
        row['P_Ox'] = pox
        zref = REF_MAP.get(str(b.refs).split(',')[0], str(b.refs).split(',')[0])
        host = load_host_json(b.Name)
        if host is not None:
            rz = host['redshift']
            zval = rz.get('z', rz.get('z_spec'))
            if zval is not None and np.isfinite(zval):
                row['Redshift'] = zval
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
        if name in NOTES:
            refs.append(NOTES[name])
        fill_mstar_sfr(row, refs, ms, sfr_c)
        row['Refs'] = ';'.join(refs)
        rows.append(row)

    cols = ['TNS_NAME', 'Repeater', 'P_Ox', 'Redshift', 'Redshift_err', 'Magnitude',
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
