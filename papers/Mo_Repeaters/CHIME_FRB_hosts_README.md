# CHIME FRB host table (DRAFT v2, 2026-10-05)

File: `CHIME_FRB_hosts.csv` (97 rows). Built by `py/build_host_table.py` from the FRB repo (`FRBs/FRB`, branch `repeater_updates`). The decisions behind it are logged in `claude_prompts/frb_host_table.md` (main repo) and `prompts/add_frbs.md` (FRB repo).

## Sample

- The 95 rows of `FRBs_base.csv` with `telescope == CHIME`. Here this means CHIME-discovered or CHIME-selected; the burst may have been localized by another instrument (EVN, DSA-110, MeerKAT, KKO).
- Plus FRB20121102A and FRB20201124A, which CHIME detected but another instrument localized.
- **16 repeaters**:
  - 14 have a host with a redshift, stellar mass and SFR.
  - FRB20190208A has a host with only a magnitude (r ≈ 27, no z).
  - FRB20181119A has no proposed host.
- **81 non-repeaters**: 18 have a host with z. Most of the rest are CHIME/KKO localizations without a host (blank).
- **No host-association cut is applied.** The `P_Ox` column gives the PATH P(O|x), so you can choose the cut.
  - Where it comes from: `FRBs_base.csv`, or else `public_hosts.csv`.
  - The priors differ between papers. Ibik+2024a values (FRB20190110C 0.779, FRB20200223B 0.899) use P(U) = 0.1, while KKO, Shah, Tian and Hewitt use their own P(U).
  - Hosts with z and P_Ox < 0.9: FRB20190110C, FRB20200223B, FRB20230311A (0.774) and FRB20231020A (0.571).
  - Blank P_Ox with a host: FRB20180814A (PATH not run) and FRB20181030A (chance-coincidence probability < 2.5×10⁻³ instead of PATH; Bhardwaj+2021b).
  - For CHIME/KKO FRBs, the P_Ox of the best candidate is given even when no host is listed.
- Repeating sources use their first-burst names. FRB20190303A appears in KKO as burst FRB20231204A; FRB20191106C appears as FRB20231128A.

## Columns

| Column | Meaning |
|---|---|
| `TNS_NAME` | e.g. `FRB20180916B` |
| `Repeater` | True/False (`FRBs_base.csv`) |
| `P_Ox` | PATH posterior P(O\|x) of the host (see Sample for priors) |
| `Redshift`, `Redshift_err` | Host redshift (all spectroscopic, so `Redshift_err` is blank) |
| `Magnitude`, `Magnitude_err` | Apparent host magnitude, AB, **corrected for Galactic extinction** |
| `Mag_flag` | 0 = detection, 1 = upper limit (none at present) |
| `Band` | Filter used. Preference: pointed imaging (GMOS, GTC, ...) > DECaL/DELVE > Pan-STARRS > SDSS, r-band. Exception: FRB20231020A has no r-band and uses Pan-STARRS_z |
| `Stellar_Mass`, `Mstar_err` | log10(M*/Msun); error symmetrized in dex |
| `Mstar_source` | `tier:method:ref`. Tier order: Prospector > Photometric (CIGALE) > Spectroscopic > Other (SDSS, NED-LVS, ...) |
| `SFR`, `SFR_err` | Msun/yr, linear. Error symmetrized |
| `SFR_flag` | 0 = measurement, 1 = upper limit, 2 = lower limit. A limit is used only when no measurement exists |
| `SFR_type` | `nebular` (Hα) > `SED` (Prospector) > `photom` / `UV+IR` / `SDSS` (in that priority) |
| `Refs` | Semicolon-separated `quantity:source` provenance |

Blank means not available.

## Caveats

- **Coverage is uneven.**
  - Repeaters: 14 of 16 have M* and SFR.
  - Non-repeaters: 8 of 18 hosts have M* (all from Leung+2025b CIGALE / NED-LVS fits), and **none** have an SFR.
  - So repeater vs non-repeater SFR comparisons are not yet possible from this table.
- **Mixed methods and IMFs** for M* and SFR (see `Mstar_source`, `SFR_type`):
  - Prospector masses are mostly Kroupa (some do not state the IMF).
  - The Leung+2025b values are Chabrier.
  - The offsets (~0.03 dex) are well below the errors.
- **Aperture-limited SFRs:**
  - Several Hα SFRs come from slit or fiber spectra and are lower limits in practice: FRB20191106C (SDSS fiber, 1.53) and FRB20190417A (1″ slit). Only FRB20220912A is formally flagged as a limit.
  - FRB20181030A uses the UV+IR total (0.36). Its Hα value (0.033) is slit-limited and is not used.
- **FRB20191106C mass:** 9.47 (Leung+2025b CIGALE) conflicts with 10.65 (Chang+2015) and with a colour-M/L estimate (≈10.4). We are checking with the Leung+2025b authors (`Refs` gives the alternative).
- **FRB20200120E:** the FRB sits in a globular cluster of M81. Its z, magnitude, M* and SFR are those of M81 (z = 0.0008; m_r = 7.6 from DECaL).
- **FRB20180814A:**
  - PATH was not run on it, so `P_Ox` is blank.
  - Its localization ellipse is under review with Dr. Michilli; the host is about 52″ from the FRB position.
- **FRB20190303A:** the host is one member of a merging pair. M* and SFR are SDSS-pipeline values (Michilli+2023), method unspecified.
- **Large, nearby galaxies:** survey catalog magnitudes may miss flux. FRB20231230A uses DELVE because its DECaL photometry is a shredded fragment.
