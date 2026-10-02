# FRB Host Table

## Goals

Generate a table of FRB host properties for Clancy's repeater analysis
He wants a simple CSV file with the following columns:
- TNS_NAME
- Redshift
- Redshift_err
- Magnitude
- Magnitude_err
- Band
- Stellar_Mass
- SFR

## Context

The FRB Repo located at `Astronomy/FRB/FRB` contains JSON files and tables of FRBs and their hosts.  In particular, the table `frb.data.FRBs.FRBs_base.csv` contains basic properties of each FRB, including whether it is a repeater.

Meanwhile, `frb.data.Galaxies` contains JSON filees for the primiary host of each FRB

Last, there are helper functions to pull this information from the Repo 

## Prompts

1. Examine the files described in the Context section and review our Goals.  Then ask me a series of questions to help you generate the table.  Put them in the Q&A section below.  
Use Opus 5.5. Log your work in the Logs section below.

2. I have answered your questions.  Read them and see if you have any additional ones.  If so, ask them in the Q&A section below.  
Use Opus 5.5. Log your work in the Logs section below.

## Missing Repeaters

These are localized repeaters with a host and/or redshift that are **not** flagged `repeater=TRUE` in `FRBs_base.csv`. Items marked (repo) come from what's in the FRB repo. Items marked (lit) are from my memory of the literature: **please verify the redshifts and references** before adding them. "Disc." is the discovery instrument and "Loc." the localization instrument.

| FRB | Disc. / Loc. | z | Ref(s) | Status in FRB repo |
|---|---|---|---|---|
| FRB20181119A | CHIME / CHIME (+EVN?) | 0.26064 (repo FRB JSON; unverified) | "Astroflash" in FRB JSON; Ibik+2024? | FRB JSON exists with `repeater: true`; not in `FRBs_base.csv`; no host JSON |
| FRB20190208A | CHIME / EVN | none (faint host, photo-z only?) | Hewitt+2024 (lit) | FRB JSON exists with `repeater: true`; not in base; no host JSON |
| FRB20180814A | CHIME / CHIME | ~0.068 (lit) | Michilli+2023; Ibik+2024 | absent |
| FRB20190110C | CHIME / CHIME | ~0.1224 (lit) | Ibik+2024 | absent |
| FRB20190303A | CHIME / CHIME | ~0.064 (lit) | Michilli+2023; Ibik+2024 | absent |
| FRB20190417A | CHIME / CHIME (+EVN) | ~0.1282 (lit) | Moroianu+2023; Ibik+2024 | absent |
| FRB20191106C | CHIME / CHIME | ~0.1078 (lit) | Ibik+2024 | absent |
| FRB20200223B | CHIME / CHIME (+EVN) | ~0.0602 (lit) | Ibik+2024; Hewitt+2024 | absent |
| FRB20220912A | CHIME / DSA-110 | ~0.0771 (lit) | Ravi+2023; Hewitt+2024 | absent |
| FRB20240114A | CHIME / MeerKAT, EVN | ~0.130 (lit) | Tian+2024; Bhardwaj+2024; Bruni+2024 | absent |
| FRB20240209A | CHIME / CHIME Outriggers | ~0.1384 (lit) | Shah+2025; Eftekhari+2025 | absent |
| FRB20220529A | FAST / VLA | ~0.184 (lit) | Li+2025 | absent (not CHIME-discovered) |

Also worth checking: the CHIME Outriggers / Leung+2025 sample, which already supplied FRB20231128A and FRB20231204A, may contain more repeaters that are in `FRBs_base.csv` with `repeater=FALSE`. The CHIME/FRB repeater catalogs (2019, 2023) can be cross-matched against the 86 CHIME rows to flag them.

## Q&A

**Q1. Sample: which FRBs go in the table?**
(a) Repeaters only (`repeater==TRUE` in `FRBs_base.csv`: 11 FRBs, 9 of which have host JSONs), (b) every FRB with a host JSON (~96, both repeaters and non-repeaters), or (c) every FRB in `FRBs_base.csv` (178), with blank host columns where there's no host? If the table includes non-repeaters, should I add a `Repeater` column (and `Telescope`)?

A:  We want it for all CHIME FRBs, given by the `telescope` column in `FRBs_base.csv` 

**Q2. Repeaters missing from `FRBs_base.csv`.**
Several well-known localized repeaters aren't in the base table at all, so they have `repeater` = FALSE/absent. Examples are FRB20220912A (DSA-110), FRB20240209A (CHIME, elliptical host) and FRB20240114A, and I didn't find host JSONs for them either. Should I (a) add them by hand from the literature, (b) leave them out, or (c) send you a list so you can update the FRB repo first? Is there a source list of repeaters that Clancy is using (e.g. CHIME repeater catalog) that I should check against?

A:  Generate a list for me that I can double check before updating the FRB repo.  Put it in the Missing Repeaters section above.

**Q3. FRB20181030A (repeater).**
Its host file `frb/data/Galaxies/FRB20181030A_host.json` doesn't parse: there's an illegal trailing comma at line 64. It's also at the top level of `Galaxies/` rather than in a `20181030A/` subfolder, so `grab_host()` will probably skip it. Shall I fix the JSON in the FRB repo, or work around it for this table only?

A:  Fix the JSON in the FRB repo.  I have moved that Repo onto a new `repeater_updates` branch.

**Q4. FRB20230814A (repeater, DSA).**
It has no redshift and no host. Include it with blanks or drop it?

A: Drop it

**Q5. Redshift source and error.**
The base-table `z` and the host JSON `z` disagree slightly for some FRBs (e.g. 20180301A: 0.33044 vs 0.3305; 20190711A: 0.52172 vs 0.522; 20201124A: 0.0982 vs 0.0979). Which should take priority? Almost no hosts carry a redshift error (only 1 of 95 has `z_err`; the rest are spectroscopic). For spec-z, should `Redshift_err` be blank, 0, or a nominal value (e.g. 0.001)? For photo-z hosts, should I use `z_phot_err`?

A: The JSON files are the source of truth; use those  

**Q6. Magnitude: which band and what kind?**
Photometry is heterogeneous: Pan-STARRS_r (64 hosts), DECaL_r (60), DELVE_r, SDSS_r, VLT_FORS2_R, GMOS_r, SOAR_r, etc.
- Apparent or absolute magnitude? `derived` has `M_r` for some hosts.
- Should I use a fixed preference order for r-band (e.g. deepest/pointed imaging first such as GMOS/FORS2/HSC, then DECaL/DELVE, then Pan-STARRS, then SDSS), with `Band` recording the filter actually used? Or force a single survey (e.g. Pan-STARRS_r) even if that means more gaps?
- Should magnitudes be corrected for Galactic extinction (`EBV` is in `photom`)?
- AB throughout? (I assume yes.)

A: It is ok that the bands vary.  Use apparent magnitudes.  Yes, use deepest first, then DECaL/DELVE, then Pan-STARRS, then SDSS.  Yes, correct for Galactic extinction.  Yes, use AB magnitudes.

**Q7. Stellar mass.**
Only 27 hosts have `Mstar` (mostly from Gordon2023 Prospector). Others have `Mstar_spec` (9) or the older `SMStar` (Mannings2021). Should I fill gaps from those in a priority order, or use `Mstar` only? Do you want log10(M*/Msun) or linear? Should I include errors (`Mstar_loerr`/`Mstar_uperr`), as two asymmetric columns or one symmetrized one?

A:  Yes, fill in gaps in this priority order:
- Gordon2023 Prospector
- Mannings2021
- Spectroscopic
- Photometric
- Other
- No mass

And specify what was used as a separate column (Mstar_source).  
Report log10(M*/Msun) and a symmetrized error column (Mstar_err).

**Q8. SFR.**
The choices are `SFR_SED` (52 hosts), `SFR_nebular` (17, from Hα) and `SFR_photom` (14). Which takes priority? Should I add an `SFR_type` column to record the source, and SFR error columns? Units are Msun/yr; linear or log?

A: Use SFR_nebular, then SFR_SED then SFR_photom.  And do add a SFR_type.  Linear with units of Msun/yr.

**Q9. Missing values and upper limits.**
For missing entries, do you want blank, NaN, or a sentinel such as -999? Some SFRs are effectively upper limits (e.g. FRB20180916B, FRB20200120E). Should I flag them?

A:  Use blank in the CSV file

**Q10. Names, output location, reproducibility.**
- TNS_NAME format: `FRB20121102A` or `20121102A`?
- Where should the CSV go, e.g. `papers/Mo_Repeaters/Data/frb_hosts.csv`?
- Do you also want the generating script saved, e.g. `papers/Mo_Repeaters/py/build_host_table.py` using `frb.galaxies.utils.build_table_of_hosts()` and `frb.frb.build_table_of_frbs()`, so the table can be regenerated?
- Should I add a `Refs` column for the provenance of each quantity?

A: Use FRB20121102A format.  Put the CSV file in the top level of the `papers/Mo_Repeaters/` directory.  Yes, add a `Refs` column for the provenance of each quantity.

### Round 2 questions (Prompt #2)

**Q11. Which CHIME rows?**
There are 86 rows with `telescope==CHIME`. 23 have a host JSON (including 20181030A). 12 more have a redshift in `FRBs_base.csv` (all Leung+2025) but no host JSON. The remaining 51 have neither a redshift nor a host. Should the CSV include (a) all 86, with blanks where needed, (b) only those with a redshift (34), or (c) only those with a host JSON (23)? For the 12 with a base-table z but no host JSON, should `Redshift` fall back to the base-table value? You said the JSON is the source of truth, so I want to check that a fallback is OK.

A:

**Q12. How "CHIME" is defined.**
`telescope` is the localizing instrument. CHIME-discovered repeaters localized elsewhere aren't counted as CHIME: FRB20201124A is listed as MeerKAT, and FRB20220912A and FRB20240114A would be DSA and MeerKAT/EVN. Should I keep `telescope==CHIME` strictly, or also include CHIME-discovered FRBs localized by other instruments?

A:

**Q13. Repeater column.**
Should I add a `Repeater` (True/False) column? It seems essential for Clancy's analysis, but it isn't in his column list.

A:

**Q14. Missing repeaters: scope of the repo update.**
Once you've checked the list above, should I (a) add them to `FRBs_base.csv` only (name, coords, DM, z, `repeater=TRUE`), so they appear with z but blank host columns, or (b) also build host JSONs with photometry (Legacy Survey / Pan-STARRS) and literature Mstar/SFR? Option (b) is a much bigger job.

A:

**Q15. FRB20181030A host JSON.**
I fixed the trailing comma, so the file now parses. Loading it through `FRBHost.by_frb()` needs a few more changes:
- Move it to `frb/data/Galaxies/20181030A/FRB20181030A_host.json`, since `by_frb` looks in a per-FRB subfolder.
- Rename `WISE_w1..w4` to `WISE_W1..W4`.
- Handle filters not in `defs.valid_filters`: `DESI_g/r/z`, `2MASS_J/H/Ks` and `Herschel_PACS_*`. Should DESI become `DECaL_*` (Legacy Survey naming), should I add 2MASS/Herschel to `defs.py`, or drop them?
- `derived.Z_spec = 0.0039` equals the redshift and looks like a copy error, since `Z_spec` is metallicity. Should I delete it?

Can I go ahead with these?

A:

**Q16. Stellar mass tiers.**
Mannings2021's `SMStar` is a stellar-mass *surface density at the FRB position* (see `defs.py`), not a total mass, so it can't fill the Mannings2021 tier. Should I drop that tier? Here's how I'd map the rest:
- Gordon2023 = `Mstar` with `Mstar_ref==Gordon2023`
- Spectroscopic = `Mstar_spec` (pPXF)
- Photometric = `Mstar` from CIGALE (other refs)
- Other = remaining literature values

For CHIME hosts specifically, only 2 of the 22 parseable host JSONs have any `Mstar` (plus 20181030A's `Mstar_spec`), so the column will be mostly blank. Is that acceptable for now?

A:

**Q17. Errors not yet specified.**
- `Redshift_err`: should it be blank for spec-z, or a nominal value? For photo-z, I'd use `z_phot_err`.
- `Mstar_err`: I'd symmetrize in log space as (dex_lo + dex_up)/2. Is that OK? Many `Mstar_spec` / CIGALE values have no error; should those be left blank?
- SFR: do you want an `SFR_err` column (symmetrized, or blank when absent)?

A:

**Q18. Magnitude details.**
- What's the ranking among pointed instruments? I'd propose: HST > Keck/LRIS, VLT/FORS2, Gemini/GMOS, HSC > NOT, SOAR, LDT and others > DECaL/DELVE/DES > Pan-STARRS > SDSS.
- Should I stay strictly with r-band (r, R, F606W?) through that ranking, and only fall back to another band when no r-band exists at all? FRB20231020A (CHIME) has only Pan-STARRS z/y and WISE.
- For extinction, I'd use `frb.galaxies.photom.extinction_correction()` (Gordon 2023 law, R_V = 3.1) with the host's `photom.EBV`, assuming the stored magnitudes are *not* already corrected. Is that right?
- Upper limits in photometry (where present): include them as blank, or as a value with a flag?

A:

**Q19. Refs format.**
I'd use one `Refs` column of semicolon-separated `quantity:ref` pairs (e.g. `z:Leung+2025;mag:DECaL_r;Mstar:Gordon2023;SFR:Gordon2023`), alongside `Band`, `Mstar_source` and `SFR_type`. Or would you prefer separate `z_ref`, `mag_ref`, `Mstar_ref` and `SFR_ref` columns?

A:

**Q20. Output names and script.**
- I'd name the CSV `papers/Mo_Repeaters/CHIME_FRB_hosts.csv`. OK?
- I'd save the generating script as `papers/Mo_Repeaters/py/build_host_table.py`, run in the `astro` conda env (it has `frb`, pandas and dust_extinction). OK?

A:

## Logs

### 2026-10-02 (Prompt #1: surveyed FRB repo, drafted questions)

Looked through `~/Astronomy/FRB/FRB/frb/data/FRBs/FRBs_base.csv`, the host JSONs in `frb/data/Galaxies/*/`, and the helper functions `frb.galaxies.utils.list_of_hosts()` / `build_table_of_hosts()` (these slurp `derived`, `photom`, `redshift`, etc. into a pandas table and add P_Ox from `public_hosts.csv`) and `frb.frb.build_table_of_frbs()`.

Findings:
- `FRBs_base.csv` has 178 FRBs (CHIME 86, DSA 47, ASKAP 40, MeerKAT 3, VLA 3), and 11 are flagged `repeater=TRUE`. 95 host JSONs parse.
- Repeaters with hosts: 20121102A, 20180301A, 20180916B, 20190520B, 20190711A, 20200120E, 20201124A, 20231128A, 20231204A. The last two have no Mstar/SFR yet. 20181030A's host JSON is malformed (trailing comma, line 64) and sits outside a subfolder. 20230814A has no z or host.
- Known repeaters 20220912A, 20240209A and 20240114A are absent from the base table and host data.
- Host coverage: r-band photometry for ~90/95 hosts, from mixed surveys. `Mstar` for 27, `SFR_SED` for 52, `SFR_nebular` for 17. Redshift errors are essentially absent (spec-z).
- Base-table and host-JSON redshifts differ at the 1e-3 level for a few FRBs.
- The system `python` (miniforge3, py3.13) has no pandas, so I used the json/csv stdlib. The table build will need an environment with `frb` + pandas installed.

Wrote the questions above (Q1–Q10). No files were changed outside this prompt file.

### 2026-10-02 (Prompt #2: read answers, drafted round-2 questions and missing-repeater list)

Read the answers to Q1–Q10 and re-examined the FRB repo (now on branch `repeater_updates`) with the new CHIME-only scope.

Findings:
- CHIME scope: 86 rows. 23 have host JSONs, 34 have a z (12 of those are Leung+2025 with no host JSON), and 51 have neither. Among the 22 parseable CHIME hosts, only 2 have `Mstar` and only 2 have SFR. r-band photometry is Pan-STARRS_r (21), DECaL_r (13), SDSS_r (10) and DELVE_r (3). FRB20231020A has no r-band.
- CHIME repeaters in the base table: 20180916B, 20181030A, 20200120E, 20231128A, 20231204A.
- `SMStar` (Mannings2021) is a surface density at the FRB position, not a total stellar mass, so it can't be a mass tier.
- `Mstar` is CIGALE unless the ref is Gordon2023 (Prospector), and `Mstar_spec` is pPXF (`defs.py`).
- The FRB-level JSONs for FRB20181119A and FRB20190208A have `repeater: true`, but neither is in `FRBs_base.csv`.
- Galactic extinction is available via `frb.galaxies.photom.extinction_correction()` (G23, R_V=3.1).
- The `astro` conda env (`~/miniforge3/envs/astro`) has `frb`, pandas and dust_extinction, so I'll use it for the build.

FRB repo edit: removed the trailing comma in `frb/data/Galaxies/FRB20181030A_host.json` (line 64), per the Q3 answer. The file now parses as JSON. Moving it to a subfolder, fixing the filter names and removing the suspect `Z_spec` are left for Q15.

I filled in the Missing Repeaters section. Literature redshifts are from memory and flagged for verification. I added questions Q11–Q20 to the Q&A section.
