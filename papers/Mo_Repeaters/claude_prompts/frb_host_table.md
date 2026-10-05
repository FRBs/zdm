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

3. I have answered your second round of questions.  Read them and see if you have any additional ones.  If so, ask them in the Q&A section below.  
Use Opus 5.5. Log your work in the Logs section below.

4. I have answered your third round of questions.  Read them and see if you have any additional ones.  If so, ask them in the Q&A section below.  
Use Opus 5.5. Log your work in the Logs section below.

5. I have answered your fourth round of questions.  Read them and see if you have any additional ones.  If so, ask them in the Q&A section below.  
Use Opus 5.5. Log your work in the Logs section below.

6. I have answered your fifth round of questions.  Read them and move on to the other tasks.
Use Opus 5.5. Log your work in the Logs section below.

7. I have worked through all of the prompts in `add_frbs.md`.  There are a few lingering TODOs, but I think you can proceed to build a draft table for Clancy.
If you have any additional questions, ask them in the Q&A section below.
Use Opus 5.5. Log your work in the Logs section below.

## Missing Repeaters

These are localized CHIME-discovered repeaters that are **not** flagged `repeater=TRUE` in `FRBs_base.csv`. As of Prompt #5, every value below was **verified against the paper PDF** in `papers/` (table, section or page given). Logs are base 10. "Ext?" says whether the paper states that the quoted magnitude is corrected for Galactic extinction.

| FRB | Disc. / Loc. | z | m_r (AB) | log M* (method, IMF) | SFR (Msun/yr, method) | Source(s) | Repo status |
|---|---|---|---|---|---|---|---|
| FRB20180814A | CHIME / CHIME baseband | 0.06835(1) spec (M23 Tab 2/3) | 17.15 PS1 Kron, no err (Ext? not stated) | 10.78 +0.12/−0.18 (Prospector; IMF n/s) | < 0.32 (log < −0.5, 95% UL; Prospector) | Michilli+2023 (arXiv:2212.11941) | absent |
| FRB20190110C | CHIME / CHIME baseband | 0.12244(6) spec (I23 Tab 3) | 18.009 ± 0.006 DESI (Ext? n/s) | 10.40 +0.02/−0.03 [2.5 +0.10/−0.17 e10] (Prospector; Chabrier) | Hα 0.1575 (slit, no internal corr.); Prospector 0.54 ± 0.04 | Ibik+2024a (arXiv:2304.02638) | absent. PATH 0.918/0.779 (P(U) = 0/0.1) |
| FRB20200223B | CHIME / CHIME baseband | 0.06024(2) spec (I23 Tab 3) | 16.080 ± 0.001 DESI (Ext? n/s) | 10.75 [5.6 +1.14/−0.93 e10] (Prospector; Chabrier) | Hα N/A (AGN); Prospector 0.59 ± 0.04 | Ibik+2024a | absent |
| FRB20190417A | CHIME / EVN | 0.12817(2) spec (Ib24b §3.2.2) | 22.42 ± 0.06 GMOS (Ext? n/s); Ib24b gives 21.47 | 7.88 +0.12/−0.14 (Prospector; IMF n/s) | Hα 0.19 ± 0.01 (slit, no internal corr.) | Moroianu+2025 (arXiv:2509.05174); Ibik+2024b (arXiv:2409.11533) | absent. (The "Moroianu+2023" I cited earlier does not exist; the host paper is Moroianu+2025) |
| FRB20220912A | CHIME / DSA-110 (+EVN) | 0.0771 ± 0.0001 spec (Ravi Tab 2) | 19.65 PS1 catalog, no err (Ext? n/s) | 10.0 ± 0.1 (Prospector; IMF n/s) | Hα ≳ 0.1 (**lower limit**, central aperture) | Ravi+2023 (arXiv:2211.09049); Hewitt+2023 (arXiv:2312.14490) | absent |
| FRB20240114A | CHIME / MeerKAT, EVN | 0.130287 ± 0.000002 spec (Bh25 Tab 3) | 21.94 ± 0.11 (Bh25 Tab 8: "extinction corrected", SDSS DR12; Tab 3 footnote says DESI) | 8.55 +0.12/−0.14 (Prospector; Kroupa). Chen+25: 8.6 ± 0.2 (M/L; Salpeter) | Hα 0.061 +0.004/−0.003 (MW + Balmer corr.). Chen+25 0.06 ± 0.01; Bruni 0.36 (secondhand; Bh25 calls it anomalous) | Bhardwaj+2025 (arXiv:2506.11915); Chen+2025 (arXiv:2502.05587); Tian+2024 (arXiv:2408.10988) | absent. Host is a dwarf satellite |
| FRB20240209A | CHIME / CHIME+KKO | 0.1384 ± 0.0004 spec (Eft Tab 2) | 16.79 ± 0.02 GMOS, 14″ aperture (Ext: **yes**, Eft Tab 1 note) | 11.34 ± 0.01 (Prospector; Kroupa) [text says 11.36] | < 0.36 (**upper limit**; Prospector 0–100 Myr) | Eftekhari+2024 (arXiv:2410.23336); Shah+2024 (arXiv:2410.23374) | absent. P(O\|x) = 0.99 |
| FRB20190208A | CHIME / EVN | **none** (too faint; z_max ≈ 0.83) | 27.32 ± 0.16 GTC (text/abstract) vs 27.17 (Tab 2) (Ext? n/s) | — | — | Hewitt+2024 (arXiv:2410.17044) | FRB JSON has `repeater: true`; not in base. P(O\|x) = 0.9995 |
| FRB20181119A | CHIME / CHIME baseband | **none**: no host proposed in any paper found (Ib24b App. A.4) | — | — | — | Michilli+2023 | FRB JSON has `repeater: true`; not in base. The repo FRB JSON's z = 0.26064 has **no literature support** |

**Already in the repo under later burst names (Q32, to be renamed):**

| Source | Repo name | z | log M* | SFR | Notes |
|---|---|---|---|---|---|
| FRB20190303A | FRB20231204A | 0.0644 (KKO Tab 2); M23: 0.06437(1) | 10.75(3) (SDSS-collab. value; method n/s) | log SFR 0.84(4) → 6.9 (SDSS-collab.; method n/s) | Host is the SW member of a merger (KKO §6.1). M23 Tables 3 and 4 swap the two members' z, so M23's r mag is ambiguous |
| FRB20191106C | FRB20231128A | 0.10775(1) (I23, SDSS); KKO 0.1079 | I23 (from Chang+2015): 10.65 [4.5 ± 1.2 e10]; **Leung+25 CIGALE: 9.47** | Hα 1.53 (SDSS fiber); Chang+15 4.75 +1.29/−1.27 | I23 calls the association "marginal"; KKO confirms it (P(O\|x) = 0.985). See Q38 |

**Dropped:** FRB20220529A (FAST-discovered; out of scope per Q21).

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

A: Use all 86, even if there is no redshift or json. It is better to be inclusive to highlight missing information with blanks. And yes, please fall back to the base table value if no .json is available.

**Q12. How "CHIME" is defined.**
`telescope` is the localizing instrument. CHIME-discovered repeaters localized elsewhere aren't counted as CHIME: FRB20201124A is listed as MeerKAT, and FRB20220912A and FRB20240114A would be DSA and MeerKAT/EVN. Should I keep `telescope==CHIME` strictly, or also include CHIME-discovered FRBs localized by other instruments?

A: Please also include CHIME-discovered FRBs localized by other instruments. Please also include CHIME-detected FRBs originally discovered by other instruments, e.g. FRB20121102A.

**Q13. Repeater column.**
Should I add a `Repeater` (True/False) column? It seems essential for Clancy's analysis, but it isn't in his column list.

A: Yes, please add this column

**Q14. Missing repeaters: scope of the repo update.**
Once you've checked the list above, should I (a) add them to `FRBs_base.csv` only (name, coords, DM, z, `repeater=TRUE`), so they appear with z but blank host columns, or (b) also build host JSONs with photometry (Legacy Survey / Pan-STARRS) and literature Mstar/SFR? Option (b) is a much bigger job.

A: The critical thing to do is to add literature Mstar/SFR to the .csv for Clancy. That is, please ensure that all pertinent information is pulled from the literature, and ensure you report the references used for each. If necessary, add references for this as a separate column of the .csv.  I also wish to build the JSON files.  To that end, generate a new prompt doc named `add_frbs.md` in `Astronomy/FRB/FRB/prompts` with a series of prompt docs to do so.  I expect you will need me to download a paper or two to do so, e.g. Leung+2025.  If so, add that to the prompt doc.

**Q15. FRB20181030A host JSON.**
I fixed the trailing comma, so the file now parses. Loading it through `FRBHost.by_frb()` needs a few more changes:
- Move it to `frb/data/Galaxies/20181030A/FRB20181030A_host.json`, since `by_frb` looks in a per-FRB subfolder.
- Rename `WISE_w1..w4` to `WISE_W1..W4`.
- Handle filters not in `defs.valid_filters`: `DESI_g/r/z`, `2MASS_J/H/Ks` and `Herschel_PACS_*`. Should DESI become `DECaL_*` (Legacy Survey naming), should I add 2MASS/Herschel to `defs.py`, or drop them?
- `derived.Z_spec = 0.0039` equals the redshift and looks like a copy error, since `Z_spec` is metallicity. Should I delete it?

Can I go ahead with these?

A: Yes, go ahead with these.

**Q16. Stellar mass tiers.**
Mannings2021's `SMStar` is a stellar-mass *surface density at the FRB position* (see `defs.py`), not a total mass, so it can't fill the Mannings2021 tier. Should I drop that tier? Here's how I'd map the rest:
- Gordon2023 = `Mstar` with `Mstar_ref==Gordon2023`
- Spectroscopic = `Mstar_spec` (pPXF)
- Photometric = `Mstar` from CIGALE (other refs)
- Other = remaining literature values

For CHIME hosts specifically, only 2 of the 22 parseable host JSONs have any `Mstar` (plus 20181030A's `Mstar_spec`), so the column will be mostly blank. Is that acceptable for now?

A: Please drop the Mannings tier. A mostly blank CHIME hosts column is acceptable.
But let's try to fill these in from the Leung papers, as described above.

**Q17. Errors not yet specified.**
- `Redshift_err`: should it be blank for spec-z, or a nominal value? For photo-z, I'd use `z_phot_err`.
- `Mstar_err`: I'd symmetrize in log space as (dex_lo + dex_up)/2. Is that OK? Many `Mstar_spec` / CIGALE values have no error; should those be left blank?
- SFR: do you want an `SFR_err` column (symmetrized, or blank when absent)?

A: Blank for spec-z error. For Mstar: yes, please symmeterize in log-space. Missing errors should be blank. An SFR_err column as described would be great, thanks.

**Q18. Magnitude details.**
- What's the ranking among pointed instruments? I'd propose: HST > Keck/LRIS, VLT/FORS2, Gemini/GMOS, HSC > NOT, SOAR, LDT and others > DECaL/DELVE/DES > Pan-STARRS > SDSS.
- Should I stay strictly with r-band (r, R, F606W?) through that ranking, and only fall back to another band when no r-band exists at all? FRB20231020A (CHIME) has only Pan-STARRS z/y and WISE.
- For extinction, I'd use `frb.galaxies.photom.extinction_correction()` (Gordon 2023 law, R_V = 3.1) with the host's `photom.EBV`, assuming the stored magnitudes are *not* already corrected. Is that right?
- Upper limits in photometry (where present): include them as blank, or as a value with a flag?

A: (i) that ranking is good. (ii) Yes, stay with r-band until you can't. (iii) I am rather confident that the JSON files have extinction corrected magnitudes.  See the `build_hosts.py` script in the FRB repo to double-check (iv) Upper limits should be included as a value with a flag

**Q19. Refs format.**
I'd use one `Refs` column of semicolon-separated `quantity:ref` pairs (e.g. `z:Leung+2025;mag:DECaL_r;Mstar:Gordon2023;SFR:Gordon2023`), alongside `Band`, `Mstar_source` and `SFR_type`. Or would you prefer separate `z_ref`, `mag_ref`, `Mstar_ref` and `SFR_ref` columns?

A:  That is fine

**Q20. Output names and script.**
- I'd name the CSV `papers/Mo_Repeaters/CHIME_FRB_hosts.csv`. OK?
- I'd save the generating script as `papers/Mo_Repeaters/py/build_host_table.py`, run in the `astro` conda env (it has `frb`, pandas and dust_extinction). OK?

A: Ok and Ok

### Round 3 questions (Prompt #3)

**Q21. The extra non-`telescope==CHIME` FRBs (from Q12).**
`FRBs_base.csv` has no discovery-instrument column, so this list comes from my knowledge of the literature. Please confirm or edit it:
- In the base table: FRB20121102A (VLA; CHIME-detected, Josephy+2019) and FRB20201124A (MeerKAT; CHIME-discovered). I don't know of CHIME detections of 20180301A, 20190520B or 20190711A, so I'd leave those out.
- From the Missing Repeaters list: everything except FRB20220529A, which is FAST-discovered and as far as I know has no CHIME detection.
- One-off FRBs localized by DSA-110, ASKAP or MeerKAT that CHIME also happened to detect: finding these properly needs a cross-match against CHIME/FRB Catalog 1/2 (or the CHIME VOEvent stream). Do you want that cross-match done (and if so, can you point me to a catalog file), or should I limit this to repeaters?

A:  We only care about the CHIME hosts for now.  I think these are the only 2 papers to consider:
    - https://ui.adsabs.harvard.edu/abs/2025ApJ...991L..25L/abstract
    - https://ui.adsabs.harvard.edu/abs/2025ApJS..280....6C/abstract
Unless there are a few one-offs from the VLBI efforts

**Q22. Instrument columns.**
Now that the sample goes beyond `telescope==CHIME`, should I add `Loc_Telescope` (the localizing instrument, from the base table) and a `CHIME_role` column (`discovered` / `detected` / `localized`) so Clancy can make cuts? Or is `Repeater` enough?

A:  I think `Repeater` is enough.

**Q23. Extinction for hand-built JSONs.**
I checked `frb/builds/build_hosts.py`: hosts built through it are dust-corrected with `photom.correct_photom_table()`, using `EBV` from `nebular.get_ebv()`. So I will **not** correct those again. But some JSONs were hand-entered rather than built, e.g. FRB20181030A (photometry copied from Bhardwaj+2021b), and probably any host I make from the literature for the missing repeaters.
- Should I treat hand-entered photometry as uncorrected and apply the G23 correction in the table script, or assume it's already corrected?
- FRB20181030A's JSON has `EBV = 0.42`. That looks high for a field at b ≈ +40°, and might be an A_V or a host value. Can you check it against Bhardwaj+2021b, or should I query IRSA (`nebular.get_ebv`) and replace it?

A: I have put `Bhardwaj_2021_ApJL_919_L24.pdf` in the `papers/` folder.  Read it and react accordingly.

**Q24. Upper-limit flags.**
The repo encodes magnitude upper limits as `<band>_err = 999` (`defs.py`, line 30). None of the CHIME hosts' r-band entries are limits at the moment, but I'll handle the case anyway. Proposal:
- Add `Mag_flag`: `0` = detection, `1` = upper limit (`Magnitude` holds the limit and `Magnitude_err` is blank).
- Do the same for SFR (`SFR_flag`), since some literature SFRs (e.g. FRB20180916B, FRB20200120E) are limits. Q9 said "blank", but Q18 said limits should carry a value plus a flag. Should SFR follow Q18?

A: Yes, do your proposal.

**Q25. FRB20181030A JSON edits (Q15): blocked on permissions.**
My attempt to move and rewrite the JSON in the FRB repo was denied by the Claude Code auto-mode permission check, so **none of the Q15 changes have been made yet**. The file still sits at `frb/data/Galaxies/FRB20181030A_host.json`, with only the trailing comma fixed. Would you rather (a) make the edits yourself, or (b) approve the action so I can do it next round? The planned edits are:
- Move it to `frb/data/Galaxies/20181030A/FRB20181030A_host.json`.
- `DESI_g/r/z` → `DECaL_g/r/z`. The original file also has a typo: `DESI_r`'s ref key is a duplicate `DESI_g_ref`.
- `2MASS_J/H/Ks` → `2MASS_j/h/k`, since `defs.py` already defines lowercase 2MASS bands, so no `defs.py` change is needed.
- `WISE_w1..w4` → `WISE_W1..W4`.
- Delete `derived.Z_spec`.
- Herschel: the entries labelled `Herschel_PACS_250/350/500` are really **SPIRE** bands (PACS is 70/100/160 µm). Should I drop all five Herschel entries (they'd remain in git history and Bhardwaj+2021b), or add `Herschel_PACS_100/160` and `Herschel_SPIRE_250/350/500` to `defs.valid_filters`? Adding them may need filter curves if anything downstream (e.g. CIGALE or the extinction code) uses them.

A: (b) I approve the action

**Q26. Stellar-mass tier for non-Gordon Prospector fits.**
Literature masses for CHIME hosts (Ibik+2024, Leung+2025, Bhardwaj+2024 and similar) are often Prospector fits too. Should "Prospector from any paper" share the top tier, labelled e.g. `Mstar_source = Prospector:Ibik+2024`? Or keep the top tier for Gordon2023 only and put the others under "Other"? A related question: if a host has both a repo value (e.g. CIGALE) and a literature Prospector value, which should win?

A:  Yes, Prospector from any paper is the top tier.  Use the Repo value, but add to the `add_frbs.md` propmt doc the action to update.

**Q27. Papers I need.**
To fill in Mstar, SFR and z (and later the JSONs via `add_frbs.md`), I expect to need the PDFs or tables from:
- Leung+2025 (CHIME Outriggers localizations and hosts)
- Ibik+2024 (CHIME repeater hosts)
- Bhardwaj+2024 (CHIME/KKO hosts)
- Shah+2025 and Eftekhari+2025 (FRB20240209A)
- Ravi+2023 and Hewitt+2024 (FRB20220912A)
- Bruni+2024 and Bhardwaj+2024 (FRB20240114A)
- Michilli+2023 and Moroianu+2023 (FRB20180814A/20190303A and FRB20190417A)
Where should I look for them, e.g. `papers/Mo_Repeaters/Literature/` or `Astronomy/FRB/FRB/frb/data/Galaxies/Literature/`? Do you want me to fetch the arXiv versions myself with web access, or will you download them? Machine-readable tables (CDS/journal) would be best where they exist.

A:  Access them yourself via `arxiv` and put in the `papers/` folder.

**Q28. Verifying the Missing Repeaters list.**
The literature redshifts in that list came from memory. Should I verify them (and the references) against the papers in Q27 before you update the FRB repo, or will you check them yourself?

A: Yes, verify against the papers.  Do not use your memory

**Q29. Order of work.**
I suggest this order: (1) write `add_frbs.md` in the FRB repo, (2) build `CHIME_FRB_hosts.csv` now from the repo plus literature values, with literature-only entries tagged in `Refs`, and (3) regenerate the CSV once the new JSONs and base-table rows exist. So the CSV may temporarily contain values that aren't in the FRB repo yet. OK?

A:  That is a fine order

### Round 4 questions (Prompt #4)

**Q30. FRB20181030A: the SFR is mislabelled.**
I checked the JSON against Bhardwaj+2021b (`papers/Bhardwaj_2021_ApJL_919_L24.pdf`), and three things are wrong:
- `photom.EBV = 0.42` is the *host's* Balmer-decrement E(B−V) (Table 6), not the Galactic value. The paper's photometry (Table 10) is already corrected for Galactic extinction (Schlafly & Finkbeiner 2011), so we should not correct it again. The fix script swaps in the IRSA Galactic value, as `build_hosts.py` does.
- `Mstar_spec = 5.8e9` is a **Prospector** fit (+1.6/−2.0 × 10⁹), not pPXF. The script moves it to `Mstar` with `Mstar_loerr`/`Mstar_uperr` and `Mstar_ref = Bhardwaj2021b`. It now falls in the Prospector tier (Q26).
- `SFR_nebular = 0.35` is really SFR_total = SFR(NUV) + 0.6·SFR(TIR) = 0.36 Msun/yr (log SFR = −0.45 ± 0.1). The true Hα SFR is 0.033 Msun/yr, but it comes from a slit covering only part of the galaxy, so the paper calls it a significant underestimate.

How should the SFR be stored? Options: (a) add a new key such as `SFR_UVIR` ("UV+IR") to `defs.valid_derived` and store 0.36 ± 0.08 there. In the CSV, `SFR_type` would then read `UV+IR`. (b) Store 0.36 under `SFR_photom`. (c) Store `SFR_nebular = 0.033` as a lower limit (`SFR_flag`). I recommend (a). I've left `SFR_nebular` untouched until you decide.

>A. Store 0.36 under SFR_photom and set SFR_nebular=0.0333 as a lower limit

**Q31. FRB20181030A Herschel entries.**
Your answer to Q25 approved the edits but didn't pick between dropping the Herschel bands and adding them to `defs.py`. The fix script **drops** all five, since nothing in the CSV needs them. Is that OK? Note that the five entries are PACS 100/160 and SPIRE 250/350/500.

>A. Yes, that is OK

**Q32. Two CHIME repeaters are listed under their later bursts.**
The KKO catalog (CHIME/FRB+2025, ApJS 280, 6) says FRB20231204A is a repeat burst of **FRB20190303A** (§6.1), and FRB20231128A is a repeat burst of **FRB20191106C** (§6.6). The repo stores both (base row and host JSON) under the 2023 names.
- In the CSV, should `TNS_NAME` be the source name (the first burst: FRB20190303A, FRB20191106C), or keep the repo's 2023 names? I recommend the source name, plus a note in `Refs`. Otherwise the two sources could end up in Clancy's table twice once the "missing" rows are added.
- For `add_frbs.md`: rename the repo entries (base row + `Galaxies/20231204A/` → `20190303A/`, etc.), or keep the 2023 names and add an alias?

>A. Ok, use the original source name.  And, yes, rename the repo entries.

**Q33. Insecure redshifts in the base table.**
Twelve CHIME rows in `FRBs_base.csv` have a `z` with `refs = Leung+2025` but no host JSON: 20230410A, 20230616A, 20230702A, 20230828A, 20230918A, 20230923A, 20230924A, 20231006B, 20231102A, 20231223D, 20231224A and 20240210C. In the KKO paper, every one of them is in **Table 3** (the "remaining" localizations without a secure host), with P(O|x) between 0.000 and 0.747. The paper prints no redshift for any of them. For example, 20230828A has P(O|x) = 0.000 yet z = 0.6268 in the base table. Do you know where those z values came from (perhaps photo-z of the top PATH candidate)? For the CSV, should I (a) blank them, (b) keep them and add a `P_Ox` column so Clancy can cut, or (c) keep them and flag them in `Refs`? I recommend (b). Related: the base-table z for 20231201A (0.119, also printed in Leung+2025 Table 1) disagrees with the JSON and the KKO paper (0.1119). The JSON wins, per Q5, and I'll add a fix of the base table to `add_frbs.md`.

>A. Ignore the FRBs with P(O|x) < 0.9.  Keep the JSON value and add the fix for the base table.

**Q34. Stellar masses from Leung+2025 (ApJL 991, L25).**
Table 1 gives log M* for these CHIME hosts: 20230222B (10.19, NED-LVS), 20230926A (10.49, CIGALE), 20231011A (9.59, CIGALE), 20231123A (9.42, CIGALE), 20231128A (9.47, CIGALE), 20231201A (9.47, CIGALE), 20231223C (10.40, NED-LVS), 20231229A (9.87, CIGALE+NED-LVS), 20231230A (10.04, CIGALE+NED-LVS) and 20220912A (10.00, published). Points to settle:
- All values were **converted to a Chabrier IMF**: published Kroupa masses were shifted by −0.03 dex, and NED-LVS masses from Salpeter. The repo's Prospector masses (Gordon2023, Bhardwaj2021b) are Kroupa. Should I shift everything to one IMF, or leave as-is and just note the IMF? The offset (0.03 dex) is far below the errors, so I recommend leaving as-is.
- Tiers: CIGALE → "Photometric" (`CIGALE:Leung+2025`); NED-LVS → "Other" (`NED-LVS:Leung+2025`). For the "bc" hosts I'd use CIGALE. OK?
- There are no per-object errors. The paper quotes ≤0.16 dex for CIGALE and 0.3 dex for NED-LVS. Should I put these nominal values in `Mstar_err`, or leave it blank (Q17 said "missing errors blank")?

>A. (i) leave as is; (ii) That tier scheme is fine; (iii) Leave blank

**Q35. SFRs for the KKO hosts.**
Neither paper gives SFRs, and the KKO host JSONs have no `derived` block at all. So SFR will be blank for all ~20 Outrigger hosts. Should it stay blank for now, or should `add_frbs.md` include a task to run CIGALE (or Hα from the Lick/Keck/Gemini spectra, if you have them) to get SFR and M*?

>A. Blank for now

**Q36. VLBI one-offs.**
You said to add "a few one-offs from the VLBI efforts". I can't find these reliably without a pointer, and I won't rely on memory (Q28). One candidate I came across while searching is the ApJL 991 paper "A Deep VLA Search for a Persistent Radio Counterpart to the One-off FRB 20250316A". I haven't confirmed that it's a CHIME/Outrigger localization. Can you name the VLBI papers or FRBs, or should I skip one-offs for this pass?

>A. Use the literature sources for the missing FRBs described in the Table above.
Otherwise, let it go.

### Round 5 questions (Prompt #5)

**Q37. SFR limits vs. the SFR priority order.**
(i) The repo has no convention for a *lower* limit in `derived`. `defs.py` defines only `_err = 999` (upper limit) and `-999` (no measurement). For the FRB20181030A Hα SFR (0.033) and the FRB20220912A Hα SFR (≳ 0.1), I propose `SFR_nebular_err = -998` to mean lower limit, documented next to line 30 of `defs.py`. Alternatively, you could pick another convention.
(ii) With the Q8 priority (nebular > SED > photom), FRB20181030A's CSV row would report the 0.033 **lower limit** rather than the 0.36 measurement. Proposal: a limit is used only when no tier holds a measured value. With that rule, 20181030A → 0.36 (UV+IR, `SFR_photom`), 20220912A → ≳ 0.1 with `SFR_flag = 2` (lower limit), 20240209A → < 0.36 with `SFR_flag = 1`, and 20180814A → < 0.32 with `SFR_flag = 1`. OK?

>A. I agree with your proposal.

**Q38. FRB20191106C (= 20231128A) stellar mass: a 1.2 dex conflict.**
Ibik+2024a quotes 4.5 ± 1.2 × 10¹⁰ (log 10.65) from Chang+2015; I haven't checked that paper or its method. Leung+2025 fits it with CIGALE and gets log 9.47 (Chabrier). Under the agreed tiers (Prospector > CIGALE > spectroscopic > other), Leung's CIGALE value wins, because Chang+2015 isn't Prospector. Should I follow the tiers (9.47) or flag the conflict? I recommend following the tiers, recording the Chang value in `Refs`, and adding a task to `add_frbs.md` to redo the fit with Prospector.

>A. I agree with your recommendation.

**Q39. Galactic extinction on literature magnitudes.**
Only two of the literature-only hosts state that their r magnitude is extinction-corrected: 20240209A (Eftekhari) and 20240114A (Bhardwaj Tab 8). The rest (Michilli, Ibik, Moroianu, Ravi, Hewitt) say nothing. Options: (a) apply the G23 correction (IRSA E(B−V)) in the table script to magnitudes not stated as corrected, (b) use them as-is and note `MWext:unknown` in `Refs`, or (c) in the CSV, use public Legacy Survey / Pan-STARRS catalog magnitudes, corrected by the script, instead of the paper values. I recommend (c) where a catalog detection exists, falling back to (a). This also matches what `build_hosts.py` would produce once the JSONs are built.

>A. I agree with your recommendation.

**Q40. Default choices: speak up only if you disagree.**
- FRB20240114A: Bhardwaj+2025 values (z = 0.130287, Prospector mass, Hα SFR 0.061), not Chen+2025 or Bruni.
- FRB20190417A: Moroianu+2025 (m_r = 22.42, log M* = 7.88), not Ibik+2024b (21.47).
- FRB20190208A: no redshift; m_r = 27.32 (text/abstract) rather than 27.17 (Table 2). Included as a repeater row with blank z/M*/SFR.
- FRB20181119A: included as a repeater with every host column blank. The repo FRB JSON's z = 0.26064 has no literature support, so `add_frbs.md` will remove it.
- FRB20190303A host values: tier "Other", labelled `SDSS:Michilli+2023`. The r magnitude comes from the repo's 20231204A JSON, not M23, because of the M23 Table 3/4 swap.
- Masses with no stated IMF (Michilli, Moroianu, Ravi) are used as-is (Q34).
- Mass-weighted log errors are symmetrized as (lo + up)/2 in dex. For linear-quoted masses (Ibik), the errors are converted to dex first.

>A. I agree with your choices.

### Round 7 questions (Prompt #7)

**Q41. The P(O|x) ≥ 0.9 cut now removes two spectroscopic repeater hosts.**
With the Ibik+2024a P(U) = 0.1 values (`add_frbs.md` Q3), FRB20190110C (0.779) and FRB20200223B (0.899) are blanked. The draft therefore has 12 repeaters with a host, not 14. The cut also mixes priors: KKO, Shah, Tian and Hewitt each use their own P(U), and 20180814A has no PATH value at all, so it passes. Options:
- (a) Keep the cut as is.
- (b) **Recommended:** add a `P_Ox` column and stop blanking. Clancy then chooses the cut, and the rows show what each value is based on.
- (c) Use the Ibik P(U) = 0 values (0.918, 0.994) for consistency with the other papers' default priors.

>A. ok, go with (b)

**Q42. No SFRs for the non-repeaters.**
All 16 non-repeater hosts (KKO) have no SFR, and only 8 have M*. A repeater vs non-repeater comparison is limited to z, magnitude and (partly) M*. Q35 said "blank for now". Is that still fine for the draft going to Clancy, or should the README ask Clancy whether he needs SFRs, which would mean, for example, Hα from the KKO Lick/Keck/Gemini spectra?

>A. Go with what we have for now

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

### 2026-10-04 (Prompt #3: read round-2 answers, drafted round-3 questions)

Read the answers to Q11–Q20 and checked the points they raised in the FRB repo (branch `repeater_updates`).

Findings:
- Extinction (Q18 iii): confirmed in `frb/builds/build_hosts.py` (lines 338–345). Photometry for built hosts is dust-corrected via `photom.correct_photom_table(merge_tbl, EBV, ...)`, with EBV from `nebular.get_ebv()`. So the table script will **not** re-correct those hosts. All 22 parseable CHIME host JSONs carry `photom.EBV`. Hand-entered JSONs (e.g. FRB20181030A) don't go through this path, and their status is asked in Q23. FRB20181030A has `EBV = 0.42`, which looks suspicious.
- Upper limits: `defs.py` (line 30) encodes magnitude limits as `_err = 999`, and several non-CHIME hosts use this convention. No CHIME r-band entries are limits right now.
- FRB20181030A filters vs. `defs.valid_filters`: 2MASS is already defined, in lowercase (`2MASS_j/h/k`), so it only needs a rename. WISE needs uppercase `W1..W4`. DESI maps to `DECaL_*`. The JSON's `Herschel_PACS_250/350/500` are actually SPIRE bands. The original JSON also mislabels `DESI_r`'s ref key as a duplicate `DESI_g_ref`.
- The Q12 expansion can't be read off `FRBs_base.csv`, which has no discovery-instrument column. I proposed a list in Q21 from the literature (20121102A, 20201124A, plus the Missing Repeaters list minus 20220529A).

Attempted FRB repo edit (Q15): moving `FRB20181030A_host.json` into `Galaxies/20181030A/` and rewriting the filter names / dropping `Z_spec` was **denied by the Claude Code auto-mode permission classifier**. No changes were made; the file is still at the top level of `Galaxies/` with only the round-2 trailing-comma fix. The planned edits are listed in Q25.

Added questions Q21–Q29. No other files were changed.

### 2026-10-04 (Prompt #4: acted on round-3 answers, drafted round-4 questions)

Read the answers to Q21–Q29.

Papers (Q21, Q27): found the two papers from Q21 through Crossref/arXiv and saved them to `papers/Mo_Repeaters/papers/`:
- ApJS 280, 6 = CHIME/FRB Collaboration 2025, "A Catalog of Local Universe FRBs from CHIME/FRB and the KKO Outrigger" (arXiv:2502.11217), saved as `CHIME_2025_ApJS_280_6_arXiv2502.11217.pdf`.
- ApJL 991, L25 = Leung+2025, "Stellar Mass–DM Correlations Constrain Baryonic Feedback in FRB Host Galaxies" (arXiv:2507.16816), saved as `Leung_2025_ApJL_991_L25_arXiv2507.16816.pdf`.
I haven't downloaded the other Q27 papers yet.

Findings from the papers:
- KKO Table 1/2 has 22 gold FRBs (z for 19), and these match the repo's Leung+2025 host JSONs. FRB20231204A ≡ FRB20190303A and FRB20231128A ≡ FRB20191106C (same repeating sources). See Q32.
- The 12 base-table z values with no host JSON all belong to KKO Table 3 FRBs with P(O|x) ≤ 0.75, and the paper gives no z for them (Q33).
- 20231201A: z = 0.1119 in the JSON and KKO, but 0.119 in the base table and Leung+2025.
- Leung+2025 Table 1 supplies log M* (Chabrier) for 9 CHIME hosts plus 20220912A, with no per-object errors and no SFRs (Q34, Q35).
- Updated the Missing Repeaters table: verified rows are marked, the duplicate sources are flagged, and the duplicated section header is removed.

Bhardwaj+2021b (Q23): the JSON's `EBV = 0.42` is the host Balmer E(B−V), not Galactic. The JSON photometry matches the paper's Table 10, which is already corrected for Galactic extinction (e.g. DESI r: −2.5 log10(9.26e−6) = 12.583). `Mstar_spec` is actually a Prospector mass, and `SFR_nebular` (0.35) is actually SFR_total from UV+IR (Q30).

FRB repo edit (Q25): my attempt to write the corrected JSON into `Galaxies/20181030A/` was **denied again by the auto-mode permission classifier**. That command also contained a `git rm`, which I shouldn't have included given the CLAUDE.md git rule. Nothing in the FRB repo changed. Instead I wrote `papers/Mo_Repeaters/py/fix_frb20181030A_host.py`, which: moves the file into the subfolder; renames DESI→DECaL, 2MASS_J/H/Ks→2MASS_j/h/k and WISE_w→WISE_W; fixes the duplicate `DESI_g_ref` key; drops Herschel; deletes `Z_spec`; swaps in the Galactic EBV from IRSA; moves the Prospector mass to `Mstar` with errors; and sets `AV_nebular` to 1.3 ± 0.2. It leaves `SFR_nebular` alone (Q30). The user needs to run it and do the `git rm`/`git add`.

Added questions Q30–Q36. I have not started `add_frbs.md` or the CSV build (Q29 steps 1–2), pending these answers.

### 2026-10-04 (Prompt #5: acted on round-4 answers, verified Missing Repeaters against the literature, drafted round-5 questions)

Read the answers to Q30–Q36.

FRB20181030A fix script (`py/fix_frb20181030A_host.py`), per Q30/Q31: it now sets `SFR_photom = 0.36 ± 0.08` (UV+IR; Bhardwaj2021b) and `SFR_nebular = 0.033` (Hα, aperture-limited). How to flag the lower limit is pending Q37. Herschel is still dropped. The FRB repo is unchanged: the script hasn't been run, and the old JSON is still at the top level of `Galaxies/`.

Literature verification (Q36/Q28): two subagents found the papers through the arXiv API/search and saved 12 PDFs to `papers/`: Michilli+2023, Ibik+2024a,b, Moroianu+2025, Ravi+2023, Hewitt+2023, Hewitt+2024, Bhardwaj+2025, Chen+2025, Tian+2024, Shah+2024 and Eftekhari+2024. They extracted z, m_r, M* and SFR with table/section citations. I spot-checked the key values myself against the PDFs: Ibik Table 3 (masses, SFRs), Eftekhari Table 2 (11.34 ± 0.01; the text says 11.36) and Michilli Table 3 (10.78, SFR < −0.5 dex). I rewrote the Missing Repeaters section with only verified values.

Corrections to my earlier memory-based list:
- FRB20181119A has no proposed host anywhere, so the repo's z = 0.26064 is unsupported.
- The FRB20190417A host paper is Moroianu+2025, not 2023.
- FRB20240114A z is 0.130287 (not ~0.130), with m_r ~ 21.9, a dwarf satellite host.
- FRB20220529A is dropped (FAST).

Other issues found:
- FRB20191106C mass conflict: 10.65 (Chang+15 via Ibik) vs 9.47 (Leung CIGALE).
- Michilli+2023 Tables 3 and 4 swap the z of the two FRB20190303A merger members.
- Several papers give a magnitude in the text that differs from their own table (20190208A, 20240209A mass).
- Most literature magnitudes don't say whether they are extinction-corrected.

The arXiv metadata had no journal volume/page for any of these papers, so the bibliographic references are arXiv IDs plus DOIs where available.

Added questions Q37–Q40. I haven't started `add_frbs.md` or the CSV build; I'll do both in the next prompt once Q37–Q40 are settled.

### 2026-10-04 (Prompt #6: built CHIME_FRB_hosts.csv and wrote add_frbs.md)

Read the answers to Q37–Q40 (all proposals accepted). Did Q29 steps 1–2.

**`CHIME_FRB_hosts.csv`** (top of `papers/Mo_Repeaters/`) is built by `py/build_host_table.py` (`astro` env).
- **Columns:** TNS_NAME, Repeater, Redshift, Redshift_err, Magnitude, Magnitude_err, Mag_flag, Band, Stellar_Mass (log10 Msun), Mstar_err (dex), Mstar_source, SFR (Msun/yr), SFR_err, SFR_flag (0 = measured, 1 = upper limit, 2 = lower limit), SFR_type, Refs.
- **Coverage:** 97 rows, 16 of them repeaters.
  - 86 rows are `telescope==CHIME` in the base table.
  - FRB20121102A and FRB20201124A are included per Q12.
  - 9 are literature-only repeaters from the Missing Repeaters table.
  - Of the 97: 30 have z, 31 a magnitude, 22 M*, 14 SFR.
  - All 16 repeaters have a name and Repeater flag. 14 have z, M* and SFR; the exceptions are 20190208A (magnitude only) and 20181119A (all blank).
- **Q33, my interpretation:** rows with P(O|x) < 0.9 are **kept** (as Q11 asked), but their host values are blanked and `Refs` says `host_ignored:P(O|x)=...`. Setting `DROP_LOW_POX = True` in the script drops them instead. This blanks 20230311A (P = 0.774), even though KKO gives it a secure z = 0.1918, and 20231020A (P = 0.571, JSON z = 0.0455).
- **Q32:** FRB20231204A → FRB20190303A and FRB20231128A → FRB20191106C (the old names are noted in `Refs`).
- **Precedence:** repo host JSON first, then the base-table z, then literature values hard-coded in the script (`LIT_HOSTS`, `LIT_SUPPLEMENT`), each tagged in `Refs`.
- **FRB20181030A:** the repo JSON hasn't been fixed yet, so the script reads the old top-level file and applies the Bhardwaj2021b corrections in memory (`FIX_181030A`).
- **Q37 SFR rule:** a measured SFR wins over a limit in the priority order. Results: 20181030A → 0.36 (UV+IR); 20220912A → 0.1 lower limit; 20180814A and 20240209A → upper limits.
- **Q39 magnitudes:** literature-only hosts take Legacy Survey / Pan-STARRS r magnitudes from `frb.surveys` (nearest object within 2″), corrected with G23 and the IRSA E(B−V) (S&F mean). The query results are cached in `py/lit_host_photom_cache.json`.
  - Exception: 20240209A uses the Eftekhari GMOS value, which the paper states is already corrected.
  - Fallbacks: 20190208A (GTC; corrected using the SDSS_r curve) and 20190417A (GMOS; GMOS_r curve), because neither has a catalog detection.
  - Check: the uncorrected catalog values match the papers (20190110C DECaL 18.009; 20220912A PS1 ≈ 19.65).
- **Bug fixed during the build:** FRB20231206A's repo `DECaL_r` is `inf` (err = −99). The script now skips non-finite or negative-error magnitudes, so it uses Pan-STARRS_r instead. Its host JSON should be repaired.

**`~/Astronomy/FRB/FRB/prompts/add_frbs.md`** contains nine prompts:
1. Run the 20181030A fix.
2. Document the −998 lower-limit convention in `defs.py`.
3. Rename the two later-burst repeaters.
4. Base-table fixes: 20231201A z typo; the 12 low-P(O|x) z values; 20181119A's unsupported z.
5. Add the missing repeaters to `FRBs_base.csv`.
6. Build their host JSONs via `public_hosts.csv` + `build_hosts.main()`.
7. Add literature derived-quantity tables via `all_refs.csv`.
8. Prospector refits (20191106C mass conflict, CIGALE/NED-LVS hosts).
9. Regenerate the CSV and strip the hard-coded literature values.

Found while writing it: `Literature/bhardwaj2021_derived_FRB20181030A.csv` actually contains FRB20201124A values.

**Other edits:**
- `py/fix_frb20181030A_host.py` now writes `SFR_nebular_err = -998` (lower limit, Q37).
- The FRB repo is otherwise unchanged: the fix script still needs to be run by the user, and `add_frbs.md` is the only new file there.

### 2026-10-05 (Prompt #7: built the draft table for Clancy)

Read the `add_frbs.md` logs (prompts 1–9, Q1–Q13 and the TODO list). The repo now holds all the literature values. `py/build_host_table.py` (as updated in `add_frbs.md` prompt 9) reads only the repo.

- Reran `py/build_host_table.py`. The output was identical to the CSV from `add_frbs.md` prompt 9.
- Fixed two small bugs in the script:
  - A NaN redshift (FRB20190208A) wrote a `z:hostJSON` ref even though no z was used. Now a z is only taken (and referenced) if it is finite.
  - `sym_err` now returns the flag value for ±999 / −998 errors instead of combining them with real errors, and gives no log error for those.
  - The only change in the CSV is that `Refs` entry.

**Draft table: `CHIME_FRB_hosts.csv`.**
- **Sample:** 97 rows, made up of the 95 `telescope == CHIME` rows plus 20121102A and 20201124A.
  - 16 are repeaters: 12 have z, M* and SFR; 20190208A has a magnitude only; 20181119A has no host; 20190110C and 20200223B are blanked by the P(O|x) cut.
  - 16 non-repeaters have a host with z. 8 of these have M*, and none has an SFR.
- **Added `CHIME_FRB_hosts_README.md` for Clancy:** sample definition, columns and flags, tiers, and caveats. The caveats cover:
  - coverage;
  - mixed methods and IMFs;
  - aperture-limited Hα;
  - the 20191106C mass conflict;
  - 20200120E/M81;
  - the 20180814A TODOs;
  - the 20190303A merger;
  - catalog magnitudes of large galaxies.
- **Checked against the TODOs:** 20180814A (ellipse, P_Ox) and 20191106C (Leung mass) are pending outside replies. The draft notes both.

Added Q41 (P(O|x) cut) and Q42 (non-repeater SFRs).
