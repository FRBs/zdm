# LumFunc prompts

## Goals

Work on the `local_repeaters` PR.

## Prompts

1. Read this file.  Execute the 1st task under "Pull Request"
2. Read this file.  Execute the 2nd task under "Pull Request"

## Pull Request

1. There is an open pull request for the `local_repeaters` branch.  Please review the new code and submit a review as `profxj`.  Include both a summary and individual comments at the appropriate places in the code.   
If you have any questions, put them in the Q&A section below and I will answer them.  Use Opus 5.5.  Log your work.

2. The "Tox env 3.11-test-alldeps" is not running.  Please check why and fix it.  Use Opus 5.5.  Log your work.

3. While we are at it, please review this open PR, similar to what you did for PR #91: 
`https://github.com/FRBs/zdm/pull/84`.  Use Opus 5.5.  Log your work.

## Q&A

(From PR review, task 1. The questions marked "for PR author" are also posted on the PR.)

- Q: I submitted the review as a plain "COMMENT", not "REQUEST_CHANGES". Do you want it escalated to request changes because of the KS-test / percentile bugs?
- Q (for PR author): `HoffmannRepeaters26Pn` still has `lC = 3.3249`, which looks like a placeholder. Is this intended?
- Q (for PR author): Was `zdm/data/Surveys/extract_DSA.py` deleted on purpose? It documents how `DSA.ecsv` was built.

## Logs

### 2026-10-07 (Review of PR #91 "Local repeaters")

PR #91 (FRBs/zdm, author cwjames1983, head = local `ec9d3fb`) has 28 files, +1485/-241. Most of it is paper material for `papers/2026_FitRepetition` (renamed from `FitRepetition2025`). Submitted a COMMENT review as profxj with a summary and 23 inline comments:
https://github.com/FRBs/zdm/pull/91#pullrequestreview-5443532124

Main findings (all checked against the code and data in the repo):
- `CHIME.py`: the KS tests are misused. `ks_2samp` is run on arrays of CDF values instead of `kstest(DMs, model_cdf)`. There is a fake (DM=3000, CDF=1) point in the empirical CDF. Host DM is probably double-counted (`g.dmvals + 10**lmean`). `plot_dm_z` is defined twice, so the first version can't be called. `chimefrbcat2.npy` is required but not in the repo. `main()` is unguarded.
- `papers/MeerTRAP/plot_z_comparison.py`: `cumsum` of a density without `*dz`, so the z_95 and "P(z>2)" values are wrong; the latter is also really P(z<2). CHIME dec bin 0 is skipped (the append comes after `continue`). The zDM figure call is commented out, leaving a lot of dead code. There is a "redshifty" typo.
- `run_mcmc.slurm` points to `zdm/data/MCMC/params.json`, which is not the paper `params.json` and does not fit lRmin/lRmax/Rgamma. It references `DSA_34`, which only exists in the paper `Surveys/` dir. It nearly duplicates `slurm/run_mcmc.slurm`.
- Most of the paper `Surveys/*.ecsv` are byte-identical to `zdm/data/Surveys`. The paper `MeerTRAPcoherent.ecsv` differs only by a Unicode minus (U+2212) in a Dec.
- `zdm/figures.py`: the Macquart overlay fix (z instead of index, plus the missing `pcosmic` import) is correct. The legend location change is global, so I suggested a kwarg. Also flagged leftover commented-out survey-specific styling.
- `zdm/states.py`: `HoffmannRepeaters26Pn` has `lC=3.3249` (looks like a placeholder). `Prep` has a docstring and comment that contradict each other.
- `check_average_survey.py`: the `misc_functions` import is now unused.
### 2026-10-07 (Task 2: "Tox env 3.11-test-alldeps" not running; diagnosed, not fixed)

Cause: branch protection on `main` requires a status check named exactly "Tox env 3.11-test-alldeps". Since commit `15623f5` ("Update ci_tests.yml"), `.github/workflows/ci_tests.yml` only runs Python 3.12. The jobs it produces are `Tox env 3.12-{test,test-alldeps,test-astropydev}`, so the required 3.11 check never reports and GitHub waits forever. All 3.12 jobs on PR #91 pass.

Fix (not applied; the permission system blocked the branch-protection change): in Settings → Branches → `main`, replace the required check with "Tox env 3.12-test-alldeps", or run
`gh api -X PATCH repos/FRBs/zdm/branches/main/protection/required_status_checks -F strict=true -f 'checks[][context]=Tox env 3.12-test-alldeps' -F 'checks[][app_id]=15368'`.
Alternative: add '3.11' back to the CI matrix.

### 2026-10-07 (Task 3: review of PR #84 "Mawsons lensing code")

PR #84 (cwjames1983; head MWSammons/zdm `fe89f1a`; 45 files, +2480/-494; 55 ahead / 20 behind `main`; AI-assisted merge) adds cluster lensing and cluster DM/scattering support.

I reviewed the full diff and checked the findings against files fetched at the PR head. Blockers:
1. `misc_functions.py`/`pcosmic.py`: unused top-level `import magnificationMapper` (a module in `zdm/scripts/`) gives `ModuleNotFoundError`.
2. `survey.py`: `self.cluster` is set after `reinit()` → `init_widths()` reads it, so every Survey raises `AttributeError`.
3. `energetics.py`: the `SplineLog` branch was dropped from `vector_cum_gamma_spline`, but the splines are still built in log–log space, so LF=2 is wrong.
4. `grid.calc_rates`: `Z_FRACTION` is applied twice in standard mode.
5. `initialise_grids` passes cluster args positionally to `repeat_Grid`, which maps them onto `opdir/Exact/MC/verbose`.
6. `MC_sample/loading.py`: wrong positional order into `initialise_grids` → `TypeError`. The PR also removes `edir`.

Other points: the lensed LF ignores use_log10, hard-codes Planck18 and does disk I/O inside `calc_pdv`; `Z_PHOTO` smearing is skipped in cluster mode; scripts hard-code `/arc/...` paths and `lC = 2.3-9`; much of the `survey.py` diff is reformatting and deleted comments. Positive: the `s.meta` → `self.meta` fix for width_method 4.

Posting failed: GitHub returned HTTP 500 for every write to PR #84. That includes REST reviews (even a bare review with no comments), GraphQL `addPullRequestReview`, and a plain `gh pr comment`. githubstatus.com reported all systems operational. I confirmed nothing was posted. Ready-to-post files are in the session scratchpad: `review84.json` (REST payload, event=COMMENT, 16 inline comments) and `review84_comment.md` (a single-comment version with permalinks).

Update: GitHub recovered and a retry succeeded. The COMMENT review with 16 inline comments is posted as profxj: https://github.com/FRBs/zdm/pull/84#pullrequestreview-5444468748
