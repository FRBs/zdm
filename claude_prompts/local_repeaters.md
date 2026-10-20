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

3. One or more of the CI tests are failing.  Please investigate and let me know
how we might fix them.  Let me know what you think in QA/Tests.
Use Opus 5.5.  Log your work.

## Q&A

### Tests

(From task 3b, CI failures, 2026-10-20.)

**What is failing.** The latest run on this PR (run 39387908567, 2026-10-20) fails all three jobs. `3.12-test` and `3.12-test-alldeps` fail at the `pip install` step inside tox with `ResolutionImpossible`:

```
The user requested ne2001 0.0.2 (from git+https://github.com/FRBs/ne2001.git)
frb 0.1.dev0 depends on ne2001 0.0.2 (from git+https://github.com/FRBs/ne2001)
```

`3.12-test-astropydev` did not fail on its own; it was cancelled by fail-fast when the other two jobs died. No zdm test actually ran. Nothing in this PR caused it: the identical branch passed on 2026-10-07, and pip is 26.2.1 in both runs.

**Why.** On 2026-10-11 the FRB repo merged PR #237 (survey_revamp), which includes commit 293d58f "consistency with ne2001 repo". It changed FRB's `pyproject.toml` direct-URL dependencies from `.../ne2001.git` and `.../astropath.git` to `.../ne2001` and `.../astropath` (no `.git`). Our `setup.cfg` still lists `ne2001 @ git+https://github.com/FRBs/ne2001.git` and `astropath @ git+https://github.com/FRBs/astropath.git`. pip treats two direct-URL requirements for the same package as satisfiable only if the URL strings match exactly, so the resolver gives up even though both point to the same commit. The passing run pulled FRB at 996fcda (pre-merge, still `.git`); the failing run pulled 85ecfdd (the merge).

**How to fix (pick one).**
1. Match FRB's spelling in `setup.cfg` lines 49 and 51: drop the `.git` from the ne2001 and astropath URLs. Verified locally with `pip install --dry-run`: the current spec reproduces the `ResolutionImpossible` error, and the spec without `.git` resolves cleanly (`Would install FRB-0.1.dev0 ... astropath-0.3.dev0 ... ne2001-0.0.2 ...`).
2. Delete the ne2001 and astropath lines from `setup.cfg` altogether and keep only `frb @ ...`. FRB already depends on both, so they get installed anyway, and we stop having to keep the URL strings in sync with FRB. This is the more robust option; the cost is that the dependency is implicit.

Either way, the fix is to `setup.cfg` on this branch (or on `main` and then merged in). Nothing about the CI workflow or tox needs to change for this. I did not apply the change since the prompt asked for an assessment.

**Two related notes.** The `README`/CLAUDE.md install instructions still say `pip install git+https://github.com/FRBs/ne2001.git#egg=ne2001`; that is harmless for a manual install but could be updated for consistency. And the earlier failure on `main` (run 37287081933, 2026-10-05) was a different, 3.11 `test-astropydev` failure that predates the 3.12-only matrix; it is not related to this one.

### PR


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

### 2026-10-20 (Task 3b: CI tests failing on PR #91; diagnosed, fix proposed in Q&A/Tests)

All three `Tox env 3.12-*` jobs on run 39387908567 fail. `test` and `test-alldeps` die in `pip install` with `ResolutionImpossible` on `ne2001`; `test-astropydev` is just cancelled by fail-fast. The same branch passed on 2026-10-07 with the same pip (26.2.1), so the cause is external.

Cause: FRB PR #237 (merged 2026-10-11, commit 293d58f) rewrote FRB's direct-URL deps as `git+https://github.com/FRBs/ne2001` and `.../astropath` (no `.git`). zdm's `setup.cfg` still uses the `.git` spelling. pip requires direct-URL requirements for the same package to match textually, so the two specs conflict even though they resolve to the same commit. The passing run had FRB at 996fcda (old spelling); the failing run had 85ecfdd (the merge).

Verified with `pip install --dry-run` in fresh venvs in the scratchpad: the current three-URL spec reproduces the error (exit 1); dropping `.git` from the ne2001 and astropath URLs resolves (exit 0).

Proposed fixes, written up in Q&A → Tests: (1) drop `.git` on `setup.cfg` lines 49 and 51, or (2) remove the ne2001/astropath lines and rely on FRB's own dependency on them. Did not edit `setup.cfg`, since the prompt asked for an assessment. Also noted that the 2026-10-05 failure on `main` was a separate 3.11 astropydev issue.

### 2026-10-20 (Task 3b follow-up: applied option 1)

User chose option 1 (keep the explicit dependencies, match FRB's URL spelling). Recommended it over removing the lines because zdm imports ne2001 (survey.py, MC_sample/mc.py) and astropath (optical.py, optical_numerics.py, scripts/Path/*) directly, so they should stay declared; the cost of keeping the URL spelling in sync with FRB is low and any future drift fails loudly at install time.

Changes: dropped `.git` from the ne2001 and astropath URLs in `setup.cfg` (lines 49 and 51), and from the manual `pip install` line in `CLAUDE.md` and `docs/installation.rst` (two places) for consistency. The `frb @ .../FRB.git` line is untouched since nothing else depends on FRB by URL. Verified with `pip install --dry-run` of the edited package in a fresh venv: resolves with no conflict. Not committed; user handles git.
