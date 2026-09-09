# LumFunc prompts

## Goals

Work on the `NewQinLF` PR.

## Prompts

1. Read this file.  Execute the 1st task under "Pull Request"
2. Read this file.  Execute the 2nd task under "Pull Request"

## Pull Request


1. There is an open pull request for the `NewQinLF` branch.  Please review the new code and submit a review as `profxj`.  If you have any questions, put them in the Q&A section below and I will answer them.  Use Opus 5.  Log your work.

2. Clancy has written to me:

> This PR is intended to fix issues with PR #87. In that PR, Claude found four categories of issues. Category 1 ("Regressions from the iteration.py conflict resolution (blocking)" should now be completely fixed due to a rewind. Could you please explicitly check that this is the case. Category 2 ("Issues in the new LF code") will have some fixes and other issues will be ignored as being deemed either features or irrelevant. Please analyses these issues and determine which have been fixed, and which are still outstanding. Category 3 "Structural suggestion (would collapse much of the above)" - no attempt has been made to fix this. Could you please propose modifications to the code that would be a solution to this problem. Category 4 ("Minor") - please recheck this category. Please also highlight any new issues that you find.

>In this, I'm asking Claude to propose a fix to category 3. Should we ask it to generate a new commit to that effect? Create a new branch fixing that issue?

Please read the above and make suggestions in the Q&A section below.  Use Opus 5.  Log your work.

3. The CI tests are failing on GitHub. Please check them and discuss them with me in the Q&A section below.  Use Fable if you can.  Log your work.

## Q&A

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-09-09 (Reviewed PR #89 (NewQinLF → main) and submitted a request-changes review as profxj)

Executed prompt 1: the 1st task under "Pull Request". Found the open PR ([FRBs/zdm#89](https://github.com/FRBs/zdm/pull/89), `NewQinLF` → `main`, by cwjames1983) — the successor to closed PR #87, re-merged from main. First verified the #87 findings were addressed: `minimise_const_only2`, the PATH machinery, ptauw `dkis2` coefficients, `get_rates()` masking, and `lEmax=41.84` all restored; `Grid.update()` stays deleted; `grid_kwargs` now passed to `get_zdm_grid`; MeerTRAP Unicode-minus and the scripts-copy `DSA_34` reference fixed. Ran the multi-agent code-review scan (the forked orchestrator also did a full line-by-line pass when its finders were slow to schedule, then six finders reported independently — strong cross-agent consensus) and spot-checked every blocking claim by hand. Submitted CHANGES_REQUESTED with 5 blocking items, all small fixes that crash the PR's own shipped configurations: (1) half-applied rename `zt_tomult`→`tz_tomult` (NameError on any `--ptauw` run); (2) `bweights`/`wweights` undefined in `calc_likelihoods_2D` — main used `zbweights`/`zwweights` (NameError on the `--Pn --pwb` config both slurm scripts run); (3) `if nthreads < 1:` before the None-check with new default `nthreads=None` (TypeError on every default invocation; plus the thread-capping comment has no code behind it); (4) `np.empty_like` + mask-fill leaves uninitialized memory for NaN thresholds in LF 4/5/6; (5) the new papers/FitRepetition2025 slurm script is a stale pre-fix copy referencing nonexistent `DSA_34`/`params2.json`. Silent-behavior items: `nz/ndm/zmax/dmmax` ignored when `g0info is None`; `run_slice.py` lEmin 38.39→30.0 with the old value commented out; `SplineMin`/`NSpline` knot changes break bit-level reproducibility and add ~70% spline-build cost; the broken-Schechter spline still has no domain guard (priors sit exactly at the 1e-9 boundary, `np.clip` masks extrapolation); `energetics.reset()` still rebuilds the spline every posterior evaluation (~0.24 s measured → CPU-hours per run). Structural #87 carryovers noted as follow-up material (LF registry, 7 copies of the integral helper with inconsistent gamma≈0 tests, constraint split with the try/except still commented out, drifting JSON configs, the copy-pasted paper script with deprecated `pkg_resources` under the new python_requires>=3.12). Suggested two smoke tests that would have caught all three crashes (none of the new tests exercise ptauw/pwb/default-nthreads). No questions for the user were needed, so the Q&A section stays empty. Learned: the new tests passing while the flagship config crashes is a coverage gap, not a safety signal. Note: the prompt asked for Opus 5; this session runs on Fable 5 and cannot switch models mid-session. Addendum: a late-arriving cross-file check surfaced one further verified item — `ConvertToMeaningfulConstant`'s new LF 4/5/6 branches return a shape-(1,) ndarray and a normalized survival fraction where the LF 0 branch returns an unnormalized float (both quirks inherited from the pre-existing else branch) — posted as a follow-up comment on the PR.
