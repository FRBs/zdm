# LumFunc prompts

## Goals

Work on the `QinLumFuncs` PR.

## Prompts

1. Read this file.  Execute the 1st task under "Pull Request"
2. Read this file.  Execute the 2nd task under "Pull Request"

## Pull Request

1. There is an open pull request for the `QinLumFuncs` branch.  Please review the new code and submit a review as `profxj`.  Use Fable if you can.  Log your work.

2. The CI tests are failing on GitHub. Please check them and discuss them with me in the Q&A section below.  Use Opus if you can.  Log your work.

## Q&A

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-09-05 (Reviewed PR #87 (QinLumFuncs → main) and submitted a request-changes review as profxj)

Executed prompt 1: the 1st task under "Pull Request". Found the open PR ([FRBs/zdm#87](https://github.com/FRBs/zdm/pull/87), `QinLumFuncs` → `main`, by cwjames1983, merging Qin Wu's luminosity-function work via PR #85). Scope was clean this time: `main` already contains zdmp, so the PR diff vs e85d656 (14 files, +2030/−796) is exactly the new work; the local working tree was already on the branch at the PR head. Ran the 8-agent code-review scan (started 2026-09-04; the last finders completed overnight — I initially miscounted and thought one was still pending), then spot-checked every headline claim line-by-line against the branch and `main` before publishing. Central finding: commit fcafe35 "Restore iteration module after conflict resolution" restored a pre-`main` version of `iteration.py` — `minimise_const_only2` deleted but called at MCMC.py:397 (any `--Pn` run crashes), the entire PATH machinery deleted with live callers, main's ptauw bilinear coefficients reverted (dkis1 where main has dkis2), `get_rates()`/`get_dm_bias()` masking dropped, negative-DMEG FRBs now raise instead of being penalised, and a hand-verified mangled merge at iteration.py:285-298 that unconditionally overwrites the sigmaDMG-weighted pvals, resets llsum with `=`, and leaves the `pdmz=False` branch referencing an undefined `pdm`. New-LF-code issues: broken-Schechter igamma spline silently extrapolates outside [1e-6, 1e6] under the shipped priors (lEb=36, lEmax=45 → x=1e-9); `energetics.reset()` per posterior evaluation rebuilds the mpmath spline every MCMC step; the `grid_kwargs` resolution fix is dead code (never passed to `get_zdm_grid`); `lEmax` default silently changed 41.84 → 43.0; `Grid.update()` reintroduced despite main's deliberate deletion. Suggested a single LF dispatch registry in energetics to collapse the four hand-synced dispatch sites and the five copies of the (r**g−1)/g helper. Submitted CHANGES_REQUESTED under profxj — unlike #83, merging this would break `main` functionality outright — with credit for the LF implementations, the new tests, and the shipped MCMC configs; recommended re-merging iteration.py from main and re-applying the LF changes on top. Note: the prompt asked for Fable — this session is Fable.
