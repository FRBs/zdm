# zdmp prompts

## Goals

## Prompts

1. Read this file.  Execute the 1st task under "Pull Request"

## Pull Request

1. There is an open pull request for the `zdmp` branch.  Please review the new code and submit a review as `profxj`.  Use Fable if you can.  Log your work.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-08-11 (Reviewed PR #82 (zdmp → main) and submitted a request-changes review as profxj)

Executed prompt 1: the 1st task under "Pull Request". Found the open PR ([FRBs/zdm#82](https://github.com/FRBs/zdm/pull/82), `zdmp` → `main`, by cwjames1983) and reviewed the full diff using an 8-agent code-review scan (angles: core grids, survey/iteration, optical/repeat/MCMC, cross-file tracing, removed-behavior audit, conventions, reuse/simplification, efficiency), then manually spot-checked the highest-impact claims against the branch before publishing. Submitted a CHANGES_REQUESTED review under the profxj account (gh CLI was already authenticated as profxj) with ~20 blocking correctness findings, 7 stale references to code deleted in the PR, 4 silent behavior changes, plus performance and duplication notes. Highlights: `Grid.update()`'s DMhalo branch calls `get_efficiency_from_wlist` with the old signature (TypeError in MCMC); `calc_pdv` accumulates `+=` into `b_fractions`/`w_fractions` without zeroing, so rates roughly double after every `grid.update()`; `calc_rates` compounds `sfr *= fz` on repeated calls; `lRmin`/`lRmax` are swapped in both HoffmannRepeaters26 presets and the case string at states.py:66 never matches (CHIME scattering silently overwrites the fitted parameters); `calc_likelihoods_2D` contains a duplicated ~90-line likelihood block whose stray recomputation overwrites the MW-uncertainty-weighted values; the console-script entry points (`zdm_build_cube`, `zdm_pzdm`) point at deleted/nonexistent files. Learned: the session ran on Fable per the prompt's request; verification-by-spot-check caught no false positives among the checked items, and the review notes it was prepared with Claude Code assistance.