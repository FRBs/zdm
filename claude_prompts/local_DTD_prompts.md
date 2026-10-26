# Delay Time Distribution 

## Goals

Work on the `local_DTD` PR.

## Pull Request

1. There is an open pull request for the `local_DTD` branch.  Please review the new code and submit a review as `profxj`.  Include both a summary and individual comments at the appropriate places in the code.   
If you have any questions, put them in the Q&A section below and I will answer them.  
Use Opus 5.5.  Log your work.

2. I have answered your questions in the Q&A section below.  Please modify our review of our PR accordingly.
Use Opus 5.5.  Log your work.

## Q&A

(From PR review, task 1, 2026-10-22. The questions marked "for PR author" are also posted on the PR.)

- Q: I submitted the review as a plain "COMMENT". Two of the findings are real bugs (narrow DTDs return ~1e-31 instead of ~SFRD; repeat_grid still calls `source_function(zvals, sfr_n)` and so breaks for modes 2 and 4). Do you want it escalated to "REQUEST_CHANGES"?
>A. Yes, please escalate to "REQUEST_CHANGES".
- Q: Modes 2 and 3 are not normalised to 1 at z=0, so `lC` changes meaning (about 100x lower expected rates for mode 2 on CRAFT_ICS_1300, 3.75x higher for mode 3). Do you want this treated as a blocker, or is it acceptable for `lC` to absorb it?
>A. We will leave it to the author
- Q (for PR author): Is the missing z=0 normalisation in modes 2/3 intentional?
>A. Be sure to ask this in our Review
- Q (for PR author): Where do the Kremer+23 fit coefficients come from, and over what redshift range are they valid?
>A. Be sure to ask this in our Review
- Q (for PR author): Should the SFRD term inside modes 2 and 4 be raised to `sfr_n`, or is SFR^1 the intended model?
>A. Be sure to ask this in our Review
- Q (for PR author): Will `pl_alpha`, `pl_tmin`, `pl_tmax` be fitted in MCMC? Mode 2 adds ~17 s per grid build, and MCMC rebuilds every survey grid at every step.
>A. Be sure to ask this in our Review
- Q (for PR author): Could tests be added for the new modes (normalisation, the `f_GC=0` limit, and a delta-like DTD reproducing the SFRD)?
>A. Be sure to ask this in our Review

## Logs
### 2026-10-22 (Review of PR #92 "Delay time distribution (DTD)")

PR #92 (FRBs/zdm, author cwjames1983, Alexa's work; head `356028e`, matches the local `local_DTD` checkout) adds three new `FRBdemo.source_evolution` modes: (2) a power-law delay-time distribution convolved with the MD14 SFRD, (3) a fitted WD-WD globular-cluster merger-rate curve from Kremer+23, and (4) a mixture of SFRD and that curve with fraction `f_GC`. Three files: `cosmology.py`, `grid.py`, `parameters.py` (+245/-7).

Per the prompt, the review itself was done by an Opus 5.5 subagent (full diff, traced from `Grid.set_evolution`, numerical checks in throwaway scripts). I verified the payload mechanically (all 20 inline comment lines inside diff hunks), reran the check scripts, and confirmed the headline findings against the code before posting. Submitted a COMMENT review as profxj with a summary and 20 inline comments:
https://github.com/FRBs/zdm/pull/92#pullrequestreview-5614475722

Main findings:
- Narrow DTDs silently return ~1e-31: `quad` over z' in [z, 30] with a 1e-40 floor never samples the support when `pl_tmax` is small (up to ~0.3 Gyr). A direct integral over delay time gives ~SFRD(z) (e.g. 0.019 at z=0.1). Wide DTDs, including the defaults, agree to 0.2%.
- Modes 2 and 3 are not normalised to 1 at z=0 (mode 2 gives 0.026, mode 3 gives 3.75), unlike modes 0, 1 and 4. `lC` is documented as the z=0 rate, so its meaning shifts; for mode 2 it becomes degenerate with the DTD parameters.
- `repeat_grid.py` lines 170 and 229 still call `source_function(zvals, sfr_n)`, so repeater grids raise IndexError for mode 2 and silently use `sfr_n` as `f_GC` for mode 4.
- Mode 2 mixes cosmologies: dt/dz follows `state.cosmo` but lookback times use hard-coded astropy Planck18. With H0=73, Om=0.35 n(z=1) moves from 0.0828 to 0.0732.
- Mode 2 is slow (~20 s per grid build at nz=500 vs 3 s for mode 0); no tests for modes 2-4; `f_GC` and tmin<tmax unchecked; duplicated WD-WD curve code; unused `z_at_value` import; docstrings omit mode 4.
- Positive: modes 0/1 take exactly the previous code path, so default behaviour is unchanged. Mode 4 with f_GC=0 reproduces `sfr_evolution(z,1)` to 2e-8.

Tests: full suite passes on the PR head (31 passed) in the `astro` conda env. Note the base `python` in this shell is the miniforge base env without scipy, so pytest fails at collection there; use `/Users/xavier/miniforge3/envs/astro/bin/python`. Also, the local `main` branch is stale (e0f985b), so `git diff main...HEAD` shows 35 files; `origin/main...HEAD` matches the PR.

Questions for the maintainer and author are in Q&A above. Check scripts and the review payload are in the session scratchpad (`checks/`, `review92.json`, `review92_notes.md`).

### 2026-10-22 (Task 2: updated the PR #92 review per the Q&A answers)

Answers: escalate to REQUEST_CHANGES; the z=0 normalisation of modes 2/3 is the author's call, not a blocker; keep the five author questions in the review.

GitHub cannot change a submitted review's state (COMMENTED → CHANGES_REQUESTED), and the 20 inline comments are attached to the original review, so the update was done in two steps. An Opus 5.5 subagent redrafted the text from the posted body and the notes; I checked it against the answers and posted it.
1. Edited the body of the original review (id 5614475722) via `PUT /pulls/92/reviews/{id}`. It now has an "updated 2026-10-22" line, and sorts the findings into Blocking (narrow DTDs return ~1e-31; `repeat_grid` breaks for modes 2/4), Strongly recommended but not blocking (mixed cosmologies, performance, tests, smaller items), and "Your call" (z=0 normalisation, with the numbers kept and pointing to question 1). Positives and the five questions are unchanged.
   https://github.com/FRBs/zdm/pull/92#pullrequestreview-5614475722
2. Submitted a new, short REQUEST_CHANGES review (id 5614578485) as profxj with no inline comments. It links the original review, lists the two blocking items with pointers to the inline comments at `cosmology.py:720` and `grid.py:469`, states that the normalisation and the rest are not blocking, and asks the author to answer the five questions.
   https://github.com/FRBs/zdm/pull/92#pullrequestreview-5614578485

PR state checked before posting: head still 356028e, no replies or other reviews. The PR now shows "Changes requested" from profxj. Drafts are in the scratchpad as `review92_body_v2.md` and `review92_rc_body.md`.
