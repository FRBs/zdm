# Delay Time Distribution 

## Goals

Work on the `local_DTD` PR.

## Pull Request

1. There is an open pull request for the `local_DTD` branch.  Please review the new code and submit a review as `profxj`.  Include both a summary and individual comments at the appropriate places in the code.   
If you have any questions, put them in the Q&A section below and I will answer them.  
Use Opus 5.5.  Log your work.

2. I have answered your questions in the Q&A section below.  Please modify our review of our PR accordingly.
Use Opus 5.5.  Log your work.

3. Clancy has addressed most of the requests.  Please review the updates
and responses.  If you have any questions or concerns, put them in the "Q&A/Round 2" section below and I will answer them.  Use Opus 5.5.  Log your work.

4. I have answered the Round 2 questions.  Please read them and then post
a new PR review on GitHub.  Use Opus 5.5.  Log your work.

## Q&A

### Round 2

(From task 3, 2026-11-11, reviewing head `4799de4`. A draft follow-up review is in the session scratchpad, `review92_round2_draft.md`.)

- Q: Clancy fixed both round-1 blockers (narrow DTDs and `repeat_grid`), but the updates add two new ones. (a) In `grid.py:489-493` the `alpha_method==1` branch overwrites `self.sfr` on the second line, so the `(1+z)^-alpha` factor is lost. This changes the **default** behaviour for everyone: on CRAFT_ICS_1300, rates are 2.2x higher than on main, by exactly (1+z)^1.54. (b) `grid.py:470` still reads `pl_alpha` after the rename to `pl_beta`, so mode 2 can't build a Grid (AttributeError). Should I post the draft as a new REQUEST_CHANGES review from profxj, with inline comments on these two lines and the smaller items?
>A. Yes, let us bring this to his attention as a new REQUEST_CHANGES review.
- Q: Clancy replied "handled via a clip" for the `f_GC` range, but no clip is in the code (f_GC=1.5 gives n=-3.46 at z=1.67). Raise it again, or accept it given the reply that negative values "might tell us something physical"? (Negative n gives NaN in log-likelihood either way.)
>A. Let this one go
- Q: The new lognormal DTD mode (5) brings back two round-1 issues in new code: it mixes Planck18 and zdm cosmologies, and it takes 8 s per 500 z. Treat these as blocking, or as recommendations like before?
>A.  Make these recommendations.
- Q: `zdm/tests/test_source_evolution.py` isn't a pytest test: there's no `test_*` function and no asserts. Do you want real tests to be a merge condition, or should I offer to write them (I could add them on this branch myself if you prefer)?
>A. Let's offer to write them
- Q: `claude_prompts/local_DTD_prompts.md` (this file, added in your commit 0ad5c62) is part of the PR diff. Do you want it kept out of main? The PR also carries unrelated changes: `rmult` in `loading.py`, and debug edits to `write_macquart_relation.py` (H0=70, a print).
>A. Let's keep it in the PR.

### Round 1

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

### 2026-11-11 (Task 3: reviewed Clancy's updates and replies on PR #92)

PR head is now `4799de4`, which matches the local `local_DTD`. There are 13 commits since the reviewed head 356028e, and the PR now touches 8 files: `repeat_grid.py`, `loading.py`, `write_macquart_relation.py`, a new `tests/test_source_evolution.py`, and this prompts file have been added. Clancy replied to 18 of the 20 inline comments. The two without replies are the positive note and the `repeat_grid` blocker, which is fixed anyway. Nothing was posted to GitHub this round. The draft follow-up review and the questions are in Q&A Round 2.

Checked with scripts in the scratchpad `checks/`. The baseline is a `git archive` of origin/main (2bc8446), not a worktree.
- **Fixed:** narrow DTDs. The new mode 2 integrates the power law exactly per z bin, with SFRD held constant per bin. It agrees with an independent delay-time integral (zdm `H(z)`) to better than 0.3% for β=-1 and 0.01-14 Gyr, 10-20 Myr, 1 Myr-0.3 Gyr, β=1, and β=-1.5. It takes 0.02 s for nz=500, down from ~17 s, and follows `state.cosmo`.
- **Fixed:** `repeat_grid` (now uses `self.base_sfr`), z=0 normalisation of modes 2 and 3, the tmin>tmax check, the shared WD-WD curve, `else: raise` in `set_evolution`, and the unused import.
- **New blocker:** `grid.py:489-493`. The second assignment overwrites the first, so `(1+z)^-alpha` is lost when `alpha_method==1`, which is the default. For the default state on CRAFT_ICS_1300, `sum(rates)` is 7.95e-5 on main and 1.77e-4 on the PR, and the `sfr` ratio is exactly (1+z)^1.54. The 31 existing tests still pass, so nothing catches it.
- **New blocker:** `grid.py:470` reads `pl_alpha`, which was renamed to `pl_beta`, so a mode-2 Grid raises AttributeError.
- **Other issues:**
  - No `f_GC` clip, despite the reply (f_GC=1.5 gives n<0).
  - Mode 4 raises TypeError for scalar z.
  - Mode 4's docstring describes a CCSN clip and `k` values that aren't in the code, and `k_type` is unused.
  - Mode 5 (lognormal, new) uses Planck18 lookback times with zdm `H(z)` (3-5% shift for H0=73, Om=0.35, against ≤1% for mode 2). It takes 8.2 s for nz=500 and is about 1% off a reference at τ=1, σ=0.3.
  - `test_source_evolution.py` holds no pytest test.
  - The help-string pieces run together with no separators, and the help text and Notation are out of date.
- **Round-1 questions:** normalisation and `sfr_n` are answered. The Kremer+23 fit range and MCMC use are still open.

Note for later: `zdm/tests/test_source_evolution.py` imports matplotlib and writes `source_evolution.pdf` if run directly.

### 2026-11-11 (Task 4: posted the round-2 REQUEST_CHANGES review on PR #92)

Used the Round 2 answers to cut down the task-3 draft:
- the `f_GC` clip item was dropped;
- the lognormal-mode issues (mixed cosmology, speed, ~1% accuracy) were put under "Recommended (not blocking)";
- for tests, the review offers to write them on this branch rather than asking for them;
- the prompts file and the unrelated changes (`rmult`, `write_macquart_relation.py`) are not mentioned.

The PR head was still `4799de4` before posting, and there was no new activity since Clancy's 2026-11-06 replies.

Submitted as profxj with event REQUEST_CHANGES on commit `4799de4`. It has a summary and 11 inline comments:
https://github.com/FRBs/zdm/pull/92#pullrequestreview-5867115212

The summary has five parts:
- **Resolved:** narrow DTDs, `repeat_grid`, normalisation and the smaller items.
- **Two new blockers:** `grid.py:492` overwrites the `(1+z)^-alpha` factor, changing default rates by (1+z)^1.54; `grid.py:470` still reads `pl_alpha`, so mode 2 raises AttributeError.
- **Recommendations:** mode 5's cosmology mixing and speed, mode 4 failing for scalar z, and the docs.
- **Tests:** the offer to write them.
- **Open round-1 questions:** the Kremer+23 fit range, and whether the DTD parameters will be fitted in MCMC.

The inline comments are at:
- `grid.py`: 470, 492
- `cosmology.py`: 609, 842, 866, 884, 906, 943
- `parameters.py`: 150, 180
- `tests/test_source_evolution.py`: 10

All of these lines are inside the PR's diff hunks. The payload is in the session scratchpad as `review92_round2.json`.
