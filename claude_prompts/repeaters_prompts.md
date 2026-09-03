# repeaters prompts

## Goals

Work on the `repeaters` PR.

## Prompts

1. Read this file.  Execute the 1st task under "Pull Request"
2. Read this file.  Execute the 2nd task under "Pull Request"

## Pull Request

1. There is an open pull request for the `repeaters` branch.  Please review the new code and submit a review as `profxj`.  Use Fable if you can.  Log your work.

2. The CI tests are failing on GitHub. Please check them and discuss them with me in the Q&A section below.  Use Opus if you can.  Log your work.

## Q&A

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-08-26 (Reviewed PR #83 (repeaters → main) and submitted a review as profxj)

Executed prompt 1: the 1st task under "Pull Request". Found the open PR ([FRBs/zdm#83](https://github.com/FRBs/zdm/pull/83), `repeaters` → `main`, by cwjames1983, "Pulling in Jordan's latest modifications for fitting the repeaters distribution"). Key discovery: the PR head (d1fe7d7) is a merge of `zdmp` into `repeaters`, so GitHub's 83-file/+9565 diff against `main` mostly repeats PR #82; the code unique to `repeaters` is `git diff origin/zdmp...repeaters` — 12 files, +293/−283 (MCMC.py, iteration.py, survey.py, run_slice.py, two slurm scripts, four survey .ecsv files, a notebook, one paper script). Cloned the repo into the session scratchpad (`--filter=blob:none`) and checked out `repeaters` there rather than switching branches in the working tree, then ran the code-review skill scoped to that delta and spot-checked every finding against the branch (read `init_zs`, recomputed the DSA threshold with astropy — 405.6 pc/cm³ from 20221027A, 25 of 40 localised FRBs lose z; confirmed `PENALTY = 100` vs finite log10 pmf down to ≈ −308; confirmed `DSA_34.ecsv`/`params2.json` don't exist; confirmed the profiler wrapper has no call sites; confirmed MeerTRAPcoherent lost `TOBS` so it drops out of the Pn term). Submitted a COMMENT review under profxj with 4 fix-before-merge items (CHIME `Z = -0.064` sign typo on repeater 20190303A; slurm script referencing nonexistent survey/params files; `plot_slice` nested inside the mkdir guard; hard-coded `vals2 = [39.0]` forcing lEmin), 3 confirm-intent questions (DSA −1/−2 encoding, MeerTRAP TOBS removal, non-monotonic Poisson underflow penalty), 4 minor items, and credit for three genuine fixes (`os`→`obs` un-shadows the `os` module; `lp += -PENALTY`; duplicate `calc_max_dm()` removed). Chose COMMENT rather than REQUEST_CHANGES this time since the branch is a research-iteration branch and the user approved #82 after my request-changes; offered to re-post as request-changes. Also noted the branch is behind `zdmp` (merge-base 6bac170 vs bccffb4), so its CI will keep failing at pip install until zdmp is merged again. Note: the prompt asked for Fable — this session is Fable.
