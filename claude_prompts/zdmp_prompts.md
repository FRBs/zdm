# zdmp prompts

## Goals

## Prompts

1. Read this file.  Execute the 1st task under "Pull Request"
2. Read this file.  Execute the 2nd task under "Pull Request"

## Pull Request

1. There is an open pull request for the `zdmp` branch.  Please review the new code and submit a review as `profxj`.  Use Fable if you can.  Log your work.

2. The CI tests are failing on GitHub. Please check them and discuss them with me in the Q&A section below.  Use Opus if you can.  Log your work.

3. I have answered your questions in the Q&A section below.  Please review them and proceed.  Use Opus if you can.  Log your work.  Also draft the request for Clancy for me and add it to the Q&A section below.

## Q&A

### Claude — 2026-08-25: why CI is red on PR #82 (zdmp)

**Short version:** CI never reaches the tests — pip refuses to install zdm's dependencies. A second, unrelated failure is waiting behind that one. And `main` is broken the same way, so this is not something the `zdmp` branch introduced.

**1. What the CI run actually says**

- Latest run [32819583399](https://github.com/FRBs/zdm/actions/runs/32819583399) on a6e5b81: all 6 matrix jobs red. Only two (3.11- and 3.12-test-alldeps) actually failed; the other four were fail-fast *canceled* and never ran anything.
- The failing step is tox installing zdm's own `install_requires` into its isolated venv:
  ```
  The conflict is caused by:
      The user requested ne2001 0.0.2 (from git+https://github.com/FRBs/ne2001)
      frb 0.1.dev0 depends on ne2001 0.0.2 (from git+https://github.com/FRBs/ne2001.git)
  ```
  pip treats two direct-URL requirements for the same package as a conflict unless the URL strings are byte-identical. `.git` vs no `.git` is enough.
- Why now: FRB moved to `pyproject.toml` on 2026-04-20 (FRBs/FRB@26648b82) and declares `ne2001 @ git+https://github.com/FRBs/ne2001.git`. zdm `main`'s last green run was 2026-04-14; `main`'s run today (7bc6d8e) fails with the identical error.
- Second trap behind the first: FRB also declares `astropath @ git+https://github.com/FRBs/astropath.git`, while zdmp's setup.cfg adds `astropath @ git+https://github.com/FRBs/astropath` (no `.git`). Fix ne2001 alone and pip fails on astropath next.
- Side note: the workflow's "Install zdm requirements" step pip-installs ne2001/frb/astropath into the runner's *system* Python, but tox builds an isolated venv and re-clones everything. That step is ~1 min of CI time that has no effect on the outcome.

**2. What fails once install works** (reproduced locally in the `astro` env, Python 3.13: `8 passed, 1 error in 90 s`)

- `zdm/tests/test_path_prior.py` errors at *collection*: line 85 calls `test_path_priors()` at module import, which constructs `opt.loudas_model()`, whose default data file is `p_mr_distributions_dz0.01_z_in_0_2.0.h5`. That file is not in the repo — only `..._z_in_0_1.2.h5` is tracked — and `*.h5` is in `.gitignore` (line 23), so `git add` silently skipped it when commit 4fa232f changed the default from 1.2 → 2.0. It isn't anywhere on this machine either; presumably Clancy has it.
- Because it is a collection error, pytest aborts the whole session (`Interrupted: 1 error during collection`) — zero tests run. So CI stays red after the pip fix.
- The 8 tests that do run all pass on 3.13. (CI's matrix is 3.11/3.12 + astropy-dev. FRB now requires numpy>=2.2, astropy>=7.1, scipy>=1.17, so the effective floor is well above zdm's declared minimums — worth knowing, not a failure.)
- Correction to my PR review: I flagged `test_scat_methods.py:110` as calling the deleted `survey.geometric_lognormals` — that line is commented out. False positive; I posted a correction comment on the PR.

**3. Proposed fixes (smallest first)**

- (a) **setup.cfg**: `ne2001 @ git+https://github.com/FRBs/ne2001.git`, `astropath @ git+https://github.com/FRBs/astropath.git` (and `frb @ ...FRB.git` for consistency), mirroring FRB's strings exactly. Two-line fix, unblocks immediately; fragile, since it breaks again whenever FRB changes its URL string. Alternative: drop the git deps from `install_requires` and put them in `tox.ini` `deps =`, so pip only ever sees FRB's URL. More robust, but changes what `pip install zdm` does for users.
- (b) **test_path_prior**: either `git add -f` the 2.0 `.h5` (the 1.2 file is 457 KB, so size is not a concern) or `pytest.skip` when the file is absent — and remove the module-level `test_path_priors()` call. Same pattern in `test_scat_methods.py:153`; pytest collects `test_*` functions itself, and module-level calls run the full computation even under `--collect-only`.
- (c) Fix (a) also needs to land on `main`, which is red for the same reason.

**Questions for you**

- **Q1.** For (a): match FRB's URLs exactly, or move the git deps out of `install_requires`? I lean toward matching URLs now and separately deciding whether zdm should be coupled to FRB's URL choices at all.
>A. Ok, match URLs for now.
- **Q2.** The 2.0 `.h5` file: ask Clancy to force-add it, or make the test skip when it's missing? The `loudas_model` default pointing at an untracked file breaks *users*, not just CI, so I'd argue for committing the data (or reverting the default to 1.2).
>A. I will ask Clancy to force-add the file.
- **Q3.** Should I apply (a) + (b) on this branch now for you to commit, or do you want it to go via `main` first?
>A. This branch
- **Q4.** Remove the ineffective "Install zdm requirements" workflow step while we're in there?
>A. Yes, remove them

### Claude — 2026-08-26: draft request for Clancy

> Hi Clancy,
>
> Two quick things on the `zdmp` PR (#82), both about getting CI green:
>
> 1. **Missing data file.** `optical.loudas_model()` now defaults to `p_mr_distributions_dz0.01_z_in_0_2.0.h5`, but that file never made it into the repo — `*.h5` is in `.gitignore`, so `git add` skipped it silently when the default changed from the 1.2 file (commit 4fa232f). Only `..._z_in_0_1.2.h5` is tracked. Could you force-add it and push to `zdmp`?
>
>    ```
>    git add -f zdm/data/optical/p_mr_distributions_dz0.01_z_in_0_2.0.h5
>    git commit -m "Add z<2.0 p(m_r) distributions used by loudas_model"
>    git push origin zdmp
>    ```
>
>    Without it, `test_path_prior.py` errors on every machine that isn't yours, and pytest aborts the whole run. If you'd rather not fight `.gitignore` each time, adding `!zdm/data/optical/*.h5` below the `*.h5` line would whitelist that directory.
>
> 2. **Heads-up on the pip conflict.** CI (on `main` too) has been failing since April at the install step: FRB's `pyproject.toml` now requires `ne2001 @ git+https://github.com/FRBs/ne2001.git`, and pip treats that as conflicting with our `...ne2001` (no `.git`). I've pushed a fix to `zdmp` that matches FRB's URL strings exactly for `ne2001`, `frb`, and `astropath`, removed the CI step that installed those into the system Python (tox's venv never saw it), and moved the module-level `test_*()` calls in `test_path_prior.py` and `test_scat_methods.py` behind `if __name__ == "__main__":` guards so pytest doesn't run them at import.
>
> Once your `.h5` lands, CI should be green and we can work through the review comments.
>
> Thanks,
> Xavier

*(Draft assumes you will have committed and pushed the four fixes on this branch before sending; adjust "I've pushed" if not.)*


## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-08-26 (Applied the CI fixes on zdmp per Q&A answers; drafted the request for Clancy)

Executed prompt 3: the 3rd task under "Pull Request". Read the user's answers to Q1–Q4 and applied them on the `zdmp` branch (uncommitted, for the user to commit): (a) `setup.cfg` — `ne2001`, `frb`, and `astropath` direct-URL requirements now end in `.git`, byte-identical to FRB's `pyproject.toml` strings, which removes pip's `ResolutionImpossible`; (Q4) `.github/workflows/ci_tests.yml` — removed the "Install zdm requirements" step, which installed into the runner's system Python that tox's isolated venv never sees; (b) `zdm/tests/test_path_prior.py` and `test_scat_methods.py` — moved the module-level `test_*()` calls behind `if __name__ == "__main__":` guards so pytest no longer executes them at import/collection. Did not add a skip for the missing `.h5`, per Q2 (Clancy will force-add the file). Verification: `configparser` and `yaml` both parse the edited files; a `pip install --dry-run .` in a fresh venv (astro Python 3.13) resolves the full dependency set with exit 0 — the exact step CI fails on; local pytest went from "Interrupted: 1 error during collection, 0 tests run" to `8 passed, 1 failed` (the failure is `test_path_priors` on the missing `p_mr_distributions_dz0.01_z_in_0_2.0.h5`, expected until Clancy's file lands). Drafted the request to Clancy (force-add command, `.gitignore` whitelist suggestion, summary of the fixes) and added it to the Q&A section. Learned: pytest's collection-error abort was masking the fact that the rest of the suite passes; the `__main__` guard alone turns a red-everything run into a single attributable failure. Note: the prompt asked for Opus; this session runs on Fable and cannot switch models mid-session.

### 2026-08-25 (Diagnosed the failing CI on PR #82 and wrote up a Q&A discussion)

Executed prompt 2: the 2nd task under "Pull Request". Pulled the CI status for PR #82 (`gh pr checks`, run 32819583399 on a6e5b81): all six matrix jobs red, but only the two `test-alldeps` jobs actually failed — the rest were fail-fast canceled. Downloaded the failed job logs via the Actions REST API (`gh run view --log` returned nothing) and found tox never reaches pytest: pip `ResolutionImpossible` because zdm's setup.cfg requests `ne2001 @ git+https://github.com/FRBs/ne2001` while FRB (since its 2026-04-20 move to pyproject.toml) requires `ne2001 @ git+https://github.com/FRBs/ne2001.git`; pip treats non-identical direct-URL strings as conflicting. Confirmed `main` fails identically as of 2026-08-25 (last green 2026-04-14), so this is upstream drift, not the branch. Found the same `.git` mismatch for `astropath`, which will fail next. Reproduced the suite locally in the astro env (Python 3.13): 8 passed, 1 collection error — `test_path_prior.py` runs `test_path_priors()` at import and needs `p_mr_distributions_dz0.01_z_in_0_2.0.h5`, which is untracked because `*.h5` is gitignored (commit 4fa232f changed the default filename without the data). Wrote the analysis, proposed fixes, and four questions into the Q&A section of this file. Learned: my PR review's `test_scat_methods.py:110` item was a false positive (the call is commented out); posted a correction comment on the PR. Also noted the workflow's "Install zdm requirements" step installs into the system Python, which tox's isolated venv never sees. Note: the prompt asked for Opus; this session runs on Fable and cannot switch models mid-session.

### 2026-08-11 (Reviewed PR #82 (zdmp → main) and submitted a request-changes review as profxj)

Executed prompt 1: the 1st task under "Pull Request". Found the open PR ([FRBs/zdm#82](https://github.com/FRBs/zdm/pull/82), `zdmp` → `main`, by cwjames1983) and reviewed the full diff using an 8-agent code-review scan (angles: core grids, survey/iteration, optical/repeat/MCMC, cross-file tracing, removed-behavior audit, conventions, reuse/simplification, efficiency), then manually spot-checked the highest-impact claims against the branch before publishing. Submitted a CHANGES_REQUESTED review under the profxj account (gh CLI was already authenticated as profxj) with ~20 blocking correctness findings, 7 stale references to code deleted in the PR, 4 silent behavior changes, plus performance and duplication notes. Highlights: `Grid.update()`'s DMhalo branch calls `get_efficiency_from_wlist` with the old signature (TypeError in MCMC); `calc_pdv` accumulates `+=` into `b_fractions`/`w_fractions` without zeroing, so rates roughly double after every `grid.update()`; `calc_rates` compounds `sfr *= fz` on repeated calls; `lRmin`/`lRmax` are swapped in both HoffmannRepeaters26 presets and the case string at states.py:66 never matches (CHIME scattering silently overwrites the fitted parameters); `calc_likelihoods_2D` contains a duplicated ~90-line likelihood block whose stray recomputation overwrites the MW-uncertainty-weighted values; the console-script entry points (`zdm_build_cube`, `zdm_pzdm`) point at deleted/nonexistent files. Learned: the session ran on Fable per the prompt's request; verification-by-spot-check caught no false positives among the checked items, and the review notes it was prepared with Claude Code assistance.