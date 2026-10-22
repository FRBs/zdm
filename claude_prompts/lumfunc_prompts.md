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

### Claude — 2026-09-09: response to Clancy's request (category-by-category status, Category 3 proposal, and commit-vs-branch suggestion)

Most of the checking Clancy asks for was done as part of [my PR #89 review submitted earlier today](https://github.com/FRBs/zdm/pull/89#pullrequestreview) — here is the explicit category-by-category accounting he requested, plus the Category 3 design and my suggestion on his workflow question.

**Category 1 (iteration.py regressions) — confirmed fixed, with two fresh casualties of the rewind.** Explicit checklist against the #87 review: `minimise_const_only2` present and called ✔; the PATH machinery (`get_joint_path_zdm_likelihoods`, `get_PATH_lls`, `sum_path_lls`, `construct_popt`, the `PATH=` kwarg) restored ✔; the mangled pvals/llsum merge block gone ✔; ptauw `dkis2` coefficients restored ✔; `get_rates()`/`get_dm_bias()` masking restored ✔; negative-DMEG penalty behavior back to main's ✔; `lEmax=41.84` default restored ✔; `Grid.update()` stays deleted ✔. **However**, the rewind/conflict commits introduced two *new* NameErrors of the same flavor they were fixing: `tz_tomult` (iteration.py:750, defined as `zt_tomult` at 731 — crashes any `--ptauw` run) and `bweights`/`wweights` (iteration.py:1721-1743, main used `zbweights`/`zwweights` — crashes the `--Pn --pwb` config both shipped slurm scripts run). Both are one-line renames; both are in my #89 review's blocking list.

**Category 2 (new LF code) — fixed vs outstanding, item by item:**

| #87 item | Status |
|---|---|
| igamma spline evaluated outside its domain under shipped priors | **Partially fixed.** `SplineMin` −6→−9 now covers `break_schechter.json` *exactly at the boundary* (lEb=36, lEmax=45 → x=1e-9). Still no domain guard, and `np.clip(result,0,1)` masks extrapolation garbage for any wider prior. Residual ask: validate `x >= 10**SplineMin` and re-tabulate via the existing `reinit=True` machinery. (Top-end extrapolation beyond 1e6 was tested and decays benignly to 0.) |
| `energetics.reset()` rebuilding the spline every posterior evaluation | **Outstanding.** Measured ~0.24 s per rebuild (now 1500 mpmath calls after NSpline 1000→1500) — ~CPU-hours per MCMC run even when the relevant gamma is fixed. Cache keyed on gamma, spared by `reset()` when unchanged. |
| `grid_kwargs` dead code | **Fixed** when `g0info` is passed (`**grid_kwargs` now reaches `get_zdm_grid`). **New gap:** when `g0info is None`, the caller's `nz/ndm/zmax/dmmax` arguments are silently ignored (grid_kwargs stays `{}`), so those signature parameters are dead. |
| Constraint enforcement split (ValueError deep in energetics vs MCMC prior check) | **Outstanding**, and the protective `try/except` in `calc_log_posterior` is still commented out (MCMC.py:230) — a slice/cube scan crossing `lEb` dies with an uncaught ValueError instead of scoring −inf. |
| `lEmax` default 41.84→43.0 | **Fixed** (41.84 restored). |
| `Grid.update()` reintroduced | **Fixed** (stays deleted). |

If some of these are "deemed features," the two I'd most want an explicit *intended* from Clancy on: the `SplineMin`/`NSpline` knot changes (they shift LF-2 likelihoods at the bit level vs main, breaking exact reproducibility of prior fits) and `run_slice.py`'s `lEmin: 30.0` with `38.394...` commented out on the same line.

**Category 3 (structural) — concrete proposal.** One registry in `energetics.py` that owns everything per-LF, consumed by all four current dispatch sites:

```python
# energetics.py
@dataclass(frozen=True)
class LFModel:
    name: str
    array_cum: Callable;  vector_cum: Callable
    array_diff: Callable; vector_diff: Callable
    extra_params: tuple[str, ...] = ()   # state.energy attrs beyond (Emin, Emax, gamma), e.g. ('gamma2', 'lEb')
    log_params: tuple[str, ...] = ()     # subset stored as log10, converted with 10**, e.g. ('lEb',)
    constraint: Callable | None = None   # params dict -> bool, e.g. lambda p: p['lEmin'] < p['lEb'] < p['lEmax']
    soft_cutoff: bool = False            # replaces the magic [1, 2, 6] Emax_boost list

LF_REGISTRY: dict[int, LFModel] = {
    0: LFModel('power_law', array_cum_power_law, vector_cum_power_law, ...),
    # 1-3 gamma variants ...
    4: LFModel('broken_power_law', ..., extra_params=('gamma2','lEb'), log_params=('lEb',),
               constraint=lambda p: p['lEmin'] < p['lEb'] < p['lEmax']),
    5: LFModel('double_broken_power_law', ..., extra_params=('gamma2','gamma3','lEb','lEb2'),
               log_params=('lEb','lEb2'),
               constraint=lambda p: p['lEmin'] < p['lEb'] < p['lEb2'] < p['lEmax']),
    6: LFModel('broken_schechter', ..., extra_params=('gamma2','lEb'), log_params=('lEb',),
               constraint=lambda p: p['lEmin'] < p['lEb'] < p['lEmax'], soft_cutoff=True),
}

def lf_params(state, Emin, Emax, gamma):
    m = LF_REGISTRY[state.energy.luminosity_function]
    extras = tuple((10**v if k in m.log_params else v)
                   for k in m.extra_params for v in [getattr(state.energy, k)])
    return m, (Emin, Emax, gamma) + extras
```

The four consumers collapse to one-liners: **`grid.init_luminosity_functions`** — the 7-branch elif ladder plus twelve wrapper methods become one generic closure per slot (`self.array_cum_lf = lambda Eth, Emin, Emax, g, *_: m.array_cum(Eth, *params)`), ~85 lines → ~10; **`iteration.ConvertToMeaningfulConstant`** — `m, p = lf_params(state, Emin, Emax, gamma); factor = float(m.vector_cum(np.array([Eref]), *p)[0])`, and the final `else` raises `KeyError` on unknown codes instead of silently computing the gamma-function constant (this also fixes the shape-(1,) ndarray and lets the LF-0 normalization inconsistency be addressed in one place); **`MCMC.valid_parameter_combination`** — `return m.constraint(merged) if m.constraint else True`, no hard-coded `(4, 6)`/`5`; **`grid.GenMCSample`** — `Emax_boost = 3.0 if m.soft_cutoff else 0.0`, no magic list. Fold in at the same time: one module-level `_powerlaw_integral(ratio, gamma)` with a single `np.isclose` zero-test (replacing the seven inconsistent copies), one shared flatten/compute/reshape wrapper for the six `array_*` functions, and named constants (`LF_BROKEN_SCHECHTER = 6`) for the remaining literals including the JSON configs' documentation. Net effect ≈ −200 lines, behavior-identical, and `test_energetics.py` is a genuine oracle for it (its expected values are derived independently of the implementation).

**Category 4 (minor) — recheck:** the rewind fixed three of five: the psnr nested Python loop is gone, the 1D/2D `bEobs` vs `bEths` inconsistency is gone (both use `bEths` now), and the commented `np.save` with undefined variables is gone. Still open: **BLAS thread capping regressed** — the old `os.environ["OMP_NUM_THREADS"]="1"` line was deleted and replaced by a comment *claiming* the capping (MCMC.py:459-460) with no code behind it (set the env vars in the launch scripts or use `threadpoolctl` post-import); and the `.flatten()` (11×) vs `.ravel()` (2×) inconsistency in the energetics array wrappers remains (cosmetic unless arrays are non-contiguous — the shared wrapper from Category 3 removes it).

**New issues:** all in the #89 review — the three crashes above plus `np.empty_like` leaving uninitialized memory for NaN thresholds in LF 4/5/6 (use `np.full_like(Eth, np.nan)`), the stale `papers/FitRepetition2025/slurm/run_mcmc.slurm` (nonexistent `DSA_34`/`params2.json`), the `nthreads=None` default (crashes at `nthreads < 1`, and once fixed silently changes the non-Slurm default from 1 worker to `os.cpu_count()`), the `ConvertToMeaningfulConstant` addendum, and the test-coverage gap (all new tests pass while the flagship `--Pn --pwb`/`--ptauw` configs crash — two smoke tests proposed in the review).

**Suggestion on Clancy's workflow question (commit vs branch):** have Claude implement Category 3, but as a **separate branch and PR — not a commit inside #89**. Reasons: (1) #89 should stay a reviewable bug-fix PR; the refactor is ~−200 lines across four files and deserves its own diff where "behavior-identical" is checkable at a glance; (2) the refactor conflicts textually with the #89 blocking fixes (same functions), so sequencing matters — land the one-line crash fixes in #89 first, then cut the refactor branch from the updated `NewQinLF` (e.g. `lf_registry`), PR it into `NewQinLF` (or `main`, if #89 merges first); (3) a standalone PR can carry the two smoke tests and be validated by the unchanged test suite. Mechanics: I can implement it in this working tree whenever you say the word — under this repo's conventions you'd do the `git checkout -b lf_registry`, commit, and push, and I'll write the code and verify tests before/after; alternatively Clancy can run the same request on his side, or I can hand over a `git format-patch`-style patch file. Effort is small (one session). I'd also suggest Clancy replies to the #89 review items with intended/won't-fix so the next review pass doesn't re-litigate the "deemed features."


## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-09-09 (Answered Clancy's category-by-category request in Q&A; proposed the LF-registry design and a branch-based workflow)

Executed prompt 2: the 2nd task under "Pull Request". Read Clancy's message and wrote the response into the Q&A section: (Category 1) explicitly confirmed every #87 iteration.py regression is fixed by the rewind — with the caveat that the rewind itself introduced two new NameErrors (`tz_tomult`, `bweights`/`wweights`), both already in the #89 blocking list; (Category 2) built a fixed-vs-outstanding table — fixed: lEmax default, Grid.update() deletion, grid_kwargs wiring (when g0info given); partially fixed: spline domain (SplineMin now covers the shipped priors exactly at the 1e-9 boundary but still no guard and np.clip masks garbage); outstanding: energetics.reset() per-evaluation spline rebuild, constraint split with the try/except still commented out, plus a new gap (nz/ndm/zmax/dmmax dead when g0info is None); flagged the SplineMin/NSpline knot change and run_slice lEmin=30.0 as the two "deemed features?" needing an explicit intended from Clancy; (Category 3) wrote a concrete registry design — a frozen `LFModel` dataclass + `LF_REGISTRY` dict in energetics owning functions, extra/log params, ordering constraint, and soft_cutoff, with one-line consumers replacing the four dispatch sites, plus folding the seven integral-helper copies and six array wrappers — net ≈ −200 lines, behavior-identical, oracled by test_energetics; (Category 4) recheck via fresh greps: three of five minors fixed by the rewind (psnr Python loop, bEobs/bEths, commented np.save), two open (BLAS thread-capping is now comment-only after the old OMP_NUM_THREADS line was deleted; flatten/ravel inconsistency 11 vs 2); (new issues) pointed to the #89 review list. Workflow suggestion to Xavier/Clancy: implement Category 3 as a separate branch (`lf_registry`) and PR — not a commit inside #89 — sequenced after #89's one-line crash fixes land, carrying the two proposed smoke tests; offered to write the code in this working tree with the user doing the git operations, per repo convention. Learned: the rewind quietly resolved three #87 "minor" items for free, but also deleted the one line of real BLAS capping while keeping a comment that claims it — comments asserting behavior are worth grepping for their implementation. Note: the prompt asked for Opus 5; this session runs on Fable 5 and cannot switch models mid-session.

### 2026-09-09 (Reviewed PR #89 (NewQinLF → main) and submitted a request-changes review as profxj)

Executed prompt 1: the 1st task under "Pull Request". Found the open PR ([FRBs/zdm#89](https://github.com/FRBs/zdm/pull/89), `NewQinLF` → `main`, by cwjames1983) — the successor to closed PR #87, re-merged from main. First verified the #87 findings were addressed: `minimise_const_only2`, the PATH machinery, ptauw `dkis2` coefficients, `get_rates()` masking, and `lEmax=41.84` all restored; `Grid.update()` stays deleted; `grid_kwargs` now passed to `get_zdm_grid`; MeerTRAP Unicode-minus and the scripts-copy `DSA_34` reference fixed. Ran the multi-agent code-review scan (the forked orchestrator also did a full line-by-line pass when its finders were slow to schedule, then six finders reported independently — strong cross-agent consensus) and spot-checked every blocking claim by hand. Submitted CHANGES_REQUESTED with 5 blocking items, all small fixes that crash the PR's own shipped configurations: (1) half-applied rename `zt_tomult`→`tz_tomult` (NameError on any `--ptauw` run); (2) `bweights`/`wweights` undefined in `calc_likelihoods_2D` — main used `zbweights`/`zwweights` (NameError on the `--Pn --pwb` config both slurm scripts run); (3) `if nthreads < 1:` before the None-check with new default `nthreads=None` (TypeError on every default invocation; plus the thread-capping comment has no code behind it); (4) `np.empty_like` + mask-fill leaves uninitialized memory for NaN thresholds in LF 4/5/6; (5) the new papers/FitRepetition2025 slurm script is a stale pre-fix copy referencing nonexistent `DSA_34`/`params2.json`. Silent-behavior items: `nz/ndm/zmax/dmmax` ignored when `g0info is None`; `run_slice.py` lEmin 38.39→30.0 with the old value commented out; `SplineMin`/`NSpline` knot changes break bit-level reproducibility and add ~70% spline-build cost; the broken-Schechter spline still has no domain guard (priors sit exactly at the 1e-9 boundary, `np.clip` masks extrapolation); `energetics.reset()` still rebuilds the spline every posterior evaluation (~0.24 s measured → CPU-hours per run). Structural #87 carryovers noted as follow-up material (LF registry, 7 copies of the integral helper with inconsistent gamma≈0 tests, constraint split with the try/except still commented out, drifting JSON configs, the copy-pasted paper script with deprecated `pkg_resources` under the new python_requires>=3.12). Suggested two smoke tests that would have caught all three crashes (none of the new tests exercise ptauw/pwb/default-nthreads). No questions for the user were needed, so the Q&A section stays empty. Learned: the new tests passing while the flagship config crashes is a coverage gap, not a safety signal. Note: the prompt asked for Opus 5; this session runs on Fable 5 and cannot switch models mid-session. Addendum: a late-arriving cross-file check surfaced one further verified item — `ConvertToMeaningfulConstant`'s new LF 4/5/6 branches return a shape-(1,) ndarray and a normalized survival fraction where the LF 0 branch returns an unnormalized float (both quirks inherited from the pre-existing else branch) — posted as a follow-up comment on the PR.
