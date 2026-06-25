# Design — admixture rate through time (time-inhomogeneous directional mugration EM)

Status: design / plan (admix-dating branch). Builds on the tslai engine (§1–§4 of CLAUDE.md).

## 0. Goal

Make the ancestry CTMC **time-inhomogeneous** and estimate the cross-ancestry transition rate
as a **function of (backward) time** — directional `q_AB(t)`, `q_BA(t)` — by EM. Output: an
admixture / cross-coalescence **rate-through-time profile**, with the rate captured *as sharply
as the data support* (penalised splines, locally adaptive — see `explore/spline_resolution.*`).

Honest framing (settle empirically): the *vertical* mugration rate-through-time is a **relative
cross-ancestry coalescence-rate** quantity (à la MSMC cross-coalescence). It locates source
**divergence** and **gene-flow epochs**, and its directional asymmetry encodes asymmetric gene
flow. It is **not** a direct single-pulse-time estimator — that lives in the horizontal
(tract-length) signal we already recover. The two are complementary.

## 1. Model

- Time axis `t ≥ 0` = node age (generations backward).
- Time-inhomogeneous generator `Q(t)` (2-state v1; K-way by generator swap):
  `Q(t) = [[-q_AB(t), q_AB(t)], [q_BA(t), -q_BA(t)]]`.
- **Two discretisations, decoupled** (the key design choice from the spanning-branches analysis):
  - a **fine log-time grid** `{cells g}` for the E-step *accumulation* (geometric/"even-power"
    spacing — equal expected branch-occupation per cell);
  - a **smooth rate model** for the M-step: `q_AB(t)=exp(s_AB(log t))`, `s_AB` a penalised
    B-spline (bins are the coarse, piecewise-constant fallback).
- Branch transition under `Q(t)`: split the branch at cell boundaries; it is the time-ordered
  product `P_branch = P_1 P_2 … P_n`, `P_i = expm(Q_i·d_i)` (child→parent, `Q_i` the cell rate).

## 2. E-step — fine-grid time-resolved sufficient statistics

Custom pruning per marginal tree / per root using the **composite** per-branch transition, then
per branch accumulate endpoint-conditioned expected **dwell** and **directional jumps into each
cell**, edge-blocked and span-weighted (§3.3 blocking preserved).

Per branch (child `c@t_c`, parent `p@t_p`), sub-intervals `i=1..n` from child upward, cell `g(i)`:

- up-likelihood `L_c` (child subtree), outside message `u` (parent toward child) — from the pass.
- chain the transitions:
  - left vector `a_i = L_c · P_1 ⋯ P_{i-1}` (state at the **bottom** of sub-interval `i`; `a_1=L_c`);
  - right vector `b_i = P_{i+1} ⋯ P_n · u` (state at the **top** of sub-interval `i`; `b_n=u`);
  - normaliser `Z = L_c · P_branch · u`.
- reward in sub-interval `i` for reward-matrix `E` (Van Loan integral `R_i = ∫ P(τ)E P(d_i−τ)dτ`
  = `branch_stats.vanloan_integral(Q_i, d_i, E)`):
  `reward_i = (a_i · R_i · b_i) / Z`.
  - dwell in `m`:  `E = e_m e_mᵀ`  → accumulate into `D_m(g(i))`.
  - jumps `m→n`:  `E = q_mn^{(i)} e_m e_nᵀ` → accumulate into `J_{mn}(g(i))`.
- multiply by edge span `w = (right − left)`; bank **once on edge entry** (`edge_diffs`).

Cost: `O(Σ_branches (#cells spanned) · K²)`; deep branches span many cells but are few. Reuse
`branch_stats.vanloan_integral` and the `accumulate.py` edge-blocked loop pattern.

Inherit the tskit edge cases verbatim (§4): per-root forests, isolated samples = missing-info
(no contribution), **skip root branches**, polytomies (product over children), full-span coverage.

**Sanity invariant (unit test):** with `Q(t)` constant, the per-cell `D_m(g)`, `J_{mn}(g)` summed
over the cells a branch spans must equal `branch_expected_stats(Q, t_p−t_c, ξ)` (the homogeneous
whole-branch totals). This validates the split.

## 3. M-step — directional penalised-Poisson spline

The accumulated `{D_A(g), D_B(g), J_AB(g), J_BA(g)}` are an inhomogeneous-Poisson dataset:

- `q_AB(t)=exp(s_AB(log t))` fit by **penalised Poisson regression** — `J_AB(g) ~ Poisson(q_AB(g)·D_A(g))`,
  log-link with **offset `log D_A(g)`**, smooth `s_AB` (B-spline on `log t`), 2nd-difference
  roughness penalty, smoothing parameter by **REML/GCV** (auto-balances fit vs smoothness → sharp
  where powered, smooth where not). Likewise `q_BA(t)` from `J_BA`, `D_B`.
- Restrict to the **informative window** (`D > floor`); the data-poor extremes are unidentifiable.
- **Coarse fallback** = piecewise-constant bins: `q_AB(bin)=Σ J_AB / Σ D_A`.

(Per the exploration: penalties don't blur a well-powered abrupt feature, because the
exposure-weighted data term dominates where events are dense; REML handles the sparse tail.)

## 4. EM loop

- Init `q_AB(t)=q_BA(t)=const` from the homogeneous `tslai.fit`.
- Iterate: E-step (fine-grid stats under current `Q(t)`) → M-step (refit the two splines) until the
  rate profile / observed-data log-likelihood converges.
- `π` held uniform (`estimate_pi=False`, §6 — avoids the π degeneracy); credibility `w` as in the
  homogeneous model (or fixed for v1). Smoothing parameter: re-select by REML each M-step, or fix
  after a burn-in for stability (decide empirically).

## 5. Identifiability, power, calibration

- Rate identifiable only where occupation `D(t) > 0` **and** there is transition information;
  restrict/penalise the extremes.
- **E-step resolution floor:** endpoint-conditioned rewards spread each branch over its span, so the
  *E-step itself* smooths at ≈ the branch-length scale — this, not the spline, likely sets the
  achievable sharpness. Measure it (recover-a-step resolution).
- **Time calibration (§6):** the profile lives on the ARG's node-age axis; mis-calibrated times
  distort it. True ARG is exact on sims; for real data prefer **SINGER** (calibrated times).
- **Mugration bias (§6):** a proxy for the structured coalescent — interpret as *relative* cross-
  ancestry rate, not literal demographic migration rate.

## 6. Validation plan (known truth)

- **Clean split** → expect `q(t)≈0` below `T_split`, sharp rise at `T_split` (the exploration shows
  this is recoverable sharply). **Split + gene-flow pulse** → bump at the pulse time. **Ongoing
  migration** → plateau. Vary `T_split`, pulse time/strength, and A↔B **asymmetry**.
- Check the fitted `q_AB(t)`, `q_BA(t)` recover the known features at the right times + the right
  directional asymmetry — on the **true ARG** first, then tsinfer / SINGER.
- **Oracle check:** compare the EM profile to the model-free pairwise cross-coalescence hazard
  (`explore/spline_resolution.py` extraction) — does the mugration-EM recover the same curve?
- **Resolution:** smallest separable feature vs the E-step floor.

## 7. Module layout (admix-dating branch)

```
src/tslai/dating/
  __init__.py        # public: fit_rate_through_time, RateThroughTime
  grid.py            # log-time fine grid + cell/branch-split helpers
  estep.py           # time-inhomogeneous pruning + per-cell endpoint-conditioned dwell/jumps
  mstep.py           # directional penalised-Poisson-spline fit (+ bins fallback, REML/GCV)
  em.py              # the EM loop
  profile.py         # RateThroughTime result (t, q_AB, q_BA, CIs) + plot
explore/             # known-scenario recovery experiments (extends spline_resolution.py)
tests/test_dating_*  # unit + integration
```

## 8. Build order (checkable rungs)

1. `grid` + the **per-branch per-cell endpoint-conditioned stats** (`estep` core) + the §2
   sum-invariant unit test (per-cell totals = homogeneous whole-branch totals).
2. **Stage-1 shortcut for a fast first signal:** homogeneous E-step (reuse `tslai` pruning ξ) but
   *bin the per-branch rewards by time* → a first rate-through-time profile from one `tslai.fit`.
   Validate it shows the `T_split` feature. (No new pruning yet — quickest go/no-go.)
3. `mstep` directional penalised spline + test (recovers a known rate from synthetic events/exposure
   — the exploration's machinery, promoted).
4. Full **time-inhomogeneous E-step** pruning (composite per-branch `P`) + `em` loop; validate
   clean-split recovery (true ARG).
5. Directional asymmetry + pulse + ongoing-migration scenarios; oracle comparison.
6. Calibration front ends (tsinfer/SINGER); resolution-floor measurement; write up.

## 9. Risks / open questions

- **Cost** of the inhomogeneous E-step (per-branch cell splitting). Mitigate: vectorise, coarser
  grid where occupation is low, `expm` caching per (cell-Q, duration).
- **E-step smoothing floor** may dominate the achievable sharpness (measure early — rung 4/6).
- **EM × smoothing-parameter** interaction (re-select λ each step vs fix) — watch convergence.
- **Interpretation** — keep the "relative cross-coalescence rate, not pulse time" framing honest.
- **Scope creep** — this is a *new estimator* riding the same engine; keep it a separate subpackage
  (`tslai.dating`), not entangled with the LAI painter.
```

---

## Findings (rungs 1–5, MEASURED)

- **Rung 1** — per-cell endpoint-conditioned E-step is exact (sum-invariant vs whole-branch
  `branch_expected_stats`).
- **Rung 2** — Stage-1 binned profile (homogeneous E-step) shows the `T_split` feature but
  **smears the onset early** (the resolution floor).
- **Rung 3** — directional penalised-Poisson spline M-step recovers a known step rate.
- **Rung 4** — the full time-inhomogeneous EM **sharpens the onset to ≈`T_split`** (1852 vs the
  Stage-1 1002), log-likelihood monotone. **Deep-tail instability is real** (rate diverges ~1e9
  in data-poor deep cells); fixed by restricting the M-step to the informative occupation window.
- **Rung 5** — the profile **resolves demographic structure**: a gene-flow **pulse** appears as a
  bump at `T_pulse`; **asymmetric** flow gives `q_AB ≠ q_BA`; **ongoing** migration gives an
  elevated recent rate. The clean-split case is onset-only.

### Direction convention (important)

tslai's jumps are **parent→child = old→young = forward in time**, so a *backward-time* A→B
mass-migration registers as **forward-time B→A** — i.e. it shows up in `q_BA`, not `q_AB`. State
the profile's directionality in forward time, or flip when reporting admixture (backward-time)
direction.

### Open / caveats

- Profiles are noisy at small sample/sequence size and at the recent time boundary; the rate is
  only trustworthy in the informative occupation window.
- It estimates a *relative cross-coalescence rate* (divergence + gene-flow epochs), not literal
  migration rates (mugration bias, §5).
- Time-calibration dependence (§5): use SINGER's calibrated times on real data (rung 6).
