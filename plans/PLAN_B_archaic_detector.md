# Plan B — Reference-free archaic detector (separate branch)

Promote the **depth** signal from Plan A — used there as a *diagnostic flag* on top of the
supervised painter — into a **generative latent state**: a 2-state CTMC where state 0 =
"modern background" (anchored by the panel) and state 1 = "deep/archaic", whose emission is
keyed to **branch depth, not a label**. Reference-free for the foreign state.

Research-grade; carries the §6 identifiability risk → its own branch.

## Dependency
Branch from `main` **after Plan A merges**, inheriting Engine 2 (the depth statistic), the
ghost/archaic simulators, and the validation kit. Do not duplicate the depth work.

## The promotion
- Plan A: `depth` is a *flag* — "this tract is a deep outlier ⇒ candidate foreign".
- Plan B: `depth` *defines an emission* — the foreign state's likelihood at a node is a
  function of that node's coalescence depth with the panel, so the model can **infer**
  archaic tracts with no archaic reference, with calibrated posteriors.

## Phases

### Phase 1 — model formulation (prototype two, pick empirically)
- (a) **Depth-dependent prior**: a lineage is "archaic-eligible" where its subtending branch
  is anomalously deep; the prior on state 1 rises with depth.
- (b) **Deep-coalescence emission**: state-1 emission ∝ P(coalescence with the panel exceeds
  a depth threshold τ). τ learned or set from the panel's background depth distribution.
- Identifiability pinned by the modern anchor (panel references hard-clamped to state 0) +
  the depth prior — this is what breaks the §6 collapse a label-less state would otherwise
  cause.

### Phase 2 — inference
- Extend `pruning`/`em`: the state-1 "emission" is a per-branch, **time-dependent** factor
  (not a tip label). Pruning is already generator-agnostic (a CLAUDE.md invariant), so the
  work is the depth-keyed E-step factor + its M-step; reuse `branch_stats` (Van Loan).
- Keep `branch_expected_stats` untouched (the Phasic seam).

### Phase 3 — identifiability study (the crux)
- Show modern anchor + depth prior break the §6 collapse (label-switch, `Q→0/∞`).
- Sensitivity to τ; rank-vs-time robustness; degenerate-case tests mirroring `test_em`.

### Phase 4 — validation vs baselines  (on Plan A's archaic-like / ghost sims)
- vs **Plan A's supervised ghost flag** — does the generative state beat the diagnostic?
- vs **Relate deep-branch labelling** (Speidel et al., 2019) — the prior-art baseline.
- vs the **supervised painter *with* an archaic reference** — the accuracy ceiling.
- Metrics: tract recall/precision, calibration, behaviour vs source-divergence depth (should
  win where the donor is deeply diverged — the favorable corner).

### Phase 5 — go / no-go
- If the generative reference-free state beats the Plan A flag and is identifiable →
  productionize behind an opt-in. If not → document as a research result (the flag may
  suffice). Test a "single ancient genome as a *soft* anchor" middle ground as a fallback.

## Decisions (resolve empirically in Phases 1/4, not up front)
- Model formulation: depth-prior vs deep-coalescence emission (prototype both).
- Strictly reference-free vs one optional ancient genome as a soft anchor.

## Risks
- **Identifiability** (the big one, §6) — a label-less state can collapse / switch; the depth
  prior + modern anchor are the guardrails to prove out.
- Calibration dependence of depth (Relate panmictic-prior bias) — prefer rank/quantile.
- Conflating ILS deep outliers with true archaic — the no-ghost / ILS control bounds this.
