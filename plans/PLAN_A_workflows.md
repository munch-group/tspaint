# Plan A — Reference QC, Anonymous Foreign Tracts, Ghost Detection → main

Promote the primitives already on the `worktree-impure-refs` branch (per-tip soft
credibility, `output.loo_posterior_table`, the impure-reference kit) into three
**user-facing, supervised diagnostic workflows**, validated on appropriate sims, and
merge to `main`.

## Architectural keystone

The three workflows **nest** and share **two engines**:

- **Engine 1 — the LOO map** (`output.loo_posterior_table`): the leave-one-out "outside
  message" — what the rest of the genealogy says about a tip *ignoring its own label*.
  Already built.
- **Engine 2 — a per-locus foreignness statistic** (`introgression.foreignness_track`):
  the one new primitive. Per (sample, locus) it returns three components:
  - `loo` — the leave-one-out posterior (Engine 1), for "dissent from the assigned label";
  - `fit` — `max_s loo[s]`, the genealogy's confidence in *any* panel state (low ⇒
    "fits nothing");
  - `depth` — TMRCA to the nearest labelled reference, **rank-normalised genome-wide**
    by default (calibration-robust; absolute-time optional).

Reference QC ⊂ Anonymous foreign tracts ⊂ Ghost detection — each adds one criterion:

| workflow | criterion | engine use |
|---|---|---|
| Reference QC | LOO dissent from a reference's **own label** | Engine 1 |
| Anonymous foreign tracts | dissent from the **assigned/expected** source, or low `fit` | Engine 1 + `fit` |
| Ghost detection | low `fit` **AND** high `depth` (deep outlier, fits no source) | Engine 1 + `fit` + `depth` |

`depth` is what separates a genuine unsampled-source tract from a merely uninformative
(50/50) tract — the latter is shallow, the former deep.

## Resolved decisions

1. **API surface:** top-level functions (`tspaint.reference_qc`, `foreign_tracts`,
   `detect_ghost`) plus thin convenience methods on `Painting`.
2. **Depth statistic:** rank/quantile by default (`depth="rank"`), absolute time optional
   (`depth="time"`).
3. **Merge cadence:** the current branch is already independently mergeable; build the
   workflows here, then merge — incremental PRs preferred at merge time (user's call).

## Phases (test after each)

### Phase 1 — the foreignness primitive  ✦ keystone
- New module `src/tspaint/introgression.py`:
  - `ForeignnessSegment(left, right, loo, fit, depth, status)`.
  - `foreignness_track(ts, Q, pi, emissions, labels, focal=None, depth="rank")` — one
    pass over marginal trees: `loo=res.loo[s]`, `fit=max(loo)`, raw depth = min over
    labelled refs of `node_time[tree.mrca(s, ref)]`; post-pass rank-normalises depth
    span-weighted genome-wide.
- Tests `tests/test_introgression.py`: toy trees — a hard-clamped tip dissents in `loo`;
  a planted deep tip gets `depth≈1` and low `fit`; coverage `[0, L)`, valid ranges.
- **Exit:** calibrated component tracks per sample; generator-agnostic.

### Phase 2 — Workflow 1: Reference QC  (validates on existing impure-ref sim)
- `tspaint.reference_qc(ts, labels, *, refine=True, anchor_frac=0.5)`:
  - Pass 1: hard-clamp all refs; per-ref LOO self-agreement `a_i = mean_genome loo[label]`
    (+ per-ref LOO introgression map). LOO is leave-one-out, so this flags foreignness even
    with hard clamps (~0.64 recall measured).
  - Pass 2 (`refine`): anchors = top `anchor_frac` by `a_i`; soften the rest; refit ⇒
    learned `w_i` + sharper maps (~0.92 recall measured).
  - Returns `ReferenceQC` with `.credibility` (per-ref `a_i`/`w_i`), `.introgression_map(ref)`,
    `.flagged_tracts(ref, deadband=…)`, `.summary()` (ranked table).
- Validate (`experiments`): on `simulate_admixture_impure_refs` — panel-level precision/recall
  of credibility flags vs truly-impure refs; per-tract recall vs census truth.
- **Exit:** ships without any new simulator; measured panel + tract precision/recall.

### Phase 3 — simulators for the unsupervised cases
- `sim.simulate_admixture_with_ghost`: a 3rd source **C** contributing to ADMIX but **not
  sampled as a reference**; census truth labels tracts A/B/C ⇒ C-tracts are ghost ground
  truth. Parameterised so a very deep C-divergence gives the **archaic-like** favorable
  regime.
- Controls reuse existing sims: 2-source `simulate_admixture` = no-ghost / deep-ILS control
  (deep coalescence, no introgression) for the false-positive floor.
- Tests `tests/test_sim.py`: C-tracts present in queries, census truth covers A/B/C, refs
  only A/B.

### Phase 4 — Workflows 2 & 3
- `tspaint.foreign_tracts(ts, labels, samples, *, threshold=…)` — segments where the
  foreignness (LOO dissent from assigned source / low `fit`) exceeds threshold.
- `tspaint.detect_ghost(ts, labels, samples, *, fit_thresh, depth_thresh)` — low `fit` AND
  high `depth` ⇒ ghost tracts + a genome-wide ghost burden per sample.
- Calibrate thresholds (`experiments`): recall/precision of ghost tracts on the ghost sim;
  **false-positive rate on the no-ghost / deep-ILS control** (the honesty gate).
- **Exit:** measured precision/recall + a stated false-positive floor.

### Phase 5 — docs, public API, merge prep
- Notebook `docs/notebooks/` (or extend `painting.ipynb`) — the three workflows end-to-end.
- Export entry points in `__init__`; `[MEASURED]` notes in CLAUDE.md §9.
- **Exit:** green suite, docs example, ready for PR(s) to main.

## Risks / honesty gates
- Ghost vs tree-error / ILS deep outliers — **must** measure the false-positive rate on the
  no-ghost control; report it, don't hide it.
- Depth statistic depends on branch-length calibration (§6 Relate panmictic bias) — rank
  default mitigates; validate rank-vs-time.
- Reference QC must keep a trusted anchor core (§6) — never soften the whole panel; the
  two-pass auto-anchor respects this.
