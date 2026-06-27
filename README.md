# tspaint — Tree-Sequence Local Ancestry Inference

Soft, calibrated **local ancestry** along individual haplotypes, inferred from an
**inferred tree sequence** (Relate → tskit, or tsinfer/tsdate). Ancestry is modelled
as a discrete character evolving up the branches of each marginal tree under a
continuous-time Markov chain (CTMC, "mugration"); reference haplotypes label tips
through a soft noise model with **learned per-tip credibility**; and
`(Q, root frequencies, {w_i})` are fit by **EM** whose sufficient statistics are
accumulated **per tree-sequence edge, span-weighted** — so a clade persisting across
many trees is counted once. Output: for every haplotype, at every position, a
calibrated posterior `P(ancestry)` — soft, not a hard call.

The same machinery, run on reference tips, is a leave-one-out
introgression/mislabel detector. See [`CLAUDE.md`](CLAUDE.md) for the authoritative
spec, the math, and the design decisions.

## Status

Research implementation. The inference engine and the soft-LAI deliverable are
implemented and validated on simulated truth:

| Component | Module | Validation |
|---|---|---|
| Ancestry CTMC, emission, credibility | `model.py` | unit |
| Per-branch dwell/jump stats (Van Loan) | `branch_stats.py` | analytic 2-state + quadrature |
| Felsenstein pruning (polytomy/forest/isolated, leave-one-out) | `pruning.py` | brute-force enumeration |
| Edge-blocked span-weighted sufficient stats | `accumulate.py` | no-double-count keystone |
| Blocked EM `(Q, π, {w_i})` | `em.py` | monotone log-lik; Q recovery; mislabel detection |
| Per-haplotype posteriors + missing-info tagging | `output.py` | coverage, prior-fallback |
| Accuracy / calibration / flicker metrics | `validate.py`, `experiments.py` | end-to-end on admixture sims |
| tsinfer inferred-ARG front end | `io_tsinfer.py` | inferred-ARG accuracy (bounded by ARG accuracy) |
| ARG-posterior ensemble merge | `ensemble.py` | average paintings across tree-sequence samples + uncertainty band |
| SINGER posterior front end | `io_singer.py` | run SINGER MCMC → posterior ARG samples (largely lifts the §9 ARG bound) |
| Head-to-head harness + RFMix comparator | `compare.py`, `io_rfmix.py` | score painters (tspaint, ARG-native baseline, RFMix) on one truth; tspaint matches RFMix accuracy, stays calibrated |
| Hard segmentation (deadband) + fragmentation metrics | `output.py`, `validate.py` | recovers the true tract-length distribution for dating (CLAUDE.md §9) |
| High-level `paint()` API | `api.py` | one-call fit + paint → `Painting` |
| Horizontal BP/EP smoother | `bp/` | wins on inferred ARGs (breakpoint F1 0.71→0.98), redundant on true ARG (CLAUDE.md §7) |
| Reference QC / foreign tracts / ghost detection | `introgression.py`, `sim.py` | impure-ref discrimination + LOO introgression recall ~0.9; ghost detection recall 0.58 / precision 1.0 at ~1% no-ghost-control false-positive (CLAUDE.md §9) |

On strong-structure msprime sims (the true ARG), painting accuracy is ~1.0 with good
calibration, and breakpoint flicker is ~1000× below the true-tract discontinuity — so the
blocked-EM approximation is sufficient on the true ARG, where the `hard_segments` deadband
recovers the tract-length distribution for dating. On **inferred** ARGs a horizontal BP/EP
smoother adds value (it suppresses tree-inference-induced spurious switches a per-position
threshold cannot — breakpoint F1 0.71→0.98): use `tspaint.paint(..., smooth=True)` or `tspaint.bp`
(CLAUDE.md §7). On a **tsinfer-inferred** ARG, painting is
bounded by ARG accuracy: ~0.88 with dense variants down toward chance when variants
are sparse — tree accuracy, not tract length, is the binding constraint (§9). With
**SINGER** (Bayesian posterior ARG sampling) that bound largely disappears: single
posterior samples paint ~0.99 even at sparse data where tsinfer gives chance, and merging
the posterior ensemble adds a calibrated uncertainty band — the ARG inference method
matters far more than the merge (§7.4).

**Scaling.** The E-step is O(#trees × #nodes) per EM iteration (after exact `expm`
caching); at a fixed region length #trees is region-bounded, so runtime is roughly
linear in sample size. Measured (true ARG, 0.1 Mb region): ~480 haplotypes fits in
~5 s (4 iterations), ~1.2 s/iteration; tsinfer inference adds ~1–3 s. **~500
haplotypes is comfortable for region/chromosome-scale analyses**, with accuracy and
flicker unaffected by sample size. The genome E-step is an exact map-reduce over
independent trees, so it parallelises near-linearly across cores: `paint(..., n_jobs=N)`
/ `tspaint.fit(..., n_jobs=N)` (and the `tspaint --cores` CLI) split it bit-exactly over
a process pool (`tspaint.parallel`; see [`paral-assess.md`](paral-assess.md)), and across
the cluster each `tspaint` subcommand is a GWF job over SINGER windows × ensemble members.

**Outstanding:** the remaining external comparators (**RFMix is wired** via
`tspaint.compare.rfmix_paint`; MOSAIC/FLARE and the ARG-native ARGMix / Pearson & Durbin need
separate installs); the Relate `--compress` front end (`io_relate.py`); and the regime where
merging SINGER posterior samples improves *accuracy* rather than only adding a calibrated band
(CLAUDE.md §7.4).

*Closed:* the §6 order-only / ranked variant (measured — **not beneficial**; the short-region
failure was π-identifiability, fixed by holding π uniform, `estimate_pi=False`); the RFMix
head-to-head (tspaint matches RFMix's accuracy while staying calibrated); the segment-fragmentation
question for dating (the `hard_segments` deadband recovers the true tract-length distribution, and
on *inferred* ARGs the horizontal BP smoother — now in `tspaint.bp` / `paint(smooth=True)` — helps
further, CLAUDE.md §7, §9).

## Install

```bash
pixi install          # dev environment (tskit, msprime, numpy, scipy, ...)
pixi run test         # run the test suite
```

## Quickstart

```python
import tspaint

# 1. Get a tree sequence — here, simulate admixture with known truth
#    (or build one from genotypes: tspaint.io.tsinfer / relate / singer — ts | VCF Zarr | VCF).
ts = tspaint.simulate_admixture(n_admix=10, n_ref=10, sequence_length=1e6,
                              T_admix=100, T_split=5000, Ne=1000, random_seed=1)

# 2. Label the reference haplotypes (sample-node id -> ancestry state) and paint the rest.
pop = ts.tables.nodes.population
name = {p: ts.population(p).metadata["name"] for p in range(ts.num_populations)}
state = {p: i for i, p in enumerate(p for p, n in name.items() if n in ("A", "B"))}
labels = {int(s): state[pop[s]] for s in ts.samples() if pop[s] in state}

painting = tspaint.paint(ts, labels)          # EM-fit (Q[, π, w]) on references, paint the queries
painting.posteriors[painting.queries[0]]    # soft per-position posterior over ancestry (Segments)
painting.segments(deadband=0.4)             # hard ancestry tracts (for tract-length / dating)
```

## Command-line interface (GWF / cluster)

The `tspaint` command wraps the library so a pipeline runs as file-in/file-out jobs (e.g. under
[GWF](https://gwf.app)) with no glue Python. Computed results are `.npz`; hand-authored inputs are
text (labels JSON `{"<node>": <state>}`; id-lists inline `3,4,5` or `@file`). `--cores/-j` defaults
to `$SLURM_JOB_CPUS_PER_NODE`.

```bash
tspaint simulate --n-admix 8 --n-ref 8 -o sim.trees --labels-out labels.json   # or your own data
tspaint fit   sim.trees --labels labels.json -j 8 -o params.npz                 # one pooled fit
tspaint paint sim.trees --params params.npz -j 8 -o painting.npz                # paint (per member)
tspaint merge painting.npz -o merged.npz                                        # ensemble mean + band
tspaint date  sim.trees --labels labels.json -o rtt.npz                         # also qc/introgress/ghost/archaic
```

For a large chromosome, build the SINGER posterior ensemble window-by-window
(`tspaint trees singer-window` per 5 Mb window, then `tspaint trees merge-arg` per member) and
feed the members to `fit` / `paint` / `merge`. See [`examples/workflow.py`](examples/workflow.py)
for a runnable GWF `Workflow` and [`paral-assess.md`](paral-assess.md) for the parallelism design
and the bit-exactness contract.

## Public API

**Core** — `tspaint.paint(ts, labels, queries=None, *, deadband=…)` returns a `Painting` with
`.posteriors` (soft `Segment` tracks), `.segments(deadband=…)` (hard tracts), and the fitted
`.Q / .pi / .w`. Building blocks: `tspaint.fit`, `posterior_table`, `hard_segments`, `Segment`,
`make_generator_2state`. Simulation: `simulate_admixture`, `local_ancestry_truth`.

**Dating (optional)** — `tspaint.fit_rate_through_time(ts, labels)` estimates the directional
cross-ancestry rate through time (`RateThroughTime` with `.q_AB / .q_BA / .plot()`) — *when* the
labelled ancestries diverged / exchanged genes. `Painting.rate_through_time()` reuses a painting's
fit; it returns a new profile and does **not** change the painting (Q(t) gives no painting-accuracy
gain — the paths stay side by side).

**Namespaces**

| Namespace | What |
|---|---|
| `tspaint.metrics` | `balanced_accuracy`, `reliability_curve`, `breakpoint_precision_recall`, `switch_density`, `tract_boundary_error`, … |
| `tspaint.compare` | painters `tspaint_paint`, `nearest_reference_paint`, `rfmix_paint` + `head_to_head` |
| `tspaint.io` | unified front ends `tsinfer` / `relate` / `singer` (accept ts \| VCF Zarr \| VCF) + `add_mutations` |
| `tspaint.bp` | horizontal BP/EP smoother (`bp_paint`, `bp_smooth`) — also `paint(smooth=True)`; helps on inferred ARGs (§7) |
| `tspaint.dating` | admixture rate through time (time-inhomogeneous directional mugration EM): `fit_rate_through_time`, `RateThroughTime`, `paint_qt` |
| `tspaint.experiments` | end-to-end drivers: `admixture_experiment`, `age_sweep`, `fragmentation_experiment`, `singer_ensemble_experiment`, … |
| `tspaint.introgression` | reference QC (`reference_qc`), anonymous foreign tracts (`foreign_tracts`), ghost-source detection (`detect_ghost`) — built on the `loo_posterior_table` introgression lens |

Lower-level machinery is in the named submodules (`tspaint.model`, `tspaint.pruning`,
`tspaint.accumulate`, `tspaint.em`, `tspaint.output`, `tspaint.ensemble`, `tspaint.ranked`);
`tspaint.parallel` (bit-exact multi-core E-step), `tspaint.serialize` (`.npz` save/load), and
`tspaint.cli` (the `tspaint` command).

See `notebooks/` for the persistence go/no-go (00), accuracy / calibration /
accuracy-vs-age (01), the flicker / `bp/` decision (02), and haplotype paintings +
runtime/correctness/flicker scaling across regimes (03).
