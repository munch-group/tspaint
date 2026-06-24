# tslai — Tree-Sequence Local Ancestry Inference

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
| Head-to-head harness | `compare.py` | score painters (tslai, ARG-native baseline, external) on one truth |

On strong-structure msprime sims (the true ARG), painting accuracy is ~1.0 with good
calibration, and breakpoint flicker is ~1000× below the true-tract discontinuity —
so the blocked-EM approximation is sufficient and the deferred loopy-BP alternative
(`bp/`) is not needed (CLAUDE.md §7.3). On a **tsinfer-inferred** ARG, painting is
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
flicker unaffected by sample size; whole-genome at that size is hours per fit — the
incremental-forest / vectorized-pruning lever (CLAUDE.md §3.3).

**Outstanding:** the **§6 order-only / ranked-topology variant** — the head-to-head shows
tslai's CTMC is sensitive to branch lengths, which tsinfer resolves poorly on short/sparse
regions (SINGER's calibrated times avoid it; CLAUDE.md §9); plugging real external
comparators (RFMix/MOSAIC/FLARE, ARGMix, Pearson & Durbin) into the `compare.py` harness;
the Relate `--compress` front end (`io_relate.py`); and finding the regime where merging
SINGER posterior samples improves *accuracy* (it currently adds a calibrated uncertainty
band only; CLAUDE.md §7.4).

## Install

```bash
pixi install          # dev environment (tskit, msprime, numpy, scipy, ...)
pixi run test         # run the test suite
```

## Quickstart

```python
import tslai

# simulate admixture with known local ancestry, fit, paint, and score
r = tslai.admixture_experiment(T_admix=300, n_admix=6, n_ref=10,
                               sequence_length=4e5, f_A=0.5, seed=1)
print("per-base accuracy:", r["accuracy"])
```

See `notebooks/` for the persistence go/no-go (00), accuracy / calibration /
accuracy-vs-age (01), the flicker / `bp/` decision (02), and haplotype paintings +
runtime/correctness/flicker scaling across regimes (03).
