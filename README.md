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

On strong-structure msprime sims (the true ARG), painting accuracy is ~1.0 with good
calibration, and breakpoint flicker is ~1000× below the true-tract discontinuity —
so the blocked-EM approximation is sufficient and the deferred loopy-BP alternative
(`bp/`) is not needed (CLAUDE.md §7.3). On a **tsinfer-inferred** ARG, painting is
bounded by ARG accuracy: ~0.88 with dense variants down toward chance when variants
are sparse — tree accuracy, not tract length, is the binding constraint (§9).

**Outstanding:** the Relate `--compress` front end (`io_relate.py`), and the
hard-regime / head-to-head validation — weak structure, ancient admixture, and
comparison vs. RFMix/MOSAIC/FLARE and the ARG-native neighbours (CLAUDE.md §9, §10).

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

See `notebooks/` for the persistence go/no-go (00), accuracy & calibration (01),
and the flicker / `bp/` decision (02).
