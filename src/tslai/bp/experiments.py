"""Reproducible BP-vs-deadband comparison (CLAUDE.md §7).

Quantifies how much the horizontal BP smoother adds over the per-position
:func:`tslai.output.hard_segments` deadband for *segmentation* fidelity (the admixture-dating
object), on the true vs the inferred (tsinfer) ARG. The headline finding:

* **true ARG** — the per-tree posteriors are clean, the deadband is near-optimal, BP adds
  nothing (and a little variance);
* **inferred ARG** — tree-inference scatters spurious breakpoints the deadband cannot tell from
  real ones by confidence alone, but BP's spatial smoothing recovers the tract structure
  (breakpoint F1 ~0.71 → ~0.98 at T_admix=500). §7's horizontal coupling earns its keep on the
  realistic input.
"""
from __future__ import annotations

import numpy as np

from ..compare import tslai_paint
from ..output import hard_segments
from ..sim import simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
from ..validate import map_truth, breakpoint_precision_recall, balanced_accuracy
from .horizontal import bp_smooth_track

__all__ = ["bp_vs_deadband_experiment"]


def _setup(T_admix, seed, infer, mutation_rate, **sim_kw):
    ts = simulate_admixture(random_seed=seed, T_admix=T_admix, **sim_kw)
    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[npop[s]] for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]
    true_segs = map_truth({q: local_ancestry_truth(ts)[0][q] for q in queries}, sop)
    work = ts
    if infer:
        from ..io_tsinfer import add_mutations, infer_tree_sequence
        work = infer_tree_sequence(add_mutations(ts, rate=mutation_rate, random_seed=seed))
    return work, labels, queries, true_segs


def _ratio_f1(seg_by_node, true_segs, queries, length, true_density, tol):
    precs, recs, nsw = [], [], 0
    for q in queries:
        pr = breakpoint_precision_recall(seg_by_node[q], true_segs[q], tol)
        if not np.isnan(pr["precision"]):
            precs.append(pr["precision"])
        if not np.isnan(pr["recall"]):
            recs.append(pr["recall"])
        nsw += pr["n_inferred"]
    p = float(np.mean(precs)) if precs else 0.0
    r = float(np.mean(recs)) if recs else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return nsw / ((length / 1e6) * len(queries)) / true_density, f1


def bp_vs_deadband_experiment(*, T_admix=500.0, infer=False, seeds=(1, 2, 3), n_admix=8, n_ref=8,
                              sequence_length=2e6, Ne=1000, T_split=5000.0, f_A=0.5,
                              recombination_rate=1e-8, mutation_rate=4e-7, tol=1e5,
                              deadbands=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                              epsilons=(0.3, 0.2, 0.1, 0.05, 0.02, 0.01)):
    """Compare BP smoothing vs the deadband for segmentation, over ``seeds``.

    For each method the operating point whose switch-density ratio is closest to 1 (the
    dating-relevant point) is selected per seed; returns mean±std breakpoint F1 there, plus
    raw and BP per-base balanced accuracy. ``infer=True`` runs on tsinfer-inferred ARGs.
    """
    pi = np.full(2, 0.5)
    sim_kw = dict(n_admix=n_admix, n_ref=n_ref, sequence_length=sequence_length, Ne=Ne,
                  T_split=T_split, f_A=f_A, recombination_rate=recombination_rate)
    db_f1, bp_f1, raw_acc, bp_acc = [], [], [], []
    for seed in seeds:
        work, labels, queries, true_segs = _setup(T_admix, seed, infer, mutation_rate, **sim_kw)
        td = np.mean([sum(1 for k in range(1, len(true_segs[q]))
                          if true_segs[q][k][2] != true_segs[q][k - 1][2])
                      / (sequence_length / 1e6) for q in queries])
        soft = tslai_paint(work, labels, queries)
        raw_acc.append(balanced_accuracy(soft, true_segs, samples=queries))
        db = [_ratio_f1({q: hard_segments(soft[q], c) for q in queries}, true_segs, queries,
                        sequence_length, td, tol) for c in deadbands]
        sm = {e: {q: bp_smooth_track(soft[q], pi, e) for q in queries} for e in epsilons}
        bp = [_ratio_f1({q: hard_segments(sm[e][q]) for q in queries}, true_segs, queries,
                        sequence_length, td, tol) for e in epsilons]
        best_e = min(epsilons, key=lambda e: abs(
            _ratio_f1({q: hard_segments(sm[e][q]) for q in queries}, true_segs, queries,
                      sequence_length, td, tol)[0] - 1))
        bp_acc.append(balanced_accuracy(sm[best_e], true_segs, samples=queries))
        db_f1.append(min(db, key=lambda x: abs(x[0] - 1))[1])
        bp_f1.append(min(bp, key=lambda x: abs(x[0] - 1))[1])
    return {
        "T_admix": T_admix, "inferred": infer, "n_seeds": len(seeds),
        "deadband_f1": (float(np.mean(db_f1)), float(np.std(db_f1))),
        "bp_f1": (float(np.mean(bp_f1)), float(np.std(bp_f1))),
        "raw_balanced_accuracy": float(np.mean(raw_acc)),
        "bp_balanced_accuracy": float(np.mean(bp_acc)),
    }
