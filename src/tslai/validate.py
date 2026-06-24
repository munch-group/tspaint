"""Validation metrics for simulated-truth experiments (CLAUDE.md §7.3, §9) — Rung 8.

Operates on the segment tracks from :func:`tslai.output.posterior_table` and the
true local-ancestry tracts from :func:`tslai.sim.local_ancestry_truth` (mapped from
population ids to ancestry-state indices via :func:`map_truth`).

* :func:`per_base_accuracy` — span-weighted fraction correctly painted.
* :func:`reliability_curve` — calibration (predicted vs. empirical) by probability bin.
* :func:`breakpoint_flicker` — §7.3 posterior discontinuity across segment boundaries.
* :func:`tract_boundary_error` — localization error of inferred vs. true switch points.
"""
from __future__ import annotations

import numpy as np

from .output import MISSING_INFO

__all__ = ["map_truth", "per_base_accuracy", "balanced_accuracy", "mean_confidence",
           "reliability_curve", "breakpoint_flicker", "tract_boundary_error",
           "breakpoint_precision_recall", "switch_density"]


def map_truth(truth, state_of_pop):
    """Map truth tracts ``{sample: [(left, right, pop_id)]}`` to ancestry-state indices."""
    return {int(s): [(l, r, state_of_pop[p]) for (l, r, p) in segs]
            for s, segs in truth.items()}


def _walk_overlap(post_segs, truth_segs):
    """Yield ``(lo, hi, Segment, true_state)`` over the common refinement of two
    piecewise-constant tracks (both sorted, covering the same interval)."""
    i = j = 0
    while i < len(post_segs) and j < len(truth_segs):
        a = post_segs[i]
        bl, br, bstate = truth_segs[j]
        lo = max(a.left, bl)
        hi = min(a.right, br)
        if hi > lo:
            yield lo, hi, a, bstate
        if a.right <= br:
            i += 1
        else:
            j += 1


def per_base_accuracy(tracks, truth_states, samples=None, exclude_missing=True):
    """Span-weighted fraction where ``argmax`` of the posterior matches the truth."""
    total = correct = 0.0
    for s in (samples if samples is not None else tracks):
        s = int(s)
        for lo, hi, seg, tstate in _walk_overlap(tracks[s], truth_states[s]):
            if exclude_missing and seg.status == MISSING_INFO:
                continue
            w = hi - lo
            total += w
            if int(np.argmax(seg.posterior)) == tstate:
                correct += w
    return correct / total if total > 0 else float("nan")


def balanced_accuracy(tracks, truth_states, samples=None, exclude_missing=True, K=2):
    """Mean of per-true-class accuracies — robust to class imbalance, so an
    uninformative painter (argmax tie-broken to one class) scores ~0.5, not the
    majority-class fraction. The honest "does it discriminate?" metric."""
    correct = np.zeros(K)
    total = np.zeros(K)
    for s in (samples if samples is not None else tracks):
        for lo, hi, seg, tstate in _walk_overlap(tracks[int(s)], truth_states[int(s)]):
            if exclude_missing and seg.status == MISSING_INFO:
                continue
            w = hi - lo
            total[tstate] += w
            if int(np.argmax(seg.posterior)) == tstate:
                correct[tstate] += w
    present = total > 0
    return float(np.mean(correct[present] / total[present])) if present.any() else float("nan")


def mean_confidence(tracks, samples=None, state=0, exclude_missing=True):
    """Span-weighted mean ``|2*P(state) - 1|`` — 0 = uninformative (P≈0.5), 1 =
    confident. Distinguishes "the tree can't tell" from a wrong confident call."""
    num = den = 0.0
    for s in (samples if samples is not None else tracks):
        for seg in tracks[int(s)]:
            if exclude_missing and seg.status == MISSING_INFO:
                continue
            w = seg.right - seg.left
            den += w
            num += w * abs(2.0 * float(seg.posterior[state]) - 1.0)
    return num / den if den > 0 else float("nan")


def reliability_curve(tracks, truth_states, state=0, n_bins=10, exclude_missing=True):
    """Calibration of ``P(state)``: span-weighted predicted vs. empirical per bin.

    A well-calibrated painter has ``pred ≈ emp`` in every populated bin.
    """
    pred = np.zeros(n_bins)
    emp = np.zeros(n_bins)
    wsum = np.zeros(n_bins)
    for s in tracks:
        for lo, hi, seg, tstate in _walk_overlap(tracks[int(s)], truth_states[int(s)]):
            if exclude_missing and seg.status == MISSING_INFO:
                continue
            p = float(seg.posterior[state])
            w = hi - lo
            b = min(int(p * n_bins), n_bins - 1)
            pred[b] += p * w
            emp[b] += (1.0 if tstate == state else 0.0) * w
            wsum[b] += w
    m = wsum > 0
    return {"pred": pred[m] / wsum[m], "emp": emp[m] / wsum[m], "weight": wsum[m]}


def breakpoint_flicker(tracks, sample, state=0):
    """§7.3: posterior discontinuity across a sample's segment boundaries."""
    segs = tracks[int(sample)]
    diffs = []
    flips = 0
    for a, b in zip(segs, segs[1:]):
        diffs.append(abs(float(a.posterior[state]) - float(b.posterior[state])))
        if int(np.argmax(a.posterior)) != int(np.argmax(b.posterior)):
            flips += 1
    n = len(diffs)
    return {"mean_abs_diff": float(np.mean(diffs)) if n else 0.0,
            "flip_rate": flips / n if n else 0.0,
            "n_boundaries": n}


def _switch_positions(items, state_of, left_of):
    pos = []
    prev = None
    for it in items:
        st = state_of(it)
        if prev is not None and st != prev:
            pos.append(left_of(it))
        prev = st
    return pos


def tract_boundary_error(tracks, truth_states, sample):
    """Localization error: distance from each true switch to the nearest inferred one."""
    inferred = _switch_positions(tracks[int(sample)],
                                 lambda seg: int(np.argmax(seg.posterior)),
                                 lambda seg: seg.left)
    true_sw = _switch_positions(truth_states[int(sample)], lambda t: t[2], lambda t: t[0])
    if not true_sw:
        return {"n_true_switches": 0, "median_error": float("nan"), "mean_error": float("nan")}
    errs = [min((abs(t - i) for i in inferred), default=float("inf")) for t in true_sw]
    return {"n_true_switches": len(true_sw),
            "median_error": float(np.median(errs)),
            "mean_error": float(np.mean(errs))}


def breakpoint_precision_recall(inferred_segs, true_segs, tol):
    """Precision/recall of inferred ancestry switch points vs. truth, matched within ``tol`` bp.

    Inputs are **hard** segment lists ``[(left, right, state)]`` (e.g. from
    :func:`tslai.output.hard_segments` and :func:`map_truth`). **Precision** = fraction of
    inferred switches lying near a true switch — low precision means spurious fragmentation,
    which biases tract-length / admixture-pulse dating *older*. **Recall** = fraction of true
    switches recovered — low recall means missed or over-smoothed switches (biases *younger*).
    The two trade off against the :func:`tslai.output.hard_segments` ``deadband`` (CLAUDE.md §9).
    """
    isw = _switch_positions(inferred_segs, lambda s: s[2], lambda s: s[0])
    tsw = _switch_positions(true_segs, lambda s: s[2], lambda s: s[0])
    prec = float(np.mean([any(abs(i - t) <= tol for t in tsw) for i in isw])) if isw else float("nan")
    rec = float(np.mean([any(abs(t - i) <= tol for i in isw) for t in tsw])) if tsw else float("nan")
    return {"precision": prec, "recall": rec, "n_inferred": len(isw), "n_true": len(tsw)}


def switch_density(segs, length):
    """Ancestry switches per unit length (the quantity admixture-pulse dating reads as time).

    ``segs`` is a hard segment list ``[(left, right, state)]``; over-fragmentation inflates
    this and biases the inferred pulse older, over-smoothing deflates it and biases younger.
    """
    n = len(_switch_positions(segs, lambda s: s[2], lambda s: s[0]))
    return n / length if length > 0 else float("nan")
