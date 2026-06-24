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

__all__ = ["map_truth", "per_base_accuracy", "reliability_curve",
           "breakpoint_flicker", "tract_boundary_error"]


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
