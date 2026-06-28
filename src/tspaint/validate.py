"""Validation metrics for simulated-truth experiments (CLAUDE.md §7.3, §9) — Rung 8.

Operates on the segment tracks from :func:`tspaint.output.posterior_table` and the
true local-ancestry tracts from :func:`tspaint.sim.local_ancestry_truth` (mapped from
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
           "breakpoint_precision_recall", "switch_density",
           "global_proportion", "true_proportion", "accuracy_by_segment_size",
           "DEFAULT_SIZE_BINS"]

#: Default log-spaced true-segment-length bins (bp) for :func:`accuracy_by_segment_size`.
DEFAULT_SIZE_BINS = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7])


def map_truth(truth, state_of_pop):
    """Remap truth tracts from population ids to ancestry-state indices.

    Parameters
    ----------
    truth : dict[int, list[tuple[float, float, int]]]
        Truth tracts ``{sample: [(left, right, pop_id)]}`` (e.g. from
        :func:`tspaint.sim.local_ancestry_truth`).
    state_of_pop : dict[int, int]
        Mapping from population id to ancestry-state index.

    Returns
    -------
    dict[int, list[tuple[float, float, int]]]
        The same tracts with each ``pop_id`` replaced by its ancestry-state
        index, keyed by integer sample id.
    """
    return {int(s): [(l, r, state_of_pop[p]) for (l, r, p) in segs]
            for s, segs in truth.items()}


def _walk_overlap(post_segs, truth_segs):
    """Walk the common refinement of two piecewise-constant tracks.

    Parameters
    ----------
    post_segs : list[Segment]
        Posterior segments, sorted and covering the interval.
    truth_segs : list[tuple[float, float, int]]
        Truth ``(left, right, state)`` tracts, sorted over the same interval.

    Yields
    ------
    tuple[float, float, Segment, int]
        ``(lo, hi, segment, true_state)`` for each overlapping sub-interval of
        the common refinement.
    """
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
    """Span-weighted per-base painting accuracy.

    Computes the span-weighted fraction of the genome where the ``argmax`` of the
    posterior matches the truth.

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Posterior segment tracks per sample (e.g. from
        :func:`tspaint.output.posterior_table`).
    truth_states : dict[int, list[tuple[float, float, int]]]
        True ancestry-state tracts per sample (e.g. from :func:`map_truth`).
    samples : iterable[int], optional
        Samples to score; defaults to all keys of ``tracks``.
    exclude_missing : bool, optional
        If True, skip segments tagged :data:`tspaint.output.MISSING_INFO`.

    Returns
    -------
    float
        Span-weighted accuracy, or ``nan`` if no span was scored.
    """
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
    """Mean of per-true-class painting accuracies.

    Robust to class imbalance: an uninformative painter (argmax tie-broken to one
    class) scores ~0.5 rather than the majority-class fraction. The honest "does
    it discriminate?" metric.

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Posterior segment tracks per sample.
    truth_states : dict[int, list[tuple[float, float, int]]]
        True ancestry-state tracts per sample (e.g. from :func:`map_truth`).
    samples : iterable[int], optional
        Samples to score; defaults to all keys of ``tracks``.
    exclude_missing : bool, optional
        If True, skip segments tagged :data:`tspaint.output.MISSING_INFO`.
    K : int, optional
        Number of ancestry states.

    Returns
    -------
    float
        Mean over present classes of the span-weighted per-class accuracy, or
        ``nan`` if no class is present.

    Examples
    --------
    >>> from tspaint.output import Segment, INFORMATIVE
    >>> import numpy as np
    >>> tracks = {0: [Segment(0, 10, np.array([0.9, 0.1]), INFORMATIVE),
    ...               Segment(10, 20, np.array([0.2, 0.8]), INFORMATIVE)]}
    >>> truth = {0: [(0, 10, 0), (10, 20, 1)]}
    >>> balanced_accuracy(tracks, truth)
    1.0
    """
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
    """Span-weighted mean painter confidence ``|2*P(state) - 1|``.

    A value of 0 is uninformative (``P ≈ 0.5``) and 1 is fully confident. This
    distinguishes "the tree can't tell" from a wrong confident call.

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Posterior segment tracks per sample.
    samples : iterable[int], optional
        Samples to score; defaults to all keys of ``tracks``.
    state : int, optional
        Ancestry-state index whose posterior probability is read.
    exclude_missing : bool, optional
        If True, skip segments tagged :data:`tspaint.output.MISSING_INFO`.

    Returns
    -------
    float
        Span-weighted mean confidence, or ``nan`` if no span was scored.
    """
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
    """Calibration curve of ``P(state)``: predicted vs. empirical per bin.

    A well-calibrated painter has ``pred ≈ emp`` in every populated bin.

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Posterior segment tracks per sample.
    truth_states : dict[int, list[tuple[float, float, int]]]
        True ancestry-state tracts per sample (e.g. from :func:`map_truth`).
    state : int, optional
        Ancestry-state index whose posterior probability is binned.
    n_bins : int, optional
        Number of equal-width probability bins on ``[0, 1]``.
    exclude_missing : bool, optional
        If True, skip segments tagged :data:`tspaint.output.MISSING_INFO`.

    Returns
    -------
    dict
        Keys ``"pred"``, ``"emp"`` and ``"weight"``, each a 1-D array over the
        populated bins giving the span-weighted mean predicted probability, the
        empirical frequency of ``state``, and the total span weight respectively.
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
    """Posterior discontinuity across a sample's segment boundaries (CLAUDE.md §7.3).

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Posterior segment tracks per sample.
    sample : int
        Sample id whose consecutive-segment boundaries are examined.
    state : int, optional
        Ancestry-state index whose posterior probability is compared.

    Returns
    -------
    dict
        Keys ``"mean_abs_diff"`` (mean ``|P_left(state) - P_right(state)|`` over
        boundaries), ``"flip_rate"`` (fraction of boundaries where the ``argmax``
        state changes) and ``"n_boundaries"`` (number of segment boundaries).
    """
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
    """Left coordinates of state changes in a sequence of segments.

    Parameters
    ----------
    items : iterable
        Ordered segments / tracts.
    state_of : callable
        Maps an item to its (hashable) state.
    left_of : callable
        Maps an item to its left coordinate.

    Returns
    -------
    list[float]
        Left coordinates of items whose state differs from the preceding item.
    """
    pos = []
    prev = None
    for it in items:
        st = state_of(it)
        if prev is not None and st != prev:
            pos.append(left_of(it))
        prev = st
    return pos


def tract_boundary_error(tracks, truth_states, sample):
    """Tract-boundary localization error against the truth.

    For each true ancestry switch, measures the distance to the nearest inferred
    switch (from the posterior ``argmax``).

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Posterior segment tracks per sample.
    truth_states : dict[int, list[tuple[float, float, int]]]
        True ancestry-state tracts per sample (e.g. from :func:`map_truth`).
    sample : int
        Sample id to score.

    Returns
    -------
    dict
        Keys ``"n_true_switches"`` (number of true switch points),
        ``"median_error"`` and ``"mean_error"`` (median/mean distance from each
        true switch to the nearest inferred switch). The errors are ``nan`` when
        there are no true switches and ``inf`` when there are no inferred ones.
    """
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
    """Precision/recall of inferred ancestry switch points vs. truth.

    Switches are matched within ``tol`` bp. **Precision** is the fraction of
    inferred switches lying near a true switch — low precision means spurious
    fragmentation, which biases tract-length / admixture-pulse dating *older*.
    **Recall** is the fraction of true switches recovered — low recall means
    missed or over-smoothed switches (biases *younger*). The two trade off
    against the :func:`tspaint.output.hard_segments` ``deadband`` (CLAUDE.md §9).

    Parameters
    ----------
    inferred_segs : list[tuple[float, float, int]]
        Inferred **hard** segments ``[(left, right, state)]`` (e.g. from
        :func:`tspaint.output.hard_segments`).
    true_segs : list[tuple[float, float, int]]
        True hard segments ``[(left, right, state)]`` (e.g. from
        :func:`map_truth`).
    tol : float
        Matching tolerance in base pairs between inferred and true switches.

    Returns
    -------
    dict
        Keys ``"precision"``, ``"recall"``, ``"n_inferred"`` and ``"n_true"``.
        ``"precision"`` is ``nan`` when there are no inferred switches and
        ``"recall"`` is ``nan`` when there are no true switches.
    """
    isw = _switch_positions(inferred_segs, lambda s: s[2], lambda s: s[0])
    tsw = _switch_positions(true_segs, lambda s: s[2], lambda s: s[0])
    prec = float(np.mean([any(abs(i - t) <= tol for t in tsw) for i in isw])) if isw else float("nan")
    rec = float(np.mean([any(abs(t - i) <= tol for i in isw) for t in tsw])) if tsw else float("nan")
    return {"precision": prec, "recall": rec, "n_inferred": len(isw), "n_true": len(tsw)}


def switch_density(segs, length):
    """Ancestry switches per unit length.

    This is the quantity admixture-pulse dating reads as time: over-fragmentation
    inflates it and biases the inferred pulse older, over-smoothing deflates it
    and biases younger.

    Parameters
    ----------
    segs : list[tuple[float, float, int]]
        Hard segment list ``[(left, right, state)]``.
    length : float
        Sequence length to normalise by.

    Returns
    -------
    float
        Number of ancestry switches divided by ``length``, or ``nan`` if
        ``length <= 0``.
    """
    n = len(_switch_positions(segs, lambda s: s[2], lambda s: s[0]))
    return n / length if length > 0 else float("nan")


def global_proportion(tracks, state=0, samples=None, exclude_missing=True):
    """Painter's estimated global ancestry proportion of ``state``.

    Span-weighted mean of the posterior probability of ``state`` over all (sample, position) —
    i.e. the soft global ancestry fraction the painter assigns to ``state``. For a one-hot (hard)
    painter this is the fraction of the genome called ``state``. Compare to
    :func:`true_proportion`; the signed difference is the painter's global-proportion bias.

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Posterior segment tracks per sample.
    state : int, optional
        Ancestry-state index whose proportion is estimated.
    samples : iterable[int], optional
        Samples to include; defaults to all keys of ``tracks``.
    exclude_missing : bool, optional
        If True, skip :data:`tspaint.output.MISSING_INFO` spans.

    Returns
    -------
    float
        Span-weighted mean posterior of ``state``, or ``nan`` if no span was scored.
    """
    num = den = 0.0
    for s in (samples if samples is not None else tracks):
        for seg in tracks[int(s)]:
            if exclude_missing and seg.status == MISSING_INFO:
                continue
            w = seg.right - seg.left
            den += w
            num += w * float(seg.posterior[state])
    return num / den if den > 0 else float("nan")


def true_proportion(truth_states, state=0, samples=None):
    """True global ancestry proportion of ``state`` (span-weighted fraction of the truth).

    Parameters
    ----------
    truth_states : dict[int, list[tuple[float, float, int]]]
        True ancestry-state tracts per sample (e.g. from :func:`map_truth`).
    state : int, optional
        Ancestry-state index whose true proportion is computed.
    samples : iterable[int], optional
        Samples to include; defaults to all keys of ``truth_states``.

    Returns
    -------
    float
        Span-weighted fraction of the truth equal to ``state``, or ``nan`` if empty.
    """
    num = den = 0.0
    for s in (samples if samples is not None else truth_states):
        for (l, r, st) in truth_states[int(s)]:
            w = r - l
            den += w
            if int(st) == state:
                num += w
    return num / den if den > 0 else float("nan")


def accuracy_by_segment_size(tracks, truth_states, bins=None, samples=None, exclude_missing=True):
    """Per-base painting accuracy stratified by **true** ancestry-tract length.

    The headline of the segment-length analysis (CLAUDE.md §9): tract-/copying-based methods
    degrade as tracts shorten, so accuracy as a function of true segment size separates methods
    that hold up on short (old-admixture) tracts from those that don't. Each true tract is
    assigned to a length bin; within it, the span where the painter's ``argmax`` matches the true
    state counts as correct, aggregated per bin.

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Posterior segment tracks per sample.
    truth_states : dict[int, list[tuple[float, float, int]]]
        True ancestry-state tracts per sample (e.g. from :func:`map_truth`).
    bins : array_like, optional
        Monotone tract-length bin edges in bp (``len(bins) - 1`` bins). Defaults to
        :data:`DEFAULT_SIZE_BINS`.
    samples : iterable[int], optional
        Samples to score; defaults to all keys of ``tracks``.
    exclude_missing : bool, optional
        If True, skip :data:`tspaint.output.MISSING_INFO` spans.

    Returns
    -------
    dict
        Keys ``"edges"`` (the bin edges), ``"accuracy"``, ``"weight"`` (scored span) and
        ``"n_segments"`` — each a 1-D array over the bins. ``accuracy`` is ``nan`` in empty bins.
    """
    bins = np.asarray(DEFAULT_SIZE_BINS if bins is None else bins, float)
    nb = len(bins) - 1
    correct, total = np.zeros(nb), np.zeros(nb)
    nseg = np.zeros(nb, np.int64)
    for s in (samples if samples is not None else tracks):
        s = int(s)
        post = tracks[s]
        for (tl, tr, tst) in truth_states[s]:
            b = int(np.searchsorted(bins, tr - tl, side="right")) - 1
            if b < 0 or b >= nb:
                continue
            nseg[b] += 1
            for lo, hi, seg, st in _walk_overlap(post, [(tl, tr, tst)]):
                if exclude_missing and seg.status == MISSING_INFO:
                    continue
                w = hi - lo
                total[b] += w
                if int(np.argmax(seg.posterior)) == st:
                    correct[b] += w
    acc = np.full(nb, np.nan)
    m = total > 0
    acc[m] = correct[m] / total[m]
    return {"edges": bins, "accuracy": acc, "weight": total, "n_segments": nseg}
