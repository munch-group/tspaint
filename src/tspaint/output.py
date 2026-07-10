"""Deliverable: per-haplotype, per-position ancestry posteriors (CLAUDE.md §2.4,
§4) — Rung 6.

For every sample, at every position, the down-pass posterior over ancestry states,
returned as span-resolved :class:`Segment`s covering the whole sequence. Isolated
spans are tagged ``MISSING_INFO`` — the tree carries no information there, the
posterior falls back to the prior ``π``, and this is **distinct from a 50-50
uncertain call** (conflating them is a real interpretive error for introgression
mapping, CLAUDE.md §4.2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import emissions_for
from .pruning import prune_tree

__all__ = ["Segment", "INFORMATIVE", "MISSING_INFO", "posterior_table",
           "loo_posterior_table", "missing_info_mask", "posterior_at", "hard_segments"]

INFORMATIVE = "informative"
MISSING_INFO = "missing-info"


@dataclass
class Segment:
    """A contiguous span carrying one soft ancestry posterior.

    Attributes
    ----------
    left, right : float
        Half-open genomic interval ``[left, right)`` the segment covers.
    posterior : numpy.ndarray
        ``(K,)`` posterior over ancestry states on ``[left, right)``.
    status : str
        :data:`INFORMATIVE` or :data:`MISSING_INFO` — the latter tags an isolated
        span where the tree carries no information and ``posterior`` is the prior
        ``π`` fallback, distinct from a 50-50 uncertain call (CLAUDE.md §4.2).
    """
    left: float
    right: float
    posterior: np.ndarray   # (K,) posterior over ancestry states on [left, right)
    status: str             # INFORMATIVE or MISSING_INFO


def _paint_tracks(ts, Q, pi, emissions, focal, merge_tol, pick, tree_range=None,
                  progress=False):
    """Shared engine for the per-sample segment painters.

    Runs the down-pass per marginal tree and records, for each focal sample, the
    posterior selected by ``pick(res, sample, prior)`` — ``γ`` for
    :func:`posterior_table`, the leave-one-out outside message for
    :func:`loo_posterior_table` — merging adjacent identical segments.

    ``tree_range=(lo, hi)`` restricts painting to that half-open marginal-tree-index range
    (the rest is skipped); since each segment's posterior comes from its own tree's pruning
    — independent of which trees a chunk covers — concatenating the per-range tracks in
    genome order and re-merging at the seams reproduces the full-genome result exactly
    (:func:`tspaint.parallel.posterior_table_parallel`).

    ``progress=True`` shows a per-marginal-tree :mod:`tqdm` bar; it is honoured only for the
    full-genome pass (``tree_range is None``), since a chunked worker (``tree_range`` set) is a
    subprocess whose progress is reported per-chunk by the parent (:func:`posterior_table_parallel`).
    """
    pi = np.asarray(pi, float)
    node_time = ts.tables.nodes.time
    samples = [int(s) for s in (ts.samples() if focal is None else focal)]
    tracks = {s: [] for s in samples}

    lo, hi = (0, ts.num_trees) if tree_range is None else tree_range
    tree_iter = ts.trees()
    if progress and tree_range is None:
        from tqdm.auto import tqdm
        tree_iter = tqdm(tree_iter, total=(hi - lo), desc="painting", unit="tree")
    for ti, tree in enumerate(tree_iter):
        if ti < lo:
            continue
        if ti >= hi:
            break
        left = tree.interval.left
        right = tree.interval.right
        res = prune_tree(tree, emissions_for(emissions, left, right), Q, node_time, pi)
        for s in samples:
            post = pick(res, s, pi)
            status = MISSING_INFO if s in res.missing_info else INFORMATIVE
            segs = tracks[s]
            if (segs and segs[-1].right == left and segs[-1].status == status
                    and np.allclose(segs[-1].posterior, post, atol=merge_tol, rtol=0)):
                segs[-1].right = right
            else:
                segs.append(Segment(left, right, np.array(post, float), status))
    return tracks


def posterior_table(ts, Q, pi, emissions, focal=None, merge_tol=1e-12, tree_range=None,
                    progress=False):
    """Per-sample ancestry posterior as contiguous segments covering ``[0, L)``.

    Runs the down-pass per marginal tree and records each focal sample's posterior
    and info-status; adjacent identical segments are merged.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence whose marginal trees are pruned.
    Q : (K, K) numpy.ndarray
        Ancestry CTMC generator.
    pi : (K,) array_like
        Root frequencies ``π`` (the prior fallback on uninformative spans).
    emissions : dict[int, numpy.ndarray]
        Per-sample emission vector (see :func:`tspaint.em.build_emissions`).
    focal : iterable[int], optional
        Samples to record; defaults to every sample in ``ts``.
    merge_tol : float, optional
        Absolute tolerance for merging adjacent segments with equal posterior.
        Default ``1e-12``.
    tree_range : tuple[int, int], optional
        Half-open ``(lo, hi)`` range of marginal-tree indices to paint; trees outside it are
        skipped. Default ``None`` → the whole genome (``(0, ts.num_trees)``). Enables the
        chunked / parallel pass: since each segment's posterior comes from its own tree's
        pruning (independent of which trees a chunk covers), concatenating the per-range tracks
        in genome order and re-merging at the seams reproduces the full-genome result exactly
        (:func:`tspaint.parallel.posterior_table_parallel`).
    progress : bool, optional
        Show a per-marginal-tree :mod:`tqdm` progress bar (full-genome pass only).
        Default ``False``.

    Returns
    -------
    dict[int, list[Segment]]
        Per focal sample, the down-pass posterior as contiguous
        :class:`Segment`\\ s covering ``[0, L)``.
    """
    return _paint_tracks(ts, Q, pi, emissions, focal, merge_tol,
                         lambda res, s, prior: res.gamma[s], tree_range=tree_range,
                         progress=progress)


def loo_posterior_table(ts, Q, pi, emissions, focal=None, merge_tol=1e-12, tree_range=None,
                        progress=False):
    """Per-sample **leave-one-out** ancestry posterior as contiguous segments.

    Like :func:`posterior_table` but paints the *outside message* — what the rest of
    the tree says about each focal sample's ancestry, **excluding that sample's own
    emission** (``PruneResult.loo``). For a labelled reference this is the
    introgression / mislabel map: where its own genealogy dissents from its label
    (CLAUDE.md §2.3, §9). Unlike the down-pass :func:`posterior_table`, it is *not*
    suppressed by a confident (e.g. hard-clamped, one-hot) tip emission, so it surfaces a
    reference's foreign tracts even where the down-pass posterior is pinned to the label.

    Parameters
    ----------
    ts, Q, pi, emissions, focal, merge_tol, tree_range, progress
        As for :func:`posterior_table` (``progress`` shows a per-marginal-tree bar for the
        full-genome pass). For a parallel leave-one-out paint use
        :func:`tspaint.parallel.loo_posterior_table_parallel`.

    Returns
    -------
    dict[int, list[Segment]]
        Per focal sample, the leave-one-out posterior as contiguous
        :class:`Segment`\\ s covering ``[0, L)``.
    """
    return _paint_tracks(ts, Q, pi, emissions, focal, merge_tol,
                         lambda res, s, prior: res.loo.get(s, prior), tree_range=tree_range,
                         progress=progress)


def missing_info_mask(ts, focal=None):
    """Per-sample spans where the tree is uninformative (isolated sample).

    Topology-only (independent of ``Q``/``π``/emissions): an isolated sample is a
    root with no children over that span (CLAUDE.md §4.2).

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to scan.
    focal : iterable[int], optional
        Samples to report; defaults to every sample in ``ts``.

    Returns
    -------
    dict[int, list[tuple[float, float]]]
        Per sample, the ``(left, right)`` spans where it is an isolated root.
    """
    samples = [int(s) for s in (ts.samples() if focal is None else focal)]
    mask = {s: [] for s in samples}
    for tree in ts.trees():
        left, right = tree.interval.left, tree.interval.right
        roots = set(tree.roots)
        for s in samples:
            if s in roots and len(tree.children(s)) == 0:
                spans = mask[s]
                if spans and spans[-1][1] == left:
                    spans[-1] = (spans[-1][0], right)
                else:
                    spans.append((left, right))
    return mask


def posterior_at(tracks, sample, position):
    """Posterior vector for ``sample`` at a genomic ``position``.

    Parameters
    ----------
    tracks : dict[int, list[Segment]]
        Per-sample segment tracks (e.g. from :func:`posterior_table`).
    sample : int
        Sample-node id to look up.
    position : float
        Genomic position.

    Returns
    -------
    numpy.ndarray or None
        The ``(K,)`` posterior on the segment covering ``position``, or ``None``
        if no segment covers it.
    """
    for seg in tracks[int(sample)]:
        if seg.left <= position < seg.right:
            return seg.posterior
    return None


#: The documented default confidence dead-band for the **high-level** segmentation surface
#: (:meth:`tspaint.Painting.segments` / :func:`tspaint.paint` / the plots / the benchmark scorer).
#: ``0.4`` is the value CLAUDE.md §9 recommends for admixture-pulse dating: it suppresses the
#: low-confidence (~P=0.5) ``argmax`` flips that fragment long tracts ~3× under naive segmentation
#: while a calibrated posterior keeps the true switches. The low-level primitive
#: :func:`hard_segments` keeps its own neutral default of ``0.0`` (pure ``argmax``); the objects and
#: :func:`tspaint.paint` supply :data:`DEFAULT_DEADBAND`. Override anywhere with ``deadband=``.
DEFAULT_DEADBAND = 0.4

#: The default dead-band for the **introgression / reference-QC** read-out
#: (:meth:`tspaint.introgression.ReferenceQC.summary` / ``.mask`` / ``.flagged_tracts``). This is a **different
#: quantity** from :data:`DEFAULT_DEADBAND`: it gates the leave-one-out *dissent margin*
#: ``loo[foreign] - loo[label]`` (how strongly a reference's own genealogy disowns its label),
#: not the top-two segmentation margin — hence its own value (``0.3``). Override with ``deadband=``.
DEFAULT_QC_DEADBAND = 0.3


def hard_segments(track, deadband=0.0):
    """Collapse a soft posterior ``track`` into hard ancestry segments.

    Produces ``[(left, right, state), ...]`` — the object a downstream tract-length /
    admixture-pulse dating analysis consumes.

    Parameters
    ----------
    track : list[Segment]
        Soft posterior segments for one sample (e.g. from :func:`posterior_table`).
    deadband : float, optional
        Confidence dead-band on the top-two posterior **margin** ``max(P) - 2nd-max(P)``.
        It **confirms** switches: a locus is a *confident anchor* only where
        ``margin >= deadband``; a tract boundary is kept only between two confident anchors
        of different states (a low-confidence run that dips and returns to the same confident
        state is flicker and is suppressed). Default ``0.0`` (every ``argmax`` change is a
        boundary) for this low-level primitive; the high-level surface
        (:meth:`tspaint.Painting.segments`, :func:`tspaint.paint`, the plots) defaults to
        :data:`DEFAULT_DEADBAND`.

    Returns
    -------
    list[tuple[float, float, int]]
        Hard ``(left, right, state)`` ancestry segments.

    Notes
    -----
    **Reversal-invariant** (a chromosome has no privileged left/right): the deadband only
    decides *whether* a switch is real, while the boundary is placed at the **argmax
    crossover** (where the two states' posteriors cross) — a direction-free point in the
    data — rather than at the first confident locus reached by the scan. So segmenting the
    track or its mirror image yields the same physical borders. (Contrast a one-pass causal
    filter, which would attribute a blurred switch to whichever confident tract precedes it
    in the scan direction.) When a sub-deadband run separates two *different* confident
    anchors, the border is the argmax-change boundary nearest the run's midpoint (unique for
    a monotone transition, the usual case).

    A positive ``deadband`` suppresses the low-confidence (~P=0.5) flips that fragment long
    tracts under naive ``argmax`` (``deadband=0``). Because the posterior is calibrated, a
    modest dead-band (≈0.3–0.5) recovers the true switch density and tract-length
    distribution where naive argmax over-fragments ~3× (CLAUDE.md §9) — a tunable
    precision/recall dial a fixed hard segmenter (e.g. RFMix's CRF) does not expose.
    ``MISSING_INFO`` spans are never anchors (missing ≠ a confident opposite call) and are
    filled by the nearest confident anchor; with no confident anchor at all the track falls
    back to raw per-locus ``argmax``.
    """
    segs = list(track)
    n = len(segs)
    if n == 0:
        return []

    amax = np.empty(n, dtype=int)
    conf = np.zeros(n, dtype=bool)
    psum = np.zeros_like(np.asarray(segs[0].posterior, float))
    for i, seg in enumerate(segs):
        p = np.asarray(seg.posterior, float)
        psum += p
        amax[i] = int(np.argmax(p))
        order = np.sort(p)
        conf[i] = (seg.status != MISSING_INFO) and (float(order[-1] - order[-2]) >= deadband)

    state = np.empty(n, dtype=int)
    anchors = np.flatnonzero(conf)
    if anchors.size == 0:
        # nothing clears the dead-band -> no switch is confirmed -> one tract of the most probable
        # state overall (reversal-invariant: a sum over loci, not the scan-first call).
        state[:] = int(np.argmax(psum))
    else:
        first, last = int(anchors[0]), int(anchors[-1])
        state[: first + 1] = amax[first]             # leading run -> first confident state
        state[last:] = amax[last]                    # trailing run -> last confident state
        for a, b in zip(anchors[:-1], anchors[1:]):
            a, b, sa, sb = int(a), int(b), int(amax[a]), int(amax[b])
            if sa == sb:
                state[a : b + 1] = sa                # dip returning to the same state -> flicker
                continue
            # a real switch: border at the argmax crossover in (a, b] nearest the sub-deadband run's
            # midpoint (reversal-invariant; unique for a monotone crossing, the usual case).
            mid = 0.5 * (segs[a].right + segs[b].left)
            changes = [k for k in range(a + 1, b + 1) if amax[k] != amax[k - 1]]
            dists = [abs(segs[k].left - mid) for k in changes]
            dmin = min(dists)
            tied = [k for k, d in zip(changes, dists) if d <= dmin + 1e-9]
            if len(tied) == 1:
                k = tied[0]
                state[a:k] = sa
                state[k : b + 1] = sb
            else:
                # crossings equidistant from the centre (symmetric): split each gap locus by which
                # side of the midpoint it falls on — direction-free (a position tie-break would not be).
                lo = min(sa, sb)
                for j in range(a, b + 1):
                    c = 0.5 * (segs[j].left + segs[j].right)
                    state[j] = sa if c < mid else (sb if c > mid else lo)

    out = []
    for i, seg in enumerate(segs):
        st = int(state[i])
        if out and out[-1][2] == st and out[-1][1] == seg.left:
            out[-1] = (out[-1][0], seg.right, st)
        else:
            out.append((seg.left, seg.right, st))
    return out
