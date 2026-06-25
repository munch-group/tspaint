"""Single-pass horizontal BP/EP smoother (CLAUDE.md §7).

Blocked EM (§3) computes each tip's posterior per marginal tree *independently* and folds the
horizontal (along-genome) coupling into the edge-blocking — it never propagates *uncertainty*
across persist-but-reparent breakpoints (§3.5). The §9 fragmentation finding is that omission
surfacing in the hard segmentation: at older admixture the per-tree posterior wobbles near 0.5
and `argmax` flips, over-fragmenting tracts (which biases admixture-pulse dating older).

This module adds the missing propagation: for each query tip, a genome-axis forward-backward
over the per-tree beliefs ``gamma_t`` (the vertical / pruning evidence) with a per-breakpoint
near-identity transition (switch penalty ``epsilon``). A run of weak-but-consistent evidence
reinforces (so a real but low-confidence switch is recovered), while an isolated low-confidence
flip is smoothed away — using neighbour evidence the per-position `output.hard_segments`
deadband cannot.

This is the **single-pass EP approximation**: the vertical evidence is fixed and the horizontal
sweep runs once (the first half of §7.2's schedule, factorised per tip). Full loopy BP would
iterate the smoothed tip beliefs back into the vertical pruning of the shared internal nodes;
that is the deferred extension (:func:`bp_paint` exposes ``n_sweeps`` as the hook).
"""
from __future__ import annotations

import numpy as np

from ..em import fit, build_emissions
from ..model import make_generator_2state
from ..output import posterior_table, Segment

__all__ = ["bp_smooth", "bp_smooth_track", "bp_paint"]


def bp_smooth(emissions, pi, epsilon):
    """Forward-backward over per-segment beliefs along the genome.

    Parameters
    ----------
    emissions : array_like, shape (T, K)
        Per-segment beliefs ``gamma_t`` (the vertical / pruning evidence) for one tip.
    pi : array_like, shape (K,)
        Per-tree prior, divided out so ``gamma_t`` enters as a likelihood (a no-op
        constant when ``pi`` is uniform, the painter default).
    epsilon : float
        Per-breakpoint switch probability of the near-identity transition
        ``A = (1 - epsilon) I + epsilon / K``: ``epsilon -> 0`` is maximal smoothing
        (one tract), ``epsilon -> (K-1)/K`` is no coupling (recovers the per-segment
        input).

    Returns
    -------
    numpy.ndarray, shape (T, K)
        The horizontally-smoothed posterior.
    """
    em = np.asarray(emissions, float)
    T, K = em.shape
    pi = np.asarray(pi, float)
    like = em / pi
    like = like / like.sum(1, keepdims=True)
    A = (1.0 - epsilon) * np.eye(K) + epsilon / K * np.ones((K, K))
    alpha = np.zeros((T, K))
    alpha[0] = like[0] * pi
    alpha[0] /= alpha[0].sum()
    for t in range(1, T):
        alpha[t] = like[t] * (alpha[t - 1] @ A)
        alpha[t] /= alpha[t].sum()
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (like[t + 1] * beta[t + 1])
        beta[t] /= beta[t].sum()
    g = alpha * beta
    g /= g.sum(1, keepdims=True)
    return g


def bp_smooth_track(track, pi, epsilon):
    """Smooth one tip's :class:`~tslai.output.Segment` track along the genome.

    Intervals and status are preserved; posteriors are replaced by the
    horizontally-smoothed ones.

    Parameters
    ----------
    track : list[Segment]
        One tip's piecewise-constant painting.
    pi : array_like, shape (K,)
        Per-tree prior (see :func:`bp_smooth`).
    epsilon : float
        Per-breakpoint switch probability (see :func:`bp_smooth`).

    Returns
    -------
    list[Segment]
        The track with smoothed posteriors (a copy of ``track`` if it has < 2
        segments).
    """
    if len(track) < 2:
        return list(track)
    em = np.array([seg.posterior for seg in track], float)
    g = bp_smooth(em, pi, epsilon)
    return [Segment(seg.left, seg.right, g[i], seg.status) for i, seg in enumerate(track)]


def bp_paint(ts, labels, queries, K=2, *, epsilon=1e-2, max_iter=6, Q0=None,
             soft_refs=None, estimate_pi=False, n_sweeps=1):
    """tslai painter with horizontal BP smoothing.

    EM-fits ``(Q[, pi, w])``, paints the queries, then smooths each tip's track
    along the genome (switch penalty ``epsilon``) — adding the propagation of
    uncertainty across persist-but-reparent breakpoints that blocked EM omits (§7).

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence to paint.
    labels : dict[int, int]
        Map from reference sample-node id to ancestry state.
    queries : iterable[int]
        Sample-node ids of the haplotypes to paint.
    K : int, optional
        Number of ancestry states (default 2).
    epsilon : float, optional
        Per-breakpoint switch penalty for the horizontal smoother (default ``1e-2``).
    max_iter : int, optional
        Maximum EM iterations (default 6).
    Q0 : numpy.ndarray, optional
        Initial generator; defaults to ``make_generator_2state(1e-3, 1e-3)``.
    soft_refs : iterable[int], optional
        Reference sample ids whose label credibility ``w`` is learned (the rest are
        hard-clamped anchors).
    estimate_pi : bool, optional
        Whether to estimate the root frequencies ``pi`` (default False; see §6).
    n_sweeps : int, optional
        Reserved for the full-loopy extension (re-feeding smoothed beliefs into the
        vertical pass); only the single-pass ``n_sweeps=1`` is implemented (default 1).

    Returns
    -------
    dict[int, list[Segment]]
        ``{query_node: [Segment]}`` with the smoothed posteriors, duck-compatible
        with the :mod:`tslai.validate` metrics and :func:`tslai.output.hard_segments`.

    Raises
    ------
    NotImplementedError
        If ``n_sweeps != 1`` (the full-loopy extension is deferred).
    """
    if n_sweeps != 1:
        raise NotImplementedError("only the single-pass (n_sweeps=1) EP smoother is implemented")
    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    res = fit(ts, labels, K=K, Q0=Q0, max_iter=max_iter, soft_refs=soft_refs,
              estimate_pi=estimate_pi)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    soft = posterior_table(ts, res.Q, res.pi, emissions, focal=queries)
    return {q: bp_smooth_track(soft[q], res.pi, epsilon) for q in soft}
