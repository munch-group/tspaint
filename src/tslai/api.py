"""High-level entry point for tslai (CLAUDE.md §2.4).

:func:`paint` is the one call most users need: fit the ancestry CTMC on the labelled
references and return per-haplotype, per-position ancestry posteriors as a :class:`Painting`.
Everything else (the EM, pruning, sufficient statistics, metrics, comparators, I/O front ends)
is the machinery underneath, available in the submodules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .em import fit, build_emissions
from .model import make_generator_2state
from .output import posterior_table, hard_segments, posterior_at, Segment

__all__ = ["paint", "Painting"]


@dataclass
class Painting:
    """Result of :func:`paint`: the soft local-ancestry posteriors plus the fitted model.

    Attributes
    ----------
    posteriors : dict[int, list[Segment]]
        Per query haplotype, the down-pass posterior over ancestry states as contiguous
        :class:`~tslai.output.Segment`\\ s covering ``[0, L)`` (the soft, calibrated deliverable).
    Q, pi, w : the fitted generator, root frequencies, and learned per-tip credibility.
    loglik_history : observed-data log-likelihood per EM step (non-decreasing).
    queries : the painted sample-node ids.
    """
    posteriors: dict
    Q: np.ndarray
    pi: np.ndarray
    w: dict
    loglik_history: list
    queries: list
    default_deadband: float = 0.0

    def segments(self, deadband=None):
        """Hard ancestry segments ``{query: [(left, right, state)]}`` for downstream
        tract-length / dating analysis. ``deadband`` (default :attr:`default_deadband`) suppresses
        low-confidence flips that fragment long tracts — see :func:`tslai.output.hard_segments`."""
        db = self.default_deadband if deadband is None else deadband
        return {q: hard_segments(t, db) for q, t in self.posteriors.items()}

    def posterior_at(self, sample, position):
        """Posterior vector for ``sample`` at a genomic ``position`` (or ``None``)."""
        return posterior_at(self.posteriors, sample, position)

    def __repr__(self):
        return (f"Painting(queries={len(self.queries)}, K={self.pi.shape[0]}, "
                f"Q={np.array2string(self.Q, precision=2)}, "
                f"pi={np.array2string(self.pi, precision=2)})")


def paint(ts, labels, queries=None, *, K=2, soft_refs=None, estimate_pi=False, deadband=0.0,
          Q0=None, max_iter=12, tol=1e-7, alpha=20.0, beta=1.0, w0=0.9):
    """Infer soft local ancestry along query haplotypes from a tree sequence.

    EM-fits the ancestry CTMC ``(Q[, π, per-tip credibility w])`` on the labelled reference tips
    (:func:`tslai.fit`), then returns the per-position posterior over ancestry states for each
    query haplotype as a :class:`Painting`.

    Parameters
    ----------
    ts : tskit.TreeSequence
        An inferred (tsinfer / Relate ``--compress``) or true tree sequence; sample nodes are
        haplotypes. Use :mod:`tslai.io` to obtain one from genotypes.
    labels : dict[int, int]
        Reference sample-node id → ancestry-state index in ``0..K-1``.
    queries : iterable[int], optional
        Sample nodes to paint; defaults to every sample not in ``labels``.
    K : int
        Number of ancestry states (2 by default; pass a ``K×K`` ``Q0`` for K-way).
    soft_refs : set[int], optional
        Reference tips whose credibility ``w_i`` is *learned* (the rest stay hard-clamped
        anchors — never let the whole panel float, CLAUDE.md §6).
    estimate_pi : bool
        Estimate root frequencies ``π`` rather than holding them uniform. Default ``False``:
        ``π`` is a prior on the arbitrary GMRCA state and estimating it from washing deep
        branches is the degeneracy of CLAUDE.md §6; uniform is the robust choice.
    deadband : float
        Stored as :attr:`Painting.default_deadband` for :meth:`Painting.segments`.
    Q0 : (K, K) array, optional
        Initial generator (default a slow symmetric 2-state generator).
    max_iter, tol, alpha, beta, w0 : EM controls (see :func:`tslai.fit`).

    Returns
    -------
    Painting
    """
    labels = {int(k): int(v) for k, v in labels.items()}
    if queries is None:
        queries = [int(s) for s in ts.samples() if int(s) not in labels]
    else:
        queries = [int(q) for q in queries]
    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    res = fit(ts, labels, K=K, Q0=Q0, max_iter=max_iter, tol=tol, soft_refs=soft_refs,
              estimate_pi=estimate_pi, alpha=alpha, beta=beta, w0=w0)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    posteriors = posterior_table(ts, res.Q, res.pi, emissions, focal=queries)
    return Painting(posteriors=posteriors, Q=res.Q, pi=res.pi, w=res.w,
                    loglik_history=res.loglik_history, queries=queries,
                    default_deadband=deadband)
