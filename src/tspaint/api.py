"""High-level entry point for tspaint (CLAUDE.md §2.4).

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
        :class:`~tspaint.output.Segment`\\ s covering ``[0, L)`` (the soft, calibrated
        deliverable).
    Q : numpy.ndarray
        The fitted generator.
    pi : numpy.ndarray
        The fitted root frequencies.
    w : dict
        The learned per-tip credibility.
    loglik_history : list
        Observed-data log-likelihood per EM step (non-decreasing).
    queries : list
        The painted sample-node ids.
    ts : tskit.TreeSequence
        The tree sequence painted (retained so :meth:`rate_through_time` can reuse the fit;
        already in memory, so no extra cost).
    labels : dict[int, int]
        The reference labels used for the fit (retained for :meth:`rate_through_time`).
    default_deadband : float
        Default dead-band passed to :meth:`segments`. Default ``0.0``.
    """
    posteriors: dict
    Q: np.ndarray
    pi: np.ndarray
    w: dict
    loglik_history: list
    queries: list
    ts: object = None
    labels: dict = None
    default_deadband: float = 0.0

    def segments(self, deadband=None):
        """Hard ancestry segments for downstream tract-length / dating analysis.

        Parameters
        ----------
        deadband : float, optional
            Confidence dead-band suppressing low-confidence flips that fragment long
            tracts. Defaults to :attr:`default_deadband`. See
            :func:`tspaint.output.hard_segments`.

        Returns
        -------
        dict[int, list[tuple[float, float, int]]]
            Per query, hard ``(left, right, state)`` segments.
        """
        db = self.default_deadband if deadband is None else deadband
        return {q: hard_segments(t, db) for q, t in self.posteriors.items()}

    def posterior_at(self, sample, position):
        """Posterior vector for ``sample`` at a genomic ``position``.

        Parameters
        ----------
        sample : int
            Sample-node id to look up.
        position : float
            Genomic position.

        Returns
        -------
        numpy.ndarray or None
            The ``(K,)`` posterior covering ``position``, or ``None`` if uncovered.
        """
        return posterior_at(self.posteriors, sample, position)

    def rate_through_time(self, edges=None, **kwargs):
        """Estimate the admixture (cross-ancestry) rate through time, reusing this fit.

        Fits the time-inhomogeneous directional mugration EM
        (:func:`tspaint.fit_rate_through_time`) **warm-started from this painting's fitted
        ``(Q, π, w)``**, so the homogeneous fit is not repeated. This is a *different
        deliverable* from the painting — the cross-ancestry transition rates ``q_AB(t)``,
        ``q_BA(t)`` as functions of (backward) time, locating divergence / gene-flow epochs and
        their direction. It returns a **new** :class:`~tspaint.dating.RateThroughTime` and does
        **not** modify :attr:`posteriors` (CLAUDE.md: Q(t) gives no painting-accuracy gain, so the
        paths stay side by side).

        Parameters
        ----------
        edges : array_like, optional
            Log-time grid edges; an auto grid is built from the node ages when ``None``.
        **kwargs
            Forwarded to :func:`tspaint.fit_rate_through_time` (e.g. ``n_cells``, ``n_iter``,
            ``n_knots``).

        Returns
        -------
        tspaint.dating.RateThroughTime
            The directional rate-through-time profile (``.centers``, ``.q_AB``, ``.q_BA``,
            ``.plot()``).
        """
        if self.ts is None or self.labels is None:
            raise ValueError("Painting was constructed without ts/labels; cannot date. Use "
                             "tspaint.fit_rate_through_time(ts, labels) directly.")
        from .em import FitResult
        from .dating import fit_rate_through_time
        warm = FitResult(self.Q, self.pi, self.w, self.loglik_history)
        return fit_rate_through_time(self.ts, self.labels, edges, fit_result=warm, **kwargs)

    def __repr__(self):
        return (f"Painting(queries={len(self.queries)}, K={self.pi.shape[0]}, "
                f"Q={np.array2string(self.Q, precision=2)}, "
                f"pi={np.array2string(self.pi, precision=2)})")


def paint(ts, labels, queries=None, *, K=2, soft_refs=None, estimate_pi=False, deadband=0.0,
          smooth=False, epsilon=1e-2, Q0=None, max_iter=12, tol=1e-7, alpha=20.0, beta=1.0,
          w0=0.9):
    """Infer soft local ancestry along query haplotypes from a tree sequence.

    EM-fits the ancestry CTMC ``(Q[, π, per-tip credibility w])`` on the labelled reference tips
    (:func:`tspaint.fit`), then returns the per-position posterior over ancestry states for each
    query haplotype as a :class:`Painting`.

    Parameters
    ----------
    ts : tskit.TreeSequence
        An inferred (tsinfer / Relate ``--compress``) or true tree sequence; sample nodes are
        haplotypes. Use :mod:`tspaint.io` to obtain one from genotypes.
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
    smooth : bool
        Apply the horizontal BP/EP smoother (:mod:`tspaint.bp`) to the posteriors along the
        genome. **Recommended on inferred (tsinfer / Relate) ARGs**, where tree inference
        scatters spurious breakpoints a per-position deadband cannot filter; redundant on a
        true/known ARG (CLAUDE.md §7). Default ``False``.
    epsilon : float
        Per-breakpoint switch penalty for ``smooth`` (smaller ⇒ more smoothing).
    Q0 : (K, K) array, optional
        Initial generator (default a slow symmetric 2-state generator).
    max_iter, tol, alpha, beta, w0 : EM controls (see :func:`tspaint.fit`).

    Returns
    -------
    Painting
        The soft per-position ancestry posteriors for each query plus the fitted
        ``(Q, π, w)`` and EM log-likelihood history.

    See Also
    --------
    tspaint.fit : The underlying blocked-EM fit.
    tspaint.output.hard_segments : Collapse the soft posteriors into hard tracts.

    Examples
    --------
    >>> import tspaint
    >>> ts = tspaint.simulate_admixture(n_admix=10, n_ref=10)
    >>> labels = {0: 0, 1: 0, 2: 1, 3: 1}   # reference sample-node -> ancestry state
    >>> painting = tspaint.paint(ts, labels)
    >>> painting.segments(deadband=0.4)      # hard ancestry tracts for dating
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
    if smooth:
        from .bp import bp_smooth_track
        posteriors = {q: bp_smooth_track(t, res.pi, epsilon) for q, t in posteriors.items()}
    return Painting(posteriors=posteriors, Q=res.Q, pi=res.pi, w=res.w,
                    loglik_history=res.loglik_history, queries=queries,
                    ts=ts, labels=labels, default_deadband=deadband)
