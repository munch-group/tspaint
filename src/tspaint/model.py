"""Ancestry CTMC model: generator ``Q``, root frequencies ``π``, tip emission and
the soft per-tip credibility noise model (CLAUDE.md §2).

Generator-agnostic by design: 2-state today, K-way by swapping the generator.
Nothing here touches tskit — these are pure array operations.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = [
    "make_generator_2state",
    "make_generator_symmetric",
    "validate_generator",
    "transition_matrix",
    "stationary_distribution",
    "tip_emission",
    "query_emission",
]


def make_generator_2state(q_AB, q_BA):
    """Build the 2-state ancestry CTMC generator (CLAUDE.md §2.1).

    Parameters
    ----------
    q_AB : float
        Instantaneous A->B rate.
    q_BA : float
        Instantaneous B->A rate.

    Returns
    -------
    numpy.ndarray
        The ``(2, 2)`` generator ``Q = [[-q_AB, q_AB], [q_BA, -q_BA]]``.

    Examples
    --------
    >>> make_generator_2state(1e-3, 1e-3)
    array([[-0.001,  0.001],
           [ 0.001, -0.001]])
    """
    return np.array([[-q_AB, q_AB], [q_BA, -q_BA]], float)


def make_generator_symmetric(K, rate):
    """Symmetric ``K``-state ancestry CTMC generator (CLAUDE.md §2.1, K-way).

    Every ordered state pair ``m != n`` gets rate ``rate / (K - 1)``, so each state's **total exit
    rate is ``rate``** — this equals :func:`make_generator_2state` at ``K = 2``
    (``make_generator_symmetric(2, r)`` is ``make_generator_2state(r, r)``). The default initial
    generator :func:`tspaint.fit` uses for a K-way fit: a slow, source-symmetric start with no
    built-in preference between ancestries.

    Parameters
    ----------
    K : int
        Number of ancestry states (``>= 2``).
    rate : float
        Total per-state exit rate.

    Returns
    -------
    numpy.ndarray
        The ``(K, K)`` symmetric generator (rows sum to zero, off-diagonals equal).

    Raises
    ------
    ValueError
        If ``K < 2``.

    Examples
    --------
    >>> make_generator_symmetric(3, 1e-3)
    array([[-0.001 ,  0.0005,  0.0005],
           [ 0.0005, -0.001 ,  0.0005],
           [ 0.0005,  0.0005, -0.001 ]])
    """
    K = int(K)
    if K < 2:
        raise ValueError("K must be >= 2")
    Q = np.full((K, K), float(rate) / (K - 1), float)
    np.fill_diagonal(Q, -float(rate))
    return Q


def validate_generator(Q, atol=1e-9):
    """Check that ``Q`` is a valid CTMC generator.

    Validates that ``Q`` is square, its rows sum to zero, and its off-diagonal
    rates are non-negative.

    Parameters
    ----------
    Q : array_like
        Candidate ``(K, K)`` generator matrix.
    atol : float, optional
        Absolute tolerance for the row-sum and non-negativity checks.

    Returns
    -------
    numpy.ndarray
        ``Q`` as a float array (unchanged), once validated.

    Raises
    ------
    ValueError
        If ``Q`` is not square, its rows do not sum to zero, or any
        off-diagonal rate is negative.
    """
    Q = np.asarray(Q, float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    if not np.allclose(Q.sum(axis=1), 0.0, atol=atol):
        raise ValueError("generator rows must sum to zero")
    off = Q.copy()
    np.fill_diagonal(off, 0.0)
    if np.any(off < -atol):
        raise ValueError("off-diagonal rates must be non-negative")
    return Q


def transition_matrix(Q, t):
    """Branch transition probabilities ``P(t) = expm(Q t)`` (CLAUDE.md §2.1).

    Parameters
    ----------
    Q : array_like
        ``(K, K)`` CTMC generator.
    t : float
        Branch length (time elapsed along the branch).

    Returns
    -------
    numpy.ndarray
        ``(K, K)`` transition-probability matrix over a branch of length ``t``.
    """
    return expm(np.asarray(Q, float) * t)


def stationary_distribution(Q):
    """Stationary distribution ``π`` of a CTMC generator.

    Solves ``π Q = 0`` subject to ``π >= 0`` and ``Σ π = 1`` as the normalised
    left null vector via an augmented least-squares system.

    Parameters
    ----------
    Q : array_like
        ``(K, K)`` CTMC generator.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` stationary distribution (non-negative, sums to 1).
    """
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    A = np.vstack([Q.T, np.ones(K)])
    b = np.concatenate([np.zeros(K), [1.0]])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def tip_emission(label, w, pi):
    """Soft-clamp emission for a labelled tip (CLAUDE.md §2.2).

    The emission is the credibility-weighted noise model
    ``e(s) = w * 1[s == label] + (1 - w) * pi(s)``: ``w = 1`` gives a hard clamp
    (one-hot), while ``w -> 0`` lets the tip be effectively re-inferred from the
    rest of the tree (like a query).

    Parameters
    ----------
    label : int
        Observed ancestry-state index for the tip.
    w : float
        Per-tip credibility in ``[0, 1]``.
    pi : array_like
        ``(K,)`` root frequencies used as the noise fallback.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` Felsenstein emission likelihood vector for the tip.
    """
    pi = np.asarray(pi, float)
    e = (1.0 - w) * pi.copy()
    e[label] += w
    return e


def query_emission(pi):
    """Flat / root-frequency emission for an unlabelled query tip (CLAUDE.md §2.2).

    Parameters
    ----------
    pi : array_like
        ``(K,)`` root frequencies, used directly as the query tip's emission.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` emission likelihood vector (a copy of ``pi``).
    """
    return np.asarray(pi, float).copy()


class MaskedEmissions:
    """Position-dependent Felsenstein emissions — **fragment masking** (CLAUDE.md §2.3).

    Wraps the base per-tip emission vectors and, for the marginal tree covering a genomic interval,
    returns them with each *masked* reference tip switched to the query (unlabelled) emission over
    its masked spans. So a contaminated reference anchors at full strength on its clean spans and is
    treated as a query (contributes no label information) exactly where it is flagged foreign —
    the local generalisation of the global per-tip credibility ``w_i``.

    Parameters
    ----------
    base : dict[int, numpy.ndarray]
        Base per-tip emission vectors (:func:`build_emissions`).
    mask : dict[int, list[tuple[float, float]]]
        Per reference-node the half-open ``[left, right)`` spans to mask out (unlabel).
    pi : array_like
        Root frequencies for the query emission used on masked spans.

    Notes
    -----
    A reference is masked for a tree if the tree interval's **midpoint** falls in one of its spans
    (tracts span many marginal trees, so the ≤1-interval boundary rounding is negligible). The
    per-interval emission depends only on that interval, so the E-step stays exactly summable over
    tree-range chunks (the byte-exact parallel property is preserved).
    """

    def __init__(self, base, mask, pi):
        self.base = base
        self._q = query_emission(pi)
        # accept (left, right) or (left, right, score) spans (ReferenceQC.mask / foreign_tracts)
        self.mask = {int(r): sorted((float(s[0]), float(s[1])) for s in spans)
                     for r, spans in (mask or {}).items() if spans}
        self._active = None                      # cache: (frozenset masked-here, overlay dict)

    def for_interval(self, left, right):
        """The emission dict for the marginal tree covering ``[left, right)``.

        Parameters
        ----------
        left, right : float
            Half-open genomic interval ``[left, right)`` of the marginal tree.

        Returns
        -------
        dict[int, numpy.ndarray]
            The per-tip emission dict for this interval: ``base`` with every reference masked
            here switched to the query (unlabelled) emission. Returns the ``base`` dict itself
            unchanged (an identity fast path) when the mask is empty or nothing is masked over
            this interval.

        Notes
        -----
        A reference is masked here iff the interval **midpoint** ``0.5 * (left + right)`` falls
        in one of its masked spans (the ≤1-interval boundary rounding is negligible against
        tract lengths). The returned dict is shared — the ``base`` dict on the fast path, else
        a cached overlay reused across identical intervals — so the caller must **not** mutate
        it.
        """
        if not self.mask:
            return self.base
        mid = 0.5 * (float(left) + float(right))
        here = frozenset(r for r, spans in self.mask.items()
                         if any(l <= mid < rr for (l, rr) in spans))
        if not here:
            return self.base
        if self._active is not None and self._active[0] == here:
            return self._active[1]               # unchanged since the last interval — reuse overlay
        overlay = dict(self.base)
        for r in here:
            overlay[r] = self._q
        self._active = (here, overlay)
        return overlay


def emissions_for(emissions, left, right):
    """The emission dict for a marginal tree interval — position-dependent for a
    :class:`MaskedEmissions`, else the plain ``emissions`` dict unchanged (backward compatible).

    Parameters
    ----------
    emissions : dict[int, numpy.ndarray] or MaskedEmissions
        A plain per-tip emission dict, or a :class:`MaskedEmissions`. Duck-typed: any object
        exposing a ``for_interval(left, right)`` method is dispatched to it, anything else is
        returned unchanged.
    left, right : float
        Half-open genomic interval ``[left, right)`` of the marginal tree.

    Returns
    -------
    dict[int, numpy.ndarray]
        For a :class:`MaskedEmissions`, ``emissions.for_interval(left, right)`` (the
        per-interval dict); for a plain dict, ``emissions`` itself unchanged.
    """
    fn = getattr(emissions, "for_interval", None)
    return fn(left, right) if fn is not None else emissions
