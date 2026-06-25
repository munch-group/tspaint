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
