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
    """2-state ancestry generator ``Q = [[-q_AB, q_AB], [q_BA, -q_BA]]`` (CLAUDE.md §2.1)."""
    return np.array([[-q_AB, q_AB], [q_BA, -q_BA]], float)


def validate_generator(Q, atol=1e-9):
    """Check ``Q`` is a valid CTMC generator: square, rows sum to 0, off-diag >= 0."""
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
    """Branch transition probabilities ``P(t) = expm(Q t)`` (CLAUDE.md §2.1)."""
    return expm(np.asarray(Q, float) * t)


def stationary_distribution(Q):
    """Stationary ``π`` solving ``π Q = 0``, ``π >= 0``, ``Σ π = 1``.

    Solved as the normalised left null vector via an augmented least-squares system.
    """
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    A = np.vstack([Q.T, np.ones(K)])
    b = np.concatenate([np.zeros(K), [1.0]])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def tip_emission(label, w, pi):
    """Soft-clamp emission for a labelled tip (CLAUDE.md §2.2)::

        e(s) = w * 1[s == label] + (1 - w) * pi(s)

    ``w = 1`` -> hard clamp (one-hot); ``w -> 0`` -> the tip is effectively
    re-inferred from the rest of the tree (like a query).
    """
    pi = np.asarray(pi, float)
    e = (1.0 - w) * pi.copy()
    e[label] += w
    return e


def query_emission(pi):
    """Flat / root-frequency emission for an unlabelled query tip (CLAUDE.md §2.2)."""
    return np.asarray(pi, float).copy()
