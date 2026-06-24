"""Per-branch endpoint-conditioned CTMC sufficient statistics (CLAUDE.md §3.2).

Expected dwell time per state and expected jump counts per ordered pair along a
single branch of length ``t`` under generator ``Q``, conditioned on the posterior
over the branch's endpoint states ``xi``. Computed via the Van Loan
block-triangular matrix exponential (Van Loan, 1978).

This module is **generator-agnostic** (2-state today, K-way by swapping ``Q``) and
is the designated **Phasic seam** (CLAUDE.md §12): replace the ``expm`` block calls
with Phasic's reward-accumulated phase-type machinery once the interface settles,
keeping :func:`branch_expected_stats` as the stable boundary.

References
----------
* Van Loan (1978) — block-triangular matrix exponential for these integrals.
* Hobolth & Jensen (2011), *J. Appl. Probab.* 48(4):911-924 — summary statistics
  for endpoint-conditioned CTMCs (expected dwell times & jump counts).
* Tataru & Hobolth (2011), *BMC Bioinformatics* 12:465 — compares EXPM (Van Loan),
  eigendecomposition, and uniformization for exactly these conditional
  expectations; uniformization/eigendecomposition are the documented fallbacks if
  ``expm`` is unstable at small or stiff ``Q*t`` (CLAUDE.md §8.7).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = ["branch_expected_stats", "branch_kernel", "stats_from_kernel", "vanloan_integral"]


def vanloan_integral(Q, t, E):
    """Top-right block of ``expm([[Q, E], [0, Q]] * t)``.

    Equals ``∫_0^t expm(Q τ) E expm(Q (t-τ)) dτ`` (Van Loan, 1978). For a reward
    indicator ``E`` this is the joint (un-normalized) expected reward together with
    the branch endpoints.
    """
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    block = np.zeros((2 * K, 2 * K))
    block[:K, :K] = Q
    block[K:, K:] = Q
    block[:K, K:] = E
    return expm(block * t)[:K, K:]


def branch_expected_stats(Q, t, xi):
    """Endpoint-conditioned expected dwell times and jump counts on one branch.

    Parameters
    ----------
    Q : (K, K) array_like
        CTMC generator (rows sum to zero).
    t : float
        Branch length (> 0). Root branches (length 0 by tskit convention) must be
        skipped by the caller (CLAUDE.md §3.4, §4.5); ``t <= 0`` returns zeros.
    xi : (K, K) array_like
        Posterior over ``(parent_state, child_state)`` for this branch, normalised
        to sum to 1. The branch's span weight is applied separately by the
        accumulator (CLAUDE.md §3.3).

    Returns
    -------
    dwell : (K,) ndarray
        Expected time spent in each state along the branch, under ``xi``. Sums to
        ``t`` for any normalised ``xi`` (a useful invariant).
    jumps : (K, K) ndarray
        Expected number of ``m -> n`` transitions (``m != n``), under ``xi``.
    """
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    kernel = branch_kernel(Q, t)
    if kernel is None:
        return np.zeros(K), np.zeros((K, K))
    return stats_from_kernel(kernel, xi)


def branch_kernel(Q, t):
    """The ``(Q, t)``-dependent part of :func:`branch_expected_stats` — the per-reward
    conditional-expectation matrices ``E[reward | s_p, s_c]``. **Cacheable by ``t``**
    (the per-branch posterior ``xi`` is applied separately, cheaply, via
    :func:`stats_from_kernel`), so a sweep computes the Van Loan ``expm`` once per
    distinct branch length rather than once per edge (CLAUDE.md §3.3). Returns ``None``
    for ``t <= 0`` (root branches are skipped by the caller, §3.4)."""
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    if t <= 0:
        return None
    P = expm(Q * t)
    # Conditioning divides the joint Van Loan integral by the endpoint transition
    # probability. Guard underflowed entries; a consistent xi puts ~0 mass there.
    Psafe = np.where(P > 1e-300, P, 1e-300)

    dwell_cond = []
    for m in range(K):
        E = np.zeros((K, K))
        E[m, m] = 1.0
        dwell_cond.append(vanloan_integral(Q, t, E) / Psafe)   # E[time in m | s_p, s_c]

    jump_cond = {}
    for m in range(K):
        for n in range(K):
            if m == n or Q[m, n] == 0.0:
                continue
            E = np.zeros((K, K))
            E[m, n] = Q[m, n]
            jump_cond[(m, n)] = vanloan_integral(Q, t, E) / Psafe  # E[# m->n jumps | s_p, s_c]

    return (dwell_cond, jump_cond, K)


def stats_from_kernel(kernel, xi):
    """Apply the endpoint posterior ``xi`` to a cached :func:`branch_kernel` to get
    expected dwell times and jump counts (cheap; no matrix exponentials)."""
    dwell_cond, jump_cond, K = kernel
    xi = np.asarray(xi, float)
    dwell = np.array([float(np.sum(xi * dwell_cond[m])) for m in range(K)])
    jumps = np.zeros((K, K))
    for (m, n), cond in jump_cond.items():
        jumps[m, n] = float(np.sum(xi * cond))
    return dwell, jumps
