"""Per-branch endpoint-conditioned CTMC sufficient statistics (CLAUDE.md §3.2).

Expected dwell time per state and expected jump counts per ordered pair along a
single branch of length ``t`` under generator ``Q``, conditioned on the posterior
over the branch's endpoint states ``xi``. Computed via the Van Loan
block-triangular matrix exponential (Van Loan, 1978).

This module is **generator-agnostic** (2-state today, K-way by swapping ``Q``) and
is the designated **Phasic seam** (CLAUDE.md §12):

    branch_expected_stats(Q, t, xi) -> (dwell, jumps)

is the *whole* seam — the one function the E-step calls, and the one a Phasic backend
replaces. Everything below it (the Van Loan ``expm``, the ``/P`` conditioning, the memo
on ``(Q, t)``) is a private implementation detail of *this* backend. A replacement is
free to restructure all of it: to cache on ``Q`` instead of ``t``, to batch, to cache
nothing. That freedom is the point — an earlier version of this module exposed the
cache split (``branch_kernel`` + ``stats_from_kernel``) as public API, which both
prejudged the caching strategy and left the *documented* seam dead code in the E-step.

``vanloan_integral`` stays public because :mod:`tspaint.dating.estep` needs the raw
un-normalised integral per time cell — a separate consumer with a different shape.

References
----------
* Van Loan (1978) — block-triangular matrix exponential for these integrals.
* Hobolth & Jensen (2011), *J. Appl. Probab.* 48(4):911-924 — summary statistics
  for endpoint-conditioned CTMCs (expected dwell times & jump counts).
* Tataru & Hobolth (2011), *BMC Bioinformatics* 12:465 — compares EXPM (Van Loan),
  eigendecomposition, and uniformization for exactly these conditional
  expectations. **[MEASURED]** ``expm`` is not the weak link: against a 60-digit
  ground truth it is accurate to ~1e-16 relative everywhere tspaint operates, while
  the eigendecomposition route silently loses ``eps / (lambda*t)`` — and tspaint runs
  at small ``lambda*t``. See ``examples/phasic_seam.py``.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import expm

__all__ = ["branch_expected_stats", "vanloan_integral"]

# How many (Q, t) branch kernels to memoise. Purely an implementation detail of THIS
# backend. Node times are near-continuous, so most branch lengths are distinct: on a 4 Mb
# msprime sim, 4313 edges carry 3369 distinct lengths (~1.3 edges/length), so the memo
# saves ~22% of the expm calls, not a factor of anything. Bounded, because a genome-scale
# sweep would otherwise hold millions of kernels; sibling edges that DO share a length
# (cherries, whose tips are coincident at time 0) enter together, so locality is high and
# a modest cache captures essentially all of the available reuse.
_KERNEL_CACHE_SIZE = 8192


def vanloan_integral(Q, t, E):
    """Top-right block of ``expm([[Q, E], [0, Q]] * t)``.

    Equals ``∫_0^t expm(Q τ) E expm(Q (t-τ)) dτ`` (Van Loan, 1978). For a reward
    indicator ``E`` this is the **joint** expectation of the reward and the branch's
    endpoint: ``V[i, j] = E[reward(path) · 1{X_t = j} | X_0 = i]``, equivalently
    ``E[reward | X_0=i, X_t=j] · P[i, j]``. It is *not* a probability matrix (that is
    ``P = expm(Q t)``, a factor of it) and carries the reward's units — time for a dwell
    ``E``, a count for a jump ``E``.

    Public because :mod:`tspaint.dating.estep` consumes the raw un-normalised integral
    per time cell under a time-inhomogeneous generator. The EM path does not call it
    directly; it goes through :func:`branch_expected_stats`.

    Parameters
    ----------
    Q : (K, K) array_like
        CTMC generator (rows sum to zero).
    t : float
        Branch length (or, for the dating grid, a time-cell duration).
    E : (K, K) array_like
        Reward indicator (``e_m e_m^T`` for dwell time in state ``m``;
        ``q_{mn} e_m e_n^T`` for ``m -> n`` jumps).

    Returns
    -------
    (K, K) ndarray
        The top-right block ``∫_0^t expm(Q τ) E expm(Q (t-τ)) dτ``.
    """
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    block = np.zeros((2 * K, 2 * K))
    block[:K, :K] = Q
    block[K:, K:] = Q
    block[:K, K:] = E
    return expm(block * t)[:K, K:]


def branch_expected_stats(Q, t, xi):
    """**THE SEAM** (CLAUDE.md §3.2, §12) — endpoint-conditioned expected dwell times and
    jump counts on one branch.

    The E-step calls exactly this, once per edge (:mod:`tspaint.accumulate`). A Phasic
    backend replaces exactly this, and owns its own caching.

    Parameters
    ----------
    Q : (K, K) array_like
        CTMC generator, ``Q[m, n]`` = rate ``m -> n``, rows sum to zero.
    t : float
        Branch length ``node_time[parent] - node_time[child]``, in generations. Root
        branches are length 0 by tskit convention (CLAUDE.md §3.4, §4.5); ``t <= 0``
        returns zeros.
    xi : (K, K) array_like
        Posterior over ``(parent_state, child_state)`` for this branch, normalised to sum
        to 1 — i.e. ``prune_tree(...).xi[(parent, child)]``. Row = parent = the CTMC's
        *start* (time runs parent -> child, old -> young). The branch's **span weight is
        applied by the caller** (``S_dwell += span * dwell``), not here.

    Returns
    -------
    dwell : (K,) ndarray
        Expected time spent in each state along the branch, under ``xi``. Sums to ``t``
        for any normalised ``xi`` (a useful invariant — the branch has to be somewhere).
    jumps : (K, K) ndarray
        Expected number of ``m -> n`` transitions (``m != n``), under ``xi``. Diagonal is
        zero, as is any entry where ``Q[m, n] == 0``.
    """
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    if t <= 0:
        return np.zeros(K), np.zeros((K, K))

    dwell_cond, jump_cond = _branch_kernel_cached(Q.tobytes(), K, float(t))

    xi = np.asarray(xi, float)
    dwell = np.array([float(np.sum(xi * dwell_cond[m])) for m in range(K)])
    jumps = np.zeros((K, K))
    for (m, n), cond in jump_cond.items():
        jumps[m, n] = float(np.sum(xi * cond))
    return dwell, jumps


@lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def _branch_kernel_cached(q_bytes, K, t):
    """Memoised ``(Q, t)`` half of :func:`branch_expected_stats` — the expensive part.

    Keyed on ``Q``'s bytes as well as ``t``, so it stays correct across EM iterations (each
    of which fits a new ``Q``) and across threads, rather than relying on a
    reset-when-Q-changes global. Implementation detail; not part of the seam contract.
    """
    Q = np.frombuffer(q_bytes, dtype=np.float64).reshape(K, K)
    return _branch_kernel(Q, t)


def _branch_kernel(Q, t):
    """``E[reward | s_parent, s_child]`` per reward: the Van Loan integral, divided
    elementwise by ``P = expm(Q t)`` to turn the joint into a conditional.

    Returns ``(dwell_cond, jump_cond)``: ``dwell_cond[m]`` is the ``(K, K)`` matrix
    ``E[time in m | s_p, s_c]``, and ``jump_cond[(m, n)]`` the ``(K, K)`` matrix
    ``E[# m->n jumps | s_p, s_c]`` (only for ``m != n`` with ``Q[m, n] != 0``).
    """
    K = Q.shape[0]
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
            jump_cond[(m, n)] = vanloan_integral(Q, t, E) / Psafe  # E[# m->n | s_p, s_c]

    return dwell_cond, jump_cond
