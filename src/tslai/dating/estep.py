"""Time-resolved E-step for admixture rate through time (admix-dating design §2).

Per branch, the endpoint-conditioned expected **dwell** and **directional jumps** are computed
**per time cell** (branch split at cell boundaries; the Van Loan reward of each sub-interval
sandwiched by the forward/backward transition products and conditioned on the branch endpoints
``xi``), then accumulated edge-blocked and span-weighted (CLAUDE.md §3.3).

Rung-1 invariant: with a homogeneous generator, the per-cell statistics summed over the cells a
branch spans equal :func:`tslai.branch_stats.branch_expected_stats` for the whole branch — the
additive property of ``∫_0^T = Σ_cells ∫_cell``.

``accumulate_time_binned`` is the **Stage-1 shortcut** (design build-order rung 2): it reuses the
existing *homogeneous* pruning ``xi`` and merely bins the rewards by time — a first
rate-through-time profile from one ``tslai.fit``, before the full time-inhomogeneous pruning.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
import tskit

from ..branch_stats import vanloan_integral
from ..pruning import prune_tree
from .grid import split_branch, cell_centers

__all__ = ["branch_cell_stats", "accumulate_time_binned", "rate_through_time_binned"]


def branch_cell_stats(Q_of_cell, t_c, t_p, xi, edges):
    """Per-cell endpoint-conditioned expected dwell and directional jumps for one branch.

    Parameters
    ----------
    Q_of_cell : callable
        ``Q_of_cell(cell_index) -> (K, K)`` generator for that time cell (homogeneous case
        returns the same ``Q`` for every cell).
    t_c, t_p : float
        Child and parent node ages (``t_p > t_c``).
    xi : (K, K) array_like
        Endpoint joint posterior ``xi[s_p, s_c]`` for the branch.
    edges : array_like
        Grid cell edges.

    Returns
    -------
    dwell : dict[int, ndarray]
        ``cell_index -> (K,)`` expected dwell per state, conditioned on ``xi``.
    jumps : dict[int, ndarray]
        ``cell_index -> (K, K)`` expected ``m -> n`` jumps, conditioned on ``xi``.
    """
    subs = split_branch(t_c, t_p, edges)
    xi = np.asarray(xi, float)
    K = xi.shape[0]
    Ps = [expm(Q_of_cell(k) * d) for (k, d) in subs]
    n = len(subs)
    eye = np.eye(K)

    pre = [eye]                                   # pre[j] = P_0 @ ... @ P_{j-1} (parent side)
    for P in Ps:
        pre.append(pre[-1] @ P)
    Pbranch = pre[n]
    Psafe = np.where(Pbranch > 1e-300, Pbranch, 1e-300)
    suf = [None] * n                              # suf[j] = P_{j+1} @ ... @ P_{n-1} (child side)
    acc = eye
    for j in range(n - 1, -1, -1):
        suf[j] = acc
        acc = Ps[j] @ acc

    dwell, jumps = {}, {}
    for j, (k, d) in enumerate(subs):
        Qk = Q_of_cell(k)
        L, R = pre[j], suf[j]
        dw = np.zeros(K)
        for m in range(K):
            E = np.zeros((K, K))
            E[m, m] = 1.0
            Mj = L @ vanloan_integral(Qk, d, E) @ R
            dw[m] = float(np.sum(xi * (Mj / Psafe)))
        jm = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                if a == b or Qk[a, b] == 0.0:
                    continue
                E = np.zeros((K, K))
                E[a, b] = Qk[a, b]
                Mj = L @ vanloan_integral(Qk, d, E) @ R
                jm[a, b] = float(np.sum(xi * (Mj / Psafe)))
        dwell[k] = dwell.get(k, np.zeros(K)) + dw
        jumps[k] = jumps.get(k, np.zeros((K, K))) + jm
    return dwell, jumps


def accumulate_time_binned(ts, Q, pi, emissions, edges):
    """Edge-blocked, span-weighted per-cell dwell + directional jumps (Stage-1, homogeneous Q).

    Mirrors :func:`tslai.accumulate.accumulate_sufficient_statistics` (bank each entering edge
    once, weighted by its span) but bins each branch's rewards by time cell.

    Returns
    -------
    D : (n_cells, K) ndarray
        Span-weighted expected dwell per cell per state.
    J : (n_cells, K, K) ndarray
        Span-weighted expected directional jumps per cell.
    """
    node_time = ts.tables.nodes.time
    K = Q.shape[0]
    ncell = len(edges) - 1
    D = np.zeros((ncell, K))
    J = np.zeros((ncell, K, K))
    Qf = lambda k: Q                                  # noqa: E731 (homogeneous E-step)
    for (interval, _eout, ein), tree in zip(ts.edge_diffs(), ts.trees()):
        res = prune_tree(tree, emissions, Q, node_time, pi)
        for e in ein:
            c, p = e.child, e.parent
            if tree.parent(c) == tskit.NULL:
                continue
            t_c, t_p = float(node_time[c]), float(node_time[p])
            if t_p <= t_c:
                continue
            xi = res.xi[(p, c)]
            w = e.right - e.left
            dwell, jumps = branch_cell_stats(Qf, t_c, t_p, xi, edges)
            for k, dw in dwell.items():
                D[k] += w * dw
            for k, jm in jumps.items():
                J[k] += w * jm
    return D, J


def rate_through_time_binned(ts, labels, edges, *, max_iter=8, Q0=None, estimate_pi=False,
                             soft_refs=None):
    """Stage-1 admixture-rate-through-time profile: fit a homogeneous ``Q``, then bin the
    per-branch sufficient statistics by time.

    Returns
    -------
    dict
        ``centers`` (cell centres), ``q_AB``/``q_BA`` (directional rate per cell =
        ``jumps/dwell``), ``D``/``J`` (raw per-cell dwell/jumps), and the fitted ``Q``.
    """
    from ..em import fit, build_emissions
    from ..model import make_generator_2state

    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    res = fit(ts, labels, Q0=Q0, max_iter=max_iter, estimate_pi=estimate_pi, soft_refs=soft_refs)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    D, J = accumulate_time_binned(ts, res.Q, res.pi, emissions, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_AB = J[:, 0, 1] / np.where(D[:, 0] > 0, D[:, 0], np.nan)
        q_BA = J[:, 1, 0] / np.where(D[:, 1] > 0, D[:, 1], np.nan)
    return {"centers": cell_centers(edges), "q_AB": q_AB, "q_BA": q_BA,
            "D": D, "J": J, "Q": res.Q}
