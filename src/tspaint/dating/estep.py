"""Time-resolved E-step for admixture rate through time (admix-dating design §2).

Per branch, the endpoint-conditioned expected **dwell** and **directional jumps** are computed
**per time cell** (branch split at cell boundaries; the Van Loan reward of each sub-interval
sandwiched by the forward/backward transition products and conditioned on the branch endpoints
``xi``), then accumulated edge-blocked and span-weighted (CLAUDE.md §3.3).

Rung-1 invariant: with a homogeneous generator, the per-cell statistics summed over the cells a
branch spans equal :func:`tspaint.branch_stats.branch_expected_stats` for the whole branch — the
additive property of ``∫_0^T = Σ_cells ∫_cell``.

``accumulate_time_binned`` is the **Stage-1 shortcut** (design build-order rung 2): it reuses the
existing *homogeneous* pruning ``xi`` and merely bins the rewards by time — a first
rate-through-time profile from one ``tspaint.fit``, before the full time-inhomogeneous pruning.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
import tskit

from ..branch_stats import vanloan_integral
from ..pruning import prune_tree
from .grid import split_branch, cell_centers

__all__ = ["branch_cell_stats", "accumulate_time_binned", "rate_through_time_binned",
           "composite_transition", "accumulate_time_binned_tv", "paint_qt"]


def composite_transition(Q_of_cell, t_c, t_p, edges):
    """Composite parent→child transition ``∏ expm(Q_k·d_k)`` for a branch under a
    time-inhomogeneous generator (indexed ``[s_p, s_c]``, the pruning convention)."""
    subs = split_branch(t_c, t_p, edges)
    K = Q_of_cell(subs[0][0]).shape[0]
    P = np.eye(K)
    for (k, d) in subs:                      # parent -> child order
        P = P @ expm(Q_of_cell(k) * d)
    return P


def _prune_root_tv(tree, root, emissions, node_time, pi, Pbranch, K):
    """Up/down pass for one root under a time-inhomogeneous generator. ``Pbranch(t_c, t_p)``
    gives the composite branch transition. Returns ``(xi, loglik)`` — mirrors
    :func:`tspaint.pruning.prune_root` with the per-branch composite transition."""
    Lnorm, cumscale, msg = {}, {}, {}
    for u in tree.nodes(root, order="postorder"):
        e = emissions.get(u)
        L = np.array(e, float) if e is not None else np.ones(K)
        cs = 0.0
        for c in tree.children(u):
            L = L * msg[c]
            cs += cumscale[c]
        s = L.sum()
        if s > 0:
            L = L / s
            cs += np.log(s)
        Lnorm[u] = L
        cumscale[u] = cs
        parent = tree.parent(u)
        if parent != tskit.NULL:
            msg[u] = Pbranch(float(node_time[u]), float(node_time[parent])) @ L

    b_root = pi * Lnorm[root]
    Zr = b_root.sum()
    loglik = (np.log(Zr) if Zr > 0 else -np.inf) + cumscale[root]
    xi = {}
    U = {root: pi.copy()}
    gamma = {root: (b_root / Zr if Zr > 0 else pi.copy())}
    if tree.is_sample(root) and len(tree.children(root)) == 0:
        gamma[root] = pi.copy()                          # isolated sample -> prior
    for p in tree.nodes(root, order="preorder"):
        children = list(tree.children(p))
        if not children:
            continue
        ep = emissions.get(p)
        base = U[p] * np.asarray(ep, float) if ep is not None else U[p]
        msgs = [msg[c] for c in children]
        k = len(children)
        prefix = [None] * k
        suffix = [None] * k
        acc = np.ones(K)
        for i in range(k):
            prefix[i] = acc
            acc = acc * msgs[i]
        acc = np.ones(K)
        for i in range(k - 1, -1, -1):
            suffix[i] = acc
            acc = acc * msgs[i]
        for i, c in enumerate(children):
            cavity = base * prefix[i] * suffix[i]
            Pc = Pbranch(float(node_time[c]), float(node_time[p]))
            Uc = cavity @ Pc
            sUc = Uc.sum()
            U[c] = Uc / sUc if sUc > 0 else np.full(K, 1.0 / K)
            bc = U[c] * Lnorm[c]
            sbc = bc.sum()
            gamma[c] = bc / sbc if sbc > 0 else np.full(K, 1.0 / K)
            M = (cavity[:, None] * Pc) * Lnorm[c][None, :]
            sM = M.sum()
            xi[(p, c)] = M / sM if sM > 0 else np.full((K, K), 1.0 / (K * K))
    return xi, gamma, float(loglik)


def accumulate_time_binned_tv(ts, Q_of_cell, pi, emissions, edges):
    """Time-**inhomogeneous** E-step: per-cell dwell + directional jumps under ``Q_of_cell``,
    edge-blocked and span-weighted. Returns ``(D, J, loglik)`` (the full rung-4 E-step)."""
    node_time = ts.tables.nodes.time
    K = np.asarray(pi).shape[0]
    ncell = len(edges) - 1
    D = np.zeros((ncell, K))
    J = np.zeros((ncell, K, K))
    loglik = 0.0
    cache = {}

    def Pbranch(t_c, t_p):
        key = (t_c, t_p)
        P = cache.get(key)
        if P is None:
            P = composite_transition(Q_of_cell, t_c, t_p, edges)
            cache[key] = P
        return P

    for (interval, _eout, ein), tree in zip(ts.edge_diffs(), ts.trees()):
        span = interval[1] - interval[0]
        xi_all = {}
        for r in tree.roots:
            xi, _gamma, ll = _prune_root_tv(tree, r, emissions, node_time, pi, Pbranch, K)
            xi_all.update(xi)
            loglik += span * ll
        for e in ein:
            c, p = e.child, e.parent
            if tree.parent(c) == tskit.NULL:
                continue
            t_c, t_p = float(node_time[c]), float(node_time[p])
            if t_p <= t_c:
                continue
            dwell, jumps = branch_cell_stats(Q_of_cell, t_c, t_p, xi_all[(p, c)], edges)
            w = e.right - e.left
            for k, dw in dwell.items():
                D[k] += w * dw
            for k, jm in jumps.items():
                J[k] += w * jm
    return D, J, loglik


def paint_qt(ts, emissions, Q_of_cell, pi, edges, focal):
    """Paint focal tips under a time-**inhomogeneous** generator ``Q_of_cell`` — the same
    deliverable as :func:`tspaint.output.posterior_table` but using the per-cell rates from the
    admixture-rate-through-time fit (so recent branches with a low rate wash less). Returns
    ``{sample: [Segment]}``; isolated spans are tagged ``MISSING_INFO``."""
    from ..output import Segment, INFORMATIVE, MISSING_INFO
    node_time = ts.tables.nodes.time
    K = np.asarray(pi).shape[0]
    samples = [int(s) for s in focal]
    tracks = {s: [] for s in samples}
    cache = {}

    def Pbranch(t_c, t_p):
        key = (t_c, t_p)
        P = cache.get(key)
        if P is None:
            P = composite_transition(Q_of_cell, t_c, t_p, edges)
            cache[key] = P
        return P

    for tree in ts.trees():
        left, right = tree.interval.left, tree.interval.right
        gamma_all = {}
        missing = set()
        for r in tree.roots:
            _xi, gamma, _ll = _prune_root_tv(tree, r, emissions, node_time, pi, Pbranch, K)
            gamma_all.update(gamma)
            if tree.is_sample(r) and len(tree.children(r)) == 0:
                missing.add(r)
        for s in samples:
            post = np.asarray(gamma_all.get(s, pi), float)
            status = MISSING_INFO if s in missing else INFORMATIVE
            segs = tracks[s]
            if (segs and segs[-1].right == left and segs[-1].status == status
                    and np.allclose(segs[-1].posterior, post)):
                segs[-1].right = right
            else:
                segs.append(Segment(left, right, post, status))
    return tracks


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

    Mirrors :func:`tspaint.accumulate.accumulate_sufficient_statistics` (bank each entering edge
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
