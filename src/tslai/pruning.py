"""Felsenstein pruning per marginal tree, per root (CLAUDE.md §3.1, §4) — Rung 3.

Two-pass sum-product message passing on each marginal tree:

* up-pass (post-order): partial likelihoods ``L_u(s)``, product over **all** children
  (polytomy-safe via ``tree.children`` / sib pointers); a node's own emission (if it
  is a labelled tip / query) multiplies in, so internal samples are handled too.
* down-pass (pre-order): posterior marginals ``γ_u(s)`` and joint parent-child
  posteriors ``ξ_{(p,c)}(s_p, s_c)`` — the (K, K) expected-transition object per
  branch — via leave-one-out (cavity) products, avoiding message division.

Transition convention: ``P_c = expm(Q * t_c)`` indexed ``[parent_state, child_state]``
with ``t_c = time[parent] - time[child] > 0``. Root state enters via ``π``.

Invariants honoured (CLAUDE.md §4, §12):

* prune **per root** — a marginal tree may be a forest (``for r in tree.roots``);
* **skip root branches** (no parent edge); root enters via ``π`` only;
* **isolated samples** (root sample with no children) = *missing-info*, tagged and
  set to the prior ``π`` — distinct from a 50-50 uncertain call;
* arbitrary arity (polytomies) via ``tree.children``.

Likelihoods are normalised per node for numerical stability; the discarded scale is
accumulated so :attr:`PruneResult.loglik` is the exact tree log-likelihood (used to
monitor EM and as a strong correctness check). Posteriors are scale-invariant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tskit

from .model import transition_matrix

__all__ = ["PruneResult", "prune_root", "prune_tree"]


@dataclass
class PruneResult:
    gamma: dict           # node -> (K,) posterior marginal P(state | tree)
    xi: dict              # (parent, child) -> (K, K) joint posterior
    root_marginal: dict   # root -> (K,) posterior (also in gamma)
    missing_info: set     # isolated-sample nodes (the tree says nothing about them)
    loo: dict             # node -> (K,) leave-one-out marginal: the outside message,
                          # excluding the node's OWN emission — credibility evidence that
                          # avoids a label confirming itself (CLAUDE.md §2.3, §3.3)
    loglik: float         # total log-likelihood, summed over roots


def _transition_cache(Q):
    cache = {}

    def get(t):
        key = float(t)
        P = cache.get(key)
        if P is None:
            P = transition_matrix(Q, t)
            cache[key] = P
        return P

    return get


def prune_root(tree, root, emissions, Q, node_time, pi, Pget=None):
    """Up/down pass for a single root of a marginal tree (see module docstring)."""
    pi = np.asarray(pi, float)
    K = pi.shape[0]
    if Pget is None:
        Pget = _transition_cache(Q)

    Lnorm = {}       # node -> normalised partial likelihood
    cumscale = {}    # node -> accumulated log normalisation of its subtree
    msg = {}         # child -> message to its parent: P_c @ L_c  (K,)

    # --- up-pass (post-order) ---
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
            t = node_time[parent] - node_time[u]
            msg[u] = Pget(t) @ L

    # --- root belief / likelihood ---
    b_root = pi * Lnorm[root]
    Zr = b_root.sum()
    loglik = (np.log(Zr) if Zr > 0 else -np.inf) + cumscale[root]

    gamma = {root: (b_root / Zr if Zr > 0 else pi.copy())}
    xi = {}
    U = {root: pi.copy()}   # message into a node from the parent side (the "outside")

    missing = set()
    if tree.is_sample(root) and len(tree.children(root)) == 0:
        # isolated sample: the tree carries no information -> fall back to prior, tag it
        missing.add(root)
        gamma[root] = pi.copy()

    # --- down-pass (pre-order) ---
    for p in tree.nodes(root, order="preorder"):
        children = list(tree.children(p))
        if not children:
            continue
        # cavity base = outside message * p's OWN emission (matters when p is an
        # internal sample); children messages are folded in leave-one-out below.
        ep = emissions.get(p)
        base = U[p] * np.asarray(ep, float) if ep is not None else U[p]
        msgs = [msg[c] for c in children]
        k = len(children)
        # leave-one-out products of child messages (no division)
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
            cavity = base * prefix[i] * suffix[i]    # above p, p's own emission, c's siblings
            Pc = Pget(node_time[p] - node_time[c])
            Uc = cavity @ Pc                         # down-message into c
            sUc = Uc.sum()
            U[c] = Uc / sUc if sUc > 0 else np.full(K, 1.0 / K)
            bc = U[c] * Lnorm[c]
            sbc = bc.sum()
            gamma[c] = bc / sbc if sbc > 0 else np.full(K, 1.0 / K)
            M = (cavity[:, None] * Pc) * Lnorm[c][None, :]
            sM = M.sum()
            xi[(p, c)] = M / sM if sM > 0 else np.full((K, K), 1.0 / (K * K))

    return PruneResult(gamma, xi, {root: gamma[root]}, missing, dict(U), float(loglik))


def prune_tree(tree, emissions, Q, node_time, pi):
    """Prune every root of a marginal tree; aggregate γ, ξ, root marginals,
    missing-info tags and the total log-likelihood (CLAUDE.md §3.1, §4)."""
    Pget = _transition_cache(Q)
    gamma, xi, root_marginal, missing, loo = {}, {}, {}, set(), {}
    loglik = 0.0
    for r in tree.roots:
        res = prune_root(tree, r, emissions, Q, node_time, pi, Pget)
        gamma.update(res.gamma)
        xi.update(res.xi)
        root_marginal.update(res.root_marginal)
        missing |= res.missing_info
        loo.update(res.loo)
        loglik += res.loglik
    return PruneResult(gamma, xi, root_marginal, missing, loo, float(loglik))
