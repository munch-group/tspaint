"""Structured / blocked EM (CLAUDE.md §3).

E-step = exact Felsenstein pruning per marginal tree, per root (:mod:`tslai.pruning`);
sufficient statistics accumulated per edge, span-weighted (:mod:`tslai.accumulate`);
M-step = the closed forms below (CLAUDE.md §3.4).

The closed-form M-step is implemented now (it is pure and unit-testable); the
E-step orchestration loop :func:`fit` lands with Rungs 4-5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .accumulate import accumulate_sufficient_statistics
from .model import make_generator_2state, tip_emission, query_emission

__all__ = ["m_step_Q", "m_step_pi", "m_step_w", "fit", "FitResult", "build_emissions"]


def m_step_Q(S_dwell, S_jumps):
    """Closed-form generator MLE: ``q_mn = S_jumps[m, n] / S_dwell[m]`` (m != n),
    with ``q_mm = -Σ_n q_mn`` (CLAUDE.md §3.4)."""
    S_dwell = np.asarray(S_dwell, float)
    S_jumps = np.asarray(S_jumps, float)
    K = S_dwell.shape[0]
    Q = np.zeros((K, K))
    for m in range(K):
        if S_dwell[m] > 0:
            for n in range(K):
                if n != m:
                    Q[m, n] = S_jumps[m, n] / S_dwell[m]
        Q[m, m] = -Q[m].sum()
    return Q


def m_step_pi(S_root):
    """Closed-form root-frequency MLE: ``π = S_root / Σ S_root`` (CLAUDE.md §3.4)."""
    S_root = np.asarray(S_root, float)
    total = S_root.sum()
    if total <= 0:
        K = S_root.shape[0]
        return np.full(K, 1.0 / K)
    return S_root / total


def m_step_w(agree, disagree, alpha, beta):
    """Per-tip credibility MAP under ``Beta(alpha, beta)`` (CLAUDE.md §3.4)::

        w = (alpha - 1 + agree) / (alpha + beta - 2 + agree + disagree)
    """
    denom = alpha + beta - 2.0 + agree + disagree
    return (alpha - 1.0 + agree) / denom


def _as_dataset_lists(ts, labels):
    if isinstance(ts, (list, tuple)):
        return list(ts), list(labels)
    return [ts], [labels]


@dataclass
class FitResult:
    Q: np.ndarray            # fitted generator
    pi: np.ndarray           # fitted root frequencies
    w: dict                  # learned per-tip credibility for soft refs (empty if hard-clamp)
    loglik_history: list     # observed-data log-likelihood per E-step (non-decreasing)


def build_emissions(ts, labels, w, pi):
    """Emissions per sample: soft-clamp ``w_i * 1[label] + (1-w_i) * pi`` for labelled
    tips (anchors default to ``w = 1``), query emission otherwise (CLAUDE.md §2.2)."""
    emissions = {}
    for s in ts.samples():
        s = int(s)
        if s in labels:
            emissions[s] = tip_emission(labels[s], w.get(s, 1.0), pi)
        else:
            emissions[s] = query_emission(pi)
    return emissions


def fit(ts, labels, *, K=2, Q0=None, pi0=None, max_iter=200, tol=1e-7,
        soft_refs=None, alpha=20.0, beta=1.0, w0=0.9, estimate_pi=True):
    """Blocked EM for ``(Q, π, {w_i})`` (CLAUDE.md §3, §11.1.5-6).

    Parameters
    ----------
    ts : tskit.TreeSequence or list thereof
        One genome, or several independent tree sequences whose statistics are pooled.
    labels : dict[int, int] or list thereof
        Per-sample label index for the reference tips; samples absent are queries.
    soft_refs : set[int], optional
        Labelled tips whose credibility ``w_i`` is **learned** (MAP under
        ``Beta(alpha, beta)``). All other labelled tips are hard-clamped anchors
        (``w ≡ 1``). With ``soft_refs=None`` every reference is an anchor (Rung 5).
        At least one anchor is required when ``soft_refs`` is non-empty — never let
        the whole panel float (CLAUDE.md §6).
    alpha, beta : Beta prior on credibility (default mass near 1).
    w0 : initial credibility for soft refs.

    Returns
    -------
    FitResult
    """
    ts_list, lab_list = _as_dataset_lists(ts, labels)
    Q = np.array(Q0, float) if Q0 is not None else make_generator_2state(0.1, 0.1)
    pi = np.array(pi0, float) if pi0 is not None else np.full(K, 1.0 / K)

    soft = set(int(s) for s in soft_refs) if soft_refs else set()
    if soft:
        all_labels = set().union(*[set(int(k) for k in l) for l in lab_list])
        if not (all_labels - soft):
            raise ValueError(
                "keep a hard-clamped anchor set; never let the whole panel float "
                "(CLAUDE.md §6)")
    w = {s: float(w0) for s in soft}   # anchors stay at w = 1 implicitly

    history = []
    prev = -np.inf
    for _ in range(max_iter):
        S_dwell = np.zeros(K)
        S_jumps = np.zeros((K, K))
        S_root = np.zeros(K)
        S_cred = {}
        loglik = 0.0
        for tsi, labi in zip(ts_list, lab_list):
            emissions = build_emissions(tsi, labi, w, pi)
            ss = accumulate_sufficient_statistics(tsi, Q, pi, emissions,
                                                  labels=labi, soft_refs=soft or None)
            S_dwell += ss.S_dwell
            S_jumps += ss.S_jumps
            S_root += ss.S_root
            loglik += ss.loglik
            for node, cred in ss.S_cred.items():
                S_cred[node] = S_cred.get(node, np.zeros(2)) + cred
        history.append(loglik)

        Q = m_step_Q(S_dwell, S_jumps)
        # pi is a prior on the (arbitrary) GMRCA state. When deep branches wash it is
        # unidentifiable from the root marginals (they echo pi) and drifts to a degenerate
        # extreme -> confident-wrong painting on sparse ARGs / the order-only variant
        # (CLAUDE.md §6). estimate_pi=False holds it fixed (uniform unless pi0 given).
        if estimate_pi:
            pi = m_step_pi(S_root)
        for s in soft:
            if s in S_cred:
                agree, disagree = S_cred[s]
                w[s] = m_step_w(agree, disagree, alpha, beta)

        if len(history) > 1 and abs(loglik - prev) < tol:
            break
        prev = loglik

    return FitResult(Q, pi, w, history)
