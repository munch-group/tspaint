"""Structured / blocked EM (CLAUDE.md §3).

E-step = exact Felsenstein pruning per marginal tree, per root (:mod:`tspaint.pruning`);
sufficient statistics accumulated per edge, span-weighted (:mod:`tspaint.accumulate`);
M-step = the closed forms below (CLAUDE.md §3.4).

The closed-form M-step is implemented now (it is pure and unit-testable); the
E-step orchestration loop :func:`fit` lands with Rungs 4-5.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from functools import reduce

import numpy as np

from .accumulate import accumulate_sufficient_statistics
from .model import make_generator_symmetric, tip_emission, query_emission, MaskedEmissions

__all__ = ["m_step_Q", "m_step_pi", "m_step_w", "fit", "FitResult", "build_emissions"]


def m_step_Q(S_dwell, S_jumps):
    """Closed-form generator MLE from expected dwell times and jump counts.

    ``q_mn = S_jumps[m, n] / S_dwell[m]`` for ``m != n``, with
    ``q_mm = -Σ_n q_mn`` (CLAUDE.md §3.4).

    Parameters
    ----------
    S_dwell : (K,) array_like
        Total expected dwell time per state (span-weighted).
    S_jumps : (K, K) array_like
        Total expected ``m -> n`` jump counts per ordered pair (span-weighted).

    Returns
    -------
    (K, K) numpy.ndarray
        The fitted generator ``Q`` (rows sum to 0).
    """
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
    """Closed-form root-frequency MLE: ``π = S_root / Σ S_root`` (CLAUDE.md §3.4).

    Parameters
    ----------
    S_root : (K,) array_like
        Total expected root-state mass (span-weighted, pooled over roots).

    Returns
    -------
    (K,) numpy.ndarray
        The fitted root frequencies ``π``; uniform if ``S_root`` sums to ``<= 0``.
    """
    S_root = np.asarray(S_root, float)
    total = S_root.sum()
    if total <= 0:
        K = S_root.shape[0]
        return np.full(K, 1.0 / K)
    return S_root / total


def m_step_w(agree, disagree, alpha, beta):
    """Per-tip credibility MAP under a ``Beta(alpha, beta)`` prior (CLAUDE.md §3.4).

    Parameters
    ----------
    agree : float
        Expected (span-weighted) mass where the tip posterior agrees with its label.
    disagree : float
        Expected (span-weighted) mass where it disagrees.
    alpha, beta : float
        Beta-prior hyperparameters (default mass near 1).

    Returns
    -------
    float
        The MAP credibility ``w``.

    Notes
    -----
    ::

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
    """Fitted parameters returned by :func:`fit`.

    Attributes
    ----------
    Q : numpy.ndarray
        Fitted generator.
    pi : numpy.ndarray
        Fitted root frequencies.
    w : dict
        Learned per-tip credibility for soft refs (empty if hard-clamp).
    loglik_history : list
        Observed-data log-likelihood per E-step (non-decreasing).
    """
    Q: np.ndarray            # fitted generator
    pi: np.ndarray           # fitted root frequencies
    w: dict                  # learned per-tip credibility for soft refs (empty if hard-clamp)
    loglik_history: list     # observed-data log-likelihood per E-step (non-decreasing)


def build_emissions(ts, labels, w, pi, mask=None):
    """Build the per-sample Felsenstein emission vectors (CLAUDE.md §2.2).

    Labelled tips get the soft-clamp ``w_i * 1[label] + (1 - w_i) * pi`` (anchors
    default to ``w = 1``); every other sample gets the query (flat / root-freq)
    emission.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Provides the sample-node ids.
    labels : dict[int, int]
        Reference sample-node id → ancestry-state index.
    w : dict[int, float]
        Per-tip credibility; tips absent default to ``1.0`` (hard clamp).
    pi : (K,) array_like
        Root frequencies ``π`` used by the soft-clamp and query emissions.
    mask : dict[int, list[tuple]], optional
        Fragment masking (CLAUDE.md §2.3): per reference sample-node id, the half-open
        ``[left, right)`` spans over which that reference emits the **query** (unlabelled)
        emission instead of its label — so it anchors at full strength on its clean spans and
        contributes no label information exactly where flagged foreign (the local,
        position-dependent generalisation of the credibility ``w_i``). Each span is
        ``(left, right)`` or ``(left, right, score)`` (any third element is ignored). Typically
        produced by :meth:`~tspaint.introgression.ReferenceQC.mask` /
        :func:`tspaint.foreign_tracts`. Default ``None`` (no masking).

    Returns
    -------
    dict[int, numpy.ndarray] or MaskedEmissions
        Per sample-node id, the ``(K,)`` emission vector. With ``mask`` given, a
        :class:`~tspaint.model.MaskedEmissions` wrapping those vectors instead — resolving to a
        per-interval dict on demand (:func:`tspaint.model.emissions_for`).
    """
    emissions = {}
    for s in ts.samples():
        s = int(s)
        if s in labels:
            emissions[s] = tip_emission(labels[s], w.get(s, 1.0), pi)
        else:
            emissions[s] = query_emission(pi)
    # ``mask`` (fragment masking, §2.3): a reference emits the query (unlabelled) emission over its
    # masked spans, so it anchors only where it is clean — position-dependent, resolved per tree.
    return MaskedEmissions(emissions, mask, pi) if mask else emissions


def fit(ts, labels, *, K=2, Q0=None, pi0=None, max_iter=200, tol=1e-7,
        soft_refs=None, alpha=20.0, beta=1.0, priors=None, w0=0.9, estimate_pi=True,
        n_jobs=None, progress=False, mask=None):
    """Blocked EM for ``(Q, π, {w_i})`` (CLAUDE.md §3, §11.1.5-6).

    The E-step is exact Felsenstein pruning per marginal tree, per root; sufficient
    statistics are accumulated per edge, span-weighted; the M-step is the closed forms
    :func:`m_step_Q`, :func:`m_step_pi`, :func:`m_step_w`. Iterates to a log-likelihood
    tolerance or ``max_iter``.

    Parameters
    ----------
    ts : tskit.TreeSequence or list thereof
        One genome, or several independent tree sequences whose statistics are pooled.
    labels : dict or list thereof
        Per-reference label index; samples absent are queries. Keys may be integer sample-node
        indices or sample-ID strings when ``ts`` was stamped by :func:`tspaint.io.singer` /
        :func:`tspaint.io.tsinfer` (:mod:`tspaint.ids`). ``soft_refs`` / ``priors`` keys likewise.
    K : int, optional
        Number of ancestry states. Default ``2``.
    Q0 : (K, K) array_like, optional
        Initial generator; defaults to a symmetric ``K``-state generator
        (:func:`tspaint.make_generator_symmetric`) whose rate is scaled to the tree-sequence time
        axis (so ``Q0 · t`` starts at O(1)). Works for any ``K``.
    pi0 : (K,) array_like, optional
        Initial / fixed root frequencies; defaults to uniform.
    max_iter : int, optional
        Maximum number of EM iterations. Default ``200``.
    tol : float, optional
        Stop when the log-likelihood changes by less than ``tol``. Default ``1e-7``.
    soft_refs : set[int], optional
        Labelled tips whose credibility ``w_i`` is **learned** (MAP under
        ``Beta(alpha, beta)``). All other labelled tips are hard-clamped anchors
        (``w ≡ 1``). With ``soft_refs=None`` every reference is an anchor (Rung 5).
        At least one anchor is required when ``soft_refs`` is non-empty — never let
        the whole panel float (CLAUDE.md §6).
    alpha, beta : float, optional
        Default ``Beta(alpha, beta)`` prior on credibility (mass near 1), applied to
        every soft ref not named in ``priors``.
    priors : dict[int, tuple[float, float]], optional
        Per-tip ``Beta`` prior overrides ``{tip: (alpha_i, beta_i)}`` for the
        graded-trust setting — give the references believed purer a stronger prior
        (mass closer to 1) than the rest. Keys must be a subset of ``soft_refs``
        (hard-clamped anchors have no learned ``w``); a key outside ``soft_refs``
        raises ``ValueError``. Note at genome scale the span-weighted credibility
        evidence typically swamps the prior, so ``w_i`` converges to the reference's
        empirical purity almost regardless of prior strength — the prior's role is the
        identifiability backstop and short/low-info-region regularisation (CLAUDE.md §6).
    w0 : float, optional
        Initial credibility for soft refs. Default ``0.9``.
    estimate_pi : bool, optional
        Re-estimate ``π`` each M-step rather than holding ``pi0`` fixed. Default
        ``True``. See Notes.
    n_jobs : int, optional
        Worker processes for the E-step (the dominant cost). Default ``None`` → all CPUs / the SLURM
        allocation (:func:`tspaint.parallel.resolve_cores`); pass ``1`` for the serial path,
        **byte-identical** to single-core. ``>1`` runs the genome E-step over
        member × tree-range chunks on a persistent :class:`~concurrent.futures.ProcessPoolExecutor`
        (reused across EM iterations); the result is ``allclose`` to serial, differing only by
        floating-point reduction order (:mod:`tspaint.parallel`).
    progress : bool, optional
        Show a :mod:`tqdm.auto` ``EM fit`` bar over the iterations, with the running
        log-likelihood (notebook widget in Jupyter, text in a terminal). Default ``False``.
    mask : dict, optional
        Fragment masking (CLAUDE.md §2.3): per reference the half-open ``[left, right)`` spans
        over which it emits the **query** (unlabelled) emission instead of its label, so it
        anchors only on its clean spans and contributes nothing where flagged foreign. Keys may
        be integer sample-node indices or sample-ID strings (resolved like ``labels`` /
        ``soft_refs`` / ``priors``); each span is ``(left, right)`` or ``(left, right, score)``.
        Threaded through both the serial and the byte-exact parallel E-step
        (:func:`build_emissions` → :class:`~tspaint.model.MaskedEmissions`). Typically produced
        by :meth:`~tspaint.introgression.ReferenceQC.mask` / :func:`tspaint.foreign_tracts`.
        Default ``None``.

    Returns
    -------
    FitResult
        The fitted ``(Q, π, {w_i})`` and the log-likelihood history.

    Raises
    ------
    ValueError
        If ``soft_refs`` covers every labelled tip (no hard-clamped anchor left).

    Notes
    -----
    ``π`` is a prior on the (arbitrary) GMRCA state. When deep branches wash it is
    unidentifiable from the root marginals (they echo ``π``) and drifts to a degenerate
    extreme — confident-wrong painting on sparse ARGs / the order-only variant
    (CLAUDE.md §6). ``estimate_pi=False`` holds it fixed (uniform unless ``pi0`` given).
    """
    ts_list, lab_list = _as_dataset_lists(ts, labels)
    # labels / soft_refs / priors keys may be sample-ID strings (stamped by io.singer/io.tsinfer)
    # or integer node indices; resolve to node ids (idempotent for already-integer keys).
    from .ids import resolve_labels, resolve_ids, resolve_nodes
    lab_list = [resolve_labels(t, l) for t, l in zip(ts_list, lab_list)]
    # every reference label must be a valid ancestry-state index 0..K-1; otherwise build_emissions'
    # e[label] indexes past the K-vector — a cryptic IndexError deep in the (possibly parallel) E-step.
    # Fail here with the real cause: K too small, or labels not numbered 0..K-1.
    states = {int(st) for l in lab_list for st in l.values()}
    if states and (min(states) < 0 or max(states) >= K):
        raise ValueError(
            f"reference label states are {sorted(states)} but K={K} (valid states are 0..{K - 1}); "
            f"pass K={max(states) + 1} if you have that many ancestries, or renumber the labels to be "
            "0-based and contiguous (0..K-1)")
    if soft_refs is not None:
        soft_refs = resolve_ids(ts_list[0], soft_refs)
    if priors:
        priors = {node: pv for k, pv in priors.items() for node in resolve_nodes(ts_list[0], k)}
    if mask:                                    # fragment masking: {ref -> [(l,r)]}, keys id or node
        mask = {node: spans for k, spans in mask.items() for node in resolve_nodes(ts_list[0], k)}
    if Q0 is not None:
        Q = np.array(Q0, float)
    else:
        # Scale the initial generator to the tree-sequence TIME axis so ``Q0 * t`` starts at O(1) —
        # an informative E-step — regardless of the absolute node-age scale. A fixed rate washes out
        # (``expm(Q0 t) -> stationary``) on a large, calibrated axis (e.g. tsdate node ages, or a
        # deep genealogy), which freezes the EM at the initial Q and collapses the painting toward
        # ``pi``. Rate ``1 / mean(internal node age)`` puts the typical branch near ``Q t ~ 1``.
        ages = np.concatenate([np.asarray(t.tables.nodes.time, float) for t in ts_list])
        ages = ages[ages > 0]
        r0 = 1.0 / float(np.mean(ages)) if ages.size else 0.1
        Q = make_generator_symmetric(K, r0)     # K-way symmetric start (== 2-state at K=2)
    pi = np.array(pi0, float) if pi0 is not None else np.full(K, 1.0 / K)

    soft = set(int(s) for s in soft_refs) if soft_refs else set()
    if soft:
        all_labels = set().union(*[set(int(k) for k in l) for l in lab_list])
        if not (all_labels - soft):
            raise ValueError(
                "keep a hard-clamped anchor set; never let the whole panel float "
                "(CLAUDE.md §6)")
    tip_priors = {int(k): (float(a), float(b)) for k, (a, b) in priors.items()} if priors else {}
    extra = set(tip_priors) - soft
    if extra:
        raise ValueError(
            f"priors given for non-soft tips {sorted(extra)}; a per-tip Beta prior "
            "applies only to soft_refs (hard-clamped anchors have no learned w)")
    w = {s: float(w0) for s in soft}   # anchors stay at w = 1 implicitly

    from .parallel import resolve_cores
    n_jobs = resolve_cores(n_jobs)          # None -> all CPUs / SLURM allocation

    def estep_serial():
        """The pooled E-step over the ensemble — byte-identical to single-core."""
        S_dwell, S_jumps = np.zeros(K), np.zeros((K, K))
        S_root, S_cred, loglik = np.zeros(K), {}, 0.0
        for tsi, labi in zip(ts_list, lab_list):
            emissions = build_emissions(tsi, labi, w, pi, mask)
            ss = accumulate_sufficient_statistics(tsi, Q, pi, emissions,
                                                  labels=labi, soft_refs=soft or None)
            S_dwell += ss.S_dwell
            S_jumps += ss.S_jumps
            S_root += ss.S_root
            loglik += ss.loglik
            for node, cred in ss.S_cred.items():
                S_cred[node] = S_cred.get(node, np.zeros(2)) + cred
        return S_dwell, S_jumps, S_root, S_cred, loglik

    history = []
    prev = -np.inf
    with ExitStack() as stack:
        executor = paths = None
        if n_jobs > 1:
            from .parallel import make_pool, as_path
            executor = make_pool(n_jobs)
            if executor is not None:
                stack.callback(executor.shutdown)
                paths = [stack.enter_context(as_path(t)) for t in ts_list]   # dump once, reuse

        def estep_parallel():
            """Same statistics over member × tree-range chunks on the persistent pool."""
            from .parallel import _accumulate_range, add_suffstats, genome_chunks
            per_member = max(1, -(-n_jobs // len(ts_list)))   # ceil: ~n_jobs tasks total
            tasks = [(paths[i], lo, hi, Q, pi, w, lab_list[i], soft or None, mask)
                     for i in range(len(ts_list))
                     for (lo, hi) in genome_chunks(ts_list[i], per_member)]
            futures = [executor.submit(_accumulate_range, *t) for t in tasks]
            ss = reduce(add_suffstats, (f.result() for f in futures))   # task order = deterministic
            return ss.S_dwell, ss.S_jumps, ss.S_root, ss.S_cred, ss.loglik

        bar = None
        if progress:
            from tqdm.auto import tqdm
            bar = tqdm(total=max_iter, desc="EM fit", unit="iter")
            stack.callback(bar.close)

        for _ in range(max_iter):
            S_dwell, S_jumps, S_root, S_cred, loglik = (
                estep_parallel() if executor is not None else estep_serial())
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
                    a, b = tip_priors.get(s, (alpha, beta))
                    w[s] = m_step_w(agree, disagree, a, b)

            if bar is not None:
                bar.update(1)
                bar.set_postfix(loglik=f"{loglik:.1f}")
            if len(history) > 1 and abs(loglik - prev) < tol:
                break
            prev = loglik

    return FitResult(Q, pi, w, history)
