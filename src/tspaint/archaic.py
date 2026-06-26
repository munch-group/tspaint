"""Reference-free archaic / ghost detection (Plan B; CLAUDE.md §9, plans/PLAN_B_archaic_detector.md).

Plan A's :func:`tspaint.detect_ghost` flags tracts from an unsampled source with a **fixed
threshold** on nearest-reference coalescence depth. Plan B promotes that depth signal from a
flag to a **generative latent state**: per sample, a 2-state HMM along the genome whose hidden
state is ``modern`` vs ``archaic`` and whose emission is keyed to **branch depth, not a label**
— so the archaic state needs no reference.

Formulation (the plan's "deep-coalescence emission" variant, realised on the genome axis —
where the reference-free signal actually lives, since with no archaic tip there is nothing to
propagate vertically):

* observation per (sample, tree-interval) = ``log`` of the nearest-**modern**-reference
  coalescence time;
* the **modern** emission ``N(μ_m, σ_m²)`` is **anchored** by the reference panel's own
  nearest-other-reference depth distribution (the identifiability anchor — it pins the modern
  state and breaks §6 label-switching);
* the **archaic** emission ``N(μ_a, σ_a²)`` is learned, constrained ``μ_a ≥ μ_m + δ`` (deeper);
* the latent state switches at recombination breakpoints (a 2-state transition matrix), learned;
* fit by Baum–Welch **pooled across samples** (shared emission / transition params, per-sample
  state posteriors), giving a **calibrated** ``P(archaic)`` per locus instead of a hard call.

It is detection (*that* a tract descends from outside the panel), not attribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tskit

__all__ = ["ArchaicResult", "detect_archaic"]

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _nearest_modern_depth(tree, s, modern_ids, node_time):
    """Coalescent time to the nearest modern reference (excluding ``s`` itself)."""
    best = np.inf
    for r in modern_ids:
        if r == s:
            continue
        m = tree.mrca(s, r)
        if m == tskit.NULL:
            continue
        t = node_time[m]
        if t < best:
            best = t
    return float(best) if np.isfinite(best) else float("nan")


def _log_depth_track(ts, modern_ids, samples):
    """Per sample, contiguous ``(left, right, log_depth)`` of nearest-modern-ref coalescence."""
    node_time = ts.tables.nodes.time
    mids = [int(r) for r in modern_ids]
    tracks = {int(s): [] for s in samples}
    for tree in ts.trees():
        left, right = tree.interval.left, tree.interval.right
        for s in samples:
            d = _nearest_modern_depth(tree, int(s), mids, node_time)
            ld = float(np.log(d)) if (np.isfinite(d) and d > 0) else float("nan")
            segs = tracks[int(s)]
            same = segs and (segs[-1][2] == ld or (np.isnan(segs[-1][2]) and np.isnan(ld)))
            if segs and segs[-1][1] == left and same:
                segs[-1] = (segs[-1][0], right, ld)
            else:
                segs.append((left, right, ld))
    return tracks


def _anchor_modern(ref_tracks, q=0.98):
    """Modern depth anchor: span-weighted mean, std, and a high quantile of the reference
    log-depths.

    The quantile ``q_ref`` is the **deepest coalescence the modern panel itself produces**
    (≈ the source-divergence depth) — the principled archaic floor. Anchoring there rather than
    at ``μ_m + kσ_m`` is what stops the model from mistaking deep *within-panel* (cross-source)
    coalescences for archaic.
    """
    vals, ws = [], []
    for segs in ref_tracks.values():
        for (l, r, ld) in segs:
            if not np.isnan(ld):
                vals.append(ld)
                ws.append(r - l)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals = np.asarray(vals, float)
    ws = np.asarray(ws, float)
    mu = np.average(vals, weights=ws)
    sd = max(float(np.sqrt(np.average((vals - mu) ** 2, weights=ws))), 1e-3)
    order = np.argsort(vals)
    cw = np.cumsum(ws[order])
    cw /= cw[-1]
    q_ref = float(np.interp(q, cw, vals[order]))
    return mu, sd, q_ref


def _emission(obs, mu, sd):
    """``(T, 2)`` Gaussian emission likelihoods; missing observations are uninformative (1)."""
    T = obs.shape[0]
    B = np.ones((T, 2))
    ok = ~np.isnan(obs)
    x = obs[ok]
    for k in range(2):
        B[ok, k] = np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) / (sd[k] * _SQRT2PI)
    B[ok] = np.clip(B[ok], 1e-300, None)
    return B


def _forward_backward(B, A, pi0):
    """Scaled forward–backward: returns ``gamma (T,2)``, accumulated ``xi (2,2)``, loglik."""
    T = B.shape[0]
    alpha = np.zeros((T, 2))
    c = np.zeros(T)
    alpha[0] = pi0 * B[0]
    c[0] = alpha[0].sum() or 1.0
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = alpha[t].sum() or 1.0
        alpha[t] /= c[t]
    beta = np.zeros((T, 2))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((2, 2))
    for t in range(T - 1):
        xi += (alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]) / c[t + 1]
    return gamma, xi, float(np.log(c).sum())


@dataclass
class ArchaicResult:
    """Result of :func:`detect_archaic`.

    Attributes
    ----------
    posteriors : dict[int, list[tuple[float, float, float]]]
        Per sample, contiguous ``(left, right, P(archaic))`` over the genome.
    burden : dict[int, float]
        Per sample, the span-weighted mean ``P(archaic)`` (the genome-wide archaic burden).
    mu, sd : numpy.ndarray
        ``(2,)`` learned emission means / stds on log-depth (index 0 modern, 1 archaic).
    A : numpy.ndarray
        ``(2, 2)`` learned per-breakpoint transition matrix.
    pi0 : numpy.ndarray
        ``(2,)`` learned initial distribution.
    loglik_history : list
        Pooled Baum–Welch log-likelihood per iteration (non-decreasing).
    """
    posteriors: dict
    burden: dict
    mu: np.ndarray
    sd: np.ndarray
    A: np.ndarray
    pi0: np.ndarray
    loglik_history: list

    def tracts(self, sample, threshold=0.5):
        """Merged archaic tracts ``[(left, right)]`` where ``P(archaic) >= threshold``."""
        out = []
        for (l, r, p) in self.posteriors[int(sample)]:
            if p >= threshold:
                if out and abs(out[-1][1] - l) < 1e-9:
                    out[-1] = (out[-1][0], r)
                else:
                    out.append((l, r))
        return out


def detect_archaic(ts, labels, samples=None, *, max_iter=50, tol=1e-3, delta=None,
                   init_archaic_burden=0.1, init_switch=0.02):
    """Reference-free archaic / ghost detection via a depth-emission HMM (Plan B).

    Fits a per-sample 2-state (modern / archaic) HMM along the genome on nearest-modern-reference
    coalescence depth, the modern emission anchored by the panel, the archaic emission learned and
    constrained deeper. Returns a **calibrated** per-locus ``P(archaic)`` — no archaic reference
    required.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence to analyse.
    labels : dict[int, int] or iterable[int]
        The modern reference sample ids (a ``{ref: state}`` painting label dict is accepted — only
        the ids are used; every reference is "modern"). They anchor the modern depth distribution.
    samples : iterable[int], optional
        Samples to scan; defaults to every non-reference sample.
    max_iter, tol : int, float, optional
        Baum–Welch iteration cap and log-likelihood tolerance.
    delta : float, optional
        Minimum gap ``μ_a - μ_m`` (log-depth) enforced on the archaic mean. Defaults to
        ``0.5 * σ_m`` (half the modern depth spread).
    init_archaic_burden, init_switch : float, optional
        Initial archaic fraction and per-breakpoint switch probability.

    Returns
    -------
    ArchaicResult
        Per-sample ``P(archaic)`` posteriors / tracts / burden plus the learned HMM parameters.

    Raises
    ------
    ValueError
        If the modern anchor cannot be estimated (no informative reference depths).
    """
    modern_ids = [int(r) for r in (labels.keys() if isinstance(labels, dict)
                                   else labels)]
    modern_set = set(modern_ids)
    if samples is None:
        samples = [int(s) for s in ts.samples() if int(s) not in modern_set]
    else:
        samples = [int(s) for s in samples]

    ref_tracks = _log_depth_track(ts, modern_ids, modern_ids)
    mu_m, sd_m, q_ref = _anchor_modern(ref_tracks)
    if not np.isfinite(mu_m):
        raise ValueError("could not anchor the modern depth distribution (no informative references)")
    sd_m = max(sd_m, (q_ref - mu_m) / 2.0)   # widen modern to cover its own deep (cross-source) tail
    if delta is None:
        delta = sd_m                  # margin above the panel's deepest coalescence (§6 guard)
    archaic_floor = q_ref + delta      # archaic must be DEEPER than any modern (cross-source) coalescence

    sample_tracks = _log_depth_track(ts, modern_ids, samples)
    obs_list, span_list = [], []
    for s in samples:
        segs = sample_tracks[s]
        obs_list.append(np.array([ld for (_l, _r, ld) in segs], float))
        span_list.append(np.array([r - l for (l, r, _ld) in segs], float))

    # --- Baum-Welch (modern emission fixed = the anchor; archaic learned, deeper) ---
    mu = np.array([mu_m, archaic_floor + sd_m])
    sd = np.array([sd_m, sd_m])
    A = np.array([[1.0 - init_switch, init_switch], [init_switch, 1.0 - init_switch]])
    pi0 = np.array([1.0 - init_archaic_burden, init_archaic_burden])
    history, prev = [], -np.inf
    for _ in range(max_iter):
        acc_xi = np.zeros((2, 2))
        acc_pi = np.zeros(2)
        g_sum = np.zeros(2)
        gx = np.zeros(2)
        gxx = np.zeros(2)
        ll = 0.0
        for obs, span in zip(obs_list, span_list):
            if obs.shape[0] == 0:
                continue
            B = _emission(obs, mu, sd)
            gamma, xi, loglik = _forward_backward(B, A, pi0)
            ll += loglik
            acc_xi += xi
            acc_pi += gamma[0]
            ok = ~np.isnan(obs)
            w = span[ok]
            x = obs[ok]
            for k in range(2):
                g = gamma[ok, k] * w           # span-weighted emission sufficient stats
                g_sum[k] += g.sum()
                gx[k] += (g * x).sum()
                gxx[k] += (g * x * x).sum()
        history.append(ll)
        # M-step: transitions + initial; archaic emission learned, modern emission fixed
        row = acc_xi.sum(axis=1, keepdims=True)
        A = np.where(row > 0, acc_xi / np.where(row > 0, row, 1.0), A)
        # floor the switch probabilities so the HMM keeps modelling tracts (avoid the
        # A -> identity degenerate fixed point when within-haplotype switches are rare)
        A[0, 1] = max(A[0, 1], 1e-3)
        A[1, 0] = max(A[1, 0], 1e-3)
        A = A / A.sum(axis=1, keepdims=True)
        if acc_pi.sum() > 0:
            pi0 = acc_pi / acc_pi.sum()
        if g_sum[1] > 0:
            mu[1] = gx[1] / g_sum[1]
            var = max(gxx[1] / g_sum[1] - mu[1] ** 2, 1e-6)
            sd[1] = np.sqrt(var)
        mu[1] = max(mu[1], archaic_floor)      # keep the archaic state beyond the modern panel's range
        sd[1] = min(sd[1], sd_m)               # keep archaic tight (not a wide deep background)
        if len(history) > 1 and abs(ll - prev) < tol:
            break
        prev = ll

    # --- decode: per-sample P(archaic) posteriors, tracts, burden ---
    posteriors, burden = {}, {}
    for s, obs, span in zip(samples, obs_list, span_list):
        segs = sample_tracks[s]
        if obs.shape[0] == 0:
            posteriors[s] = []
            burden[s] = float("nan")
            continue
        B = _emission(obs, mu, sd)
        gamma, _xi, _ll = _forward_backward(B, A, pi0)
        pa = gamma[:, 1]
        posteriors[s] = [(segs[i][0], segs[i][1], float(pa[i])) for i in range(len(segs))]
        tot = span.sum()
        burden[s] = float((pa * span).sum() / tot) if tot > 0 else float("nan")

    return ArchaicResult(posteriors=posteriors, burden=burden, mu=mu, sd=sd, A=A, pi0=pi0,
                         loglik_history=history)
