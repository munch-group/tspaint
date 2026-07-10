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

from .output import Segment, INFORMATIVE, DEFAULT_DEADBAND
from .track import SoftTrack

__all__ = ["GhostResult", "detect_ghost", "ArchaicResult", "detect_archaic"]

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


def _depth_track(ts, modern_ids, samples, tree_range=None):
    """Per sample, contiguous ``(left, right, depth)`` of nearest-modern-ref coalescence **time**.

    Raw coalescent time (``nan`` where no modern reference is reachable); the ``log`` (``depth="time"``)
    or genome-wide ``rank`` (``depth="rank"``) transform is applied afterwards by :func:`_make_transform`.
    ``tree_range=(lo, hi)`` restricts the pass to that half-open marginal-tree-index range (a chunk
    worker for :func:`tspaint.parallel.depth_track_parallel`); ``None`` = the whole genome.
    """
    node_time = ts.tables.nodes.time
    mids = [int(r) for r in modern_ids]
    tracks = {int(s): [] for s in samples}
    lo, hi = (0, ts.num_trees) if tree_range is None else tree_range
    for ti, tree in enumerate(ts.trees()):
        if ti < lo:
            continue
        if ti >= hi:
            break
        left, right = tree.interval.left, tree.interval.right
        for s in samples:
            d = _nearest_modern_depth(tree, int(s), mids, node_time)
            d = float(d) if np.isfinite(d) else float("nan")
            segs = tracks[int(s)]
            same = segs and (segs[-1][2] == d or (np.isnan(segs[-1][2]) and np.isnan(d)))
            if segs and segs[-1][1] == left and same:
                segs[-1] = (segs[-1][0], right, d)
            else:
                segs.append((left, right, d))
    return tracks


def _make_transform(depth, *track_dicts):
    """The depth → observation transform: ``log`` time (``depth="time"``) or genome-wide span-weighted
    **rank** in ``[0, 1]`` (``depth="rank"``, monotonic ⇒ calibration-invariant).

    For ``"rank"`` the rank function is built from the pooled finite depths across ``track_dicts``
    (the references + samples of one member), so the modern panel occupies the low ranks and a deep
    ghost the high ranks regardless of branch-length calibration.
    """
    if depth == "time":
        return lambda t: float(np.log(t)) if (np.isfinite(t) and t > 0) else float("nan")
    if depth != "rank":
        raise ValueError("depth must be 'time' or 'rank'")
    vals, ws = [], []
    for tracks in track_dicts:
        for segs in tracks.values():
            for (l, r, t) in segs:
                if np.isfinite(t):
                    vals.append(t); ws.append(r - l)
    if not vals:
        return lambda t: float("nan")
    vals = np.asarray(vals, float)
    order = np.argsort(vals)
    v = vals[order]
    cw = np.cumsum(np.asarray(ws, float)[order])
    cw /= cw[-1]
    return lambda t: float(np.interp(t, v, cw)) if np.isfinite(t) else float("nan")


def _apply_transform(tracks, tfm):
    return {s: [(l, r, tfm(t)) for (l, r, t) in segs] for s, segs in tracks.items()}


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
class GhostResult(SoftTrack):
    """Result of :func:`detect_ghost` — per-locus ``P(ghost)`` from the depth-emission HMM.

    "Ghost" = ancestry from a population not in the reference panel; the signal is *depth*, so the
    detector targets **deep** (archaic-like) ghosts specifically (a shallow recent ghost will not
    trip it).

    A :class:`~tspaint.track.SoftTrack`, so it shares the painting read-out surface:
    :meth:`~tspaint.track.SoftTrack.segments` (hard ghost/modern tracts with the calibrated
    dead-band), :meth:`~tspaint.track.SoftTrack.posterior_at`, and
    :meth:`~tspaint.track.SoftTrack.plot` (whose colour scale highlights ``P(ghost)``).

    Attributes
    ----------
    posteriors : dict[int, list[Segment]]
        Per sample, contiguous :class:`~tspaint.output.Segment`\\ s tiling ``[0, L)`` whose
        ``(2,)`` ``posterior`` is ``[P(modern), P(ghost)]`` (state 0 modern, 1 ghost). For an
        **ensemble** input these are :class:`~tspaint.ensemble.MergedSegment`\\ s — the per-member
        mean plus a ``posterior_std`` band — but behave identically.
    burden : dict[int, float]
        Per sample, the span-weighted mean ``P(ghost)`` (the genome-wide ghost burden).
    mu, sd : numpy.ndarray
        ``(2,)`` learned emission means / stds on the depth observation — ``log`` coalescence time
        (``depth="time"``) or its genome-wide rank (``depth="rank"``); index 0 modern, 1 ghost.
    A : numpy.ndarray
        ``(2, 2)`` learned per-breakpoint transition matrix.
    pi0 : numpy.ndarray
        ``(2,)`` learned initial distribution.
    loglik_history : list
        Pooled Baum–Welch log-likelihood per iteration (non-decreasing).
    default_deadband : float
        Default dead-band passed to :meth:`~tspaint.track.SoftTrack.segments`. Defaults to
        :data:`~tspaint.output.DEFAULT_DEADBAND`.
    """

    # The ghost detector's "interesting" state is state 1 (ghost); the plot highlights P(ghost).
    _hi_state = 1
    _hi_label = "P(ghost)"

    posteriors: dict
    burden: dict
    mu: np.ndarray
    sd: np.ndarray
    A: np.ndarray
    pi0: np.ndarray
    loglik_history: list
    default_deadband: float = DEFAULT_DEADBAND
    _seqlen: float = None

    def tracts(self, sample, threshold=0.5):
        """Merged ghost tracts ``[(left, right)]`` where ``P(ghost) >= threshold``.

        Parameters
        ----------
        sample : int
            Sample-node id whose ghost tracts to extract (a key of :attr:`posteriors`).
        threshold : float, optional
            Minimum ``P(ghost)`` — ``posterior[1]``, the ghost state (state 1) — for a locus to
            count as ghost. Default ``0.5``.

        Returns
        -------
        list[tuple[float, float]]
            Merged ``(left, right)`` ghost spans (adjacent qualifying segments joined).
        """
        out = []
        for seg in self.posteriors[int(sample)]:
            if float(seg.posterior[1]) >= threshold:
                if out and abs(out[-1][1] - seg.left) < 1e-9:
                    out[-1] = (out[-1][0], seg.right)
                else:
                    out.append((seg.left, seg.right))
        return out

    def _summary_title(self, truth=None, deadband=None, samples=None):
        # The detector's states are [modern, ghost], not ancestry A/B, so opt out of SoftTrack's
        # ancestry-summary default title (plot() stays title-less unless a title is passed).
        return None

    def _prepare_truth(self, truth):
        # The truth for a ghost detector is a set of ghost tracts (e.g. Simulation.ghost_states): each
        # segment marks a truly-ghost region, whatever its own state index (the sources embed the ghost
        # above them, so it is 2/3/…). Binarise every segment to the detector's ghost state (state 1) so
        # the truth band reads white (non-ghost gaps) + one ghost colour — the same scheme as the soft
        # and hard bands — instead of the ancestry A/B/ghost legend.
        if not truth:
            return truth
        hi = self._hi_state
        return {q: [(float(l), float(r), hi) for (l, r, *_rest) in segs] for q, segs in truth.items()}

    def _make_colorizer(self, K, n_source, *, cmap, colors, alpha):
        # Single-colour "ghost highlight": state 1 (ghost) gets one hue, state 0 (modern) is white,
        # legend is the P(ghost) colour bar — not the ancestry A/B/… scheme (CLAUDE.md §9).
        from .track import _Colorizer
        return _Colorizer(K, hi=self._hi_state, hi_label=self._hi_label, n_source=n_source,
                          cmap=cmap, colors=colors, alpha=alpha, highlight=True)


def _baum_welch(obs_list, span_list, mu_m, sd_m, archaic_floor, *, max_iter, tol,
                init_archaic_burden, init_switch, ghost_scale=None):
    """Pooled Baum–Welch: modern emission fixed at the anchor, ghost emission learned (deeper).

    ``obs_list`` / ``span_list`` are the per-sequence observation / span arrays — pooled across an
    ensemble's members × samples so one shared ``(mu, sd, A, pi0)`` is learned. ``ghost_scale`` sets
    the ghost emission's initial offset above the floor and the cap on its learned std; it defaults
    to ``sd_m`` (the unbounded log-time behaviour) but is set smaller for the bounded rank scale.
    Returns ``(mu, sd, A, pi0, loglik_history)``.
    """
    if ghost_scale is None:
        ghost_scale = sd_m
    mu = np.array([mu_m, archaic_floor + ghost_scale])
    sd = np.array([sd_m, ghost_scale])
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
        # M-step: transitions + initial; ghost emission learned, modern emission fixed
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
        mu[1] = max(mu[1], archaic_floor)      # keep the ghost state beyond the modern panel's range
        sd[1] = min(sd[1], ghost_scale)        # keep ghost tight (not a wide deep background)
        if len(history) > 1 and abs(ll - prev) < tol:
            break
        prev = ll
    return mu, sd, A, pi0, history


def detect_ghost(ts, labels, samples=None, *, depth="time", max_iter=50, tol=1e-3, delta=None,
                 init_archaic_burden=0.1, init_switch=0.02, n_jobs=None, progress=False):
    """Reference-free ghost / archaic introgression detection via a depth-emission HMM.

    The dedicated ghost-search tool (Task 2; CLAUDE.md §9). A 2-state (modern / ghost) HMM along the
    genome on nearest-modern-reference coalescence depth — the modern emission anchored by the
    panel, the **ghost** emission learned and constrained deeper. Returns a **calibrated** per-locus
    ``P(ghost)`` with no ghost/archaic reference required. The signal is depth, so it targets **deep
    (archaic-like)** ghosts. (Renamed from ``detect_archaic``, which remains a deprecated alias.)

    Parameters
    ----------
    ts : tskit.TreeSequence or list[tskit.TreeSequence]
        One tree sequence, or **an ensemble** (e.g. SINGER posterior samples) — the HMM is fit once
        pooled across all members, decoded per member, and the per-locus ``P(ghost)`` **averaged**
        across members (marginalising ARG uncertainty for accuracy; CLAUDE.md §7.4). All members
        must share the same sample ids.
    labels : dict[int, int] or iterable[int]
        The modern reference sample ids (a ``{ref: state}`` painting label dict is accepted — only
        the ids are used; every reference is "modern"). They anchor the modern depth distribution.
    samples : iterable[int], optional
        Samples to scan; defaults to every non-reference sample.
    depth : {"time", "rank"}, optional
        The depth observation: ``"time"`` (default) uses ``log`` nearest-reference coalescence time;
        ``"rank"`` uses its genome-wide span-weighted **rank** — a monotonic transform, hence
        **calibration-invariant** (robust to branch-length miscalibration, e.g. on a Relate ARG).
    max_iter, tol : int, float, optional
        Baum–Welch iteration cap and log-likelihood tolerance.
    delta : float, optional
        Margin placing the ghost emission floor above ``q_ref``, the panel's deepest-coalescence
        quantile (in the ``depth`` units), so the ghost state stays identifiable (CLAUDE.md §6).
        Its default and bounds depend on ``depth``:

        * ``depth="time"`` — the floor is ``q_ref + delta``, unbounded; defaults to ``σ_modern``.
        * ``depth="rank"`` — rank space is bounded in ``[0, 1]``, so the log-time rule would
          overshoot the ceiling and park the ghost above every observation. The floor is instead
          ``q_ref + min(delta, room / 2)`` with ``room = max(1 - q_ref, 1e-3)``; a supplied
          ``delta`` is therefore **capped at half the remaining room**, and the default (``None``)
          is ``room / 2``. ``σ_modern`` is not used in this mode.
    init_archaic_burden, init_switch : float, optional
        Initial ghost fraction and per-breakpoint switch probability.
    n_jobs : int, optional
        Worker processes for the nearest-modern-ref coalescence pass (split by genome chunk, one
        shared pool across ensemble members; exactly equal to serial). Default: all CPUs / the SLURM
        allocation (:func:`tspaint.parallel.resolve_cores`); pass ``1`` for serial.
    progress : bool, optional
        Show a progress bar for the depth pass. Default ``False``.

    Returns
    -------
    GhostResult
        Per-sample ``P(ghost)`` posteriors / tracts / burden plus the learned HMM parameters. For an
        ensemble the posteriors are the per-member mean.

    Raises
    ------
    ValueError
        If the modern anchor cannot be estimated (no informative reference depths), ``depth`` is
        invalid, or the ensemble is empty.
    """
    members = list(ts) if isinstance(ts, (list, tuple)) else [ts]
    if not members:
        raise ValueError("detect_ghost got an empty ensemble; pass at least one tree sequence")
    if depth not in ("time", "rank"):
        raise ValueError("depth must be 'time' or 'rank'")
    from .ids import resolve_ids
    modern_ids = resolve_ids(members[0], list(labels.keys() if isinstance(labels, dict) else labels))
    modern_set = set(modern_ids)
    if samples is None:
        samples = [int(s) for s in members[0].samples() if int(s) not in modern_set]
    else:
        samples = resolve_ids(members[0], samples)

    # per-member transformed depth tracks (log-time, or per-member calibration-invariant rank); the
    # nearest-modern-ref coalescence pass is the heavy part -> parallelise it over genome chunks,
    # sharing one worker pool across ensemble members.
    from .parallel import make_pool, depth_track_parallel
    ref_per_member, samp_per_member = [], []
    ex = make_pool(n_jobs)                        # None when serial (n_jobs <= 1)
    try:
        for g in members:
            ref_t = depth_track_parallel(g, modern_ids, modern_ids, executor=ex, n_jobs=n_jobs,
                                         progress=progress)
            samp_t = depth_track_parallel(g, modern_ids, samples, executor=ex, n_jobs=n_jobs,
                                          progress=progress)
            tfm = _make_transform(depth, ref_t, samp_t)
            ref_per_member.append(_apply_transform(ref_t, tfm))
            samp_per_member.append(_apply_transform(samp_t, tfm))
    finally:
        if ex is not None:
            ex.shutdown()

    # anchor the modern state from the pooled reference observations across members
    pooled_ref = {}
    for mi, ref_t in enumerate(ref_per_member):
        for r, segs in ref_t.items():
            pooled_ref[(mi, r)] = segs
    mu_m, sd_m, q_ref = _anchor_modern(pooled_ref)
    if not np.isfinite(mu_m):
        raise ValueError("could not anchor the modern depth distribution (no informative references)")
    sd_m = max(sd_m, (q_ref - mu_m) / 2.0)   # widen modern to cover its own deep (cross-source) tail
    if depth == "rank":
        # Rank space is bounded in [0, 1]: the unbounded log-time rule ``floor = q_ref + sd_m``
        # overshoots the ceiling (q_ref -> 1, sd_m ~ 0.25), centring the ghost state *above* every
        # observation so P(ghost) collapses to 0 (no detection). Place the floor in the room that
        # remains above the panel's deepest rank, and scale the ghost emission to that room.
        room = max(1.0 - q_ref, 1e-3)
        ghost_scale = 0.5 * room
        archaic_floor = q_ref + (0.5 * room if delta is None else min(delta, 0.5 * room))
    else:
        ghost_scale = sd_m
        if delta is None:
            delta = sd_m              # margin above the panel's deepest coalescence (§6 guard)
        archaic_floor = q_ref + delta  # ghost must be DEEPER than any modern (cross-source) coalescence

    # pooled Baum-Welch over all (member, sample) observation sequences
    obs_list, span_list = [], []
    for samp_t in samp_per_member:
        for s in samples:
            segs = samp_t[s]
            obs_list.append(np.array([o for (_l, _r, o) in segs], float))
            span_list.append(np.array([r - l for (l, r, _o) in segs], float))
    mu, sd, A, pi0, history = _baum_welch(obs_list, span_list, mu_m, sd_m, archaic_floor,
                                          max_iter=max_iter, tol=tol,
                                          init_archaic_burden=init_archaic_burden,
                                          init_switch=init_switch, ghost_scale=ghost_scale)

    # decode per member as Segment tables ([P(modern), P(ghost)] per locus), then merge each
    # sample across members (mean posterior + uncertainty band) reusing the painting ensemble merge.
    per_member_tables = []
    for samp_t in samp_per_member:
        table = {}
        for s in samples:
            segs = samp_t[s]
            if not segs:
                table[s] = []
                continue
            obs = np.array([o for (_l, _r, o) in segs], float)
            B = _emission(obs, mu, sd)
            gamma, _xi, _ll = _forward_backward(B, A, pi0)   # gamma[:,0]=P(modern), [:,1]=P(ghost)
            table[s] = [Segment(segs[i][0], segs[i][1], gamma[i].copy(), INFORMATIVE)
                        for i in range(len(segs))]
        per_member_tables.append(table)

    if len(per_member_tables) == 1:
        posteriors = per_member_tables[0]
    else:
        from .ensemble import merge_posterior_tables
        posteriors = merge_posterior_tables(per_member_tables, samples=samples)

    burden = {}
    for s in samples:
        segs = posteriors[s]
        tot = sum(seg.right - seg.left for seg in segs)
        burden[s] = (float(sum(seg.posterior[1] * (seg.right - seg.left) for seg in segs) / tot)
                     if tot > 0 else float("nan"))

    return GhostResult(posteriors=posteriors, burden=burden, mu=mu, sd=sd, A=A, pi0=pi0,
                       loglik_history=history, _seqlen=float(members[0].sequence_length))


# --- deprecated aliases (the detector was renamed detect_archaic -> detect_ghost) ------------

ArchaicResult = GhostResult


def detect_archaic(*args, **kwargs):
    """Deprecated alias for :func:`detect_ghost` (the detector was renamed ``detect_archaic`` ->
    ``detect_ghost``).

    Forwards every positional and keyword argument to :func:`detect_ghost` unchanged and emits a
    :class:`DeprecationWarning`; prefer :func:`detect_ghost` in new code.

    Returns
    -------
    GhostResult
        Exactly what :func:`detect_ghost` returns (``ArchaicResult`` is a deprecated alias of
        :class:`GhostResult`).
    """
    import warnings
    warnings.warn("tspaint.detect_archaic is deprecated; use tspaint.detect_ghost",
                  DeprecationWarning, stacklevel=2)
    return detect_ghost(*args, **kwargs)
