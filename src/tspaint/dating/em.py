"""EM loop for admixture rate through time (admix-dating design §4, rung 4).

Iterate the time-inhomogeneous E-step (:func:`tspaint.dating.estep.accumulate_time_binned_tv`) and
the directional penalised-spline M-step (:func:`tspaint.dating.mstep.directional_rate_splines`):

    init  q_AB(t)=q_BA(t)=const  (from the homogeneous tspaint.fit)
    E     accumulate per-cell occupation D and directional jumps J under the current Q(t)
    M     refit q_AB(t), q_BA(t) as penalised-Poisson splines -> new Q(t)

until the (span-integrated) log-likelihood converges. The payoff over the Stage-1 binned profile
is a *sharp* rate-through-time: the per-cell rate is no longer smeared by a single global Q.
"""
from __future__ import annotations

import re
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .grid import cell_centers, log_time_grid, assert_calibrated
from .estep import accumulate_time_binned_tv
from .mstep import directional_rate_splines

__all__ = ["RateThroughTime", "make_Q_of_cell", "fit_rate_through_time",
           "split_time", "split_times", "EnsembleRateThroughTime"]

def _diverging_sm(cmap, colors):
    """A ``[0, 1]`` ScalarMappable, optionally from a custom ``colors`` diverging colormap."""
    import matplotlib
    if colors:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
    return matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap)

def _letters(K):
    """Default ancestry-state names when none are given: ``A, B, C, …`` (``S26, S27, …`` past Z)."""
    return [chr(65 + i) if i < 26 else f"S{i}" for i in range(int(K))]


def _q_label(names, m, n):
    return f"q({names[m]}→{names[n]})"                           # e.g. q(A→B) / q(A_prox→A)


def _pair_curves(ax, centers, q, scale, colors, *, lw=2, alpha=1.0, label=True, names=None):
    """Plot every directional rate in ``q`` (``(n_cells, K, K)``) onto a single ``ax``, one line per
    ordered off-diagonal pair. Each **unordered** pair ``{m, n}`` shares a colour: ``q_{mn}``
    (``m<n``) solid, ``q_{nm}`` dashed — so ``q_AB`` / ``q_BA`` read as one colour, two styles. Lines
    are labelled by state ``names`` (default ``A, B, …``). The caller adds the legend;
    ``label=False`` suppresses the labels (faint per-member ensemble curves)."""
    q = np.asarray(q, float)
    names = names if names is not None else _letters(q.shape[1])
    pairs = _unordered_pairs(q.shape[1])
    pal = list(colors) if colors is not None else [f"C{i}" for i in range(max(len(pairs), 1))]
    for i, (m, n) in enumerate(pairs):
        c = pal[i % len(pal)]
        ax.plot(centers, q[:, m, n] * scale, "-", lw=lw, alpha=alpha, color=c,
                label=(_q_label(names, m, n) if label else None))
        ax.plot(centers, q[:, n, m] * scale, "--", lw=lw, alpha=alpha, color=c,
                label=(_q_label(names, n, m) if label else None))


def _one_pair(ax, centers, q, m, n, scale, color, *, lw=2, alpha=1.0, label=True, names=None):
    """Draw the single unordered pair ``{m, n}`` on ``ax``: ``q_{mn}`` solid, ``q_{nm}`` dashed."""
    q = np.asarray(q, float)
    names = names if names is not None else _letters(q.shape[1])
    ax.plot(centers, q[:, m, n] * scale, "-", lw=lw, alpha=alpha, color=color,
            label=(_q_label(names, m, n) if label else None))
    ax.plot(centers, q[:, n, m] * scale, "--", lw=lw, alpha=alpha, color=color,
            label=(_q_label(names, n, m) if label else None))


def _facet_axes(n_pairs, ax):
    """``n_pairs`` stacked axes (shared x) — create a figure when ``ax`` is ``None``, else validate a
    supplied array of axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        _f, axes = plt.subplots(n_pairs, 1, figsize=(7, 2.2 * n_pairs + 0.8),
                                sharex=True, squeeze=False)
        return list(axes[:, 0])
    axes = list(np.atleast_1d(ax).ravel())
    if len(axes) < n_pairs:
        raise ValueError(f"faceted plot needs {n_pairs} axes for {n_pairs} ancestry pairs; "
                         f"got {len(axes)}")
    return axes[:n_pairs]


@dataclass
class RateThroughTime:
    """Result of :func:`fit_rate_through_time` — the directional cross-ancestry rate profile.

    Attributes
    ----------
    centers : (n_cells,) ndarray
        Cell centres (generations ago).
    q : (n_cells, K, K) ndarray
        Directional transition rate per cell for every ordered ancestry pair — ``q[:, m, n]`` is the
        rate ``m → n`` (forward in time: parent→child = old→young; a *backward*-time admixture A→B
        shows up in ``q[:, 1, 0]`` = :attr:`q_BA`). The diagonal is 0. For ``K > 2`` there are
        ``K·(K-1)`` directional rates: read any of them with :meth:`rate` — **by population name**
        (``rate("A", "B")``) or index (``rate(0, 1)``) — enumerate the unordered pairs with
        :attr:`pairs`, and get per-pair split times with :func:`split_times`. :attr:`q_AB` /
        :attr:`q_BA` are **2-state-only** convenience aliases (they raise for ``K != 2`` rather than
        silently returning just the ``0↔1`` slice).
    D : (n_cells, K) ndarray
        Per-cell occupation (the exposure / informative window).
    J : (n_cells, K, K) ndarray
        Per-cell directional expected jumps.
    loglik_history : list
        Span-integrated log-likelihood per EM iteration (non-decreasing).
    states : list[str] or None
        Ancestry-state **names** in state-index order (``states[i]`` names state ``i``), from the
        population names in the fit's ``labels``. ``None`` when the labels were plain integer states,
        in which case the names default to letters ``A, B, C, …`` (:attr:`state_names`). All
        reporting — :attr:`pairs`, :meth:`rate`, :func:`split_times`, :meth:`plot` — is by name.
    """
    centers: np.ndarray
    q: np.ndarray
    D: np.ndarray
    J: np.ndarray
    loglik_history: list
    states: list = None

    @property
    def K(self):
        """Number of ancestry states."""
        return int(np.asarray(self.q).shape[1])

    @property
    def state_names(self):
        """Ancestry-state names in state-index order — :attr:`states` if given, else ``A, B, C, …``."""
        return list(self.states) if self.states is not None else _letters(self.K)

    def _state_index(self, x):
        """Resolve an ancestry state given as an integer index **or** a population name to its index."""
        if isinstance(x, (int, np.integer)):
            return int(x)
        names = self.state_names
        if x in names:
            return names.index(x)
        raise KeyError(f"unknown ancestry state {x!r}; states are {names}")

    @property
    def pairs(self):
        """The ``K·(K-1)/2`` unordered ancestry pairs as **name** tuples ``(name_m, name_n)`` with
        ``m < n`` (the keys of :func:`split_times`; the facets of :meth:`plot`)."""
        names = self.state_names
        return [(names[m], names[n]) for (m, n) in _unordered_pairs(self.K)]

    @property
    def q_AB(self):
        """Directional rate ``0 → 1`` (``A → B``) — ``q[:, 0, 1]`` (2-state only).

        Raises
        ------
        ValueError
            If ``K != 2``. Use ``rate(0, 1)`` for the ``0→1`` slice, :meth:`rate` for any pair, or
            :attr:`q` for the full ``(n_cells, K, K)`` array.
        """
        self._require_2state("q_AB")
        return np.asarray(self.q, float)[:, 0, 1]

    @property
    def q_BA(self):
        """Directional rate ``1 → 0`` (``B → A``) — ``q[:, 1, 0]`` (2-state only).

        Raises
        ------
        ValueError
            If ``K != 2``. Use ``rate(1, 0)`` for the ``1→0`` slice, :meth:`rate` for any pair, or
            :attr:`q` for the full ``(n_cells, K, K)`` array.
        """
        self._require_2state("q_BA")
        return np.asarray(self.q, float)[:, 1, 0]

    def _require_2state(self, name):
        if self.K != 2:
            raise ValueError(
                f"{name} is a 2-state alias; this profile has K={self.K}. Use rate(m, n) for any "
                "directional rate, .pairs to enumerate them, .q for the full array, and "
                "split_times(rtt) for per-pair split times.")

    def rate(self, m, n):
        """Directional rate ``m → n`` per cell — ``q[:, m, n]``.

        Parameters
        ----------
        m, n : int or str
            Source and destination ancestry states, each a population **name** (e.g. ``"A"``,
            resolved via :attr:`state_names`) or an integer state index.

        Returns
        -------
        numpy.ndarray
            ``(n_cells,)`` directional rate ``m → n`` per cell.
        """
        return np.asarray(self.q, float)[:, self._state_index(m), self._state_index(n)]

    def plot(self, ax=None, scale=1.0, colors=None, facet=None, logy=False, mark_split=True):
        """Plot the directional rate-through-time profile (log-time x-axis).

        Each **unordered** ancestry pair ``{m, n}`` is drawn as ``q_{mn}`` (``m<n``) **solid** and
        ``q_{nm}`` **dashed** in one colour (so for ``K=2`` ``q_AB`` and ``q_BA`` are one colour,
        two styles). Because the pairs can sit on very different scales, ``K > 2`` **facets by
        default** — one subplot per pair, each with its own independent y-axis — while ``K = 2``
        stays a single overlaid axis. ``scale`` multiplies the rates (e.g. pass ``Ne`` to show
        ``rate × N``); ``colors`` overrides the per-pair colour cycle.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or array of Axes, optional
            Axes to draw on. Default ``None`` — a new figure is created (a single axes when not
            faceting, or a column of ``len(pairs)`` axes when faceting). When faceting, an
            array of at least ``len(pairs)`` axes may be supplied.
        scale : float, optional
            Multiply the plotted rates (e.g. pass ``Ne`` to show ``rate × N``). Default ``1.0``.
        colors : list, optional
            Per-pair colour override. Default ``None`` — the default Matplotlib colour cycle.
        facet : bool, optional
            One subplot per unordered pair (independent y-axes) vs. a single overlaid axis. Default
            ``None`` — facet when ``K > 2``, overlay when ``K == 2``.
        logy : bool, optional
            Use a log y-axis. Default ``False`` (linear; each facet still autoscales independently).
        mark_split : bool, optional
            Draw a dotted vertical line at each pair's split time (:func:`split_times`). Default
            ``True`` (drawn per facet, and for the single ``K = 2`` axis).

        Returns
        -------
        matplotlib.axes.Axes or list[matplotlib.axes.Axes]
            The single axes (overlay) or the list of per-pair axes (faceted).
        """
        import matplotlib.pyplot as plt
        names = self.state_names
        idx_pairs = _unordered_pairs(self.K)                    # (m, n) integer indices
        name_pairs = self.pairs                                 # (name_m, name_n), same order
        if facet is None:
            facet = len(idx_pairs) > 1
        splits = split_times(self) if mark_split else {}        # keyed by name pairs
        ylabel = "rate" + (" × scale" if scale != 1.0 else "")

        if not facet:
            if ax is None:
                _fig, ax = plt.subplots(figsize=(7, 4.2))
            _pair_curves(ax, self.centers, self.q, scale, colors, names=names)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("time (generations ago)")
            ax.set_ylabel(ylabel)
            if mark_split and len(idx_pairs) == 1 and np.isfinite(splits.get(name_pairs[0], np.nan)):
                ax.axvline(splits[name_pairs[0]], color="k", lw=1.0, ls=":")
            ax.legend()
            return ax

        axes = _facet_axes(len(idx_pairs), ax)
        pal = list(colors) if colors is not None else [f"C{i}" for i in range(len(idx_pairs))]
        for a, i, (m, n), np_pair in zip(axes, range(len(idx_pairs)), idx_pairs, name_pairs):
            _one_pair(a, self.centers, self.q, m, n, scale, pal[i % len(pal)], names=names)
            a.set_xscale("log")
            if logy:
                a.set_yscale("log")
            a.set_ylabel(ylabel)
            if mark_split and np.isfinite(splits.get(np_pair, np.nan)):
                a.axvline(splits[np_pair], color="k", lw=1.0, ls=":")
            a.legend(fontsize=8, loc="upper left")
        axes[-1].set_xlabel("time (generations ago)")
        return axes


def _unordered_pairs(K):
    """The ``K·(K-1)/2`` unordered ancestry pairs ``(m, n)`` with ``m < n``."""
    return list(combinations(range(int(K)), 2))


def _pair_split_time(centers, rate, exposure=None, exposure_frac=0.01):
    """Split time of one unordered pair from its combined rate ``rate = q_mn + q_nm`` vs. time.

    The rate **steps** at the split (not a peak): for two genuine source ancestries it is ~0 while
    they are still separate looking backward and **rises** once ``t`` exceeds the split (a rising
    onset); for a reference-*proxy* pair — a source and a proxy that diverged from it — it is
    **high at recent times and falls after the split** (a falling edge), because the two labels are
    genealogically interchangeable until they split. This estimator handles **both**: it locates
    the half-amplitude crossing in whichever direction the rate transitions with age.

    Parameters
    ----------
    centers : (n_cells,) array_like
        Cell centres, ascending time (generations ago).
    rate : (n_cells,) array_like
        The unordered pair's combined directional rate ``q[:, m, n] + q[:, n, m]`` per cell.
    exposure : (n_cells,) array_like, optional
        Per-cell occupation for the two states (``D[:, m] + D[:, n]``); cells below
        ``exposure_frac`` of its max are ignored so a deep no-data cell cannot masquerade as the
        transition. Default ``None`` — no exposure guard.
    exposure_frac : float, optional
        Occupation threshold as a fraction of the max. Default ``0.01``.

    Returns
    -------
    float
        The split-time estimate (generations ago), or ``nan`` if the rate has no clear transition.
    """
    c = np.asarray(centers, float)
    r = np.asarray(rate, float).astype(float, copy=True)
    if exposure is not None:
        e = np.asarray(exposure, float)
        emax = float(np.nanmax(e)) if e.size else 0.0
        if emax > 0:
            r = np.where(e >= exposure_frac * emax, r, np.nan)
    fin = np.isfinite(r)
    if int(fin.sum()) < 2:
        return float("nan")
    rv, cv = r[fin], c[fin]                              # exposed cells only, ascending time
    r_hi, r_lo = float(np.max(rv)), float(np.min(rv))
    if r_hi <= 0 or (r_hi - r_lo) <= 0:
        return float("nan")
    thr = r_lo + 0.5 * (r_hi - r_lo)                     # half amplitude
    k = max(1, len(rv) // 4)
    young = float(np.mean(rv[:k]))                       # recent (small t)
    deep = float(np.mean(rv[-k:]))                       # ancient (large t)
    if deep >= young:                                    # low -> high with age: rising onset
        hits = np.nonzero(rv >= thr)[0]
    else:                                                # high -> low with age: proxy falling edge
        hits = np.nonzero(rv <= thr)[0]
    return float(cv[hits[0]]) if hits.size else float("nan")


def split_times(rtt, exposure_frac=0.01):
    """Per-**pair** split / divergence times from a directional rate profile.

    Generalises :func:`split_time` to any ``K``: for each unordered ancestry pair the split is
    estimated from that pair's own combined rate ``q[:, m, n] + q[:, n, m]`` and its own occupation
    ``D[:, m] + D[:, n]``, so pairs on very different rate scales each get their own
    (exposure-guarded) transition — and a deep, faint pair is not swamped by a recent, strong one.
    Each pair auto-detects a **rising onset** (source–source) or a **falling edge**
    (reference-proxy) — see :func:`_pair_split_time`. The dict is keyed by **population-name** pairs
    (``rtt.state_names``; letters ``A, B, …`` when the fit had integer-state labels).

    Parameters
    ----------
    rtt : RateThroughTime or EnsembleRateThroughTime
        A fitted profile (uses ``.centers``, ``.q``, ``.state_names`` and, when present, ``.D``).
    exposure_frac : float, optional
        Ignore cells whose pair occupation is below this fraction of its max. Default ``0.01``.

    Returns
    -------
    dict[tuple[str, str], float]
        ``{(name_m, name_n): split_time}`` (generations ago) for every unordered pair ``m < n``; a
        pair with no clear transition maps to ``nan``.
    """
    c = np.asarray(rtt.centers, float)
    q = np.asarray(rtt.q, float)
    D = getattr(rtt, "D", None)
    D = np.asarray(D, float) if D is not None else None
    names = getattr(rtt, "state_names", None) or _letters(q.shape[1])
    out = {}
    for (m, n) in _unordered_pairs(q.shape[1]):
        rate = q[:, m, n] + q[:, n, m]
        expo = (D[:, m] + D[:, n]) if D is not None else None
        out[(names[m], names[n])] = _pair_split_time(c, rate, expo, exposure_frac)
    return out


def split_time(rtt, exposure_frac=0.01):
    """Estimate the split / divergence time from a **2-state** :class:`RateThroughTime` profile.

    The combined cross-ancestry rate ``q_AB(t) + q_BA(t)`` **steps** at the split (not a peak). For
    two genuine source ancestries it is ~0 below the split (they are still separate looking
    backward) and rises once ``t`` exceeds it — a *rising onset*. For a reference-*proxy* pair it is
    instead high at recent times and **falls after the split** — a *falling edge*. This returns the
    half-amplitude crossing in whichever direction the rate transitions with age (cells with
    negligible occupation ``D`` are ignored, so a deep no-data cell cannot masquerade as it).

    Parameters
    ----------
    rtt : RateThroughTime
        A fitted 2-state directional rate profile.
    exposure_frac : float, optional
        Ignore cells whose occupation ``D`` is below this fraction of the max. Default ``0.01``.

    Returns
    -------
    float
        The split-time estimate (generations ago), or ``nan`` if the cross-rate has no clear
        transition.

    Raises
    ------
    ValueError
        If the profile has ``K != 2``. Use :func:`split_times` for a per-pair dict.
    """
    K = int(np.asarray(rtt.q).shape[1])
    if K != 2:
        raise ValueError(
            f"split_time is for K=2; this profile has K={K}. Use split_times(rtt) for a per-pair "
            "dict of split times (each pair auto-detects a rising or falling transition).")
    st = split_times(rtt, exposure_frac=exposure_frac)          # one unordered pair at K=2
    return next(iter(st.values()), float("nan"))


@dataclass
class EnsembleRateThroughTime:
    """Rate through time across an ARG ensemble — one :class:`RateThroughTime` per member, with a
    **confidence interval on the split time** from the ensemble spread.

    Returned by :meth:`tspaint.Painting.rate_through_time` when the painting was built from an
    ensemble of tree sequences (e.g. SINGER posterior samples): each member is dated on the shared
    fit, and the per-member split-time estimates become an interval — the ARG-uncertainty band on
    *when* the ancestries diverged. For ``K > 2`` use the per-pair methods (:meth:`pair_split_times`
    / :meth:`pair_split_time_ci`); the scalar :meth:`split_time` / :meth:`split_time_ci` are
    2-state-only.

    Attributes
    ----------
    members : list[RateThroughTime]
        Per-member directional rate profiles, on a shared time grid (so they average).
    split_times : numpy.ndarray
        ``(M,)`` per-member scalar split-time estimates (generations ago) for a 2-state fit; all
        ``nan`` for ``K > 2`` (use :meth:`pair_split_times`). Build with :meth:`from_members`.
    """
    members: list
    split_times: np.ndarray

    @classmethod
    def from_members(cls, members):
        """Build an ensemble from per-member :class:`RateThroughTime` profiles.

        Populates :attr:`split_times` with each member's scalar split time for a 2-state fit
        (``nan`` per member for ``K > 2`` — use :meth:`pair_split_times` there).

        Parameters
        ----------
        members : iterable[RateThroughTime]
            Per-member directional rate profiles, sharing one time grid.

        Returns
        -------
        EnsembleRateThroughTime
            The assembled ensemble.
        """
        members = list(members)
        sts = np.array([split_time(m) if m.K == 2 else float("nan") for m in members], float)
        return cls(members, sts)

    @property
    def K(self):
        """Number of ancestry states."""
        return int(self.members[0].K)

    @property
    def state_names(self):
        """Ancestry-state names in state-index order — from the members (letters when unnamed)."""
        return self.members[0].state_names

    @property
    def pairs(self):
        """The unordered ancestry pairs as **name** tuples ``(name_m, name_n)`` with ``m < n``
        (the keys of :meth:`pair_split_times`)."""
        names = self.state_names
        return [(names[m], names[n]) for (m, n) in _unordered_pairs(self.K)]

    @property
    def centers(self):
        """Shared cell centres (generations ago) — taken from the first member (all members share
        one time grid, so they average)."""
        return self.members[0].centers

    @property
    def q(self):
        """Ensemble-mean directional rate array ``(n_cells, K, K)``."""
        return np.nanmean(np.stack([np.asarray(m.q, float) for m in self.members], axis=0), axis=0)

    @property
    def D(self):
        """Ensemble-mean per-cell occupation ``(n_cells, K)`` (the exposure for :func:`split_times`)."""
        return np.nanmean(np.stack([np.asarray(m.D, float) for m in self.members], axis=0), axis=0)

    @property
    def q_AB(self):
        """Ensemble-mean ``q_AB(t)`` — ``q[:, 0, 1]`` (2-state only; raises for ``K != 2``)."""
        self._require_2state("q_AB")
        return self.q[:, 0, 1]

    @property
    def q_BA(self):
        """Ensemble-mean ``q_BA(t)`` — ``q[:, 1, 0]`` (2-state only; raises for ``K != 2``)."""
        self._require_2state("q_BA")
        return self.q[:, 1, 0]

    def _require_2state(self, name):
        if self.K != 2:
            raise ValueError(
                f"{name} is a 2-state alias; this ensemble has K={self.K}. Use .q (mean array), "
                ".rate via a member, .pairs, and pair_split_times() / pair_split_time_ci().")

    def split_time(self, statistic="median"):
        """Point estimate of the split time over members (``"median"`` or ``"mean"``; 2-state only).

        Parameters
        ----------
        statistic : str, optional
            ``"median"`` (default) or ``"mean"`` over the finite per-member split-time estimates.

        Returns
        -------
        float
            The point-estimate split time (generations ago), or ``nan`` if no member has a finite
            estimate.

        Raises
        ------
        ValueError
            If ``K != 2``. Use :meth:`pair_split_times` for a per-pair dict.
        """
        self._require_2state("split_time")
        v = self.split_times[np.isfinite(self.split_times)]
        if v.size == 0:
            return float("nan")
        return float(np.mean(v) if statistic == "mean" else np.median(v))

    def split_time_ci(self, level=0.95):
        """Percentile confidence interval ``(lo, hi)`` on the split time from the ensemble spread
        (2-state only).

        Parameters
        ----------
        level : float, optional
            Central mass of the interval (e.g. ``0.95`` for a 95% interval). Default ``0.95``.

        Returns
        -------
        tuple of float
            ``(lo, hi)`` percentile interval on the split time (generations ago), or
            ``(nan, nan)`` if no member has a finite estimate.

        Raises
        ------
        ValueError
            If ``K != 2``. Use :meth:`pair_split_time_ci` for a per-pair dict.
        """
        self._require_2state("split_time_ci")
        v = self.split_times[np.isfinite(self.split_times)]
        if v.size == 0:
            return (float("nan"), float("nan"))
        a = (1.0 - level) / 2.0 * 100.0
        return (float(np.percentile(v, a)), float(np.percentile(v, 100.0 - a)))

    def _per_member_pair_splits(self):
        """``{(m, n): [split per member]}`` from each member's :func:`split_times`."""
        per = {}
        for mem in self.members:
            for pr, t in split_times(mem).items():
                per.setdefault(pr, []).append(t)
        return per

    def pair_split_times(self, statistic="median"):
        """Per-**pair** point-estimate split time over members (works for any ``K``).

        Parameters
        ----------
        statistic : str, optional
            ``"median"`` (default) or ``"mean"`` over the finite per-member estimates of each pair.

        Returns
        -------
        dict[tuple[int, int], float]
            ``{(m, n): split_time}`` (generations ago), ``nan`` where no member has a finite value.
        """
        agg = np.nanmean if statistic == "mean" else np.nanmedian
        out = {}
        for pr, vals in self._per_member_pair_splits().items():
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            out[pr] = float(agg(v)) if v.size else float("nan")
        return out

    def pair_split_time_ci(self, level=0.95):
        """Per-**pair** percentile confidence interval ``(lo, hi)`` on the split time (any ``K``).

        Parameters
        ----------
        level : float, optional
            Central mass of the interval. Default ``0.95``.

        Returns
        -------
        dict[tuple[int, int], tuple[float, float]]
            ``{(m, n): (lo, hi)}`` (generations ago), ``(nan, nan)`` where no member has a finite
            value.
        """
        a = (1.0 - level) / 2.0 * 100.0
        out = {}
        for pr, vals in self._per_member_pair_splits().items():
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            out[pr] = ((float(np.percentile(v, a)), float(np.percentile(v, 100.0 - a)))
                       if v.size else (float("nan"), float("nan")))
        return out

    def plot(self, ax=None, scale=1.0, level=0.95, colors=None, facet=None, logy=False):
        """Per-member rate curves (faint) + the ensemble mean + the per-pair split-time CI band.

        Every directional rate is drawn, each **unordered** ancestry pair in one colour with
        ``q_{mn}`` (``m<n``) solid and ``q_{nm}`` dashed. Because the pairs can sit on very
        different scales, ``K > 2`` **facets by default** — one subplot per pair, each with its own
        y-axis and its own split-time CI band — while ``K = 2`` stays a single overlaid axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or array of Axes, optional
            Axes to draw on. Default ``None`` — a new figure (single axes, or a column of
            ``len(pairs)`` axes when faceting).
        scale : float, optional
            Multiply the plotted rates (e.g. pass ``Ne`` to show ``rate × N``). Default ``1.0``.
        level : float, optional
            Confidence level for the shaded split-time band (:meth:`pair_split_time_ci`).
            Default ``0.95``.
        colors : list, optional
            Per-pair colour override. Default ``None`` — the default Matplotlib colour cycle.
        facet : bool, optional
            One subplot per unordered pair vs. a single overlaid axis. Default ``None`` — facet when
            ``K > 2``, overlay when ``K == 2``.
        logy : bool, optional
            Use a log y-axis. Default ``False``.

        Returns
        -------
        matplotlib.axes.Axes or list[matplotlib.axes.Axes]
            The single axes (overlay) or the list of per-pair axes (faceted).
        """
        names = self.state_names
        idx_pairs = _unordered_pairs(self.K)                    # (m, n) integer indices
        name_pairs = self.pairs                                 # (name_m, name_n), same order
        if facet is None:
            facet = len(idx_pairs) > 1
        ci = self.pair_split_time_ci(level)                     # keyed by name pairs
        pt = self.pair_split_times()
        ylabel = "rate" + (" × scale" if scale != 1.0 else "")

        if not facet:
            import matplotlib.pyplot as plt
            if ax is None:
                _f, ax = plt.subplots(figsize=(7, 4.2))
            for mem in self.members:                    # faint per-member curves (no labels)
                _pair_curves(ax, mem.centers, mem.q, scale, colors, lw=0.6, alpha=0.3, label=False,
                             names=names)
            _pair_curves(ax, self.centers, self.q, scale, colors, lw=2.5, names=names)   # mean
            lo, hi = ci.get(name_pairs[0], (float("nan"), float("nan")))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.axvspan(lo, hi, color="0.5", alpha=0.2, label=f"split {int(level * 100)}% CI")
            st = pt.get(name_pairs[0], float("nan"))
            if np.isfinite(st):
                ax.axvline(st, color="k", lw=1.0, ls=":")
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("time (generations ago)")
            ax.set_ylabel(ylabel)
            ax.legend()
            return ax

        axes = _facet_axes(len(idx_pairs), ax)
        pal = list(colors) if colors is not None else [f"C{i}" for i in range(len(idx_pairs))]
        for a, i, (m, n), np_pair in zip(axes, range(len(idx_pairs)), idx_pairs, name_pairs):
            col = pal[i % len(pal)]
            for mem in self.members:
                _one_pair(a, mem.centers, mem.q, m, n, scale, col, lw=0.6, alpha=0.3, label=False,
                          names=names)
            _one_pair(a, self.centers, self.q, m, n, scale, col, lw=2.5, names=names)
            lo, hi = ci.get(np_pair, (float("nan"), float("nan")))
            if np.isfinite(lo) and np.isfinite(hi):
                a.axvspan(lo, hi, color="0.5", alpha=0.2, label=f"split {int(level * 100)}% CI")
            st = pt.get(np_pair, float("nan"))
            if np.isfinite(st):
                a.axvline(st, color="k", lw=1.0, ls=":")
            a.set_xscale("log")
            if logy:
                a.set_yscale("log")
            a.set_ylabel(ylabel)
            a.legend(fontsize=8, loc="upper left")
        axes[-1].set_xlabel("time (generations ago)")
        return axes

    def __repr__(self):
        if self.K != 2:
            pts = self.pair_split_times()
            body = ", ".join(f"{a}{b}={pts[(a, b)]:.0f}" for (a, b) in self.pairs)
            return f"EnsembleRateThroughTime(M={len(self.members)}, K={self.K}, split_times[{body}] gen)"
        lo, hi = self.split_time_ci()
        return (f"EnsembleRateThroughTime(M={len(self.members)}, "
                f"split_time={self.split_time():.0f} [{lo:.0f}, {hi:.0f}] gen)")


def make_Q_of_cell(q, q_BA=None):
    """Build a ``cell_index -> (K, K) generator`` callable from per-cell directional rates.

    ``q`` is the ``(n_cells, K, K)`` off-diagonal rate array; the returned generator sets each row's
    diagonal to ``-Σ`` of its off-diagonals. The legacy 2-state call ``make_Q_of_cell(q_AB, q_BA)``
    (two ``(n_cells,)`` arrays) is still accepted.

    Parameters
    ----------
    q : numpy.ndarray
        The ``(n_cells, K, K)`` off-diagonal rate array (its diagonal is ignored / overwritten). In
        the legacy 2-state call this is instead ``q_AB``, the ``(n_cells,)`` A→B rate array.
    q_BA : array_like, optional
        Legacy 2-state B→A rate array ``(n_cells,)``. Default ``None`` — ``q`` is taken as the full
        ``(n_cells, K, K)`` array; when given, ``q`` and ``q_BA`` are assembled into a
        ``(n_cells, 2, 2)`` array.

    Returns
    -------
    callable
        ``Q(cell_index) -> (K, K)`` generator with each row's diagonal set to ``-Σ`` of its
        off-diagonals (rows sum to 0); per-cell results are cached.
    """
    if q_BA is not None:                                        # legacy (q_AB, q_BA) -> (n_cells, 2, 2)
        q_AB = np.asarray(q, float)
        arr = np.zeros((len(q_AB), 2, 2))
        arr[:, 0, 1] = q_AB
        arr[:, 1, 0] = np.asarray(q_BA, float)
        q = arr
    q = np.asarray(q, float)
    cache = {}

    def Q(k):
        Qk = cache.get(k)
        if Qk is None:
            Qk = np.array(q[k], float)
            np.fill_diagonal(Qk, 0.0)
            Qk[np.diag_indices_from(Qk)] = -Qk.sum(axis=1)      # rows sum to 0
            cache[k] = Qk
        return Qk

    return Q


def _resolve_state_labels(labels):
    """Split ``labels`` into ``(int_state_labels, state_names)``.

    Label **values** may be integer states (``0, 1, …``) or population **names** (strings). If any
    value is a non-integer string, all values are treated as names: they are mapped to contiguous
    states in **sorted** order and the ordered name list is returned so the fit can report by name.
    Otherwise the values are cast to ``int`` and ``state_names`` is ``None`` (letters ``A, B, …``).
    """
    vals = list(labels.values())
    named = any(isinstance(v, str) and not re.fullmatch(r"-?\d+", v) for v in vals)
    if not named:
        return {k: int(v) for k, v in labels.items()}, None
    names = sorted({str(v) for v in vals})
    idx = {nm: i for i, nm in enumerate(names)}
    return {k: idx[str(v)] for k, v in labels.items()}, names


def fit_rate_through_time(ts, labels, edges=None, *, n_cells=40, n_iter=15, em_init=8, Q0=None,
                          estimate_pi=False, soft_refs=None, n_knots=20, tol=1e-4,
                          floor=1e-9, fit_result=None, mask=None, n_jobs=None):
    """Fit the time-inhomogeneous directional admixture-rate-through-time profile by EM.

    Parameters
    ----------
    ts : tskit.TreeSequence
    labels : dict
        Reference sample id -> ancestry state. Each key is a sample-node index or a sample-ID
        string; each value is either an **integer** state (``0, 1, …``) or a **population name**
        (a string, e.g. ``"A"`` / ``"B"``). When names are given they are mapped to states in
        sorted order and carried through, so the result reports rates and split times **by name**
        (:attr:`RateThroughTime.state_names`); integer states default to the letters ``A, B, C, …``.
    edges : array_like, optional
        Fine log-time grid edges (:func:`tspaint.dating.log_time_grid`). If ``None`` (default), a
        grid of ``n_cells`` log-spaced cells is built automatically from the node ages.
    n_cells : int
        Number of log-time cells when ``edges`` is auto-constructed (ignored if ``edges`` given).
    n_iter : int
        Maximum EM iterations.
    em_init : int
        Iterations of the homogeneous :func:`tspaint.fit` used to initialise (ignored when
        ``fit_result`` is supplied).
    Q0 : (K, K) array_like, optional
        Initial generator for the internal homogeneous :func:`tspaint.fit`. Default ``None`` —
        :func:`tspaint.fit` scales a default symmetric generator to the calibrated node-age axis
        (so the warm-start fit does not wash out on a deep time scale). Ignored when ``fit_result``
        is supplied.
    estimate_pi : bool, optional
        Whether the internal fit re-estimates the root frequencies ``π``. Default ``False`` — hold
        ``π`` fixed. ``π`` is a prior on the arbitrary GMRCA state, so holding it fixed is the
        robust default; estimating it from washing deep roots is what breaks (CLAUDE.md §6,
        π-identifiability). Ignored when ``fit_result`` is supplied.
    soft_refs : set[int], optional
        Labelled tips whose credibility ``w_i`` the internal fit **learns** (the rest stay
        hard-clamped anchors). Default ``None`` — every reference is an anchor. Ignored when
        ``fit_result`` is supplied.
    n_knots : int
        Spline knots for the M-step.
    tol : float
        Relative log-likelihood change for convergence.
    floor : float, optional
        Lower clamp on every per-cell off-diagonal rate — at initialisation and after each M-step
        (also the fill value for any NaN rate) — keeping the generator strictly positive and
        well-defined for the next E-step's ``expm`` composites. Default ``1e-9``.
    fit_result : tspaint.em.FitResult, optional
        A precomputed homogeneous fit to **warm-start** from (its ``Q`` seeds the rate splines and
        its ``w``/``pi`` build the tip emissions). When given, the internal homogeneous
        :func:`tspaint.fit` is skipped — used by :meth:`tspaint.Painting.rate_through_time` to reuse
        the painting's fit rather than refitting.
    mask : dict, optional
        **Fragment masking** (CLAUDE.md §2.3): ``{ref: [(left, right), ...]}`` per-reference spans
        over which that reference emits the query (unlabelled) emission, as for
        :func:`tspaint.paint` / :func:`tspaint.fit`. Keys may be node ids or sample-ID strings.
        Applied to both the internal warm-start fit and the time-inhomogeneous E-step, so a
        painting made with a mask dates under the same emissions.
        :meth:`tspaint.Painting.rate_through_time` forwards its painting's mask automatically.
        Default ``None`` (no masking).
    n_jobs : int, optional
        Worker processes for the time-inhomogeneous E-step, split across genome tree-ranges
        (:func:`tspaint.parallel.dating_estep_parallel`). Default ``None`` = all CPUs / the SLURM
        allocation (:func:`tspaint.parallel.resolve_cores`); pass ``1`` for serial. A pool is built
        once and reused across all EM iterations. The parallel E-step is ``allclose`` (not
        bit-identical) to serial — the per-cell statistics sum over tree-ranges in chunk order,
        and float ``+`` is not associative.

    Returns
    -------
    RateThroughTime
        The fitted directional rate profile: the log-time grid (``centers``), the per-cell
        generator array ``q`` (with the :attr:`~RateThroughTime.q_AB` /
        :attr:`~RateThroughTime.q_BA` views), the accumulated dwell / jump statistics ``D`` and
        ``J``, and the EM ``loglik_history``. Call :meth:`RateThroughTime.plot` to draw the
        profile, or :func:`tspaint.dating.split_time` to read off the divergence epoch.
    """
    from ..em import fit, build_emissions
    from ..ids import resolve_labels, resolve_nodes

    labels, states = _resolve_state_labels(labels)     # values may be population names -> int states
    labels = resolve_labels(ts, labels)                # keys may be sample-ID strings or node indices
    if mask:                                           # fragment masking (§2.3): keys id or node
        mask = {node: spans for k, spans in mask.items() for node in resolve_nodes(ts, k)}
    if edges is None:                                  # auto log-time grid from the node ages
        nt = np.asarray(ts.tables.nodes.time, float)
        assert_calibrated(nt)                          # reject a raw (uncalibrated) tsinfer ARG
        pos = nt[nt > 0]
        edges = log_time_grid(max(1.0, float(pos.min())), float(pos.max()) * 1.05, n_cells)
    if fit_result is not None:                         # warm-start: reuse a precomputed fit
        res = fit_result
    else:
        # Q0=None -> fit() scales the initial generator to the (calibrated) node-age axis, so the
        # warm-start fit does not wash out on a deep time scale (same guard as tspaint.paint). K is
        # taken from the labels so a cold-start K>2 dating works without an explicit Q0.
        K = 1 + max((int(v) for v in labels.values()), default=1)
        res = fit(ts, labels, K=K, Q0=Q0, max_iter=em_init, estimate_pi=estimate_pi,
                  soft_refs=soft_refs, mask=mask)
    emissions = build_emissions(ts, labels, res.w, res.pi, mask)
    pi = res.pi
    centers = cell_centers(edges)
    ncell = len(centers)
    K = int(np.asarray(res.Q).shape[0])

    diag = np.arange(K)
    q = np.zeros((ncell, K, K))                        # per-cell off-diagonal rate array, init from Q
    q[:] = np.maximum(np.asarray(res.Q, float), floor)
    q[:, diag, diag] = 0.0

    from ..parallel import resolve_cores
    nj = resolve_cores(n_jobs)
    use_pool = nj > 1 and ts.num_trees > 1

    history = []
    D = J = None
    with ExitStack() as stack:
        executor = path = None
        if use_pool:                                   # one pool + one ts dump, reused every iteration
            from ..parallel import make_pool, as_path
            executor = make_pool(nj)
            if executor is not None:
                stack.callback(executor.shutdown)
                path = stack.enter_context(as_path(ts))

        def estep(q_now):
            if executor is not None:
                from ..parallel import dating_estep_parallel
                return dating_estep_parallel(ts, q_now, pi, labels, res.w, edges, mask=mask,
                                             path=path, n_jobs=nj, executor=executor)
            return accumulate_time_binned_tv(ts, make_Q_of_cell(q_now), pi, emissions, edges)

        for it in range(n_iter):
            D, J, ll = estep(q)
            history.append(ll)
            sp = directional_rate_splines(D, J, centers, n_knots=n_knots)
            q = np.maximum(np.nan_to_num(np.asarray(sp["q"], float), nan=floor), floor)
            q[:, diag, diag] = 0.0                      # the diagonal is not a rate
            if it > 0 and abs(history[-1] - history[-2]) < tol * (abs(history[-2]) + 1e-12):
                break
    return RateThroughTime(centers, q, D, J, history, states=states)
