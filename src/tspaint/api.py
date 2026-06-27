"""High-level entry point for tspaint (CLAUDE.md §2.4).

:func:`paint` is the one call most users need: fit the ancestry CTMC on the labelled
references and return per-haplotype, per-position ancestry posteriors as a :class:`Painting`.
Everything else (the EM, pruning, sufficient statistics, metrics, comparators, I/O front ends)
is the machinery underneath, available in the submodules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .em import fit, build_emissions
from .model import make_generator_2state
from .output import posterior_table, hard_segments, posterior_at, Segment

__all__ = ["paint", "Painting"]


@dataclass
class Painting:
    """Result of :func:`paint`: the soft local-ancestry posteriors plus the fitted model.

    Attributes
    ----------
    posteriors : dict[int, list[Segment]]
        Per query haplotype, the down-pass posterior over ancestry states as contiguous
        :class:`~tspaint.output.Segment`\\ s covering ``[0, L)`` (the soft, calibrated
        deliverable). When :func:`paint` was given an **ensemble** of tree sequences these are
        :class:`~tspaint.ensemble.MergedSegment`\\ s — the ensemble-mean posterior plus a
        ``posterior_std`` ARG-uncertainty band — but otherwise behave identically (same
        ``left``/``right``/``posterior``/``status``).
    Q : numpy.ndarray
        The fitted generator.
    pi : numpy.ndarray
        The fitted root frequencies.
    w : dict
        The learned per-tip credibility.
    loglik_history : list
        Observed-data log-likelihood per EM step (non-decreasing).
    queries : list
        The painted sample-node ids.
    ts : tskit.TreeSequence or list[tskit.TreeSequence]
        The tree sequence painted — or the ensemble of them — retained so
        :meth:`introgression_map` / :meth:`rate_through_time` can reuse the fit (already in
        memory, so no extra cost).
    labels : dict[int, int]
        The reference labels used for the fit (retained for :meth:`rate_through_time`).
    default_deadband : float
        Default dead-band passed to :meth:`segments`. Default ``0.0``.
    _member_posteriors : list[dict] or None
        For an **ensemble** painting, the individual per-member posterior tables (each a
        ``dict[int, list[Segment]]``) whose per-position mean is :attr:`posteriors`; ``None`` for a
        single tree sequence. Drives the per-member :meth:`rate_through_time` split-time interval.
    """
    posteriors: dict
    Q: np.ndarray
    pi: np.ndarray
    w: dict
    loglik_history: list
    queries: list
    ts: object = None
    labels: dict = None
    default_deadband: float = 0.0
    n_jobs: int = 1          # worker processes used to paint; the default for rate_through_time
    _seqlen: float = None    # sequence length carried by a reloaded painting (ts is None then)
    _member_posteriors: list = None    # per-member posterior tables for an ensemble (else None)

    @property
    def length(self):
        """Sequence length of the painted genome (the first member, for an ensemble)."""
        if self.ts is None:
            return self._seqlen
        t = self.ts[0] if isinstance(self.ts, (list, tuple)) else self.ts
        return float(t.sequence_length)

    def save(self, path):
        """Write this painting to ``path`` as ``.npz`` (the segment table plus the fitted model).

        Reload with :meth:`load`. The painted tree sequence itself is **not** stored (keep the
        ``.trees`` file); a reloaded painting therefore has ``ts=None`` but retains
        :attr:`length`, the posteriors, ``Q``/``π``/``w`` and the labels.
        """
        from .serialize import save_painting
        save_painting(path, self.posteriors, Q=self.Q, pi=self.pi, w=self.w,
                      queries=self.queries, labels=self.labels, seqlen=self.length,
                      deadband=self.default_deadband)

    @staticmethod
    def load(path):
        """Reload a painting written by :meth:`save` (``ts`` is ``None``; see :meth:`save`)."""
        from .serialize import load_painting, load_painting_meta
        tracks = load_painting(path)
        m = load_painting_meta(path)
        return Painting(posteriors=tracks, Q=m.get("Q"), pi=m.get("pi"), w=m.get("w", {}),
                        loglik_history=[], queries=m.get("queries", []), ts=None,
                        labels=m.get("labels"), default_deadband=m.get("deadband", 0.0) or 0.0,
                        _seqlen=m.get("seqlen"))

    def segments(self, deadband=None):
        """Hard ancestry segments for downstream tract-length / dating analysis.

        Parameters
        ----------
        deadband : float, optional
            Confidence dead-band suppressing low-confidence flips that fragment long
            tracts. Defaults to :attr:`default_deadband`. See
            :func:`tspaint.output.hard_segments`.

        Returns
        -------
        dict[int, list[tuple[float, float, int]]]
            Per query, hard ``(left, right, state)`` segments.
        """
        db = self.default_deadband if deadband is None else deadband
        return {q: hard_segments(t, db) for q, t in self.posteriors.items()}

    def posterior_at(self, sample, position):
        """Posterior vector for ``sample`` at a genomic ``position``.

        Parameters
        ----------
        sample : int
            Sample-node id to look up.
        position : float
            Genomic position.

        Returns
        -------
        numpy.ndarray or None
            The ``(K,)`` posterior covering ``position``, or ``None`` if uncovered.
        """
        return posterior_at(self.posteriors, sample, position)

    def rate_through_time(self, edges=None, *, n_jobs=None, **kwargs):
        """Estimate the admixture (cross-ancestry) rate through time, reusing this fit.

        Fits the time-inhomogeneous directional mugration EM
        (:func:`tspaint.fit_rate_through_time`) **warm-started from this painting's fitted
        ``(Q, π, w)``**, so the homogeneous fit is not repeated. This is a *different
        deliverable* from the painting — the cross-ancestry transition rates ``q_AB(t)``,
        ``q_BA(t)`` as functions of (backward) time, locating divergence / gene-flow epochs and
        their direction. It returns a **new** :class:`~tspaint.dating.RateThroughTime` and does
        **not** modify :attr:`posteriors` (CLAUDE.md: Q(t) gives no painting-accuracy gain, so the
        paths stay side by side).

        Parameters
        ----------
        edges : array_like, optional
            Log-time grid edges; an auto grid is built from the node ages when ``None``.
        n_jobs : int, optional
            For an **ensemble** painting, worker processes dating the members **in parallel**
            (one per member — they are independent). Defaults to the painting's own
            :attr:`n_jobs` (so a painting fit in parallel dates in parallel); pass an explicit
            value to override. Ignored for a single tree sequence (the dating E-step is not yet
            tree-parallel).
        **kwargs
            Forwarded to :func:`tspaint.fit_rate_through_time` (e.g. ``n_cells``, ``n_iter``,
            ``n_knots``).

        Returns
        -------
        tspaint.dating.RateThroughTime or tspaint.dating.EnsembleRateThroughTime
            For a single tree sequence, the directional rate-through-time profile (``.centers``,
            ``.q_AB``, ``.q_BA``, ``.plot()``). For an **ensemble** painting, an
            :class:`~tspaint.dating.EnsembleRateThroughTime`: every member is dated on the shared
            fit and grid, and the per-member split-time estimates form a **confidence interval on
            the split time** (``.split_time()``, ``.split_time_ci()``) — the ARG-uncertainty band
            on *when* the ancestries diverged.
        """
        if self.ts is None or self.labels is None:
            raise ValueError("Painting was constructed without ts/labels; cannot date. Use "
                             "tspaint.fit_rate_through_time(ts, labels) directly.")
        from .em import FitResult
        from .dating import fit_rate_through_time
        warm = FitResult(self.Q, self.pi, self.w, self.loglik_history)
        if isinstance(self.ts, (list, tuple)):
            # ensemble: date each member on the shared fit/grid (in parallel across members);
            # the per-member split-time estimates become a confidence interval on the split.
            from .dating import log_time_grid, split_time, EnsembleRateThroughTime
            from .parallel import date_members_parallel
            members = list(self.ts)
            if edges is None:                       # one shared grid so the member profiles align
                nt = np.concatenate([np.asarray(g.tables.nodes.time, float) for g in members])
                pos = nt[nt > 0]
                n_cells = kwargs.pop("n_cells", 40)
                edges = log_time_grid(max(1.0, float(pos.min())), float(pos.max()) * 1.05, n_cells)
            nj = self.n_jobs if n_jobs is None else n_jobs
            rtts = date_members_parallel(members, self.labels, warm, edges, kwargs, n_jobs=nj)
            return EnsembleRateThroughTime(rtts, np.array([split_time(r) for r in rtts], float))
        return fit_rate_through_time(self.ts, self.labels, edges, fit_result=warm, **kwargs)

    def introgression_map(self, sample):
        """Leave-one-out introgression map for ``sample``, reusing this painting's fit.

        Returns what the rest of the genealogy says about ``sample`` *excluding its own
        emission* (:func:`tspaint.output.loo_posterior_table`). Unlike :attr:`posteriors` (the
        down-pass), it is not suppressed by a confident tip emission, so it surfaces a labelled
        reference's own foreign tracts — the reference-introgression / mislabel lens
        (CLAUDE.md §2.3, §9). For a panel-wide audit see :func:`tspaint.reference_qc`.

        Parameters
        ----------
        sample : int
            Sample-node id to map (a labelled reference or a query).

        Returns
        -------
        list[tspaint.output.Segment]
            The leave-one-out posterior as contiguous segments covering ``[0, L)`` (for an
            ensemble painting, the ensemble-mean leave-one-out map as
            :class:`~tspaint.ensemble.MergedSegment`\\ s).
        """
        if self.ts is None or self.labels is None:
            raise ValueError("Painting was constructed without ts/labels; cannot map "
                             "introgression. Use tspaint.output.loo_posterior_table directly.")
        from .output import loo_posterior_table
        sample = int(sample)
        if isinstance(self.ts, (list, tuple)):
            from .ensemble import merge_posterior_tables
            tables = [loo_posterior_table(g, self.Q, self.pi,
                                          build_emissions(g, self.labels, self.w, self.pi),
                                          focal=[sample]) for g in self.ts]
            return merge_posterior_tables(tables, samples=[sample])[sample]
        emissions = build_emissions(self.ts, self.labels, self.w, self.pi)
        return loo_posterior_table(self.ts, self.Q, self.pi, emissions, focal=[sample])[sample]

    def __repr__(self):
        return (f"Painting(queries={len(self.queries)}, K={self.pi.shape[0]}, "
                f"Q={np.array2string(self.Q, precision=2)}, "
                f"pi={np.array2string(self.pi, precision=2)})")

    def plot(self, truth=None, title=None):
        
        qs = self.queries
        segments = self.segments(deadband=0.4)

        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap='coolwarm')
        fig = plt.figure(figsize=(9, 0.3 * len(qs) + 1))
        gs = fig.add_gridspec(len(qs), 2, width_ratios=[1,0.03], hspace=0)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(len(qs))]

        for i, q in enumerate(qs):
            ymin, ymax = 0, 1.5
            if truth:
                ymin = -0.5
                for (l, r, s) in truth[q]:
                    axes[i].barh(-0.25, r - l, left=l, height=0.5,
                            color=sm.to_rgba(1.0 if s == 0 else 0.0), edgecolor="none")
            for (l, r, s) in segments[q]:
                axes[i].barh(0.25, r - l, left=l, height=0.5,
                        color=sm.to_rgba(1.0 if s == 0 else 0.0), edgecolor="none")
            for seg in self.posteriors[q]:
                axes[i].barh(1, seg.right - seg.left, left=seg.left, height=1,
                        color=sm.to_rgba(seg.posterior[0]), edgecolor="none")
            # for state in [0, 1]:
            #     xs, ys = [], []
            #     for seg in self.posteriors[q]:
            #         xs += [seg.left, seg.right]
            #         ys += [float(seg.posterior[state])] * 2
            #     axes[i].plot(xs, ys, color=sm.to_rgba(1.0 if state == 0 else 0.0))

            axes[i].set_ylim(ymin, ymax)
            axes[i].set_xlim(0, self.length)
            axes[i].set_ylabel(f'hapl. {i}', rotation=0, fontsize=7, color="0.0", horizontalalignment="right")
            axes[i].yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            if i < len(axes)-1:
                axes[i].xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            axes[i].tick_params(axis='x', bottom=True)
            axes[i].spines['top'].set_visible(True)
            axes[i].spines['bottom'].set_visible(True)
            axes[i].spines['left'].set_visible(True)
            axes[i].spines['right'].set_visible(True)
            axes[i].grid(False)

        ax = fig.add_subplot(gs[:, 1])
        ax.set_axis_off()
        cb = fig.colorbar(sm, ax=ax, fraction=0.5, pad=0.01)
        cb.set_label("P(ancestry A)")
        if title:
            axes[0].set_title(title)
        plt.tight_layout()

def paint(ts, labels, queries=None, *, K=2, soft_refs=None, estimate_pi=False, deadband=0.0,
          smooth=False, epsilon=1e-2, Q0=None, max_iter=12, tol=1e-7, alpha=20.0, beta=1.0,
          priors=None, w0=0.9, n_jobs=1):
    """Infer soft local ancestry along query haplotypes from a tree sequence.

    EM-fits the ancestry CTMC ``(Q[, π, per-tip credibility w])`` on the labelled reference tips
    (:func:`tspaint.fit`), then returns the per-position posterior over ancestry states for each
    query haplotype as a :class:`Painting`.

    Parameters
    ----------
    ts : tskit.TreeSequence or list[tskit.TreeSequence]
        An inferred (tsinfer / Relate ``--compress``) or true tree sequence; sample nodes are
        haplotypes. Use :mod:`tspaint.io` to obtain one from genotypes. **Pass a list of tree
        sequences** — e.g. the posterior ARG ensemble from :func:`tspaint.io.singer` — to paint
        from the ensemble: one ``(Q, π, w)`` is fit pooled across all members (the M-step is
        scale-invariant), each member is painted with it, and the per-position posteriors are
        **averaged**. This marginalises ARG uncertainty — the binding constraint on real data —
        and the spread becomes a calibrated uncertainty band (CLAUDE.md §7.4). All members must
        share the same sample ids (true of a SINGER ensemble, where sample order is preserved).
    labels : dict[int, int]
        Reference sample-node id → ancestry-state index in ``0..K-1``. Applied to every member
        of an ensemble.
    queries : iterable[int], optional
        Sample nodes to paint; defaults to every sample not in ``labels``.
    K : int
        Number of ancestry states (2 by default; pass a ``K×K`` ``Q0`` for K-way).
    soft_refs : set[int], optional
        Reference tips whose credibility ``w_i`` is *learned* (the rest stay hard-clamped
        anchors — never let the whole panel float, CLAUDE.md §6). Softening a slightly
        impure reference (rather than hard-clamping it) lets the genealogy override its
        label over its foreign tracts — both painting queries there correctly and mapping
        the reference's own introgression (CLAUDE.md §2.3).
    estimate_pi : bool
        Estimate root frequencies ``π`` rather than holding them uniform. Default ``False``:
        ``π`` is a prior on the arbitrary GMRCA state and estimating it from washing deep
        branches is the degeneracy of CLAUDE.md §6; uniform is the robust choice.
    deadband : float
        Stored as :attr:`Painting.default_deadband` for :meth:`Painting.segments`.
    smooth : bool
        Apply the horizontal BP/EP smoother (:mod:`tspaint.bp`) to the posteriors along the
        genome. **Recommended on inferred (tsinfer / Relate) ARGs**, where tree inference
        scatters spurious breakpoints a per-position deadband cannot filter; redundant on a
        true/known ARG (CLAUDE.md §7). Default ``False``.
    epsilon : float
        Per-breakpoint switch penalty for ``smooth`` (smaller ⇒ more smoothing).
    Q0 : (K, K) array, optional
        Initial generator (default a slow symmetric 2-state generator).
    priors : dict[int, tuple[float, float]], optional
        Per-tip ``Beta(alpha_i, beta_i)`` prior overrides for the graded-trust setting
        (keys ⊆ ``soft_refs``); see :func:`tspaint.fit`.
    max_iter, tol, alpha, beta, w0 : EM controls (see :func:`tspaint.fit`).
    n_jobs : int
        Worker processes for the genome E-step and painting. ``1`` (default) is serial and
        byte-identical to single-core; ``>1`` saturates the node via a process pool
        (:mod:`tspaint.parallel`) — the painting is *exactly* equal to serial, the fit
        ``allclose`` (floating-point reduction order). The CLI resolves this from
        ``$SLURM_JOB_CPUS_PER_NODE``.

    Returns
    -------
    Painting
        The soft per-position ancestry posteriors for each query plus the fitted
        ``(Q, π, w)`` and EM log-likelihood history. For an ensemble input the posteriors are
        the ensemble mean with an ARG-uncertainty band (:class:`~tspaint.ensemble.MergedSegment`,
        with a ``posterior_std``).

    See Also
    --------
    tspaint.fit : The underlying blocked-EM fit.
    tspaint.output.hard_segments : Collapse the soft posteriors into hard tracts.
    tspaint.ensemble.merge_posterior_tables : The per-position averaging used for an ensemble.

    Examples
    --------
    >>> import tspaint
    >>> ts = tspaint.simulate_admixture(n_admix=10, n_ref=10)
    >>> labels = {0: 0, 1: 0, 2: 1, 3: 1}   # reference sample-node -> ancestry state
    >>> painting = tspaint.paint(ts, labels)
    >>> painting.segments(deadband=0.4)      # hard ancestry tracts for dating

    Paint from a SINGER posterior ensemble — the mean painting carries an uncertainty band:

    >>> ensemble = tspaint.io.singer(vcf, Ne=1e4, mutation_rate=1e-8, recombination_rate=1e-8)
    >>> painting = tspaint.paint(ensemble, labels)            # one pooled fit; averaged posteriors
    >>> painting.posteriors[q][0].posterior_std               # per-position ARG-uncertainty band
    """
    labels = {int(k): int(v) for k, v in labels.items()}
    members = list(ts) if isinstance(ts, (list, tuple)) else None    # an ARG ensemble?
    if members is not None and not members:
        raise ValueError("paint() got an empty ensemble; pass at least one tree sequence")
    ref_ts = members[0] if members is not None else ts
    if queries is None:
        queries = [int(s) for s in ref_ts.samples() if int(s) not in labels]
    else:
        queries = [int(q) for q in queries]
    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)

    # One pooled θ fit shared across the ensemble (the M-step is scale-invariant).
    res = fit(members if members is not None else ts,
              [labels] * len(members) if members is not None else labels,
              K=K, Q0=Q0, max_iter=max_iter, tol=tol, soft_refs=soft_refs,
              estimate_pi=estimate_pi, alpha=alpha, beta=beta, priors=priors, w0=w0,
              n_jobs=n_jobs)

    def _paint_member(g):
        if n_jobs and int(n_jobs) > 1:
            from .parallel import posterior_table_parallel
            table = posterior_table_parallel(g, res.Q, res.pi, w=res.w, labels=labels,
                                             focal=queries, n_jobs=n_jobs)
        else:
            emissions = build_emissions(g, labels, res.w, res.pi)
            table = posterior_table(g, res.Q, res.pi, emissions, focal=queries)
        if smooth:
            from .bp import bp_smooth_track
            table = {q: bp_smooth_track(t, res.pi, epsilon) for q, t in table.items()}
        return table

    member_tables = None
    if members is None:
        posteriors = _paint_member(ts)
    else:
        from .ensemble import merge_posterior_tables
        member_tables = [_paint_member(g) for g in members]      # kept for the per-member CI
        posteriors = merge_posterior_tables(member_tables, samples=queries)

    return Painting(posteriors=posteriors, Q=res.Q, pi=res.pi, w=res.w,
                    loglik_history=res.loglik_history, queries=queries,
                    ts=ts, labels=labels, default_deadband=deadband, n_jobs=n_jobs,
                    _member_posteriors=member_tables)
