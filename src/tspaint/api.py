"""High-level entry point for tspaint (CLAUDE.md §2.4).

:func:`paint` is the one call most users need: fit the ancestry CTMC on the labelled
references and return per-haplotype, per-position ancestry posteriors as a :class:`Painting`.
Everything else (the EM, pruning, sufficient statistics, metrics, comparators, I/O front ends)
is the machinery underneath, available in the submodules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .em import fit, build_emissions
from .model import make_generator_2state
from .output import posterior_table, Segment
from .track import SoftTrack

__all__ = ["paint", "Painting"]


@dataclass
class Painting(SoftTrack):
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

    @property
    def samples(self):
        """The painted query haplotypes — the row order used by :meth:`plot`."""
        return self.queries

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


def paint(ts, labels, queries=None, *, refs=False, K=2, soft_refs=None, estimate_pi=False,
          deadband=0.0, smooth=False, epsilon=1e-2, Q0=None, max_iter=12, tol=1e-7, alpha=20.0,
          beta=1.0, priors=None, w0=0.9, n_jobs=1):
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
    labels : dict
        Reference → ancestry-state index in ``0..K-1``, applied to every member of an ensemble.
        Each key is an integer **sample-node index** or a **sample-ID string**: when ``ts`` came
        from :func:`tspaint.io.singer` / :func:`tspaint.io.tsinfer` the source's sample ids are
        stamped on the nodes, so a base id (e.g. ``"NA12878"``) labels **both** its haplotypes and a
        per-haplotype id (``"NA12878_1"``) labels one (:mod:`tspaint.ids`).
    queries : iterable, optional
        Samples to paint (node indices or sample-ID strings, as for ``labels``); defaults to every
        sample not in ``labels``.
    refs : bool or iterable, optional
        Also paint reference haplotypes, arranged around the queries so the plot is framed by its
        anchors: **group-0 references ("ref1") occupy the first (top) rows and the other groups
        ("ref2", ...) the last (bottom) rows**, with the queries in between. ``True`` includes every
        reference; an iterable includes only the named reference individuals (node indices or
        sample-ID strings, a diploid id expanding to both haplotypes) — a
        :class:`ValueError` is raised if any of them is not a labelled reference. Default ``False``
        (paint queries only). References stay hard-clamped anchors in the fit, so a clamped
        reference paints as a flat bar of its own label colour (use ``soft_refs`` /
        :meth:`Painting.introgression_map` to see a reference dissent from its label).
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

    Label references by **sample-ID string** — no need to know the node order — when the tree
    sequence came from a front end (:func:`tspaint.io.singer` / :func:`tspaint.io.tsinfer` stamp
    the source's ids); a diploid base id labels both its haplotypes:

    >>> ts = tspaint.io.singer(vcf, Ne=1e4, mutation_rate=1e-8, recombination_rate=1e-8)
    >>> painting = tspaint.paint(ts, {"NA12878": 0, "NA12889": 1})

    Paint from a SINGER posterior ensemble — the mean painting carries an uncertainty band:

    >>> ensemble = tspaint.io.singer(vcf, Ne=1e4, mutation_rate=1e-8, recombination_rate=1e-8)
    >>> painting = tspaint.paint(ensemble, labels)            # one pooled fit; averaged posteriors
    >>> painting.posteriors[q][0].posterior_std               # per-position ARG-uncertainty band
    """
    members = list(ts) if isinstance(ts, (list, tuple)) else None    # an ARG ensemble?
    if members is not None and not members:
        raise ValueError("paint() got an empty ensemble; pass at least one tree sequence")
    ref_ts = members[0] if members is not None else ts
    # labels / queries may be keyed by sample-ID string (from io.singer/io.tsinfer's stamped ids)
    # or by integer node index; resolve both to node ids (soft_refs / priors are resolved by fit).
    from .ids import resolve_labels, resolve_ids, resolve_nodes
    labels = resolve_labels(ref_ts, labels)
    if queries is None:
        queries = [int(s) for s in ref_ts.samples() if int(s) not in labels]
    else:
        queries = resolve_ids(ref_ts, queries)

    # Optionally paint the reference haplotypes too, framing the queries: group-0 refs ("ref1") as
    # the first (top) rows, the other reference groups ("ref2", ...) as the last (bottom) rows.
    if refs:
        if refs is True:
            ref_nodes = list(labels)                          # every reference, in label order
        else:
            ref_nodes, absent = [], []
            for item in refs:                                 # an explicit set of reference individuals
                try:
                    nodes = resolve_nodes(ref_ts, item)
                except KeyError:
                    nodes = []
                if nodes and all(n in labels for n in nodes):
                    ref_nodes.extend(nodes)
                else:
                    absent.append(item)
            if absent:
                raise ValueError(
                    f"refs {absent} are not reference individuals (absent from labels); pass only "
                    "labelled references, or refs=True to include them all")
        ref_set = set(ref_nodes)
        top = [n for n in ref_nodes if labels[n] == 0]        # ref1 (state 0) -> first rows
        bottom = [n for n in ref_nodes if labels[n] != 0]     # ref2 (other states) -> bottom rows
        queries = top + [q for q in queries if q not in ref_set] + bottom

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
