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
from .output import posterior_table, Segment, DEFAULT_DEADBAND
from .track import SoftTrack

__all__ = ["paint", "Painting", "WindowedPainting"]

#: Hatch for a reference's **masked** (unlabelled) spans in :meth:`Painting.plot` — a cross-hatch,
#: visually distinct from the ghost overlay (:data:`_GHOST_HATCH`) so the two read apart on one plot.
_MASK_HATCH = "xxx"
#: Hatch for **ghost** tracts overlaid on the ancestry painting in :meth:`Painting.plot` (diagonal).
_GHOST_HATCH = "///"


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
        Default dead-band passed to :meth:`segments` (and the plots) — the confidence dead-band on
        the top-two posterior margin ``max(P) − 2nd-max(P)`` (:func:`tspaint.output.hard_segments`).
        Default :data:`~tspaint.output.DEFAULT_DEADBAND` (``0.4``).
    n_jobs : int or None
        Worker processes used to paint (resolved from :func:`paint`'s ``n_jobs`` via
        :func:`tspaint.parallel.resolve_cores`), reused as the default for
        :meth:`rate_through_time`'s per-member ensemble dating. ``None`` on a painting not produced
        by :func:`paint` (e.g. reloaded with :meth:`load` or reassembled from windows).
    _member_posteriors : list[dict] or None
        For an **ensemble** painting, the individual per-member posterior tables (each a
        ``dict[int, list[Segment]]``) whose per-position mean is :attr:`posteriors`; ``None`` for a
        single tree sequence. Drives the per-member :meth:`rate_through_time` split-time interval.
    mask : dict or None
        The fragment mask applied when painting — ``{ref-node: [(left, right), ...]}`` spans
        treated as unlabelled (CLAUDE.md §2.3), resolved to node-id keys; ``None`` when unmasked.
        :meth:`plot` reads it to cross-hatch each reference's masked spans (``mark_masked``).
    """
    posteriors: dict
    Q: np.ndarray
    pi: np.ndarray
    w: dict
    loglik_history: list
    queries: list
    ts: object = None
    labels: dict = None
    default_deadband: float = DEFAULT_DEADBAND
    n_jobs: int = None       # workers used to paint; the default for rate_through_time (None -> all CPUs)
    _seqlen: float = None    # sequence length carried by a reloaded painting (ts is None then)
    _member_posteriors: list = None    # per-member posterior tables for an ensemble (else None)
    mask: dict = None        # fragment mask used to paint ({ref: [(l,r)]}); marks masked ref spans in plot()

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

    def plot(self, truth=None, segments=False, title=None, cmap='coolwarm', colors=None, alpha=None, figsize=None, return_plot=False,
             deadband=None, row_labels=None, mark_spans=None, samples=None, curves=False, *, refs=True, mark_masked=True,
             ghost=None, ghost_threshold=0.5):
        """Strip plot of the painting — **reference-aware** (extends :meth:`SoftTrack.plot`).

        **To display reference haplotypes you must paint them** — ``refs`` here does **not** add rows,
        it only annotates references that are already in the painted focal set. Include them at paint
        time: ``paint(ts, labels, refs=True)`` (all references, arranged around the queries) or
        ``paint(ts, labels, queries=queries + [ref], mask=mask)`` (specific ones). Then, for any painted
        row that is a labelled reference (in :attr:`labels`), ``refs`` (default ``True``):

        * labels the row with its **nominal ancestry** — e.g. ``ref 39 (A)`` instead of ``hapl. i`` —
          so a reference that paints a foreign (e.g. B) tract stands out against its own A label; and
        * when the painting was produced with a fragment ``mask``, **hatches each reference's masked
          (unlabelled) spans** over its soft band, showing where the reference was un-anchored and its
          posterior became tree-inferred (the mechanism that reveals its introgression).

        Pass ``refs=False`` / ``mark_masked=False`` to suppress either. A reference's masked spans are
        drawn with a **cross-hatch** (:data:`_MASK_HATCH`), distinct from the ghost overlay below.

        ``samples`` selects which haplotypes to plot (defaults to all queries) and accepts the same
        forms as the former :meth:`plot_posterior`: an ``int`` / list of ``int`` (haploid sample-node
        indices) or a ``str`` / list of ``str`` (individual / sample ids, a diploid id expanding to
        both haplotype rows), resolved via :meth:`_resolve_plot_samples`. Rows are then labelled by the
        resolved id / node unless overridden by ``row_labels``.

        Pass ``ghost=`` a :class:`~tspaint.GhostResult` (from :func:`tspaint.detect_ghost`) to **hatch
        each haplotype's ghost tracts over its ancestry bands** — the ghost search (where the ancestry
        is foreign to the whole panel) drawn on top of the source painting with a **diagonal** hatch
        (:data:`_GHOST_HATCH`, distinct from a masked span's cross-hatch). The tracts are
        :meth:`GhostResult.tracts` at ``ghost_threshold`` (default ``P(ghost) ≥ 0.5``); only haplotypes
        the ghost result also covers are annotated.

        ``curves=True`` draws each row's posterior as **line curves** instead of the colour band — one
        line per ancestry state in its segment hue, ``y = P(state)`` over the genome — so
        ``painting.plot(samples=[0], curves=True)`` plots the ``K`` posterior curves for the first query.
        ``truth`` and — with ``segments=True`` — the hard segments then show as thin state-coloured bands
        below the curves (each ~10 % of the curve height). With ``ghost=`` in this mode the ghost
        ``P(ghost)`` posterior is added as a separate black curve rather than a hatch.

        When ``title`` is left ``None`` it defaults to a compact painting summary — total ancestry
        proportions and fragmentation, plus breakpoint precision / recall when ``truth`` is given
        (:meth:`summary`). Pass an explicit ``title`` (including ``""``) to override. All other
        arguments (``cmap``, ``colors``, ``deadband``, ``return_plot`` …) behave as in
        :meth:`SoftTrack.plot`.

        Parameters
        ----------
        truth : dict[int, list[tuple[float, float, int]]], optional
            Ground-truth tracts ``(left, right, state)`` per haplotype, drawn as a reference track
            below each row. Default ``None`` (no truth track).
        segments : bool, optional
            Also draw the hard segments (:meth:`segments` at ``deadband``) as a band per row.
            Default ``False``.
        title : str, optional
            Title for the top row. Default ``None`` → the compact performance-stats summary
            (:meth:`summary`); pass ``""`` for no title, or any string to override.
        cmap : optional
            Unused (kept for backward compatibility); the confidence-as-opacity rendering takes the
            per-state hues from ``colors``. Default ``'coolwarm'``.
        colors : list, optional
            Per-state hues (state ``k`` → ``colors[k]``; the two extremes are the state hues for
            ``K = 2``). Default ``None`` → matplotlib's colour cycle (see :meth:`SoftTrack.plot`).
        alpha : float, optional
            Maximum opacity — fades the whole plot; the soft band is drawn at
            ``confidence · alpha``. Default ``None`` (fully opaque, i.e. ``1``).
        figsize : (float, float), optional
            Figure size. Default ``None`` (auto from the number of rows).
        return_plot : bool, optional
            Return the matplotlib ``(figure, axes)`` instead of ``None``. Default ``False``.
        deadband : float, optional
            Dead-band on the top-two posterior margin ``max(P) − 2nd-max(P)`` for the hard-segment
            overlay (:func:`tspaint.output.hard_segments`). Default ``None`` →
            :attr:`default_deadband`.
        row_labels : dict[int, str], optional
            Per-sample y-axis label overrides (**highest priority**). Painted references are
            otherwise auto-labelled ``ref <node> (<ancestry>)`` when ``refs=True``, and ids resolved
            from ``samples`` supply their own labels. Default ``None``.
        mark_spans : dict[int, list[tuple[float, float]]], optional
            Extra per-sample spans to hatch over each row; the reference-mask and ghost overlays
            below are merged into it. Default ``None``.
        samples : int or str or iterable, optional
            Which painted haplotypes to plot, in order: an ``int`` / list of ``int`` (haploid
            sample-node indices) or a ``str`` / list of ``str`` (individual / sample ids, a diploid
            id expanding to both haplotype rows), resolved via :meth:`_resolve_plot_samples`. A
            :class:`ValueError` is raised if the selection is empty or includes an unpainted
            haplotype. Default ``None`` (all queries).
        curves : bool, optional
            Draw each row's posterior as line curves (one per ancestry state) instead of the colour
            band. Default ``False``.
        refs : bool, optional
            For painted rows that are labelled references (in :attr:`labels`): label the row with
            its nominal ancestry (``ref <node> (A)``) and, when a ``mask`` was used, hatch its
            masked spans. Keyword-only. Default ``True``.
        mark_masked : bool, optional
            Cross-hatch (:data:`_MASK_HATCH`) each reference's masked (unlabelled) spans when this
            painting was produced with a fragment ``mask``. Keyword-only. Default ``True``.
        ghost : tspaint.GhostResult, optional
            A ghost-detector result (:func:`tspaint.detect_ghost`); its ghost tracts are hatched
            over each haplotype's ancestry bands (diagonal, :data:`_GHOST_HATCH`) — or, with
            ``curves=True``, added as a black ``P(ghost)`` curve. Keyword-only. Default ``None``.
        ghost_threshold : float, optional
            Threshold for the ``ghost`` tracts — a locus is ghost where
            ``P(ghost) ≥ ghost_threshold`` (:meth:`GhostResult.tracts`). Keyword-only.
            Default ``0.5``.

        Returns
        -------
        tuple or None
            ``(figure, list_of_axes)`` if ``return_plot`` is ``True``, else ``None``.
        """
        resolved = {}
        if samples is not None:                         # resolve ids / nodes (folds in plot_posterior)
            nodes, resolved = self._resolve_plot_samples(samples)
            if not nodes:
                raise ValueError("plot: `samples` selected no haplotypes")
            absent = [n for n in nodes if n not in self.posteriors]
            if absent:
                raise ValueError(
                    f"haplotypes {absent} were not painted (not among the painted samples "
                    f"{list(self.samples)}); include them via paint(..., queries=...) or refs=...")
            samples = nodes

        # row labels — priority: caller row_labels > reference label > resolved id / node name
        row_labels = dict(row_labels or {})
        if refs and self.labels:
            for q in self.samples:
                if q in self.labels:
                    row_labels.setdefault(q, f"ref {q} ({chr(65 + int(self.labels[q]))})")
        for n, lbl in resolved.items():
            row_labels.setdefault(n, lbl)

        if mark_masked and self.mask:                   # masked ref spans -> cross-hatch
            mark_spans = dict(mark_spans or {})
            for q, spans in self.mask.items():
                if q in self.samples:
                    mark_spans.setdefault(q, [(float(s[0]), float(s[1]), _MASK_HATCH) for s in spans])
        curve_overlays = None
        if ghost is not None:
            gpost = getattr(ghost, "posteriors", None)
            if gpost is None or not hasattr(ghost, "tracts"):
                raise TypeError("ghost= expects a GhostResult from tspaint.detect_ghost")
            if curves:                                  # ghost P(ghost) -> black dashed curve
                gstate = int(getattr(ghost, "_hi_state", 1))
                rows = samples if samples is not None else list(self.samples)
                curve_overlays = {q: [(gpost[q], gstate, "black", "solid", "Ghost")]
                                  for q in rows if q in gpost}
            else:                                       # ghost tracts -> diagonal hatch (distinct from mask)
                mark_spans = dict(mark_spans or {})
                for q in self.samples:
                    if q in gpost:
                        gt = [(float(a), float(b), _GHOST_HATCH)
                              for (a, b) in ghost.tracts(q, threshold=ghost_threshold)]
                        if gt:
                            mark_spans[q] = list(mark_spans.get(q, [])) + gt
        # title=None is defaulted to the performance-stats summary by SoftTrack.plot (shared with
        # SegmentTrack); the reference-aware row_labels / mark_spans above are the painting-only part.
        return super().plot(truth=truth, segments=segments, title=title, cmap=cmap, colors=colors, alpha=alpha,
                            figsize=figsize, return_plot=return_plot, deadband=deadband, row_labels=row_labels,
                            mark_spans=mark_spans, samples=samples, curves=curves, curve_overlays=curve_overlays)

    def _resolve_plot_samples(self, individuals):
        """Resolve ``individuals`` to ``(ordered node list, {node: row-label})``.

        ``int`` (or list of ``int``) are haploid **sample-node indices**; ``str`` (or list of ``str``)
        are **individual / sample ids** resolved against the painted tree sequence's stamped ids
        (:func:`tspaint.ids.resolve_nodes`) — a diploid id expanding to *both* haplotype nodes. Order
        is preserved and duplicates removed.
        """
        items = [individuals] if isinstance(individuals, (int, np.integer, str)) else list(individuals)
        ref_ts = self.ts[0] if isinstance(self.ts, (list, tuple)) else self.ts
        nodes, labels = [], {}
        for it in items:
            if isinstance(it, (int, np.integer)):
                n = int(it)
                nodes.append(n)
                labels.setdefault(n, f"hapl. {n}")
            else:                                            # an individual / sample id string
                if ref_ts is None:
                    raise ValueError(
                        f"cannot resolve id {it!r}: this painting has no tree sequence (ts=None, e.g. "
                        "reloaded from disk). Pass integer sample-node indices instead.")
                from .ids import resolve_nodes
                hap_nodes = resolve_nodes(ref_ts, it)
                for h, n in enumerate(hap_nodes):
                    nodes.append(n)
                    labels[n] = it if len(hap_nodes) == 1 else f"{it} (hap {h})"
        seen, uniq = set(), []
        for n in nodes:                                      # de-duplicate, preserving order
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq, labels

    def plot_posterior(self, individuals, **kwargs):
        """Deprecated alias for ``plot(samples=individuals, ...)``.

        The sample selection / id resolution it provided now lives directly in :meth:`plot` (its
        ``samples`` argument), so ``painting.plot(samples=[0, 3])`` or
        ``painting.plot(samples="NA12878")`` is the preferred call. Kept for backward compatibility.

        Parameters
        ----------
        individuals : int or str or iterable
            The haplotype(s) to plot — forwarded to :meth:`plot` as ``samples`` (see there): an
            ``int`` / list of ``int`` (haploid sample-node indices) or a ``str`` / list of ``str``
            (individual / sample ids, a diploid id expanding to both haplotypes).
        **kwargs
            Forwarded verbatim to :meth:`plot` (``truth``, ``deadband``, ``return_plot``, …).

        Returns
        -------
        tuple or None
            Whatever :meth:`plot` returns — ``(figure, axes)`` when ``return_plot=True``, else
            ``None``.

        Examples
        --------
        >>> painting.plot(samples=0)                          # one haploid node
        >>> painting.plot(samples=[0, 3, 5])                  # several haploid nodes
        >>> painting.plot(samples="NA12878")                 # both haplotypes of a diploid id
        >>> painting.plot(samples=["NA12878", "NA12889"], truth=truth)
        """
        return self.plot(samples=individuals, **kwargs)

    def save(self, path):
        """Write this painting to ``path`` as ``.npz`` (the segment table plus the fitted model).

        Reload with :meth:`load`. The painted tree sequence itself is **not** stored (keep the
        ``.trees`` file); a reloaded painting therefore has ``ts=None`` but retains
        :attr:`length`, the posteriors, ``Q``/``π``/``w`` and the labels.

        Parameters
        ----------
        path : str
            Destination ``.npz`` path (written exactly as given, no extension appended). Delegates
            to :func:`tspaint.serialize.save_painting` (format ``"tspaint-painting"``: the flat
            per-segment table plus the fitted-model metadata
            ``Q``/``π``/``w``/``queries``/``labels``/``seqlen``/``deadband``).
        """
        from .serialize import save_painting
        save_painting(path, self.posteriors, Q=self.Q, pi=self.pi, w=self.w,
                      queries=self.queries, labels=self.labels, seqlen=self.length,
                      deadband=self.default_deadband)

    @staticmethod
    def load(path):
        """Reload a painting written by :meth:`save` (``ts`` is ``None``; see :meth:`save`).

        Parameters
        ----------
        path : str
            Path to a ``.npz`` written by :meth:`save` (format ``"tspaint-painting"``).

        Returns
        -------
        Painting
            The reloaded painting: :attr:`posteriors`, ``Q``/``π``/``w``, ``queries``, ``labels``
            and :attr:`default_deadband` are restored from the file, while ``ts`` is ``None`` and
            ``loglik_history`` is empty (neither is persisted). Delegates to
            :func:`tspaint.serialize.load_painting` and
            :func:`~tspaint.serialize.load_painting_meta`.
        """
        from .serialize import load_painting, load_painting_meta
        tracks = load_painting(path)
        m = load_painting_meta(path)
        return Painting(posteriors=tracks, Q=m.get("Q"), pi=m.get("pi"), w=m.get("w", {}),
                        loglik_history=[], queries=m.get("queries", []), ts=None,
                        labels=m.get("labels"),
                        default_deadband=(DEFAULT_DEADBAND if m.get("deadband") is None
                                          else float(m["deadband"])),
                        _seqlen=m.get("seqlen"))

    def rate_through_time(self, edges=None, *, n_jobs=None, **kwargs):
        """Estimate the admixture (cross-ancestry) rate through time, reusing this fit.

        Fits the time-inhomogeneous directional mugration EM
        (:func:`tspaint.fit_rate_through_time`) **warm-started from this painting's fitted
        ``(Q, π, w)``**, so the homogeneous fit is not repeated. This is a *different
        deliverable* from the painting — the cross-ancestry transition rates as functions of
        (backward) time, locating divergence / gene-flow epochs and their direction. Works for any
        ``K``: the result carries all ``K·(K-1)`` directional rates ``q[:, m, n]`` (``.q_AB`` /
        ``.q_BA`` are the ``K=2`` slices), and ``.plot()`` draws each ancestry pair in one colour
        (solid / dashed for the two directions). It returns a **new**
        :class:`~tspaint.dating.RateThroughTime` and does **not** modify :attr:`posteriors`
        (CLAUDE.md: Q(t) gives no painting-accuracy gain, so the paths stay side by side).

        Parameters
        ----------
        edges : array_like, optional
            Log-time grid edges; an auto grid is built from the node ages when ``None``.
        n_jobs : int, optional
            Worker processes. For a **single** tree sequence the dating E-step is split across
            genome tree-ranges (:func:`tspaint.parallel.dating_estep_parallel`); for an **ensemble**
            the members are dated in parallel (one worker per member — they are independent).
            Defaults to the painting's own :attr:`n_jobs` (so a painting fit in parallel dates in
            parallel); pass an explicit value to override.
        **kwargs
            Forwarded to :func:`tspaint.fit_rate_through_time` (e.g. ``n_cells``, ``n_iter``,
            ``n_knots``). When this painting was produced with a fragment ``mask``, that mask is
            forwarded too, so the dating E-step uses the same emissions the painting did; pass an
            explicit ``mask=`` (or ``mask=None``) to override.

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
        if self.mask and "mask" not in kwargs:      # date under the same emissions we painted with
            kwargs["mask"] = self.mask
        nj = self.n_jobs if n_jobs is None else n_jobs
        if isinstance(self.ts, (list, tuple)):
            # ensemble: date each member on the shared fit/grid (in parallel across members);
            # the per-member split-time estimates become a confidence interval on the split.
            from .dating import log_time_grid, EnsembleRateThroughTime
            from .dating.grid import assert_calibrated
            from .parallel import date_members_parallel
            members = list(self.ts)
            if edges is None:                       # one shared grid so the member profiles align
                nt = np.concatenate([np.asarray(g.tables.nodes.time, float) for g in members])
                assert_calibrated(nt)               # reject uncalibrated (raw tsinfer) members
                pos = nt[nt > 0]
                n_cells = kwargs.pop("n_cells", 40)
                edges = log_time_grid(max(1.0, float(pos.min())), float(pos.max()) * 1.05, n_cells)
            rtts = date_members_parallel(members, self.labels, warm, edges, kwargs, n_jobs=nj)
            return EnsembleRateThroughTime.from_members(rtts)
        return fit_rate_through_time(self.ts, self.labels, edges, fit_result=warm, n_jobs=nj,
                                     **kwargs)

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


def _stitch_window_tables(tables, bounds, queries):
    """Concatenate per-window posterior tables into one genome-wide table per query.

    Each window was painted over the full ``[0, L)`` with its genealogy confined to its own
    ``[lo, hi)`` (the flanks paint as missing-info); we keep each segment's slice inside ``[lo, hi)``
    and lay the windows end to end. Segments are clipped at the window bounds, so the stitched track
    tiles ``[0, L)`` exactly — the same result as painting the whole tree sequence at once.
    """
    stitched = {q: [] for q in queries}
    for table, (lo, hi) in zip(tables, bounds):
        for q in queries:
            for s in table.get(q, []):
                left, right = max(s.left, lo), min(s.right, hi)
                if left < right:
                    stitched[q].append(Segment(left, right, s.posterior, s.status))
    return stitched


@dataclass
class WindowedPainting:
    """Handle over a windowed, streamed painting written to a directory by
    ``paint(ts, …, window_size=…, out_dir=…)``.

    The heavy per-position posteriors stay on disk — one ``window_NNNNN.npz`` :class:`Painting` per
    genomic window, plus a ``manifest.json`` describing the single shared fit — so this object is
    lightweight (just the fit and the window index). Iterate windows lazily with :meth:`windows` (one
    loaded at a time), or reassemble the whole genome with :meth:`painting` (holds it all).

    Attributes
    ----------
    out_dir : str
        Directory holding the ``window_NNNNN.npz`` files and ``manifest.json``.
    window_size, seqlen : float
        Window width and the painted genome's sequence length (bp).
    bounds : list[tuple[int, float, float]]
        ``(k, lo, hi)`` per window, in genome order.
    Q, pi, w, queries, labels
        The single shared fit — the same for every window.
    default_deadband : float
        Default dead-band carried into every per-window / reassembled :class:`Painting` (for
        :meth:`Painting.segments`). Default :data:`~tspaint.output.DEFAULT_DEADBAND` (``0.4``).
    """
    out_dir: str
    window_size: float
    seqlen: float
    bounds: list
    Q: np.ndarray
    pi: np.ndarray
    w: dict
    queries: list
    labels: dict = None
    default_deadband: float = DEFAULT_DEADBAND

    def _window_path(self, k):
        import os
        return os.path.join(self.out_dir, f"window_{int(k):05d}.npz")

    @property
    def n_windows(self):
        """Number of windows the genome was tiled into."""
        return len(self.bounds)

    def windows(self):
        """Yield ``(lo, hi, Painting)`` per window, loading one file at a time (bounded memory).

        Returns
        -------
        Iterator[tuple[float, float, Painting]]
            One ``(lo, hi, Painting)`` per window in genome order — the window's ``[lo, hi)`` bounds
            and its :class:`Painting`, loaded from ``window_NNNNN.npz`` on demand (one held at a
            time).
        """
        for k, lo, hi in self.bounds:
            yield lo, hi, Painting.load(self._window_path(k))

    def painting(self):
        """Reassemble the full genome-wide :class:`Painting` from the per-window files.

        Loads every window and concatenates by genomic position — this holds the whole-genome
        posteriors in memory (the cost streaming avoided), so call it only when you can afford it,
        or after narrowing to a manageable region.

        Returns
        -------
        Painting
            The stitched genome-wide painting (``ts=None``; carries the shared fit
            ``Q``/``π``/``w``, ``queries``, ``labels`` and :attr:`default_deadband`, and tiles
            ``[0, L)`` exactly as a single-pass paint would).
        """
        tables, bnds = [], []
        for lo, hi, p in self.windows():
            tables.append(p.posteriors)
            bnds.append((lo, hi))
        stitched = _stitch_window_tables(tables, bnds, self.queries)
        return Painting(posteriors=stitched, Q=self.Q, pi=self.pi, w=self.w, loglik_history=[],
                        queries=self.queries, ts=None, labels=self.labels,
                        default_deadband=self.default_deadband, _seqlen=self.seqlen)

    @staticmethod
    def load(out_dir):
        """Reopen a directory previously written by windowed :func:`paint` (reads ``manifest.json``).

        Parameters
        ----------
        out_dir : str
            A directory written by ``paint(..., window_size=…, out_dir=out_dir)`` — it must hold a
            ``manifest.json`` (the shared fit and window bounds) alongside the ``window_NNNNN.npz``
            files.

        Returns
        -------
        WindowedPainting
            A lightweight handle over the per-window files (the window index and shared fit only;
            the per-window posteriors stay on disk until :meth:`windows` / :meth:`painting`).
        """
        import json
        import os
        with open(os.path.join(out_dir, "manifest.json")) as f:
            m = json.load(f)
        return WindowedPainting(
            out_dir=out_dir, window_size=float(m["window_size"]), seqlen=float(m["seqlen"]),
            bounds=[tuple(b) for b in m["bounds"]],
            Q=np.asarray(m["Q"], float), pi=np.asarray(m["pi"], float),
            w={int(k): float(v) for k, v in (m.get("w") or {}).items()},
            queries=[int(q) for q in m["queries"]],
            labels=({int(k): int(v) for k, v in m["labels"].items()} if m.get("labels") else None),
            default_deadband=(DEFAULT_DEADBAND if m.get("deadband") is None
                              else float(m["deadband"])))


def _stream_windowed(ts, out_dir, window_size, res, labels, queries, paint_member, *,
                     deadband=DEFAULT_DEADBAND, progress=False):
    """Paint ``ts`` window-by-window with the fixed fit ``res``, streaming each window to ``out_dir``.

    Writes the shared-fit ``manifest.json`` up front (so a crash mid-run still leaves a loadable
    directory), then paints each window, saves its ``[lo, hi)`` slice, and releases it before cutting
    the next. Skips a window whose ``.npz`` already exists — the run is resumable.
    """
    import json
    import math
    import os
    from .io_relate import _iter_windows

    os.makedirs(out_dir, exist_ok=True)
    L = float(ts.sequence_length)
    n = max(1, math.ceil(L / window_size))
    bounds = [(k, k * window_size, min((k + 1) * window_size, L)) for k in range(n)]
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"window_size": window_size, "seqlen": L, "bounds": bounds,
                   "Q": np.asarray(res.Q).tolist(), "pi": np.asarray(res.pi).tolist(),
                   "w": {int(k): float(v) for k, v in (res.w or {}).items()},
                   "queries": [int(q) for q in queries],
                   "labels": {int(k): int(v) for k, v in (labels or {}).items()},
                   "deadband": float(deadband)}, f)

    it = _iter_windows(ts, window_size)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=n, desc="painting", unit="window")
    for k, lo, hi, w in it:
        path = os.path.join(out_dir, f"window_{k:05d}.npz")
        if os.path.exists(path):                        # resume: already painted
            del w
            continue
        table = paint_member(w)                         # fixed (Q, π, w) from the one global fit
        clipped = _stitch_window_tables([table], [(lo, hi)], queries)   # this window's [lo, hi) slice
        Painting(posteriors=clipped, Q=res.Q, pi=res.pi, w=res.w, loglik_history=res.loglik_history,
                 queries=queries, ts=None, labels=labels, default_deadband=deadband,
                 _seqlen=L).save(path)
        del table, clipped, w                           # release before cutting the next window
    return WindowedPainting(out_dir=out_dir, window_size=window_size, seqlen=L, bounds=bounds,
                            Q=res.Q, pi=res.pi, w=res.w, queries=list(queries), labels=labels,
                            default_deadband=deadband)


def paint(ts, labels, queries=None, *, refs=False, K=2, soft_refs=None, estimate_pi=False,
          deadband=DEFAULT_DEADBAND, smooth=False, epsilon=1e-2, Q0=None, max_iter=12, tol=1e-7, alpha=20.0,
          beta=1.0, priors=None, w0=0.9, mask=None, window_size=None, out_dir=None, n_jobs=None,
          progress=False):
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
        Stored as :attr:`Painting.default_deadband` for :meth:`Painting.segments` (and the plots).
        Defaults to :data:`~tspaint.output.DEFAULT_DEADBAND` (``0.4``, the CLAUDE.md §9 dating value);
        pass ``0.0`` for raw ``argmax`` tracts.
    smooth : bool
        Apply the horizontal BP/EP smoother (:mod:`tspaint.bp`) to the posteriors along the
        genome. **Recommended on inferred (tsinfer / Relate) ARGs**, where tree inference
        scatters spurious breakpoints a per-position deadband cannot filter; redundant on a
        true/known ARG (CLAUDE.md §7). Default ``False``.
    epsilon : float
        Per-breakpoint switch penalty for ``smooth`` (smaller ⇒ more smoothing).
    window_size : float, optional
        **Stream** the painting of a genome-scale tree sequence in genomic windows of this width (bp),
        writing one ``Painting`` per window to ``out_dir`` and returning a lightweight
        :class:`WindowedPainting` handle instead of a single in-memory :class:`Painting`. The model is
        fit **once** on the whole tree sequence, then each window is cut, painted with those fixed
        ``(Q, π, w)``, saved, and **released** before the next — so peak memory is bounded by a single
        window rather than the whole-genome posterior table. The run is **resumable**: a window whose
        ``.npz`` already exists in ``out_dir`` is skipped. This is the memory-bounded path for a
        genome-wide **Relate** ARG (:func:`tspaint.io.relate` → this), where ``EstimatePopulationSize``
        wants the whole chromosome but the full painting will not fit in RAM. Requires ``out_dir``.
        ``None`` (default) paints the whole tree sequence in one call — for the common case, prefer
        that with ``n_jobs`` (which already parallelises the whole-genome paint across cores);
        windowing is only worthwhile when the *output* is too large to hold. Note: with ``smooth=True``
        the genome-axis smoother runs within each window, so smoothing does not cross window seams.
    out_dir : str, optional
        Destination directory for windowed streaming (see ``window_size``); created if absent. Holds
        one ``window_NNNNN.npz`` per window plus a ``manifest.json`` describing the shared fit, so the
        directory is self-contained — reassemble later with :meth:`WindowedPainting.painting` or
        :meth:`WindowedPainting.load`. Required when ``window_size`` is set (and only used then).
    Q0 : (K, K) array, optional
        Initial generator (default a slow symmetric 2-state generator).
    priors : dict[int, tuple[float, float]], optional
        Per-tip ``Beta(alpha_i, beta_i)`` prior overrides for the graded-trust setting
        (keys ⊆ ``soft_refs``); see :func:`tspaint.fit`.
    mask : dict, optional
        **Fragment masking** (CLAUDE.md §2.3): ``{ref: [(left, right), ...]}`` per-reference spans to
        treat as **unlabelled** (the reference emits the query emission there), so a contaminated
        reference anchors only on its clean spans instead of being down-weighted as a whole
        individual (``soft_refs``). Keys may be node ids or sample-ID strings. Feed it directly from
        :meth:`~tspaint.introgression.ReferenceQC.mask` or from :func:`tspaint.foreign_tracts`.
        Applied to both the fit and the painting; exactly equal for any ``n_jobs``.
    max_iter, tol, alpha, beta, w0 : EM controls (see :func:`tspaint.fit`).
    n_jobs : int
        Worker processes for the genome E-step and painting. Default ``None`` → all CPUs / the SLURM
        allocation (:func:`tspaint.parallel.resolve_cores`); pass ``1`` for serial (byte-identical to
        single-core). ``>1`` saturates the node via a process pool
        (:mod:`tspaint.parallel`) — the painting is *exactly* equal to serial, the fit
        ``allclose`` (floating-point reduction order). The CLI resolves this from
        ``$SLURM_JOB_CPUS_PER_NODE``.
    progress : bool
        Show :mod:`tqdm` progress bars. First an ``EM fit`` bar over the EM iterations (the
        slow part on a large ARG — one bar with the running log-likelihood), then a
        ``painting`` bar: per marginal tree when serial (``n_jobs == 1``), per genome chunk
        when parallel (``n_jobs > 1``), and per ensemble member when ``ts`` is a list of tree
        sequences. Uses :mod:`tqdm.auto` — the notebook widget in Jupyter, a text bar in a
        terminal. Default ``False`` (no bar; behaviour is otherwise unchanged).

    Returns
    -------
    Painting or WindowedPainting
        A :class:`Painting` — the soft per-position ancestry posteriors for each query plus the fitted
        ``(Q, π, w)`` and EM log-likelihood history. For an ensemble input the posteriors are the
        ensemble mean with an ARG-uncertainty band (:class:`~tspaint.ensemble.MergedSegment`, with a
        ``posterior_std``). When ``window_size`` is set, a :class:`WindowedPainting` handle over the
        per-window files in ``out_dir`` instead (iterate windows lazily, or ``.painting()`` to
        reassemble the full genome-wide painting).

    See Also
    --------
    tspaint.fit : The underlying blocked-EM fit.
    tspaint.output.hard_segments : Collapse the soft posteriors into hard tracts.
    tspaint.ensemble.merge_posterior_tables : The per-position averaging used for an ensemble.

    Examples
    --------
    >>> import tspaint
    >>> sim = tspaint.simulate_admixture(tspaint.sim.admixture_demography(), n_query=10, n_reference=10)
    >>> painting = tspaint.paint(sim.ts, sim.labels, queries=sim.queries)
    >>> painting.segments(deadband=0.4)      # hard ancestry tracts for dating

    Label references by **sample-ID string** — no need to know the node order — when the tree
    sequence came from a front end (:func:`tspaint.io.singer` / :func:`tspaint.io.tsinfer` stamp
    the source's ids); a diploid base id labels both its haplotypes:

    >>> ts = tspaint.io.singer(vcf, _Ne=1e4, _m=1e-8, _r=1e-8)
    >>> painting = tspaint.paint(ts, {"NA12878": 0, "NA12889": 1})

    Paint from a SINGER posterior ensemble — the mean painting carries an uncertainty band:

    >>> ensemble = tspaint.io.singer(vcf, _Ne=1e4, _m=1e-8, _r=1e-8, ts=20)
    >>> painting = tspaint.paint(ensemble, labels)            # one pooled fit; averaged posteriors
    >>> painting.posteriors[q][0].posterior_std               # per-position ARG-uncertainty band
    """
    from .parallel import resolve_cores
    n_jobs = resolve_cores(n_jobs)          # None -> all CPUs / SLURM allocation (see resolve_cores)
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
    if mask:                                    # fragment masking (§2.3): {ref -> [(l,r)]}, id or node
        mask = {node: spans for k, spans in mask.items() for node in resolve_nodes(ref_ts, k)}

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

    # Q0 left as given; when None, fit() scales the initial generator to the tree-sequence time
    # axis (so Q0*t ~ 1) — a fixed rate washes out on a large calibrated axis (e.g. tsdate node
    # ages) and collapses the painting toward pi. Pass an explicit Q0 to override.

    # One pooled θ fit shared across the ensemble (the M-step is scale-invariant).
    res = fit(members if members is not None else ts,
              [labels] * len(members) if members is not None else labels,
              K=K, Q0=Q0, max_iter=max_iter, tol=tol, soft_refs=soft_refs,
              estimate_pi=estimate_pi, alpha=alpha, beta=beta, priors=priors, w0=w0,
              n_jobs=n_jobs, progress=progress, mask=mask)

    def _paint_member(g, member_progress=False):
        if n_jobs and int(n_jobs) > 1:
            from .parallel import posterior_table_parallel
            table = posterior_table_parallel(g, res.Q, res.pi, w=res.w, labels=labels,
                                             focal=queries, n_jobs=n_jobs,
                                             progress=member_progress, mask=mask)
        else:
            emissions = build_emissions(g, labels, res.w, res.pi, mask)
            table = posterior_table(g, res.Q, res.pi, emissions, focal=queries,
                                    progress=member_progress)
        if smooth:
            from .bp import bp_smooth_track
            table = {q: bp_smooth_track(t, res.pi, epsilon) for q, t in table.items()}
        return table

    # Windowed streaming: fit once (above), then paint-and-release one window at a time to disk,
    # returning a lightweight handle rather than a whole-genome Painting (bounded memory, resumable).
    if window_size is not None:
        if out_dir is None:
            raise ValueError(
                "window_size requires out_dir=...: it streams one Painting per window to that "
                "directory (bounded memory, resumable). For a whole-genome painting in memory use "
                "paint(ts, n_jobs=...) without window_size — n_jobs already parallelises it.")
        if members is not None:
            raise ValueError("window_size streams a single (chromosome-length) tree sequence; paint "
                             "ensemble members separately")
        return _stream_windowed(ts, out_dir, float(window_size), res, labels, queries,
                                _paint_member, deadband=deadband, progress=progress)
    if out_dir is not None:
        raise ValueError("out_dir is only used with window_size=... (windowed streaming)")

    member_tables = None
    if members is None:
        # Single tree sequence: the per-tree (or per-chunk) bar is the progress axis.
        posteriors = _paint_member(ts, member_progress=progress)
    else:
        from .ensemble import merge_posterior_tables
        # Ensemble: one tick per member (each member is a full genome paint); the inner
        # per-tree/per-chunk bar is suppressed to avoid one nested bar per member.
        member_iter = members
        if progress:
            from tqdm.auto import tqdm
            member_iter = tqdm(members, desc="painting", unit="member")
        member_tables = [_paint_member(g) for g in member_iter]  # kept for the per-member CI
        posteriors = merge_posterior_tables(member_tables, samples=queries)

    return Painting(posteriors=posteriors, Q=res.Q, pi=res.pi, w=res.w,
                    loglik_history=res.loglik_history, queries=queries,
                    ts=ts, labels=labels, default_deadband=deadband, n_jobs=n_jobs,
                    _member_posteriors=member_tables, mask=mask)
