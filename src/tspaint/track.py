"""The read-out surface shared by every per-haplotype, per-locus soft posterior.

Both deliverables in tspaint are the same *shape* of object — a dict mapping each haplotype to a
list of contiguous :class:`~tspaint.output.Segment`\\ s that tile ``[0, L)``, each carrying a soft
``(K,)`` posterior — even though they come from very different models:

* :class:`tspaint.Painting` — ancestry from the CTMC blocked-EM fit (``posterior`` = over ancestry
  states), plus an ancestry-specific tail (``Q``/``π``/``w``, :meth:`~tspaint.Painting.rate_through_time`,
  :meth:`~tspaint.Painting.introgression_map`);
* :class:`tspaint.GhostResult` — ``P(ghost)`` from the depth-emission HMM (``posterior`` =
  ``[P(modern), P(ghost)]``), plus a ghost-specific tail (``μ``/``σ``/``A``/``π₀``, ``burden``,
  :meth:`~tspaint.GhostResult.tracts`).

:class:`SoftTrack` factors out the model-agnostic read-out — :meth:`segments` (hard tracts with the
calibrated dead-band), :meth:`posterior_at`, and :meth:`plot` — so both classes share one
implementation instead of each re-deriving it. It is a behaviour mixin, not a dataclass: a subclass
is the dataclass and supplies the data attributes the mixin reads (see :class:`SoftTrack`).
"""
from __future__ import annotations

import numpy as np

from .output import hard_segments, posterior_at, Segment, INFORMATIVE, DEFAULT_DEADBAND

__all__ = ["SoftTrack", "SegmentTrack", "compare_tracks"]


def _diverging_sm(cmap, colors):
    """A ``[0, 1]`` ScalarMappable, optionally from a custom ``colors`` diverging colormap."""
    import matplotlib
    if colors:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
    return matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap)


def _draw_track_row(ax, soft_segs, hard_segs, truth_segs, *, hi, sm, length, ylabel, mark_segs=None):
    """Draw one haplotype strip: soft posterior gradient (top), hard segments (middle), truth (below).

    Shared by :meth:`SoftTrack.plot` and :func:`compare_tracks` so every track/tool renders the same
    way. ``soft_segs`` are :class:`~tspaint.output.Segment`\\ s (gradient by ``posterior[hi]``);
    ``hard_segs`` / ``truth_segs`` are ``(left, right, state)`` tuples (solid, highlighted at ``hi``).
    ``mark_segs`` (``[(left, right), ...]``, e.g. a reference's *masked* / unlabelled spans) are
    hatched over the soft band.
    """
    import matplotlib
    ymin = -0.25 if truth_segs is not None else 0.0
    if truth_segs is not None:
        for (l, r, s) in truth_segs:
            ax.barh(-0.25, r - l, left=l, height=0.5, color=sm.to_rgba(1.0 if s == hi else 0.0),
                    edgecolor="none")
    for (l, r, s) in hard_segs:
        ax.barh(0.25, r - l, left=l, height=0.5, color=sm.to_rgba(1.0 if s == hi else 0.0),
                edgecolor="none")
    ax.axhline(0, c='black', lw=0.25)
    ax.axhline(0.5, c='black', lw=0.25)
    for seg in soft_segs:
        ax.barh(1, seg.right - seg.left, left=seg.left, height=1,
                color=sm.to_rgba(seg.posterior[hi]), edgecolor="none")
    for (l, r) in (mark_segs or ()):                # masked / unlabelled spans: hatch over the soft band
        ax.barh(1, r - l, left=l, height=1, facecolor="none", hatch="////",
                edgecolor="black", linewidth=0.0)
    ax.set_ylim(ymin, 1.5)
    ax.set_xlim(0, length)
    ax.set_ylabel(ylabel, rotation=0, fontsize=7, color="0.0", horizontalalignment="right")
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    for sp in ("top", "bottom", "left", "right"):
        ax.spines[sp].set_visible(True)
    ax.grid(False)


class SoftTrack:
    """Mixin providing the shared read-out surface for a per-locus soft posterior along the genome.

    A subclass must be a :func:`~dataclasses.dataclass` that provides:

    * ``posteriors`` — ``dict[int, list[Segment | MergedSegment]]`` tiling ``[0, L)`` per haplotype
      (e.g. from :func:`tspaint.output.posterior_table` or
      :func:`tspaint.ensemble.merge_posterior_tables`);
    * ``default_deadband`` — :class:`float`, the default dead-band for :meth:`segments`;
    * ``_seqlen`` — the genome length (or override :attr:`length`).

    A subclass may set the class attributes ``_hi_state`` / ``_hi_label`` to control which state's
    probability drives the :meth:`plot` colour scale and how the colour bar is labelled (the
    "highlighted" state, drawn at the top of the diverging colormap). Defaults highlight state 0 and
    label it ``"P(ancestry A)"`` (so :class:`tspaint.Painting` needs no override).
    """

    # Which state the plot highlights (top of the diverging colormap) and its colour-bar label.
    # Subclasses override (e.g. GhostResult uses state 1 = ghost, "P(ghost)").
    _hi_state = 0
    _hi_label = "P(ancestry A)"

    @property
    def length(self):
        """Sequence length of the painted genome (default: the stored ``_seqlen``)."""
        return self._seqlen

    @property
    def samples(self):
        """Ordered haplotype ids carried by this track — the row order used by :meth:`plot`."""
        return list(self.posteriors)

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
            Per haplotype, hard ``(left, right, state)`` segments.
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

    def plot(self, truth=None, title=None, cmap='coolwarm', colors=None, return_plot=False,
             deadband=None, row_labels=None, mark_spans=None):
        """Stacked strip plot of the soft posterior — one row per haplotype.

        Each haplotype row shows, top to bottom: the **soft** per-position posterior for the
        highlighted state (``self._hi_label``, e.g. ``P(ancestry A)`` for a painting or ``P(ghost)``
        for the ghost detector) as a colour gradient, the **hard** segments (:meth:`segments` at the
        ``deadband`` below), and — when ``truth`` is given — the true tracts as a reference track
        beneath. A shared colour bar maps the highlighted state's probability (highlighted state at
        the top of ``cmap``, the other at the bottom). For an ensemble result the rows show the
        ensemble-mean posterior. Requires matplotlib.

        Parameters
        ----------
        truth : dict[int, list[tuple[float, float, int]]], optional
            Ground-truth tracts ``(left, right, state)`` per haplotype, drawn as a reference track
            below each row. No truth track is drawn when ``None`` (default).
        title : str, optional
            Title placed on the top haplotype axes.
        cmap : str or matplotlib.colors.Colormap, optional
            Diverging colormap mapping the highlighted state's probability ``∈ [0, 1]`` (highlighted
            state → ``1.0``, the other → ``0.0``). Default ``'coolwarm'``. Ignored when ``colors`` is
            given.
        colors : list, optional
            Colours to build a custom diverging colormap from, overriding ``cmap`` — e.g.
            ``["#2c7bb6", "#ffffbf", "#d7191c"]``.
        return_plot : bool, optional
            Return the Matplotlib ``(figure, axes)`` so the caller can further customise or save
            the plot, instead of ``None``. Default ``False``.
        deadband : float, optional
            Dead-band for the hard-segment overlay. Defaults to :attr:`default_deadband` (so the
            plotted tracts match :meth:`segments`); pass a value to override for this plot only.
        row_labels : dict[int, str], optional
            Per-sample y-axis label overriding the default ``hapl. i`` (e.g. mark reference rows with
            their nominal ancestry). Samples absent keep the default.
        mark_spans : dict[int, list[tuple[float, float]]], optional
            Per-sample ``[(left, right), ...]`` spans to **hatch** over the soft band — e.g. a
            reference's masked / unlabelled spans, so the plot shows *where* a reference was un-anchored.

        Returns
        -------
        tuple or None
            ``(figure, list_of_axes)`` if ``return_plot`` is ``True``, else ``None``.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        hi = self._hi_state
        qs = self.samples
        segments = self.segments(deadband=deadband)     # None -> the track's default_deadband

        sm = _diverging_sm(cmap, colors)
        fig = plt.figure(figsize=(9, 0.3 * len(qs) + 1))
        gs = fig.add_gridspec(len(qs), 2, width_ratios=[1, 0.03], hspace=0)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(len(qs))]

        for i, q in enumerate(qs):
            ylabel = (row_labels or {}).get(q, f'hapl. {i}')
            _draw_track_row(axes[i], self.posteriors[q], segments[q],
                            truth.get(q) if truth else None,          # rows with no truth omit the track
                            hi=hi, sm=sm, length=self.length, ylabel=ylabel,
                            mark_segs=(mark_spans or {}).get(q))
            axes[i].tick_params(axis='x', bottom=True)
            if i < len(axes) - 1:
                axes[i].xaxis.set_major_locator(matplotlib.ticker.NullLocator())

        ax = fig.add_subplot(gs[:, 1])
        ax.set_axis_off()
        cb = fig.colorbar(sm, ax=ax, fraction=0.5, pad=0.01)
        cb.set_label(self._hi_label)
        if title:
            axes[0].set_title(title)
        plt.tight_layout()
        if return_plot:
            return fig, axes


def _infer_K(data):
    """Number of ancestry states across a segments dict (max soft-posterior length / hard state + 1)."""
    K = 2
    for segs in data.values():
        for s in segs:
            K = max(K, int(np.asarray(s.posterior).shape[0]) if isinstance(s, Segment) else int(s[2]) + 1)
    return K


def _to_segments(seglist, K):
    """Normalise one sample's list to soft :class:`~tspaint.output.Segment`\\ s.

    :class:`Segment`\\ s pass through; a hard ``(left, right, state)`` tuple becomes a one-hot
    ``Segment`` so the shared renderer draws it as a solid state bar.
    """
    out = []
    for s in seglist:
        if isinstance(s, Segment):
            out.append(s)
        else:
            left, right, state = s
            post = np.zeros(K)
            post[int(state)] = 1.0
            out.append(Segment(float(left), float(right), post, INFORMATIVE))
    return out


def _infer_length(posteriors):
    return max((float(s.right) for segs in posteriors.values() for s in segs), default=0.0)


class SegmentTrack(SoftTrack):
    """Wrap any per-sample segments in a plottable track — hard tuples *or* soft posteriors.

    Gives ``painting.segments()`` (hard ``(left, right, state)`` tuples), the comparator painters
    (``rfmix_paint`` / ``gnomix`` / ``tspaint_paint`` — soft :class:`~tspaint.output.Segment`\\ s), or any
    ``{sample: segments}`` dict the same read-out as a :class:`~tspaint.Painting`: :meth:`plot`,
    :meth:`segments`, :meth:`posterior_at`. Hard tuples become one-hot segments and render as solid
    per-state bars; soft segments render as the posterior gradient — so different tools plot in one
    consistent style (compare several at once with :func:`compare_tracks`). Purely a view: it does not
    alter the segments it wraps.

    Parameters
    ----------
    segments : dict or SoftTrack
        ``{sample: [Segment | (left, right, state)]}``, or an existing track / :class:`~tspaint.Painting`
        (its ``posteriors`` and length are reused).
    length : float, optional
        Sequence length for the x-axis; defaults to the largest segment ``right``.
    deadband : float, optional
        Default dead-band for :meth:`segments`. Defaults to :data:`~tspaint.output.DEFAULT_DEADBAND`.
    hi_state : int, optional
        The state highlighted by :meth:`plot`'s colour scale (top of the colormap). Default ``0``.
    hi_label : str, optional
        Colour-bar label; defaults to the painting's ``"P(ancestry A)"``.
    K : int, optional
        Number of ancestry states (for one-hot conversion of hard tuples); inferred when ``None``.
    """

    def __init__(self, segments, *, length=None, deadband=DEFAULT_DEADBAND, hi_state=0,
                 hi_label=None, K=None):
        if isinstance(segments, SoftTrack):
            data = segments.posteriors
            if length is None:
                length = segments.length
        else:
            data = segments
        K = K if K is not None else _infer_K(data)
        self.posteriors = {s: _to_segments(list(v), K) for s, v in data.items()}
        self._seqlen = float(length) if length is not None else _infer_length(self.posteriors)
        self.default_deadband = deadband
        self._hi_state = hi_state
        if hi_label is not None:
            self._hi_label = hi_label


def compare_tracks(tracks, sample, *, truth=None, length=None, cmap='coolwarm', colors=None,
                   title=None, return_plot=False, deadband=None):
    """Stack several tools' calls for **one haplotype**, one row per tool, for visual comparison.

    Parameters
    ----------
    tracks : mapping
        ``{tool name: segments}`` — each value a ``{sample: segments}`` dict, a
        :class:`~tspaint.Painting`, or a :class:`SegmentTrack` (soft or hard; normalised via
        :class:`SegmentTrack`).
    sample : int
        The haplotype (sample id / key) to compare across tools. Call once per haplotype.
    truth : dict, optional
        Ground-truth ``{sample: [(left, right, state)]}`` drawn as a reference row at the bottom.
    length : float, optional
        x-axis length; defaults to the largest ``right`` across the tools.
    cmap, colors, title, return_plot
        As for :meth:`SoftTrack.plot`.

    Returns
    -------
    tuple or None
        ``(figure, axes)`` if ``return_plot`` else ``None``.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    names = list(tracks)
    sts = {name: (t if isinstance(t, SegmentTrack) else SegmentTrack(t, length=length))
           for name, t in tracks.items()}
    hi = 0
    L = float(length) if length is not None else max((st.length for st in sts.values()), default=0.0)
    truth_seg = truth.get(sample) if truth else None
    n_rows = len(names) + (1 if truth_seg is not None else 0)

    sm = _diverging_sm(cmap, colors)
    fig = plt.figure(figsize=(9, 0.4 * n_rows + 1))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1, 0.03], hspace=0)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]

    for i, name in enumerate(names):
        st = sts[name]
        _draw_track_row(axes[i], st.posteriors.get(sample, []),
                        st.segments(deadband=deadband).get(sample, []), None,
                        hi=hi, sm=sm, length=L, ylabel=name)
        axes[i].tick_params(axis='x', bottom=True)
        if i < n_rows - 1:
            axes[i].xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    if truth_seg is not None:
        _draw_track_row(axes[-1], [], truth_seg, None, hi=hi, sm=sm, length=L, ylabel="truth")
        axes[-1].tick_params(axis='x', bottom=True)

    ax = fig.add_subplot(gs[:, 1])
    ax.set_axis_off()
    fig.colorbar(sm, ax=ax, fraction=0.5, pad=0.01)
    if title:
        axes[0].set_title(title)
    plt.tight_layout()
    if return_plot:
        return fig, axes
