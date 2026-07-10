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

# __all__ = ["SoftTrack", "SegmentTrack", "compare_tracks", "STATE_COLORS"]
__all__ = ["SoftTrack", "SegmentTrack", "compare_tracks"]


#: Categorical per-state colours for **K > 2** ancestry rendering (2-state tracks use a diverging
#: colormap instead — see :meth:`SoftTrack.plot`). The reference qualitative palette, validated
#: colourblind-safe (worst adjacent-pair CVD ΔE ≈ 24); the plot's state **legend** carries the
#: labels (A, B, C, …), the required relief for the two lower-contrast hues. Assigned in fixed order
#: by ancestry-state index (never cycled while ``K`` is within range; beyond its length it wraps —
#: rare for local ancestry). Override per-plot with ``plot(colors=[...])``.
#STATE_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


def _diverging_sm(cmap, colors):
    """A ``[0, 1]`` ScalarMappable, optionally from a custom ``colors`` diverging colormap."""
    import matplotlib
    if colors:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
    return matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap)


def _K_of(posteriors):
    """Number of ancestry states ``K``, read from a posterior vector (``Segment`` or ``MergedSegment``)."""
    for segs in posteriors.values():
        if segs:
            return int(np.asarray(segs[0].posterior).shape[0])
    return 2

def _hex_palette():
    """The matplotlib default categorical colour cycle (``axes.prop_cycle``) as full-opacity hex
    strings. Robust to a cycle that yields RGB tuples, hex strings or named colours."""
    import matplotlib
    cyc = matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", [".15"])
    return [matplotlib.colors.to_hex(c) for c in cyc]


class _Colorizer:
    """The single state→colour rule every mark on a track shares, so soft/hard/truth stay consistent.

    **Confidence as opacity** (one rule for every ``K``): the colour is always the **argmax** state's
    hue, and the winning probability drives that hue's **opacity** — faint = uncertain, solid =
    confident. The confidence is the winning probability normalised out of its ``[1/K, 1]`` range,
    ``conf = (max_p − 1/K) / (1 − 1/K)`` (0 at a uniform 'no-information' posterior, 1 at a one-hot),
    and the soft opacity is ``conf · alpha`` — spanning ``[0, alpha]``. ``alpha`` (default 1) sets the
    opacity of a fully-confident locus and **fades the whole plot uniformly**: the hard-segment and
    truth bars are drawn at opacity ``alpha`` (:meth:`hard`), the soft band at ``conf · alpha``. A
    smaller ``alpha`` fades everything (e.g. ``alpha=0.5`` ⇒ at most 50% opaque). At any ``alpha`` a
    one-hot (fully-confident) posterior renders identically to its hard call: :meth:`soft` == :meth:`hard`.

    Only the **legend/colour-bar** depends on ``K`` (``self.categorical = K > 2``):

    * **K ≤ 2** — a diverging colour bar whose two ends are the two state hues and whose centre is
      transparent (the soft rule swept over ``P(state hi)``).
    * **K > 2** — a per-state legend of hues (``A, B, C, …``, then ``ghost`` for truth-only states).
    """

    def __init__(self, K, *, hi=0, hi_label="P(ancestry A)", cmap="coolwarm", colors=None, alpha=None,
                 n_source=None, highlight=False):
        import matplotlib
        self.K = int(K)
        # states 0..n_source-1 are painted ancestry sources (legend A, B, C, …); any states above them
        # are truth-only foreign states embedded above the sources — the ghost(s) — labelled "ghost".
        self.n_source = int(n_source) if n_source is not None else self.K
        # highlight mode: a *single* interesting state (``hi``) carries a colour and **every other state
        # is white** — the binary "is it ``hi``?" read-out (e.g. the ghost detector: white modern + one
        # ghost hue). It always uses the single ``P(hi)`` colour bar, never the categorical A/B/… legend.
        self.highlight = bool(highlight)
        self.categorical = (self.K > 2) and not self.highlight   # legend (K>2) vs diverging colour bar
        self.hi = int(hi)
        self.hi_label = hi_label
        self.alpha = 1.0 if alpha is None else float(alpha)  # max opacity: soft opacity spans [0, alpha]
        # one FULL-OPACITY per-state palette for every K (opacity is applied per-locus in soft()).
        if colors:
            base = list(colors)
            if self.K == 2:                                  # the two extremes are the two state hues
                base = [base[0], base[-1]]                   # (a legacy 3-stop diverging list still works)
            elif len(base) < self.K:
                raise ValueError(f"colors must supply at least K={self.K} entries")
        else:
            base = _hex_palette()
        self.palette = np.array([matplotlib.colors.to_rgba(base[k % len(base)]) for k in range(self.K)],
                                float)
        self.palette[:, 3] = 1.0                             # hues are opaque; soft() sets the alpha

    def _opacity(self, max_p):
        """Confidence→opacity: normalise ``max_p`` out of ``[1/K, 1]`` to ``[0, 1]``, then scale into the
        ``[0, alpha]`` range — ``alpha`` is the opacity of a fully-confident locus (default 1)."""
        conf = (float(max_p) - 1.0 / self.K) / (1.0 - 1.0 / self.K) if self.K > 1 else 1.0
        return float(np.clip(conf, 0.0, 1.0)) * self.alpha

    def hard(self, state):
        """RGBA for a hard ``(…, state)`` segment (and the truth bars) — the state's hue at opacity
        ``alpha`` (default 1, fully opaque). In **highlight** mode every state other than ``hi`` is
        **white** (the binary is-it-``hi`` read-out). ``alpha`` fades the whole plot, hard/truth included."""
        if self.highlight and int(state) != self.hi:
            return (1.0, 1.0, 1.0, self.alpha)              # non-highlight states -> white
        r, g, b = self.palette[int(state) % self.K][:3]
        return (r, g, b, self.alpha)

    def soft(self, posterior):
        """RGBA for a soft posterior — the argmax state's hue at ``opacity = confidence · alpha`` (spans
        ``[0, alpha]``; see the class docstring). A one-hot posterior → :meth:`hard` at any ``alpha``. In
        **highlight** mode it is always the ``hi`` hue at ``opacity = P(hi) · alpha`` — white where
        ``P(hi) ≈ 0``, solid where ``P(hi) ≈ 1`` (a ``P(hi)`` heat strip on a white ground)."""
        p = np.asarray(posterior, float)
        if p.shape[0] != self.K:                            # a track with fewer/more states than the panel
            q = np.zeros(self.K)
            q[: min(p.shape[0], self.K)] = p[: self.K]
            p = q
        if self.highlight:
            r, g, b = self.palette[self.hi][:3]
            return (r, g, b, float(np.clip(p[self.hi], 0.0, 1.0)) * self.alpha)
        k = int(np.argmax(p))
        r, g, b = self.palette[k][:3]
        return (r, g, b, self._opacity(p[k]))

    def decorate(self, fig, ax):
        """Fill the spare right-hand axis: a diverging colour bar (K ≤ 2) or a per-state legend (K > 2)."""
        ax.set_axis_off()
        if not self.categorical:
            # sweep P(state hi) 0→1; each sample IS the soft colour (hue + confidence-opacity), so the
            # bar reads hue → transparent centre → other hue, exactly like the strip.
            import matplotlib
            rows = []
            for x in np.linspace(0.0, 1.0, 256):
                p = np.zeros(max(self.K, 2))
                p[self.hi] = x
                p[1 - self.hi] = 1.0 - x                     # the other of the two states
                rows.append(self.soft(p))
            sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1),
                                              cmap=matplotlib.colors.ListedColormap(rows))
            cb = fig.colorbar(sm, ax=ax, fraction=0.5, pad=0.01)
            cb.set_label(self.hi_label)
            return
        import matplotlib.patches as mpatches
        n_ghost = self.K - self.n_source

        def _label(k):
            if k < self.n_source:                            # a painted ancestry source: A, B, C, …
                return chr(65 + k)
            g = k - self.n_source                            # a truth-only foreign state: the ghost(s)
            return "ghost" if n_ghost == 1 else f"ghost {g + 1}"

        handles = [mpatches.Patch(facecolor=tuple(self.palette[k]), edgecolor="none", label=_label(k))
                   for k in range(self.K)]
        ax.legend(handles=handles, loc="center left", frameon=False, fontsize=8,
                  title="ancestry", handlelength=1.0, handleheight=1.0,
                  borderaxespad=0.0)


#: Fixed width ratio of the right-hand legend / colour-bar column, used by every strip plot so they all
#: reserve the same space — a ghost plot (``K=2`` P(ghost) colour bar) and a ``K>2`` painting
#: (categorical A/B/… legend) then default to the same figure layout instead of differing by the legend
#: column. (matplotlib's auto-layout still fits a colour bar and a legend slightly differently, so the
#: plotted strips can still differ by a few %; this equalises the *reserved* column.)
_LEGEND_W = 0.12

#: Default hatch for a marked span in :func:`_draw_track_row`. A per-span 3rd element overrides it, so
#: :meth:`tspaint.Painting.plot` draws masked reference spans and ghost tracts in **distinct** hatches.
_DEFAULT_HATCH = "////"


def _draw_track_row(ax, soft_segs, hard_segs, truth_segs, *, colorizer, length, ylabel, mark_segs=None):
    """Draw one haplotype strip: soft posterior (top), hard segments (middle), truth (below).

    Shared by :meth:`SoftTrack.plot` and :func:`compare_tracks` so every track/tool renders the same
    way. ``soft_segs`` are :class:`~tspaint.output.Segment`\\ s (coloured by ``colorizer.soft``);
    ``hard_segs`` / ``truth_segs`` are ``(left, right, state)`` tuples (solid ``colorizer.hard``). The
    ``colorizer`` gives soft, hard and truth marks one consistent state→colour mapping (diverging for
    ≤ 2 states, categorical for more). ``mark_segs`` are spans hatched over the hard and soft bands —
    each ``(left, right)`` or ``(left, right, hatch)`` (its own hatch, else :data:`_DEFAULT_HATCH`); this
    is how a reference's *masked* / unlabelled spans and overlaid *ghost* tracts get distinct hatches.
    """
    import matplotlib
    ymin = 0.0
    ymax = ymin

    def _hatch_band(y):                                     # hatch every mark_seg over the band at y
        for m in (mark_segs or ()):
            ax.barh(y, m[1] - m[0], left=m[0], height=1, facecolor="none",
                    hatch=(m[2] if len(m) > 2 else _DEFAULT_HATCH), edgecolor="none")

    if truth_segs:
        if ymax: ax.axhline(ymax, c='black', lw=0.25)
        ymax += 0.5
        for (l, r, s) in truth_segs:
            ax.barh(ymax-0.25, r - l, left=l, height=0.5, color=colorizer.hard(s), edgecolor="none")
    if hard_segs:
        if ymax: ax.axhline(ymax, c='black', lw=0.25)
        ymax += 0.5
        for (l, r, s) in hard_segs:
            ax.barh(ymax-0.25, r - l, left=l, height=0.5, color=colorizer.hard(s), edgecolor="none")
        if mark_segs:
            _hatch_band(ymax-0.5)                          # masked / ghost spans: hatch over the hard band
    if soft_segs:
        if ymax: ax.axhline(ymax, c='black', lw=0.25)
        ymax += 0.5
        for seg in soft_segs:
            ax.barh(ymax-0.25, seg.right - seg.left, left=seg.left, height=0.5,
                    color=colorizer.soft(seg.posterior), edgecolor="none")
        if mark_segs:
            _hatch_band(ymax-0.5)                          # masked / ghost spans: hatch over the soft band

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, length)
    ax.set_ylabel(ylabel, rotation=0, fontsize=7, color="0.0", horizontalalignment="right")
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    for sp in ("top", "bottom", "left", "right"):
        ax.spines[sp].set_visible(True)
    ax.grid(False)


def _curve_step(segs, state):
    """Step ``(xs, ys)`` for the piecewise-constant ``P(state=state)`` across ``segs`` (a Segment list)."""
    xs, ys = [], []
    for s in segs:
        p = np.asarray(s.posterior, float)
        y = float(p[state]) if state < p.shape[0] else 0.0
        xs += [s.left, s.right]
        ys += [y, y]
    return xs, ys


#: Height of a hard-segment / truth band under the curves in :func:`_draw_curve_row` — 10% of the
#: ``[0, 1]`` curve height, so the bands stay a thin annotation and the curves keep the focus.
_CURVE_BAND_H = 0.1


def _draw_curve_row(ax, soft_segs, *, colorizer, length, ylabel, overlays=None,
                    hard_segs=None, truth_segs=None):
    """Draw one haplotype row as posterior **curves** (the ``curves=True`` alternative to the colour band).

    One line per ancestry state in its segment hue (``colorizer.palette[k]`` — the same colour that fills
    the state's segments), ``y = P(state)`` in ``[0, 1]`` with a faint fill under each curve.
    ``overlays`` is an optional list of extra ``(segs, state, colour, linestyle, label)`` lines — e.g.
    the ghost detector's ``P(ghost)``. Mirroring :func:`_draw_track_row`, the **hard segments**
    (``hard_segs``) and the **truth** tracts (``truth_segs``) — each ``(left, right, state)`` in the
    state's solid ``colorizer.hard`` colour — are stacked as thin bands just below the curves (hard
    first, then truth), but each band is only :data:`_CURVE_BAND_H` (~10 %) of the curve height so the
    curves stay the focus.
    """
    import matplotlib
    band, gap = _CURVE_BAND_H, 0.02
    y = 0.0                                                 # cursor: descends below the curves per band
    for segs in (hard_segs, truth_segs):                   # hard band, then truth band (like _draw_track_row)
        if not segs:
            continue
        ax.axhline(y, c="black", lw=0.25)                  # separator above the band
        top = y - gap
        for (l, r, s) in segs:
            ax.barh(top - band / 2, r - l, left=l, height=band, color=colorizer.hard(s), edgecolor="none")
        y = top - band
    for k in range(colorizer.n_source):                    # one curve per painted ancestry state
        xs, ys = _curve_step(soft_segs, k)
        ax.fill_between(xs, ys, color=tuple(colorizer.palette[k][:3]), alpha=0.3)
        ax.plot(xs, ys, color=tuple(colorizer.palette[k][:3]), lw=1, label=chr(65 + k))
    for (segs, state, color, ls, label) in (overlays or ()):
        xs, ys = _curve_step(segs, state)
        ax.plot(xs, ys, color=color, lw=1.2, ls=ls, label=label)
    ax.set_ylim(min(y, 0.0) - 0.03, 1.03)
    ax.set_xlim(0, length)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis='y', labelsize=6)
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel(ylabel, rotation=0, fontsize=7, color="0.0", horizontalalignment="right")
    ax.grid(True, axis='y', lw=0.3, alpha=0.4)
    for sp in ("top", "bottom", "left", "right"):
        ax.spines[sp].set_visible(True)


def _draw_curve_legend(fig, ax, colorizer, overlay_styles):
    """A line legend for :func:`_draw_curve_row`: the per-state hues (``A``, ``B``, …) plus any overlay
    lines (``overlay_styles`` = ``(colour, linestyle, label)``, e.g. the ghost dashed line)."""
    import matplotlib.lines as mlines
    ax.set_axis_off()
    handles = [mlines.Line2D([], [], color=tuple(colorizer.palette[k][:3]), lw=1.6, label=chr(65 + k))
               for k in range(colorizer.n_source)]
    for (color, ls, label) in overlay_styles:
        handles.append(mlines.Line2D([], [], color=color, lw=1.6, ls=ls, label=label))
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=8,
              title="posterior", handlelength=1.6, borderaxespad=0.0)


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

    def posteriors_as_frame(self, samples=None):
        """The soft per-position posterior as a pandas ``DataFrame`` — one row per segment.

        **Wide** layout: the interval columns ``haplotype``, ``left``, ``right`` followed by one
        probability column per ancestry state — ``A``, ``B``, … (state ``k`` → ``chr(65 + k)``,
        matching the plot legend and :meth:`segments_as_frame`) — plus a trailing ``status`` column
        (:data:`~tspaint.output.INFORMATIVE` or :data:`~tspaint.output.MISSING_INFO`, so a
        prior-fallback span is not mistaken for a genuine 50-50 call, CLAUDE.md §4.2). For an
        **ensemble** painting each state additionally gets a ``<letter>_std`` column carrying the
        per-position ARG-uncertainty band (:attr:`~tspaint.ensemble.MergedSegment.posterior_std`).

        pandas is imported lazily, so it is only needed when this method is called.

        Parameters
        ----------
        samples : iterable[int], optional
            Haplotypes to include, in the given order; defaults to all of :attr:`samples`.

        Returns
        -------
        pandas.DataFrame
            One row per ``(haplotype, segment)``; columns
            ``haplotype, left, right, <states…>[, <states…>_std], status``.
        """
        import pandas as pd

        qs = list(self.samples if samples is None else samples)
        K = _K_of(self.posteriors)
        letters = [chr(65 + k) for k in range(K)]
        has_std = any(getattr(s, "posterior_std", None) is not None
                      for q in qs for s in self.posteriors.get(q, []))
        cols = {"haplotype": [], "left": [], "right": []}
        for L in letters:
            cols[L] = []
        if has_std:
            for L in letters:
                cols[f"{L}_std"] = []
        cols["status"] = []
        for q in qs:
            for seg in self.posteriors.get(q, []):
                cols["haplotype"].append(q)
                cols["left"].append(float(seg.left))
                cols["right"].append(float(seg.right))
                p = np.asarray(seg.posterior, float)
                for k, L in enumerate(letters):
                    cols[L].append(float(p[k]) if k < p.shape[0] else float("nan"))
                if has_std:
                    sd = getattr(seg, "posterior_std", None)
                    sd = np.asarray(sd, float) if sd is not None else None
                    for k, L in enumerate(letters):
                        cols[f"{L}_std"].append(
                            float(sd[k]) if sd is not None and k < sd.shape[0] else float("nan"))
                cols["status"].append(seg.status)
        return pd.DataFrame(cols)

    def segments_as_frame(self, deadband=None, samples=None):
        """The hard ancestry tracts as a pandas ``DataFrame`` — one row per tract.

        Columns ``haplotype``, ``start``, ``end``, ``ancestry`` — the calibrated-dead-band hard
        segmentation (:meth:`segments`) with the called state rendered as a letter ``A``, ``B``, …
        (state ``k`` → ``chr(65 + k)``, matching the plot legend and :meth:`posteriors_as_frame`).
        This is the tract-length / admixture-dating object as a table (CLAUDE.md §9).

        pandas is imported lazily, so it is only needed when this method is called.

        Parameters
        ----------
        deadband : float, optional
            Confidence dead-band for the hard segmentation (see :meth:`segments`); defaults to
            :attr:`default_deadband`.
        samples : iterable[int], optional
            Haplotypes to include, in the given order; defaults to all of :attr:`samples`.

        Returns
        -------
        pandas.DataFrame
            One row per ``(haplotype, tract)``; columns ``haplotype, start, end, ancestry``.
        """
        import pandas as pd

        hard = self.segments(deadband=deadband)
        qs = list(self.samples if samples is None else samples)
        cols = {"haplotype": [], "start": [], "end": [], "ancestry": []}
        for q in qs:
            for (left, right, state) in hard.get(q, []):
                cols["haplotype"].append(q)
                cols["start"].append(float(left))
                cols["end"].append(float(right))
                cols["ancestry"].append(chr(65 + int(state)))
        return pd.DataFrame(cols)

    def summary(self, truth=None, deadband=None, samples=None):
        """Quality summary of this track — every :mod:`tspaint.metrics` metric in one object.

        Bundles the model-free read-out any soft ancestry track exposes — the **total ancestry
        proportions**, **confidence**, **fragmentation** (switches/Mb) and boundary **flicker**,
        none needing a reference — plus, against ``truth``, **accuracy** / **balanced accuracy**,
        **breakpoint precision / recall**, the inferred ÷ true switch-density **ratio**,
        **tract-boundary error**, and the **calibration** / **size-stratified accuracy** curves
        (CLAUDE.md §9). Computed at the same dead-band :meth:`segments` / :meth:`plot` use, so the
        numbers describe the plotted hard tracts. Shared by :class:`~tspaint.Painting` and any
        :class:`SegmentTrack`, so a wrapped RFMix / gnomix / tspaint painting summarises the same way.

        Parameters
        ----------
        truth : dict[int, list[tuple[float, float, int]]], optional
            True ancestry-state tracts per sample (e.g. from :func:`tspaint.validate.map_truth`).
            When given, adds the truth-dependent metrics (accuracy, precision / recall,
            boundary error, reliability, size-stratified accuracy, …).
        deadband : float, optional
            Dead-band for the hard segmentation the fragmentation / precision / recall read.
            Defaults to :attr:`default_deadband` (matching :meth:`segments` and :meth:`plot`).
        samples : iterable[int], optional
            Haplotypes to summarise; defaults to all of :attr:`samples`.

        Returns
        -------
        tspaint.validate.PaintingSummary
            A dict-compatible object of the metrics with a formatted ``repr`` and a
            :meth:`~tspaint.validate.PaintingSummary.plot_size_stratified` method
            (see :func:`tspaint.validate.painting_summary` for the keys).
        """
        from .validate import painting_summary
        K = 2
        for segs in self.posteriors.values():           # infer K from a posterior (Segment or Merged)
            if segs:
                K = int(np.asarray(segs[0].posterior).shape[0])
                break
        samples = self.samples if samples is None else list(samples)
        return painting_summary(self.posteriors, self.segments(deadband=deadband), self.length,
                                truth=truth, samples=samples, K=K)

    def _summary_title(self, truth=None, deadband=None, samples=None):
        """One-line rendering of :meth:`summary` for the default :meth:`plot` title (``None`` when
        there is nothing to summarise). Subclasses whose states are not ancestry (e.g. the ghost
        detector) override this to opt out of the default title."""
        s = self.summary(truth=truth, deadband=deadband, samples=samples)
        if not s["n_samples"]:
            return None

        def fmt(x, spec):
            return spec.format(x) if isinstance(x, float) and not np.isnan(x) else "–"

        toks = []
        if truth is not None:                       # precision / recall need the reference tracts
            toks.append(f"precision {fmt(s['precision'], '{:.2f}')}")
            toks.append(f"recall {fmt(s['recall'], '{:.2f}')}")
            toks.append(f"fragmentation {fmt(s.get('switch_ratio', float('nan')), '{:.2f}')}×")
        else:                                       # no truth: report the raw inferred switch density
            toks.append(f"fragmentation {fmt(s['switch_per_mb'], '{:.2f}')} sw/Mb")
        names = [chr(65 + k) for k in range(len(s["proportion"]))]
        prop = " ".join(f"{nm} {fmt(p * 100, '{:.0f}')}%" for nm, p in zip(names, s["proportion"]))
        toks.append(f"ancestry {prop}")
        return " · ".join(toks)

    def _prepare_truth(self, truth):
        """Hook: transform the ``truth`` dict before it drives :meth:`plot` (its colours, ``K``, and
        truth band). Identity by default; :class:`~tspaint.archaic.GhostResult` overrides it to
        binarise the truth to its ghost state so the truth band reads white + one ghost colour."""
        return truth

    def _make_colorizer(self, K, n_source, *, cmap, colors, alpha):
        """Hook: build the :class:`_Colorizer` for :meth:`plot`. The ancestry A/B/… scheme by default;
        :class:`~tspaint.archaic.GhostResult` overrides it for the single-colour ghost highlight."""
        return _Colorizer(K, hi=self._hi_state, hi_label=self._hi_label, n_source=n_source,
                          cmap=cmap, colors=colors, alpha=alpha)

    def plot(self, truth=None, segments=False, title=None, cmap='coolwarm', colors=None, figsize=None,
             return_plot=False, alpha=None, deadband=None, row_labels=None, mark_spans=None, samples=None,
             curves=False, curve_overlays=None):

        """Stacked strip plot of the soft posterior — one row per haplotype.

        Each haplotype row shows, top to bottom: the **soft** per-position posterior as a colour, the
        **hard** segments (:meth:`segments` at the ``deadband`` below), and — when ``truth`` is given
        — the true tracts as a reference track beneath. Soft, hard and truth marks share **one**
        state→colour rule (:class:`_Colorizer`): the colour is the **argmax** state's hue and its
        **opacity is the confidence** — faint where uncertain, solid where confident — so a confident
        soft locus matches its hard call and the truth bar (both solid). Works for any ``K``; only the
        key differs — a **diverging colour bar** (two state hues fading through a transparent centre)
        for ``K ≤ 2``, a per-state **legend** (``A``, ``B``, ``C``, … ``ghost``) for ``K > 2``.

        For an ensemble result the rows show the ensemble-mean posterior. Requires matplotlib.

        Parameters
        ----------
        truth : dict[int, list[tuple[float, float, int]]], optional
            Ground-truth tracts ``(left, right, state)`` per haplotype, drawn as a reference track
            below each row. No truth track is drawn when ``None`` (default).
        segments : bool, optional
            Whether to plot hard segments using deadband. Default False.
        title : str, optional
            Title for the top haplotype axes. When ``None`` (default) it becomes a compact
            performance-stats summary — total ancestry proportions and fragmentation, plus breakpoint
            precision / recall when ``truth`` is given (:meth:`summary`). Pass ``""`` for no title.
        cmap : optional
            Unused (kept for backward compatibility). The confidence-as-opacity rendering does not use
            a colormap; pass per-state hues via ``colors``.
        colors : list, optional
            Per-state hues (state ``k`` → ``colors[k]``); defaults to matplotlib's colour cycle. For
            ``K > 2`` must supply at least ``K`` colours. For ``K = 2`` the two **extremes**
            ``colors[0]`` / ``colors[-1]`` are taken as the two state hues (so a legacy 3-stop diverging
            list like ``["#2c7bb6", "#ffffbf", "#d7191c"]`` still works — its middle stop is replaced by
            the transparent centre).
        alpha : float, optional
            Maximum opacity — **fades the whole plot** (default ``1``). The hard-segment and truth
            bars are drawn at opacity ``alpha``; the soft band at ``opacity = confidence · alpha``
            (spanning ``[0, alpha]``; ``confidence`` is the winning probability normalised out of
            ``[1/K, 1]``). E.g. ``alpha=0.5`` ⇒ nothing more than 50 % opaque.
        figsize : (float, float), optional
            Figure size. Default None.
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
        samples : iterable[int], optional
            Restrict the plot to these haplotypes, in the given order (one row each); defaults to all
            of :attr:`samples`. :meth:`tspaint.Painting.plot_posterior` uses this to plot a chosen
            subset by node index or individual id.
        curves : bool, optional
            Draw each row's soft posterior as **line curves** (``plt.plot``) instead of the colour band:
            one line per ancestry state in that state's segment hue, ``y = P(state)`` over the genome
            (e.g. ``plot(samples=[0], curves=True)`` shows the ``K`` posterior curves for one query).
            ``truth`` and — with ``segments=True`` — the hard segments are then drawn as thin solid
            state-coloured **bands** below the curves (each ~10 % of the curve height), mirroring the
            colour-band layout; the mask hatch overlay is not drawn in this mode. Default ``False``.
        curve_overlays : dict[int, list[tuple]], optional
            Extra curves per sample for ``curves=True`` — each ``(segments, state, colour, linestyle,
            label)`` (e.g. :meth:`tspaint.Painting.plot` passes the ghost ``P(ghost)`` posterior as a
            black dashed line via ``ghost=``).

        Returns
        -------
        tuple or None
            ``(figure, list_of_axes)`` if ``return_plot`` is ``True``, else ``None``.
        """
        import matplotlib
        import matplotlib.pyplot as plt



        qs = list(self.samples if samples is None else samples)
        missing = [q for q in qs if q not in self.posteriors]
        if missing:
            raise KeyError(f"samples {missing} are not painted in this track; "
                           f"available: {list(self.posteriors)}")
        segments = self.segments(deadband=deadband) if segments else []
        truth = self._prepare_truth(truth)              # identity, except GhostResult binarises to ghost

        # size the colour scale to cover both the painting states and any truth states — so a ghost
        # state embedded in the truth (index beyond the painting's K) gets its own colour in the truth
        # band, and is labelled "ghost" (not the next source letter) in the legend.
        n_source = _K_of(self.posteriors)
        K = n_source
        if truth:
            K = max(K, 1 + max((int(s[2]) for segs in truth.values() for s in segs), default=-1))
        colorizer = self._make_colorizer(K, n_source, cmap=cmap, colors=colors, alpha=alpha)
        fig = plt.figure(figsize=figsize if figsize else (10, min(0.25 * len(qs) + 1, 10)))
        gs = fig.add_gridspec(len(qs), 2, width_ratios=[1, _LEGEND_W], hspace=0)   # same legend column for every plot
        axes = [fig.add_subplot(gs[i, 0]) for i in range(len(qs))]

        for i, q in enumerate(qs):
            ylabel = (row_labels or {}).get(q, f'hapl. {i}')
            if curves:                                      # soft posterior as line curves, not a band
                _draw_curve_row(axes[i], self.posteriors[q], colorizer=colorizer, length=self.length,
                                ylabel=ylabel, overlays=(curve_overlays or {}).get(q),
                                hard_segs=(segments.get(q) if segments else None),
                                truth_segs=(truth.get(q) if truth else None))
            else:
                _draw_track_row(axes[i], self.posteriors[q], segments[q] if segments else [],
                                truth.get(q) if truth else None,      # rows with no truth omit the track
                                colorizer=colorizer, length=self.length, ylabel=ylabel,
                                mark_segs=(mark_spans or {}).get(q))
            axes[i].tick_params(axis='x', bottom=True)
            if i < len(axes) - 1:
                axes[i].xaxis.set_major_locator(matplotlib.ticker.NullLocator())

        legend_ax = fig.add_subplot(gs[:, 1])
        if curves:                                          # a line legend (states + overlays) for curves
            styles, seen = [], set()
            for lst in (curve_overlays or {}).values():
                for (_segs, _state, color, ls, label) in lst:
                    if (color, ls, label) not in seen:
                        seen.add((color, ls, label))
                        styles.append((color, ls, label))
            _draw_curve_legend(fig, legend_ax, colorizer, styles)
        else:
            colorizer.decorate(fig, legend_ax)
        if title is None:                               # default to the performance-stats summary
            title = self._summary_title(truth=truth, deadband=deadband, samples=qs)
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
    ``{sample: segments}`` dict the same read-out as a :class:`~tspaint.Painting`: :meth:`plot`
    (including its performance-stats default title via :meth:`~SoftTrack.summary` — proportions,
    fragmentation, and precision / recall against ``truth``), :meth:`segments`, :meth:`posterior_at`.
    Hard tuples become one-hot segments and render as solid per-state bars; soft segments render as
    the posterior gradient — so different tools plot in one consistent style (compare several at once
    with :func:`compare_tracks`). Purely a view: it does not alter the segments it wraps.

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
                   alpha=None, title=None, return_plot=False, deadband=None):
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
    cmap : optional
        Unused (kept for backward compatibility). Default ``'coolwarm'``.
        See :meth:`SoftTrack.plot`.
    colors : list, optional
        Per-state hues shared by every tool row and the truth row (the two extremes are the state
        hues for ``K = 2``). Default ``None`` → matplotlib's colour cycle.
    alpha : float, optional
        Maximum opacity (the soft band is drawn at ``confidence · alpha``). Default ``None`` (fully
        opaque, i.e. ``1``).
    title : str, optional
        Title for the top row. Default ``None`` — no title (unlike :meth:`SoftTrack.plot`, no
        summary is synthesised here).
    return_plot : bool, optional
        Return the matplotlib ``(figure, axes)`` instead of ``None``. Default ``False``.
    deadband : float, optional
        Dead-band on the top-two posterior margin ``max(P) − 2nd-max(P)`` for each tool's hard
        segments (:meth:`SoftTrack.segments` → :func:`tspaint.output.hard_segments`). Default
        ``None`` — each wrapped track's :attr:`~SoftTrack.default_deadband` (``0.4`` for a
        :class:`SegmentTrack`).

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
    L = float(length) if length is not None else max((st.length for st in sts.values()), default=0.0)
    truth_seg = truth.get(sample) if truth else None
    n_rows = len(names) + (1 if truth_seg is not None else 0)

    # one shared K (and colour rule) across every tool row and the truth row; states above the tools'
    # painted states are truth-only foreign (ghost) states, labelled "ghost" in the legend.
    n_source = max([_K_of(st.posteriors) for st in sts.values()] + [2])
    K = max([n_source] + [int(s[2]) + 1 for s in (truth_seg or [])])
    colorizer = _Colorizer(K, cmap=cmap, colors=colors, alpha=alpha, n_source=n_source)
    fig = plt.figure(figsize=(9, 0.4 * n_rows + 1))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1, _LEGEND_W], hspace=0)   # same legend column for every plot
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]

    for i, name in enumerate(names):
        st = sts[name]
        _draw_track_row(axes[i], st.posteriors.get(sample, []),
                        st.segments(deadband=deadband).get(sample, []), None,
                        colorizer=colorizer, length=L, ylabel=name)
        axes[i].tick_params(axis='x', bottom=True)
        if i < n_rows - 1:
            axes[i].xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    if truth_seg is not None:
        _draw_track_row(axes[-1], [], truth_seg, None, colorizer=colorizer, length=L, ylabel="truth")
        axes[-1].tick_params(axis='x', bottom=True)

    colorizer.decorate(fig, fig.add_subplot(gs[:, 1]))
    if title:
        axes[0].set_title(title)
    plt.tight_layout()
    if return_plot:
        return fig, axes
